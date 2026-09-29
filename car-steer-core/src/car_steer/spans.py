"""Tokenizer-aligned context lookup and the three-step output-bias state machine."""

from dataclasses import dataclass
import string

import torch


@dataclass(frozen=True)
class ContextToken:
    token_id: int
    prompt_position: int
    character_offset: int


@dataclass(frozen=True)
class ContextSpan:
    token_ids: tuple[int, ...]
    prompt_positions: tuple[int, ...]
    character_offset: int
    attention_score: float


def make_prompt(context: str, question: str) -> tuple[str, int, int]:
    prefix = "Context: "
    prompt = prefix + context + "\nQuestion: " + question + "\nAnswer with the shortest span from the context:"
    return prompt, len(prefix), len(prefix) + len(context)


def context_tokens(prompt: str, token_ids: list[int], offsets: list[tuple[int, int]],
                   start: int, end: int) -> list[ContextToken]:
    """Map context tokens back to the model's actual prompt tokenization.

    A first token may also cover whitespace immediately before the context. It
    may never consume the 'Context:' label, question, or instructions.
    """
    result = []
    for position, (token_id, (a, b)) in enumerate(zip(token_ids, offsets)):
        if b <= a or b <= start or b > end:
            continue
        if a < start and not prompt[a:start].isspace():
            continue
        result.append(ContextToken(token_id, position, max(a, start) - start))
    return result


def build_stop_tokens(tokenizer, vocab_size: int, eos_ids: set[int]) -> set[int]:
    known = set(tokenizer.get_vocab().values())
    # Padded output rows without a tokenizer representation cannot anchor a span.
    stops = (set(range(vocab_size)) - known) | eos_ids
    for token_id in sorted(known):
        if not 0 <= token_id < vocab_size:
            continue
        decoded = tokenizer.decode([token_id], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if decoded.isspace() or any(char in string.punctuation for char in decoded):
            stops.add(token_id)
    return stops


def select_span(first_token: int, tokens: list[ContextToken], stop_ids: set[int],
                attention: torch.Tensor | None) -> ContextSpan | None:
    candidates = []
    for index, token in enumerate(tokens):
        if token.token_id != first_token or token.token_id in stop_ids:
            continue
        span = []
        for current in tokens[index:index + 10]:
            if current.token_id in stop_ids:
                break
            if span and current.prompt_position != span[-1].prompt_position + 1:
                break
            span.append(current)
        candidates.append(span)
    if not candidates:
        return None
    if len(candidates) > 1 and attention is None:
        raise RuntimeError("Multiple context matches require penultimate prefill attention")
    scored = []
    for span in candidates:
        positions = tuple(t.prompt_position for t in span)
        score = float(attention[list(positions)].mean().item()) if attention is not None else 0.0
        scored.append(ContextSpan(tuple(t.token_id for t in span), positions,
                                  span[0].character_offset, score))
    return min(scored, key=lambda span: (-span.attention_score, span.character_offset))


class ShortSpanBias:
    """A positive bias on a single expected token; stop permanently on mismatch."""

    def __init__(self, target: int | None, enabled: bool, initial: float, decay: float,
                 tokens: list[ContextToken], stops: set[int], attention: torch.Tensor | None):
        self.target, self.initial, self.decay = target, initial, decay
        self.tokens, self.stops, self.attention = tokens, stops, attention
        self.active = enabled and target is not None and initial > 0
        self.stop_reason = None if self.active else "disabled"
        self.span: ContextSpan | None = None
        self.events: list[dict] = []
        self._pending: tuple[int, int | None, float] | None = None

    def _stop(self, reason: str):
        self.active, self.stop_reason = False, reason

    def apply(self, logits: torch.Tensor, step: int) -> torch.Tensor:
        if self._pending is not None:
            raise RuntimeError("Observe the emitted token before applying another bias")
        if self.active and step >= 3:
            self._stop("three_step_limit")
        expected, amount = None, 0.0
        if self.active:
            if step == 0:
                expected = self.target
            elif self.span is not None and step < len(self.span.token_ids):
                expected = self.span.token_ids[step]
            else:
                self._stop("span_end")
            if expected is not None:
                amount = self.initial * self.decay ** step
        # Use float32 for adding small logit increments even with BF16 weights.
        adjusted = logits.float().clone()
        if expected is not None:
            adjusted[expected] += amount
        self._pending = (step, expected, amount)
        return adjusted

    def observe(self, emitted: int):
        if self._pending is None:
            raise RuntimeError("apply() must precede observe()")
        step, expected, amount = self._pending
        self._pending = None
        if self.active:
            if emitted in self.stops:
                self._stop("stopping_token")
            elif emitted != expected:
                self._stop("emitted_token_mismatch")
            elif step == 0:
                self.span = select_span(emitted, self.tokens, self.stops, self.attention)
                if self.span is None:
                    self._stop("no_context_match")
            if self.active and step >= 2:
                self._stop("three_step_limit")
            elif self.active and self.span is not None and step + 1 >= len(self.span.token_ids):
                self._stop("span_end")
        self.events.append({"step": step, "expected_token_id": expected,
                            "bias": amount, "emitted_token_id": emitted,
                            "stop_reason": self.stop_reason})
