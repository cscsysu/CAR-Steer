"""Single-example greedy generation with one causally steered prefill pass."""

from dataclasses import asdict, dataclass
import threading

import torch
import transformers

from .config import SteeringConfig
from .geometry import vocabulary_mean
from .hooks import PrefillMonitor
from .spans import ShortSpanBias, build_stop_tokens, context_tokens, make_prompt


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int]
    prompt_tokens: int
    finish_reason: str
    prefill: dict
    bias_steps: list[dict]
    selected_span: dict | None

    def to_dict(self):
        return asdict(self)


class CARSteer:
    """Wrap a fixed HF Qwen3/Llama/Mistral CausalLM and a fast tokenizer.

    Load with attn_implementation='eager' on one device. Construct the wrapper
    after model.to(device/dtype); its cached vocabulary mean belongs to that
    device and weight version. Each generate() call owns its hooks and KV cache.
    """

    def __init__(self, model, tokenizer, config: SteeringConfig):
        if transformers.__version__ != "4.51.2":
            raise RuntimeError("This release targets transformers==4.51.2")
        model_type = getattr(model.config, "model_type", None)
        if model_type not in {"qwen3", "llama", "mistral"}:
            raise ValueError("Supported model families: qwen3, llama, mistral")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("A fast tokenizer is required for context offset mapping")
        if getattr(model.config, "_attn_implementation", None) != "eager":
            raise ValueError("Load the model with attn_implementation='eager'")
        if getattr(model, "is_quantized", False):
            raise ValueError("Use floating-point model weights; quantized heads are unsupported")
        devices = {p.device for p in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type == "meta":
            raise ValueError("Place all model parameters on one CPU or CUDA device")
        self.device = next(iter(devices))
        self.weight = model.get_output_embeddings().weight
        self.dtype = self.weight.dtype
        depth = len(model.model.layers)
        if config.model_type is not None and config.model_type != model_type:
            raise ValueError("Preset model family does not match the loaded model")
        if config.num_layers is not None and config.num_layers != depth:
            raise ValueError("Preset decoder depth does not match the loaded model")
        config.bounds(depth)
        self.model, self.tokenizer, self.config = model.eval(), tokenizer, config
        self.eos_ids = set()
        for source in (tokenizer, model.config, model.generation_config):
            values = getattr(source, "eos_token_id", None)
            if values is not None:
                self.eos_ids.update(values if isinstance(values, (list, tuple)) else [values])
        with torch.inference_mode():
            self.mean = vocabulary_mean(self.weight)
        self.stops = build_stop_tokens(tokenizer, self.weight.shape[0], self.eos_ids)
        self._lock = threading.Lock()

    @torch.inference_mode()
    def generate(self, context: str, question: str) -> GenerationResult:
        if not isinstance(context, str) or not isinstance(question, str):
            raise TypeError("context and question must be strings")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("A CARSteer wrapper can handle only one request at a time")
        try:
            if self.weight.device != self.device or self.weight.dtype != self.dtype:
                raise RuntimeError("Reconstruct CARSteer after moving or recasting model weights")
            return self._generate(context, question)
        finally:
            self._lock.release()

    def _generate(self, context: str, question: str) -> GenerationResult:
        prompt, start, end = make_prompt(context, question)
        encoded = self.tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True,
                                 return_tensors="pt", truncation=False)
        ids = encoded["input_ids"].to(self.device)
        mask = encoded["attention_mask"].to(self.device)
        positions = encoded["offset_mapping"][0].tolist()
        tokens = context_tokens(prompt, ids[0].tolist(), positions, start, end)
        length = ids.shape[1]
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit is not None and length + self.config.max_new_tokens > limit:
            raise ValueError("Prompt plus generation exceeds max_position_embeddings; shorten the context explicitly")
        with PrefillMonitor(self.model, self.config, self.mean) as monitor:
            output = self.model(input_ids=ids, attention_mask=mask, use_cache=True,
                                return_dict=True, output_attentions=False,
                                output_hidden_states=False, logits_to_keep=1)
        # Hooks are removed before any decoding call.
        trace = monitor.trace
        bias = ShortSpanBias(trace.target_token_id,
                             trace.gate_open and self.config.bias_enabled,
                             self.config.initial_bias, self.config.bias_decay,
                             tokens, self.stops, monitor.attention)
        generated, finish = [], "length"
        for step in range(self.config.max_new_tokens):
            scores = bias.apply(output.logits[0, -1, :], step)
            token_id = int(scores.argmax().item())
            generated.append(token_id)
            bias.observe(token_id)
            if token_id in self.eos_ids:
                finish = "eos"
                break
            if step + 1 < self.config.max_new_tokens:
                mask = torch.cat((mask, mask.new_ones((1, 1))), dim=-1)
                output = self.model(input_ids=ids.new_tensor([[token_id]]),
                                    attention_mask=mask, past_key_values=output.past_key_values,
                                    use_cache=True, return_dict=True, output_attentions=False,
                                    output_hidden_states=False, logits_to_keep=1)
        text = self.tokenizer.decode(generated, skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        return GenerationResult(text, generated, length, finish, trace.to_dict(),
                                bias.events, asdict(bias.span) if bias.span else None)
