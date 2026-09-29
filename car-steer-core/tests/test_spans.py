import pytest
import torch

from car_steer.spans import (ContextToken, ShortSpanBias, build_stop_tokens,
                             context_tokens, make_prompt, select_span)


def test_context_only_mapping(tokenizer):
    prompt, start, end = make_prompt("Mira Sen.", "Who is Mira?")
    encoded = tokenizer(prompt, return_offsets_mapping=True)
    found = context_tokens(prompt, encoded["input_ids"], encoded["offset_mapping"], start, end)
    assert [t.token_id for t in found] == [6, 7, 8]
    # A leading-space token can begin one character before the context itself.
    assert context_tokens(prompt, [26], [(start-1, start+4)], start, end)[0].character_offset == 0
    assert context_tokens(prompt, [26], [(0, start+4)], start, end) == []


def test_stop_set_and_leading_spaces(tokenizer):
    stops = build_stop_tokens(tokenizer, len(tokenizer) + 1, {2})
    assert {2, 5, 8, 14, 23, 27, 28, len(tokenizer)} <= stops
    assert not {24, 25, 26} & stops
    tokens = [ContextToken(26, 0, 0), ContextToken(7, 1, 6)]
    assert select_span(24, tokens, stops, None) is None
    assert select_span(26, tokens, stops, None).token_ids == (26, 7)


def test_longest_span_then_attention_then_earliest():
    tokens = [ContextToken(t, i, i * 2) for i, t in enumerate([6, 7, 8, 6, 9, 10, 8])]
    weights = torch.tensor([.9, .1, 0, .5, .5, .5, 0])
    # Equal mean attention; longer candidates do not win a tie over an earlier start.
    span = select_span(6, tokens, {8}, weights)
    assert span.token_ids == (6, 7) and span.character_offset == 0
    weights[3:6] = .6
    assert select_span(6, tokens, {8}, weights).token_ids == (6, 9, 10)
    with pytest.raises(RuntimeError):
        select_span(6, tokens, {8}, None)
    assert len(select_span(6, [ContextToken(6 if i == 0 else 7, i, i)
                             for i in range(15)], set(), None).token_ids) == 10


def test_three_steps_only_and_single_token_bias():
    tokens = [ContextToken(t, i, i) for i, t in enumerate([6, 7, 9, 10])]
    bias = ShortSpanBias(6, True, 2, .5, tokens, {8}, None)
    for step, (token, amount) in enumerate([(6, 2), (7, 1), (9, .5), (10, 0)]):
        scores = bias.apply(torch.zeros(32), step)
        assert scores.sum().item() == amount
        assert scores[token].item() == amount
        bias.observe(token)
    assert bias.stop_reason == "three_step_limit"


@pytest.mark.parametrize("emitted,reason", [(11, "emitted_token_mismatch"), (8, "stopping_token")])
def test_early_stop_is_permanent(emitted, reason):
    bias = ShortSpanBias(6, True, 2, .5, [ContextToken(6, 0, 0)], {8}, None)
    bias.apply(torch.zeros(32), 0)
    bias.observe(emitted)
    assert bias.stop_reason == reason
    assert torch.equal(bias.apply(torch.ones(32), 1), torch.ones(32))
    bias.observe(6)
    assert not bias.active


def test_no_match_span_end_and_disabled():
    for tokens, reason in [([], "no_context_match"), ([ContextToken(6, 0, 0)], "span_end")]:
        bias = ShortSpanBias(6, True, 2, .5, tokens, set(), None)
        bias.apply(torch.zeros(32), 0)
        bias.observe(6)
        assert bias.stop_reason == reason
    bias = ShortSpanBias(6, False, 2, .5, [], set(), None)
    assert bias.apply(torch.zeros(32), 0).sum() == 0
    bias.observe(7)
    assert bias.stop_reason == "disabled"
