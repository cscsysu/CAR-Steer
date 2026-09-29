"""Real HF decoder classes with tiny random weights; no downloads or score claims."""

import pytest
import torch

from car_steer import CARSteer, SteeringConfig
from car_steer.geometry import residual_update, vocabulary_mean
from car_steer.hooks import PrefillMonitor
from car_steer.spans import make_prompt


CONTEXT, QUESTION = "Mira Sen is the director.", "Who is the director?"


def inputs(tokenizer):
    prompt = make_prompt(CONTEXT, QUESTION)[0]
    encoded = tokenizer(prompt, return_tensors="pt")
    return {key: encoded[key] for key in ("input_ids", "attention_mask")}


def raw_states(model, encoded):
    states, handles = [], []
    for block in model.model.layers:
        handles.append(block.register_forward_hook(
            lambda module, args, output: states.append(output[0].detach().clone())))
    try:
        model(**encoded, use_cache=True, logits_to_keep=1)
    finally:
        for handle in handles:
            handle.remove()
    return states


def arrange_active_gate(model, tokenizer, inject_block=4):
    """A deterministic numerical fixture with an early candidate and r=0.005."""
    encoded = inputs(tokenizer)
    states = raw_states(model, encoded)
    first, fourth = states[0][0, -1].float(), states[inject_block - 1][0, -1].float()
    axis = fourth / fourth.norm()
    perpendicular = first - first.dot(axis) * axis
    perpendicular /= perpendicular.norm()
    direction = perpendicular * (1 - .005**2)**.5 + .005 * axis
    model.lm_head.weight.zero_()
    model.lm_head.weight[6].copy_(direction)
    return encoded


@torch.inference_mode()
def test_no_crossing_matches_hf_greedy_and_resets(tiny_model, tokenizer):
    runner = CARSteer(tiny_model, tokenizer, SteeringConfig(threshold=2., max_new_tokens=4))
    expected = tiny_model.generate(**inputs(tokenizer), do_sample=False, max_new_tokens=4,
                                   use_cache=True, pad_token_id=tokenizer.pad_token_id)
    result = runner.generate(CONTEXT, QUESTION)
    assert result.token_ids == expected[0, result.prompt_tokens:].tolist()
    assert result.prefill["trigger_center"] is None
    assert all(step["bias"] == 0 for step in result.bias_steps)
    assert runner.generate(CONTEXT, QUESTION).to_dict() == result.to_dict()
    assert all(not b._forward_hooks and not b.self_attn._forward_hooks for b in tiny_model.model.layers)


@torch.inference_mode()
def test_active_prefill_equals_explicit_intervention_and_cache(tiny_model, tokenizer):
    encoded = arrange_active_gate(tiny_model, tokenizer)
    config = SteeringConfig(threshold=0., center_end=3, max_new_tokens=4)
    mean = vocabulary_mean(tiny_model.lm_head.weight)
    baseline = tiny_model(**encoded, use_cache=True, logits_to_keep=1)
    with PrefillMonitor(tiny_model, config, mean) as monitor:
        actual = tiny_model(**encoded, use_cache=True, logits_to_keep=1)
    assert monitor.trace.trigger_center == 3
    assert monitor.trace.injection_block == 4
    assert monitor.trace.target_token_id == 6
    assert monitor.trace.target_peak_layer <= 3
    assert monitor.trace.alignment == pytest.approx(.005, abs=2e-6)
    assert monitor.trace.residual_applied
    assert monitor.attention.shape == (encoded["input_ids"].shape[1],)
    direction = tiny_model.lm_head.weight[6].float() - mean

    def manual(module, args, output):
        state = output[0].clone()
        state[:, -1, :] = residual_update(state[:, -1, :], direction, config.alpha)[0]
        return (state,) + output[1:]

    handle = tiny_model.model.layers[3].register_forward_hook(manual)
    try:
        expected = tiny_model(**encoded, use_cache=True, logits_to_keep=1)
    finally:
        handle.remove()
    assert torch.allclose(actual.logits, expected.logits, atol=1e-6)
    for index, (key, value) in enumerate(actual.past_key_values):
        expected_key, expected_value = expected.past_key_values[index]
        assert torch.allclose(key, expected_key, atol=1e-6)
        assert torch.allclose(value, expected_value, atol=1e-6)
        base_key, base_value = baseline.past_key_values[index]
        if index <= 3:
            assert torch.equal(key, base_key) and torch.equal(value, base_value)
        else:
            assert not torch.equal(value, base_value)
    runner = CARSteer(tiny_model, tokenizer, config)
    result = runner.generate(CONTEXT, QUESTION)
    assert result.prefill["residual_applied"]
    assert result.bias_steps[0]["bias"] == 2.0
    assert result.token_ids[0] == 6
    assert runner.generate(CONTEXT, QUESTION).to_dict() == result.to_dict()


@torch.inference_mode()
def test_closed_gate_matches_baseline(tiny_model, tokenizer):
    encoded = inputs(tokenizer)
    fourth = raw_states(tiny_model, encoded)[3][0, -1]
    tiny_model.lm_head.weight.zero_()
    tiny_model.lm_head.weight[6].copy_(fourth / fourth.norm())
    runner = CARSteer(tiny_model, tokenizer, SteeringConfig(threshold=0., center_end=3,
                                                          max_new_tokens=4))
    result = runner.generate(CONTEXT, QUESTION)
    expected = tiny_model.generate(**encoded, do_sample=False, max_new_tokens=4, use_cache=True)
    assert result.prefill["trigger_center"] == 3
    assert not result.prefill["gate_open"]
    assert result.token_ids == expected[0, result.prompt_tokens:].tolist()
    assert all(step["bias"] == 0 for step in result.bias_steps)


def test_context_limit_and_preset_checks(tiny_model, tokenizer):
    runner = CARSteer(tiny_model, tokenizer, SteeringConfig(threshold=2.))
    with pytest.raises(ValueError, match="exceeds"):
        runner.generate("Mira " * 250, QUESTION)
    with pytest.raises(ValueError, match="Preset"):
        CARSteer(tiny_model, tokenizer, SteeringConfig.preset("qwen3-8b"))


@torch.inference_mode()
def test_last_block_injection_does_not_rewrite_any_cache(tiny_model, tokenizer):
    encoded = arrange_active_gate(tiny_model, tokenizer, inject_block=6)
    baseline = tiny_model(**encoded, use_cache=True, logits_to_keep=1)
    config = SteeringConfig(threshold=0., center_start=5, center_end=5)
    with PrefillMonitor(tiny_model, config, vocabulary_mean(tiny_model.lm_head.weight)) as monitor:
        actual = tiny_model(**encoded, use_cache=True, logits_to_keep=1)
    assert monitor.trace.trigger_center == 5 and monitor.trace.injection_block == 6
    assert monitor.trace.target_peak_layer <= 4  # floor(0.8 * 6), despite center 5.
    assert monitor.trace.residual_applied
    assert not torch.equal(actual.logits, baseline.logits)
    for (k, v), (base_k, base_v) in zip(actual.past_key_values, baseline.past_key_values):
        assert torch.equal(k, base_k) and torch.equal(v, base_v)
    assert monitor.attention is not None  # Available even though the decision was later.
