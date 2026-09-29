from types import SimpleNamespace

import pytest
import torch
from torch import nn

from car_steer import SteeringConfig
from car_steer.geometry import vocabulary_mean
from car_steer.hooks import PrefillMonitor


class ToyBlock(nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.register_buffer("delta", torch.tensor(delta))
        self.self_attn = nn.Identity()

    def forward(self, state):
        return (state + self.delta,)


class ToyModel:
    def __init__(self):
        self.model = SimpleNamespace(layers=nn.ModuleList([
            ToyBlock([0., 0.]), ToyBlock([1., 0.]), ToyBlock([1., 0.]),
            ToyBlock([0., 1.]), ToyBlock([1., 1.]), ToyBlock([1., 1.])]), norm=nn.Identity())
        self.head = nn.Linear(2, 2, bias=False)
        # h4=(2,1), dot(v,h4)>0 but cosine is <0.01; earlier h1..h3 favor token 0.
        v = torch.tensor([1., -1.99])
        with torch.no_grad():
            self.head.weight.copy_(torch.stack((v, -v)))

    def get_output_embeddings(self):
        return self.head

    def run(self):
        x = torch.zeros(1, 2, 2)
        states = []
        for layer in self.model.layers:
            x = layer(x)[0]
            states.append(x.clone())
        return states


def test_causal_output_update_only_last_token_and_single_trigger():
    model = ToyModel()
    config = SteeringConfig(threshold=.5, bias_enabled=False)
    with torch.inference_mode():
        baseline = model.run()
        monitor = PrefillMonitor(model, config, vocabulary_mean(model.head.weight))
        with monitor:
            states = model.run()
        trace = monitor.trace
        assert trace.trigger_center == 3 and trace.injection_block == 4
        assert trace.target_peak_layer == 3 and trace.target_token_id == 0
        assert trace.residual_applied and len(trace.curvatures) == 1
        for before, after in zip(baseline[:3], states[:3]):
            assert torch.equal(before, after)
        assert torch.equal(states[3][:, 0, :], baseline[3][:, 0, :])
        assert not torch.equal(states[3][:, -1, :], baseline[3][:, -1, :])
        assert torch.allclose(states[4] - baseline[4], states[3] - baseline[3])
        assert all(not layer._forward_hooks for layer in model.model.layers)
        assert all(torch.equal(a, b) for a, b in zip(model.run(), baseline))


def test_strict_threshold_and_candidate_excludes_current_block():
    model = ToyModel()
    with torch.inference_mode():
        monitor = PrefillMonitor(model, SteeringConfig(threshold=1., center_end=3, bias_enabled=False),
                                 vocabulary_mean(model.head.weight))
        with monitor:
            model.run()
        assert monitor.trace.curvatures[0]["value"] == 1
        assert monitor.trace.trigger_center is None  # Equality never triggers.


def test_zero_gate_stops_scan_and_hook_cleanup_after_exception():
    model = ToyModel()
    with torch.no_grad():
        model.head.weight.copy_(torch.tensor([[1., 0.], [-1., 0.]]))
    with torch.inference_mode():
        monitor = PrefillMonitor(model, SteeringConfig(threshold=.5, bias_enabled=False),
                                 vocabulary_mean(model.head.weight))
        with monitor:
            baseline = model.run()
        assert monitor.trace.status == "gate_closed"
        assert len(monitor.trace.curvatures) == 1
        assert all(torch.equal(a, b) for a, b in zip(baseline, model.run()))
        with pytest.raises(RuntimeError, match="test failure"):
            with PrefillMonitor(model, monitor.config, monitor.mean):
                raise RuntimeError("test failure")
        assert all(not layer._forward_hooks for layer in model.model.layers)
