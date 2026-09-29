import pytest
import torch

from car_steer.config import SteeringConfig
from car_steer.geometry import (HistoricalPeak, curvature, gate_multiplier,
                                residual_update, vocabulary_mean)


@pytest.mark.parametrize("end,expected", [([2., 0.], 0), ([1., 1.], 1), ([0., 0.], 2)])
def test_known_turns(end, expected):
    assert curvature(torch.tensor([0., 0.]), torch.tensor([1., 0.]),
                     torch.tensor(end)) == pytest.approx(expected)


def test_degenerate_curvature():
    assert curvature(torch.zeros(2), torch.zeros(2), torch.ones(2)) == 0
    assert curvature(torch.zeros(2), torch.tensor([1e-8, 0.]), torch.ones(2)) == 0


@pytest.mark.parametrize("r,scale", [(-0.1, 0), (0, 2), (.000999, 2), (.001, 1),
                                    (.009999, 1), (.01, .3), (.099999, .3), (.1, 0), (1, 0)])
def test_gate_boundaries(r, scale):
    assert gate_multiplier(r) == scale


def test_norm_scaling_and_centering():
    weight = torch.randn(17, 8)
    assert torch.allclose(vocabulary_mean(weight, 3), weight.mean(0), atol=1e-7)
    h, v = torch.tensor([[2., 0.]]), torch.tensor([.005, (1 - .005**2)**.5])
    updated, r, alpha, opened = residual_update(h, v, .8)
    assert opened and r == pytest.approx(.005) and alpha == .8
    assert (updated - h).norm().item() == pytest.approx(.8 * h.norm().item())
    assert not residual_update(h, torch.zeros(2), 1)[3]


def test_historical_ties_and_probability_not_logit():
    peak = HistoricalPeak()
    peak.observe(1, torch.tensor([1., 1., 0.]))
    peak.observe(2, torch.tensor([1., 1., 0.]))
    assert peak.best.layer == 1 and peak.best.token_id == 0
    peak.observe(3, torch.tensor([99., 100., 99.]))
    assert peak.best.layer == 3 and peak.best.token_id == 1
    peak.observe(4, torch.tensor([999., 999., 999.]))
    assert peak.best.layer == 3  # Absolute logit magnitude cannot select the peak.


def test_paper_presets():
    qwen = SteeringConfig.preset("qwen3-8b")
    assert qwen.bounds(36) == (3, 31)
    assert SteeringConfig.preset("llama-3.1-8b").bounds(32) == (3, 31)
    with pytest.raises(ValueError):
        SteeringConfig(threshold=.3, center_end=6).bounds(6)
