"""Equations (1)--(3); all layer labels exposed by this package are 1-based."""

from dataclasses import dataclass
import math

import torch


def curvature(h_prev: torch.Tensor, h_center: torch.Tensor,
              h_next: torch.Tensor) -> float:
    """Center-indexed turn; available only once h_next exists. Eq. (1)."""
    incoming = h_center.double() - h_prev.double()
    outgoing = h_next.double() - h_center.double()
    a, b = incoming.norm(), outgoing.norm()
    if a.item() < 1e-6 or b.item() < 1e-6:
        return 0.0
    value = 1.0 - (incoming.flatten() @ outgoing.flatten()) / (a * b)
    if not torch.isfinite(value):
        raise FloatingPointError("Non-finite residual curvature")
    return value.clamp(0, 2).item()


def gate_multiplier(alignment: float) -> float:
    """Mutually exclusive intervals in Eq. (3)."""
    if not math.isfinite(alignment):
        raise FloatingPointError("Non-finite gate alignment")
    if alignment < 0 or alignment >= 0.1:
        return 0.0
    if alignment < 0.001:
        return 2.0
    if alignment < 0.01:
        return 1.0
    return 0.3


def vocabulary_mean(weight: torch.Tensor, chunk_size: int = 4096) -> torch.Tensor:
    """Mean over HF output-embedding rows, without a full float32 matrix copy."""
    total = torch.zeros(weight.shape[1], dtype=torch.float32, device=weight.device)
    for rows in weight.split(chunk_size):
        total.add_(rows.float().sum(dim=0))
    return total / weight.shape[0]


def residual_update(h: torch.Tensor, direction: torch.Tensor, alpha: float):
    """Return (updated h, cosine r, effective alpha, gate is open)."""
    state, vector = h.float(), direction.float()
    h_norm, v_norm = state.norm(), vector.norm()
    if not torch.isfinite(h_norm) or not torch.isfinite(v_norm):
        raise FloatingPointError("Non-finite injection state or direction")
    if h_norm.item() < 1e-6 or v_norm.item() < 1e-6:
        return h, None, 0.0, False
    r = ((state.flatten() @ vector.flatten()) / (h_norm * v_norm)).clamp(-1, 1).item()
    multiplier = gate_multiplier(r)
    effective = alpha * multiplier
    updated = state + effective * (h_norm / v_norm) * vector
    return updated.to(h.dtype), r, effective, multiplier > 0


@dataclass(frozen=True)
class Candidate:
    layer: int
    token_id: int
    probability: float


class HistoricalPeak:
    """The register is updated in increasing layer order, after trigger evaluation."""

    def __init__(self):
        self.best: Candidate | None = None

    def observe(self, layer: int, logits: torch.Tensor):
        flat = logits.float().reshape(-1)
        if not torch.isfinite(flat).all():
            raise FloatingPointError("Non-finite logit-lens projection")
        # torch.argmax chooses the first (smallest) token ID on exact ties.
        token_id = int(flat.argmax().item())
        probability = torch.softmax(flat, dim=-1)[token_id].item()
        candidate = Candidate(layer, token_id, probability)
        if self.best is None or probability > self.best.probability:
            self.best = candidate
