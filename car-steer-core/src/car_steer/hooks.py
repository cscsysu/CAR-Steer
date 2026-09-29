"""Causal prefill hooks. No hook is retained during autoregressive decoding."""

from collections import deque
from dataclasses import asdict, dataclass, field

import torch

from .config import SteeringConfig
from .geometry import HistoricalPeak, curvature, residual_update


@dataclass
class PrefillTrace:
    center_start: int
    center_end: int
    candidate_cap: int
    curvatures: list[dict] = field(default_factory=list)
    trigger_center: int | None = None
    injection_block: int | None = None
    target_token_id: int | None = None
    target_peak_layer: int | None = None
    target_peak_probability: float | None = None
    alignment: float | None = None
    effective_alpha: float = 0.0
    gate_open: bool = False
    residual_applied: bool = False
    status: str = "no_crossing"

    def to_dict(self):
        return asdict(self)


class PrefillMonitor:
    def __init__(self, model, config: SteeringConfig, mean: torch.Tensor):
        self.model, self.config, self.mean = model, config, mean
        self.layers = model.model.layers
        self.depth = len(self.layers)
        self.start, self.end = config.bounds(self.depth)
        self.cap = (4 * self.depth) // 5
        self.trace = PrefillTrace(self.start, self.end, self.cap)
        self.history = deque(maxlen=3)
        self.peak = HistoricalPeak()
        self.attention: torch.Tensor | None = None
        self.handles = []

    def __enter__(self):
        try:
            for index, layer in enumerate(self.layers):
                self.handles.append(layer.register_forward_hook(self._block_hook(index + 1)))
            self.handles.append(self.layers[-2].self_attn.register_forward_hook(self._attention_hook))
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _attention_hook(self, module, inputs, output):
        if not self.config.bias_enabled:
            return
        if not self.trace.gate_open:
            # At L-1, a trigger at L-1 itself is still undecided until block L.
            if self.trace.trigger_center is not None or self.end < self.depth - 1:
                return
        weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
        if weights is None:
            raise RuntimeError("Penultimate attention weights unavailable; load with attn_implementation='eager'")
        # Only keep one query row averaged over heads, never all layers' attention.
        self.attention = weights[0, :, -1, :].detach().float().mean(dim=0).clone()

    def _block_hook(self, block: int):
        def hook(module, inputs, output):
            if self.trace.trigger_center is not None or block > self.end + 1:
                return None
            states = output[0] if isinstance(output, tuple) else output
            if states.ndim != 3 or states.shape[0] != 1:
                raise ValueError("Core inference requires a single unpadded prompt")
            h = states[:, -1, :].detach().clone()
            self.history.append(h)
            center = block - 1
            if self.start <= center <= self.end:
                kappa = curvature(*self.history)
                self.trace.curvatures.append({"center": center, "observed_after_block": block,
                                              "value": kappa})
                if kappa > self.config.threshold:
                    best = self.peak.best
                    if best is None:
                        raise RuntimeError("No historical candidate at an eligible trigger")
                    trace = self.trace
                    trace.trigger_center, trace.injection_block = center, block
                    trace.target_token_id, trace.target_peak_layer = best.token_id, best.layer
                    trace.target_peak_probability = best.probability
                    weight = self.model.get_output_embeddings().weight
                    direction = weight[best.token_id].float() - self.mean
                    updated, trace.alignment, trace.effective_alpha, trace.gate_open = residual_update(
                        h, direction, self.config.alpha)
                    trace.status = "gate_open" if trace.gate_open else "gate_closed"
                    if trace.alignment is None:
                        trace.status = "degenerate_state_or_direction"
                    if trace.gate_open and self.config.residual_enabled and trace.effective_alpha > 0:
                        modified = states.clone()
                        modified[:, -1, :] = updated
                        trace.residual_applied = True
                        trace.status = "injected"
                        return (modified,) + output[1:] if isinstance(output, tuple) else modified
                    return None  # First crossing is final, including when the gate closes.
            # The current completed block must not enter the register before the test above:
            # the decision at center l uses candidates through l, never through l+1.
            if block <= min(self.cap, self.end):
                normalized = self.model.model.norm(h)
                self.peak.observe(block, self.model.get_output_embeddings()(normalized))
            return None
        return hook
