"""Paper presets and explicit inference settings."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SteeringConfig:
    threshold: float
    alpha: float = 1.0
    initial_bias: float = 2.0
    bias_decay: float = 0.5
    center_start: int = 3
    center_end: int | None = None
    max_new_tokens: int = 32
    residual_enabled: bool = True
    bias_enabled: bool = True
    # Presets validate the architecture and depth instead of silently transferring.
    model_type: str | None = None
    num_layers: int | None = None

    def __post_init__(self):
        for name in ("threshold", "alpha", "initial_bias", "bias_decay"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.threshold < 0 or self.alpha < 0 or self.initial_bias < 0:
            raise ValueError("threshold, alpha and initial_bias must be nonnegative")
        if not 0 <= self.bias_decay <= 1:
            raise ValueError("bias_decay must lie in [0, 1]")
        if self.center_start < 3 or self.max_new_tokens < 1:
            raise ValueError("center_start must be >= 3; max_new_tokens must be positive")

    def bounds(self, depth: int) -> tuple[int, int]:
        end = depth - 1 if self.center_end is None else self.center_end
        if not self.center_start <= end < depth:
            raise ValueError("Curvature centers must satisfy 3 <= start <= end < depth")
        return self.center_start, end

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def preset(cls, name: str, **overrides) -> "SteeringConfig":
        presets = {
            "qwen3-8b": dict(threshold=0.385, center_end=31,
                            model_type="qwen3", num_layers=36),
            "llama-3.1-8b": dict(threshold=0.412, center_end=31,
                                model_type="llama", num_layers=32),
            "mistral-7b": dict(threshold=0.368, center_end=31,
                              model_type="mistral", num_layers=32),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset {name!r}; choose {list(presets)}")
        return cls(**(presets[name] | overrides))
