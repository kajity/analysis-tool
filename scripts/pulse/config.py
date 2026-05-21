from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass(frozen=True)
class PulseAnalysisConfig:
    max_points_per_trace: int | None = 1000
    max_display_traces: int | None = 400
    valid_pulse_range_start: int = 320
    valid_pulse_range_stop: int = 330
    valid_pulse_diff_threshold: float = 1000.0
    spectrum_bins: str | int = "auto"
    histogram_min: float | None = None
    histogram_max: float | None = None
    spectrum_chunk_size: int = 512
    negative_pulses: bool = True
    optimal_filter_template_normalize: bool = True
    baseline_drift_correction: bool = False
    baseline_drift_baseline_min: float | None = None
    baseline_drift_baseline_max: float | None = None
    baseline_drift_pha_min: float | None = None
    baseline_drift_pha_max: float | None = None

    def validated(self) -> PulseAnalysisConfig:
        if self.max_points_per_trace is not None and self.max_points_per_trace < 1:
            raise ValueError("max_points_per_trace must be positive or null.")
        if self.max_display_traces is not None and self.max_display_traces < 1:
            raise ValueError("max_display_traces must be positive or null.")
        if self.valid_pulse_range_start < 0:
            raise ValueError("valid_pulse_range_start must be non-negative.")
        if self.valid_pulse_range_stop < self.valid_pulse_range_start:
            raise ValueError(
                "valid_pulse_range_stop must be greater than or equal to "
                "valid_pulse_range_start."
            )
        if self.valid_pulse_diff_threshold < 0:
            raise ValueError("valid_pulse_diff_threshold must be non-negative.")
        if isinstance(self.spectrum_bins, int) and self.spectrum_bins < 1:
            raise ValueError("spectrum_bins must be positive when it is an integer.")
        if isinstance(self.spectrum_bins, str) and self.spectrum_bins != "auto":
            raise ValueError('spectrum_bins must be "auto" or a positive integer.')
        if (
            self.histogram_min is not None
            and self.histogram_max is not None
            and self.histogram_max <= self.histogram_min
        ):
            raise ValueError("histogram_max must be greater than histogram_min.")
        if self.spectrum_chunk_size < 1:
            raise ValueError("spectrum_chunk_size must be positive.")
        if (
            self.baseline_drift_baseline_min is not None
            and self.baseline_drift_baseline_max is not None
            and self.baseline_drift_baseline_max <= self.baseline_drift_baseline_min
        ):
            raise ValueError(
                "baseline_drift_baseline_max must be greater than "
                "baseline_drift_baseline_min."
            )
        if (
            self.baseline_drift_pha_min is not None
            and self.baseline_drift_pha_max is not None
            and self.baseline_drift_pha_max <= self.baseline_drift_pha_min
        ):
            raise ValueError(
                "baseline_drift_pha_max must be greater than baseline_drift_pha_min."
            )
        return self

    def with_updates(self, **updates: Any) -> PulseAnalysisConfig:
        return replace(self, **updates).validated()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config() -> PulseAnalysisConfig:
    return PulseAnalysisConfig().validated()


def load_config(path: Path | None) -> PulseAnalysisConfig:
    if path is None:
        return default_config()
    loaded = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(loaded, dict):
        raise ValueError(f"Pulse config must be a mapping: {path}")
    config = default_config()
    allowed = set(config.to_dict())
    unknown = sorted(set(loaded) - allowed)
    if unknown:
        raise ValueError(f"Unknown pulse config keys in {path}: {', '.join(unknown)}")
    return config.with_updates(**loaded)


def save_config(config: PulseAnalysisConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(config.to_dict()), f=path)
    return path


def parse_config_updates(
    config: PulseAnalysisConfig,
    updates: dict[str, str],
) -> PulseAnalysisConfig:
    parsed: dict[str, Any] = {}
    for key, value in updates.items():
        text = value.strip()
        if key in {
            "max_points_per_trace",
            "max_display_traces",
            "spectrum_chunk_size",
        }:
            parsed[key] = None if text.lower() in {"", "none", "null"} else int(text)
        elif key in {"valid_pulse_range_start", "valid_pulse_range_stop"}:
            parsed[key] = int(text)
        elif key == "valid_pulse_diff_threshold":
            parsed[key] = float(text)
        elif key == "spectrum_bins":
            parsed[key] = "auto" if text.lower() == "auto" else int(text)
        elif key in {
            "histogram_min",
            "histogram_max",
            "baseline_drift_baseline_min",
            "baseline_drift_baseline_max",
            "baseline_drift_pha_min",
            "baseline_drift_pha_max",
        }:
            parsed[key] = None if text.lower() in {"", "none", "null"} else float(text)
        elif key in {
            "negative_pulses",
            "optimal_filter_template_normalize",
            "baseline_drift_correction",
        }:
            parsed[key] = text.lower() in {"1", "true", "yes", "on"}
        else:
            raise ValueError(f"Unknown pulse config key: {key}")
    return config.with_updates(**parsed)
