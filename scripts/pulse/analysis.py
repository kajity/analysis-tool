from __future__ import annotations

import numpy as np


def differential_signal(aligned_signal: np.ndarray) -> np.ndarray:
    """Calculate adjacent-sample differences for aligned traces.

    Inputs:
        aligned_signal: Baseline-aligned signal with shape (traces, samples).
    Description:
        Computes the first difference along the sample axis for each trace.
    Returns:
        Differential signal with one fewer sample, shape (traces, samples - 1).
    """
    return np.diff(aligned_signal, axis=1)


def valid_pulse_mask(
    differential_signal: np.ndarray,
    start: int,
    stop: int,
    threshold: float,
) -> np.ndarray:
    """Identify accepted traces from differences outside the valid pulse range.

    Inputs:
        differential_signal: Difference signal calculated by `differential_signal()`.
        start: Start index of the valid pulse range on the differential axis.
        stop: Stop index of the valid pulse range on the differential axis.
        threshold: Maximum allowed absolute difference outside the valid range.
    Description:
        Accepts traces where every absolute difference outside start:stop is less
        than or equal to threshold.
    Returns:
        Boolean accepted/rejected mask for each trace, shape (traces,).
    """
    if start < 0 or stop < start:
        raise ValueError("valid pulse range must be a non-negative (start, stop) pair.")
    if threshold < 0:
        raise ValueError("valid pulse threshold must be non-negative.")

    diff_count = differential_signal.shape[1]
    outside_valid_range = np.ones(diff_count, dtype=bool)
    outside_valid_range[min(start, diff_count) : min(stop, diff_count)] = False
    if not np.any(outside_valid_range):
        return np.ones(differential_signal.shape[0], dtype=bool)

    # A pulse is valid only when every diff outside the configured pulse window
    # stays within the allowed threshold.
    outside_diff = np.abs(differential_signal[:, outside_valid_range])
    return np.all(outside_diff <= threshold, axis=1)


def pulse_heights(
    shaped_signal: np.ndarray,
    vertical_resolution: float,
    negative_pulses: bool,
) -> np.ndarray:
    """Calculate pulse heights from accepted traces.

    Inputs:
        shaped_signal: Baseline-aligned signal after rejection, shape
            (traces, samples).
        vertical_resolution: Scale factor converting ADC counts to vertical units.
        negative_pulses: If True, negative pulses are reported as positive heights.
    Description:
        Uses the minimum sample value in each trace as the pulse height and applies
        vertical_resolution. When negative_pulses is True, the sign is inverted.
    Returns:
        Pulse height array for accepted traces, shape (traces,).
    """
    if shaped_signal.size == 0:
        return np.array([], dtype=float)

    heights = np.min(shaped_signal, axis=1) * vertical_resolution
    if negative_pulses:
        heights *= -1
    return heights


def noise_records(background_signal: np.ndarray) -> np.ndarray:
    """Create mean-subtracted noise records from background samples.

    Inputs:
        background_signal: Background-window signal for each trace, shape
            (traces, samples).
    Description:
        Computes the mean of each trace and subtracts it from that trace.
    Returns:
        Background noise signal with the DC component removed, or an empty float
        array when the input is empty.
    """
    if background_signal.size == 0:
        return np.array([], dtype=float)
    return background_signal - np.average(background_signal, axis=1, keepdims=True)


def filter_template_fft(
    template_fft: np.ndarray,
    noise_psd: np.ndarray,
) -> np.ndarray:
    """Build the optimal-filter template FFT from signal FFT and noise PSD.

    Inputs:
        template_fft: rFFT of the accepted-pulse average template.
        noise_psd: Power spectral density estimated from background noise records.
    Description:
        Computes template_fft / noise_psd for frequency bins with positive noise
        power. The DC bin is always excluded because background records are
        mean-subtracted.
    Returns:
        Complex array with the same shape as template_fft. Unusable bins are zero.
    """
    usable_bins = min(template_fft.size, noise_psd.size)
    if usable_bins == 0:
        return np.array([], dtype=complex)

    result = np.zeros_like(template_fft)
    noise_power = noise_psd[:usable_bins]
    positive = noise_power > 0
    # Noise records are mean-subtracted, so the DC bin has no useful variance
    # estimate and would otherwise dominate signal/noise^2.
    positive[0] = False
    usable_result = result[:usable_bins]
    usable_result[positive] = (
        template_fft[:usable_bins][positive] / noise_power[positive]
    )
    return result


def filter_height_normalization(
    template: np.ndarray,
    filter_template: np.ndarray,
) -> float:
    """Calculate the normalization factor for optimal-filter pulse heights.

    Inputs:
        template: Time-domain template built from the accepted-pulse average.
        filter_template: Time-domain optimal-filter template.
    Description:
        Computes the dot product over the shared sample range of both templates.
    Returns:
        Normalization factor used to divide pulse-height projections, or 0.0 when
        either input is empty.
    """
    if template.size == 0 or filter_template.size == 0:
        return 0.0
    usable_samples = min(template.size, filter_template.size)
    return float(template[:usable_samples] @ filter_template[:usable_samples])


def baseline_drift_corrected_pulse_heights(
    baseline: np.ndarray,
    pha: np.ndarray,
    enabled: bool,
    baseline_min: float | None = None,
    baseline_max: float | None = None,
    pha_min: float | None = None,
    pha_max: float | None = None,
) -> tuple[np.ndarray, float, float, float, int, np.ndarray]:
    """Remove a linear baseline drift term from optimal-filter pulse heights.

    Fit points are finite baseline/PHA pairs that also fall inside the optional
    baseline and PHA ranges. The corrected values preserve the pulse-height scale
    at the mean baseline of the fit points.
    """
    baseline_values = np.asarray(baseline, dtype=float)
    pha_values = np.asarray(pha, dtype=float)
    corrected = pha_values.copy()
    finite = np.isfinite(baseline_values) & np.isfinite(pha_values)
    fit_mask = finite.copy()
    if baseline_min is not None:
        fit_mask &= baseline_values >= baseline_min
    if baseline_max is not None:
        fit_mask &= baseline_values <= baseline_max
    if pha_min is not None:
        fit_mask &= pha_values >= pha_min
    if pha_max is not None:
        fit_mask &= pha_values <= pha_max

    fit_count = int(np.count_nonzero(fit_mask))
    if fit_count:
        reference_baseline = float(np.mean(baseline_values[fit_mask]))
        intercept = float(np.mean(pha_values[fit_mask]))
    else:
        reference_baseline = float("nan")
        intercept = float("nan")

    if not enabled or fit_count < 2 or np.ptp(baseline_values[fit_mask]) == 0:
        return corrected, 0.0, intercept, reference_baseline, fit_count, fit_mask

    slope, intercept = np.polyfit(
        baseline_values[fit_mask], pha_values[fit_mask], deg=1
    )
    corrected[finite] = pha_values[finite] - slope * (
        baseline_values[finite] - reference_baseline
    )
    return (
        corrected,
        float(slope),
        float(intercept),
        reference_baseline,
        fit_count,
        fit_mask,
    )


def histogram_range(
    values: np.ndarray,
    histogram_min: float | None,
    histogram_max: float | None,
) -> tuple[float, float] | None:
    """Resolve the histogram range from configuration and data values.

    Inputs:
        values: Values to histogram.
        histogram_min: Explicit lower bound. If None, the minimum value is used.
        histogram_max: Explicit upper bound. If None, the maximum value is used.
    Description:
        Returns None for NumPy's automatic range when both bounds are unset. When
        only one bound is set, fills the other bound from the data min/max.
    Returns:
        (lower, upper) for np.histogram(range=...), or None for automatic range.
    """
    if histogram_min is None and histogram_max is None:
        return None
    if values.size == 0:
        return None
    lower = float(np.min(values)) if histogram_min is None else histogram_min
    upper = float(np.max(values)) if histogram_max is None else histogram_max
    if upper <= lower:
        raise ValueError("histogram range must have max greater than min.")
    return lower, upper
