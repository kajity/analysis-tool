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
    fixed_slope: float | None = None,
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

    if not enabled:
        return corrected, 0.0, intercept, reference_baseline, fit_count, fit_mask
    if fixed_slope is None:
        if fit_count < 2 or np.ptp(baseline_values[fit_mask]) == 0:
            return corrected, 0.0, intercept, reference_baseline, fit_count, fit_mask
        slope, intercept = np.polyfit(
            baseline_values[fit_mask], pha_values[fit_mask], deg=1
        )
    else:
        if fit_count < 1:
            return corrected, 0.0, intercept, reference_baseline, fit_count, fit_mask
        slope = float(fixed_slope)
        intercept = float(
            np.mean(pha_values[fit_mask] - slope * baseline_values[fit_mask])
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


def baseline_pha_kmeans_clusters(
    baseline: np.ndarray,
    pha: np.ndarray,
    fit_mask: np.ndarray,
    slope: float,
    cluster_count: int = 2,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Cluster selected baseline/PHA points by c = pha - slope * baseline.

    The common slope and per-cluster c centers are updated every iteration. Labels
    are returned for the full input length. Points outside `fit_mask` are
    assigned -1. Cluster labels are ordered by centroid from low to high.
    """
    if cluster_count < 1:
        raise ValueError("cluster_count must be positive.")
    baseline_values = np.asarray(baseline, dtype=float)
    pha_values = np.asarray(pha, dtype=float)
    selected = (
        np.asarray(fit_mask, dtype=bool)
        & np.isfinite(baseline_values)
        & np.isfinite(pha_values)
    )
    labels = np.full(pha_values.shape, -1, dtype=int)
    x_values = baseline_values[selected]
    y_values = pha_values[selected]
    if y_values.size == 0:
        return labels, np.array([], dtype=float), float(slope), 0

    current_slope = float(slope)
    c_values = y_values - current_slope * x_values
    centers = np.linspace(
        float(np.min(c_values)), float(np.max(c_values)), cluster_count
    )
    selected_labels = np.full(y_values.shape, -1, dtype=int)
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        distances = np.abs(c_values[:, np.newaxis] - centers[np.newaxis, :])
        next_labels = np.argmin(distances, axis=1)
        next_slope, next_centers = _fit_common_slope_and_cluster_centers(
            x_values,
            y_values,
            next_labels,
            cluster_count,
            current_slope,
            centers,
        )
        order = np.argsort(next_centers)
        remap = np.empty_like(order)
        remap[order] = np.arange(cluster_count)
        next_labels = remap[next_labels]
        next_centers = next_centers[order]
        next_c_values = y_values - next_slope * x_values

        if (
            np.array_equal(next_labels, selected_labels)
            and np.allclose(next_centers, centers)
            and np.isclose(next_slope, current_slope)
        ):
            selected_labels = next_labels
            centers = next_centers
            current_slope = next_slope
            c_values = next_c_values
            break

        selected_labels = next_labels
        centers = next_centers
        current_slope = next_slope
        c_values = next_c_values

    labels[selected] = selected_labels
    return labels, centers, current_slope, iterations


def _fit_common_slope_and_cluster_centers(
    baseline: np.ndarray,
    pha: np.ndarray,
    labels: np.ndarray,
    cluster_count: int,
    previous_slope: float,
    previous_centers: np.ndarray,
) -> tuple[float, np.ndarray]:
    nonempty_clusters = [
        cluster_index
        for cluster_index in range(cluster_count)
        if np.any(labels == cluster_index)
    ]
    if not nonempty_clusters:
        return previous_slope, previous_centers.copy()

    columns = [baseline]
    for cluster_index in nonempty_clusters:
        columns.append((labels == cluster_index).astype(float))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, pha, rcond=None)
    slope = float(coefficients[0])
    centers = previous_centers.copy()
    for coefficient_index, cluster_index in enumerate(nonempty_clusters, start=1):
        centers[cluster_index] = float(coefficients[coefficient_index])
    return slope, centers


def cluster_pha_timeline(
    pha: np.ndarray,
    selected_mask: np.ndarray,
    boundary: float,
    neighbor_count: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign PHA timeline clusters from a boundary and nearby selected points.

    Each selected point is first classified by the boundary. The final class is
    resolved by a majority vote over up to `neighbor_count` selected points before
    and after that point in timeline order. Ties keep the boundary class.
    """
    pha_values = np.asarray(pha, dtype=float)
    selected = np.asarray(selected_mask, dtype=bool)
    upper_by_boundary = selected & (pha_values >= boundary)
    assigned_upper = upper_by_boundary.copy()
    selected_indices = np.flatnonzero(selected)
    neighbor_count = max(0, int(neighbor_count))

    for position, index in enumerate(selected_indices):
        if neighbor_count == 0:
            break
        start = max(0, position - neighbor_count)
        stop = min(selected_indices.size, position + neighbor_count + 1)
        neighbor_indices = np.concatenate(
            (selected_indices[start:position], selected_indices[position + 1 : stop])
        )
        if neighbor_indices.size == 0:
            continue
        upper_count = int(np.count_nonzero(upper_by_boundary[neighbor_indices]))
        lower_count = int(neighbor_indices.size - upper_count)
        if upper_count > lower_count:
            assigned_upper[index] = True
        elif lower_count > upper_count:
            assigned_upper[index] = False

    upper_cluster = selected & assigned_upper
    lower_cluster = selected & ~assigned_upper
    return lower_cluster, upper_cluster


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
