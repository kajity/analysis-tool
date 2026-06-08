from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from scripts.pulse import analysis as pulse_analysis
from scripts.pulse.config import PulseAnalysisConfig, load_config, save_config
from scripts.pulse.datasource import PulseDataSource
from scripts.pulse.main import parse_args
from scripts.pulse.pipeline import (
    PhaClusterResult,
    PhaTimelineResult,
    PulsePipeline,
    PulseStage,
)
from scripts.pulse.pulse_io import open_hdf5_pulse_data
from scripts.pulse.rendering import PulsePlotRenderer
from scripts.pulse.workflow import (
    PulseWorkflow,
    PulseWorkflowController,
    _fit_terminal_line,
    pulse_steps,
    save_pulse_plots,
)


def write_test_hdf5(path: Path) -> None:
    wave = np.array(
        [
            [1, -1, 1, -1, 0, -1, -3, -1],
            [3, -3, 3, -3, 0, 10, -1, 0],
            [2, -2, 2, -2, 0, -2, -2, 0],
            [1, 1, -1, -1, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    with h5py.File(path, "w") as h5_file:
        waveform = h5_file.create_group("waveform")
        waveform.create_dataset("wave", data=wave)
        waveform.create_dataset("vres", data=np.float32(0.5))
        waveform.create_dataset("hres", data=np.float32(0.1))


def write_drift_test_hdf5(path: Path) -> None:
    rows = []
    for baseline in [0.0, 2.0, 4.0, 6.0]:
        amplitude = 10.0 + 2.0 * baseline
        background = baseline + np.array([1.0, -1.0, 1.0, -1.0])
        signal = baseline + np.array([0.0, -amplitude, -0.5 * amplitude, 0.0])
        rows.append(np.concatenate([background, signal]))
    wave = np.asarray(rows, dtype=np.float32)
    with h5py.File(path, "w") as h5_file:
        waveform = h5_file.create_group("waveform")
        waveform.create_dataset("wave", data=wave)
        waveform.create_dataset("vres", data=np.float32(1.0))
        waveform.create_dataset("hres", data=np.float32(0.1))


class PulseCliArgumentTest(unittest.TestCase):
    def test_drift_correction_can_be_set_from_args(self) -> None:
        enabled = parse_args(["input.hdf5", "--drift-correction"])
        disabled = parse_args(["input.hdf5", "--no-drift-correction"])

        self.assertTrue(enabled.baseline_drift_correction)
        self.assertFalse(disabled.baseline_drift_correction)

    def test_pha_clustering_can_be_set_from_args(self) -> None:
        enabled = parse_args(["input.hdf5", "--pha-clustering"])
        disabled = parse_args(["input.hdf5", "--no-pha-clustering"])

        self.assertTrue(enabled.pha_clustering)
        self.assertFalse(disabled.pha_clustering)


class PulseWorkflowNavigationTest(unittest.TestCase):
    def test_can_finish_from_any_step(self) -> None:
        workflow = PulseWorkflow(steps=("one", "two", "three"))

        self.assertTrue(workflow.can_finish)
        self.assertTrue(workflow.finish())
        self.assertTrue(workflow.finished)
        self.assertFalse(workflow.can_finish)

    def test_can_jump_directly_to_step(self) -> None:
        workflow = PulseWorkflow(steps=("one", "two", "three"))

        self.assertTrue(workflow.go_to_step(2))
        self.assertEqual(workflow.current_step, "three")
        self.assertEqual(workflow.events, ["step:three"])
        self.assertFalse(workflow.go_to_step(3))

    def test_default_steps_use_reduction_and_ph_labels(self) -> None:
        self.assertIn(PulseStage.REDUCTION.value, PulseWorkflow().steps)
        self.assertIn(PulseStage.PH.value, PulseWorkflow().steps)
        self.assertIn(PulseStage.PHA_TIMELINE.value, PulseWorkflow().steps)
        self.assertNotIn("Reject/Shaping", PulseWorkflow().steps)
        self.assertNotIn("Preprocess", PulseWorkflow().steps)
        self.assertNotIn("Spectrum", PulseWorkflow().steps)
        self.assertNotIn(
            PulseStage.DRIFT_CORRECTED_PHA.value,
            PulseWorkflow().steps,
        )

    def test_drift_correction_steps_are_enabled_from_config(self) -> None:
        steps = pulse_steps(PulseAnalysisConfig(baseline_drift_correction=True))

        self.assertIn(PulseStage.DRIFT_CORRECTED_PHA.value, steps)

    def test_pha_cluster_step_is_enabled_from_config(self) -> None:
        steps = pulse_steps(PulseAnalysisConfig(pha_clustering=True))

        self.assertIn(PulseStage.PHA_CLUSTER.value, steps)
        self.assertIn(PulseStage.LOWER_CLUSTER_PHA.value, steps)
        self.assertLess(
            steps.index(PulseStage.BASELINE_PHA.value),
            steps.index(PulseStage.PHA_CLUSTER.value),
        )
        self.assertLess(
            steps.index(PulseStage.PHA_CLUSTER.value),
            steps.index(PulseStage.LOWER_CLUSTER_PHA.value),
        )


class PulseProgressOutputTest(unittest.TestCase):
    def test_fit_terminal_line_truncates_to_width(self) -> None:
        text = "Saving figure 10/10: Drift-Corrected PHA -> /very/long/output/path.png"

        fitted = _fit_terminal_line(text, 32)

        self.assertLessEqual(len(fitted), 32)
        self.assertIn("...", fitted)

    def test_fit_terminal_line_handles_tiny_width(self) -> None:
        self.assertEqual(_fit_terminal_line("abcdef", 2), "ab")


class PulseWorkflowControllerTest(unittest.TestCase):
    def test_finish_prints_minimal_analysis_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=1,
                    valid_pulse_range_stop=2,
                    valid_pulse_diff_threshold=3,
                    spectrum_chunk_size=2,
                ).validated()
                controller = PulseWorkflowController(
                    pulse_data,
                    max_points_per_trace=None,
                    max_traces=None,
                    config=config,
                )
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        controller.render()
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        controller.finish()
                finally:
                    controller.ui.close()

            terminal_text = output.getvalue()
            self.assertIn("[pulse] Analysis summary", terminal_text)
            self.assertIn("accepted: 3", terminal_text)
            self.assertIn("rejected: 1", terminal_text)
            self.assertIn("PHA points: 3", terminal_text)
            self.assertIn("drift correction: disabled", terminal_text)
            self.assertNotIn("UI info summary", terminal_text)
            self.assertNotIn("valid range:", terminal_text)

    def test_analysis_summary_includes_drift_fit_result_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_drift_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=0,
                    valid_pulse_range_stop=3,
                    valid_pulse_diff_threshold=0,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                    baseline_drift_correction=True,
                ).validated()
                controller = PulseWorkflowController(
                    pulse_data,
                    max_points_per_trace=None,
                    max_traces=None,
                    config=config,
                )
                try:
                    summary = controller.analysis_summary_text()
                finally:
                    controller.ui.close()

            self.assertIn("drift correction: enabled", summary)
            self.assertIn("drift slope:", summary)
            self.assertIn("drift reference baseline:", summary)
            self.assertIn("drift fit points:", summary)

    def test_analysis_summary_includes_pha_cluster_result_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=1,
                    valid_pulse_range_stop=2,
                    valid_pulse_diff_threshold=3,
                    spectrum_chunk_size=2,
                    pha_clustering=True,
                    pha_cluster_boundary=0.0,
                ).validated()
                controller = PulseWorkflowController(
                    pulse_data,
                    max_points_per_trace=None,
                    max_traces=None,
                    config=config,
                )
                try:
                    summary = controller.analysis_summary_text()
                finally:
                    controller.ui.close()

            self.assertIn("PHA clustering: enabled", summary)
            self.assertIn("cluster selected points:", summary)
            self.assertIn("cluster boundary: 0", summary)

    def test_config_values_include_pha_cluster_controls_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                controller = PulseWorkflowController(
                    pulse_data,
                    max_points_per_trace=None,
                    max_traces=None,
                    config=PulseAnalysisConfig(
                        pha_clustering=True,
                        pha_cluster_pha_min=1.0,
                        pha_cluster_pha_max=2.0,
                        pha_cluster_boundary=1.5,
                    ),
                )
                try:
                    values = controller.config_values()
                finally:
                    controller.ui.close()

            self.assertEqual(values["pha_cluster_pha_min"], "1")
            self.assertEqual(values["pha_cluster_pha_max"], "2")
            self.assertEqual(values["pha_cluster_boundary"], "1.5")

    def test_config_values_include_drift_cluster_count_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                controller = PulseWorkflowController(
                    pulse_data,
                    max_points_per_trace=None,
                    max_traces=None,
                    config=PulseAnalysisConfig(
                        baseline_drift_correction=True,
                        baseline_drift_clustering=True,
                        baseline_drift_cluster_count=3,
                    ),
                )
                try:
                    values = controller.config_values()
                finally:
                    controller.ui.close()

            self.assertEqual(values["baseline_drift_cluster_count"], "3")

    def test_optional_config_text_limits_float_precision(self) -> None:
        self.assertEqual(
            PulseWorkflowController._optional_config_text(None, 1.23456789),
            "1.23457",
        )
        self.assertEqual(PulseWorkflowController._optional_config_text(None, None), "")


class PulseAnalysisFunctionTest(unittest.TestCase):
    def test_valid_pulse_mask_rejects_threshold_outside_valid_range(self) -> None:
        diff = np.array(
            [
                [0.0, 10.0, 1.0],
                [4.0, 10.0, 1.0],
            ]
        )

        mask = pulse_analysis.valid_pulse_mask(
            diff,
            start=1,
            stop=2,
            threshold=3.0,
        )

        np.testing.assert_array_equal(mask, np.array([True, False]))

    def test_pulse_heights_respect_negative_pulse_sign(self) -> None:
        shaped = np.array(
            [
                [0.0, -2.0, -1.0],
                [0.0, -4.0, 1.0],
            ]
        )

        np.testing.assert_allclose(
            pulse_analysis.pulse_heights(shaped, 0.5, negative_pulses=True),
            np.array([1.0, 2.0]),
        )
        np.testing.assert_allclose(
            pulse_analysis.pulse_heights(shaped, 0.5, negative_pulses=False),
            np.array([-1.0, -2.0]),
        )

    def test_histogram_range_accepts_both_and_one_sided_limits(self) -> None:
        values = np.array([0.0, 1.0, 2.0])

        self.assertEqual(
            pulse_analysis.histogram_range(values, 0.5, 1.5),
            (0.5, 1.5),
        )
        self.assertEqual(
            pulse_analysis.histogram_range(values, 0.5, None),
            (0.5, 2.0),
        )
        self.assertEqual(
            pulse_analysis.histogram_range(values, None, 1.5),
            (0.0, 1.5),
        )
        self.assertIsNone(pulse_analysis.histogram_range(values, None, None))

    def test_noise_records_subtract_row_means(self) -> None:
        background = np.array(
            [
                [1.0, 3.0, 5.0],
                [2.0, 2.0, 2.0],
            ]
        )

        np.testing.assert_allclose(
            pulse_analysis.noise_records(background),
            np.array(
                [
                    [-2.0, 0.0, 2.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )

    def test_filter_template_fft_excludes_dc_bin(self) -> None:
        template_fft = np.array([10.0 + 0j, 6.0 + 0j, 8.0 + 0j])
        noise_psd = np.array([1.0, 2.0, 0.0])

        np.testing.assert_allclose(
            pulse_analysis.filter_template_fft(template_fft, noise_psd),
            np.array([0.0 + 0j, 3.0 + 0j, 0.0 + 0j]),
        )

    def test_baseline_drift_correction_preserves_mean_pulse_height(self) -> None:
        pha_corrected, slope, intercept, reference_baseline, fit_count, fit_mask = (
            pulse_analysis.baseline_drift_corrected_pulse_heights(
                np.array([0.0, 1.0, 2.0]),
                np.array([10.0, 12.0, 14.0]),
                enabled=True,
            )
        )

        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 10.0)
        self.assertAlmostEqual(reference_baseline, 1.0)
        self.assertEqual(fit_count, 3)
        np.testing.assert_array_equal(fit_mask, np.array([True, True, True]))
        np.testing.assert_allclose(pha_corrected, np.array([12.0, 12.0, 12.0]))

    def test_baseline_drift_correction_can_use_fixed_slope(self) -> None:
        pha_corrected, slope, intercept, reference_baseline, fit_count, fit_mask = (
            pulse_analysis.baseline_drift_corrected_pulse_heights(
                np.array([0.0, 1.0, 2.0]),
                np.array([10.0, 13.0, 16.0]),
                enabled=True,
                fixed_slope=2.0,
            )
        )

        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 11.0)
        self.assertAlmostEqual(reference_baseline, 1.0)
        self.assertEqual(fit_count, 3)
        np.testing.assert_array_equal(fit_mask, np.array([True, True, True]))
        np.testing.assert_allclose(pha_corrected, np.array([12.0, 13.0, 14.0]))

    def test_baseline_pha_kmeans_clusters_shifted_parallel_groups(self) -> None:
        labels, centers, slope, iterations = (
            pulse_analysis.baseline_pha_kmeans_clusters(
                baseline=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
                pha=np.array([10.0, 12.0, 14.0, 20.0, 22.0, 24.0]),
                fit_mask=np.array([True, True, True, True, True, True]),
                slope=2.0,
            )
        )

        np.testing.assert_array_equal(labels, np.array([0, 0, 0, 1, 1, 1]))
        np.testing.assert_allclose(centers, np.array([10.0, 20.0]))
        self.assertAlmostEqual(slope, 2.0)
        self.assertGreaterEqual(iterations, 1)

    def test_baseline_pha_kmeans_updates_slope_and_c_values(self) -> None:
        labels, centers, slope, _ = pulse_analysis.baseline_pha_kmeans_clusters(
            baseline=np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            pha=np.array([10.0, 13.0, 16.0, 20.0, 23.0, 26.0]),
            fit_mask=np.array([True, True, True, True, True, True]),
            slope=0.0,
        )

        np.testing.assert_array_equal(labels, np.array([0, 0, 0, 1, 1, 1]))
        self.assertAlmostEqual(slope, 3.0)
        np.testing.assert_allclose(centers, np.array([10.0, 20.0]))

    def test_baseline_pha_kmeans_accepts_configurable_cluster_count(self) -> None:
        labels, centers, slope, _ = pulse_analysis.baseline_pha_kmeans_clusters(
            baseline=np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            pha=np.array([10.0, 12.0, 20.0, 22.0, 30.0, 32.0]),
            fit_mask=np.array([True, True, True, True, True, True]),
            slope=2.0,
            cluster_count=3,
        )

        np.testing.assert_array_equal(labels, np.array([0, 0, 1, 1, 2, 2]))
        np.testing.assert_allclose(centers, np.array([10.0, 20.0, 30.0]))
        self.assertAlmostEqual(slope, 2.0)

    def test_baseline_drift_correction_can_be_disabled(self) -> None:
        pha_corrected, slope, _, _, fit_count, fit_mask = (
            pulse_analysis.baseline_drift_corrected_pulse_heights(
                np.array([0.0, 1.0, 2.0]),
                np.array([10.0, 12.0, 14.0]),
                enabled=False,
            )
        )

        self.assertEqual(slope, 0.0)
        self.assertEqual(fit_count, 3)
        np.testing.assert_array_equal(fit_mask, np.array([True, True, True]))
        np.testing.assert_allclose(pha_corrected, np.array([10.0, 12.0, 14.0]))

    def test_baseline_drift_correction_skips_constant_baseline(self) -> None:
        pha_corrected, slope, _, _, fit_count, fit_mask = (
            pulse_analysis.baseline_drift_corrected_pulse_heights(
                np.array([1.0, 1.0, 1.0]),
                np.array([10.0, 12.0, 14.0]),
                enabled=True,
            )
        )

        self.assertEqual(slope, 0.0)
        self.assertEqual(fit_count, 3)
        np.testing.assert_array_equal(fit_mask, np.array([True, True, True]))
        np.testing.assert_allclose(pha_corrected, np.array([10.0, 12.0, 14.0]))

    def test_baseline_drift_correction_uses_configured_fit_ranges(self) -> None:
        _, slope, _, reference_baseline, fit_count, fit_mask = (
            pulse_analysis.baseline_drift_corrected_pulse_heights(
                np.array([0.0, 1.0, 2.0, 3.0]),
                np.array([10.0, 12.0, 14.0, 50.0]),
                enabled=True,
                baseline_min=0.5,
                baseline_max=2.5,
                pha_min=11.0,
                pha_max=15.0,
            )
        )

        self.assertEqual(fit_count, 2)
        np.testing.assert_array_equal(fit_mask, np.array([False, True, True, False]))
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(reference_baseline, 1.5)

    def test_analysis_functions_accept_empty_arrays(self) -> None:
        empty_records = np.empty((0, 3))

        self.assertEqual(
            pulse_analysis.differential_signal(empty_records).shape, (0, 2)
        )
        self.assertEqual(
            pulse_analysis.valid_pulse_mask(empty_records, 1, 2, 3.0).shape,
            (0,),
        )
        self.assertEqual(
            pulse_analysis.pulse_heights(empty_records, 0.5, True).shape,
            (0,),
        )
        self.assertEqual(pulse_analysis.noise_records(empty_records).shape, (0,))
        self.assertEqual(
            pulse_analysis.filter_template_fft(
                np.array([], dtype=complex),
                np.array([], dtype=float),
            ).shape,
            (0,),
        )
        self.assertEqual(
            pulse_analysis.filter_height_normalization(
                np.array([], dtype=float),
                np.array([], dtype=float),
            ),
            0.0,
        )
        self.assertIsNone(pulse_analysis.histogram_range(np.array([]), 0.0, 1.0))


class PulsePipelineTest(unittest.TestCase):
    def test_spectrum_counts_shaped_pulse_heights(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    max_points_per_trace=None,
                    max_display_traces=None,
                    valid_pulse_range_start=1,
                    valid_pulse_range_stop=2,
                    valid_pulse_diff_threshold=3,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                spectrum = pipeline.ph_spectrum()

            self.assertEqual(spectrum.accepted_count, 3)
            self.assertEqual(spectrum.rejected_count, 1)
            np.testing.assert_allclose(
                np.sort(spectrum.pulse_heights),
                np.array([0.0, 1.0, 1.5]),
            )
            self.assertEqual(int(spectrum.counts.sum()), 3)

    def test_config_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config = PulseAnalysisConfig(
                valid_pulse_range_start=10,
                valid_pulse_range_stop=20,
                valid_pulse_diff_threshold=42.0,
                spectrum_bins=16,
                baseline_drift_correction=True,
                baseline_drift_baseline_min=1.0,
                baseline_drift_baseline_max=2.0,
                baseline_drift_pha_min=10.0,
                baseline_drift_pha_max=20.0,
                baseline_drift_clustering=True,
                baseline_drift_cluster_count=3,
                baseline_drift_cluster_slope=2.0,
            ).validated()

            save_config(config, config_path)
            self.assertEqual(load_config(config_path), config)

    def test_histogram_range_limits_spectrum_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    max_points_per_trace=None,
                    max_display_traces=None,
                    valid_pulse_range_start=1,
                    valid_pulse_range_stop=2,
                    valid_pulse_diff_threshold=3,
                    spectrum_bins=2,
                    histogram_min=0.5,
                    histogram_max=1.5,
                    spectrum_chunk_size=2,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                spectrum = pipeline.ph_spectrum()

            self.assertEqual(int(spectrum.counts.sum()), 2)
            np.testing.assert_allclose(spectrum.bin_edges, np.array([0.5, 1.0, 1.5]))

    def test_histogram_range_accepts_only_min_or_max(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                min_config = PulseAnalysisConfig(
                    max_points_per_trace=None,
                    max_display_traces=None,
                    valid_pulse_range_start=1,
                    valid_pulse_range_stop=2,
                    valid_pulse_diff_threshold=3,
                    spectrum_bins=2,
                    histogram_min=0.5,
                    spectrum_chunk_size=2,
                ).validated()
                min_spectrum = PulsePipeline(
                    PulseDataSource(pulse_data),
                    min_config,
                ).ph_spectrum()

            with open_hdf5_pulse_data(input_path) as pulse_data:
                max_config = min_config.with_updates(
                    histogram_min=None,
                    histogram_max=1.0,
                )
                max_spectrum = PulsePipeline(
                    PulseDataSource(pulse_data),
                    max_config,
                ).ph_spectrum()

            self.assertEqual(int(min_spectrum.counts.sum()), 2)
            np.testing.assert_allclose(min_spectrum.bin_edges[0], 0.5)
            self.assertEqual(int(max_spectrum.counts.sum()), 2)
            np.testing.assert_allclose(max_spectrum.bin_edges[-1], 1.0)

    def test_optimal_filter_prep_estimates_template_and_noise_psd(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    max_points_per_trace=None,
                    max_display_traces=None,
                    valid_pulse_range_start=1,
                    valid_pulse_range_stop=2,
                    valid_pulse_diff_threshold=3,
                    spectrum_chunk_size=2,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                prep = pipeline.optimal_filter_prep()

            self.assertEqual(prep.accepted_count, 3)
            self.assertEqual(prep.rejected_count, 1)
            np.testing.assert_allclose(
                prep.template,
                np.array([0.0, 0.6, 1.0, 0.2]),
                rtol=1e-6,
            )
            self.assertEqual(prep.template_fft.shape, (3,))
            self.assertEqual(prep.noise_psd.shape, (3,))
            self.assertEqual(prep.noise_fft.shape, (3,))
            self.assertEqual(prep.filter_template_fft.shape, (3,))
            self.assertEqual(prep.filter_template.shape, (4,))
            self.assertGreater(np.max(np.abs(prep.filter_template)), 0)

    def test_pha_spectrum_counts_projected_pulses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=1,
                    valid_pulse_range_stop=2,
                    valid_pulse_diff_threshold=3,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                result = pipeline.pha_spectrum()

            self.assertEqual(result.accepted_count, 3)
            self.assertEqual(result.rejected_count, 1)
            self.assertEqual(int(result.counts.sum()), 3)
            self.assertEqual(result.pulse_heights.shape, (3,))
            self.assertGreater(abs(result.normalization), 0)
            self.assertTrue(np.all(np.isfinite(result.pulse_heights)))

    def test_baseline_pha_pairs_baseline_and_pha(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=1,
                    valid_pulse_range_stop=2,
                    valid_pulse_diff_threshold=3,
                    spectrum_chunk_size=2,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                result = pipeline.baseline_pha()

            self.assertEqual(result.accepted_count, 3)
            self.assertEqual(result.rejected_count, 1)
            self.assertEqual(result.baseline.shape, (3,))
            self.assertEqual(result.pha.shape, (3,))
            np.testing.assert_allclose(result.baseline, np.zeros(3))
            self.assertTrue(np.all(np.isfinite(result.pha)))
            self.assertGreater(abs(result.normalization), 0)

    def test_pha_timeline_pairs_accepted_pulse_index_and_pha(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=1,
                    valid_pulse_range_stop=2,
                    valid_pulse_diff_threshold=3,
                    spectrum_chunk_size=2,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                baseline_pha = pipeline.baseline_pha()
                timeline = pipeline.pha_timeline()

            np.testing.assert_array_equal(timeline.pulse_indices, np.arange(3))
            np.testing.assert_allclose(timeline.pha, baseline_pha.pha)
            self.assertEqual(timeline.accepted_count, 3)
            self.assertEqual(timeline.rejected_count, 1)

    def test_pha_timeline_uses_drift_corrected_pha_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_drift_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=0,
                    valid_pulse_range_stop=3,
                    valid_pulse_diff_threshold=0,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                    baseline_drift_correction=True,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                corrected = pipeline.drift_corrected_pha_spectrum()
                timeline = pipeline.pha_timeline()

            np.testing.assert_allclose(timeline.pha, corrected.pha)

    def test_pha_timeline_renderer_uses_markers_without_lines(self) -> None:
        fig, ax = plt.subplots()
        try:
            renderer = PulsePlotRenderer.__new__(PulsePlotRenderer)
            renderer._draw_pha_timeline(
                ax,
                PhaTimelineResult(
                    pulse_indices=np.array([0, 1, 2]),
                    pha=np.array([10.0, 11.0, 9.0]),
                    accepted_count=3,
                    rejected_count=0,
                    normalization=1.0,
                ),
            )

            self.assertEqual(len(ax.lines), 0)
            self.assertEqual(len(ax.collections), 1)
        finally:
            plt.close(fig)

    def test_pha_cluster_uses_nearby_points_to_resolve_isolated_crossings(
        self,
    ) -> None:
        pha = np.array([10.0] * 10 + [20.0] + [10.0] * 10)
        selected = np.ones(pha.shape, dtype=bool)

        lower, upper = pulse_analysis.cluster_pha_timeline(pha, selected, boundary=15.0)

        np.testing.assert_array_equal(lower, np.ones(pha.shape, dtype=bool))
        np.testing.assert_array_equal(upper, np.zeros(pha.shape, dtype=bool))

    def test_pha_cluster_uses_drift_cluster_boundary_when_config_boundary_is_unset(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_drift_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=0,
                    valid_pulse_range_stop=3,
                    valid_pulse_diff_threshold=0,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                    baseline_drift_correction=True,
                    baseline_drift_clustering=True,
                    pha_clustering=True,
                    pha_cluster_boundary=None,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                baseline_pha = pipeline.baseline_pha()
                cluster = pipeline.pha_cluster()

            self.assertIsNotNone(baseline_pha.drift)
            assert baseline_pha.drift is not None
            self.assertIsNotNone(baseline_pha.drift.cluster_boundary)
            self.assertEqual(cluster.boundary, baseline_pha.drift.cluster_boundary)

    def test_pha_cluster_config_boundary_overrides_drift_cluster_boundary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_drift_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=0,
                    valid_pulse_range_stop=3,
                    valid_pulse_diff_threshold=0,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                    baseline_drift_correction=True,
                    baseline_drift_clustering=True,
                    pha_clustering=True,
                    pha_cluster_boundary=16.0,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                cluster = pipeline.pha_cluster()

            self.assertEqual(cluster.boundary, 16.0)

    def test_pha_cluster_splits_selected_timeline_points_by_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_drift_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=0,
                    valid_pulse_range_stop=3,
                    valid_pulse_diff_threshold=0,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                    pha_clustering=True,
                    pha_cluster_pha_min=12.0,
                    pha_cluster_pha_max=20.0,
                    pha_cluster_boundary=16.0,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                cluster = pipeline.pha_cluster()

            np.testing.assert_array_equal(
                cluster.selected_mask,
                (cluster.pha >= 12.0) & (cluster.pha <= 20.0),
            )
            expected_lower, expected_upper = pulse_analysis.cluster_pha_timeline(
                cluster.pha, cluster.selected_mask, boundary=16.0
            )
            np.testing.assert_array_equal(cluster.lower_cluster_mask, expected_lower)
            np.testing.assert_array_equal(cluster.upper_cluster_mask, expected_upper)

    def test_lower_cluster_pha_histogram_uses_lower_cluster_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_drift_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=0,
                    valid_pulse_range_stop=3,
                    valid_pulse_diff_threshold=0,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                    pha_clustering=True,
                    pha_cluster_pha_min=12.0,
                    pha_cluster_pha_max=20.0,
                    pha_cluster_boundary=16.0,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                cluster = pipeline.pha_cluster()
                lower_histogram = pipeline.lower_cluster_pha_spectrum()

            np.testing.assert_array_equal(
                lower_histogram.pha,
                cluster.pha[cluster.lower_cluster_mask],
            )
            self.assertEqual(
                int(np.sum(lower_histogram.counts)),
                int(np.count_nonzero(cluster.lower_cluster_mask)),
            )

    def test_pha_cluster_renderer_draws_cluster_collections(self) -> None:
        fig, ax = plt.subplots()
        try:
            renderer = PulsePlotRenderer.__new__(PulsePlotRenderer)
            renderer._draw_pha_cluster(
                ax,
                PhaClusterResult(
                    pulse_indices=np.array([0, 1, 2]),
                    pha=np.array([10.0, 16.0, 22.0]),
                    selected_mask=np.array([True, True, False]),
                    lower_cluster_mask=np.array([True, False, False]),
                    upper_cluster_mask=np.array([False, True, False]),
                    pha_min=9.0,
                    pha_max=20.0,
                    boundary=15.0,
                    accepted_count=3,
                    rejected_count=0,
                    normalization=1.0,
                ),
            )

            self.assertGreaterEqual(len(ax.collections), 3)
            self.assertGreaterEqual(len(ax.lines), 3)
        finally:
            plt.close(fig)

    def test_baseline_drift_correction_removes_linear_pha_dependence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_drift_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=0,
                    valid_pulse_range_stop=3,
                    valid_pulse_diff_threshold=0,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                    baseline_drift_correction=True,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                result = pipeline.baseline_pha()
                corrected = pipeline.drift_corrected_pha_spectrum()

            raw_slope = np.polyfit(result.baseline, result.pha, deg=1)[0]
            self.assertIsNotNone(result.drift)
            assert result.drift is not None
            corrected_slope = np.polyfit(
                result.baseline, result.drift.pha_corrected, deg=1
            )[0]
            self.assertGreater(abs(raw_slope), 0)
            self.assertAlmostEqual(result.drift.slope, raw_slope)
            self.assertAlmostEqual(corrected_slope, 0.0, places=10)
            self.assertAlmostEqual(
                float(np.mean(result.drift.pha_corrected)), float(np.mean(result.pha))
            )
            np.testing.assert_allclose(corrected.pha, result.drift.pha_corrected)
            self.assertEqual(int(corrected.counts.sum()), result.accepted_count)

    def test_baseline_pha_clustering_records_fit_cluster_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_drift_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=0,
                    valid_pulse_range_stop=3,
                    valid_pulse_diff_threshold=0,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                    baseline_drift_correction=True,
                    baseline_drift_clustering=True,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                result = pipeline.baseline_pha()

            self.assertIsNotNone(result.drift)
            assert result.drift is not None
            self.assertIsNotNone(result.drift.cluster_labels)
            assert result.drift.cluster_labels is not None
            self.assertEqual(result.drift.cluster_labels.shape, result.pha.shape)
            self.assertEqual(
                int(np.count_nonzero(result.drift.cluster_labels >= 0)),
                result.drift.fit_count,
            )

    def test_baseline_pha_clustering_uses_configured_cluster_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_drift_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=0,
                    valid_pulse_range_stop=3,
                    valid_pulse_diff_threshold=0,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                    baseline_drift_correction=True,
                    baseline_drift_clustering=True,
                    baseline_drift_cluster_count=3,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                result = pipeline.baseline_pha()

            self.assertIsNotNone(result.drift)
            assert result.drift is not None
            self.assertIsNotNone(result.drift.cluster_centers)
            assert result.drift.cluster_centers is not None
            self.assertEqual(result.drift.cluster_centers.shape, (3,))

    def test_baseline_drift_correction_disabled_uses_raw_pha(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_drift_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=0,
                    valid_pulse_range_stop=3,
                    valid_pulse_diff_threshold=0,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                    baseline_drift_correction=False,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                result = pipeline.baseline_pha()
                corrected = pipeline.drift_corrected_pha_spectrum()

            self.assertIsNone(result.drift)
            np.testing.assert_allclose(corrected.pha, result.pha)

    def test_drift_correction_config_change_invalidates_cached_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_drift_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=0,
                    valid_pulse_range_stop=3,
                    valid_pulse_diff_threshold=0,
                    spectrum_bins=3,
                    spectrum_chunk_size=2,
                    baseline_drift_correction=True,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)
                pipeline.baseline_pha()
                pipeline.drift_corrected_pha_spectrum()

                pipeline.update_config(
                    config.with_updates(baseline_drift_correction=False)
                )

                self.assertNotIn(
                    PulseStage.BASELINE_PHA.value,
                    pipeline._cache,
                )
                self.assertNotIn(
                    PulseStage.DRIFT_CORRECTED_PHA.value,
                    pipeline._cache,
                )

    def test_optimal_filter_stages_share_prep_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                config = PulseAnalysisConfig(
                    valid_pulse_range_start=1,
                    valid_pulse_range_stop=2,
                    valid_pulse_diff_threshold=3,
                    spectrum_chunk_size=2,
                ).validated()
                pipeline = PulsePipeline(PulseDataSource(pulse_data), config)

                prep = pipeline.optimal_filter_prep()
                self.assertIs(
                    pipeline.result_for_stage(PulseStage.REDUCTION.value),
                    pipeline.reduction_view(),
                )
                self.assertIs(
                    pipeline.result_for_stage(PulseStage.PH.value),
                    pipeline.ph_spectrum(),
                )
                self.assertIs(
                    pipeline.result_for_stage("Optimal Filter Signal FFT"),
                    prep,
                )
                self.assertIs(
                    pipeline.result_for_stage("Optimal Filter Noise FFT"),
                    prep,
                )
                self.assertIs(
                    pipeline.result_for_stage("Optimal Filter Template"),
                    prep,
                )
                self.assertIs(
                    pipeline.result_for_stage("PHA"),
                    pipeline.pha_spectrum(),
                )
                self.assertIs(
                    pipeline.result_for_stage(PulseStage.PHA_TIMELINE.value),
                    pipeline.pha_timeline(),
                )
                self.assertIs(
                    pipeline.result_for_stage(PulseStage.PHA_CLUSTER.value),
                    pipeline.pha_cluster(),
                )
                self.assertIs(
                    pipeline.result_for_stage(PulseStage.LOWER_CLUSTER_PHA.value),
                    pipeline.lower_cluster_pha_spectrum(),
                )
                self.assertIs(
                    pipeline.result_for_stage(PulseStage.BASELINE_PHA.value),
                    pipeline.baseline_pha(),
                )
                self.assertIs(
                    pipeline.result_for_stage(PulseStage.DRIFT_CORRECTED_PHA.value),
                    pipeline.drift_corrected_pha_spectrum(),
                )

    def test_save_pulse_plots_writes_single_npy_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            output_dir = Path(tmpdir) / "outputs"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                paths = save_pulse_plots(
                    pulse_data,
                    output_dir,
                    steps=(),
                    config=PulseAnalysisConfig(
                        valid_pulse_range_start=1,
                        valid_pulse_range_stop=2,
                        valid_pulse_diff_threshold=3,
                        spectrum_bins=3,
                        spectrum_chunk_size=2,
                    ),
                )

            names = {path.name for path in paths}
            self.assertIn("pulse-results.npy", names)
            self.assertNotIn("spectrum.npy", names)
            self.assertNotIn("optimal-filter-template.npy", names)
            self.assertNotIn("spectrum.csv", names)

            payload = np.load(
                output_dir / "pulse" / "pulse-results.npy",
                allow_pickle=True,
            ).item()
            self.assertIn("spectrum", payload)
            self.assertIn("optimal_filter_template", payload)
            self.assertIn("optimal_filter_template_fft", payload)
            self.assertIn("optimal_filter_pha_timeline", payload)
            self.assertIn("optimal_filter_baseline_pulse_height", payload)
            self.assertNotIn("optimal_filter_drift_correction", payload)
            self.assertNotIn("optimal_filter_drift_corrected_pulse_height", payload)
            self.assertEqual(int(payload["spectrum"]["count"].sum()), 3)
            self.assertEqual(
                payload["optimal_filter_baseline_pulse_height"]["baseline"].shape,
                (3,),
            )
            np.testing.assert_array_equal(
                payload["optimal_filter_pha_timeline"]["pulse_index"],
                np.arange(3),
            )

    def test_save_pulse_plots_includes_drift_outputs_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            output_dir = Path(tmpdir) / "outputs"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                paths = save_pulse_plots(
                    pulse_data,
                    output_dir,
                    steps=(),
                    config=PulseAnalysisConfig(
                        valid_pulse_range_start=1,
                        valid_pulse_range_stop=2,
                        valid_pulse_diff_threshold=3,
                        spectrum_bins=3,
                        spectrum_chunk_size=2,
                        baseline_drift_correction=True,
                    ),
                )

            self.assertIn("pulse-results.npy", {path.name for path in paths})
            payload = np.load(
                output_dir / "pulse" / "pulse-results.npy",
                allow_pickle=True,
            ).item()
            self.assertIn("optimal_filter_drift_correction", payload)
            self.assertIn("optimal_filter_drift_corrected_pulse_height", payload)
            self.assertEqual(
                payload["optimal_filter_baseline_pulse_height"]["pha_corrected"].shape,
                (3,),
            )

    def test_save_pulse_plots_can_write_csv_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            output_dir = Path(tmpdir) / "outputs"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                paths = save_pulse_plots(
                    pulse_data,
                    output_dir,
                    steps=(),
                    config=PulseAnalysisConfig(
                        valid_pulse_range_start=1,
                        valid_pulse_range_stop=2,
                        valid_pulse_diff_threshold=3,
                        spectrum_bins=3,
                        spectrum_chunk_size=2,
                    ),
                    array_format="csv",
                )

            names = {path.name for path in paths}
            self.assertIn("spectrum.csv", names)
            self.assertIn("optimal-filter-template.csv", names)
            self.assertIn("optimal-filter-pha-timeline.csv", names)
            self.assertIn("optimal-filter-baseline-pulse-height.csv", names)
            self.assertNotIn("optimal-filter-drift-corrected-pulse-height.csv", names)
            self.assertNotIn("pulse-results.npy", names)

    def test_save_pulse_plots_includes_pha_cluster_outputs_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "pulse.hdf5"
            output_dir = Path(tmpdir) / "outputs"
            write_test_hdf5(input_path)
            with open_hdf5_pulse_data(input_path) as pulse_data:
                paths = save_pulse_plots(
                    pulse_data,
                    output_dir,
                    steps=(),
                    config=PulseAnalysisConfig(
                        valid_pulse_range_start=1,
                        valid_pulse_range_stop=2,
                        valid_pulse_diff_threshold=3,
                        spectrum_bins=3,
                        spectrum_chunk_size=2,
                        pha_clustering=True,
                        pha_cluster_boundary=0.0,
                    ),
                    array_format="csv",
                )

            names = {path.name for path in paths}
            self.assertIn("optimal-filter-pha-cluster.csv", names)
            self.assertIn("optimal-filter-lower-cluster-pulse-height.csv", names)


if __name__ == "__main__":
    unittest.main()
