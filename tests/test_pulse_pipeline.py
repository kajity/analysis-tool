from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.pulse.config import PulseAnalysisConfig, load_config, save_config
from scripts.pulse.datasource import PulseDataSource
from scripts.pulse.pipeline import PulsePipeline
from scripts.pulse.pulse_io import open_hdf5_pulse_data
from scripts.pulse.workflow import save_pulse_plots


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
                spectrum = pipeline.spectrum()

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
                spectrum = pipeline.spectrum()

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
                ).spectrum()

            with open_hdf5_pulse_data(input_path) as pulse_data:
                max_config = min_config.with_updates(
                    histogram_min=None,
                    histogram_max=1.0,
                )
                max_spectrum = PulsePipeline(
                    PulseDataSource(pulse_data),
                    max_config,
                ).spectrum()

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

    def test_optimal_filter_pulse_height_counts_projected_pulses(self) -> None:
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
                result = pipeline.optimal_filter_pulse_height()

            self.assertEqual(result.accepted_count, 3)
            self.assertEqual(result.rejected_count, 1)
            self.assertEqual(int(result.counts.sum()), 3)
            self.assertEqual(result.pulse_heights.shape, (3,))
            self.assertGreater(abs(result.normalization), 0)
            self.assertTrue(np.all(np.isfinite(result.pulse_heights)))

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
                    pipeline.result_for_stage("Optimal Filter Pulse Height"),
                    pipeline.optimal_filter_pulse_height(),
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
            self.assertEqual(int(payload["spectrum"]["count"].sum()), 3)

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
            self.assertNotIn("pulse-results.npy", names)


if __name__ == "__main__":
    unittest.main()
