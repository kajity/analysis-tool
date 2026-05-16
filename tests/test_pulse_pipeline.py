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


def write_test_hdf5(path: Path) -> None:
    wave = np.array(
        [
            [0, 0, 0, 0, 0, -1, -3, -1],
            [0, 0, 0, 0, 0, 10, -1, 0],
            [0, 0, 0, 0, 0, -2, -2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
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


if __name__ == "__main__":
    unittest.main()
