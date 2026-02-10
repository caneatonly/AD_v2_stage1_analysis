import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sim_flip.analysis.config import load_yaml
from sim_flip.analysis.raw_preprocess import preprocess_run_to_csv
from sim_flip.analysis.segment_lock import segment_run_csv


class TestSegmentLock(unittest.TestCase):
    @staticmethod
    def _write_raw_free_decay_txt(path: Path, excitation_sign: float) -> None:
        dt = 0.02
        t = np.arange(0.0, 40.0, dt)
        t0 = 5.0
        tau = np.clip(t - t0, 0.0, None)
        gate = 1.0 / (1.0 + np.exp(-(t - t0) / 0.2))
        base = 20.0 * gate * np.exp(-0.08 * tau) * np.cos(2.0 * np.pi * 0.25 * tau)
        angle_x = excitation_sign * base
        angle_y = excitation_sign * 0.15 * base
        gyro_x = np.gradient(angle_x, dt)
        gyro_y = np.gradient(angle_y, dt)
        raw = pd.DataFrame(
            {
                "angleX": angle_x,
                "angleY": angle_y,
                "gyroX": gyro_x,
                "gyroY": gyro_y,
                "time": t * 1000.0,
            }
        )
        raw.to_csv(path, index=False, sep=" ")

    def test_segment_export_and_manifest_update(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            run_id = "R20260209_01"
            t = np.arange(0.0, 40.0, 0.001)
            # Decaying free-decay-like waveform around 90 deg.
            theta_deg = 90.0 - 20.0 * np.exp(-0.07 * t) * np.cos(2 * np.pi * 0.18 * t)
            q_rad_s = np.deg2rad(np.gradient(theta_deg, t))
            q_dot_rad_s2 = np.gradient(q_rad_s, t)
            run_df = pd.DataFrame(
                {
                    "run_id": run_id,
                    "t_s": t,
                    "theta_rad": np.deg2rad(theta_deg),
                    "theta_deg": theta_deg,
                    "q_rad_s": q_rad_s,
                    "q_deg_s": np.rad2deg(q_rad_s),
                    "q_dot_rad_s2": q_dot_rad_s2,
                    "q_dot_deg_s2": np.rad2deg(q_dot_rad_s2),
                    "fs_hz": 1000.0,
                }
            )
            run_csv = tmp / f"{run_id}.csv"
            run_df.to_csv(run_csv, index=False)
            manifest = tmp / "experiment_manifest.csv"
            protocol = Path("sim_flip/configs/id_protocol.yaml")
            out_seg_dir = tmp / "segments"

            out = segment_run_csv(
                run_csv_path=run_csv,
                protocol_path=protocol,
                manifest_path=manifest,
                out_dir=out_seg_dir,
            )
            self.assertTrue(out.manifest_path.exists())
            self.assertTrue(out.metadata_json_path.exists())
            self.assertEqual(len(out.segment_paths), 1)
            mdf = pd.read_csv(manifest)
            self.assertIn("segment_id", mdf.columns)
            self.assertTrue((mdf["run_id"] == run_id).any())

    def test_reverse_excitation_still_locks_first_peak_single_segment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            protocol_path = Path("sim_flip/configs/id_protocol.yaml")
            protocol = load_yaml(protocol_path)
            seg_cfg = protocol["segmentation"]
            dt = float(protocol["preprocess"]["resample"]["dt_s"])
            fs = 1.0 / dt
            min_dist = int(max(1, float(seg_cfg["min_peak_distance_s"]) * fs))
            prominence = float(seg_cfg["valley_prominence_deg"])
            min_t = float(seg_cfg["start_time_min_s"])

            manifest = tmp / "experiment_manifest.csv"
            theta0_by_sign: dict[float, float] = {}

            for sign, run_id in [(1.0, "R20260210_01"), (-1.0, "R20260210_02")]:
                raw_txt = tmp / f"{run_id}.txt"
                self._write_raw_free_decay_txt(raw_txt, excitation_sign=sign)

                pre = preprocess_run_to_csv(
                    raw_txt_path=raw_txt,
                    run_id=run_id,
                    protocol_path=protocol_path,
                    out_csv_path=tmp / f"{run_id}.csv",
                    out_qc_path=tmp / f"{run_id}_qc.json",
                )
                seg = segment_run_csv(
                    run_csv_path=pre.csv_path,
                    protocol_path=protocol_path,
                    manifest_path=manifest,
                    out_dir=tmp / "segments",
                )

                self.assertEqual(len(seg.segment_paths), 1)
                seg_df = pd.read_csv(seg.segment_paths[0])
                theta0 = float(seg_df["theta_deg"].iloc[0])
                theta0_by_sign[sign] = theta0
                self.assertGreater(theta0, 90.0)

                t_s = pre.df["t_s"].to_numpy(dtype=float)
                theta_deg = pre.df["theta_deg"].to_numpy(dtype=float)
                peaks, _ = find_peaks(theta_deg, distance=min_dist, prominence=prominence)
                valid = [int(i) for i in peaks if float(t_s[int(i)]) >= min_t]
                self.assertTrue(valid)
                first_peak_idx = valid[0]
                self.assertAlmostEqual(theta0, float(theta_deg[first_peak_idx]), places=3)

            self.assertAlmostEqual(theta0_by_sign[1.0], theta0_by_sign[-1.0], places=2)


if __name__ == "__main__":
    unittest.main()
