import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sim_flip.analysis.segment_lock import segment_run_csv


class TestSegmentLock(unittest.TestCase):
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
            self.assertGreaterEqual(len(out.segment_paths), 1)
            mdf = pd.read_csv(manifest)
            self.assertIn("segment_id", mdf.columns)
            self.assertTrue((mdf["run_id"] == run_id).any())


if __name__ == "__main__":
    unittest.main()

