import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sim_flip.analysis.raw_preprocess import preprocess_run_to_csv


class TestRawPreprocess(unittest.TestCase):
    def test_preprocess_txt_to_csv(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            run_id = "R20260209_01"
            txt = tmp / f"{run_id}.txt"
            n = 3000
            t = np.arange(n)
            angle_x = 5.0 * np.sin(2 * np.pi * 0.3 * t / 1000.0)
            angle_y = 0.2 * np.sin(2 * np.pi * 0.3 * t / 1000.0 + 0.3)
            gyro_x = np.gradient(angle_x) * 1000.0
            gyro_y = np.gradient(angle_y) * 1000.0
            raw = pd.DataFrame(
                {
                    "angleX": angle_x,
                    "angleY": angle_y,
                    "gyroX": gyro_x,
                    "gyroY": gyro_y,
                    "time": t,
                }
            )
            raw.to_csv(txt, index=False, sep=" ")

            out_csv = tmp / "run.csv"
            out_qc = tmp / "qc.json"
            res = preprocess_run_to_csv(raw_txt_path=txt, run_id=run_id, out_csv_path=out_csv, out_qc_path=out_qc)
            self.assertTrue(res.csv_path.exists())
            self.assertTrue(res.qc_path.exists())
            self.assertIn("q_dot_rad_s2", res.df.columns)
            self.assertIn("theta_deg", res.df.columns)
            self.assertAlmostEqual(float(res.df["fs_hz"].iloc[0]), 1000.0, places=6)
            dt = np.diff(res.df["t_s"].to_numpy(dtype=float))
            self.assertTrue(np.all(dt > 0))


if __name__ == "__main__":
    unittest.main()

