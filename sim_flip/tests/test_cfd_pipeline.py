import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sim_flip.analysis.cfd_pipeline import run_cfd_review_pipeline


class TestCfdPipeline(unittest.TestCase):
    def test_build_full_table_from_angle_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "cfd"
            src.mkdir(parents=True, exist_ok=True)
            for a in range(0, 181, 5):
                n = 700
                # simple synthetic convergence series
                cx = np.full(n, -0.3 + 0.002 * a)
                cz = np.full(n, -0.1 - 0.01 * a)
                cm = np.full(n, 0.005 * np.sin(np.deg2rad(a)))
                pd.DataFrame({"Cx": cx, "Cz": cz, "Cm": cm}).to_csv(src / f"C_{a}_overmesh.csv", index=False)

            out_table = tmp / "cfd_table_clean.csv"
            out_report = tmp / "report.json"
            out_fig = tmp / "figs"
            out = run_cfd_review_pipeline(
                source_dir=src,
                out_table_path=out_table,
                out_report_path=out_report,
                out_figure_dir=out_fig,
                n_steady_last=500,
                angle_step_deg=5,
            )
            self.assertTrue(out.cfd_table_path.exists())
            self.assertTrue(out.report_path.exists())
            table = pd.read_csv(out.cfd_table_path)
            self.assertEqual(int(table["alpha_deg"].min()), 0)
            self.assertEqual(int(table["alpha_deg"].max()), 180)


if __name__ == "__main__":
    unittest.main()

