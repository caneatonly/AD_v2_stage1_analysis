import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from sim_flip.scripts import run_sim_real_eval


class TestRunSimRealEval(unittest.TestCase):
    @staticmethod
    def _write_params(path: Path) -> None:
        import yaml

        params = yaml.safe_load(Path("sim_flip/configs/params_nominal.yaml").read_text(encoding="utf-8"))
        path.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")

    @staticmethod
    def _write_segment_csv(path: Path, *, run_id: str, segment_id: str) -> None:
        t = np.linspace(0.0, 2.0, 101)
        theta_deg = 90.0 + 6.0 * np.exp(-0.25 * t) * np.cos(2.0 * np.pi * 0.7 * t)
        q_rad_s = np.gradient(np.deg2rad(theta_deg), t)
        df = pd.DataFrame(
            {
                "run_id": run_id,
                "segment_id": segment_id,
                "sample_idx": np.arange(len(t), dtype=int),
                "t_rel_s": t,
                "theta_rad": np.deg2rad(theta_deg),
                "theta_deg": theta_deg,
                "q_rad_s": q_rad_s,
            }
        )
        df.to_csv(path, index=False)

    @staticmethod
    def _run_script(
        *,
        segment_dir: Path,
        manifest: Path,
        params: Path,
        out_dir: Path,
        eval_split_tags: str,
        cv_fold: str,
    ) -> None:
        argv = [
            "run_sim_real_eval.py",
            "--segment-dir",
            str(segment_dir),
            "--manifest",
            str(manifest),
            "--params",
            str(params),
            "--out-dir",
            str(out_dir),
            "--eval-split-tags",
            eval_split_tags,
            "--cv-fold",
            cv_fold,
        ]
        with patch("sys.argv", argv):
            run_sim_real_eval.main()

    def test_no_selected_segment_exits_without_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            seg_dir = tmp / "segments"
            seg_dir.mkdir(parents=True, exist_ok=True)
            self._write_segment_csv(seg_dir / "R20260210_01_S01.csv", run_id="R20260210_01", segment_id="R20260210_01_S01")

            manifest = tmp / "manifest.csv"
            pd.DataFrame(
                [
                    {"segment_id": "R20260210_01_S01", "split_tag": "train", "cv_fold": "holdout_v1"},
                ]
            ).to_csv(manifest, index=False)
            params = tmp / "params.yaml"
            self._write_params(params)

            out_dir = tmp / "eval_out"
            self._run_script(
                segment_dir=seg_dir,
                manifest=manifest,
                params=params,
                out_dir=out_dir,
                eval_split_tags="val",
                cv_fold="holdout_v1",
            )

            self.assertFalse((out_dir / "sim_real_metrics.csv").exists())
            self.assertFalse((out_dir / "sim_real_metrics.json").exists())

    def test_only_matching_split_tag_and_fold_are_evaluated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            seg_dir = tmp / "segments"
            seg_dir.mkdir(parents=True, exist_ok=True)
            self._write_segment_csv(seg_dir / "R20260210_01_S01.csv", run_id="R20260210_01", segment_id="R20260210_01_S01")
            self._write_segment_csv(seg_dir / "R20260210_02_S01.csv", run_id="R20260210_02", segment_id="R20260210_02_S01")

            manifest = tmp / "manifest.csv"
            pd.DataFrame(
                [
                    {"segment_id": "R20260210_01_S01", "split_tag": "val", "cv_fold": "holdout_v1"},
                    {"segment_id": "R20260210_02_S01", "split_tag": "train", "cv_fold": "holdout_v1"},
                ]
            ).to_csv(manifest, index=False)
            params = tmp / "params.yaml"
            self._write_params(params)

            out_dir = tmp / "eval_out"
            self._run_script(
                segment_dir=seg_dir,
                manifest=manifest,
                params=params,
                out_dir=out_dir,
                eval_split_tags="val",
                cv_fold="holdout_v1",
            )

            out_csv = out_dir / "sim_real_metrics.csv"
            self.assertTrue(out_csv.exists())
            out = pd.read_csv(out_csv)
            self.assertEqual(len(out), 1)
            self.assertEqual(str(out.loc[0, "segment_id"]), "R20260210_01_S01")


if __name__ == "__main__":
    unittest.main()
