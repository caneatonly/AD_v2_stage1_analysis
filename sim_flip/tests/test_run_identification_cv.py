import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

import sim_flip.analysis.config as config_mod
from sim_flip.scripts import run_identification_cv


class TestRunIdentificationCv(unittest.TestCase):
    MANIFEST_COLUMNS = [
        "run_id",
        "segment_id",
        "repeat_id",
        "theta0_meas_deg",
        "q0_meas_rad_s",
        "t_start_s",
        "t_end_s",
        "segmentation_mode",
        "split_tag",
        "cv_fold",
        "notes",
    ]

    @staticmethod
    def _write_raw_txt(path: Path, *, sign: float = 1.0, broken: bool = False) -> None:
        dt = 0.02
        t = np.arange(0.0, 60.0, dt)
        t0 = 3.0
        tau = np.clip(t - t0, 0.0, None)
        gate = 1.0 / (1.0 + np.exp(-(t - t0) / 0.2))
        base = 25.0 * gate * np.exp(-0.015 * tau) * np.cos(2.0 * np.pi * 0.8 * tau)
        angle_x = sign * base
        angle_y = sign * 0.15 * base
        gyro_x = np.gradient(angle_x, dt)
        gyro_y = np.gradient(angle_y, dt)
        if broken:
            df = pd.DataFrame({"angleX": angle_x, "gyroX": gyro_x, "gyroY": gyro_y, "time": t * 1000.0})
        else:
            df = pd.DataFrame(
                {
                    "angleX": angle_x,
                    "angleY": angle_y,
                    "gyroX": gyro_x,
                    "gyroY": gyro_y,
                    "time": t * 1000.0,
                }
            )
        df.to_csv(path, index=False, sep=" ")

    @staticmethod
    def _write_protocol(path: Path, *, min_cycles: int = 8) -> None:
        base = yaml.safe_load(Path("sim_flip/configs/id_protocol.yaml").read_text(encoding="utf-8"))
        base["segmentation"]["max_segments_per_run"] = 1
        base["segmentation"]["stable_q_abs_max_rad_s"] = 1e-6
        base["segmentation"]["stable_theta_std_max_deg"] = 1e-6
        base["segmentation"]["min_peak_distance_s"] = 2.0
        base["step3_energy"]["min_cycles"] = int(min_cycles)
        base["bootstrap"]["n_boot"] = 0
        path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")

    @staticmethod
    def _write_params(path: Path) -> None:
        params = {
            "constants": {"rho": 1000.0, "g": 9.81},
            "rigid_body": {"Iyy": 0.2, "I_water_inner": 0.03},
            "added_mass_outer": {"M_qdot_outer": 0.01},
            "buoyancy_restore": {"B_mass": 2.0, "x_b": 0.05},
            "drag": {"Cd_x": 0.7, "Cd_z": 0.8},
            "cfd": {"table_path": "sim_flip/data/cfd_table_clean.csv", "interpolation": "pchip"},
            "initial_condition_default": {"u0": 0.0, "w0": 0.0, "q0": 0.0, "theta0_deg": 90.0},
            "integration": {"method": "RK45", "dt_out": 0.01, "t_span_s": [0.0, 10.0]},
        }
        path.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")

    @classmethod
    def _write_manifest(cls, path: Path, rows: list[dict[str, object]]) -> None:
        defaults: dict[str, object] = {
            "run_id": "",
            "segment_id": "",
            "repeat_id": 1,
            "theta0_meas_deg": np.nan,
            "q0_meas_rad_s": np.nan,
            "t_start_s": np.nan,
            "t_end_s": np.nan,
            "segmentation_mode": "preset",
            "split_tag": "train",
            "cv_fold": "holdout_v1",
            "notes": "manual_note",
        }
        out_rows = []
        for row in rows:
            r = defaults.copy()
            r.update(row)
            out_rows.append(r)
        pd.DataFrame(out_rows, columns=cls.MANIFEST_COLUMNS).to_csv(path, index=False)

    @staticmethod
    def _run_script(
        *,
        sim_root: Path,
        raw_dir: Path,
        protocol: Path,
        manifest: Path,
        params: Path,
        out_dir: Path,
        extra_args: list[str] | None = None,
    ) -> None:
        argv = [
            "run_identification_cv.py",
            "--raw-dir",
            str(raw_dir),
            "--protocol",
            str(protocol),
            "--manifest",
            str(manifest),
            "--params",
            str(params),
            "--out-dir",
            str(out_dir),
            "--fit-split-tag",
            "train",
            "--cv-fold",
            "holdout_v1",
        ]
        if extra_args:
            argv.extend(extra_args)
        with patch.object(config_mod, "sim_flip_root", return_value=sim_root):
            with patch("sys.argv", argv):
                run_identification_cv.main()

    def test_global_fit_uses_train_segments_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            sim_root = tmp / "sim_flip"
            raw_dir = tmp / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            self._write_raw_txt(raw_dir / "R20260210_01.txt", sign=1.0)
            self._write_raw_txt(raw_dir / "R20260210_02.txt", sign=-1.0)

            manifest = tmp / "manifest.csv"
            self._write_manifest(
                manifest,
                [
                    {"run_id": "R20260210_01", "segment_id": "R20260210_01_S01", "split_tag": "train", "cv_fold": "holdout_v1"},
                    {"run_id": "R20260210_02", "segment_id": "R20260210_02_S01", "split_tag": "train", "cv_fold": "holdout_v1"},
                ],
            )
            protocol = tmp / "protocol.yaml"
            self._write_protocol(protocol, min_cycles=8)
            params = tmp / "params.yaml"
            self._write_params(params)
            out_dir = tmp / "results"

            self._run_script(
                sim_root=sim_root,
                raw_dir=raw_dir,
                protocol=protocol,
                manifest=manifest,
                params=params,
                out_dir=out_dir,
            )

            global_dir = out_dir / "global"
            self.assertTrue((global_dir / "identified_params.json").exists())
            self.assertTrue((global_dir / "identified_params.yaml").exists())
            self.assertTrue((global_dir / "frequency.csv").exists())
            self.assertTrue((global_dir / "multistart_convergence.csv").exists())
            self.assertTrue((global_dir / "selected_segments.csv").exists())
            self.assertTrue((global_dir / "preprocess_status.csv").exists())
            self.assertTrue((out_dir / "identification_summary.csv").exists())

            self.assertFalse((out_dir / "R20260210_01").exists())
            self.assertFalse((out_dir / "R20260210_02").exists())

            params = json.loads((global_dir / "identified_params.json").read_text(encoding="utf-8"))
            self.assertEqual(params["fit_scope"], "global")
            self.assertEqual(params["run_id"], "global")
            self.assertEqual(int(params["n_segments_selected"]), 2)
            self.assertEqual(sorted(params["selected_run_ids"]), ["R20260210_01", "R20260210_02"])

            selected = pd.read_csv(global_dir / "selected_segments.csv")
            self.assertEqual(len(selected), 2)
            self.assertEqual(sorted(selected["run_id"].astype(str).tolist()), ["R20260210_01", "R20260210_02"])

    def test_failed_run_is_recorded_and_pipeline_continues(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            sim_root = tmp / "sim_flip"
            raw_dir = tmp / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            self._write_raw_txt(raw_dir / "R20260210_01.txt", sign=1.0)
            self._write_raw_txt(raw_dir / "R20260210_02.txt", sign=-1.0, broken=True)

            manifest = tmp / "manifest.csv"
            self._write_manifest(
                manifest,
                [
                    {"run_id": "R20260210_01", "segment_id": "R20260210_01_S01", "split_tag": "train", "cv_fold": "holdout_v1"},
                    {"run_id": "R20260210_02", "segment_id": "R20260210_02_S01", "split_tag": "train", "cv_fold": "holdout_v1"},
                ],
            )
            protocol = tmp / "protocol.yaml"
            self._write_protocol(protocol, min_cycles=6)
            params = tmp / "params.yaml"
            self._write_params(params)
            out_dir = tmp / "results"

            self._run_script(
                sim_root=sim_root,
                raw_dir=raw_dir,
                protocol=protocol,
                manifest=manifest,
                params=params,
                out_dir=out_dir,
            )

            global_dir = out_dir / "global"
            params = json.loads((global_dir / "identified_params.json").read_text(encoding="utf-8"))
            self.assertEqual(int(params["n_segments_selected"]), 1)
            self.assertIn("R20260210_02", params["skipped_runs"])

            status = pd.read_csv(global_dir / "preprocess_status.csv")
            bad = status.loc[status["run_id"].astype(str) == "R20260210_02"].iloc[0]
            self.assertFalse(bool(bad["preprocess_ok"]))

    def test_no_train_segment_raises_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            sim_root = tmp / "sim_flip"
            raw_dir = tmp / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            self._write_raw_txt(raw_dir / "R20260210_01.txt", sign=1.0)

            manifest = tmp / "manifest.csv"
            self._write_manifest(
                manifest,
                [
                    {"run_id": "R20260210_01", "segment_id": "R20260210_01_S01", "split_tag": "val", "cv_fold": "holdout_v1"},
                ],
            )
            protocol = tmp / "protocol.yaml"
            self._write_protocol(protocol, min_cycles=6)
            params = tmp / "params.yaml"
            self._write_params(params)
            out_dir = tmp / "results"

            with self.assertRaises(RuntimeError) as ctx:
                self._run_script(
                    sim_root=sim_root,
                    raw_dir=raw_dir,
                    protocol=protocol,
                    manifest=manifest,
                    params=params,
                    out_dir=out_dir,
                )
            self.assertIn("No segments selected", str(ctx.exception))

    def test_prepare_only_then_fit_only_two_step_flow(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            sim_root = tmp / "sim_flip"
            raw_dir = tmp / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            self._write_raw_txt(raw_dir / "R20260210_01.txt", sign=1.0)
            self._write_raw_txt(raw_dir / "R20260210_02.txt", sign=-1.0)

            manifest = tmp / "manifest.csv"
            self._write_manifest(
                manifest,
                [
                    {"run_id": "R20260210_01", "segment_id": "R20260210_01_S01", "split_tag": "train", "cv_fold": "holdout_v1"},
                    {"run_id": "R20260210_02", "segment_id": "R20260210_02_S01", "split_tag": "train", "cv_fold": "holdout_v1"},
                ],
            )
            protocol = tmp / "protocol.yaml"
            self._write_protocol(protocol, min_cycles=8)
            params = tmp / "params.yaml"
            self._write_params(params)
            out_dir = tmp / "results"

            self._run_script(
                sim_root=sim_root,
                raw_dir=raw_dir,
                protocol=protocol,
                manifest=manifest,
                params=params,
                out_dir=out_dir,
                extra_args=["--prepare-only"],
            )
            global_dir = out_dir / "global"
            self.assertTrue((global_dir / "preprocess_status.csv").exists())
            self.assertFalse((global_dir / "identified_params.json").exists())

            self._run_script(
                sim_root=sim_root,
                raw_dir=raw_dir,
                protocol=protocol,
                manifest=manifest,
                params=params,
                out_dir=out_dir,
                extra_args=["--fit-only"],
            )
            self.assertTrue((global_dir / "identified_params.json").exists())
            self.assertTrue((global_dir / "selected_segments.csv").exists())


if __name__ == "__main__":
    unittest.main()
