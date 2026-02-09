from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sim_flip.analysis.config import (
    default_manifest_path,
    default_protocol_path,
    ensure_pipeline_dirs,
    load_yaml,
)
from sim_flip.analysis.frequency import estimate_frequency_many
from sim_flip.analysis.id_step3_energy import fit_damping_nnls
from sim_flip.analysis.id_step4_ode import fit_params_ode
from sim_flip.analysis.raw_preprocess import preprocess_run_to_csv, run_id_from_path
from sim_flip.analysis.segment_lock import segment_run_csv
from sim_flip.src.dynamics import load_params_yaml


def _model_cfg_from_params(path: Path) -> dict[str, float]:
    params = load_params_yaml(path)
    rb = params["rigid_body"]
    am = params["added_mass_outer"]
    buoy = params["buoyancy_restore"]
    return {
        "I0": float(rb["Iyy"] - am["M_qdot_outer"]),
        "Iwater": float(rb["I_water_inner"]),
        "B_force": float(buoy["B_mass"]) * float(params["constants"]["g"]),
        "x_b": float(buoy["x_b"]),
    }


def _select_segment_ids(
    *,
    manifest_path: Path,
    run_id: str,
    split_tag: str,
    cv_fold: str,
) -> list[str]:
    if not manifest_path.exists():
        return []
    mdf = pd.read_csv(manifest_path)
    if mdf.empty:
        return []
    req = {"run_id", "segment_id", "split_tag", "cv_fold"}
    if not req.issubset(mdf.columns):
        raise ValueError(f"manifest missing columns: {sorted(req - set(mdf.columns))}")

    mask = (mdf["run_id"].astype(str) == str(run_id)) & (mdf["split_tag"].astype(str) == str(split_tag))
    if cv_fold.strip():
        mask = mask & (mdf["cv_fold"].fillna("").astype(str) == cv_fold.strip())
    out = mdf.loc[mask, "segment_id"].astype(str).tolist()
    return sorted(set(out))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed identification pipeline with CV-ready outputs.")
    parser.add_argument("--raw-dir", type=Path, default=Path("sim_flip/data/raw/runs"))
    parser.add_argument("--protocol", type=Path, default=default_protocol_path())
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--params", type=Path, default=Path("sim_flip/configs/params_nominal.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("sim_flip/results/identification"))
    parser.add_argument("--fit-split-tag", type=str, default="train", help="Only use segments with this split_tag.")
    parser.add_argument("--cv-fold", type=str, default="", help="Optional cv_fold filter (exact match).")
    args = parser.parse_args()

    ensure_pipeline_dirs()
    protocol = load_yaml(args.protocol)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for txt in sorted(args.raw_dir.glob("R*.txt")):
        run_id = run_id_from_path(txt)
        pre = preprocess_run_to_csv(raw_txt_path=txt, run_id=run_id, protocol_path=args.protocol)
        seg = segment_run_csv(run_csv_path=pre.csv_path, protocol_path=args.protocol, manifest_path=args.manifest)
        selected_ids = _select_segment_ids(
            manifest_path=args.manifest,
            run_id=run_id,
            split_tag=args.fit_split_tag,
            cv_fold=args.cv_fold,
        )
        if not selected_ids:
            print(f"[skip] {run_id}: no segment matched split_tag={args.fit_split_tag!r}, cv_fold={args.cv_fold!r}")
            continue

        path_map = {p.stem: p for p in seg.segment_paths}
        selected_paths = [path_map[sid] for sid in selected_ids if sid in path_map]
        segment_dfs = [pd.read_csv(p) for p in selected_paths]
        if not segment_dfs:
            print(f"[skip] {run_id}: selected segment files not found in derived directory")
            continue

        freq_df = estimate_frequency_many(segment_dfs, protocol=protocol)
        freq_path = args.out_dir / f"{run_id}_frequency.csv"
        freq_df.to_csv(freq_path, index=False)

        model_cfg = _model_cfg_from_params(args.params)
        I_sys = model_cfg["I0"] + 0.5 * model_cfg["Iwater"]
        K_sys = model_cfg["B_force"] * model_cfg["x_b"]
        step3 = fit_damping_nnls(segment_dfs=segment_dfs, I_sys=I_sys, K_sys=K_sys, protocol=protocol)
        step4 = fit_params_ode(
            segment_dfs=segment_dfs,
            protocol=protocol,
            model_cfg=model_cfg,
            init_guess={"d_q": step3.d_q, "d_qq": step3.d_qq},
        )
        out = {
            "run_id": run_id,
            "fit_split_tag": args.fit_split_tag,
            "cv_fold": args.cv_fold,
            "selected_segment_ids": selected_ids,
            "n_segments_selected": len(segment_dfs),
            "n_segments_exported": len(seg.segment_paths),
            "step3_d_q": step3.d_q,
            "step3_d_qq": step3.d_qq,
            "step3_n_cycles": step3.n_cycles_used,
            "step4_mu_theta": step4.mu_theta,
            "step4_d_q": step4.d_q,
            "step4_d_qq": step4.d_qq,
            "step4_K_cable": step4.K_cable,
            "step4_success": step4.success,
            "step4_cost": step4.cost,
            "step4_nfev": step4.nfev,
        }
        with (args.out_dir / f"{run_id}_id.json").open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        summary_rows.append(out)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(args.out_dir / "identification_summary.csv", index=False)


if __name__ == "__main__":
    main()
