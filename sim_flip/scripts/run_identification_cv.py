from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from sim_flip.analysis.bootstrap import bootstrap_fit_segments
from sim_flip.analysis.config import (
    default_derived_segment_dir,
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


def _select_segment_rows_global(
    *,
    manifest_path: Path,
    split_tag: str,
    cv_fold: str,
) -> pd.DataFrame:
    cols = ["run_id", "segment_id", "split_tag", "cv_fold"]
    if not manifest_path.exists():
        return pd.DataFrame(columns=cols)
    mdf = pd.read_csv(manifest_path)
    if mdf.empty:
        return pd.DataFrame(columns=cols)
    req = {"run_id", "segment_id", "split_tag", "cv_fold"}
    if not req.issubset(mdf.columns):
        raise ValueError(f"manifest missing columns: {sorted(req - set(mdf.columns))}")

    mask = mdf["split_tag"].astype(str) == str(split_tag)
    if cv_fold.strip():
        mask = mask & (mdf["cv_fold"].fillna("").astype(str) == cv_fold.strip())
    out = mdf.loc[mask, cols].copy()
    if out.empty:
        return out
    out["run_id"] = out["run_id"].astype(str)
    out["segment_id"] = out["segment_id"].astype(str)
    out["split_tag"] = out["split_tag"].astype(str)
    out["cv_fold"] = out["cv_fold"].fillna("").astype(str)
    out = out.drop_duplicates(subset=["run_id", "segment_id"]).sort_values(["run_id", "segment_id"]).reset_index(drop=True)
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _save_reproducibility_artifacts(*, out_dir: Path, protocol_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    snap_path = out_dir / "protocol_snapshot.yaml"
    if protocol_path.exists():
        shutil.copy2(protocol_path, snap_path)
    else:
        snap_path.write_text("# protocol path not found\n", encoding="utf-8")

    git_path = out_dir / "git_commit.txt"
    try:
        res = subprocess.run(
            ["git", "-C", str(_repo_root()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        git_path.write_text(res.stdout.strip() + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        git_path.write_text(f"unavailable: {exc}\n", encoding="utf-8")

    env_path = out_dir / "python_env.txt"
    lines: list[str] = [f"python_executable={sys.executable}"]
    try:
        v = subprocess.run([sys.executable, "--version"], check=True, capture_output=True, text=True)
        lines.append(v.stdout.strip() or v.stderr.strip())
    except Exception as exc:  # noqa: BLE001
        lines.append(f"python_version_unavailable: {exc}")
    try:
        fr = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
        lines.append("")
        lines.append("[pip_freeze]")
        lines.extend([ln for ln in fr.stdout.splitlines() if ln.strip()])
    except Exception as exc:  # noqa: BLE001
        lines.append(f"pip_freeze_unavailable: {exc}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_scatter_png(path: Path, rows: list[dict], *, title: str) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    df = pd.DataFrame(rows)
    if not {"mu_theta", "d_q", "cost"}.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    c = df["cost"].to_numpy(dtype=float)
    sc = ax.scatter(df["mu_theta"], df["d_q"], c=c, cmap="viridis", s=20, alpha=0.85)
    ax.set_xlabel("mu_theta")
    ax.set_ylabel("d_q")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="cost")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _run_multistart_fit(
    *,
    segment_dfs: list[pd.DataFrame],
    protocol: dict,
    model_cfg: dict[str, float],
    init_from_step3: dict[str, float],
    seed: int,
) -> tuple[dict, list[dict]]:
    cfg = protocol.get("step4_ode", {})
    bnd = cfg.get("bounds", {})
    n_starts = int(cfg.get("multi_start_n", 1))
    n_starts = max(1, n_starts)

    mu_b = tuple(float(v) for v in bnd.get("mu_theta", [0.0, 1.0]))
    dq_b = tuple(float(v) for v in bnd.get("d_q", [0.0, 5.0]))
    dqq_b = tuple(float(v) for v in bnd.get("d_qq", [0.0, 5.0]))
    kc_b = tuple(float(v) for v in bnd.get("K_cable", [0.0, 5.0]))

    rng = np.random.default_rng(seed)
    init_cfg = cfg.get("init", {})
    first_start = {
        "mu_theta": float(init_cfg.get("mu_theta", 0.6)),
        "d_q": float(init_cfg.get("d_q", 0.05)),
        "d_qq": float(init_cfg.get("d_qq", 0.05)),
        "K_cable": float(init_cfg.get("K_cable", 0.0)),
    }
    first_start.update({k: float(v) for k, v in init_from_step3.items() if k in first_start})
    starts: list[dict[str, float]] = [first_start]
    for _ in range(n_starts - 1):
        starts.append(
            {
                "mu_theta": float(rng.uniform(mu_b[0], mu_b[1])),
                "d_q": float(rng.uniform(dq_b[0], dq_b[1])),
                "d_qq": float(rng.uniform(dqq_b[0], dqq_b[1])),
                "K_cable": float(rng.uniform(kc_b[0], kc_b[1])),
            }
        )

    rows: list[dict] = []
    best: dict | None = None
    for i, start in enumerate(starts):
        res = fit_params_ode(
            segment_dfs=segment_dfs,
            protocol=protocol,
            model_cfg=model_cfg,
            init_guess=start,
        )
        row = {
            "start_idx": i,
            "mu_theta_init": float(start["mu_theta"]),
            "d_q_init": float(start["d_q"]),
            "d_qq_init": float(start["d_qq"]),
            "K_cable_init": float(start["K_cable"]),
            "mu_theta": float(res.mu_theta),
            "d_q": float(res.d_q),
            "d_qq": float(res.d_qq),
            "K_cable": float(res.K_cable),
            "success": bool(res.success),
            "cost": float(res.cost),
            "nfev": int(res.nfev),
            "message": str(res.message),
        }
        rows.append(row)
        if row["success"]:
            if best is None or row["cost"] < best["cost"]:
                best = row

    if best is None:
        best = min(rows, key=lambda r: float(r["cost"]))
    return best, rows


def _deterministic_seed(base_seed: int, label: str) -> int:
    acc = int(base_seed)
    for i, ch in enumerate(str(label)):
        acc += (i + 1) * ord(ch)
    return acc


def _build_preprocess_and_segment_status(
    *,
    raw_dir: Path,
    protocol_path: Path,
    manifest_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for txt in sorted(raw_dir.glob("R*.txt")):
        run_id = txt.stem
        row = {
            "run_id": run_id,
            "preprocess_ok": False,
            "segment_ok": False,
            "n_segments_exported": 0,
            "reason": "",
        }
        try:
            run_id = run_id_from_path(txt)
            row["run_id"] = run_id
        except Exception as exc:  # noqa: BLE001
            row["reason"] = f"run_id_invalid: {exc}"
            print(f"[skip] {txt.name}: {row['reason']}")
            rows.append(row)
            continue

        try:
            pre = preprocess_run_to_csv(raw_txt_path=txt, run_id=run_id, protocol_path=protocol_path)
            row["preprocess_ok"] = True
        except Exception as exc:  # noqa: BLE001
            row["reason"] = f"preprocess_failed: {exc}"
            print(f"[skip] {run_id}: {row['reason']}")
            rows.append(row)
            continue

        try:
            seg = segment_run_csv(run_csv_path=pre.csv_path, protocol_path=protocol_path, manifest_path=manifest_path)
            row["segment_ok"] = True
            row["n_segments_exported"] = int(len(seg.segment_paths))
            if len(seg.segment_paths) == 0:
                row["reason"] = "segment_export_empty"
        except Exception as exc:  # noqa: BLE001
            row["reason"] = f"segment_failed: {exc}"
            print(f"[skip] {run_id}: {row['reason']}")
        rows.append(row)
    return pd.DataFrame(rows, columns=["run_id", "preprocess_ok", "segment_ok", "n_segments_exported", "reason"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed identification pipeline with CV-ready outputs.")
    parser.add_argument("--raw-dir", type=Path, default=Path("sim_flip/data/raw/runs"))
    parser.add_argument("--protocol", type=Path, default=default_protocol_path())
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--params", type=Path, default=Path("sim_flip/configs/params_nominal.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("sim_flip/results/identification"))
    parser.add_argument("--fit-split-tag", type=str, default="train", help="Only use segments with this split_tag.")
    parser.add_argument("--cv-fold", type=str, default="", help="Optional cv_fold filter (exact match).")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only preprocess raw txt + export segments + update manifest, then exit without fitting.",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Skip preprocess/segmentation; fit directly from existing manifest + segment CSV files.",
    )
    args = parser.parse_args()
    if args.prepare_only and args.fit_only:
        raise ValueError("--prepare-only and --fit-only cannot be used together.")

    ensure_pipeline_dirs()
    protocol = load_yaml(args.protocol)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    global_seed = int(protocol.get("reproducibility", {}).get("random_seed", 0))
    global_out = args.out_dir / "global"
    global_out.mkdir(parents=True, exist_ok=True)
    _save_reproducibility_artifacts(out_dir=global_out, protocol_path=args.protocol)

    status_path = global_out / "preprocess_status.csv"
    if args.fit_only:
        if status_path.exists():
            preprocess_status = pd.read_csv(status_path)
        else:
            preprocess_status = pd.DataFrame(
                columns=["run_id", "preprocess_ok", "segment_ok", "n_segments_exported", "reason"]
            )
            preprocess_status.to_csv(status_path, index=False)
    else:
        preprocess_status = _build_preprocess_and_segment_status(
            raw_dir=args.raw_dir,
            protocol_path=args.protocol,
            manifest_path=args.manifest,
        )
        preprocess_status.to_csv(status_path, index=False)
        if args.prepare_only:
            print(
                "[done] prepare-only completed: run/segment artifacts and manifest are updated; "
                "no identification fit was executed."
            )
            return

    selected_rows = _select_segment_rows_global(
        manifest_path=args.manifest,
        split_tag=args.fit_split_tag,
        cv_fold=args.cv_fold,
    )
    if selected_rows.empty:
        raise RuntimeError(
            f"No segments selected by split_tag={args.fit_split_tag!r}, cv_fold={args.cv_fold!r}. "
            "Update manifest split_tag/cv_fold first."
        )

    segment_dir = default_derived_segment_dir()
    segment_dfs: list[pd.DataFrame] = []
    selected_loaded: list[dict[str, str]] = []
    skipped_segments: list[dict[str, str]] = []
    for _, row in selected_rows.iterrows():
        run_id = str(row["run_id"])
        seg_id = str(row["segment_id"])
        seg_path = segment_dir / f"{seg_id}.csv"
        if not seg_path.exists():
            skipped_segments.append(
                {"run_id": run_id, "segment_id": seg_id, "reason": f"segment_csv_missing: {seg_path}"}
            )
            continue
        try:
            seg_df = pd.read_csv(seg_path)
        except Exception as exc:  # noqa: BLE001
            skipped_segments.append({"run_id": run_id, "segment_id": seg_id, "reason": f"segment_read_failed: {exc}"})
            continue
        if seg_df.empty:
            skipped_segments.append({"run_id": run_id, "segment_id": seg_id, "reason": "segment_csv_empty"})
            continue
        segment_dfs.append(seg_df)
        selected_loaded.append(
            {
                "run_id": run_id,
                "segment_id": seg_id,
                "split_tag": str(row["split_tag"]),
                "cv_fold": str(row["cv_fold"]),
            }
        )

    if skipped_segments:
        pd.DataFrame(skipped_segments).to_csv(global_out / "skipped_segments.csv", index=False)
    selected_loaded_df = pd.DataFrame(selected_loaded, columns=["run_id", "segment_id", "split_tag", "cv_fold"])
    selected_loaded_df.to_csv(global_out / "selected_segments.csv", index=False)
    if not segment_dfs:
        raise RuntimeError(
            "No selected segment CSV could be loaded from derived directory. "
            "Check preprocessing/segmentation status and manifest segment_id values."
        )

    selected_run_ids_requested = sorted(set(selected_rows["run_id"].astype(str).tolist()))
    selected_run_ids = sorted(set(selected_loaded_df["run_id"].astype(str).tolist()))
    skipped_runs = set()
    if not preprocess_status.empty:
        m = preprocess_status["run_id"].astype(str).isin(selected_run_ids_requested)
        failed = m & (
            ~preprocess_status["preprocess_ok"].fillna(False).astype(bool)
            | ~preprocess_status["segment_ok"].fillna(False).astype(bool)
        )
        skipped_runs.update(preprocess_status.loc[failed, "run_id"].astype(str).tolist())
    skipped_runs.update([str(r["run_id"]) for r in skipped_segments])
    skipped_runs = set(skipped_runs) - set(selected_run_ids)

    freq_df = estimate_frequency_many(segment_dfs, protocol=protocol)
    freq_df.to_csv(global_out / "frequency.csv", index=False)

    model_cfg = _model_cfg_from_params(args.params)
    I_sys = model_cfg["I0"] + 0.5 * model_cfg["Iwater"]
    K_sys = model_cfg["B_force"] * model_cfg["x_b"]
    step3 = fit_damping_nnls(segment_dfs=segment_dfs, I_sys=I_sys, K_sys=K_sys, protocol=protocol)
    init_guess = {"d_q": step3.d_q, "d_qq": step3.d_qq}
    best_step4, multistart_rows = _run_multistart_fit(
        segment_dfs=segment_dfs,
        protocol=protocol,
        model_cfg=model_cfg,
        init_from_step3=init_guess,
        seed=_deterministic_seed(global_seed, f"global:{args.fit_split_tag}:{args.cv_fold}"),
    )
    pd.DataFrame(multistart_rows).to_csv(global_out / "multistart_convergence.csv", index=False)
    _save_scatter_png(
        global_out / "multistart_scatter.png",
        multistart_rows,
        title=f"Multi-start convergence: global ({args.fit_split_tag}/{args.cv_fold or 'all'})",
    )

    boot_cfg = protocol.get("bootstrap", {})
    n_boot = int(boot_cfg.get("n_boot", 0))
    boot_summary_json: dict[str, Any] = {}
    if n_boot > 0:
        dt_s = float(protocol["preprocess"]["resample"]["dt_s"])
        block_s = float(boot_cfg.get("block_s", 1.0))
        boot_seed = int(boot_cfg.get("seed", global_seed))

        def _fit_fn(seg_samples: list[pd.DataFrame]) -> dict[str, float]:
            s3 = fit_damping_nnls(segment_dfs=seg_samples, I_sys=I_sys, K_sys=K_sys, protocol=protocol)
            s4 = fit_params_ode(
                segment_dfs=seg_samples,
                protocol=protocol,
                model_cfg=model_cfg,
                init_guess={"d_q": s3.d_q, "d_qq": s3.d_qq},
            )
            return {
                "mu_theta": float(s4.mu_theta),
                "d_q": float(s4.d_q),
                "d_qq": float(s4.d_qq),
                "K_cable": float(s4.K_cable),
                "cost": float(s4.cost),
            }

        boot = bootstrap_fit_segments(
            segment_dfs=segment_dfs,
            fit_fn=_fit_fn,
            n_boot=n_boot,
            block_s=block_s,
            dt_s=dt_s,
            seed=boot_seed,
        )
        boot_samples_df = pd.DataFrame(boot.samples)
        boot_samples_df.to_csv(global_out / "bootstrap_samples.csv", index=False)
        corr_df = pd.DataFrame(boot.params_corr)
        corr_df.to_csv(global_out / "bootstrap_corr.csv", index=True)
        boot_summary_json = {
            "n_success": int(boot.n_success),
            "n_total": int(boot.n_total),
            "params_mean": boot.params_mean,
            "params_ci95": {k: [v[0], v[1]] for k, v in boot.params_ci95.items()},
        }
        with (global_out / "bootstrap_summary.json").open("w", encoding="utf-8") as f:
            json.dump(boot_summary_json, f, ensure_ascii=False, indent=2)
        _save_scatter_png(
            global_out / "bootstrap_scatter.png",
            [{**r, "cost": float(r.get("cost", 0.0))} for r in boot.samples],
            title=f"Bootstrap convergence cloud: global ({args.fit_split_tag}/{args.cv_fold or 'all'})",
        )

    out = {
        "run_id": "global",
        "fit_scope": "global",
        "fit_split_tag": args.fit_split_tag,
        "cv_fold": args.cv_fold,
        "selected_run_ids": selected_run_ids,
        "selected_segment_ids": selected_loaded_df["segment_id"].astype(str).tolist(),
        "n_runs_requested": len(selected_run_ids_requested),
        "n_runs_selected": len(selected_run_ids),
        "n_segments_requested": int(len(selected_rows)),
        "n_segments_selected": int(len(segment_dfs)),
        "n_segments_skipped": int(len(skipped_segments)),
        "skipped_runs": sorted(skipped_runs),
        "skipped_segments": skipped_segments,
        "step3_d_q": step3.d_q,
        "step3_d_qq": step3.d_qq,
        "step3_n_cycles": step3.n_cycles_used,
        "step4_mu_theta": float(best_step4["mu_theta"]),
        "step4_d_q": float(best_step4["d_q"]),
        "step4_d_qq": float(best_step4["d_qq"]),
        "step4_K_cable": float(best_step4["K_cable"]),
        "step4_success": bool(best_step4["success"]),
        "step4_cost": float(best_step4["cost"]),
        "step4_nfev": int(best_step4["nfev"]),
        "bootstrap": boot_summary_json,
    }
    with (global_out / "identified_params.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with (global_out / "identified_params.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)

    summary = {
        "run_id": "global",
        "fit_scope": "global",
        "fit_split_tag": args.fit_split_tag,
        "cv_fold": args.cv_fold,
        "n_runs_requested": len(selected_run_ids_requested),
        "n_runs_selected": len(selected_run_ids),
        "n_segments_requested": int(len(selected_rows)),
        "n_segments_selected": int(len(segment_dfs)),
        "n_segments_skipped": int(len(skipped_segments)),
        "skipped_runs": ";".join(sorted(skipped_runs)),
        "step3_d_q": step3.d_q,
        "step3_d_qq": step3.d_qq,
        "step3_n_cycles": step3.n_cycles_used,
        "step4_mu_theta": float(best_step4["mu_theta"]),
        "step4_d_q": float(best_step4["d_q"]),
        "step4_d_qq": float(best_step4["d_qq"]),
        "step4_K_cable": float(best_step4["K_cable"]),
        "step4_success": bool(best_step4["success"]),
        "step4_cost": float(best_step4["cost"]),
        "step4_nfev": int(best_step4["nfev"]),
        "bootstrap_n_success": int(boot_summary_json.get("n_success", 0)),
        "bootstrap_n_total": int(boot_summary_json.get("n_total", 0)),
    }
    pd.DataFrame([summary]).to_csv(args.out_dir / "identification_summary.csv", index=False)


if __name__ == "__main__":
    main()
