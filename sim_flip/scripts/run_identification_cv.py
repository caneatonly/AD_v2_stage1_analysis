from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sim_flip.analysis.bootstrap import bootstrap_fit_segments
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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _save_reproducibility_artifacts(*, run_out_dir: Path, protocol_path: Path) -> None:
    run_out_dir.mkdir(parents=True, exist_ok=True)
    snap_path = run_out_dir / "protocol_snapshot.yaml"
    if protocol_path.exists():
        shutil.copy2(protocol_path, snap_path)
    else:
        snap_path.write_text("# protocol path not found\n", encoding="utf-8")

    git_path = run_out_dir / "git_commit.txt"
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

    env_path = run_out_dir / "python_env.txt"
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
    starts: list[dict[str, float]] = [dict(init_from_step3)]
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


def _deterministic_run_seed(base_seed: int, run_id: str) -> int:
    acc = int(base_seed)
    for i, ch in enumerate(str(run_id)):
        acc += (i + 1) * ord(ch)
    return acc


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
    global_seed = int(protocol.get("reproducibility", {}).get("random_seed", 0))

    summary_rows = []
    for txt in sorted(args.raw_dir.glob("R*.txt")):
        run_id = run_id_from_path(txt)
        run_out = args.out_dir / run_id
        run_out.mkdir(parents=True, exist_ok=True)
        _save_reproducibility_artifacts(run_out_dir=run_out, protocol_path=args.protocol)

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
        freq_path = run_out / "frequency.csv"
        freq_df.to_csv(freq_path, index=False)

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
            seed=_deterministic_run_seed(global_seed, run_id),
        )
        pd.DataFrame(multistart_rows).to_csv(run_out / "multistart_convergence.csv", index=False)
        _save_scatter_png(run_out / "multistart_scatter.png", multistart_rows, title=f"Multi-start convergence: {run_id}")

        boot_cfg = protocol.get("bootstrap", {})
        n_boot = int(boot_cfg.get("n_boot", 0))
        boot_summary_json = {}
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
            boot_samples_df.to_csv(run_out / "bootstrap_samples.csv", index=False)
            corr_df = pd.DataFrame(boot.params_corr)
            corr_df.to_csv(run_out / "bootstrap_corr.csv", index=True)
            boot_summary_json = {
                "n_success": int(boot.n_success),
                "n_total": int(boot.n_total),
                "params_mean": boot.params_mean,
                "params_ci95": {k: [v[0], v[1]] for k, v in boot.params_ci95.items()},
            }
            with (run_out / "bootstrap_summary.json").open("w", encoding="utf-8") as f:
                json.dump(boot_summary_json, f, ensure_ascii=False, indent=2)
            _save_scatter_png(
                run_out / "bootstrap_scatter.png",
                [{**r, "cost": float(r.get("cost", 0.0))} for r in boot.samples],
                title=f"Bootstrap convergence cloud: {run_id}",
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
            "step4_mu_theta": float(best_step4["mu_theta"]),
            "step4_d_q": float(best_step4["d_q"]),
            "step4_d_qq": float(best_step4["d_qq"]),
            "step4_K_cable": float(best_step4["K_cable"]),
            "step4_success": bool(best_step4["success"]),
            "step4_cost": float(best_step4["cost"]),
            "step4_nfev": int(best_step4["nfev"]),
            "bootstrap": boot_summary_json,
        }
        with (run_out / "identified_params.json").open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        with (run_out / "identified_params.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
        summary_rows.append(out)

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(args.out_dir / "identification_summary.csv", index=False)


if __name__ == "__main__":
    main()
