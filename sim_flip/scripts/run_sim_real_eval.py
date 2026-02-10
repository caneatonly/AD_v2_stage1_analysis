from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sim_flip.analysis.metrics import compute_sim_real_metrics
from sim_flip.src.dynamics import simulate


def _select_eval_ids(
    *,
    manifest_path: Path,
    split_tags: list[str],
    cv_fold: str,
) -> set[str]:
    if not manifest_path.exists():
        return set()
    mdf = pd.read_csv(manifest_path)
    if mdf.empty:
        return set()
    req = {"segment_id", "split_tag", "cv_fold"}
    if not req.issubset(mdf.columns):
        raise ValueError(f"manifest missing columns: {sorted(req - set(mdf.columns))}")
    tag_set = {t.strip() for t in split_tags if t.strip()}
    mask = mdf["split_tag"].astype(str).isin(tag_set)
    if cv_fold.strip():
        mask = mask & (mdf["cv_fold"].fillna("").astype(str) == cv_fold.strip())
    return set(mdf.loc[mask, "segment_id"].astype(str).tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Sim-real evaluation on exported segment CSV files.")
    parser.add_argument("--segment-dir", type=Path, default=Path("sim_flip/data/derived/segments"))
    parser.add_argument("--manifest", type=Path, default=Path("sim_flip/configs/experiment_manifest.csv"))
    parser.add_argument(
        "--eval-split-tags",
        type=str,
        default="val,test",
        help="Comma-separated split tags to evaluate, e.g. val,test",
    )
    parser.add_argument("--cv-fold", type=str, default="", help="Optional cv_fold filter (exact match).")
    parser.add_argument("--params", type=Path, default=Path("sim_flip/configs/params_nominal.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("sim_flip/results/sim_real_eval"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tags = [t.strip() for t in args.eval_split_tags.split(",")]
    selected_ids = _select_eval_ids(manifest_path=args.manifest, split_tags=tags, cv_fold=args.cv_fold)
    if not selected_ids:
        print(f"[warn] no segment selected by split_tags={tags} cv_fold={args.cv_fold!r}; nothing to evaluate")
        return

    rows = []
    for seg_path in sorted(args.segment_dir.glob("*_S*.csv")):
        seg = pd.read_csv(seg_path)
        if seg.empty:
            continue
        run_id = str(seg["run_id"].iloc[0])
        segment_id = str(seg["segment_id"].iloc[0])
        if selected_ids and segment_id not in selected_ids:
            continue
        t_exp = seg["t_rel_s"].to_numpy(dtype=float)
        theta_exp_deg = seg["theta_deg"].to_numpy(dtype=float)
        q_exp = seg["q_rad_s"].to_numpy(dtype=float) if "q_rad_s" in seg.columns else None

        y0 = [0.0, 0.0, float(q_exp[0] if q_exp is not None else 0.0), np.deg2rad(float(theta_exp_deg[0]))]
        sim = simulate(
            y0=y0,
            t_span=(0.0, float(t_exp[-1])),
            dt_out=float(np.mean(np.diff(t_exp))) if len(t_exp) > 2 else 0.01,
            params_path=args.params,
        )
        m = compute_sim_real_metrics(
            t_sim=sim.data["t"].to_numpy(dtype=float),
            theta_sim_deg=sim.data["theta_deg"].to_numpy(dtype=float),
            t_exp=t_exp,
            theta_exp_deg=theta_exp_deg,
            q_sim=sim.data["q"].to_numpy(dtype=float) if "q" in sim.data.columns else None,
            q_exp=q_exp,
            theta0_deg=float(theta_exp_deg[0]),
        )
        row = {
            "run_id": run_id,
            "segment_id": segment_id,
            "eval_split_tags": ",".join(tags),
            "cv_fold": args.cv_fold,
            **m.to_dict(),
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(args.out_dir / "sim_real_metrics.csv", index=False)
        with (args.out_dir / "sim_real_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
