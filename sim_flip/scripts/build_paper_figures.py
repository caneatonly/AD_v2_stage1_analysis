from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot_gapscore(metrics_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(metrics_csv)
    if "segment_id" not in df.columns or "gap_score" not in df.columns:
        raise ValueError(f"{metrics_csv} missing segment_id/gap_score")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df["segment_id"], df["gap_score"], color="#1f77b4")
    ax.set_xlabel("Segment")
    ax.set_ylabel("GapScore")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready summary figures from pipeline outputs.")
    parser.add_argument("--sim-real-metrics", type=Path, default=Path("sim_flip/results/sim_real_eval/sim_real_metrics.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.sim_real_metrics.exists():
        _plot_gapscore(args.sim_real_metrics, args.out_dir / "fig_gapscore.png")


if __name__ == "__main__":
    main()

