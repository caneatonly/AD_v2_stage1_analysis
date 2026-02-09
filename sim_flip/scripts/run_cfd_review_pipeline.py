from __future__ import annotations

import argparse
from pathlib import Path

from sim_flip.analysis.cfd_pipeline import run_cfd_review_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed CFD review pipeline and export cfd_table_clean.csv.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory with C_*_overmesh.csv files.")
    parser.add_argument("--out-table", type=Path, default=Path("sim_flip/data/cfd_table_clean.csv"))
    parser.add_argument("--out-report", type=Path, default=Path("sim_flip/results/cfd_review_report.json"))
    parser.add_argument("--out-fig-dir", type=Path, default=Path("sim_flip/results/cfd_figures"))
    parser.add_argument("--n-steady-last", type=int, default=500)
    parser.add_argument("--angle-step", type=int, default=5)
    args = parser.parse_args()

    run_cfd_review_pipeline(
        source_dir=args.source_dir,
        out_table_path=args.out_table,
        out_report_path=args.out_report,
        out_figure_dir=args.out_fig_dir,
        n_steady_last=args.n_steady_last,
        angle_step_deg=args.angle_step,
    )


if __name__ == "__main__":
    main()

