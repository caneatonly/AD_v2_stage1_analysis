from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sim_flip.src.dynamics import load_params_yaml, simulate


def _run_with_damping_scale(params: dict, scale: float, t_end: float = 20.0) -> dict:
    p = dict(params)
    p["damping"] = dict(params["damping"])
    p["damping"]["d_q"] = float(params["damping"]["d_q"]) * scale
    p["damping"]["d_q_abs"] = float(params["damping"]["d_q_abs"]) * scale
    res = simulate(y0=[0.0, 0.0, 0.0, 0.0], t_span=(0.0, t_end), dt_out=0.01, params=p)
    theta = res.data["theta_deg"].to_numpy(dtype=float)
    t = res.data["t"].to_numpy(dtype=float)
    t80 = t[theta >= 80.0]
    return {
        "k_damp_scale": float(scale),
        "theta_final_deg": float(theta[-1]),
        "theta_max_deg": float(np.max(theta)),
        "t80_s": float(t80[0]) if len(t80) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run damping sensitivity suite.")
    parser.add_argument("--params", type=Path, default=Path("sim_flip/configs/params_nominal.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("sim_flip/results/sensitivity"))
    parser.add_argument("--k-min", type=float, default=0.5)
    parser.add_argument("--k-max", type=float, default=2.0)
    parser.add_argument("--k-n", type=int, default=7)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    params = load_params_yaml(args.params)
    scales = np.linspace(args.k_min, args.k_max, args.k_n)
    rows = [_run_with_damping_scale(params, float(s)) for s in scales]
    pd.DataFrame(rows).to_csv(args.out_dir / "damping_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()

