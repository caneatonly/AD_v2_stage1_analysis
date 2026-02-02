from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sim_flip.src.dynamics import make_event_theta_deg, simulate


def compute_metrics(df) -> dict:
    theta_deg = df["theta_deg"].to_numpy()
    t = df["t"].to_numpy()
    q = df["q"].to_numpy()

    above_80 = np.where(theta_deg >= 80.0)[0]
    t_80 = float(t[above_80[0]]) if len(above_80) else float("nan")

    return {
        "t_80": t_80,
        "theta_max": float(theta_deg.max()),
        "q_max": float(np.max(np.abs(q))),
    }


def main() -> None:
    y0 = [0.0, 0.0, 0.0, 0.0]
    t_span = (0.0, 20.0)
    dt_out = 0.01

    result = simulate(
        y0=y0,
        t_span=t_span,
        dt_out=dt_out,
        events=[make_event_theta_deg(80.0)],
    )

    if not result.success:
        print(f"WARNING: solver ended early (status={result.status}): {result.message}")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = result.data
    metrics = compute_metrics(df)

    df.to_csv(out_dir / "baseline_timeseries.csv", index=False)
    with (out_dir / "baseline_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
