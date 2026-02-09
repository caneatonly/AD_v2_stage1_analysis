from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapSummary:
    n_success: int
    n_total: int
    params_mean: dict[str, float]
    params_ci95: dict[str, tuple[float, float]]


def _block_bootstrap_1d(x: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    if n <= 1:
        return x.copy()
    out = []
    while len(out) < n:
        start = int(rng.integers(0, max(1, n - block_len + 1)))
        out.extend(x[start : start + block_len].tolist())
    return np.asarray(out[:n], dtype=float)


def bootstrap_fit_segments(
    *,
    segment_dfs: list[pd.DataFrame],
    fit_fn: Callable[[list[pd.DataFrame]], dict[str, float]],
    n_boot: int,
    block_s: float,
    dt_s: float,
    seed: int = 0,
) -> BootstrapSummary:
    rng = np.random.default_rng(seed)
    block_len = max(2, int(round(block_s / dt_s)))
    keys = ["mu_theta", "d_q", "d_qq", "K_cable"]
    rows = []

    for _ in range(int(n_boot)):
        sample_segments = []
        for seg in segment_dfs:
            s = seg.copy()
            for col in ["theta_rad", "q_rad_s"]:
                s[col] = _block_bootstrap_1d(s[col].to_numpy(dtype=float), block_len, rng)
            if "q_dot_rad_s2" in s.columns:
                s["q_dot_rad_s2"] = _block_bootstrap_1d(
                    s["q_dot_rad_s2"].to_numpy(dtype=float), block_len, rng
                )
            sample_segments.append(s)

        try:
            row = fit_fn(sample_segments)
        except Exception:
            continue
        rows.append({k: float(row[k]) for k in keys if k in row})

    if not rows:
        raise RuntimeError("bootstrap_fit_segments: no successful bootstrap sample")

    df = pd.DataFrame(rows)
    mean = {c: float(df[c].mean()) for c in df.columns}
    ci95 = {
        c: (
            float(np.percentile(df[c], 2.5)),
            float(np.percentile(df[c], 97.5)),
        )
        for c in df.columns
    }
    return BootstrapSummary(
        n_success=int(len(df)),
        n_total=int(n_boot),
        params_mean=mean,
        params_ci95=ci95,
    )

