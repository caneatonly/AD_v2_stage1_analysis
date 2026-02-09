from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls


@dataclass(frozen=True)
class Step3Result:
    d_q: float
    d_qq: float
    n_cycles_used: int
    residual_norm: float


def _zero_crossings_times(x: np.ndarray, dt: float) -> np.ndarray:
    s = np.sign(x)
    s[s == 0] = 1.0
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    return idx.astype(float) * dt


def fit_damping_nnls(
    *,
    segment_dfs: list[pd.DataFrame],
    I_sys: float,
    K_sys: float,
    protocol: dict[str, Any],
) -> Step3Result:
    cfg = protocol["step3_energy"]
    min_amp = float(cfg["min_cycle_amplitude_deg"])
    min_cycles = int(cfg["min_cycles"])

    X_terms: list[list[float]] = []
    Y_terms: list[float] = []
    n_cycles = 0

    for seg in segment_dfs:
        t = seg["t_rel_s"].to_numpy(dtype=float)
        dt = float(np.mean(np.diff(t)))
        theta = seg["theta_rad"].to_numpy(dtype=float) - np.deg2rad(90.0)
        omega = seg["q_rad_s"].to_numpy(dtype=float)
        zc = _zero_crossings_times(theta, dt)
        if len(zc) < 3:
            continue
        idx = np.clip(np.round(zc / dt).astype(int), 0, len(theta) - 1)
        for i in range(len(idx) - 2):
            i0 = idx[i]
            i2 = idx[i + 2]
            if i2 <= i0 + 2:
                continue
            th = theta[i0:i2]
            om = omega[i0:i2]
            if np.rad2deg(np.max(np.abs(th))) < min_amp:
                continue
            e0 = 0.5 * I_sys * (om[0] ** 2) + 0.5 * K_sys * (th[0] ** 2)
            e1 = 0.5 * I_sys * (om[-1] ** 2) + 0.5 * K_sys * (th[-1] ** 2)
            dE = max(0.0, e0 - e1)
            i_w2 = float(np.trapz(om**2, dx=dt))
            i_w3 = float(np.trapz(np.abs(om) ** 3, dx=dt))
            if i_w2 <= 0 and i_w3 <= 0:
                continue
            X_terms.append([i_w2, i_w3])
            Y_terms.append(dE)
            n_cycles += 1

    if n_cycles < min_cycles:
        raise RuntimeError(f"Step3 insufficient cycles: {n_cycles} < {min_cycles}")

    X = np.asarray(X_terms, dtype=float)
    Y = np.asarray(Y_terms, dtype=float)
    coeffs, rnorm = nnls(X, Y)
    return Step3Result(
        d_q=float(coeffs[0]),
        d_qq=float(coeffs[1]),
        n_cycles_used=int(n_cycles),
        residual_norm=float(rnorm),
    )
