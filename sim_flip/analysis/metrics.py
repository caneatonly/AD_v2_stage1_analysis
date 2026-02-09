from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d


def first_crossing_time(t: np.ndarray, y: np.ndarray, threshold: float) -> float:
    above = y >= threshold
    idx = np.where(above)[0]
    if len(idx) == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    t0, t1 = float(t[i - 1]), float(t[i])
    y0, y1 = float(y[i - 1]), float(y[i])
    if y1 == y0:
        return t1
    frac = (threshold - y0) / (y1 - y0)
    return t0 + frac * (t1 - t0)


@dataclass(frozen=True)
class SimRealMetrics:
    rmse_theta_deg: float
    mae_theta_deg: float
    max_abs_theta_deg: float
    taae_theta_deg: float
    tase_theta_deg: float
    dt90_s: float
    dt80_s: float
    overshoot_error_deg: float
    ss_mean_error_deg: float
    ss_std_sim_deg: float
    ss_std_exp_deg: float
    rmse_q_rad_s: Optional[float]
    mae_q_rad_s: Optional[float]
    gap_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_sim_real_metrics(
    *,
    t_sim: np.ndarray,
    theta_sim_deg: np.ndarray,
    t_exp: np.ndarray,
    theta_exp_deg: np.ndarray,
    q_sim: np.ndarray | None = None,
    q_exp: np.ndarray | None = None,
    theta0_deg: float | None = None,
    target_deg: float = 90.0,
    eps: float = 1e-9,
) -> SimRealMetrics:
    interp_theta = interp1d(t_sim, theta_sim_deg, bounds_error=False, fill_value="extrapolate")
    sim_on_exp = interp_theta(t_exp)
    err = sim_on_exp - theta_exp_deg

    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    taae = float(np.mean(np.abs(err)))
    tase = float(np.mean(err))

    t90_sim = first_crossing_time(t_sim, theta_sim_deg, target_deg)
    t90_exp = first_crossing_time(t_exp, theta_exp_deg, target_deg)
    t80_sim = first_crossing_time(t_sim, theta_sim_deg, 80.0)
    t80_exp = first_crossing_time(t_exp, theta_exp_deg, 80.0)
    dt90 = t90_sim - t90_exp if np.isfinite(t90_sim) and np.isfinite(t90_exp) else float("nan")
    dt80 = t80_sim - t80_exp if np.isfinite(t80_sim) and np.isfinite(t80_exp) else float("nan")

    over = float(np.max(theta_sim_deg) - np.max(theta_exp_deg))

    n_ss = max(1, int(0.2 * len(t_exp)))
    ss_sim = sim_on_exp[-n_ss:]
    ss_exp = theta_exp_deg[-n_ss:]
    ss_mean = float(np.mean(ss_sim) - np.mean(ss_exp))
    ss_std_sim = float(np.std(ss_sim))
    ss_std_exp = float(np.std(ss_exp))

    rmse_q = None
    mae_q = None
    if q_sim is not None and q_exp is not None and len(q_sim) > 0 and len(q_exp) > 0:
        interp_q = interp1d(t_sim, q_sim, bounds_error=False, fill_value="extrapolate")
        q_sim_on_exp = interp_q(t_exp)
        q_err = q_sim_on_exp - q_exp
        rmse_q = float(np.sqrt(np.mean(q_err**2)))
        mae_q = float(np.mean(np.abs(q_err)))

    th0 = float(theta0_deg if theta0_deg is not None else theta_exp_deg[0])
    norm_th = max(abs(target_deg - th0), eps)
    n_rmse_theta = rmse / norm_th
    n_mae_theta = mae / norm_th
    n_over = abs(over) / norm_th
    if np.isfinite(dt90) and np.isfinite(t90_exp) and abs(t90_exp) > eps:
        n_t = abs(dt90) / abs(t90_exp)
    elif np.isfinite(dt80) and np.isfinite(t80_exp) and abs(t80_exp) > eps:
        n_t = abs(dt80) / abs(t80_exp)
    else:
        n_t = 1.0

    if rmse_q is not None and q_exp is not None:
        denom_q = max(float(np.percentile(np.abs(q_exp), 95)), eps)
        n_rmse_q = rmse_q / denom_q
        gap = 0.35 * n_rmse_theta + 0.20 * n_mae_theta + 0.20 * n_rmse_q + 0.15 * n_t + 0.10 * n_over
    else:
        # Re-normalize weights when q is missing.
        gap = (0.35 * n_rmse_theta + 0.20 * n_mae_theta + 0.15 * n_t + 0.10 * n_over) / 0.80

    return SimRealMetrics(
        rmse_theta_deg=rmse,
        mae_theta_deg=mae,
        max_abs_theta_deg=max_abs,
        taae_theta_deg=taae,
        tase_theta_deg=tase,
        dt90_s=float(dt90),
        dt80_s=float(dt80),
        overshoot_error_deg=over,
        ss_mean_error_deg=ss_mean,
        ss_std_sim_deg=ss_std_sim,
        ss_std_exp_deg=ss_std_exp,
        rmse_q_rad_s=rmse_q,
        mae_q_rad_s=mae_q,
        gap_score=float(gap),
    )

