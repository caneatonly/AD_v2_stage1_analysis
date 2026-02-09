from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Step4Result:
    mu_theta: float
    d_q: float
    d_qq: float
    K_cable: float
    success: bool
    cost: float
    nfev: int
    message: str


def _simulate_one_segment(
    *,
    t: np.ndarray,
    theta0_abs: float,
    omega0: float,
    mu_theta: float,
    d_q: float,
    d_qq: float,
    K_cable: float,
    model_cfg: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    I0 = float(model_cfg["I0"])
    Iwater = float(model_cfg["Iwater"])
    B_force = float(model_cfg["B_force"])
    x_b = float(model_cfg["x_b"])
    K_geo = B_force * x_b
    I = I0 + mu_theta * Iwater
    K = K_geo + K_cable

    theta_dev0 = float(theta0_abs - np.deg2rad(90.0))

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        th, om = float(y[0]), float(y[1])
        om_dot = -(d_q + d_qq * abs(om)) * om / I - (K / I) * th
        return np.asarray([om, om_dot], dtype=float)

    sol = solve_ivp(
        rhs,
        (float(t[0]), float(t[-1])),
        np.asarray([theta_dev0, omega0], dtype=float),
        t_eval=t,
        rtol=1e-6,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"Step4 ODE solve failed: {sol.message}")

    theta_abs = sol.y[0] + np.deg2rad(90.0)
    omega = sol.y[1]
    return theta_abs, omega


def fit_params_ode(
    *,
    segment_dfs: list[pd.DataFrame],
    protocol: dict[str, Any],
    model_cfg: dict[str, float],
    init_guess: dict[str, float] | None = None,
) -> Step4Result:
    cfg = protocol["step4_ode"]
    w_omega = float(cfg["lambda_omega"])
    bounds_cfg = cfg["bounds"]

    mu0 = float((init_guess or {}).get("mu_theta", cfg["init"]["mu_theta"]))
    dq0 = float((init_guess or {}).get("d_q", cfg["init"]["d_q"]))
    dqq0 = float((init_guess or {}).get("d_qq", cfg["init"]["d_qq"]))
    kc0 = float((init_guess or {}).get("K_cable", cfg["init"]["K_cable"]))

    x0 = np.asarray([mu0, dq0, dqq0, kc0], dtype=float)
    lb = np.asarray(
        [
            float(bounds_cfg["mu_theta"][0]),
            float(bounds_cfg["d_q"][0]),
            float(bounds_cfg["d_qq"][0]),
            float(bounds_cfg["K_cable"][0]),
        ],
        dtype=float,
    )
    ub = np.asarray(
        [
            float(bounds_cfg["mu_theta"][1]),
            float(bounds_cfg["d_q"][1]),
            float(bounds_cfg["d_qq"][1]),
            float(bounds_cfg["K_cable"][1]),
        ],
        dtype=float,
    )

    def residuals(x: np.ndarray) -> np.ndarray:
        mu, dq, dqq, kc = (float(v) for v in x)
        errs = []
        for seg in segment_dfs:
            t = seg["t_rel_s"].to_numpy(dtype=float)
            th_exp = seg["theta_rad"].to_numpy(dtype=float)
            om_exp = seg["q_rad_s"].to_numpy(dtype=float)
            th_sim, om_sim = _simulate_one_segment(
                t=t,
                theta0_abs=float(th_exp[0]),
                omega0=float(om_exp[0]),
                mu_theta=mu,
                d_q=dq,
                d_qq=dqq,
                K_cable=kc,
                model_cfg=model_cfg,
            )
            errs.append(th_sim - th_exp)
            errs.append(w_omega * (om_sim - om_exp))
        return np.concatenate(errs)

    res = least_squares(residuals, x0=x0, bounds=(lb, ub), method="trf")
    mu, dq, dqq, kc = (float(v) for v in res.x)
    return Step4Result(
        mu_theta=mu,
        d_q=dq,
        d_qq=dqq,
        K_cable=kc,
        success=bool(res.success),
        cost=float(res.cost),
        nfev=int(res.nfev),
        message=str(res.message),
    )

