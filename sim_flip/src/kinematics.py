# 该脚本定义了用于计算水下航行器运动学参数的函数和数据结构。
from __future__ import annotations

import math
from dataclasses import dataclass

from sim_flip.src.conventions import alpha_deg_from_uw
from sim_flip.src.cfd_table import map_alpha_rule_b


@dataclass(frozen=True)
class Kinematics:
    V: float
    Q: float
    alpha_raw_deg: float
    alpha_deg: float


def speed(u: float, w: float) -> float:
    return math.hypot(float(u), float(w))


def dynamic_pressure(*, rho: float, u: float, w: float, V_eps: float) -> tuple[float, float]:
    """Return (V, Q). If V < V_eps, return Q=0 for robustness."""

    V = speed(u, w)
    if V < float(V_eps):
        return V, 0.0
    Q = 0.5 * float(rho) * (V * V)
    return V, Q


def alpha_with_singularity_protection(
    *,
    u: float,
    w: float,
    V: float,
    V_eps: float,
    alpha_prev_deg: float | None = None,
    hold_when_V_small: bool = False,
) -> float:
    """Compute alpha_deg with a frozen definition and V≈0 protection.

    If V < V_eps:
      - hold_when_V_small=False -> alpha_deg = 0
      - hold_when_V_small=True  -> alpha_deg = alpha_prev_deg if provided else 0
    """

    if float(V) < float(V_eps):
        if hold_when_V_small and (alpha_prev_deg is not None) and math.isfinite(alpha_prev_deg):
            return float(alpha_prev_deg)
        return 0.0

    return float(alpha_deg_from_uw(float(u), float(w)))


def clamp(x: float, lo: float, hi: float) -> float:
    lo_f = float(lo)
    hi_f = float(hi)
    if lo_f > hi_f:
        raise ValueError(f"clamp bounds invalid: lo={lo_f} > hi={hi_f}")
    return min(max(float(x), lo_f), hi_f)


def compute_kinematics(
    *,
    rho: float,
    u: float,
    w: float,
    V_eps: float,
    alpha_min_deg: float,
    alpha_max_deg: float,
    alpha_prev_deg: float | None = None,
    hold_when_V_small: bool = False,
) -> Kinematics:
    """Compute (V, Q, alpha_raw_deg, alpha_deg).

    - alpha_raw_deg follows the frozen convention: atan2(w,u) in degrees.
    - alpha_deg is the CFD lookup alpha after applying Rule-B folding/sign logic,
      then clamped to [alpha_min_deg, alpha_max_deg].
    """

    V, Q = dynamic_pressure(rho=rho, u=u, w=w, V_eps=V_eps)
    alpha_raw_deg = alpha_with_singularity_protection(
        u=u,
        w=w,
        V=V,
        V_eps=V_eps,
        alpha_prev_deg=alpha_prev_deg,
        hold_when_V_small=hold_when_V_small,
    )

    alpha_lut_deg, _, _, _ = map_alpha_rule_b(alpha_raw_deg)
    alpha_deg = clamp(alpha_lut_deg, alpha_min_deg, alpha_max_deg)

    return Kinematics(V=V, Q=Q, alpha_raw_deg=alpha_raw_deg, alpha_deg=alpha_deg)
