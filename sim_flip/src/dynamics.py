from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from sim_flip.src.added_mass import AddedMassTotals, EffectiveInertia, compute_from_param_tree
from sim_flip.src.cfd_table import CfdInterpolator, load_default_cfd_interpolator
from sim_flip.src.kinematics import compute_kinematics


@dataclass(frozen=True)
class FlipState:
    u: float
    w: float
    q: float
    theta: float

# 状态诊断数据类，包含模拟过程中各个时间点的状态和力矩信息
@dataclass(frozen=True)
class FlipDiagnostics:
    t: float # 时间 s
    u: float # 前向速度 m/s
    w: float # 垂向速度 m/s
    q: float # 角速度 rad/s
    theta: float # 俯仰角 rad
    theta_deg: float # 俯仰角度 deg
    V: float # 速度幅值 m/s
    Q: float # 动压 Pa
    alpha_raw_deg: float # 未限幅攻角 deg
    alpha_deg: float # 限幅后攻角 deg
    Cx: float # 阻力系数
    Cz: float # 升力系数
    Cm: float # 助航矩系数
    X_cfd: float # CFD计算的X方向力 N
    Z_cfd: float # CFD计算的Z方向力 N
    M_cfd: float # CFD计算的力矩 N·m
    M_damp: float # 阻尼力矩 N·m
    M_bg: float # 浮力重力力矩 N·m
    M_cable: float # 缆绳力矩 N·m
    T: float # 推进器推力 N
    M_thruster: float # 推进器力矩 N·m


@dataclass(frozen=True)
class SimulationResult:
    t: Iterable[float]  # 求解器输出的时间序列（对应 t_eval，单位 s）。
    y: Iterable[Iterable[float]] # 求解器输出的状态变量时间序列数组，形状为 (4, len(t))。 y=[u,w,q,θ] 
    data: "DataFrame" 
    events: dict
    success: bool
    status: int
    message: str


class _AlphaHoldState:
    def __init__(self) -> None:
        self._last_alpha_deg: Optional[float] = None

    def update(self, alpha_deg: float) -> None:
        self._last_alpha_deg = float(alpha_deg)

    @property
    def last(self) -> Optional[float]:
        return self._last_alpha_deg


def load_params_yaml(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r") as f:
        return yaml.safe_load(f)


def _get_constants(params: dict) -> dict:
    return params["constants"]


def _get_rigid_body(params: dict) -> dict:
    return params["rigid_body"]


def _get_buoyancy(params: dict) -> dict:
    return params["buoyancy_restore"]


def _get_damping(params: dict) -> dict:
    return params["damping"]


def _get_cable(params: dict) -> dict:
    return params["cable"]


def _get_numerics(params: dict) -> dict:
    return params["numerics"]


def _theta_eq_rad(params: dict) -> float:
    cable = _get_cable(params)
    return math.radians(float(cable["theta_eq_deg"]))


def _compute_forces_and_moments(
    *,
    u: float,
    w: float,
    q: float,
    theta: float,
    params: dict,
    totals: AddedMassTotals,
    eff: EffectiveInertia,
    cfd: CfdInterpolator,
    alpha_prev_deg: Optional[float],
    T_fn: Optional[Callable[[float, FlipState, dict], float]],
    M_thruster_fn: Optional[Callable[[float, FlipState, dict], float]],
    t: float,
) -> tuple[FlipDiagnostics, tuple[float, float, float, float]]:
    constants = _get_constants(params)
    rb = _get_rigid_body(params)
    buoy = _get_buoyancy(params)
    damp = _get_damping(params)
    cable = _get_cable(params)
    numerics = _get_numerics(params)

    rho = float(constants["rho"])
    g = float(constants["g"])
    A_ref = float(constants["A_ref"])
    L_ref = float(constants["L_ref"])

    V_eps = float(numerics["V_eps"])
    alpha_hold = bool(numerics["alpha_hold_when_V_small"])

    kin = compute_kinematics(
        rho=rho,
        u=u,
        w=w,
        V_eps=V_eps,
        alpha_min_deg=cfd.alpha_min_deg,
        alpha_max_deg=cfd.alpha_max_deg,
        alpha_prev_deg=alpha_prev_deg,
        hold_when_V_small=alpha_hold,
    )

    Cx, Cz, Cm = cfd.coeffs_extended(kin.alpha_raw_deg)

    X_cfd = kin.Q * A_ref * Cx
    Z_cfd = kin.Q * A_ref * Cz
    M_cfd = kin.Q * A_ref * L_ref * Cm

    m_dry = float(rb["m_dry"])
    m_wet = float(rb["m_wet"])
    W_force = m_dry * g
    B_force = float(buoy["B_mass"]) * g

    W_minus_B = W_force - B_force

    T = 0.0
    if T_fn is not None:
        T = float(T_fn(t, FlipState(u=u, w=w, q=q, theta=theta), params))

    M_thruster = 0.0
    if M_thruster_fn is not None:
        M_thruster = float(M_thruster_fn(t, FlipState(u=u, w=w, q=q, theta=theta), params))

    d_q = float(damp["d_q"])
    d_q_abs = float(damp["d_q_abs"])

    x_b = float(buoy["x_b"])
    z_b = float(buoy["z_b"])

    M_bg = B_force * (z_b * math.sin(theta) + x_b * math.cos(theta))
    M_damp = -(d_q + d_q_abs * abs(q)) * q

    M_cable = 0.0
    if bool(cable["enabled"]):
        K_cable = float(cable["K_cable"])
        M_cable = -K_cable * (theta - _theta_eq_rad(params))

    u_dot = (
        -(m_dry - totals.Z_wdot_total) * w * q
        + X_cfd
        - W_minus_B * math.sin(theta)
        + T
    ) / eff.m_x

    w_dot = (
        +(m_dry - totals.X_udot_total) * u * q
        + Z_cfd
        + W_minus_B * math.cos(theta)
    ) / eff.m_z

    q_dot = (
        M_cfd
        + M_damp
        + M_bg
        + M_cable
        + M_thruster
    ) / eff.I_y

    theta_dot = q

    diag = FlipDiagnostics(
        t=t,
        u=u,
        w=w,
        q=q,
        theta=theta,
        theta_deg=math.degrees(theta),
        V=kin.V,
        Q=kin.Q,
        alpha_raw_deg=kin.alpha_raw_deg,
        alpha_deg=kin.alpha_deg,
        Cx=Cx,
        Cz=Cz,
        Cm=Cm,
        X_cfd=X_cfd,
        Z_cfd=Z_cfd,
        M_cfd=M_cfd,
        M_damp=M_damp,
        M_bg=M_bg,
        M_cable=M_cable,
        T=T,
        M_thruster=M_thruster,
    )

    return diag, (u_dot, w_dot, q_dot, theta_dot)


def build_rhs(
    *,
    params: dict,
    cfd: CfdInterpolator,
    T_fn: Optional[Callable[[float, FlipState, dict], float]] = None,
    M_thruster_fn: Optional[Callable[[float, FlipState, dict], float]] = None,
) -> Callable[[float, Iterable[float]], Iterable[float]]:
    totals, eff = compute_from_param_tree(params)
    alpha_hold_state = _AlphaHoldState()

    def _rhs(t: float, y: Iterable[float]) -> Iterable[float]:
        u, w, q, theta = (float(v) for v in y)
        diag, deriv = _compute_forces_and_moments(
            u=u,
            w=w,
            q=q,
            theta=theta,
            params=params,
            totals=totals,
            eff=eff,
            cfd=cfd,
            alpha_prev_deg=alpha_hold_state.last,
            T_fn=T_fn,
            M_thruster_fn=M_thruster_fn,
            t=t,
        )
        alpha_hold_state.update(diag.alpha_raw_deg)
        return deriv

    return _rhs


def evaluate_diagnostics(
    *,
    t: float,
    y: Iterable[float],
    params: dict,
    cfd: CfdInterpolator,
    alpha_prev_deg: Optional[float] = None,
    T_fn: Optional[Callable[[float, FlipState, dict], float]] = None,
    M_thruster_fn: Optional[Callable[[float, FlipState, dict], float]] = None,
) -> FlipDiagnostics:
    totals, eff = compute_from_param_tree(params)
    u, w, q, theta = (float(v) for v in y)
    diag, _ = _compute_forces_and_moments(
        u=u,
        w=w,
        q=q,
        theta=theta,
        params=params,
        totals=totals,
        eff=eff,
        cfd=cfd,
        alpha_prev_deg=alpha_prev_deg,
        T_fn=T_fn,
        M_thruster_fn=M_thruster_fn,
        t=t,
    )
    return diag


def make_event_theta_deg(target_deg: float) -> Callable[[float, Iterable[float]], float]:
    target_rad = math.radians(float(target_deg))

    def _event(t: float, y: Iterable[float]) -> float:
        return float(y[3]) - target_rad

    _event.terminal = True
    _event.direction = 1
    return _event


def make_event_q_zero() -> Callable[[float, Iterable[float]], float]:
    def _event(t: float, y: Iterable[float]) -> float:
        return float(y[2])

    _event.terminal = False
    _event.direction = 0
    return _event


def make_event_speed_small(V_eps: float) -> Callable[[float, Iterable[float]], float]:
    def _event(t: float, y: Iterable[float]) -> float:
        u = float(y[0])
        w = float(y[1])
        return math.hypot(u, w) - float(V_eps)

    _event.terminal = False
    _event.direction = -1
    return _event


def simulate(
    *,
    y0: Iterable[float],
    t_span: tuple[float, float],
    dt_out: float,
    params: Optional[dict] = None,
    params_path: Optional[str | Path] = None,
    cfd: Optional[CfdInterpolator] = None,
    solver: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    events: Optional[list[Callable[[float, Iterable[float]], float]]] = None,
    T_fn: Optional[Callable[[float, FlipState, dict], float]] = None,
    M_thruster_fn: Optional[Callable[[float, FlipState, dict], float]] = None,
) -> SimulationResult:
    import numpy as np
    import pandas as pd

    if params is None:
        if params_path is None:
            params_path = Path(__file__).resolve().parents[1] / "configs" / "params_nominal.yaml"
        params = load_params_yaml(params_path)

    if cfd is None:
        cfd = load_default_cfd_interpolator()

    rhs = build_rhs(params=params, cfd=cfd, T_fn=T_fn, M_thruster_fn=M_thruster_fn)

    t0, tf = float(t_span[0]), float(t_span[1])
    if dt_out <= 0:
        raise ValueError("dt_out must be positive")

    # Build t_eval robustly: never allow points outside t_span (SciPy raises).
    # np.arange can overshoot tf due to floating-point rounding.
    if tf == t0:
        t_eval = np.array([t0], dtype=float)
    else:
        direction = 1.0 if tf > t0 else -1.0
        span = abs(tf - t0)
        n = int(np.floor(span / dt_out))
        base = t0 + direction * dt_out * np.arange(n + 1, dtype=float)
        # Ensure we include tf exactly as the final point.
        if (direction > 0 and base[-1] < tf) or (direction < 0 and base[-1] > tf):
            t_eval = np.concatenate([base, np.array([tf], dtype=float)])
        else:
            t_eval = base

        lo, hi = (t0, tf) if t0 <= tf else (tf, t0)
        t_eval = np.clip(t_eval, lo, hi)

    try:
        from scipy.integrate import solve_ivp
    except Exception as exc:  # noqa: BLE001
        raise ImportError("scipy is required for simulate(); install sim_flip/requirements.txt") from exc

    sol = solve_ivp(
        rhs,
        (t0, tf),
        [float(v) for v in y0],
        method=solver,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        events=events,
    )

    diags = []
    alpha_hold: Optional[float] = None
    for i, t in enumerate(sol.t):
        y = sol.y[:, i]
        diag = evaluate_diagnostics(
            t=float(t),
            y=y, # 状态变量数组 [u,w,q,θ]
            params=params,
            cfd=cfd,
            alpha_prev_deg=alpha_hold,
            T_fn=T_fn,
            M_thruster_fn=M_thruster_fn,
        )
        alpha_hold = diag.alpha_raw_deg
        diags.append(diag)

    df = pd.DataFrame([d.__dict__ for d in diags])

    event_map: dict = {}
    if sol.t_events is not None and events is not None:
        for idx, ev in enumerate(events):
            event_map[getattr(ev, "__name__", f"event_{idx}")] = sol.t_events[idx]

    return SimulationResult(
        t=sol.t,
        y=sol.y,
        data=df,
        events=event_map,
        success=bool(sol.success),
        status=int(sol.status),
        message=str(sol.message),
    )
