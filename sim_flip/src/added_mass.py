from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AddedMassTotals:
    """Total (outer + permeability-scaled inner) added-mass terms.

    Sign convention: these terms are expected to be negative in typical Fossen form.
    """

    X_udot_total: float  # [kg]
    Z_wdot_total: float  # [kg]
    M_qdot_total: float  # [kg*m^2]


@dataclass(frozen=True)
class EffectiveInertia:
    """Effective inertias used in the 3-DOF ODE denominators."""

    m_x: float  # [kg]  m_dry - X_udot_total
    m_z: float  # [kg]  m_dry - Z_wdot_total
    I_y: float  # [kg*m^2] Iyy - M_qdot_total


def _assert_unit_interval(name: str, value: float) -> None:
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def compute_added_mass_totals(
    *,
    X_udot_outer: float,
    Z_wdot_outer: float,
    M_qdot_outer: float,
    m_water_inner: float,
    I_water_inner: float,
    mu_x: float,
    mu_z: float,
    mu_theta: float,
) -> AddedMassTotals:
    """Compute total added mass/inertia using the anisotropic permeability model.

    Model (as per your write-up):
    - X_udot_total = X_udot_outer - mu_x * m_water_inner
    - Z_wdot_total = Z_wdot_outer - mu_z * m_water_inner
    - M_qdot_total = M_qdot_outer - mu_theta * I_water_inner
    """

    _assert_unit_interval("mu_x", mu_x)
    _assert_unit_interval("mu_z", mu_z)
    _assert_unit_interval("mu_theta", mu_theta)

    X_udot_total = float(X_udot_outer) - float(mu_x) * float(m_water_inner)
    Z_wdot_total = float(Z_wdot_outer) - float(mu_z) * float(m_water_inner)
    M_qdot_total = float(M_qdot_outer) - float(mu_theta) * float(I_water_inner)

    return AddedMassTotals(
        X_udot_total=X_udot_total,
        Z_wdot_total=Z_wdot_total,
        M_qdot_total=M_qdot_total,
    )


def compute_effective_inertia(
    *,
    m_dry: float,
    Iyy: float,
    totals: AddedMassTotals,
) -> EffectiveInertia:
    """Compute effective inertias used in denominators.

    IMPORTANT: per the governing equations in this project, the surge/heave
    denominators use m_dry (not m_wet).
    """

    m_ref = float(m_dry)

    m_x = m_ref - float(totals.X_udot_total)
    m_z = m_ref - float(totals.Z_wdot_total)
    I_y = float(Iyy) - float(totals.M_qdot_total)

    return EffectiveInertia(m_x=m_x, m_z=m_z, I_y=I_y)


def compute_from_param_tree(params: dict) -> tuple[AddedMassTotals, EffectiveInertia]:
    """Convenience helper for our nested YAML dict structure."""

    try:
        rb = params["rigid_body"]
        outer = params["added_mass_outer"]
        mu = params["permeability"]
    except Exception as exc:  # noqa: BLE001
        raise KeyError("params must contain rigid_body/added_mass_outer/permeability") from exc

    totals = compute_added_mass_totals(
        X_udot_outer=outer["X_udot_outer"],
        Z_wdot_outer=outer["Z_wdot_outer"],
        M_qdot_outer=outer["M_qdot_outer"],
        m_water_inner=rb["m_water_inner"],
        I_water_inner=rb["I_water_inner"],
        mu_x=mu["mu_x"],
        mu_z=mu["mu_z"],
        mu_theta=mu["mu_theta"],
    )
    eff = compute_effective_inertia(m_dry=rb["m_dry"], Iyy=rb["Iyy"], totals=totals)
    return totals, eff
