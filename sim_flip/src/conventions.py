"""Global sign/units conventions for sim_flip.

Frozen choices (do not change silently):
- Body axes: x_b forward, z_b down.
- Pitch angle theta: nose-up positive; q = dtheta/dt about +y_b.
- Velocities: u along +x_b, w along +z_b.
- Angle of attack for CFD lookup (deg): alpha_deg = atan2(w, u) * 180/pi.
- CFD coefficients are body-axis: Cx along +x_b, Cz along +z_b, Cm about +y_b.
- Cm > 0 is nose-up moment.

This module is intentionally tiny; it exists so every file can import the same frozen choices.
"""

from __future__ import annotations

import math

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0


def alpha_deg_from_uw(u: float, w: float) -> float:
    """Compute alpha in degrees using the project's frozen convention."""
    return math.degrees(math.atan2(w, u))
