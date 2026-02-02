from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_right
from typing import Iterable, List, Protocol, Sequence, Tuple, Union


Number = Union[int, float]


class _Interpolator1D(Protocol):
    def eval(self, xq: float) -> float: ...


@dataclass(frozen=True)
class CfdTable:
    alpha_deg: Tuple[float, ...]
    Cx: Tuple[float, ...]
    Cz: Tuple[float, ...]
    Cm: Tuple[float, ...]

    @property
    def alpha_min_deg(self) -> float:
        return self.alpha_deg[0]

    @property
    def alpha_max_deg(self) -> float:
        return self.alpha_deg[-1]


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


def _wrap_to_180_deg(alpha_deg: float) -> float:
    """Wrap any angle (deg) to [-180, 180)."""
    a = float(alpha_deg)
    if not _is_finite(a):
        raise ValueError("alpha_deg must be finite")
    return ((a + 180.0) % 360.0) - 180.0


def map_alpha_rule_b(alpha_raw_deg: float) -> tuple[float, float, float, float]:
    """Map raw alpha to lookup alpha and coefficient sign multipliers.

    Rule B (as confirmed):
    - Use B1 folding so lookup alpha stays in [0, 90].
    - If alpha < 0: first take abs(alpha) for lookup, but apply Cz and Cm sign flip.
      (Cx unchanged)
    - If (after abs) alpha > 90: fold via alpha <- 180 - alpha, and apply:
      Cx sign flip, Cm sign flip. (Cz unchanged)

    Returns: (alpha_lut_deg, sCx, sCz, sCm)
    where coefficients should be transformed as:
      Cx' = sCx * Cx(alpha_lut)
      Cz' = sCz * Cz(alpha_lut)
      Cm' = sCm * Cm(alpha_lut)
    """

    a = _wrap_to_180_deg(alpha_raw_deg)

    sCx = 1.0
    sCz = 1.0
    sCm = 1.0

    if a < 0.0:
        # alpha<0 rule: Cz and Cm flip, Cx unchanged.
        sCz *= -1.0
        sCm *= -1.0
        a = abs(a)

    # Now a in [0, 180].
    if a > 90.0:
        # alpha>90 rule: Cx flips, Cm flips, Cz unchanged; fold to [0,90].
        sCx *= -1.0
        sCm *= -1.0
        a = 180.0 - a

    # Numerical guard.
    if a < 0.0:
        a = 0.0
    if a > 90.0:
        a = 90.0

    return float(a), float(sCx), float(sCz), float(sCm)


def _sign(x: float) -> int:
    if x > 0.0:
        return 1
    if x < 0.0:
        return -1
    return 0


class _Linear1D:
    def __init__(self, x: Sequence[float], y: Sequence[float]):
        if len(x) != len(y):
            raise ValueError("x and y length mismatch")
        if len(x) < 2:
            raise ValueError("Need at least 2 points for interpolation")
        self._x = tuple(float(v) for v in x)
        self._y = tuple(float(v) for v in y)
        self._n = len(self._x)

    def eval(self, xq: float) -> float:
        x0 = self._x[0]
        xn = self._x[-1]
        if xq <= x0:
            return self._y[0]
        if xq >= xn:
            return self._y[-1]

        i = bisect_right(self._x, xq) - 1
        if i < 0:
            i = 0
        if i > self._n - 2:
            i = self._n - 2

        xL = self._x[i]
        xR = self._x[i + 1]
        yL = self._y[i]
        yR = self._y[i + 1]
        t = (xq - xL) / (xR - xL)
        return (1.0 - t) * yL + t * yR


class _Pchip1D:
    """Shape-preserving (monotone) piecewise cubic Hermite interpolant.

    Implements Fritsch-Carlson slope limiting, and clamps outside range
    (no extrapolation).
    """

    def __init__(self, x: Sequence[float], y: Sequence[float]):
        if len(x) != len(y):
            raise ValueError("x and y length mismatch")
        if len(x) < 2:
            raise ValueError("Need at least 2 points for interpolation")

        x_arr = [float(v) for v in x]
        y_arr = [float(v) for v in y]

        for i in range(len(x_arr) - 1):
            if not (x_arr[i + 1] > x_arr[i]):
                raise ValueError("x must be strictly increasing")

        self._x = tuple(x_arr)
        self._y = tuple(y_arr)
        self._n = len(self._x)
        self._d = self._compute_slopes()

    def _compute_slopes(self) -> Tuple[float, ...]:
        n = self._n
        x = self._x
        y = self._y

        if n == 2:
            delta0 = (y[1] - y[0]) / (x[1] - x[0])
            return (delta0, delta0)

        h: List[float] = []
        delta: List[float] = []
        for i in range(n - 1):
            hi = x[i + 1] - x[i]
            h.append(hi)
            delta.append((y[i + 1] - y[i]) / hi)

        d = [0.0] * n

        # Interior slopes: weighted harmonic mean when deltas have same sign.
        for i in range(1, n - 1):
            if delta[i - 1] == 0.0 or delta[i] == 0.0:
                d[i] = 0.0
            elif delta[i - 1] * delta[i] < 0.0:
                d[i] = 0.0
            else:
                w1 = 2.0 * h[i] + h[i - 1]
                w2 = h[i] + 2.0 * h[i - 1]
                d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

        # Endpoint slopes.
        d0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
        if _sign(d0) != _sign(delta[0]):
            d0 = 0.0
        elif _sign(delta[0]) != _sign(delta[1]) and abs(d0) > abs(3.0 * delta[0]):
            d0 = 3.0 * delta[0]
        d[0] = d0

        dn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
        if _sign(dn) != _sign(delta[-1]):
            dn = 0.0
        elif _sign(delta[-1]) != _sign(delta[-2]) and abs(dn) > abs(3.0 * delta[-1]):
            dn = 3.0 * delta[-1]
        d[-1] = dn

        return tuple(float(v) for v in d)

    def eval(self, xq: float) -> float:
        x0 = self._x[0]
        xn = self._x[-1]
        if xq <= x0:
            return self._y[0]
        if xq >= xn:
            return self._y[-1]

        i = bisect_right(self._x, xq) - 1
        if i < 0:
            i = 0
        if i > self._n - 2:
            i = self._n - 2

        xL = self._x[i]
        xR = self._x[i + 1]
        yL = self._y[i]
        yR = self._y[i + 1]
        dL = self._d[i]
        dR = self._d[i + 1]
        h = xR - xL
        t = (xq - xL) / h

        t2 = t * t
        t3 = t2 * t

        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2

        return h00 * yL + h10 * h * dL + h01 * yR + h11 * h * dR


@dataclass(frozen=True)
class CfdInterpolator:
    table: CfdTable
    method: str
    _Cx: _Interpolator1D
    _Cz: _Interpolator1D
    _Cm: _Interpolator1D

    @staticmethod
    def from_csv(csv_path: Union[str, Path], method: str = "pchip") -> "CfdInterpolator":
        table = load_cfd_table(csv_path)
        return CfdInterpolator.from_table(table, method=method)

    @staticmethod
    def from_table(table: CfdTable, method: str = "pchip") -> "CfdInterpolator":
        method_norm = method.strip().lower()
        if method_norm not in {"pchip", "linear"}:
            raise ValueError("method must be 'pchip' or 'linear'")

        if method_norm == "linear":
            ctor = _Linear1D
        else:
            ctor = _Pchip1D

        return CfdInterpolator(
            table=table,
            method=method_norm,
            _Cx=ctor(table.alpha_deg, table.Cx),
            _Cz=ctor(table.alpha_deg, table.Cz),
            _Cm=ctor(table.alpha_deg, table.Cm),
        )

    @property
    def alpha_min_deg(self) -> float:
        return self.table.alpha_min_deg

    @property
    def alpha_max_deg(self) -> float:
        return self.table.alpha_max_deg

    def coeffs(self, alpha_deg: float) -> Tuple[float, float, float]:
        """Return (Cx, Cz, Cm) at alpha_deg, clamped to table range."""
        a = float(alpha_deg)
        if not _is_finite(a):
            raise ValueError("alpha_deg must be finite")
        a = min(max(a, self.alpha_min_deg), self.alpha_max_deg)
        return (
            float(self._Cx.eval(a)),
            float(self._Cz.eval(a)),
            float(self._Cm.eval(a)),
        )

    def coeffs_extended(self, alpha_raw_deg: float) -> Tuple[float, float, float]:
        """Return (Cx, Cz, Cm) using Rule-B alpha handling.

        The raw alpha is first mapped to a lookup alpha within [0, 90] (then clamped
        to the table's alpha range), and the coefficients are sign-adjusted per Rule B.
        """

        a_lut, sCx, sCz, sCm = map_alpha_rule_b(alpha_raw_deg)
        # Clamp to actual table range (in case table is not exactly 0..90).
        a_lut = min(max(a_lut, self.alpha_min_deg), self.alpha_max_deg)
        cx, cz, cm = self.coeffs(a_lut)
        return (sCx * cx, sCz * cz, sCm * cm)

    def coeffs_many(self, alpha_deg: Iterable[Number]) -> List[Tuple[float, float, float]]:
        return [self.coeffs(float(a)) for a in alpha_deg]


def default_cfd_table_path() -> Path:
    # sim_flip/src/cfd_table.py -> sim_flip/ -> data/
    pkg_root = Path(__file__).resolve().parents[1]
    return pkg_root / "data" / "cfd_table_clean.csv"


def load_cfd_table(csv_path: Union[str, Path]) -> CfdTable:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    alpha: List[float] = []
    cx: List[float] = []
    cz: List[float] = []
    cm: List[float] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"alpha_deg", "Cx", "Cz", "Cm"}
        if reader.fieldnames is None:
            raise ValueError("CSV missing header")
        missing = required.difference(set(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            a = float(row["alpha_deg"])
            x = float(row["Cx"])
            z = float(row["Cz"])
            m = float(row["Cm"])
            if not (_is_finite(a) and _is_finite(x) and _is_finite(z) and _is_finite(m)):
                raise ValueError("CSV contains non-finite values")
            alpha.append(a)
            cx.append(x)
            cz.append(z)
            cm.append(m)

    if len(alpha) < 2:
        raise ValueError("Need at least 2 CFD points")

    # Validate strictly increasing alpha.
    for i in range(len(alpha) - 1):
        if not (alpha[i + 1] > alpha[i]):
            raise ValueError("alpha_deg must be strictly increasing")

    return CfdTable(
        alpha_deg=tuple(alpha),
        Cx=tuple(cx),
        Cz=tuple(cz),
        Cm=tuple(cm),
    )


def load_default_cfd_interpolator(method: str = "pchip") -> CfdInterpolator:
    return CfdInterpolator.from_csv(default_cfd_table_path(), method=method)
