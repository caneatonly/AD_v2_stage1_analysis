from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import sim_flip_root


ANGLE_FILE_RE = re.compile(r"^C_(\d+)_overmesh\.csv$")


@dataclass(frozen=True)
class CfdPipelineOutput:
    cfd_table_path: Path
    report_path: Path
    figure_paths: list[Path]


def _steady_mean_last_n(df: pd.DataFrame, cols: list[str], n_last: int) -> dict[str, float]:
    n = min(len(df), int(n_last))
    if n <= 0:
        raise ValueError("empty dataframe for steady extraction")
    tail = df.iloc[-n:]
    return {c: float(pd.to_numeric(tail[c], errors="coerce").mean()) for c in cols}


def _load_angle_files(data_dir: Path, n_last: int) -> pd.DataFrame:
    rows = []
    for path in sorted(data_dir.glob("C_*_overmesh.csv")):
        m = ANGLE_FILE_RE.match(path.name)
        if not m:
            continue
        angle = int(m.group(1))
        df = pd.read_csv(path)
        required = {"Cx", "Cz", "Cm"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} missing columns {sorted(required - set(df.columns))}")
        means = _steady_mean_last_n(df, ["Cx", "Cz", "Cm"], n_last=n_last)
        rows.append({"alpha_deg": angle, **means, "n_samples_used": min(len(df), n_last)})
    if not rows:
        raise RuntimeError(f"No C_*_overmesh.csv files found in: {data_dir}")
    out = pd.DataFrame(rows).sort_values("alpha_deg").reset_index(drop=True)
    return out


def _validate_full_0_180(table: pd.DataFrame, angle_step_deg: int) -> list[int]:
    expected = list(range(0, 181, int(angle_step_deg)))
    got = table["alpha_deg"].astype(int).tolist()
    missing = sorted(set(expected) - set(got))
    return missing


def _trend_validation(table: pd.DataFrame) -> dict[str, Any]:
    # Lightweight trend diagnostics; hard fail stays in full-range check.
    out: dict[str, Any] = {}
    for col in ["Cx", "Cz", "Cm"]:
        y = table[col].to_numpy(dtype=float)
        dy = np.diff(y)
        out[f"{col}_diff_sign_changes"] = int(np.sum(np.sign(dy[1:]) != np.sign(dy[:-1])))
        out[f"{col}_min"] = float(np.min(y))
        out[f"{col}_max"] = float(np.max(y))
    return out


def _magnitude_check(table: pd.DataFrame, A_ref: float, L_ref: float) -> dict[str, float]:
    # Crossflow drag conversion at 90 deg.
    rho = 997.561
    _ = rho  # Keep signature stable if later extended.
    D = float(np.sqrt(4 * A_ref / np.pi))
    conv = float(np.pi * D / (4.0 * L_ref))
    cx0 = float(table.loc[table["alpha_deg"] == 0, "Cx"].iloc[0])
    cz90 = float(table.loc[table["alpha_deg"] == 90, "Cz"].iloc[0])
    cd_cross = abs(cz90) * conv
    return {
        "Cx_at_0deg": cx0,
        "Cz_at_90deg": cz90,
        "Cd_cross_at_90deg": float(cd_cross),
    }


def _compute_cp(table: pd.DataFrame, L_ref: float, cz_eps: float) -> pd.DataFrame:
    cp = table.copy()
    cp["Xcp_m"] = np.nan
    mask = np.abs(cp["Cz"]) > float(cz_eps)
    cp.loc[mask, "Xcp_m"] = -(cp.loc[mask, "Cm"] / cp.loc[mask, "Cz"]) * float(L_ref)
    return cp


def _plot_coeffs(table: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        out_path.write_text("matplotlib unavailable; coeff plot skipped\n", encoding="utf-8")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(table["alpha_deg"], table["Cx"], "o-", label="Cx")
    ax.plot(table["alpha_deg"], table["Cz"], "s-", label="Cz")
    ax.plot(table["alpha_deg"], table["Cm"], "^-", label="Cm")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("Angle of Attack alpha (deg)")
    ax.set_ylabel("Coefficient")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_cp(cp_df: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        out_path.write_text("matplotlib unavailable; xcp plot skipped\n", encoding="utf-8")
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(cp_df["alpha_deg"], cp_df["Xcp_m"], "d-", color="#6a3d9a", label="Xcp")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel("Angle of Attack alpha (deg)")
    ax.set_ylabel("Xcp (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_cfd_review_pipeline(
    *,
    source_dir: str | Path,
    out_table_path: str | Path | None = None,
    out_report_path: str | Path | None = None,
    out_figure_dir: str | Path | None = None,
    n_steady_last: int = 500,
    angle_step_deg: int = 5,
    A_ref: float = 0.0056745,
    L_ref: float = 0.625,
    cp_cz_eps: float = 0.01,
) -> CfdPipelineOutput:
    src = Path(source_dir)
    if not src.exists():
        raise FileNotFoundError(str(src))

    root = sim_flip_root()
    table_path = Path(out_table_path) if out_table_path else root / "data" / "cfd_table_clean.csv"
    report_path = Path(out_report_path) if out_report_path else root / "results" / "cfd_review_report.json"
    fig_dir = Path(out_figure_dir) if out_figure_dir else root / "results" / "cfd_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    table = _load_angle_files(src, n_last=n_steady_last)
    missing = _validate_full_0_180(table, angle_step_deg=angle_step_deg)
    if missing:
        raise RuntimeError(
            "CFD full-angle contract violated: missing alpha_deg values "
            f"{missing}. Runtime table requires strict 0..180 coverage."
        )

    table = table[["alpha_deg", "Cx", "Cz", "Cm"]].copy()
    table.to_csv(table_path, index=False)

    cp_df = _compute_cp(table, L_ref=L_ref, cz_eps=cp_cz_eps)
    coeff_fig = fig_dir / "cfd_coeffs_0_180.png"
    cp_fig = fig_dir / "cfd_xcp_0_180.png"
    _plot_coeffs(table, coeff_fig)
    _plot_cp(cp_df, cp_fig)

    report = {
        "source_dir": str(src),
        "table_path": str(table_path),
        "n_angles": int(len(table)),
        "angle_step_deg": int(angle_step_deg),
        "trend_validation": _trend_validation(table),
        "magnitude_check": _magnitude_check(table, A_ref=A_ref, L_ref=L_ref),
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return CfdPipelineOutput(
        cfd_table_path=table_path,
        report_path=report_path,
        figure_paths=[coeff_fig, cp_fig],
    )
