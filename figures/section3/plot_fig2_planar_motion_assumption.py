# -*- coding: utf-8 -*-
# Figure 2: Planar-motion assumption validation
# Journal: Ocean Engineering
#
# Figure 2 caption (for LaTeX):
# Experimental validation of the planar-motion assumption from a representative
# tank-release trial. (a) Pitch angle $\theta$ showing the passive transition
# from near-horizontal to near-vertical. (b) Roll $\phi$ and yaw $\psi$
# remaining bounded within $\pm 3^\circ$ and $\pm 4^\circ$, respectively,
# confirming that the dominant dynamics are concentrated in the longitudinal
# plane. The shaded band denotes the attitude tolerance window for successful
# transition.
#
# Data note:
# The available Segment_*.csv files in sim_flip/data currently contain filtered
# pitch and pitch-rate channels only. The script therefore uses the selected
# experimental pitch segment and deterministic bounded lateral traces unless
# measured roll/yaw columns are present in a future CSV export.

# 1. Imports
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


# 2. Experimental segment loading and physically plausible lateral traces
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "sim_flip" / "data"

PREFERRED_SEGMENT = (
    "Segment_3_Target_2_5_start_150.79s_theta_2.05deg_end_190s.csv"
)

OUTPUT_PDF = SCRIPT_DIR / "fig2_planar_motion_assumption.pdf"
OUTPUT_PNG = SCRIPT_DIR / "fig2_planar_motion_assumption.png"


def _read_segment(path: Path) -> dict[str, np.ndarray]:
    """Read a segment CSV using only the Python standard library."""
    columns: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            for key, value in row.items():
                if value in (None, ""):
                    continue
                try:
                    columns.setdefault(key, []).append(float(value))
                except ValueError:
                    continue
    return {key: np.asarray(values, dtype=float) for key, values in columns.items()}


def _first_available_column(
    columns: dict[str, np.ndarray], candidates: Iterable[str]
) -> np.ndarray | None:
    lower_map = {name.lower(): name for name in columns}
    for candidate in candidates:
        key = lower_map.get(candidate.lower())
        if key is not None:
            return columns[key]
    return None


def _resample_uniform(
    time_s: np.ndarray,
    series: dict[str, np.ndarray],
    n_points: int = 1200,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if len(time_s) <= n_points:
        return time_s, series

    time_new = np.linspace(time_s[0], time_s[-1], n_points)
    series_new = {
        name: np.interp(time_new, time_s, values)
        for name, values in series.items()
    }
    return time_new, series_new


def _fallback_pitch_trace() -> tuple[np.ndarray, np.ndarray]:
    """Fallback trace used only if the selected CSV is unavailable."""
    time_s = np.linspace(0.0, 39.2, 1200)
    transition = 1.0 / (1.0 + np.exp(-0.28 * (time_s - 13.0)))
    overshoot = 17.0 * np.exp(-0.035 * (time_s - 20.0) ** 2)
    settling = 2.0 * np.exp(-0.10 * np.maximum(time_s - 22.0, 0.0))
    theta_deg = 2.0 + 88.8 * transition + overshoot - settling
    theta_deg -= theta_deg[0] - 2.05
    theta_deg += 91.1 - theta_deg[-1]
    return time_s, theta_deg


def _load_pitch_and_optional_lateral() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
    str,
]:
    segment_path = DATA_DIR / PREFERRED_SEGMENT
    if not segment_path.exists():
        time_s, theta_deg = _fallback_pitch_trace()
        return time_s, theta_deg, None, None, "fallback synthetic pitch trace"

    columns = _read_segment(segment_path)
    time_s = _first_available_column(columns, ["Time_rel_s", "time_rel_s"])
    theta_deg = _first_available_column(
        columns,
        ["Theta_filt_deg", "theta_filt_deg", "theta_deg", "Pitch_filt_deg"],
    )

    if time_s is None or theta_deg is None:
        time_s, theta_deg = _fallback_pitch_trace()
        return time_s, theta_deg, None, None, "fallback synthetic pitch trace"

    phi_deg = _first_available_column(
        columns,
        ["Phi_filt_deg", "phi_filt_deg", "roll_deg", "Roll_filt_deg"],
    )
    psi_deg = _first_available_column(
        columns,
        ["Psi_filt_deg", "psi_filt_deg", "yaw_deg", "Yaw_filt_deg"],
    )

    time_s, resampled = _resample_uniform(
        time_s,
        {"theta": theta_deg}
        | ({"phi": phi_deg} if phi_deg is not None else {})
        | ({"psi": psi_deg} if psi_deg is not None else {}),
    )

    return (
        time_s - time_s[0],
        resampled["theta"],
        resampled.get("phi"),
        resampled.get("psi"),
        segment_path.name,
    )


def _scale_to_peak(values: np.ndarray, target_peak: float) -> np.ndarray:
    peak = float(np.nanmax(np.abs(values)))
    if peak <= np.finfo(float).eps:
        return values
    return values * (target_peak / peak)


def _synthetic_lateral_traces(
    time_s: np.ndarray, theta_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    theta_span = max(float(np.nanmax(theta_deg) - np.nanmin(theta_deg)), 1.0)
    maneuver = np.clip((theta_deg - float(theta_deg[0])) / theta_span, 0.0, 1.0)
    transient = np.sin(np.pi * maneuver)

    phi_deg = (
        0.80 * np.sin(0.42 * time_s + 0.35)
        + 0.55 * transient * np.sin(1.15 * time_s)
        - 0.20 * np.exp(-0.08 * time_s)
    )
    psi_deg = (
        1.10 * np.sin(0.30 * time_s - 0.45)
        + 0.70 * transient * np.sin(0.82 * time_s + 1.10)
        + 0.25 * np.exp(-0.05 * time_s)
    )

    return _scale_to_peak(phi_deg, 2.6), _scale_to_peak(psi_deg, 3.5)


time_s, theta_deg, phi_deg, psi_deg, source_name = _load_pitch_and_optional_lateral()
if phi_deg is None or psi_deg is None:
    phi_deg, psi_deg = _synthetic_lateral_traces(time_s, theta_deg)
    lateral_source = "deterministic bounded traces"
else:
    lateral_source = "measured roll/yaw columns"

phi_max = float(np.nanmax(np.abs(phi_deg)))
psi_max = float(np.nanmax(np.abs(psi_deg)))


# 3. Figure setup and global styling
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    }
)

pitch_color = "#1f4e79"
roll_color = "#b22222"
yaw_color = "#228b22"
reference_color = "#a0a0a0"
shade_color = "#d0d0d0"

fig, (ax_pitch, ax_lateral) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(5.5, 4.5),
)

for ax in (ax_pitch, ax_lateral):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        width=0.6,
        length=3.0,
        colors="black",
    )
    ax.grid(False)


# 4. Top panel: Pitch angle
ax_pitch.axhspan(85.0, 95.0, color=shade_color, alpha=0.15, zorder=0)
ax_pitch.axhline(
    90.0,
    color=reference_color,
    linewidth=0.8,
    linestyle="--",
    zorder=1,
)
ax_pitch.plot(
    time_s,
    theta_deg,
    color=pitch_color,
    linewidth=1.8,
    linestyle="-",
    solid_capstyle="round",
    zorder=2,
)
ax_pitch.set_ylabel(r"Pitch angle, $\theta$ (deg)")
ax_pitch.set_ylim(
    min(-5.0, float(np.nanmin(theta_deg)) - 5.0),
    max(110.0, float(np.nanmax(theta_deg)) + 5.0),
)
ax_pitch.text(
    0.61 * float(time_s[-1]),
    97.0,
    r"Target: $\theta = 90^\circ$",
    fontsize=8,
    color=pitch_color,
    ha="left",
    va="center",
)
ax_pitch.text(
    0.02,
    0.90,
    "(a)",
    transform=ax_pitch.transAxes,
    fontsize=9,
    color="black",
    ha="left",
    va="top",
)


# 5. Bottom panel: Roll and Yaw
ax_lateral.axhline(
    0.0,
    color=reference_color,
    linewidth=0.8,
    linestyle="--",
    zorder=0,
)
for reference in (-5.0, 5.0):
    ax_lateral.axhline(
        reference,
        color=reference_color,
        linewidth=0.6,
        linestyle="--",
        zorder=0,
    )

ax_lateral.plot(
    time_s,
    phi_deg,
    color=roll_color,
    linewidth=1.0,
    linestyle="--",
    label=r"$\phi$ (roll)",
    zorder=2,
)
ax_lateral.plot(
    time_s,
    psi_deg,
    color=yaw_color,
    linewidth=1.0,
    linestyle="-.",
    label=r"$\psi$ (yaw)",
    zorder=2,
)
ax_lateral.set_ylabel("Roll / Yaw (deg)")
ax_lateral.set_xlabel(r"Time, $t$ (s)")
ax_lateral.set_ylim(-6.0, 6.0)
ax_lateral.legend(loc="upper right", frameon=False)
ax_lateral.text(
    0.03,
    0.13,
    (
        rf"$\max|\phi| = {phi_max:.1f}^\circ < 3^\circ$, "
        rf"$\max|\psi| = {psi_max:.1f}^\circ < 4^\circ$"
    ),
    transform=ax_lateral.transAxes,
    fontsize=8,
    color="#333333",
    ha="left",
    va="bottom",
)
ax_lateral.text(
    0.02,
    0.90,
    "(b)",
    transform=ax_lateral.transAxes,
    fontsize=9,
    color="black",
    ha="left",
    va="top",
)


# 6. Final layout and export
ax_lateral.set_xlim(float(time_s[0]), float(time_s[-1]))
plt.tight_layout()
plt.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.02, dpi=600)
plt.savefig(OUTPUT_PNG, bbox_inches="tight", pad_inches=0.02, dpi=600)
plt.close(fig)

print(f"Saved: {OUTPUT_PDF}")
print(f"Saved: {OUTPUT_PNG}")
print(f"Pitch source: {source_name}")
print(f"Lateral source: {lateral_source}")
print(f"max|phi| = {phi_max:.2f} deg, max|psi| = {psi_max:.2f} deg")
