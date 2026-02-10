from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, savgol_filter, welch

from .config import (
    default_derived_run_dir,
    default_protocol_path,
    ensure_pipeline_dirs,
    load_yaml,
)

RUN_ID_PATTERN = re.compile(r"^R\d{8}_\d{2}$")


@dataclass(frozen=True)
class PreprocessOutput:
    run_id: str
    csv_path: Path
    qc_path: Path
    df: pd.DataFrame
    qc: dict[str, Any]


def _validate_run_id(run_id: str) -> str:
    rid = str(run_id).strip()
    if not RUN_ID_PATTERN.fullmatch(rid):
        raise ValueError(
            f"run_id must match RYYYYMMDD_## pattern, got: {run_id!r}"
        )
    return rid


def run_id_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    return _validate_run_id(stem)


def _load_raw_txt(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    df = pd.read_csv(p, sep=r"\s+", engine="python")
    required = ["angleX", "angleY", "gyroX", "gyroY", "time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"raw txt missing columns {missing}; got={df.columns.tolist()}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().copy()
    if df.empty:
        raise ValueError("raw txt became empty after numeric cleaning")
    return df


def _principal_axis_from_angle_xy(angle_xy: np.ndarray) -> np.ndarray:
    """SVD-based PCA first component with deterministic sign convention.

    SVD/PCA has an inherent sign ambiguity: ``vt[0]`` and ``-vt[0]`` are
    equally valid.  Without a sign convention the *same* physical motion
    can produce opposite ``theta_lab_deg`` projections across different
    data files, which in turn flips whether the first oscillation
    extremum appears as a valley (< 90 deg) or a peak (> 90 deg).

    Convention chosen here (deterministic):
      Project *all* samples onto the candidate axis, then find the
      sample with the **largest absolute projection** -- this is the
      excitation peak, far from equilibrium, with the highest SNR.
      If that projection is negative, flip the axis so the excitation
      peak always maps to the *positive* side.

      After the downstream mapping ``theta_deg = 90 + theta_lab_deg``,
      this guarantees that the excitation peak satisfies ``theta > 90``
      regardless of which physical direction the initial push was in.

    Why *not* use the first sample:
      In a typical free-decay experiment the recording starts *before*
      the excitation is applied.  The first data points sit near the
      equilibrium (projection ~ 0) where noise dominates, making the
      sign effectively random -- exactly the failure mode we need to
      prevent.

    Downstream consequence:
      Because excitation always maps to theta > 90 deg (peak), the
      segmentation step (``segment_lock._detect_start_indices``) only
      needs to call ``find_peaks(theta_deg)`` to locate the free-decay
      start point.
    """
    centered = angle_xy - np.mean(angle_xy, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    norm = np.linalg.norm(axis)
    if norm <= 0.0:
        raise ValueError("PCA axis norm is zero")
    axis = axis / norm

    # --- Deterministic sign: max-deviation sample projects to positive ---
    proj = centered @ axis
    idx_max_dev = int(np.argmax(np.abs(proj)))
    if proj[idx_max_dev] < 0.0:
        axis = -axis

    return axis


def _project_to_principal_axis(df_raw: pd.DataFrame) -> pd.DataFrame:
    angle_xy = df_raw[["angleX", "angleY"]].to_numpy(dtype=float)
    gyro_xy = df_raw[["gyroX", "gyroY"]].to_numpy(dtype=float)
    axis = _principal_axis_from_angle_xy(angle_xy)

    angle_proj_deg = (angle_xy - np.mean(angle_xy, axis=0)) @ axis
    gyro_proj_deg_s = gyro_xy @ axis

    t_ticks = df_raw["time"].to_numpy(dtype=float)
    t_s = (t_ticks - t_ticks[0]) / 1000.0
    if np.any(np.diff(t_s) <= 0):
        raise ValueError("time is not strictly increasing after normalization")

    out = pd.DataFrame(
        {
            "t_raw_s": t_s,
            "theta_lab_deg": angle_proj_deg,
            "q_lab_deg_s": gyro_proj_deg_s,
        }
    )
    # Freeze lab -> physics mapping from notebook.
    out["theta_deg"] = 90.0 + out["theta_lab_deg"]
    out["q_deg_s"] = out["q_lab_deg_s"]
    out["theta_rad"] = np.deg2rad(out["theta_deg"])
    out["q_rad_s"] = np.deg2rad(out["q_deg_s"])
    return out


def _resample_and_filter(df: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    pre = protocol["preprocess"]
    dt = float(pre["resample"]["dt_s"])
    if dt <= 0.0:
        raise ValueError("resample.dt_s must be > 0")

    t_raw = df["t_raw_s"].to_numpy(dtype=float)
    t_s = np.arange(t_raw[0], t_raw[-1], dt, dtype=float)
    if len(t_s) < 10:
        raise ValueError("resampled length too short")

    interp_kind = str(pre["resample"].get("kind", "quadratic"))
    theta_interp = interp1d(
        t_raw,
        df["theta_deg"].to_numpy(dtype=float),
        kind=interp_kind,
        fill_value="extrapolate",
    )(t_s)
    q_interp = interp1d(
        t_raw,
        df["q_deg_s"].to_numpy(dtype=float),
        kind=interp_kind,
        fill_value="extrapolate",
    )(t_s)

    sg = pre["savgol"]
    window = int(sg["window"])
    poly = int(sg["polyorder"])
    if window % 2 == 0:
        window += 1
    if window <= poly:
        window = poly + 3 if (poly + 3) % 2 == 1 else poly + 4

    theta_sg = savgol_filter(theta_interp, window, poly)
    q_sg = savgol_filter(q_interp, window, poly)

    fs = 1.0 / dt
    f, pxx = welch(q_sg, fs=fs, nperseg=min(int(pre["welch"]["nperseg"]), len(q_sg)))
    dom_low = float(pre["butter"]["dom_freq_min_hz"])
    dom_high = float(pre["butter"]["dom_freq_max_hz"])
    mask = (f >= dom_low) & (f <= dom_high)
    if np.any(mask):
        dom_freq = float(f[mask][np.argmax(pxx[mask])])
    else:
        dom_freq = float(pre["butter"]["fallback_dom_freq_hz"])

    mult = float(pre["butter"]["cutoff_multiplier"])
    cutoff = dom_freq * mult
    cutoff = max(float(pre["butter"]["cutoff_min_hz"]), cutoff)
    cutoff = min(float(pre["butter"]["cutoff_max_hz"]), cutoff)
    nyq = 0.5 * fs
    wn = min(0.99, cutoff / nyq)

    order = int(pre["butter"]["order"])
    b, a = butter(order, wn, btype="low")
    theta_f = filtfilt(b, a, theta_sg)
    q_f = filtfilt(b, a, q_sg)

    q_dot_deg_s2 = np.gradient(q_f, dt)

    out = pd.DataFrame(
        {
            "t_s": t_s,
            "theta_deg": theta_f,
            "q_deg_s": q_f,
            "q_dot_deg_s2": q_dot_deg_s2,
        }
    )
    out["theta_rad"] = np.deg2rad(out["theta_deg"])
    out["q_rad_s"] = np.deg2rad(out["q_deg_s"])
    out["q_dot_rad_s2"] = np.deg2rad(out["q_dot_deg_s2"])
    out["fs_hz"] = fs
    out.attrs["adaptive_dom_freq_hz"] = dom_freq
    out.attrs["adaptive_cutoff_hz"] = cutoff
    return out


def _qc_report(df_raw: pd.DataFrame, df_proc: pd.DataFrame, run_id: str) -> dict[str, Any]:
    dt = np.diff(df_proc["t_s"].to_numpy(dtype=float))
    q = {
        "run_id": run_id,
        "n_raw": int(len(df_raw)),
        "n_processed": int(len(df_proc)),
        "time_monotonic_raw": bool(np.all(np.diff(df_raw["t_raw_s"].to_numpy(dtype=float)) > 0)),
        "time_monotonic_processed": bool(np.all(dt > 0)),
        "dt_mean_s": float(np.mean(dt)) if len(dt) else float("nan"),
        "dt_std_s": float(np.std(dt)) if len(dt) else float("nan"),
        "theta_range_deg": [
            float(df_proc["theta_deg"].min()),
            float(df_proc["theta_deg"].max()),
        ],
        "q_range_rad_s": [
            float(df_proc["q_rad_s"].min()),
            float(df_proc["q_rad_s"].max()),
        ],
        "q_dot_range_rad_s2": [
            float(df_proc["q_dot_rad_s2"].min()),
            float(df_proc["q_dot_rad_s2"].max()),
        ],
        "has_nan": bool(df_proc.isna().any().any()),
    }
    return q


def preprocess_run_to_csv(
    *,
    raw_txt_path: str | Path,
    run_id: str | None = None,
    protocol_path: str | Path | None = None,
    out_csv_path: str | Path | None = None,
    out_qc_path: str | Path | None = None,
) -> PreprocessOutput:
    ensure_pipeline_dirs()
    protocol = load_yaml(protocol_path or default_protocol_path())

    rid = _validate_run_id(run_id) if run_id is not None else run_id_from_path(raw_txt_path)
    raw = _load_raw_txt(raw_txt_path)
    projected = _project_to_principal_axis(raw)
    processed = _resample_and_filter(projected, protocol=protocol)
    processed.insert(0, "run_id", rid)
    processed = processed[
        [
            "run_id",
            "t_s",
            "theta_rad",
            "theta_deg",
            "q_rad_s",
            "q_deg_s",
            "q_dot_rad_s2",
            "q_dot_deg_s2",
            "fs_hz",
        ]
    ].copy()

    qc = _qc_report(projected, processed, rid)
    qc["adaptive_dom_freq_hz"] = float(processed.attrs["adaptive_dom_freq_hz"])
    qc["adaptive_cutoff_hz"] = float(processed.attrs["adaptive_cutoff_hz"])

    csv_path = Path(out_csv_path) if out_csv_path else default_derived_run_dir() / f"{rid}.csv"
    qc_path = Path(out_qc_path) if out_qc_path else default_derived_run_dir() / f"{rid}_qc.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    processed.to_csv(csv_path, index=False)
    with qc_path.open("w", encoding="utf-8") as f:
        json.dump(qc, f, ensure_ascii=False, indent=2)

    return PreprocessOutput(run_id=rid, csv_path=csv_path, qc_path=qc_path, df=processed, qc=qc)

