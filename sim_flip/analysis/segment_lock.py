from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import (
    default_derived_segment_dir,
    default_manifest_path,
    default_protocol_path,
    ensure_pipeline_dirs,
    load_yaml,
)

MANIFEST_COLUMNS = [
    "run_id",
    "segment_id",
    "repeat_id",
    "theta0_meas_deg",
    "q0_meas_rad_s",
    "t_start_s",
    "t_end_s",
    "segmentation_mode",
    "split_tag",
    "cv_fold",
    "notes",
]


@dataclass(frozen=True)
class SegmentExport:
    run_id: str
    segment_paths: list[Path]
    metadata_json_path: Path
    manifest_path: Path


def _load_manifest(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    df = pd.read_csv(p)
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    out = df[MANIFEST_COLUMNS].copy()
    str_cols = ["run_id", "segment_id", "segmentation_mode", "split_tag", "cv_fold", "notes"]
    for col in str_cols:
        out[col] = out[col].fillna("").astype(str)
    num_cols = ["repeat_id", "theta0_meas_deg", "q0_meas_rad_s", "t_start_s", "t_end_s"]
    for col in num_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _save_manifest(path: str | Path, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[MANIFEST_COLUMNS]
    df.to_csv(p, index=False)


def _detect_start_indices(theta_deg: np.ndarray, t_s: np.ndarray, protocol: dict[str, Any]) -> np.ndarray:
    """Detect free-decay start points as peaks (theta > 90 deg).

    Coordinate contract (enforced by ``raw_preprocess._principal_axis_from_angle_xy``):
      The PCA sign convention guarantees that the excitation extremum
      (the point of maximum deviation from equilibrium) is always
      mapped to ``theta_lab > 0``, i.e. ``theta_deg > 90``.
      Therefore the free-decay start is always a **peak** of theta_deg.

    Why only peaks, not valleys:
      - Valleys (theta < 90) are intermediate oscillation troughs
        *during* the decay, not independent excitation start points.
      - Detecting valleys would create overlapping segments from
        the same decay process, inflating pseudo-sample count in
        Step 3 energy identification.

    The ``prominence`` threshold (``segmentation.valley_prominence_deg``
    in protocol YAML -- name kept for backward compatibility) gates
    how far above the local baseline a peak must rise to qualify.
    """
    seg_cfg = protocol["segmentation"]
    fs = 1.0 / float(protocol["preprocess"]["resample"]["dt_s"])
    min_dist = int(max(1, float(seg_cfg["min_peak_distance_s"]) * fs))
    prominence = float(seg_cfg["valley_prominence_deg"])

    # Detect peaks only (theta > 90 side = excitation start points)
    peaks, _ = find_peaks(theta_deg, distance=min_dist, prominence=prominence)
    if len(peaks) == 0:
        raise RuntimeError(
            "No peaks detected for lock-phase segmentation.  "
            "Check raw signal quality, PCA sign convention, "
            "and segmentation.valley_prominence_deg."
        )

    starts = []
    min_t = float(seg_cfg["start_time_min_s"])
    for idx in peaks:
        if float(t_s[idx]) >= min_t:
            starts.append(int(idx))
    if not starts:
        raise RuntimeError("Detected peaks exist but all before start_time_min_s gate")
    return np.asarray(starts, dtype=int)


def _stable_end_index(
    *,
    theta_deg: np.ndarray,
    q_rad_s: np.ndarray,
    t_s: np.ndarray,
    start_idx: int,
    protocol: dict[str, Any],
) -> int:
    seg_cfg = protocol["segmentation"]
    dt = float(protocol["preprocess"]["resample"]["dt_s"])
    min_len = int(max(1, float(seg_cfg["min_segment_length_s"]) / dt))
    stable_win = int(max(3, float(seg_cfg["stable_window_s"]) / dt))
    q_tol = float(seg_cfg["stable_q_abs_max_rad_s"])
    theta_std_tol = float(seg_cfg["stable_theta_std_max_deg"])

    n = len(theta_deg)
    i0 = min(n - 1, start_idx + min_len)
    if i0 >= n - stable_win:
        return n - 1

    for i in range(i0, n - stable_win):
        win_theta = theta_deg[i : i + stable_win]
        win_q = q_rad_s[i : i + stable_win]
        if np.std(win_theta) <= theta_std_tol and np.max(np.abs(win_q)) <= q_tol:
            return int(i + stable_win - 1)
    return n - 1


def _segment_rows_auto(
    *,
    run_id: str,
    df: pd.DataFrame,
    protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    t_s = df["t_s"].to_numpy(dtype=float)
    theta_deg = df["theta_deg"].to_numpy(dtype=float)
    q_rad_s = df["q_rad_s"].to_numpy(dtype=float)
    start_indices = _detect_start_indices(theta_deg, t_s, protocol)

    max_segments = int(protocol["segmentation"]["max_segments_per_run"])
    rows: list[dict[str, Any]] = []
    for i, idx in enumerate(start_indices[:max_segments], start=1):
        seg_id = f"{run_id}_S{i:02d}"
        end_idx = _stable_end_index(
            theta_deg=theta_deg,
            q_rad_s=q_rad_s,
            t_s=t_s,
            start_idx=int(idx),
            protocol=protocol,
        )
        rows.append(
            {
                "run_id": run_id,
                "segment_id": seg_id,
                "repeat_id": 1,
                "theta0_meas_deg": float(theta_deg[idx]),
                "q0_meas_rad_s": float(q_rad_s[idx]),
                "t_start_s": float(t_s[idx]),
                "t_end_s": float(t_s[end_idx]),
                "segmentation_mode": "auto_detected",
                "split_tag": "train",
                "cv_fold": "",
                "notes": "",
            }
        )
    return rows


def _apply_manifest_overrides(
    auto_rows: list[dict[str, Any]],
    manifest_df: pd.DataFrame,
    run_id: str,
) -> list[dict[str, Any]]:
    by_seg = {
        str(r["segment_id"]): r.copy()
        for r in auto_rows
    }
    mask = manifest_df["run_id"].astype(str) == str(run_id)
    for _, row in manifest_df.loc[mask].iterrows():
        seg_id = str(row["segment_id"])
        if seg_id not in by_seg:
            continue
        if pd.notna(row["t_start_s"]):
            by_seg[seg_id]["t_start_s"] = float(row["t_start_s"])
            by_seg[seg_id]["segmentation_mode"] = "manifest_override"
        if pd.notna(row["t_end_s"]):
            by_seg[seg_id]["t_end_s"] = float(row["t_end_s"])
            by_seg[seg_id]["segmentation_mode"] = "manifest_override"
        if pd.notna(row["split_tag"]):
            by_seg[seg_id]["split_tag"] = str(row["split_tag"])
        if pd.notna(row["cv_fold"]):
            by_seg[seg_id]["cv_fold"] = str(row["cv_fold"])
        if pd.notna(row["notes"]):
            by_seg[seg_id]["notes"] = str(row["notes"])
    return [by_seg[k] for k in sorted(by_seg.keys())]


def _export_segment_csvs(
    *,
    df: pd.DataFrame,
    rows: list[dict[str, Any]],
    out_dir: Path,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    t_all = df["t_s"].to_numpy(dtype=float)
    for row in rows:
        seg_id = str(row["segment_id"])
        t0 = float(row["t_start_s"])
        t1 = float(row["t_end_s"])
        mask = (t_all >= t0) & (t_all <= t1)
        seg = df.loc[mask, ["run_id", "t_s", "theta_rad", "theta_deg", "q_rad_s", "q_deg_s", "q_dot_rad_s2", "q_dot_deg_s2"]].copy()
        if seg.empty:
            continue
        seg["segment_id"] = seg_id
        seg["t_rel_s"] = seg["t_s"] - float(seg["t_s"].iloc[0])
        seg["sample_idx"] = np.arange(len(seg), dtype=int)
        seg = seg[
            [
                "run_id",
                "segment_id",
                "sample_idx",
                "t_rel_s",
                "theta_rad",
                "theta_deg",
                "q_rad_s",
                "q_deg_s",
                "q_dot_rad_s2",
                "q_dot_deg_s2",
            ]
        ]
        path = out_dir / f"{seg_id}.csv"
        seg.to_csv(path, index=False)
        paths.append(path)
    return paths


def _merge_manifest(manifest_df: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    out = manifest_df.copy()
    for row in rows:
        mask = (
            (out["run_id"].astype(str) == str(row["run_id"]))
            & (out["segment_id"].astype(str) == str(row["segment_id"]))
        )
        if mask.any():
            for k, v in row.items():
                out.loc[mask, k] = v
        else:
            out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)
    out = out.sort_values(["run_id", "segment_id"]).reset_index(drop=True)
    return out


def segment_run_csv(
    *,
    run_csv_path: str | Path,
    protocol_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> SegmentExport:
    ensure_pipeline_dirs()
    protocol = load_yaml(protocol_path or default_protocol_path())
    man_path = Path(manifest_path or default_manifest_path())
    seg_dir = Path(out_dir) if out_dir else default_derived_segment_dir()

    df = pd.read_csv(run_csv_path)
    required = {"run_id", "t_s", "theta_deg", "q_rad_s"}
    if not required.issubset(df.columns):
        raise ValueError(f"run csv missing columns {sorted(required - set(df.columns))}")
    run_id = str(df["run_id"].iloc[0])

    man_df = _load_manifest(man_path)
    auto_rows = _segment_rows_auto(run_id=run_id, df=df, protocol=protocol)
    final_rows = _apply_manifest_overrides(auto_rows, man_df, run_id=run_id)
    seg_paths = _export_segment_csvs(df=df, rows=final_rows, out_dir=seg_dir)

    merged = _merge_manifest(man_df, final_rows)
    _save_manifest(man_path, merged)

    meta_path = seg_dir / f"{run_id}_segments.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(final_rows, f, ensure_ascii=False, indent=2)

    return SegmentExport(
        run_id=run_id,
        segment_paths=seg_paths,
        metadata_json_path=meta_path,
        manifest_path=man_path,
    )
