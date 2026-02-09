from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch


@dataclass(frozen=True)
class FrequencyEstimate:
    segment_id: str
    hz_peak: float
    hz_zero_cross: float
    hz_acf: float
    hz_psd: float
    hz_selected: float
    method_selected: str
    consistency_flag: str


def _full_periods_from_peaks(x: np.ndarray, dt: float, min_distance_s: float, prominence: float) -> np.ndarray:
    dist = max(1, int(min_distance_s / dt))
    p_max, _ = find_peaks(x, distance=dist, prominence=prominence)
    p_min, _ = find_peaks(-x, distance=dist, prominence=prominence)
    periods = []
    for idx in (p_max, p_min):
        if len(idx) >= 2:
            periods.extend(np.diff(idx) * dt)
    return np.asarray(periods, dtype=float)


def _zero_crossings(x: np.ndarray, dt: float) -> np.ndarray:
    s = np.sign(x)
    s[s == 0] = 1.0
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if len(idx) < 3:
        return np.asarray([], dtype=float)
    t = idx.astype(float) * dt
    return t[2:] - t[:-2]


def _acf_freq(x: np.ndarray, fs: float, f_min: float, f_max: float) -> float:
    x0 = x - np.mean(x)
    acf = np.correlate(x0, x0, mode="full")[len(x0) - 1 :]
    if len(acf) < 4:
        return float("nan")
    acf = acf / max(acf[0], 1e-12)
    lags = np.arange(len(acf)) / fs
    m = (lags >= 1.0 / f_max) & (lags <= 1.0 / f_min)
    if not np.any(m):
        return float("nan")
    cands = acf[m]
    lag_sel = lags[m][int(np.argmax(cands))]
    if lag_sel <= 0.0:
        return float("nan")
    return 1.0 / lag_sel


def _psd_freq(x: np.ndarray, fs: float, f_min: float, f_max: float) -> float:
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), 4096))
    m = (f >= f_min) & (f <= f_max)
    if not np.any(m):
        return float("nan")
    return float(f[m][np.argmax(pxx[m])])


def estimate_segment_frequency(
    seg_df: pd.DataFrame,
    *,
    protocol: dict[str, Any],
) -> FrequencyEstimate:
    seg_id = str(seg_df["segment_id"].iloc[0]) if "segment_id" in seg_df.columns else "segment"
    t = seg_df["t_rel_s"].to_numpy(dtype=float)
    theta_deg = seg_df["theta_deg"].to_numpy(dtype=float)
    dt = float(np.mean(np.diff(t)))
    fs = 1.0 / dt
    x = theta_deg - np.mean(theta_deg)

    cfg = protocol["frequency_lock"]
    f_min = float(cfg["f_min_hz"])
    f_max = float(cfg["f_max_hz"])
    min_dist_s = float(cfg["min_peak_distance_s"])
    prom = float(cfg["peak_prominence_deg"])
    consistency_tol = float(cfg["consistency_tol_hz"])

    periods_peak = _full_periods_from_peaks(x, dt, min_dist_s, prom)
    hz_peak = float(1.0 / np.median(periods_peak)) if len(periods_peak) else float("nan")

    periods_zc = _zero_crossings(x, dt)
    hz_zc = float(1.0 / np.median(periods_zc)) if len(periods_zc) else float("nan")
    hz_acf = _acf_freq(x, fs, f_min=f_min, f_max=f_max)
    hz_psd = _psd_freq(x, fs, f_min=f_min, f_max=f_max)

    candidates = [
        ("peak", hz_peak),
        ("zero_cross", hz_zc),
        ("acf", hz_acf),
        ("psd", hz_psd),
    ]
    valids = [(k, v) for k, v in candidates if np.isfinite(v) and v > 0]
    if not valids:
        return FrequencyEstimate(
            segment_id=seg_id,
            hz_peak=hz_peak,
            hz_zero_cross=hz_zc,
            hz_acf=hz_acf,
            hz_psd=hz_psd,
            hz_selected=float("nan"),
            method_selected="none",
            consistency_flag="invalid",
        )

    selected_key, selected_val = sorted(valids, key=lambda kv: abs(kv[1] - np.median([v for _, v in valids])))[0]
    spread = max(v for _, v in valids) - min(v for _, v in valids)
    flag = "ok" if spread <= consistency_tol else "warning_spread"

    return FrequencyEstimate(
        segment_id=seg_id,
        hz_peak=hz_peak,
        hz_zero_cross=hz_zc,
        hz_acf=hz_acf,
        hz_psd=hz_psd,
        hz_selected=float(selected_val),
        method_selected=selected_key,
        consistency_flag=flag,
    )


def estimate_frequency_many(
    segment_dfs: list[pd.DataFrame],
    *,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    rows = [estimate_segment_frequency(df, protocol=protocol).__dict__ for df in segment_dfs]
    return pd.DataFrame(rows)

