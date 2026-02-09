from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def sim_flip_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_protocol_path() -> Path:
    return sim_flip_root() / "configs" / "id_protocol.yaml"


def default_manifest_path() -> Path:
    return sim_flip_root() / "configs" / "experiment_manifest.csv"


def default_raw_runs_dir() -> Path:
    return sim_flip_root() / "data" / "raw" / "runs"


def default_derived_run_dir() -> Path:
    return sim_flip_root() / "data" / "derived" / "run_csv"


def default_derived_segment_dir() -> Path:
    return sim_flip_root() / "data" / "derived" / "segments"


def default_results_dir() -> Path:
    return sim_flip_root() / "results"


def ensure_pipeline_dirs() -> dict[str, Path]:
    dirs = {
        "raw_runs": default_raw_runs_dir(),
        "run_csv": default_derived_run_dir(),
        "segments": default_derived_segment_dir(),
        "results": default_results_dir(),
        "configs": sim_flip_root() / "configs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML must contain a mapping object: {path}")
    return data

