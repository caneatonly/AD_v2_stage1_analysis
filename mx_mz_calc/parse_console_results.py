"""Parse STAR-CCM+ console output for the forced-acceleration case.

Outputs:
  1. mx_data/Fx_PhysicalTime.csv or mz_data/Fz_PhysicalTime.csv
     One row per physical time step, using the final reported row after
     "TimeStep n: Time t".

  2. mx_data/Residuals.csv or mz_data/Residuals.csv
     One row per solver iteration containing residuals and reported forces.

The parser is intentionally text-based because STAR-CCM+ monitor exports may
sample by iteration and can miss the last inner iteration of each time step.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
TIMESTEP_RE = re.compile(rf"^\s*TimeStep\s+(\d+):\s+Time\s+({FLOAT_RE})\s*$")


BASE_RESIDUAL_FIELDS = [
    "Iteration",
    "Continuity",
    "X_momentum",
    "Y_momentum",
    "Z_momentum",
    "Tke",
    "Sdr",
    "Cx",
    "Cm",
    "Cz",
    "TimeStep",
    "Physical_time_s",
    "Vcmd_mps",
    "wall_Yplus",
]

BASE_FORCE_TIME_FIELDS = [
    "TimeStep",
    "Physical_time_s",
    "Iteration",
    "Vcmd_mps",
    "wall_Yplus",
]


def parse_numeric_row(line: str) -> list[float] | None:
    """Return numeric STAR-CCM+ data row, or None for non-data lines."""
    parts = line.split()
    if len(parts) < 11:
        return None

    # Iteration rows begin with an integer. Other STAR-CCM+ log lines may
    # contain numbers but do not follow this tabular format.
    if not re.fullmatch(r"\d+", parts[0]):
        return None

    try:
        values = [float(x) for x in parts]
    except ValueError:
        return None

    if len(values) < 11:
        return None

    return values


def parse_console(
    input_path: Path,
    force_column: str = "Fx_body_N",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    residual_rows: list[dict[str, object]] = []
    force_time_rows: list[dict[str, object]] = []

    pending_timestep: int | None = None
    pending_time: float | None = None

    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            timestep_match = TIMESTEP_RE.match(line)
            if timestep_match:
                pending_timestep = int(timestep_match.group(1))
                pending_time = float(timestep_match.group(2))
                continue

            values = parse_numeric_row(line)
            if values is None:
                continue

            iteration = int(values[0])

            row = {
                "Iteration": iteration,
                "Continuity": values[1],
                "X_momentum": values[2],
                "Y_momentum": values[3],
                "Z_momentum": values[4],
                "Tke": values[5],
                "Sdr": values[6],
                "Cx": values[7],
                "Cm": values[8],
                "Cz": values[9],
                force_column: values[10],
                "TimeStep": "",
                "Physical_time_s": "",
                "Vcmd_mps": "",
                "wall_Yplus": "",
            }

            # Final time-step rows contain physical time and command velocity
            # after the force value. Some runs also print wall Y+ after Vcmd.
            # Use the explicit values from the row and keep the preceding
            # TimeStep marker for the time-step index.
            if len(values) >= 13:
                physical_time = values[11]
                vcmd = values[12]
                wall_yplus = values[13] if len(values) >= 14 else ""
                row["TimeStep"] = pending_timestep if pending_timestep is not None else ""
                row["Physical_time_s"] = physical_time
                row["Vcmd_mps"] = vcmd
                row["wall_Yplus"] = wall_yplus

                force_time_rows.append(
                    {
                        "TimeStep": pending_timestep if pending_timestep is not None else "",
                        "Physical_time_s": physical_time,
                        "Iteration": iteration,
                        force_column: values[10],
                        "Vcmd_mps": vcmd,
                        "wall_Yplus": wall_yplus,
                    }
                )

                pending_timestep = None
                pending_time = None
            elif pending_time is not None:
                # Keep pending_timestep until the final row with time data.
                pass

            residual_rows.append(row)

    return residual_rows, force_time_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_input = script_dir / "mx_data" / "console_resuls.txt"
    default_force = script_dir / "mx_data" / "Fx_PhysicalTime.csv"
    default_residuals = script_dir / "mx_data" / "Residuals.csv"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--force-column", default="Fx_body_N")
    parser.add_argument("--force-output", "--fx-output", type=Path, default=default_force)
    parser.add_argument("--residuals-output", type=Path, default=default_residuals)
    args = parser.parse_args()

    residual_fields = BASE_RESIDUAL_FIELDS[:10] + [args.force_column] + BASE_RESIDUAL_FIELDS[10:]
    force_time_fields = BASE_FORCE_TIME_FIELDS[:3] + [args.force_column] + BASE_FORCE_TIME_FIELDS[3:]

    residual_rows, force_time_rows = parse_console(args.input, args.force_column)
    write_csv(args.force_output, force_time_rows, force_time_fields)
    write_csv(args.residuals_output, residual_rows, residual_fields)

    print(f"Input: {args.input}")
    print(f"Wrote: {args.force_output} ({len(force_time_rows)} rows)")
    print(f"Wrote: {args.residuals_output} ({len(residual_rows)} rows)")
    if force_time_rows:
        first = force_time_rows[0]
        last = force_time_rows[-1]
        print(
            "Force_PhysicalTime range: "
            f"t={first['Physical_time_s']}..{last['Physical_time_s']} s, "
            f"TimeStep={first['TimeStep']}..{last['TimeStep']}"
        )


if __name__ == "__main__":
    main()
