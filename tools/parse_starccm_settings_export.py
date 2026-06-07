#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize STAR-CCM+ settings exported by ExportSimCfdSettings.java.

The macro creates three files:
  - settings_index.csv
  - settings_report.txt
  - export_warnings.txt

This parser turns the large CSV into an AI handoff markdown file.
It extracts stable CFD-review fields and adds coverage/index sections so a
future AI can trace back from the summary to the full macro export.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


CSV_NAME = "settings_index.csv"
REPORT_NAME = "settings_report.txt"
WARNINGS_NAME = "export_warnings.txt"


@dataclass(frozen=True)
class Row:
    section: str
    path: str
    object_class: str
    prop: str
    value: str


def read_rows(path: Path) -> list[Row]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for item in reader:
            rows.append(
                Row(
                    section=item.get("section", ""),
                    path=item.get("path", ""),
                    object_class=item.get("object_class", ""),
                    prop=item.get("property", ""),
                    value=item.get("value", ""),
                )
            )
        return rows


def first_value(rows: Iterable[Row], path: str | None = None, prop: str | None = None,
                contains_path: str | None = None, object_class: str | None = None) -> str:
    for row in rows:
        if path is not None and row.path != path:
            continue
        if contains_path is not None and contains_path not in row.path:
            continue
        if object_class is not None and object_class not in row.object_class:
            continue
        if prop is not None and row.prop != prop:
            continue
        return row.value
    return ""


def rows_where(rows: Iterable[Row], *, contains_path: str | None = None,
               prop: str | None = None, object_class_contains: str | None = None) -> list[Row]:
    out = []
    for row in rows:
        if contains_path is not None and contains_path not in row.path:
            continue
        if prop is not None and row.prop != prop:
            continue
        if object_class_contains is not None and object_class_contains not in row.object_class:
            continue
        out.append(row)
    return out


def object_blocks(rows: Iterable[Row], manager_token: str, name_prop: str = "_objectName") -> dict[str, list[Row]]:
    blocks: dict[str, list[Row]] = {}
    current_path = ""
    for row in rows:
        if manager_token not in row.path:
            continue
        if row.prop == name_prop:
            current_path = row.path
            blocks.setdefault(current_path, []).append(row)
        elif current_path and row.path.startswith(current_path):
            blocks.setdefault(current_path, []).append(row)
    return blocks


def object_name(block: list[Row]) -> str:
    for row in block:
        if row.prop == "_objectName":
            return row.value
    for row in block:
        if row.prop == "presentationName":
            return row.value
    return ""


def prop_value(block: list[Row], prop: str) -> str:
    for row in block:
        if row.prop == prop:
            return row.value
    return ""


def truncate(value: str, limit: int = 320) -> str:
    value = value.replace("\n", " ").replace("\r", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def unique_objects(rows: Iterable[Row]) -> list[Row]:
    seen: set[tuple[str, str, str]] = set()
    objects: list[Row] = []
    for row in rows:
        if row.prop != "_objectName":
            continue
        key = (row.path, row.object_class, row.value)
        if key in seen:
            continue
        seen.add(key)
        objects.append(row)
    return objects


def block_by_exact_path(rows: Iterable[Row]) -> dict[str, list[Row]]:
    blocks: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        blocks[row.path].append(row)
    return dict(blocks)


def selected_props_for_path(path_rows: list[Row], props: Iterable[str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for prop in props:
        for row in path_rows:
            if row.prop == prop and row.value:
                found.append((prop, row.value))
                break
    return found


def object_table(lines: list[str], title: str, objects: list[Row], rows_by_path: dict[str, list[Row]],
                 path_tokens: Iterable[str], props: Iterable[str], limit: int | None = None) -> None:
    tokens = tuple(path_tokens)
    selected = [obj for obj in objects if any(token in obj.path for token in tokens)]
    if not selected:
        return
    lines.append("")
    lines.append(f"### {title}")
    if limit is not None and len(selected) > limit:
        lines.append(f"- Showing first `{limit}` of `{len(selected)}` objects. Use `settings_index.csv` for the full list.")
        selected = selected[:limit]
    lines.append("")
    lines.append("| Object | Class | Path | Key properties |")
    lines.append("|---|---|---|---|")
    for obj in selected:
        prop_items = selected_props_for_path(rows_by_path.get(obj.path, []), props)
        prop_text = "<br>".join(f"{p}: `{truncate(v, 140)}`" for p, v in prop_items)
        lines.append(f"| `{truncate(obj.value, 80)}` | `{truncate(obj.object_class, 80)}` | `{truncate(obj.path, 120)}` | {prop_text or '-'} |")


def bullet(lines: list[str], key: str, value: str) -> None:
    if value:
        lines.append(f"- {key}: `{value}`")


def summarize_warnings(warnings_path: Path | None) -> tuple[int, list[str]]:
    if not warnings_path or not warnings_path.exists():
        return 0, []
    text = warnings_path.read_text(encoding="utf-8", errors="replace")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    categories = Counter()
    for line in lines:
        if "FieldFunctionManager" in line:
            categories["field function values skipped"] += 1
        elif "No getObjects()" in line:
            categories["manager has no getObjects collection"] += 1
        elif "null" in line:
            categories["null property"] += 1
        else:
            categories["other"] += 1
    summary = [f"{name}: {count}" for name, count in categories.most_common()]
    return len(lines), summary


def build_summary(rows: list[Row], report_path: Path | None, warnings_path: Path | None,
                  index_path: Path | None = None) -> str:
    lines: list[str] = []
    objects = unique_objects(rows)
    rows_by_path = block_by_exact_path(rows)
    section_counts = Counter(row.section for row in rows)
    object_class_counts = Counter(obj.object_class for obj in objects)
    sim_name = first_value(rows, path="/Simulation", prop="presentationName")
    session_path = first_value(rows, path="/Simulation", prop="sessionPath")
    session_dir = first_value(rows, path="/Simulation", prop="sessionDir")
    star_version = first_value(rows, path="/Simulation", prop="releaseNumber")
    current_version = first_value(rows, path="/Simulation", prop="currentVersion")
    workers = first_value(rows, path="/Simulation", prop="numberOfWorkers")

    lines.append("# STAR-CCM+ CFD Settings Summary")
    lines.append("")
    lines.append("## 0. How To Use This File")
    lines.append("- Purpose: AI handoff document for understanding and reviewing the current STAR-CCM+ CFD setup.")
    lines.append("- This file extracts and indexes the macro export. It does not replace the raw export files.")
    lines.append("- Keep `settings_index.csv`, `settings_report.txt`, and `export_warnings.txt` together with this summary when asking AI for CFD guidance.")
    lines.append("- Guarantee boundary: AI can rely on this file for the key CFD configuration and use the linked raw CSV/report for all macro-exported properties. STAR-CCM+ settings not exposed by the Java API still require targeted macros or GUI screenshots.")
    if index_path:
        bullet(lines, "raw settings index", str(index_path))
    if report_path:
        bullet(lines, "raw settings report", str(report_path))
    if warnings_path:
        bullet(lines, "raw export warnings", str(warnings_path))
    lines.append("")
    lines.append("## 0.1 Export Coverage")
    bullet(lines, "CSV property rows", str(len(rows)))
    bullet(lines, "unique exported objects", str(len(objects)))
    bullet(lines, "unique object classes", str(len(object_class_counts)))
    lines.append("- Largest exported sections:")
    for section, count in section_counts.most_common(12):
        lines.append(f"  - `{section}`: `{count}` rows")
    lines.append("- Most frequent exported object classes:")
    for cls, count in object_class_counts.most_common(12):
        lines.append(f"  - `{cls}`: `{count}` objects")
    lines.append("")
    lines.append("## 1. Case Identity")
    bullet(lines, "simulation", sim_name)
    bullet(lines, "session path", session_path)
    bullet(lines, "session dir", session_dir)
    if star_version or current_version:
        bullet(lines, "STAR-CCM+ version", f"{star_version} / {current_version}".strip(" /"))
    bullet(lines, "workers", workers)
    if report_path and report_path.exists():
        generated = ""
        for line in report_path.read_text(encoding="utf-8", errors="replace").splitlines()[:5]:
            if line.startswith("Generated:"):
                generated = line.replace("Generated:", "").strip()
                break
        bullet(lines, "export generated", generated)

    lines.append("")
    lines.append("## 2. Physics Continuum")
    model_children = first_value(rows, contains_path="/getContinuumManager/000_Physics 1/getModelManager",
                                 prop="children")
    bullet(lines, "enabled model list", model_children)
    bullet(lines, "regions linked to continuum",
           first_value(rows, contains_path="/getContinuumManager/000_Physics 1", prop="regions"))
    bullet(lines, "overset enabled",
           first_value(rows, contains_path="/getContinuumManager/000_Physics 1", prop="oversetEnabled"))
    bullet(lines, "chimera grid enabled",
           first_value(rows, contains_path="/getContinuumManager/000_Physics 1", prop="chimeraGridEnabled"))
    physics_model_objects = [
        obj for obj in objects
        if "/getContinuumManager" in obj.path and "/getModelManager/" in obj.path
    ]
    if physics_model_objects:
        lines.append("- enabled model objects:")
        for obj in physics_model_objects:
            if "ModelManager" in obj.object_class:
                continue
            lines.append(f"  - `{obj.value}`: `{obj.object_class}`")

    lines.append("")
    lines.append("## 3. Regions And Boundaries")
    for path, block in object_blocks(rows, "/getRegionManager").items():
        name = object_name(block)
        if not name or "/getBoundaryManager/" in path or "/getSegmentManager" in path:
            continue
        if "Region" not in block[0].object_class:
            continue
        lines.append(f"- Region `{name}`")
        boundaries = prop_value(block, "boundaries")
        if boundaries:
            lines.append(f"  - boundaries: `{boundaries}`")
        for bpath, bblock in object_blocks(rows, path + "/getBoundaryManager").items():
            bname = object_name(bblock)
            btype = prop_value(bblock, "boundaryType")
            if bname and btype:
                lines.append(f"  - `{bname}`: `{btype}`")

    lines.append("")
    lines.append("## 4. Interfaces")
    interface_blocks = object_blocks(rows, "/getInterfaceManager")
    if not interface_blocks:
        lines.append("- No interface objects found in the parsed export.")
    for _, block in interface_blocks.items():
        name = object_name(block)
        cls = block[0].object_class if block else ""
        if name:
            lines.append(f"- `{name}`: `{cls}`")
            for prop in ("interfaceType", "prismLayerShrinkage", "oversetConservation", "enabledZeroDonorCheck"):
                value = prop_value(block, prop)
                if value:
                    lines.append(f"  - {prop}: `{value}`")

    lines.append("")
    lines.append("## 5. Mesh")
    bullet(lines, "total cell count",
           first_value(rows, contains_path="/getRepresentationManager/000_Geometry/getManager/002_Volume Mesh",
                       prop="cellCount"))
    mesh_blocks = object_blocks(rows, "/star.meshing.MeshOperationManager")
    for _, block in mesh_blocks.items():
        name = object_name(block)
        if not name:
            continue
        cls = block[0].object_class
        if "MeshOperation" not in cls and "AutoMeshOperation" not in cls:
            continue
        lines.append(f"- `{name}`: `{cls}`")
        for prop in ("meshersCollection", "partsInput", "customMeshControls", "meshInParallel",
                     "mesherParallelModeOptionInput"):
            value = prop_value(block, prop)
            if value:
                lines.append(f"  - {prop}: `{value}`")
    lines.append("- Note: if base size, custom controls, prism-layer height, or y+ are absent here, the macro export needs a targeted mesh-control extension.")
    mesh_signals = [
        row for row in rows
        if any(token.lower() in (row.prop + " " + row.path + " " + row.value).lower()
               for token in ("basesize", "target", "minimum", "prism", "y+", "wall y+", "cellcount", "volume mesh"))
    ]
    if mesh_signals:
        lines.append("- mesh-related exported signals:")
        for row in mesh_signals[:30]:
            lines.append(f"  - `{truncate(row.path, 100)}` / `{row.prop}` = `{truncate(row.value, 160)}`")
        if len(mesh_signals) > 30:
            lines.append(f"  - `{len(mesh_signals) - 30}` additional mesh-related rows exist in `settings_index.csv`.")

    lines.append("")
    lines.append("## 6. Coordinate Systems")
    for _, block in object_blocks(rows, "/getCoordinateSystemManager").items():
        name = object_name(block)
        if name != "Body_frame":
            continue
        lines.append(f"- `{name}`")
        for prop in ("originInput", "xVectorInput", "xyPlaneInput", "basis0", "basis1", "basis2"):
            value = prop_value(block, prop)
            if value:
                lines.append(f"  - {prop}: `{value}`")

    lines.append("")
    lines.append("## 7. Reports")
    for _, block in object_blocks(rows, "/getReportManager").items():
        name = object_name(block)
        if name not in {"Cx", "Cz", "Cm"}:
            continue
        cls = block[0].object_class
        lines.append(f"- `{name}`: `{cls}`")
        for prop in ("coordinateSystem", "directionInput", "originInput", "force", "partsInput",
                     "referenceAreaInput", "referenceDensityInput", "referenceVelocityInput",
                     "referencePressureInput", "referenceRadiusInput"):
            value = prop_value(block, prop)
            if value:
                lines.append(f"  - {prop}: `{value}`")

    lines.append("")
    lines.append("## 8. Solvers And Stopping Criteria")
    for _, block in object_blocks(rows, "/getSolverManager").items():
        name = object_name(block)
        if not name:
            continue
        cls = block[0].object_class
        if "Solver" not in cls:
            continue
        lines.append(f"- `{name}`: `{cls}`")
        for prop in ("scheme", "maximumUnlimitedVelocityInput", "enabled", "frozen"):
            value = prop_value(block, prop)
            if value:
                lines.append(f"  - {prop}: `{value}`")
    for _, block in object_blocks(rows, "/getSolverStoppingCriterionManager").items():
        name = object_name(block)
        if not name:
            continue
        lines.append(f"- stopping criterion `{name}`: `{block[0].object_class}`")
        for prop in ("maximumNumberSteps", "innerIterationCriterion", "enabled", "satisfied"):
            value = prop_value(block, prop)
            if value:
                lines.append(f"  - {prop}: `{value}`")

    lines.append("")
    lines.append("## 9. Monitors")
    for _, block in object_blocks(rows, "/getMonitorManager").items():
        name = object_name(block)
        if not name:
            continue
        cls = block[0].object_class
        if "Monitor" not in cls:
            continue
        report = prop_value(block, "report")
        if report:
            lines.append(f"- `{name}`: `{cls}`, report `{report}`")
        else:
            lines.append(f"- `{name}`: `{cls}`")

    lines.append("")
    lines.append("## 10. Global Parameters")
    param_blocks = object_blocks(rows, "/getGlobalParameterManager")
    for _, block in param_blocks.items():
        name = object_name(block)
        if not name or name == "参数":
            continue
        quantity = prop_value(block, "quantityInput")
        references = prop_value(block, "references")
        lines.append(f"- `{name}`: `{quantity or 'no quantityInput found'}`")
        if references:
            lines.append(f"  - references: `{references}`")
        if references == "size=0 []":
            lines.append("  - review note: this parameter is exported but appears unused; do not treat it as the physical case angle unless geometry/boundary references confirm it.")

    lines.append("")
    lines.append("## 11. AI Completeness And Missing-Field Audit")
    critical_checks = [
        ("inlet velocity magnitude/direction", ("VelocityMagnitude", "velocity magnitude", "VelocityProfile", "directionInput")),
        ("outlet pressure value", ("Pressure", "pressureInput", "StaticPressure")),
        ("turbulence inlet quantities", ("Turbulence", "turbulent", "Intensity", "ViscosityRatio")),
        ("mesh base size and custom control numeric values", ("baseSize", "Base Size", "targetSurfaceSize", "minimumSurfaceSize")),
        ("prism layer numeric settings", ("NumPrism", "PrismLayer", "first prism", "Prism Cell Thickness")),
        ("wall y+ statistics/report", ("Wall Y+", "y+", "Yplus")),
        ("domain dimensions and inlet/outlet distances", ("SimpleBlockPart", "Block", "extent", "Coordinate")),
    ]
    for label, needles in critical_checks:
        matches = [
            row for row in rows
            if any(needle.lower() in (row.path + " " + row.prop + " " + row.value).lower() for needle in needles)
        ]
        if matches:
            lines.append(f"- `{label}`: found `{len(matches)}` candidate rows in `settings_index.csv`.")
            for row in matches[:5]:
                lines.append(f"  - `{truncate(row.path, 100)}` / `{row.prop}` = `{truncate(row.value, 160)}`")
            if len(matches) > 5:
                lines.append("  - additional candidate rows omitted from summary; inspect `settings_index.csv`.")
        else:
            lines.append(f"- `{label}`: not clearly exported; use a targeted STAR-CCM+ macro or GUI screenshot/manual record.")

    lines.append("")
    lines.append("## 12. Export Warnings")
    warning_count, warning_summary = summarize_warnings(warnings_path)
    lines.append(f"- warning lines: `{warning_count}`")
    for item in warning_summary:
        lines.append(f"- {item}")
    lines.append("- Interpretation: field-function value warnings are usually acceptable for settings review; missing targeted fields should be fixed with a specialized macro extension.")

    lines.append("")
    lines.append("## 13. CFD Review Checklist")
    lines.extend([
        "- Confirm the visible geometry/orientation and do not rely only on unused global parameters.",
        "- Confirm all mesh levels use identical physics, reports, reference area, reference velocity, reference density, moment origin, and body coordinate system.",
        "- Confirm inlet velocity direction/magnitude, outlet pressure, turbulence inlet quantities, and domain dimensions.",
        "- Confirm mesh controls: base size, local refinement, prism layers, first-layer height, total prism thickness, and y+ statistics.",
        "- For static AoA maps, use these cases only as quasi-steady force/moment coefficient evidence, not added-mass, damping, or free-running CFD validation.",
    ])

    lines.append("")
    lines.append("## 14. Object Inventory For AI Traceability")
    lines.append("This inventory is a compact map of important exported objects. For full property-level detail, search `settings_index.csv` by the listed path or object name.")
    object_table(
        lines,
        "Continuum, Models, Initial Conditions",
        objects,
        rows_by_path,
        ["/getContinuumManager"],
        ["_objectName", "children", "enabledModels", "regions", "oversetEnabled", "chimeraGridEnabled"],
        limit=80,
    )
    object_table(
        lines,
        "Geometry Parts And Imported/Derived Parts",
        objects,
        rows_by_path,
        ["/getGeometryPartManager", "/getPartManager"],
        ["_objectName", "presentationName", "surfaces", "parts", "inputParts", "tagsInput"],
        limit=120,
    )
    object_table(
        lines,
        "Regions, Boundaries, And Interfaces",
        objects,
        rows_by_path,
        ["/getRegionManager", "/getInterfaceManager"],
        ["_objectName", "boundaryType", "interfaceType", "boundaries", "partsInput", "region", "prismLayerShrinkage"],
        limit=120,
    )
    object_table(
        lines,
        "Mesh Operations And Representations",
        objects,
        rows_by_path,
        ["/star.meshing.MeshOperationManager", "/getRepresentationManager", "/getMeshManager"],
        ["_objectName", "meshersCollection", "cellCount", "partsInput", "customMeshControls", "meshInParallel"],
        limit=140,
    )
    object_table(
        lines,
        "Reports, Monitors, Solvers, And Stopping Criteria",
        objects,
        rows_by_path,
        ["/getReportManager", "/getMonitorManager", "/getSolverManager", "/getSolverStoppingCriterionManager"],
        ["_objectName", "report", "fieldFunction", "coordinateSystem", "directionInput", "partsInput", "scheme", "maximumNumberSteps"],
        limit=180,
    )
    object_table(
        lines,
        "Coordinate Systems, Global Parameters, Field Functions, Plots, And Scenes",
        objects,
        rows_by_path,
        ["/getCoordinateSystemManager", "/getGlobalParameterManager", "/getFieldFunctionManager", "/getPlotManager", "/getSceneManager"],
        ["_objectName", "quantityInput", "references", "originInput", "xVectorInput", "xyPlaneInput", "functionName", "sourceString"],
        limit=220,
    )

    lines.append("")
    lines.append("## 15. Raw CSV Search Guide")
    lines.append("- CSV columns: `section,path,object_class,property,value`.")
    lines.append("- To inspect a setting, search by object name, path fragment, class name, or property name.")
    lines.append("- Recommended search tokens for CFD guidance: `Physics 1`, `ModelManager`, `RegionManager`, `BoundaryManager`, `Inlet`, `Outlet`, `AD_v2_surface`, `Overmesh`, `MeshOperationManager`, `Volume Mesh`, `cellCount`, `ReportManager`, `Cx`, `Cz`, `Cm`, `Body_frame`, `SolverManager`, `StoppingCriterion`, `GlobalParameterManager`.")
    lines.append("- If a future AI needs exact values not shown in the summary, it should query `settings_index.csv` first before asking for another STAR-CCM+ export.")
    lines.append("")
    return "\n".join(lines)


def find_inputs(args: argparse.Namespace) -> tuple[Path, Path | None, Path | None, Path]:
    if args.export_dir:
        export_dir = Path(args.export_dir)
        index = export_dir / CSV_NAME
        report = export_dir / REPORT_NAME
        warnings = export_dir / WARNINGS_NAME
        output = Path(args.output) if args.output else export_dir / "cfd_settings_summary.md"
    else:
        if not args.index:
            raise SystemExit("Either --export-dir or --index must be provided.")
        index = Path(args.index)
        report = Path(args.report) if args.report else None
        warnings = Path(args.warnings) if args.warnings else None
        output = Path(args.output) if args.output else index.with_name("cfd_settings_summary.md")
    return index, report, warnings, output


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize STAR-CCM+ settings export files.")
    parser.add_argument("--export-dir", help="Folder containing settings_index.csv, settings_report.txt, and export_warnings.txt.")
    parser.add_argument("--index", help="Path to settings_index.csv.")
    parser.add_argument("--report", help="Path to settings_report.txt.")
    parser.add_argument("--warnings", help="Path to export_warnings.txt.")
    parser.add_argument("--output", help="Output markdown path. Defaults to cfd_settings_summary.md next to the CSV.")
    args = parser.parse_args()

    index, report, warnings, output = find_inputs(args)
    if not index.exists():
        raise SystemExit(f"Missing index CSV: {index}")
    rows = read_rows(index)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_summary(rows, report, warnings, index), encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
