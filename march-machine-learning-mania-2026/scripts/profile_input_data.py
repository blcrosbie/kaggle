#!/usr/bin/env python3
"""Profile all competition CSV files and write dataset reports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class FileSummary:
    file: str
    rows: int
    columns: int
    duplicate_rows: int
    missing_cells: int
    missing_pct: float
    memory_mb: float
    min_season: int | None
    max_season: int | None


def _float_or_none(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _int_or_none(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _column_profile(series: pd.Series) -> dict[str, Any]:
    null_count = int(series.isna().sum())
    non_null = int(series.notna().sum())
    out: dict[str, Any] = {
        "dtype": str(series.dtype),
        "non_null": non_null,
        "null_count": null_count,
        "null_pct": round(null_count / max(len(series), 1) * 100, 4),
        "nunique_non_null": int(series.nunique(dropna=True)),
    }
    if pd.api.types.is_numeric_dtype(series):
        out["numeric_stats"] = {
            "min": _float_or_none(series.min()),
            "max": _float_or_none(series.max()),
            "mean": _float_or_none(series.mean()),
            "std": _float_or_none(series.std()),
        }
    else:
        top_values = (
            series.astype("string")
            .fillna("<NA>")
            .value_counts(dropna=False)
            .head(5)
            .to_dict()
        )
        out["top_values"] = {str(k): int(v) for k, v in top_values.items()}
    return out


def _build_notes(
    summaries: list[FileSummary], profile_details: dict[str, Any], reports_dir: Path
) -> None:
    men_files = [s for s in summaries if Path(s.file).name.startswith("M")]
    women_files = [s for s in summaries if Path(s.file).name.startswith("W")]
    shared_files = [
        s
        for s in summaries
        if not Path(s.file).name.startswith("M") and not Path(s.file).name.startswith("W")
    ]
    total_rows = sum(s.rows for s in summaries)
    largest = sorted(summaries, key=lambda s: s.rows, reverse=True)[:8]

    lines: list[str] = [
        "# Data Profile Notes",
        "",
        f"- Total CSV files profiled: **{len(summaries)}**",
        f"- Total rows across all CSV files: **{total_rows:,}**",
        f"- Men's files: **{len(men_files)}**",
        f"- Women's files: **{len(women_files)}**",
        f"- Shared files: **{len(shared_files)}**",
        "",
        "## Largest files by rows",
    ]
    for fs in largest:
        lines.append(
            f"- `{Path(fs.file).name}`: {fs.rows:,} rows, {fs.columns} cols, missing={fs.missing_pct:.2f}%"
        )

    lines.extend(["", "## Quick quality flags"])
    for fs in summaries:
        flags: list[str] = []
        if fs.missing_pct > 5:
            flags.append(f"high missingness {fs.missing_pct:.2f}%")
        if fs.duplicate_rows > 0:
            flags.append(f"{fs.duplicate_rows} duplicate rows")
        if not flags:
            continue
        lines.append(f"- `{Path(fs.file).name}`: " + ", ".join(flags))
    if lines[-1] == "## Quick quality flags":
        lines.append("- No major quality flags from missingness/duplicate checks.")

    lines.extend(["", "## Season coverage by file"])
    for fs in sorted(summaries, key=lambda x: Path(x.file).name):
        if fs.min_season is None:
            continue
        lines.append(f"- `{Path(fs.file).name}`: {fs.min_season} to {fs.max_season}")

    check_messages: list[str] = []
    pairings = [
        ("MRegularSeasonCompactResults.csv", "MRegularSeasonDetailedResults.csv", 2003),
        ("WRegularSeasonCompactResults.csv", "WRegularSeasonDetailedResults.csv", 2010),
        ("MNCAATourneyCompactResults.csv", "MNCAATourneyDetailedResults.csv", 2003),
        ("WNCAATourneyCompactResults.csv", "WNCAATourneyDetailedResults.csv", 2010),
    ]
    for compact_name, detail_name, min_season in pairings:
        if compact_name not in profile_details or detail_name not in profile_details:
            continue
        compact = pd.DataFrame(profile_details[compact_name]["preview_rows"])
        detail = pd.DataFrame(profile_details[detail_name]["preview_rows"])
        if "Season" not in compact or "Season" not in detail:
            continue
        cmin = compact["Season"].dropna().min()
        dmin = detail["Season"].dropna().min()
        check_messages.append(
            f"- `{compact_name}` vs `{detail_name}` preview minimum seasons: {cmin} vs {dmin} (expected detailed starts around {min_season})."
        )
    lines.extend(["", "## Compact vs detailed sanity checks (preview-based)"])
    if check_messages:
        lines.extend(check_messages)
    else:
        lines.append("- Pair sanity checks unavailable.")

    (reports_dir / "data_profile_notes.md").write_text("\n".join(lines), encoding="utf-8")


def profile_csvs(input_dir: Path, reports_dir: Path, preview_rows: int) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    summaries: list[FileSummary] = []
    details: dict[str, Any] = {}

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, low_memory=False)
        file_name = csv_path.name
        rows, cols = df.shape
        missing_cells = int(df.isna().sum().sum())
        missing_pct = round(missing_cells / max(rows * cols, 1) * 100, 4)
        duplicate_rows = int(df.duplicated().sum())
        memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 3)
        min_season = _int_or_none(df["Season"].min()) if "Season" in df.columns else None
        max_season = _int_or_none(df["Season"].max()) if "Season" in df.columns else None

        summaries.append(
            FileSummary(
                file=file_name,
                rows=rows,
                columns=cols,
                duplicate_rows=duplicate_rows,
                missing_cells=missing_cells,
                missing_pct=missing_pct,
                memory_mb=memory_mb,
                min_season=min_season,
                max_season=max_season,
            )
        )

        column_profiles = {col: _column_profile(df[col]) for col in df.columns}
        details[file_name] = {
            "file": file_name,
            "shape": {"rows": rows, "columns": cols},
            "columns": list(df.columns),
            "duplicate_rows": duplicate_rows,
            "missing_cells": missing_cells,
            "missing_pct": missing_pct,
            "memory_mb": memory_mb,
            "season_range": {"min": min_season, "max": max_season},
            "column_profiles": column_profiles,
            "preview_rows": df.head(preview_rows).to_dict(orient="records"),
        }

    summary_df = pd.DataFrame([asdict(s) for s in summaries]).sort_values("file")
    summary_df.to_csv(reports_dir / "data_profile_summary.csv", index=False)
    (reports_dir / "data_profile_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    _build_notes(summaries, details, reports_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile all CSV files in input directory.")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--preview-rows", type=int, default=5)
    args = parser.parse_args()
    profile_csvs(args.input_dir, args.reports_dir, args.preview_rows)


if __name__ == "__main__":
    main()
