#!/usr/bin/env python3
"""Build historical tournament matchup training data from Kaggle files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from matchup_model_utils import FEATURE_NAMES, build_profiles, build_training_examples, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build historical tournament matchup training data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/matchup_training_data.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/matchup_training_data_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["season", "gender", "team_a", "team_b", "label", *FEATURE_NAMES]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for example in examples:
            row = {
                "season": example.season,
                "gender": example.gender,
                "team_a": example.team_a,
                "team_b": example.team_b,
                "label": int(example.label),
            }
            row.update({name: round(example.features[name], 8) for name in FEATURE_NAMES})
            writer.writerow(row)

    season_counts: dict[int, int] = {}
    gender_counts: dict[str, int] = {"men": 0, "women": 0}
    for example in examples:
        season_counts[example.season] = season_counts.get(example.season, 0) + 1
        gender_counts[example.gender] = gender_counts.get(example.gender, 0) + 1

    payload = {
        "rows": len(examples),
        "feature_names": FEATURE_NAMES,
        "gender_counts": gender_counts,
        "season_min": min(season_counts) if season_counts else None,
        "season_max": max(season_counts) if season_counts else None,
        "season_counts": season_counts,
        "output_csv": str(args.output_csv),
    }
    write_json(args.output_json, payload)

    print(f"Wrote {len(examples)} rows to {args.output_csv}")
    print(f"Summary: {args.output_json}")


if __name__ == "__main__":
    main()
