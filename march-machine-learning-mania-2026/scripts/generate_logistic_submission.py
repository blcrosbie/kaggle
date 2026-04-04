#!/usr/bin/env python3
"""Generate a Kaggle submission using pure-Python logistic regression."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from matchup_model_utils import (
    FEATURE_NAMES,
    build_matchup_features,
    build_profiles,
    build_training_examples,
    evaluate_by_season,
    read_submission_rows,
    train_logistic_model,
    write_json,
    write_submission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a submission using logistic regression matchup features.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/SampleSubmissionStage1.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/stage1_logistic.csv"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path for JSON manifest. Defaults to output + .manifest.json",
    )
    parser.add_argument(
        "--eval-start-season",
        type=int,
        default=2018,
        help="First season to include in walk-forward validation summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)
    evaluation = evaluate_by_season(
        examples,
        trainer_name="logistic",
        feature_names=FEATURE_NAMES,
        min_test_season=args.eval_start_season,
    )

    rows = read_submission_rows(args.sample_submission)
    models: dict[tuple[str, int], object] = {}
    predictions: list[tuple[str, float]] = []
    model_rows: dict[tuple[str, int], int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in rows:
        key = (gender, season)
        if key not in models:
            train_rows = [
                example
                for example in examples
                if example.gender == gender and example.season < season
            ]
            if not train_rows:
                raise ValueError(f"No training rows available for gender={gender} before season={season}")
            models[key] = train_logistic_model(train_rows, feature_names=FEATURE_NAMES)
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        pred = models[key].predict_prob(features)
        predictions.append((game_id, pred))
        model_rows[key] += 1

    write_submission(args.output, predictions)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "logistic_regression",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "feature_names": FEATURE_NAMES,
        "walkforward_evaluation": evaluation,
        "season_gender_row_counts": {
            f"{gender}_{season}": count for (gender, season), count in sorted(model_rows.items())
        },
    }
    write_json(manifest_path, payload)

    print(f"Submission written to {args.output}")
    print(f"Manifest written to {manifest_path}")
    print(f"Walk-forward Brier: {evaluation['overall_brier']}")


if __name__ == "__main__":
    main()
