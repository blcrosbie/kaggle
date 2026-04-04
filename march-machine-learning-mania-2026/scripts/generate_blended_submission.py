#!/usr/bin/env python3
"""Generate a blended Kaggle submission from logistic and boosted stump models."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from matchup_model_utils import (
    FEATURE_NAMES,
    TrainingExample,
    brier_score,
    build_matchup_features,
    build_profiles,
    build_training_examples,
    read_submission_rows,
    train_boosted_stump_model,
    train_logistic_model,
    write_json,
    write_submission,
)


def _walkforward_blend_score(
    examples: list[TrainingExample],
    logistic_weight: float,
    eval_start_season: int,
) -> dict[str, object]:
    seasons = sorted({example.season for example in examples})
    rows: list[dict[str, object]] = []
    all_preds: list[float] = []
    all_targets: list[float] = []
    for season in seasons:
        if season < eval_start_season:
            continue
        train_rows = [example for example in examples if example.season < season]
        test_rows = [example for example in examples if example.season == season]
        if not train_rows or not test_rows:
            continue
        logistic = train_logistic_model(train_rows, feature_names=FEATURE_NAMES)
        stumps = train_boosted_stump_model(train_rows, feature_names=FEATURE_NAMES)
        preds = []
        targets = []
        for example in test_rows:
            lp = logistic.predict_prob(example.features)
            sp = stumps.predict_prob(example.features)
            pred = logistic_weight * lp + (1.0 - logistic_weight) * sp
            preds.append(pred)
            targets.append(example.label)
        rows.append({"season": season, "examples": len(test_rows), "brier": round(brier_score(preds, targets), 6)})
        all_preds.extend(preds)
        all_targets.extend(targets)
    return {"overall_brier": round(brier_score(all_preds, all_targets), 6), "seasons": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a blended Stage 1 submission.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/SampleSubmissionStage1.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/stage1_blended.csv"),
    )
    parser.add_argument("--logistic-weight", type=float, default=0.75)
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)
    evaluation = _walkforward_blend_score(
        examples,
        logistic_weight=args.logistic_weight,
        eval_start_season=args.eval_start_season,
    )

    rows = read_submission_rows(args.sample_submission)
    models: dict[tuple[str, int], tuple[object, object]] = {}
    counts: dict[tuple[str, int], int] = defaultdict(int)
    predictions: list[tuple[str, float]] = []

    for game_id, season, team_a, team_b, gender in rows:
        key = (gender, season)
        if key not in models:
            train_rows = [
                example for example in examples if example.gender == gender and example.season < season
            ]
            logistic = train_logistic_model(train_rows, feature_names=FEATURE_NAMES)
            stumps = train_boosted_stump_model(train_rows, feature_names=FEATURE_NAMES)
            models[key] = (logistic, stumps)
        logistic, stumps = models[key]
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        lp = logistic.predict_prob(features)
        sp = stumps.predict_prob(features)
        pred = args.logistic_weight * lp + (1.0 - args.logistic_weight) * sp
        predictions.append((game_id, pred))
        counts[key] += 1

    write_submission(args.output, predictions)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "blended_logistic_boosted_stumps",
        "logistic_weight": args.logistic_weight,
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "feature_names": FEATURE_NAMES,
        "walkforward_evaluation": evaluation,
        "season_gender_row_counts": {f"{gender}_{season}": count for (gender, season), count in sorted(counts.items())},
    }
    write_json(manifest_path, payload)
    print(f"Submission written to {args.output}")
    print(f"Manifest written to {manifest_path}")
    print(f"Walk-forward Brier: {evaluation['overall_brier']}")


if __name__ == "__main__":
    main()
