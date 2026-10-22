#!/usr/bin/env python3
"""Generate a Stage 1 submission using a weighted ensemble of logistic experts."""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matchup_model_utils import (
    FEATURE_NAMES,
    TrainingExample,
    brier_score,
    build_matchup_features,
    build_profiles,
    build_training_examples,
    read_submission_rows,
    train_logistic_model,
    write_json,
    write_submission,
)


EXPERT_FEATURES: dict[str, list[str]] = {
    "full": FEATURE_NAMES,
    "power": [
        "elo_diff",
        "recent_elo_diff",
        "win_pct_diff",
        "recent_win_pct_diff",
        "scoring_margin_diff",
        "recent_margin_diff",
        "sos_elo_diff",
        "conf_elo_diff",
        "seed_diff",
        "seed_strength_diff",
        "massey_diff",
        "adj_net_eff_diff",
    ],
    "efficiency": [
        "off_eff_diff",
        "def_eff_diff",
        "net_eff_diff",
        "adj_net_eff_diff",
        "efg_diff",
        "tov_rate_diff",
        "orb_rate_diff",
        "ftr_diff",
        "sos_margin_diff",
        "conf_margin_diff",
    ],
    "tournament_shape": [
        "seed_diff",
        "seed_known",
        "seed_strength_diff",
        "favorite_flag",
        "massey_diff",
        "massey_known",
        "conf_tourney_win_diff",
        "recent_elo_diff",
        "recent_margin_diff",
        "games_diff",
    ],
}


def _expert_hyperparams(name: str) -> dict[str, float | int]:
    if name == "full":
        return {"epochs": 250, "learning_rate": 0.05, "l2": 0.001}
    if name == "power":
        return {"epochs": 220, "learning_rate": 0.055, "l2": 0.0008}
    if name == "efficiency":
        return {"epochs": 220, "learning_rate": 0.05, "l2": 0.0012}
    return {"epochs": 180, "learning_rate": 0.06, "l2": 0.0015}


def _generate_weight_vectors(n: int, step: int = 10) -> list[list[float]]:
    buckets = 100 // step
    vectors: list[list[float]] = []
    for combo in itertools.product(range(buckets + 1), repeat=n):
        if sum(combo) != buckets:
            continue
        vectors.append([value / buckets for value in combo])
    return vectors


def _weighted_average(values: list[float], weights: list[float]) -> float:
    total = sum(weight * value for weight, value in zip(weights, values))
    return min(0.999, max(0.001, total))


def _walkforward_expert_predictions(
    examples: list[TrainingExample],
    eval_start_season: int,
) -> tuple[dict[tuple[str, int], dict[str, list[float]]], dict[tuple[str, int], list[float]]]:
    seasons = sorted({example.season for example in examples if example.season >= eval_start_season})
    genders = sorted({example.gender for example in examples})
    predictions_by_season: dict[tuple[str, int], dict[str, list[float]]] = {}
    targets_by_season: dict[tuple[str, int], list[float]] = {}

    for gender in genders:
        for season in seasons:
            train_rows = [example for example in examples if example.gender == gender and example.season < season]
            test_rows = [example for example in examples if example.gender == gender and example.season == season]
            if not train_rows or not test_rows:
                continue
            season_preds: dict[str, list[float]] = {}
            for expert_name, feature_names in EXPERT_FEATURES.items():
                params = _expert_hyperparams(expert_name)
                model = train_logistic_model(train_rows, feature_names=feature_names, **params)
                season_preds[expert_name] = [model.predict_prob(example.features) for example in test_rows]
            predictions_by_season[(gender, season)] = season_preds
            targets_by_season[(gender, season)] = [example.label for example in test_rows]
    return predictions_by_season, targets_by_season


def _fit_meta_weights(
    predictions_by_season: dict[tuple[str, int], dict[str, list[float]]],
    targets_by_season: dict[tuple[str, int], list[float]],
) -> tuple[dict[str, float], dict[str, Any]]:
    expert_names = list(EXPERT_FEATURES.keys())
    candidate_weights = _generate_weight_vectors(len(expert_names), step=10)
    best_score = float("inf")
    best_weights = candidate_weights[0]

    for weights in candidate_weights:
        all_preds: list[float] = []
        all_targets: list[float] = []
        for season_key, expert_preds in predictions_by_season.items():
            season_targets = targets_by_season[season_key]
            for index in range(len(season_targets)):
                preds = [expert_preds[name][index] for name in expert_names]
                all_preds.append(_weighted_average(preds, weights))
            all_targets.extend(season_targets)
        score = brier_score(all_preds, all_targets)
        if score < best_score:
            best_score = score
            best_weights = weights

    per_season: list[dict[str, Any]] = []
    all_preds = []
    all_targets = []
    for season_key, expert_preds in predictions_by_season.items():
        gender, season = season_key
        season_targets = targets_by_season[season_key]
        season_blend = []
        for index in range(len(season_targets)):
            preds = [expert_preds[name][index] for name in expert_names]
            season_blend.append(_weighted_average(preds, best_weights))
        season_score = brier_score(season_blend, season_targets)
        per_season.append(
            {"gender": gender, "season": season, "examples": len(season_targets), "brier": round(season_score, 6)}
        )
        all_preds.extend(season_blend)
        all_targets.extend(season_targets)

    return (
        {name: weight for name, weight in zip(expert_names, best_weights)},
        {
            "overall_brier": round(brier_score(all_preds, all_targets), 6),
            "seasons": per_season,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using a v5 logistic ensemble.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=Path("data/SampleSubmissionStage1.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions/stage1_v5.csv"),
    )
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    predictions_by_season, targets_by_season = _walkforward_expert_predictions(
        examples,
        eval_start_season=args.eval_start_season,
    )
    meta_weights, evaluation = _fit_meta_weights(predictions_by_season, targets_by_season)

    rows = read_submission_rows(args.sample_submission)
    model_cache: dict[tuple[str, int, str], object] = {}
    row_counts: dict[str, int] = defaultdict(int)
    predictions: list[tuple[str, float]] = []
    expert_names = list(EXPERT_FEATURES.keys())

    for game_id, season, team_a, team_b, gender in rows:
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        expert_probs: list[float] = []
        for expert_name in expert_names:
            cache_key = (gender, season, expert_name)
            if cache_key not in model_cache:
                train_rows = [
                    example
                    for example in examples
                    if example.gender == gender and example.season < season
                ]
                params = _expert_hyperparams(expert_name)
                model_cache[cache_key] = train_logistic_model(
                    train_rows,
                    feature_names=EXPERT_FEATURES[expert_name],
                    **params,
                )
            model = model_cache[cache_key]
            expert_probs.append(model.predict_prob(features))
        pred = _weighted_average(expert_probs, [meta_weights[name] for name in expert_names])
        predictions.append((game_id, pred))
        row_counts[f"{gender}_{season}"] += 1

    write_submission(args.output, predictions)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "v5_weighted_logistic_ensemble",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "experts": EXPERT_FEATURES,
        "meta_weights": meta_weights,
        "walkforward_evaluation": evaluation,
        "season_gender_row_counts": dict(sorted(row_counts.items())),
    }
    write_json(manifest_path, payload)

    print(f"Submission written to {args.output}")
    print(f"Manifest written to {manifest_path}")
    print(f"Meta weights: {meta_weights}")
    print(f"Walk-forward Brier: {evaluation['overall_brier']}")


if __name__ == "__main__":
    main()
