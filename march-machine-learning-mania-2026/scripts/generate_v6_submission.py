#!/usr/bin/env python3
"""Generate a Stage 1 submission using a refined v6 logistic ensemble."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matchup_model_utils import (
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
    "efficiency_plus": [
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
        "common_opp_margin_diff",
        "common_opp_win_diff",
        "common_opp_count",
    ],
    "tournament_plus": [
        "seed_diff",
        "seed_known",
        "seed_strength_diff",
        "favorite_flag",
        "massey_diff",
        "massey_known",
        "conf_tourney_win_diff",
        "recent_elo_diff",
        "recent_margin_diff",
        "recent_win_pct_diff",
        "head_to_head_margin",
        "head_to_head_games",
        "games_diff",
    ],
    "power_plus": [
        "elo_diff",
        "recent_elo_diff",
        "win_pct_diff",
        "recent_win_pct_diff",
        "scoring_margin_diff",
        "recent_margin_diff",
        "sos_elo_diff",
        "conf_elo_diff",
        "seed_strength_diff",
        "massey_diff",
        "adj_net_eff_diff",
        "head_to_head_margin",
        "common_opp_margin_diff",
    ],
}


def _expert_hyperparams(name: str) -> dict[str, float | int]:
    if name == "efficiency_plus":
        return {"epochs": 260, "learning_rate": 0.052, "l2": 0.001}
    if name == "tournament_plus":
        return {"epochs": 220, "learning_rate": 0.058, "l2": 0.0012}
    return {"epochs": 220, "learning_rate": 0.055, "l2": 0.0009}


def _candidate_weight_vectors() -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for eff in range(45, 71, 5):
        for tourn in range(20, 41, 5):
            power = 100 - eff - tourn
            if power < 5 or power > 25:
                continue
            candidates.append(
                {
                    "efficiency_plus": eff / 100.0,
                    "tournament_plus": tourn / 100.0,
                    "power_plus": power / 100.0,
                }
            )
    return candidates


def _blend(values: dict[str, float], weights: dict[str, float]) -> float:
    total = sum(values[name] * weights[name] for name in weights)
    return min(0.999, max(0.001, total))


def _walkforward_predictions(
    examples: list[TrainingExample],
    eval_start_season: int,
) -> tuple[dict[int, dict[str, list[float]]], dict[int, list[float]]]:
    seasons = sorted({example.season for example in examples if example.season >= eval_start_season})
    preds_by_season: dict[int, dict[str, list[float]]] = {}
    targets_by_season: dict[int, list[float]] = {}
    for season in seasons:
        train_rows = [example for example in examples if example.season < season]
        test_rows = [example for example in examples if example.season == season]
        if not train_rows or not test_rows:
            continue
        season_preds: dict[str, list[float]] = {}
        for expert_name, feature_names in EXPERT_FEATURES.items():
            params = _expert_hyperparams(expert_name)
            model = train_logistic_model(train_rows, feature_names=feature_names, **params)
            season_preds[expert_name] = [model.predict_prob(example.features) for example in test_rows]
        preds_by_season[season] = season_preds
        targets_by_season[season] = [example.label for example in test_rows]
    return preds_by_season, targets_by_season


def _fit_weights(
    preds_by_season: dict[int, dict[str, list[float]]],
    targets_by_season: dict[int, list[float]],
) -> tuple[dict[str, float], dict[str, Any]]:
    best_weights = _candidate_weight_vectors()[0]
    best_score = float("inf")
    for weights in _candidate_weight_vectors():
        all_preds: list[float] = []
        all_targets: list[float] = []
        for season, season_preds in preds_by_season.items():
            season_targets = targets_by_season[season]
            for index in range(len(season_targets)):
                values = {name: season_preds[name][index] for name in EXPERT_FEATURES}
                all_preds.append(_blend(values, weights))
            all_targets.extend(season_targets)
        score = brier_score(all_preds, all_targets)
        if score < best_score:
            best_score = score
            best_weights = weights

    per_season: list[dict[str, Any]] = []
    all_preds = []
    all_targets = []
    for season, season_preds in preds_by_season.items():
        season_targets = targets_by_season[season]
        season_blend = []
        for index in range(len(season_targets)):
            values = {name: season_preds[name][index] for name in EXPERT_FEATURES}
            season_blend.append(_blend(values, best_weights))
        season_score = brier_score(season_blend, season_targets)
        per_season.append({"season": season, "examples": len(season_targets), "brier": round(season_score, 6)})
        all_preds.extend(season_blend)
        all_targets.extend(season_targets)

    return best_weights, {"overall_brier": round(brier_score(all_preds, all_targets), 6), "seasons": per_season}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using the v6 refined ensemble.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v6.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)
    preds_by_season, targets_by_season = _walkforward_predictions(examples, args.eval_start_season)
    best_weights, evaluation = _fit_weights(preds_by_season, targets_by_season)

    rows = read_submission_rows(args.sample_submission)
    model_cache: dict[tuple[str, int, str], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in rows:
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        expert_probs: dict[str, float] = {}
        for expert_name, feature_names in EXPERT_FEATURES.items():
            key = (gender, season, expert_name)
            if key not in model_cache:
                train_rows = [example for example in examples if example.gender == gender and example.season < season]
                params = _expert_hyperparams(expert_name)
                model_cache[key] = train_logistic_model(train_rows, feature_names=feature_names, **params)
            model = model_cache[key]
            expert_probs[expert_name] = model.predict_prob(features)
        pred = _blend(expert_probs, best_weights)
        predictions.append((game_id, pred))
        row_counts[f"{gender}_{season}"] += 1

    write_submission(args.output, predictions)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "v6_refined_logistic_ensemble",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "experts": EXPERT_FEATURES,
        "meta_weights": best_weights,
        "walkforward_evaluation": evaluation,
        "season_gender_row_counts": dict(sorted(row_counts.items())),
    }
    write_json(manifest_path, payload)
    print(f"Submission written to {args.output}")
    print(f"Manifest written to {manifest_path}")
    print(f"Meta weights: {best_weights}")
    print(f"Walk-forward Brier: {evaluation['overall_brier']}")


if __name__ == "__main__":
    main()
