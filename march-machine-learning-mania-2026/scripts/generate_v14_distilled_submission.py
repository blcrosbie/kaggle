#!/usr/bin/env python3
"""Generate a Stage 1 submission using distilled per-gender direct models."""

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


MEN_FEATURES = [
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
    "massey_ratio",
    "massey_gap_sq",
    "massey_mean_diff",
    "massey_best_diff",
    "massey_count_diff",
    "massey_median_rank_diff",
    "massey_mean_rank_diff",
    "massey_best_rank_diff",
    "massey_top10_count_diff",
    "massey_top25_count_diff",
    "top_rank_flag_diff",
]

WOMEN_FEATURES = [
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
    "seed_diff",
    "seed_known",
    "seed_strength_diff",
    "favorite_flag",
    "conf_tourney_win_diff",
    "recent_elo_diff",
    "recent_margin_diff",
    "head_to_head_margin",
    "head_to_head_games",
    "games_diff",
    "seed_efficiency_interaction",
    "seed_recent_interaction",
    "head_to_head_common_opp_interaction",
]

MEN_CANDIDATES = [
    {"epochs": 220, "learning_rate": 0.05, "l2": 0.001},
    {"epochs": 260, "learning_rate": 0.045, "l2": 0.001},
    {"epochs": 300, "learning_rate": 0.04, "l2": 0.0015},
    {"epochs": 220, "learning_rate": 0.055, "l2": 0.0008},
]

WOMEN_CANDIDATES = [
    {"epochs": 220, "learning_rate": 0.05, "l2": 0.001},
    {"epochs": 260, "learning_rate": 0.045, "l2": 0.001},
    {"epochs": 220, "learning_rate": 0.055, "l2": 0.0012},
]


def _select_params(
    examples: list[TrainingExample],
    gender: str,
    feature_names: list[str],
    candidates: list[dict[str, float | int]],
    eval_start_season: int,
) -> tuple[dict[str, float | int], dict[str, Any]]:
    seasons = sorted({example.season for example in examples if example.gender == gender and example.season >= eval_start_season})
    best_params = candidates[0]
    best_score = float("inf")
    best_per_season: list[dict[str, Any]] = []

    for params in candidates:
        all_preds: list[float] = []
        all_targets: list[float] = []
        per_season: list[dict[str, Any]] = []
        for season in seasons:
            train_rows = [example for example in examples if example.gender == gender and example.season < season]
            test_rows = [example for example in examples if example.gender == gender and example.season == season]
            if not train_rows or not test_rows:
                continue
            model = train_logistic_model(train_rows, feature_names=feature_names, **params)
            targets = [example.label for example in test_rows]
            preds = [model.predict_prob(example.features) for example in test_rows]
            per_season.append({"season": season, "examples": len(targets), "brier": round(brier_score(preds, targets), 6)})
            all_preds.extend(preds)
            all_targets.extend(targets)
        score = brier_score(all_preds, all_targets)
        if score < best_score:
            best_score = score
            best_params = params
            best_per_season = per_season

    return best_params, {"overall_brier": round(best_score, 6), "seasons": best_per_season}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using v14 distilled per-gender direct models.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v14.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    men_params, men_eval = _select_params(examples, "men", MEN_FEATURES, MEN_CANDIDATES, args.eval_start_season)
    women_params, women_eval = _select_params(examples, "women", WOMEN_FEATURES, WOMEN_CANDIDATES, args.eval_start_season)

    model_cache: dict[tuple[str, int], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in read_submission_rows(args.sample_submission):
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        key = (gender, season)
        if key not in model_cache:
            train_rows = [example for example in examples if example.gender == gender and example.season < season]
            if gender == "men":
                model_cache[key] = train_logistic_model(train_rows, feature_names=MEN_FEATURES, **men_params)
            else:
                model_cache[key] = train_logistic_model(train_rows, feature_names=WOMEN_FEATURES, **women_params)
        pred = model_cache[key].predict_prob(features)
        predictions.append((game_id, pred))
        row_counts[f"{gender}_{season}"] += 1

    men_examples = sum(row["examples"] for row in men_eval["seasons"])
    women_examples = sum(row["examples"] for row in women_eval["seasons"])
    overall_brier = round(
        (men_eval["overall_brier"] * men_examples + women_eval["overall_brier"] * women_examples)
        / max(men_examples + women_examples, 1),
        6,
    )

    write_submission(args.output, predictions)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "v14_distilled_direct_models",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "men_features": MEN_FEATURES,
        "women_features": WOMEN_FEATURES,
        "men_params": men_params,
        "women_params": women_params,
        "walkforward_evaluation": {
            "overall_brier": overall_brier,
            "men": men_eval,
            "women": women_eval,
        },
        "season_gender_row_counts": dict(sorted(row_counts.items())),
    }
    write_json(manifest_path, payload)
    print(f"Submission written to {args.output}")
    print(f"Manifest written to {manifest_path}")
    print(f"Men params: {men_params}")
    print(f"Women params: {women_params}")
    print(f"Walk-forward Brier: {overall_brier}")


if __name__ == "__main__":
    main()
