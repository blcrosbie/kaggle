#!/usr/bin/env python3
"""Generate a Stage 1 submission using interaction-heavy gender-specific ensembles."""

from __future__ import annotations

import argparse
import itertools
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


MEN_EXPERTS: dict[str, list[str]] = {
    "efficiency_core": [
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
    "tournament_core": [
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
    ],
    "interaction_core": [
        "seed_efficiency_interaction",
        "seed_recent_interaction",
        "recent_rank_interaction",
        "head_to_head_common_opp_interaction",
        "efficiency_rank_interaction",
        "common_opp_margin_diff",
        "head_to_head_margin",
        "adj_net_eff_diff",
        "recent_elo_diff",
        "seed_strength_diff",
    ],
    "matchup_history": [
        "common_opp_margin_diff",
        "common_opp_win_diff",
        "common_opp_count",
        "head_to_head_margin",
        "head_to_head_games",
        "recent_margin_diff",
        "recent_elo_diff",
        "adj_net_eff_diff",
    ],
}

WOMEN_EXPERTS: dict[str, list[str]] = {
    "efficiency_core": [
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
    "tournament_core": [
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
    ],
    "power_interaction": [
        "elo_diff",
        "recent_elo_diff",
        "win_pct_diff",
        "recent_win_pct_diff",
        "scoring_margin_diff",
        "recent_margin_diff",
        "sos_elo_diff",
        "conf_elo_diff",
        "seed_efficiency_interaction",
        "seed_recent_interaction",
        "head_to_head_common_opp_interaction",
    ],
}


def _expert_hyperparams(name: str) -> dict[str, float | int]:
    if "efficiency" in name:
        return {"epochs": 260, "learning_rate": 0.052, "l2": 0.001}
    if "tournament" in name:
        return {"epochs": 230, "learning_rate": 0.058, "l2": 0.0012}
    if "interaction" in name:
        return {"epochs": 240, "learning_rate": 0.055, "l2": 0.0009}
    return {"epochs": 220, "learning_rate": 0.055, "l2": 0.0009}


def _weight_vectors(n: int, step: int = 5) -> list[list[float]]:
    buckets = 100 // step
    vectors: list[list[float]] = []
    for combo in itertools.product(range(buckets + 1), repeat=n):
        if sum(combo) != buckets:
            continue
        vectors.append([value / buckets for value in combo])
    return vectors


def _blend(values: list[float], weights: list[float]) -> float:
    return min(0.999, max(0.001, sum(v * w for v, w in zip(values, weights))))


def _fit_gender_weights(
    examples: list[TrainingExample],
    gender: str,
    experts: dict[str, list[str]],
    eval_start_season: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    seasons = sorted({example.season for example in examples if example.gender == gender and example.season >= eval_start_season})
    preds_by_season: dict[int, dict[str, list[float]]] = {}
    targets_by_season: dict[int, list[float]] = {}
    for season in seasons:
        train_rows = [example for example in examples if example.gender == gender and example.season < season]
        test_rows = [example for example in examples if example.gender == gender and example.season == season]
        if not train_rows or not test_rows:
            continue
        season_preds: dict[str, list[float]] = {}
        for expert_name, feature_names in experts.items():
            model = train_logistic_model(train_rows, feature_names=feature_names, **_expert_hyperparams(expert_name))
            season_preds[expert_name] = [model.predict_prob(example.features) for example in test_rows]
        preds_by_season[season] = season_preds
        targets_by_season[season] = [example.label for example in test_rows]

    expert_names = list(experts.keys())
    candidates = _weight_vectors(len(expert_names), step=5)
    best_weights = candidates[0]
    best_score = float("inf")
    for weights in candidates:
        all_preds: list[float] = []
        all_targets: list[float] = []
        for season, season_preds in preds_by_season.items():
            targets = targets_by_season[season]
            for index in range(len(targets)):
                all_preds.append(_blend([season_preds[name][index] for name in expert_names], weights))
            all_targets.extend(targets)
        score = brier_score(all_preds, all_targets)
        if score < best_score:
            best_score = score
            best_weights = weights

    per_season: list[dict[str, Any]] = []
    all_preds = []
    all_targets = []
    for season, season_preds in preds_by_season.items():
        targets = targets_by_season[season]
        preds = [_blend([season_preds[name][index] for name in expert_names], best_weights) for index in range(len(targets))]
        per_season.append({"season": season, "examples": len(targets), "brier": round(brier_score(preds, targets), 6)})
        all_preds.extend(preds)
        all_targets.extend(targets)

    return (
        {name: weight for name, weight in zip(expert_names, best_weights)},
        {"overall_brier": round(brier_score(all_preds, all_targets), 6), "seasons": per_season},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using v9 interaction-heavy ensembles.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v9.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    men_weights, men_eval = _fit_gender_weights(examples, "men", MEN_EXPERTS, args.eval_start_season)
    women_weights, women_eval = _fit_gender_weights(examples, "women", WOMEN_EXPERTS, args.eval_start_season)

    model_cache: dict[tuple[str, int, str], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in read_submission_rows(args.sample_submission):
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        if gender == "men":
            experts = MEN_EXPERTS
            weights = men_weights
        else:
            experts = WOMEN_EXPERTS
            weights = women_weights
        expert_names = list(experts.keys())
        probs: list[float] = []
        for expert_name in expert_names:
            key = (gender, season, expert_name)
            if key not in model_cache:
                train_rows = [example for example in examples if example.gender == gender and example.season < season]
                model_cache[key] = train_logistic_model(
                    train_rows,
                    feature_names=experts[expert_name],
                    **_expert_hyperparams(expert_name),
                )
            probs.append(model_cache[key].predict_prob(features))
        pred = _blend(probs, [weights[name] for name in expert_names])
        predictions.append((game_id, pred))
        row_counts[f"{gender}_{season}"] += 1

    overall_rows = men_eval["seasons"] + women_eval["seasons"]
    total_examples = sum(row["examples"] for row in overall_rows)
    overall_brier = round(
        (
            men_eval["overall_brier"] * sum(row["examples"] for row in men_eval["seasons"])
            + women_eval["overall_brier"] * sum(row["examples"] for row in women_eval["seasons"])
        )
        / max(total_examples, 1),
        6,
    )

    write_submission(args.output, predictions)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "v9_interaction_gender_specific_ensembles",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "men_experts": MEN_EXPERTS,
        "women_experts": WOMEN_EXPERTS,
        "men_weights": men_weights,
        "women_weights": women_weights,
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
    print(f"Men weights: {men_weights}")
    print(f"Women weights: {women_weights}")
    print(f"Walk-forward Brier: {overall_brier}")


if __name__ == "__main__":
    main()
