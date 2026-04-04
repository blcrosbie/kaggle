#!/usr/bin/env python3
"""Generate a Stage 1 submission using men coach-continuity features."""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_v10_submission import (
    WOMEN_EXPERTS as V10_WOMEN_EXPERTS,
    _blend as v10_blend,
    _expert_hyperparams as v10_params,
)
from generate_v12_ranking_submission import MEN_EXPERTS as BASE_MEN_EXPERTS, _men_hyperparams as base_men_params
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


MEN_EXPERTS = {
    **BASE_MEN_EXPERTS,
    "coach_tournament": [
        "seed_diff",
        "seed_known",
        "favorite_flag",
        "conf_tourney_win_diff",
        "recent_elo_diff",
        "recent_margin_diff",
        "massey_diff",
        "adj_net_eff_diff",
        "coach_tenure_diff",
        "coach_change_diff",
    ],
}

WOMEN_WEIGHTS = {"efficiency_core": 0.5, "tournament_core": 0.5, "power_core": 0.0}


def _men_hyperparams(name: str) -> dict[str, float | int]:
    if name == "coach_tournament":
        return {"epochs": 220, "learning_rate": 0.05, "l2": 0.001}
    return base_men_params(name)


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


def _fit_men_weights(examples: list[TrainingExample], eval_start_season: int) -> tuple[dict[str, float], dict[str, Any]]:
    seasons = sorted({example.season for example in examples if example.gender == "men" and example.season >= eval_start_season})
    preds_by_season: dict[int, dict[str, list[float]]] = {}
    targets_by_season: dict[int, list[float]] = {}
    for season in seasons:
        train_rows = [example for example in examples if example.gender == "men" and example.season < season]
        test_rows = [example for example in examples if example.gender == "men" and example.season == season]
        if not train_rows or not test_rows:
            continue
        season_preds: dict[str, list[float]] = {}
        for expert_name, feature_names in MEN_EXPERTS.items():
            model = train_logistic_model(train_rows, feature_names=feature_names, **_men_hyperparams(expert_name))
            season_preds[expert_name] = [model.predict_prob(example.features) for example in test_rows]
        preds_by_season[season] = season_preds
        targets_by_season[season] = [example.label for example in test_rows]

    expert_names = list(MEN_EXPERTS.keys())
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


def _evaluate_women(examples: list[TrainingExample], eval_start_season: int) -> dict[str, Any]:
    seasons = sorted({example.season for example in examples if example.gender == "women" and example.season >= eval_start_season})
    all_preds: list[float] = []
    all_targets: list[float] = []
    per_season: list[dict[str, Any]] = []
    expert_names = list(V10_WOMEN_EXPERTS.keys())
    for season in seasons:
        train_rows = [example for example in examples if example.gender == "women" and example.season < season]
        test_rows = [example for example in examples if example.gender == "women" and example.season == season]
        if not train_rows or not test_rows:
            continue
        models = {
            name: train_logistic_model(train_rows, feature_names=feature_names, **v10_params(name))
            for name, feature_names in V10_WOMEN_EXPERTS.items()
        }
        targets = [example.label for example in test_rows]
        preds = []
        for example in test_rows:
            probs = [models[name].predict_prob(example.features) for name in expert_names]
            preds.append(v10_blend(probs, [WOMEN_WEIGHTS[name] for name in expert_names]))
        per_season.append({"season": season, "examples": len(targets), "brier": round(brier_score(preds, targets), 6)})
        all_preds.extend(preds)
        all_targets.extend(targets)
    return {"overall_brier": round(brier_score(all_preds, all_targets), 6), "seasons": per_season}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using v15 men-coach ensembles.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v15.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    men_weights, men_eval = _fit_men_weights(examples, args.eval_start_season)
    women_eval = _evaluate_women(examples, args.eval_start_season)

    model_cache: dict[tuple[str, int, str], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in read_submission_rows(args.sample_submission):
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        if gender == "men":
            expert_names = list(MEN_EXPERTS.keys())
            probs: list[float] = []
            for expert_name in expert_names:
                key = (gender, season, expert_name)
                if key not in model_cache:
                    train_rows = [example for example in examples if example.gender == gender and example.season < season]
                    model_cache[key] = train_logistic_model(
                        train_rows,
                        feature_names=MEN_EXPERTS[expert_name],
                        **_men_hyperparams(expert_name),
                    )
                probs.append(model_cache[key].predict_prob(features))
            pred = _blend(probs, [men_weights[name] for name in expert_names])
        else:
            expert_names = list(V10_WOMEN_EXPERTS.keys())
            probs: list[float] = []
            for expert_name in expert_names:
                key = (gender, season, expert_name)
                if key not in model_cache:
                    train_rows = [example for example in examples if example.gender == gender and example.season < season]
                    model_cache[key] = train_logistic_model(
                        train_rows,
                        feature_names=V10_WOMEN_EXPERTS[expert_name],
                        **v10_params(expert_name),
                    )
                probs.append(model_cache[key].predict_prob(features))
            pred = v10_blend(probs, [WOMEN_WEIGHTS[name] for name in expert_names])
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
        "model": "v15_men_coach_features_v10_women",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "men_experts": MEN_EXPERTS,
        "men_weights": men_weights,
        "women_experts": V10_WOMEN_EXPERTS,
        "women_weights": WOMEN_WEIGHTS,
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
    print(f"Walk-forward Brier: {overall_brier}")


if __name__ == "__main__":
    main()
