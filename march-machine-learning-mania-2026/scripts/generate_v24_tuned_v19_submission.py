#!/usr/bin/env python3
"""Generate a Stage 1 submission using a holdout-tuned v19-style men ensemble."""

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
from generate_v17_nested_calibrated_submission import _calibrate, _evaluate_gender_nested
from generate_v19_conf_tourney_submission import MEN_EXPERTS as V19_MEN_EXPERTS
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


MEN_PARAM_GRID: dict[str, list[dict[str, float | int]]] = {
    "efficiency_plus": [
        {"epochs": 220, "learning_rate": 0.055, "l2": 0.001},
        {"epochs": 260, "learning_rate": 0.05, "l2": 0.0008},
    ],
    "tournament_plus": [
        {"epochs": 190, "learning_rate": 0.06, "l2": 0.0014},
        {"epochs": 230, "learning_rate": 0.054, "l2": 0.0011},
    ],
    "conference_tourney": [
        {"epochs": 220, "learning_rate": 0.055, "l2": 0.001},
        {"epochs": 260, "learning_rate": 0.05, "l2": 0.0009},
    ],
}

WOMEN_WEIGHTS = {"efficiency_core": 0.5, "tournament_core": 0.5, "power_core": 0.0}


def _weight_vectors(step: int = 5) -> list[list[float]]:
    buckets = 100 // step
    vectors: list[list[float]] = []
    for combo in itertools.product(range(buckets + 1), repeat=3):
        if sum(combo) != buckets:
            continue
        vectors.append([value / buckets for value in combo])
    return vectors


def _blend(values: list[float], weights: list[float]) -> float:
    return min(0.999, max(0.001, sum(v * w for v, w in zip(values, weights))))


def _score_subset(rows: list[dict[str, Any]], seasons: set[int]) -> float:
    preds: list[float] = []
    targets: list[float] = []
    for row in rows:
        if int(row["season"]) not in seasons:
            continue
        preds.extend(row["preds"])
        targets.extend(row["targets"])
    return round(brier_score(preds, targets), 6)


def _select_men_config(examples: list[TrainingExample], eval_start_season: int) -> tuple[dict[str, int], dict[str, float], dict[str, Any]]:
    seasons = sorted({example.season for example in examples if example.gender == "men" and example.season >= eval_start_season})
    selection_seasons = {season for season in seasons if season <= 2022}
    holdout_seasons = {season for season in seasons if season >= 2023}
    expert_names = list(V19_MEN_EXPERTS.keys())

    preds_by_variant: dict[tuple[str, int, int], list[float]] = {}
    targets_by_season: dict[int, list[float]] = {}
    for season in seasons:
        train_rows = [example for example in examples if example.gender == "men" and example.season < season]
        test_rows = [example for example in examples if example.gender == "men" and example.season == season]
        if not train_rows or not test_rows:
            continue
        targets_by_season[season] = [example.label for example in test_rows]
        for expert_name in expert_names:
            for param_index, params in enumerate(MEN_PARAM_GRID[expert_name]):
                model = train_logistic_model(train_rows, feature_names=V19_MEN_EXPERTS[expert_name], **params)
                preds_by_variant[(season, param_index, expert_names.index(expert_name))] = [
                    model.predict_prob(example.features) for example in test_rows
                ]

    weight_vectors = _weight_vectors(step=5)
    best_selection = float("inf")
    best_param_choice = {name: 0 for name in expert_names}
    best_weights = {name: 0.0 for name in expert_names}
    best_eval: dict[str, Any] = {}

    for eff_idx in range(len(MEN_PARAM_GRID["efficiency_plus"])):
        for tour_idx in range(len(MEN_PARAM_GRID["tournament_plus"])):
            for conf_idx in range(len(MEN_PARAM_GRID["conference_tourney"])):
                param_choice = {"efficiency_plus": eff_idx, "tournament_plus": tour_idx, "conference_tourney": conf_idx}
                for weights in weight_vectors:
                    detailed_rows: list[dict[str, Any]] = []
                    for season in seasons:
                        targets = targets_by_season.get(season)
                        if not targets:
                            continue
                        preds = [
                            _blend(
                                [
                                    preds_by_variant[(season, eff_idx, 0)][index],
                                    preds_by_variant[(season, tour_idx, 1)][index],
                                    preds_by_variant[(season, conf_idx, 2)][index],
                                ],
                                weights,
                            )
                            for index in range(len(targets))
                        ]
                        detailed_rows.append({"season": season, "preds": preds, "targets": targets})
                    selection_brier = _score_subset(detailed_rows, selection_seasons)
                    if selection_brier < best_selection:
                        best_selection = selection_brier
                        best_param_choice = param_choice
                        best_weights = {name: weight for name, weight in zip(expert_names, weights)}
                        best_eval = {
                            "overall_brier": round(
                                brier_score(
                                    [pred for row in detailed_rows for pred in row["preds"]],
                                    [target for row in detailed_rows for target in row["targets"]],
                                ),
                                6,
                            ),
                            "selection_brier": selection_brier,
                            "holdout_brier": _score_subset(detailed_rows, holdout_seasons),
                            "seasons": [
                                {"season": row["season"], "examples": len(row["targets"]), "brier": round(brier_score(row["preds"], row["targets"]), 6)}
                                for row in detailed_rows
                            ],
                        }

    return best_param_choice, best_weights, best_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using a holdout-tuned v19-style men ensemble.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v24.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    men_param_choice, men_weights, men_eval = _select_men_config(examples, args.eval_start_season)
    women_eval, women_calibrators = _evaluate_gender_nested(examples, "women", args.eval_start_season)
    final_women_calibrator = women_calibrators[-1][1] if women_calibrators else {"alpha": 1.0, "beta": 0.0}

    model_cache: dict[tuple[str, int, str], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in read_submission_rows(args.sample_submission):
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        if gender == "men":
            probs: list[float] = []
            for expert_name in V19_MEN_EXPERTS:
                key = (gender, season, expert_name)
                if key not in model_cache:
                    train_rows = [example for example in examples if example.gender == gender and example.season < season]
                    model_cache[key] = train_logistic_model(
                        train_rows,
                        feature_names=V19_MEN_EXPERTS[expert_name],
                        **MEN_PARAM_GRID[expert_name][men_param_choice[expert_name]],
                    )
                probs.append(model_cache[key].predict_prob(features))
            pred = _blend(probs, [men_weights[name] for name in V19_MEN_EXPERTS])
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
            pred = _calibrate(pred, final_women_calibrator["alpha"], final_women_calibrator["beta"])
        predictions.append((game_id, pred))
        row_counts[f"{gender}_{season}"] += 1

    men_count = sum(row["examples"] for row in men_eval["seasons"])
    women_count = sum(row["examples"] for row in women_eval["seasons"])
    overall_brier = round(
        (men_eval["overall_brier"] * men_count + women_eval["overall_brier"] * women_count)
        / max(men_count + women_count, 1),
        6,
    )

    write_submission(args.output, predictions)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "v24_tuned_v19_men_nested_calibrated_women",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "selected_men_param_indexes": men_param_choice,
        "selected_men_params": {
            name: MEN_PARAM_GRID[name][index] for name, index in men_param_choice.items()
        },
        "men_weights": men_weights,
        "women_weights": WOMEN_WEIGHTS,
        "final_women_calibrator": final_women_calibrator,
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
    print(f"Selected men params: {payload['selected_men_params']}")
    print(f"Men weights: {men_weights}")
    print(f"Walk-forward Brier: {overall_brier}")


if __name__ == "__main__":
    main()
