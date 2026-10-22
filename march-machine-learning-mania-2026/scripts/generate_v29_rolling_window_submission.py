#!/usr/bin/env python3
"""Generate a Stage 1 submission using rolling-window v19-style men ensembles."""

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
from generate_v19_conf_tourney_submission import MEN_EXPERTS as V19_MEN_EXPERTS, _men_hyperparams as v19_params
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


WOMEN_WEIGHTS = {"efficiency_core": 0.5, "tournament_core": 0.5, "power_core": 0.0}
WINDOW_OPTIONS = [8, 10, 12, 16, 99]


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


def _rows_for_window(rows: list[TrainingExample], season: int, window: int) -> list[TrainingExample]:
    if window >= 99:
        return rows
    start_season = season - window
    return [row for row in rows if row.season >= start_season]


def _score_subset(rows: list[dict[str, Any]], seasons: set[int]) -> float:
    preds: list[float] = []
    targets: list[float] = []
    for row in rows:
        if int(row["season"]) not in seasons:
            continue
        preds.extend(row["preds"])
        targets.extend(row["targets"])
    return round(brier_score(preds, targets), 6)


def _fit_men_config(examples: list[TrainingExample], eval_start_season: int) -> tuple[int, dict[str, float], dict[str, Any]]:
    seasons = sorted({example.season for example in examples if example.gender == "men" and example.season >= eval_start_season})
    selection_seasons = {season for season in seasons if season <= 2022}
    holdout_seasons = {season for season in seasons if season >= 2023}
    expert_names = list(V19_MEN_EXPERTS.keys())
    candidates = _weight_vectors(len(expert_names), step=5)

    best_window = WINDOW_OPTIONS[0]
    best_weights = candidates[0]
    best_selection = float("inf")
    best_eval: dict[str, Any] = {}

    for window in WINDOW_OPTIONS:
        preds_by_season: dict[int, dict[str, list[float]]] = {}
        targets_by_season: dict[int, list[float]] = {}
        for season in seasons:
            full_train = [example for example in examples if example.gender == "men" and example.season < season]
            train_rows = _rows_for_window(full_train, season, window)
            test_rows = [example for example in examples if example.gender == "men" and example.season == season]
            if not train_rows or not test_rows:
                continue
            season_preds: dict[str, list[float]] = {}
            for expert_name, feature_names in V19_MEN_EXPERTS.items():
                model = train_logistic_model(train_rows, feature_names=feature_names, **v19_params(expert_name))
                season_preds[expert_name] = [model.predict_prob(example.features) for example in test_rows]
            preds_by_season[season] = season_preds
            targets_by_season[season] = [example.label for example in test_rows]

        for weights in candidates:
            detailed_rows: list[dict[str, Any]] = []
            for season, season_preds in preds_by_season.items():
                targets = targets_by_season[season]
                preds = [_blend([season_preds[name][index] for name in expert_names], weights) for index in range(len(targets))]
                detailed_rows.append({"season": season, "preds": preds, "targets": targets})
            selection_brier = _score_subset(detailed_rows, selection_seasons)
            if selection_brier < best_selection:
                best_selection = selection_brier
                best_window = window
                best_weights = weights
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

    return best_window, {name: weight for name, weight in zip(expert_names, best_weights)}, best_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using rolling-window v19-style men ensembles.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v29.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    men_window, men_weights, men_eval = _fit_men_config(examples, args.eval_start_season)
    women_eval, women_calibrators = _evaluate_gender_nested(examples, "women", args.eval_start_season)
    final_women_calibrator = women_calibrators[-1][1] if women_calibrators else {"alpha": 1.0, "beta": 0.0}

    model_cache: dict[tuple[str, int, str], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in read_submission_rows(args.sample_submission):
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        if gender == "men":
            experts = V19_MEN_EXPERTS
            expert_names = list(experts.keys())
            probs: list[float] = []
            for expert_name in expert_names:
                key = (gender, season, expert_name)
                if key not in model_cache:
                    full_train = [example for example in examples if example.gender == gender and example.season < season]
                    train_rows = _rows_for_window(full_train, season, men_window)
                    model_cache[key] = train_logistic_model(
                        train_rows,
                        feature_names=experts[expert_name],
                        **v19_params(expert_name),
                    )
                probs.append(model_cache[key].predict_prob(features))
            pred = _blend(probs, [men_weights[name] for name in expert_names])
        else:
            experts = V10_WOMEN_EXPERTS
            expert_names = list(experts.keys())
            probs = []
            for expert_name in expert_names:
                key = (gender, season, expert_name)
                if key not in model_cache:
                    train_rows = [example for example in examples if example.gender == gender and example.season < season]
                    model_cache[key] = train_logistic_model(
                        train_rows,
                        feature_names=experts[expert_name],
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
        "model": "v29_rolling_window_v19_men_nested_calibrated_women",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "selected_men_window": men_window,
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
    print(f"Selected men window: {men_window}")
    print(f"Men weights: {men_weights}")
    print(f"Walk-forward Brier: {overall_brier}")


if __name__ == "__main__":
    main()
