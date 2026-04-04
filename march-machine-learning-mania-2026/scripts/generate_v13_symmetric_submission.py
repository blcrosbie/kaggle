#!/usr/bin/env python3
"""Generate a Stage 1 submission using symmetry-augmented training."""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_v10_submission import (
    WOMEN_EXPERTS as V10_WOMEN_EXPERTS,
    _expert_hyperparams as v10_params,
)
from generate_v12_ranking_submission import MEN_EXPERTS as V12_MEN_EXPERTS, _men_hyperparams as v12_params
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


def _augment_examples(
    examples: list[TrainingExample],
    profiles,
) -> list[TrainingExample]:
    augmented: list[TrainingExample] = []
    for example in examples:
        augmented.append(example)
        swapped_features = build_matchup_features(
            profiles,
            example.gender,
            example.season,
            example.team_b,
            example.team_a,
        )
        augmented.append(
            TrainingExample(
                season=example.season,
                gender=example.gender,
                team_a=example.team_b,
                team_b=example.team_a,
                label=1.0 - example.label,
                features=swapped_features,
            )
        )
    return augmented


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
    profiles,
    gender: str,
    experts: dict[str, list[str]],
    hyperparam_fn,
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
        augmented_train = _augment_examples(train_rows, profiles)
        season_preds: dict[str, list[float]] = {}
        for expert_name, feature_names in experts.items():
            model = train_logistic_model(augmented_train, feature_names=feature_names, **hyperparam_fn(expert_name))
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
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using v13 symmetry-augmented training.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v13.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    men_weights, men_eval = _fit_gender_weights(examples, profiles, "men", V12_MEN_EXPERTS, v12_params, args.eval_start_season)
    women_weights, women_eval = _fit_gender_weights(examples, profiles, "women", V10_WOMEN_EXPERTS, v10_params, args.eval_start_season)

    model_cache: dict[tuple[str, int, str], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in read_submission_rows(args.sample_submission):
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        if gender == "men":
            experts = V12_MEN_EXPERTS
            weights = men_weights
            hyperparam_fn = v12_params
        else:
            experts = V10_WOMEN_EXPERTS
            weights = women_weights
            hyperparam_fn = v10_params
        expert_names = list(experts.keys())
        probs: list[float] = []
        for expert_name in expert_names:
            key = (gender, season, expert_name)
            if key not in model_cache:
                train_rows = [example for example in examples if example.gender == gender and example.season < season]
                augmented_train = _augment_examples(train_rows, profiles)
                model_cache[key] = train_logistic_model(
                    augmented_train,
                    feature_names=experts[expert_name],
                    **hyperparam_fn(expert_name),
                )
            probs.append(model_cache[key].predict_prob(features))
        pred = _blend(probs, [weights[name] for name in expert_names])
        predictions.append((game_id, pred))
        row_counts[f"{gender}_{season}"] += 1

    total_examples = sum(row["examples"] for row in men_eval["seasons"]) + sum(row["examples"] for row in women_eval["seasons"])
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
        "model": "v13_symmetric_training",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "men_experts": V12_MEN_EXPERTS,
        "women_experts": V10_WOMEN_EXPERTS,
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
