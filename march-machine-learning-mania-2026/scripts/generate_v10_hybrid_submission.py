#!/usr/bin/env python3
"""Generate a Stage 1 submission using v5 men and v10 women."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from generate_v10_submission import (
    WOMEN_EXPERTS as V10_WOMEN_EXPERTS,
    _blend as v10_blend,
    _expert_hyperparams as v10_params,
)
from generate_v5_submission import (
    EXPERT_FEATURES as V5_EXPERTS,
    _expert_hyperparams as v5_params,
    _weighted_average as v5_blend,
)
from matchup_model_utils import (
    brier_score,
    build_matchup_features,
    build_profiles,
    build_training_examples,
    read_submission_rows,
    train_logistic_model,
    write_json,
    write_submission,
)


V5_MEN_WEIGHTS = {"full": 0.0, "power": 0.1, "efficiency": 0.6, "tournament_shape": 0.3}
V10_WOMEN_WEIGHTS = {"efficiency_core": 0.5, "tournament_core": 0.5, "power_core": 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using a v5-men / v10-women hybrid.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v10_hybrid.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def _evaluate_v5_men(examples, eval_start_season: int) -> tuple[float, list[dict[str, float | int]]]:
    seasons = sorted({example.season for example in examples if example.gender == "men" and example.season >= eval_start_season})
    all_preds: list[float] = []
    all_targets: list[float] = []
    per_season: list[dict[str, float | int]] = []
    for season in seasons:
        train_rows = [example for example in examples if example.gender == "men" and example.season < season]
        test_rows = [example for example in examples if example.season == season and example.gender == "men"]
        if not train_rows or not test_rows:
            continue
        models = {
            name: train_logistic_model(train_rows, feature_names=feature_names, **v5_params(name))
            for name, feature_names in V5_EXPERTS.items()
        }
        preds = []
        targets = [example.label for example in test_rows]
        for example in test_rows:
            expert_probs = [models[name].predict_prob(example.features) for name in V5_EXPERTS]
            preds.append(v5_blend(expert_probs, [V5_MEN_WEIGHTS[name] for name in V5_EXPERTS]))
        per_season.append({"season": season, "examples": len(targets), "brier": round(brier_score(preds, targets), 6)})
        all_preds.extend(preds)
        all_targets.extend(targets)
    return round(brier_score(all_preds, all_targets), 6), per_season


def _evaluate_v10_women(examples, eval_start_season: int) -> tuple[float, list[dict[str, float | int]]]:
    seasons = sorted({example.season for example in examples if example.gender == "women" and example.season >= eval_start_season})
    all_preds: list[float] = []
    all_targets: list[float] = []
    per_season: list[dict[str, float | int]] = []
    for season in seasons:
        train_rows = [example for example in examples if example.gender == "women" and example.season < season]
        test_rows = [example for example in examples if example.gender == "women" and example.season == season]
        if not train_rows or not test_rows:
            continue
        models = {
            name: train_logistic_model(train_rows, feature_names=feature_names, **v10_params(name))
            for name, feature_names in V10_WOMEN_EXPERTS.items()
        }
        preds = []
        targets = [example.label for example in test_rows]
        for example in test_rows:
            expert_probs = [models[name].predict_prob(example.features) for name in V10_WOMEN_EXPERTS]
            preds.append(v10_blend(expert_probs, [V10_WOMEN_WEIGHTS[name] for name in V10_WOMEN_EXPERTS]))
        per_season.append({"season": season, "examples": len(targets), "brier": round(brier_score(preds, targets), 6)})
        all_preds.extend(preds)
        all_targets.extend(targets)
    return round(brier_score(all_preds, all_targets), 6), per_season


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    men_brier, men_seasons = _evaluate_v5_men(examples, args.eval_start_season)
    women_brier, women_seasons = _evaluate_v10_women(examples, args.eval_start_season)
    men_examples = sum(row["examples"] for row in men_seasons)
    women_examples = sum(row["examples"] for row in women_seasons)
    overall_brier = round(
        (men_brier * men_examples + women_brier * women_examples) / max(men_examples + women_examples, 1),
        6,
    )

    model_cache: dict[tuple[str, int, str], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in read_submission_rows(args.sample_submission):
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        if gender == "men":
            expert_names = list(V5_EXPERTS.keys())
            expert_probs = []
            for expert_name in expert_names:
                key = (gender, season, expert_name)
                if key not in model_cache:
                    train_rows = [
                        example for example in examples if example.gender == gender and example.season < season
                    ]
                    model_cache[key] = train_logistic_model(
                        train_rows,
                        feature_names=V5_EXPERTS[expert_name],
                        **v5_params(expert_name),
                    )
                expert_probs.append(model_cache[key].predict_prob(features))
            pred = v5_blend(expert_probs, [V5_MEN_WEIGHTS[name] for name in expert_names])
        else:
            expert_names = list(V10_WOMEN_EXPERTS.keys())
            expert_probs = []
            for expert_name in expert_names:
                key = (gender, season, expert_name)
                if key not in model_cache:
                    train_rows = [example for example in examples if example.gender == gender and example.season < season]
                    model_cache[key] = train_logistic_model(
                        train_rows,
                        feature_names=V10_WOMEN_EXPERTS[expert_name],
                        **v10_params(expert_name),
                    )
                expert_probs.append(model_cache[key].predict_prob(features))
            pred = v10_blend(expert_probs, [V10_WOMEN_WEIGHTS[name] for name in expert_names])
        predictions.append((game_id, pred))
        row_counts[f"{gender}_{season}"] += 1

    write_submission(args.output, predictions)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "v10_hybrid_v5_men_v10_women",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "men_model": {
            "source": "v5_weighted_logistic_ensemble",
            "experts": V5_EXPERTS,
            "weights": V5_MEN_WEIGHTS,
            "overall_brier": men_brier,
            "seasons": men_seasons,
        },
        "women_model": {
            "source": "v10_folded_interaction_gender_specific_ensembles",
            "experts": V10_WOMEN_EXPERTS,
            "weights": V10_WOMEN_WEIGHTS,
            "overall_brier": women_brier,
            "seasons": women_seasons,
        },
        "walkforward_evaluation": {"overall_brier": overall_brier},
        "season_gender_row_counts": dict(sorted(row_counts.items())),
    }
    write_json(manifest_path, payload)
    print(f"Submission written to {args.output}")
    print(f"Manifest written to {manifest_path}")
    print(f"Men Brier: {men_brier}")
    print(f"Women Brier: {women_brier}")
    print(f"Walk-forward Brier: {overall_brier}")


if __name__ == "__main__":
    main()
