#!/usr/bin/env python3
"""Generate a Stage 1 submission using nested-calibrated v19 men and nested-calibrated v10 women."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_v10_submission import (
    WOMEN_EXPERTS as V10_WOMEN_EXPERTS,
    _blend as v10_blend,
    _expert_hyperparams as v10_params,
)
from generate_v17_nested_calibrated_submission import _calibrate, _search_calibration
from generate_v19_conf_tourney_submission import MEN_EXPERTS as V19_MEN_EXPERTS, _blend as v19_blend, _men_hyperparams as v19_params
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


MEN_WEIGHTS = {"efficiency_plus": 0.4, "tournament_plus": 0.05, "conference_tourney": 0.55}
WOMEN_WEIGHTS = {"efficiency_core": 0.5, "tournament_core": 0.5, "power_core": 0.0}


def _evaluate_gender_nested(
    examples: list[TrainingExample],
    gender: str,
    eval_start_season: int,
) -> tuple[dict[str, Any], list[tuple[int, dict[str, float]]]]:
    seasons = sorted({example.season for example in examples if example.gender == gender and example.season >= eval_start_season})
    history_preds: list[float] = []
    history_targets: list[float] = []
    all_preds: list[float] = []
    all_targets: list[float] = []
    per_season: list[dict[str, Any]] = []
    calibrators_by_season: list[tuple[int, dict[str, float]]] = []

    for season in seasons:
        calibrator = _search_calibration(history_preds, history_targets)
        calibrators_by_season.append((season, calibrator))
        train_rows = [example for example in examples if example.gender == gender and example.season < season]
        test_rows = [example for example in examples if example.gender == gender and example.season == season]
        if not train_rows or not test_rows:
            continue
        if gender == "men":
            experts = V19_MEN_EXPERTS
            weights = MEN_WEIGHTS
            params_fn = v19_params
        else:
            experts = V10_WOMEN_EXPERTS
            weights = WOMEN_WEIGHTS
            params_fn = v10_params
        models = {
            name: train_logistic_model(train_rows, feature_names=feature_names, **params_fn(name))
            for name, feature_names in experts.items()
        }
        expert_names = list(experts.keys())
        targets = [example.label for example in test_rows]
        base_preds = []
        calibrated_preds = []
        for example in test_rows:
            probs = [models[name].predict_prob(example.features) for name in expert_names]
            if gender == "men":
                base_pred = v19_blend(probs, [weights[name] for name in expert_names])
            else:
                base_pred = v10_blend(probs, [weights[name] for name in expert_names])
            base_preds.append(base_pred)
            calibrated_preds.append(_calibrate(base_pred, calibrator["alpha"], calibrator["beta"]))
        per_season.append({"season": season, "examples": len(targets), "brier": round(brier_score(calibrated_preds, targets), 6)})
        all_preds.extend(calibrated_preds)
        all_targets.extend(targets)
        history_preds.extend(base_preds)
        history_targets.extend(targets)

    return {"overall_brier": round(brier_score(all_preds, all_targets), 6), "seasons": per_season}, calibrators_by_season


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using nested-calibrated v19 men and nested-calibrated v10 women.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v23.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    men_eval, men_calibrators = _evaluate_gender_nested(examples, "men", args.eval_start_season)
    women_eval, women_calibrators = _evaluate_gender_nested(examples, "women", args.eval_start_season)
    final_calibrators = {
        "men": men_calibrators[-1][1] if men_calibrators else {"alpha": 1.0, "beta": 0.0},
        "women": women_calibrators[-1][1] if women_calibrators else {"alpha": 1.0, "beta": 0.0},
    }

    model_cache: dict[tuple[str, int, str], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in read_submission_rows(args.sample_submission):
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        if gender == "men":
            experts = V19_MEN_EXPERTS
            weights = MEN_WEIGHTS
            params_fn = v19_params
        else:
            experts = V10_WOMEN_EXPERTS
            weights = WOMEN_WEIGHTS
            params_fn = v10_params
        expert_names = list(experts.keys())
        probs: list[float] = []
        for expert_name in expert_names:
            key = (gender, season, expert_name)
            if key not in model_cache:
                train_rows = [example for example in examples if example.gender == gender and example.season < season]
                model_cache[key] = train_logistic_model(
                    train_rows,
                    feature_names=experts[expert_name],
                    **params_fn(expert_name),
                )
            probs.append(model_cache[key].predict_prob(features))
        if gender == "men":
            pred = v19_blend(probs, [weights[name] for name in expert_names])
        else:
            pred = v10_blend(probs, [weights[name] for name in expert_names])
        calibrator = final_calibrators[gender]
        pred = _calibrate(pred, calibrator["alpha"], calibrator["beta"])
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
        "model": "v23_nested_calibrated_v19_men_v10_women",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "men_weights": MEN_WEIGHTS,
        "women_weights": WOMEN_WEIGHTS,
        "final_calibrators": final_calibrators,
        "men_nested_calibrators": [{"season": season, **params} for season, params in men_calibrators],
        "women_nested_calibrators": [{"season": season, **params} for season, params in women_calibrators],
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
    print(f"Final calibrators: {final_calibrators}")
    print(f"Walk-forward Brier: {overall_brier}")


if __name__ == "__main__":
    main()
