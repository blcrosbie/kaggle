#!/usr/bin/env python3
"""Generate a Stage 1 submission using calibrated v12-men / v10-women probabilities."""

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


MEN_WEIGHTS = {"power": 0.0, "efficiency": 0.55, "tournament_shape": 0.45, "ranking_consensus": 0.0}
WOMEN_WEIGHTS = {"efficiency_core": 0.5, "tournament_core": 0.5, "power_core": 0.0}


def _blend(values: list[float], weights: list[float]) -> float:
    return min(0.999, max(0.001, sum(v * w for v, w in zip(values, weights))))


def _calibrate(prob: float, alpha: float, beta: float) -> float:
    return min(0.999, max(0.001, 0.5 + alpha * (prob - 0.5) + beta))


def _search_calibration(preds: list[float], targets: list[float]) -> dict[str, float]:
    best = {"alpha": 1.0, "beta": 0.0}
    best_score = brier_score(preds, targets)
    for alpha_i in range(60, 121, 5):
        alpha = alpha_i / 100.0
        for beta_i in range(-5, 6):
            beta = beta_i / 100.0
            calibrated = [_calibrate(pred, alpha, beta) for pred in preds]
            score = brier_score(calibrated, targets)
            if score < best_score:
                best_score = score
                best = {"alpha": alpha, "beta": beta}
    return best


def _walkforward_base_predictions(examples: list[TrainingExample], eval_start_season: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, float]]]:
    seasons = sorted({example.season for example in examples if example.season >= eval_start_season})
    by_gender = {"men": {"preds": [], "targets": [], "seasons": []}, "women": {"preds": [], "targets": [], "seasons": []}}
    for season in seasons:
        for gender in ("men", "women"):
            train_rows = [example for example in examples if example.gender == gender and example.season < season]
            test_rows = [example for example in examples if example.gender == gender and example.season == season]
            if not train_rows or not test_rows:
                continue
            if gender == "men":
                experts = V12_MEN_EXPERTS
                weights = MEN_WEIGHTS
                params_fn = v12_params
            else:
                experts = V10_WOMEN_EXPERTS
                weights = WOMEN_WEIGHTS
                params_fn = v10_params
            models = {
                name: train_logistic_model(train_rows, feature_names=feature_names, **params_fn(name))
                for name, feature_names in experts.items()
            }
            expert_names = list(experts.keys())
            preds = []
            targets = [example.label for example in test_rows]
            for example in test_rows:
                probs = [models[name].predict_prob(example.features) for name in expert_names]
                if gender == "men":
                    preds.append(_blend(probs, [weights[name] for name in expert_names]))
                else:
                    preds.append(v10_blend(probs, [weights[name] for name in expert_names]))
            by_gender[gender]["preds"].extend(preds)
            by_gender[gender]["targets"].extend(targets)
            by_gender[gender]["seasons"].append({"season": season, "examples": len(targets), "brier": round(brier_score(preds, targets), 6)})

    calibrators = {
        gender: _search_calibration(by_gender[gender]["preds"], by_gender[gender]["targets"])
        for gender in ("men", "women")
    }

    eval_payload = {}
    for gender in ("men", "women"):
        calibrated = [
            _calibrate(pred, calibrators[gender]["alpha"], calibrators[gender]["beta"])
            for pred in by_gender[gender]["preds"]
        ]
        eval_payload[gender] = {
            "overall_brier": round(brier_score(calibrated, by_gender[gender]["targets"]), 6),
            "seasons": by_gender[gender]["seasons"],
        }
    return eval_payload["men"], eval_payload["women"], calibrators


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using v16 calibrated probabilities.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v16.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    men_eval, women_eval, calibrators = _walkforward_base_predictions(examples, args.eval_start_season)

    model_cache: dict[tuple[str, int, str], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in read_submission_rows(args.sample_submission):
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        if gender == "men":
            experts = V12_MEN_EXPERTS
            weights = MEN_WEIGHTS
            params_fn = v12_params
            calibrator = calibrators["men"]
        else:
            experts = V10_WOMEN_EXPERTS
            weights = WOMEN_WEIGHTS
            params_fn = v10_params
            calibrator = calibrators["women"]
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
            pred = _blend(probs, [weights[name] for name in expert_names])
        else:
            pred = v10_blend(probs, [weights[name] for name in expert_names])
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
        "model": "v16_calibrated_v12_men_v10_women",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "men_weights": MEN_WEIGHTS,
        "women_weights": WOMEN_WEIGHTS,
        "calibrators": calibrators,
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
    print(f"Calibrators: {calibrators}")
    print(f"Walk-forward Brier: {overall_brier}")


if __name__ == "__main__":
    main()
