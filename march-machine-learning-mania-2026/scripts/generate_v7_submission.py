#!/usr/bin/env python3
"""Generate a Stage 1 submission using a stacked logistic ensemble with calibration."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
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


BASE_EXPERTS: dict[str, list[str]] = {
    "efficiency": [
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
    "tournament": [
        "seed_diff",
        "seed_known",
        "seed_strength_diff",
        "favorite_flag",
        "massey_diff",
        "massey_ratio",
        "massey_gap_sq",
        "top_rank_flag_diff",
        "massey_known",
        "conf_tourney_win_diff",
        "recent_elo_diff",
        "recent_margin_diff",
        "head_to_head_margin",
        "head_to_head_games",
    ],
    "power_rank": [
        "elo_diff",
        "recent_elo_diff",
        "win_pct_diff",
        "recent_win_pct_diff",
        "scoring_margin_diff",
        "recent_margin_diff",
        "sos_elo_diff",
        "conf_elo_diff",
        "massey_diff",
        "massey_ratio",
        "top_rank_flag_diff",
        "adj_net_eff_diff",
        "head_to_head_margin",
        "common_opp_margin_diff",
    ],
}

META_FEATURES = [
    "efficiency_prob",
    "tournament_prob",
    "power_rank_prob",
    "expert_mean",
    "expert_spread",
    "eff_tourn_gap",
    "seed_strength_diff",
    "adj_net_eff_diff",
    "massey_diff",
    "massey_ratio",
    "top_rank_flag_diff",
    "common_opp_margin_diff",
    "head_to_head_margin",
    "recent_elo_diff",
    "gender_is_men",
    "era_early",
    "era_middle",
    "era_modern",
]


@dataclass
class PlattCalibrator:
    a: float
    b: float

    def predict(self, prob: float) -> float:
        clipped = min(0.999999, max(0.000001, prob))
        x = math.log(clipped / (1.0 - clipped))
        total = self.a * x + self.b
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, total))))


def _era_bucket(season: int) -> str:
    if season <= 2002:
        return "early"
    if season <= 2014:
        return "middle"
    return "modern"


def _expert_hyperparams(name: str) -> dict[str, float | int]:
    if name == "efficiency":
        return {"epochs": 260, "learning_rate": 0.052, "l2": 0.001}
    if name == "tournament":
        return {"epochs": 220, "learning_rate": 0.058, "l2": 0.0012}
    return {"epochs": 220, "learning_rate": 0.055, "l2": 0.0009}


def _build_meta_features(expert_probs: dict[str, float], base_features: dict[str, float], gender: str, season: int) -> dict[str, float]:
    probs = list(expert_probs.values())
    era = _era_bucket(season)
    return {
        "efficiency_prob": expert_probs["efficiency"],
        "tournament_prob": expert_probs["tournament"],
        "power_rank_prob": expert_probs["power_rank"],
        "expert_mean": sum(probs) / len(probs),
        "expert_spread": max(probs) - min(probs),
        "eff_tourn_gap": expert_probs["efficiency"] - expert_probs["tournament"],
        "seed_strength_diff": base_features.get("seed_strength_diff", 0.0),
        "adj_net_eff_diff": base_features.get("adj_net_eff_diff", 0.0),
        "massey_diff": base_features.get("massey_diff", 0.0),
        "massey_ratio": base_features.get("massey_ratio", 0.0),
        "top_rank_flag_diff": base_features.get("top_rank_flag_diff", 0.0),
        "common_opp_margin_diff": base_features.get("common_opp_margin_diff", 0.0),
        "head_to_head_margin": base_features.get("head_to_head_margin", 0.0),
        "recent_elo_diff": base_features.get("recent_elo_diff", 0.0),
        "gender_is_men": 1.0 if gender == "men" else 0.0,
        "era_early": 1.0 if era == "early" else 0.0,
        "era_middle": 1.0 if era == "middle" else 0.0,
        "era_modern": 1.0 if era == "modern" else 0.0,
    }


def _fit_platt(probs: list[float], labels: list[float], epochs: int = 120, learning_rate: float = 0.02) -> PlattCalibrator:
    a = 1.0
    b = 0.0
    xs = []
    for prob in probs:
        clipped = min(0.999999, max(0.000001, prob))
        xs.append(math.log(clipped / (1.0 - clipped)))
    for _ in range(epochs):
        grad_a = 0.0
        grad_b = 0.0
        for x, y in zip(xs, labels):
            total = a * x + b
            pred = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, total))))
            error = pred - y
            grad_a += error * x
            grad_b += error
        scale = 1.0 / max(len(xs), 1)
        a -= learning_rate * grad_a * scale
        b -= learning_rate * grad_b * scale
    return PlattCalibrator(a=a, b=b)


def _fit_group_calibrators(meta_rows: list[TrainingExample], raw_probs: list[float]) -> dict[str, PlattCalibrator]:
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for row, prob in zip(meta_rows, raw_probs):
        era = _era_bucket(row.season)
        key = f"{row.gender}_{era}"
        probs, labels = grouped.setdefault(key, ([], []))
        probs.append(prob)
        labels.append(row.label)

        gender_key = f"{row.gender}_all"
        probs, labels = grouped.setdefault(gender_key, ([], []))
        probs.append(prob)
        labels.append(row.label)

        all_key = "all_all"
        probs, labels = grouped.setdefault(all_key, ([], []))
        probs.append(prob)
        labels.append(row.label)

    calibrators: dict[str, PlattCalibrator] = {}
    for key, (probs, labels) in grouped.items():
        if len(probs) < 40:
            continue
        calibrators[key] = _fit_platt(probs, labels)
    return calibrators


def _get_calibrator(calibrators: dict[str, PlattCalibrator], gender: str, season: int) -> PlattCalibrator:
    era = _era_bucket(season)
    return (
        calibrators.get(f"{gender}_{era}")
        or calibrators.get(f"{gender}_all")
        or calibrators.get("all_all")
        or PlattCalibrator(1.0, 0.0)
    )


def _build_walkforward_meta_rows(
    examples: list[TrainingExample],
    eval_start_season: int,
) -> tuple[dict[int, list[TrainingExample]], dict[int, list[float]]]:
    seasons = sorted({example.season for example in examples if example.season >= eval_start_season})
    meta_rows_by_season: dict[int, list[TrainingExample]] = {}
    raw_probs_by_season: dict[int, list[float]] = {}
    for season in seasons:
        train_rows = [example for example in examples if example.season < season]
        test_rows = [example for example in examples if example.season == season]
        if not train_rows or not test_rows:
            continue
        expert_models = {
            name: train_logistic_model(train_rows, feature_names=features, **_expert_hyperparams(name))
            for name, features in BASE_EXPERTS.items()
        }
        season_meta_rows: list[TrainingExample] = []
        raw_probs: list[float] = []
        for row in test_rows:
            expert_probs = {name: model.predict_prob(row.features) for name, model in expert_models.items()}
            meta_features = _build_meta_features(expert_probs, row.features, row.gender, row.season)
            raw_prob = sum(expert_probs.values()) / len(expert_probs)
            season_meta_rows.append(
                TrainingExample(
                    season=row.season,
                    gender=row.gender,
                    team_a=row.team_a,
                    team_b=row.team_b,
                    label=row.label,
                    features=meta_features,
                )
            )
            raw_probs.append(raw_prob)
        meta_rows_by_season[season] = season_meta_rows
        raw_probs_by_season[season] = raw_probs
    return meta_rows_by_season, raw_probs_by_season


def _fit_meta_model(meta_train_rows: list[TrainingExample]) -> object:
    return train_logistic_model(meta_train_rows, feature_names=META_FEATURES, epochs=220, learning_rate=0.06, l2=0.001)


def _evaluate_walkforward(meta_rows_by_season: dict[int, list[TrainingExample]], eval_start_season: int) -> dict[str, Any]:
    seasons = sorted(season for season in meta_rows_by_season if season >= eval_start_season)
    season_rows: list[dict[str, Any]] = []
    all_preds: list[float] = []
    all_labels: list[float] = []
    for season in seasons:
        meta_train_rows = [
            row
            for earlier_season, rows in meta_rows_by_season.items()
            if earlier_season < season
            for row in rows
        ]
        meta_test_rows = meta_rows_by_season[season]
        if not meta_train_rows or not meta_test_rows:
            continue
        model = _fit_meta_model(meta_train_rows)
        raw_probs = [model.predict_prob(row.features) for row in meta_train_rows]
        calibrators = _fit_group_calibrators(meta_train_rows, raw_probs)
        season_preds = []
        season_labels = []
        for row in meta_test_rows:
            raw = model.predict_prob(row.features)
            pred = _get_calibrator(calibrators, row.gender, row.season).predict(raw)
            season_preds.append(pred)
            season_labels.append(row.label)
        score = brier_score(season_preds, season_labels)
        season_rows.append({"season": season, "examples": len(season_labels), "brier": round(score, 6)})
        all_preds.extend(season_preds)
        all_labels.extend(season_labels)
    return {"overall_brier": round(brier_score(all_preds, all_labels), 6), "seasons": season_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using the v7 stacked ensemble.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v7.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)
    meta_rows_by_season, _ = _build_walkforward_meta_rows(examples, args.eval_start_season)
    evaluation = _evaluate_walkforward(meta_rows_by_season, args.eval_start_season)

    rows = read_submission_rows(args.sample_submission)
    base_model_cache: dict[tuple[str, int, str], object] = {}
    meta_model_cache: dict[int, object] = {}
    calibrator_cache: dict[int, dict[str, PlattCalibrator]] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in rows:
        if season not in meta_model_cache:
            meta_train_rows = [
                row
                for earlier_season, season_rows in meta_rows_by_season.items()
                if earlier_season < season
                for row in season_rows
            ]
            meta_model_cache[season] = _fit_meta_model(meta_train_rows)
            raw_probs = [meta_model_cache[season].predict_prob(row.features) for row in meta_train_rows]
            calibrator_cache[season] = _fit_group_calibrators(meta_train_rows, raw_probs)
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        expert_probs: dict[str, float] = {}
        for expert_name, feature_names in BASE_EXPERTS.items():
            key = (gender, season, expert_name)
            if key not in base_model_cache:
                train_rows = [example for example in examples if example.gender == gender and example.season < season]
                base_model_cache[key] = train_logistic_model(
                    train_rows,
                    feature_names=feature_names,
                    **_expert_hyperparams(expert_name),
                )
            expert_probs[expert_name] = base_model_cache[key].predict_prob(features)
        meta_features = _build_meta_features(expert_probs, features, gender, season)
        raw = meta_model_cache[season].predict_prob(meta_features)
        pred = _get_calibrator(calibrator_cache[season], gender, season).predict(raw)
        predictions.append((game_id, pred))
        row_counts[f"{gender}_{season}"] += 1

    write_submission(args.output, predictions)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "v7_stacked_logistic_meta",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "base_experts": BASE_EXPERTS,
        "meta_features": META_FEATURES,
        "walkforward_evaluation": evaluation,
        "season_gender_row_counts": dict(sorted(row_counts.items())),
    }
    write_json(manifest_path, payload)
    print(f"Submission written to {args.output}")
    print(f"Manifest written to {manifest_path}")
    print(f"Walk-forward Brier: {evaluation['overall_brier']}")


if __name__ == "__main__":
    main()
