#!/usr/bin/env python3
"""Generate a Stage 1 submission using a direct men logistic over the v19-style feature union."""

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
from generate_v17_nested_calibrated_submission import _calibrate, _evaluate_gender_nested
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


MEN_FEATURE_SETS: dict[str, list[str]] = {
    "union_core": [
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
        "seed_efficiency_interaction",
        "efficiency_rank_interaction",
        "conf_tourney_elo_diff",
        "conf_tourney_win_diff",
        "conf_tourney_games_diff",
        "conf_tourney_win_pct_diff",
        "conf_tourney_champ_diff",
        "conf_tourney_last_day_diff",
        "conf_elo_diff",
        "recent_elo_diff",
        "recent_margin_diff",
        "seed_strength_diff",
    ],
    "union_plus": [
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
        "seed_efficiency_interaction",
        "efficiency_rank_interaction",
        "conf_tourney_elo_diff",
        "conf_tourney_win_diff",
        "conf_tourney_games_diff",
        "conf_tourney_win_pct_diff",
        "conf_tourney_champ_diff",
        "conf_tourney_last_day_diff",
        "conf_elo_diff",
        "recent_elo_diff",
        "recent_margin_diff",
        "seed_strength_diff",
        "seed_diff",
        "seed_known",
        "favorite_flag",
        "massey_diff",
        "massey_known",
        "games_diff",
        "seed_recent_interaction",
        "recent_rank_interaction",
    ],
    "union_plus_consistency": [
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
        "seed_efficiency_interaction",
        "efficiency_rank_interaction",
        "conf_tourney_elo_diff",
        "conf_tourney_win_diff",
        "conf_tourney_games_diff",
        "conf_tourney_win_pct_diff",
        "conf_tourney_champ_diff",
        "conf_tourney_last_day_diff",
        "conf_elo_diff",
        "recent_elo_diff",
        "recent_margin_diff",
        "seed_strength_diff",
        "seed_diff",
        "seed_known",
        "favorite_flag",
        "massey_diff",
        "games_diff",
        "seed_recent_interaction",
        "margin_std_diff",
        "close_win_pct_diff",
        "blowout_rate_diff",
        "quality_margin_diff",
    ],
}

MEN_PARAMS: dict[str, dict[str, float | int]] = {
    "fast": {"epochs": 220, "learning_rate": 0.055, "l2": 0.001},
    "balanced": {"epochs": 260, "learning_rate": 0.05, "l2": 0.0009},
    "conservative": {"epochs": 320, "learning_rate": 0.042, "l2": 0.0013},
}

WOMEN_WEIGHTS = {"efficiency_core": 0.5, "tournament_core": 0.5, "power_core": 0.0}


def _score_subset(rows: list[dict[str, Any]], seasons: set[int]) -> float:
    preds: list[float] = []
    targets: list[float] = []
    for row in rows:
        if int(row["season"]) not in seasons:
            continue
        preds.extend(row["preds"])
        targets.extend(row["targets"])
    return round(brier_score(preds, targets), 6)


def _select_men_model(examples: list[TrainingExample], eval_start_season: int) -> tuple[str, str, dict[str, Any]]:
    seasons = sorted({example.season for example in examples if example.gender == "men" and example.season >= eval_start_season})
    selection_seasons = {season for season in seasons if season <= 2022}
    holdout_seasons = {season for season in seasons if season >= 2023}

    best_key = ("", "")
    best_selection = float("inf")
    best_eval: dict[str, Any] = {}

    for feature_set_name, feature_names in MEN_FEATURE_SETS.items():
        for param_name, params in MEN_PARAMS.items():
            detailed_rows: list[dict[str, Any]] = []
            season_rows: list[dict[str, Any]] = []
            for season in seasons:
                train_rows = [example for example in examples if example.gender == "men" and example.season < season]
                test_rows = [example for example in examples if example.gender == "men" and example.season == season]
                if not train_rows or not test_rows:
                    continue
                model = train_logistic_model(train_rows, feature_names=feature_names, **params)
                preds = [model.predict_prob(example.features) for example in test_rows]
                targets = [example.label for example in test_rows]
                detailed_rows.append({"season": season, "preds": preds, "targets": targets})
                season_rows.append({"season": season, "examples": len(targets), "brier": round(brier_score(preds, targets), 6)})

            selection_brier = _score_subset(detailed_rows, selection_seasons)
            holdout_brier = _score_subset(detailed_rows, holdout_seasons)
            overall_brier = round(
                brier_score(
                    [pred for row in detailed_rows for pred in row["preds"]],
                    [target for row in detailed_rows for target in row["targets"]],
                ),
                6,
            )
            if selection_brier < best_selection:
                best_selection = selection_brier
                best_key = (feature_set_name, param_name)
                best_eval = {
                    "overall_brier": overall_brier,
                    "selection_brier": selection_brier,
                    "holdout_brier": holdout_brier,
                    "seasons": season_rows,
                }
    return best_key[0], best_key[1], best_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using a direct men logistic over the v19-style feature union.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v22.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    men_feature_set, men_param_name, men_eval = _select_men_model(examples, args.eval_start_season)
    women_eval, women_calibrators = _evaluate_gender_nested(examples, "women", args.eval_start_season)
    final_women_calibrator = women_calibrators[-1][1] if women_calibrators else {"alpha": 1.0, "beta": 0.0}

    model_cache: dict[tuple[str, int, str], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in read_submission_rows(args.sample_submission):
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        if gender == "men":
            key = (gender, season, "direct")
            if key not in model_cache:
                train_rows = [example for example in examples if example.gender == gender and example.season < season]
                model_cache[key] = train_logistic_model(
                    train_rows,
                    feature_names=MEN_FEATURE_SETS[men_feature_set],
                    **MEN_PARAMS[men_param_name],
                )
            pred = model_cache[key].predict_prob(features)
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
        "model": "v22_direct_union_men_nested_calibrated_women",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "selected_men_feature_set": men_feature_set,
        "selected_men_params": MEN_PARAMS[men_param_name],
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
    print(f"Selected men feature set: {men_feature_set}")
    print(f"Selected men params: {MEN_PARAMS[men_param_name]}")
    print(f"Walk-forward Brier: {overall_brier}")


if __name__ == "__main__":
    main()
