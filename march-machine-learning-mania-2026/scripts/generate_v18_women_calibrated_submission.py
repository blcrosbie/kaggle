#!/usr/bin/env python3
"""Generate a Stage 1 submission using v12 men and nested-calibrated v10 women."""

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
from generate_v12_ranking_submission import MEN_EXPERTS as V12_MEN_EXPERTS, _men_hyperparams as v12_params
from generate_v17_nested_calibrated_submission import _calibrate, _evaluate_gender_nested
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


MEN_WEIGHTS = {"power": 0.0, "efficiency": 0.55, "tournament_shape": 0.45, "ranking_consensus": 0.0}
WOMEN_WEIGHTS = {"efficiency_core": 0.5, "tournament_core": 0.5, "power_core": 0.0}


def _blend(values: list[float], weights: list[float]) -> float:
    return min(0.999, max(0.001, sum(v * w for v, w in zip(values, weights))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using v18 women-only nested calibration.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v18.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)

    women_eval, women_calibrators = _evaluate_gender_nested(examples, "women", args.eval_start_season)
    final_women_calibrator = women_calibrators[-1][1] if women_calibrators else {"alpha": 1.0, "beta": 0.0}

    men_examples = [example for example in examples if example.gender == "men" and example.season >= args.eval_start_season]
    women_examples = [example for example in examples if example.gender == "women" and example.season >= args.eval_start_season]
    men_eval = {"overall_brier": 0.0, "seasons": []}
    men_all_preds: list[float] = []
    men_all_targets: list[float] = []
    # Rebuild men per-season scores using the fixed v12 path.
    seasons = sorted({example.season for example in men_examples})
    for season in seasons:
        train_rows = [example for example in examples if example.gender == "men" and example.season < season]
        test_rows = [example for example in examples if example.gender == "men" and example.season == season]
        if not train_rows or not test_rows:
            continue
        models = {
            name: train_logistic_model(train_rows, feature_names=feature_names, **v12_params(name))
            for name, feature_names in V12_MEN_EXPERTS.items()
        }
        targets = [example.label for example in test_rows]
        preds = []
        expert_names = list(V12_MEN_EXPERTS.keys())
        for example in test_rows:
            probs = [models[name].predict_prob(example.features) for name in expert_names]
            preds.append(_blend(probs, [MEN_WEIGHTS[name] for name in expert_names]))
        men_eval["seasons"].append({"season": season, "examples": len(targets), "brier": round(brier_score(preds, targets), 6)})
        men_all_preds.extend(preds)
        men_all_targets.extend(targets)
    men_eval["overall_brier"] = round(brier_score(men_all_preds, men_all_targets), 6)

    model_cache: dict[tuple[str, int, str], object] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in read_submission_rows(args.sample_submission):
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        if gender == "men":
            experts = V12_MEN_EXPERTS
            weights = MEN_WEIGHTS
            params_fn = v12_params
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
            pred = _blend(probs, [weights[name] for name in expert_names])
        else:
            pred = v10_blend(probs, [weights[name] for name in expert_names])
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
        "model": "v18_v12_men_nested_calibrated_women",
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "men_weights": MEN_WEIGHTS,
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
    print(f"Final women calibrator: {final_women_calibrator}")
    print(f"Walk-forward Brier: {overall_brier}")


if __name__ == "__main__":
    main()
