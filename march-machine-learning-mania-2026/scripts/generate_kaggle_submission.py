#!/usr/bin/env python3
"""Generate a 2026 Kaggle submission using the best current hybrid baseline.

The core model uses the best branch discovered so far:
- men: v19 conference-tournament-aware ensemble
- women: v10 folded-interaction ensemble with historical calibration

For men's matchups only, live Polymarket prices can override part of the model:
- direct matchup markets get the strongest blend weight
- futures markets provide a weaker team-strength prior
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from generate_v10_submission import (
    WOMEN_EXPERTS as V10_WOMEN_EXPERTS,
    _blend as v10_blend,
    _expert_hyperparams as v10_params,
)
from generate_v19_conf_tourney_submission import (
    MEN_EXPERTS as V19_MEN_EXPERTS,
    _blend as v19_blend,
    _men_hyperparams as v19_params,
)
from matchup_model_utils import (
    build_matchup_features,
    build_profiles,
    build_training_examples,
    train_logistic_model,
)


HYBRID_MEN_WEIGHTS = {"efficiency_plus": 0.4, "tournament_plus": 0.05, "conference_tourney": 0.55}
HYBRID_WOMEN_WEIGHTS = {"efficiency_core": 0.5, "tournament_core": 0.5, "power_core": 0.0}
HYBRID_WOMEN_CALIBRATOR = {"alpha": 1.1, "beta": 0.0}
FUTURE_TYPE_WEIGHTS = {
    "champion": 1.0,
    "number_1_seed": 0.35,
    "final_four": 0.5,
    "elite_eight": 0.25,
    "sweet_sixteen": 0.15,
    "make_tournament": 0.1,
}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_polymarket_payload(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_pairwise_market_probs(payload: dict | None) -> dict[tuple[int, int], dict[str, Any]]:
    if not payload:
        return {}
    pairwise_index = payload.get("pairwise_index", {})
    market_probs: dict[tuple[int, int], dict[str, Any]] = {}
    for value in pairwise_index.values():
        team1_id = value.get("team1_id")
        team2_id = value.get("team2_id")
        try:
            key = (int(team1_id), int(team2_id))
        except (TypeError, ValueError):
            continue
        market_probs[key] = {
            "prob": _safe_float(value.get("team1_prob"), 0.0),
            "price_source": str(value.get("price_source") or ""),
            "market_liquidity": _safe_float(value.get("market_liquidity"), 0.0),
        }
    return market_probs


def _build_market_strengths(payload: dict | None) -> dict[int, float]:
    if not payload:
        return {}
    futures_index = payload.get("futures_index", {})
    strengths: dict[int, float] = {}
    for future_type, weight in FUTURE_TYPE_WEIGHTS.items():
        markets = futures_index.get(future_type, {})
        for value in markets.values():
            team_id = value.get("team_id")
            try:
                team_id_int = int(team_id)
            except (TypeError, ValueError):
                continue
            strengths[team_id_int] = strengths.get(team_id_int, 0.0) + weight * _safe_float(value.get("yes_prob"), 0.0)
    return strengths


def _relative_strength_prob(team_a: int, team_b: int, strengths: dict[int, float]) -> float | None:
    strength_a = strengths.get(team_a, 0.0)
    strength_b = strengths.get(team_b, 0.0)
    total = strength_a + strength_b
    if total <= 0:
        return None
    return strength_a / total


def _parse_submission_ids(path: Path) -> list[tuple[str, int, int, int, str]]:
    rows: list[tuple[str, int, int, int, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "ID" not in (reader.fieldnames or []) or "Pred" not in (reader.fieldnames or []):
            raise ValueError("Sample submission must contain ID and Pred columns")
        for row in reader:
            game_id = row["ID"]
            season_text, team_a_text, team_b_text = game_id.split("_")
            team_a = int(team_a_text)
            team_b = int(team_b_text)
            gender = "men" if 1000 <= team_a <= 1999 else "women"
            rows.append((game_id, int(season_text), team_a, team_b, gender))
    return rows


def _prediction_stats(predictions: list[tuple[str, float]]) -> dict[str, float]:
    values = [prob for _, prob in predictions]
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "std": variance ** 0.5,
    }


def _calibrate(prob: float, alpha: float, beta: float) -> float:
    return min(0.999, max(0.001, 0.5 + alpha * (prob - 0.5) + beta))


def _train_hybrid_models(
    examples,
    season: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    men_rows = [example for example in examples if example.gender == "men" and example.season < season]
    women_rows = [example for example in examples if example.gender == "women" and example.season < season]
    men_models = {
        name: train_logistic_model(men_rows, feature_names=feature_names, **v19_params(name))
        for name, feature_names in V19_MEN_EXPERTS.items()
    }
    women_models = {
        name: train_logistic_model(women_rows, feature_names=feature_names, **v10_params(name))
        for name, feature_names in V10_WOMEN_EXPERTS.items()
    }
    return men_models, women_models


def _hybrid_base_prob(
    gender: str,
    features: dict[str, float],
    men_models: dict[str, Any],
    women_models: dict[str, Any],
) -> float:
    if gender == "men":
        expert_names = list(V19_MEN_EXPERTS.keys())
        expert_probs = [men_models[name].predict_prob(features) for name in expert_names]
        return v19_blend(expert_probs, [HYBRID_MEN_WEIGHTS[name] for name in expert_names])
    expert_names = list(V10_WOMEN_EXPERTS.keys())
    expert_probs = [women_models[name].predict_prob(features) for name in expert_names]
    prob = v10_blend(expert_probs, [HYBRID_WOMEN_WEIGHTS[name] for name in expert_names])
    return _calibrate(prob, HYBRID_WOMEN_CALIBRATOR["alpha"], HYBRID_WOMEN_CALIBRATOR["beta"])


def generate_predictions(
    submission_rows: list[tuple[str, int, int, int, str]],
    profiles,
    men_models: dict[str, Any],
    women_models: dict[str, Any],
    pairwise_market_probs: dict[tuple[int, int], dict[str, Any]],
    market_strengths: dict[int, float],
    direct_market_weight: float,
    futures_market_weight: float,
) -> tuple[list[tuple[str, float]], dict[str, int]]:
    predictions: list[tuple[str, float]] = []
    usage = {
        "direct_market": 0,
        "futures_market": 0,
        "model_only_men": 0,
        "model_only_women": 0,
        "unknown_rows": 0,
        "clob_midpoints": 0,
        "gamma_outcome_prices": 0,
    }

    for game_id, season, team_a, team_b, gender in submission_rows:
        if gender not in {"men", "women"}:
            prob = 0.5
            usage["unknown_rows"] += 1
        else:
            features = build_matchup_features(profiles, gender, season, team_a, team_b)
            prob = _hybrid_base_prob(gender, features, men_models, women_models)
            if gender == "men":
                market_payload = pairwise_market_probs.get((team_a, team_b))
                if market_payload is not None:
                    market_prob = market_payload["prob"]
                    prob = (1.0 - direct_market_weight) * prob + direct_market_weight * market_prob
                    usage["direct_market"] += 1
                    if market_payload["price_source"] == "clob_midpoints":
                        usage["clob_midpoints"] += 1
                    else:
                        usage["gamma_outcome_prices"] += 1
                else:
                    futures_prob = _relative_strength_prob(team_a, team_b, market_strengths)
                    if futures_prob is not None:
                        prob = (1.0 - futures_market_weight) * prob + futures_market_weight * futures_prob
                        usage["futures_market"] += 1
                    else:
                        usage["model_only_men"] += 1
            else:
                usage["model_only_women"] += 1
        prob = max(0.001, min(0.999, prob))
        predictions.append((game_id, prob))
    return predictions, usage


def write_submission(predictions: list[tuple[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Pred"])
        for game_id, prob in predictions:
            writer.writerow([game_id, f"{prob:.6f}"])


def write_manifest(
    output_path: Path,
    sample_submission: Path,
    season: int,
    predictions: list[tuple[str, float]],
    polymarket_json: Path | None,
    direct_market_weight: float,
    futures_market_weight: float,
    usage: dict[str, int],
) -> Path:
    stats = _prediction_stats(predictions)
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "stage2_hybrid_v19_men_calibrated_v10_women",
        "output": str(output_path),
        "sample_submission": str(sample_submission),
        "season": season,
        "rows": len(predictions),
        "polymarket_json": str(polymarket_json) if polymarket_json else None,
        "direct_market_weight": direct_market_weight,
        "futures_market_weight": futures_market_weight,
        "future_type_weights": FUTURE_TYPE_WEIGHTS,
        "men_weights": HYBRID_MEN_WEIGHTS,
        "women_weights": HYBRID_WOMEN_WEIGHTS,
        "women_calibrator": HYBRID_WOMEN_CALIBRATOR,
        "market_usage": usage,
        "prediction_stats": {
            "min": round(stats["min"], 6),
            "max": round(stats["max"], 6),
            "mean": round(stats["mean"], 6),
            "std": round(stats["std"], 6),
        },
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 2026 Kaggle submission using the current best hybrid baseline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage2.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submissions") / f"stage2_hybrid_{datetime.now().strftime('%Y%m%d')}.csv",
    )
    parser.add_argument(
        "--polymarket-json",
        type=Path,
        default=Path("reports") / "polymarket_ncaab_matchups.json",
        help="Optional normalized Polymarket payload for men's market adjustments.",
    )
    parser.add_argument("--disable-polymarket", action="store_true")
    parser.add_argument("--direct-market-weight", type=float, default=0.75)
    parser.add_argument("--futures-market-weight", type=float, default=0.2)
    args = parser.parse_args()

    submission_rows = _parse_submission_ids(args.sample_submission)
    seasons = sorted({season for _, season, _, _, _ in submission_rows})
    if len(seasons) != 1:
        raise ValueError(f"Expected one submission season, found {seasons}")
    season = seasons[0]

    profiles = build_profiles(args.data_dir)
    examples = build_training_examples(args.data_dir, profiles=profiles)
    men_models, women_models = _train_hybrid_models(examples, season)

    polymarket_payload = None if args.disable_polymarket else _load_polymarket_payload(args.polymarket_json)
    pairwise_market_probs = _build_pairwise_market_probs(polymarket_payload)
    market_strengths = _build_market_strengths(polymarket_payload)

    predictions, usage = generate_predictions(
        submission_rows,
        profiles,
        men_models,
        women_models,
        pairwise_market_probs,
        market_strengths,
        direct_market_weight=args.direct_market_weight,
        futures_market_weight=args.futures_market_weight,
    )

    write_submission(predictions, args.output)
    manifest_path = write_manifest(
        output_path=args.output,
        sample_submission=args.sample_submission,
        season=season,
        predictions=predictions,
        polymarket_json=None if args.disable_polymarket else args.polymarket_json,
        direct_market_weight=args.direct_market_weight,
        futures_market_weight=args.futures_market_weight,
        usage=usage,
    )

    print("=" * 60)
    print("Stage 2 submission generated")
    print("=" * 60)
    print(f"Season used: {season}")
    print(f"Rows: {len(predictions)}")
    print(f"Men direct-market rows: {usage['direct_market']}")
    print(f"Men futures-adjusted rows: {usage['futures_market']}")
    print(f"Men model-only rows: {usage['model_only_men']}")
    print(f"Women model-only rows: {usage['model_only_women']}")
    print(f"CLOB-priced rows: {usage['clob_midpoints']}")
    print(f"Gamma-priced rows: {usage['gamma_outcome_prices']}")
    print(f"Output: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
