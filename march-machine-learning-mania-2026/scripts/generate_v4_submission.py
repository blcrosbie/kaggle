#!/usr/bin/env python3
"""Generate a Stage 1 submission using a date-aware pregame logistic pipeline."""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from matchup_model_utils import brier_score, read_submission_rows, write_json, write_submission


EPSILON = 1e-9
FEATURE_NAMES = [
    "elo_diff",
    "recent_elo_diff",
    "win_pct_diff",
    "recent_win_pct_diff",
    "margin_diff",
    "recent_margin_diff",
    "off_eff_diff",
    "def_eff_diff",
    "net_eff_diff",
    "efg_diff",
    "tov_rate_diff",
    "orb_rate_diff",
    "ftr_diff",
    "sos_elo_diff",
    "games_diff",
]


@dataclass
class TeamState:
    games: int = 0
    wins: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    elo: float = 1500.0
    fgm: float = 0.0
    fga: float = 0.0
    fgm3: float = 0.0
    fta: float = 0.0
    turnovers: float = 0.0
    off_reb: float = 0.0
    opp_def_reb: float = 0.0
    poss: float = 0.0
    opp_poss: float = 0.0
    sos_elo_sum: float = 0.0
    recent_results: list[tuple[float, float, float]] | None = None

    def __post_init__(self) -> None:
        if self.recent_results is None:
            self.recent_results = []


@dataclass
class GameExample:
    season: int
    gender: str
    label: float
    weight: float
    features: dict[str, float]


@dataclass
class LogisticModel:
    means: list[float]
    stds: list[float]
    weights: list[float]
    intercept: float

    def predict_raw(self, features: dict[str, float]) -> float:
        total = self.intercept
        for idx, name in enumerate(FEATURE_NAMES):
            value = features.get(name, 0.0)
            total += self.weights[idx] * ((value - self.means[idx]) / self.stds[idx])
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, total))))


@dataclass
class PlattCalibrator:
    a: float
    b: float

    def predict(self, prob: float) -> float:
        clipped = min(0.999999, max(0.000001, prob))
        x = math.log(clipped / (1.0 - clipped))
        total = self.a * x + self.b
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, total))))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < EPSILON:
        return 0.0
    return numerator / denominator


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def _state_features(state: TeamState) -> dict[str, float]:
    recent = state.recent_results[-8:]
    recent_win_pct = sum(item[0] for item in recent) / len(recent) if recent else _safe_div(state.wins, max(state.games, 1))
    recent_margin = sum(item[1] for item in recent) / len(recent) if recent else _safe_div(state.points_for - state.points_against, max(state.games, 1))
    recent_elo = recent[-1][2] if recent else state.elo
    off_eff = 100.0 * _safe_div(state.points_for, state.poss)
    def_eff = 100.0 * _safe_div(state.points_against, state.opp_poss)
    return {
        "elo": state.elo,
        "recent_elo": recent_elo,
        "win_pct": _safe_div(state.wins, max(state.games, 1)),
        "recent_win_pct": recent_win_pct,
        "margin": _safe_div(state.points_for - state.points_against, max(state.games, 1)),
        "recent_margin": recent_margin,
        "off_eff": off_eff,
        "def_eff": def_eff,
        "net_eff": off_eff - def_eff,
        "efg": _safe_div(state.fgm + 0.5 * state.fgm3, state.fga),
        "tov_rate": _safe_div(state.turnovers, state.poss),
        "orb_rate": _safe_div(state.off_reb, state.off_reb + state.opp_def_reb),
        "ftr": _safe_div(state.fta, state.fga),
        "sos_elo": _safe_div(state.sos_elo_sum, max(state.games, 1)),
        "games": float(state.games),
    }


def _build_feature_diff(team_a: TeamState, team_b: TeamState) -> dict[str, float]:
    a = _state_features(team_a)
    b = _state_features(team_b)
    return {
        "elo_diff": a["elo"] - b["elo"],
        "recent_elo_diff": a["recent_elo"] - b["recent_elo"],
        "win_pct_diff": a["win_pct"] - b["win_pct"],
        "recent_win_pct_diff": a["recent_win_pct"] - b["recent_win_pct"],
        "margin_diff": a["margin"] - b["margin"],
        "recent_margin_diff": a["recent_margin"] - b["recent_margin"],
        "off_eff_diff": a["off_eff"] - b["off_eff"],
        "def_eff_diff": b["def_eff"] - a["def_eff"],
        "net_eff_diff": a["net_eff"] - b["net_eff"],
        "efg_diff": a["efg"] - b["efg"],
        "tov_rate_diff": b["tov_rate"] - a["tov_rate"],
        "orb_rate_diff": a["orb_rate"] - b["orb_rate"],
        "ftr_diff": a["ftr"] - b["ftr"],
        "sos_elo_diff": a["sos_elo"] - b["sos_elo"],
        "games_diff": a["games"] - b["games"],
    }


def _update_state(
    state: TeamState,
    opponent_elo: float,
    won: bool,
    points_for: float,
    points_against: float,
    fgm: float,
    fga: float,
    fgm3: float,
    fta: float,
    turnovers: float,
    off_reb: float,
    opp_def_reb: float,
) -> None:
    expected = _elo_expected(state.elo, opponent_elo)
    state.elo = state.elo + 20.0 * ((1.0 if won else 0.0) - expected)
    state.games += 1
    state.wins += 1 if won else 0
    state.points_for += points_for
    state.points_against += points_against
    state.fgm += fgm
    state.fga += fga
    state.fgm3 += fgm3
    state.fta += fta
    state.turnovers += turnovers
    state.off_reb += off_reb
    state.opp_def_reb += opp_def_reb
    poss = fga - off_reb + turnovers + 0.475 * fta
    state.poss += poss
    state.opp_poss += poss
    state.sos_elo_sum += opponent_elo
    state.recent_results.append((1.0 if won else 0.0, points_for - points_against, state.elo))
    if len(state.recent_results) > 8:
        state.recent_results = state.recent_results[-8:]


def _snapshot_state(state: TeamState) -> TeamState:
    return TeamState(
        games=state.games,
        wins=state.wins,
        points_for=state.points_for,
        points_against=state.points_against,
        elo=state.elo,
        fgm=state.fgm,
        fga=state.fga,
        fgm3=state.fgm3,
        fta=state.fta,
        turnovers=state.turnovers,
        off_reb=state.off_reb,
        opp_def_reb=state.opp_def_reb,
        poss=state.poss,
        opp_poss=state.opp_poss,
        sos_elo_sum=state.sos_elo_sum,
        recent_results=list(state.recent_results or []),
    )


def _build_regular_season_examples_and_snapshots(path: Path, gender: str) -> tuple[list[GameExample], dict[int, dict[int, TeamState]]]:
    examples: list[GameExample] = []
    season_states: dict[int, dict[int, TeamState]] = defaultdict(dict)
    season_snapshots: dict[int, dict[int, TeamState]] = {}

    rows = _read_csv(path)
    rows.sort(key=lambda row: (int(row["Season"]), int(row["DayNum"])))

    current_season = None
    for row in rows:
        season = int(row["Season"])
        day = int(row["DayNum"])
        if current_season is None:
            current_season = season
        elif season != current_season:
            season_snapshots[current_season] = {
                team_id: _snapshot_state(state) for team_id, state in season_states[current_season].items()
            }
            current_season = season

        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        states = season_states[season]
        winner_state = states.setdefault(winner, TeamState())
        loser_state = states.setdefault(loser, TeamState())

        team_a = min(winner, loser)
        team_b = max(winner, loser)
        state_a = winner_state if winner == team_a else loser_state
        state_b = loser_state if loser == team_b else winner_state
        if day >= 45 and state_a.games >= 8 and state_b.games >= 8:
            label = 1.0 if winner == team_a else 0.0
            weight = 1.0
            if day >= 110:
                weight += 0.5
            if row["WLoc"] == "N":
                weight += 0.25
            examples.append(
                GameExample(
                    season=season,
                    gender=gender,
                    label=label,
                    weight=weight,
                    features=_build_feature_diff(state_a, state_b),
                )
            )

        winner_pre_elo = winner_state.elo
        loser_pre_elo = loser_state.elo
        _update_state(
            winner_state,
            opponent_elo=loser_pre_elo,
            won=True,
            points_for=float(row["WScore"]),
            points_against=float(row["LScore"]),
            fgm=float(row["WFGM"]),
            fga=float(row["WFGA"]),
            fgm3=float(row["WFGM3"]),
            fta=float(row["WFTA"]),
            turnovers=float(row["WTO"]),
            off_reb=float(row["WOR"]),
            opp_def_reb=float(row["LDR"]),
        )
        _update_state(
            loser_state,
            opponent_elo=winner_pre_elo,
            won=False,
            points_for=float(row["LScore"]),
            points_against=float(row["WScore"]),
            fgm=float(row["LFGM"]),
            fga=float(row["LFGA"]),
            fgm3=float(row["LFGM3"]),
            fta=float(row["LFTA"]),
            turnovers=float(row["LTO"]),
            off_reb=float(row["LOR"]),
            opp_def_reb=float(row["WDR"]),
        )

    if current_season is not None:
        season_snapshots[current_season] = {
            team_id: _snapshot_state(state) for team_id, state in season_states[current_season].items()
        }
    return examples, season_snapshots


def _build_tournament_examples(
    path: Path,
    gender: str,
    season_snapshots: dict[int, dict[int, TeamState]],
) -> list[GameExample]:
    examples: list[GameExample] = []
    for row in _read_csv(path):
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        team_a = min(winner, loser)
        team_b = max(winner, loser)
        season_states = season_snapshots.get(season, {})
        state_a = season_states.get(team_a, TeamState())
        state_b = season_states.get(team_b, TeamState())
        examples.append(
            GameExample(
                season=season,
                gender=gender,
                label=1.0 if winner == team_a else 0.0,
                weight=1.0,
                features=_build_feature_diff(state_a, state_b),
            )
        )
    return examples


def _fit_logistic(
    examples: list[GameExample],
    epochs: int = 40,
    learning_rate: float = 0.02,
    l2: float = 0.0005,
    seed: int = 2026,
) -> LogisticModel:
    rng = random.Random(seed)
    rows = [[example.features[name] for name in FEATURE_NAMES] for example in examples]
    labels = [example.label for example in examples]
    weights = [example.weight for example in examples]
    means = []
    stds = []
    for idx in range(len(FEATURE_NAMES)):
        column = [row[idx] for row in rows]
        mean = sum(column) / max(len(column), 1)
        variance = sum((value - mean) ** 2 for value in column) / max(len(column), 1)
        std = math.sqrt(variance) if variance > EPSILON else 1.0
        means.append(mean)
        stds.append(std)

    model_weights = [0.0] * len(FEATURE_NAMES)
    intercept = 0.0
    order = list(range(len(rows)))
    for _ in range(epochs):
        rng.shuffle(order)
        for idx in order:
            normalized = [(rows[idx][j] - means[j]) / stds[j] for j in range(len(FEATURE_NAMES))]
            total = intercept + sum(model_weights[j] * normalized[j] for j in range(len(FEATURE_NAMES)))
            pred = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, total))))
            error = (pred - labels[idx]) * weights[idx]
            intercept -= learning_rate * error
            for j in range(len(model_weights)):
                model_weights[j] -= learning_rate * (error * normalized[j] + l2 * model_weights[j])
    return LogisticModel(means=means, stds=stds, weights=model_weights, intercept=intercept)


def _fit_platt(probs: list[float], labels: list[float], epochs: int = 100, learning_rate: float = 0.01) -> PlattCalibrator:
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


def _build_all_data(data_dir: Path) -> tuple[list[GameExample], list[GameExample], dict[str, dict[int, dict[int, TeamState]]]]:
    men_reg, men_snapshots = _build_regular_season_examples_and_snapshots(data_dir / "MRegularSeasonDetailedResults.csv", "men")
    women_reg, women_snapshots = _build_regular_season_examples_and_snapshots(data_dir / "WRegularSeasonDetailedResults.csv", "women")
    men_tourney = _build_tournament_examples(data_dir / "MNCAATourneyCompactResults.csv", "men", men_snapshots)
    women_tourney = _build_tournament_examples(data_dir / "WNCAATourneyCompactResults.csv", "women", women_snapshots)
    snapshots = {"men": men_snapshots, "women": women_snapshots}
    return men_reg + women_reg, men_tourney + women_tourney, snapshots


def _train_for_target_season(
    regular_examples: list[GameExample],
    tourney_examples: list[GameExample],
    target_season: int,
    gender: str,
    train_window: int,
) -> tuple[LogisticModel, PlattCalibrator]:
    min_season = target_season - train_window
    reg_rows = [
        example
        for example in regular_examples
        if example.gender == gender and min_season <= example.season < target_season
    ]
    if not reg_rows:
        raise ValueError(f"No regular-season rows available for {gender=} {target_season=}")
    model = _fit_logistic(reg_rows)

    calib_rows = [
        example
        for example in tourney_examples
        if example.gender == gender and min_season <= example.season < target_season
    ]
    if not calib_rows:
        return model, PlattCalibrator(a=1.0, b=0.0)
    raw_probs = [model.predict_raw(example.features) for example in calib_rows]
    labels = [example.label for example in calib_rows]
    calibrator = _fit_platt(raw_probs, labels)
    return model, calibrator


def _evaluate_walkforward(
    regular_examples: list[GameExample],
    tourney_examples: list[GameExample],
    eval_start_season: int,
    train_window: int,
) -> dict[str, Any]:
    seasons = sorted({example.season for example in tourney_examples if example.season >= eval_start_season})
    season_rows: list[dict[str, Any]] = []
    all_preds: list[float] = []
    all_labels: list[float] = []
    for season in seasons:
        season_examples = [example for example in tourney_examples if example.season == season]
        if not season_examples:
            continue
        preds: list[float] = []
        labels: list[float] = []
        by_gender: dict[str, list[GameExample]] = defaultdict(list)
        for example in season_examples:
            by_gender[example.gender].append(example)
        for gender, gender_rows in by_gender.items():
            model, calibrator = _train_for_target_season(
                regular_examples,
                tourney_examples,
                target_season=season,
                gender=gender,
                train_window=train_window,
            )
            for example in gender_rows:
                pred = calibrator.predict(model.predict_raw(example.features))
                preds.append(pred)
                labels.append(example.label)
        score = brier_score(preds, labels)
        season_rows.append({"season": season, "examples": len(labels), "brier": round(score, 6)})
        all_preds.extend(preds)
        all_labels.extend(labels)
    return {"overall_brier": round(brier_score(all_preds, all_labels), 6), "seasons": season_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Stage 1 submission using the v4 date-aware pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-submission", type=Path, default=Path("data/SampleSubmissionStage1.csv"))
    parser.add_argument("--output", type=Path, default=Path("submissions/stage1_v4.csv"))
    parser.add_argument("--eval-start-season", type=int, default=2018)
    parser.add_argument("--train-window", type=int, default=6)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regular_examples, tourney_examples, snapshots = _build_all_data(args.data_dir)
    evaluation = _evaluate_walkforward(
        regular_examples,
        tourney_examples,
        eval_start_season=args.eval_start_season,
        train_window=args.train_window,
    )

    rows = read_submission_rows(args.sample_submission)
    model_cache: dict[tuple[str, int], tuple[LogisticModel, PlattCalibrator]] = {}
    predictions: list[tuple[str, float]] = []
    row_counts: dict[str, int] = defaultdict(int)

    for game_id, season, team_a, team_b, gender in rows:
        key = (gender, season)
        if key not in model_cache:
            model_cache[key] = _train_for_target_season(
                regular_examples,
                tourney_examples,
                target_season=season,
                gender=gender,
                train_window=args.train_window,
            )
        model, calibrator = model_cache[key]
        state_a = snapshots.get(gender, {}).get(season, {}).get(team_a, TeamState())
        state_b = snapshots.get(gender, {}).get(season, {}).get(team_b, TeamState())
        raw = model.predict_raw(_build_feature_diff(state_a, state_b))
        pred = calibrator.predict(raw)
        predictions.append((game_id, pred))
        row_counts[f"{gender}_{season}"] += 1

    write_submission(args.output, predictions)
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "v4_date_aware_logistic",
        "feature_names": FEATURE_NAMES,
        "sample_submission": str(args.sample_submission),
        "output": str(args.output),
        "rows": len(predictions),
        "train_window": args.train_window,
        "walkforward_evaluation": evaluation,
        "regular_training_examples": len(regular_examples),
        "tournament_calibration_examples": len(tourney_examples),
        "season_gender_row_counts": dict(sorted(row_counts.items())),
    }
    write_json(manifest_path, payload)

    print(f"Submission written to {args.output}")
    print(f"Manifest written to {manifest_path}")
    print(f"Walk-forward Brier: {evaluation['overall_brier']}")


if __name__ == "__main__":
    main()
