#!/usr/bin/env python3
"""Utilities for building NCAA matchup features and pure-Python baseline models."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPSILON = 1e-9
FEATURE_NAMES = [
    "elo_diff",
    "win_pct_diff",
    "scoring_margin_diff",
    "points_for_diff",
    "points_against_diff",
    "off_eff_diff",
    "def_eff_diff",
    "net_eff_diff",
    "efg_diff",
    "tov_rate_diff",
    "orb_rate_diff",
    "ftr_diff",
    "seed_diff",
    "seed_known",
    "massey_diff",
    "massey_known",
    "games_diff",
    "sos_elo_diff",
    "sos_margin_diff",
    "conf_elo_diff",
    "conf_margin_diff",
    "conf_tourney_win_diff",
    "conf_tourney_games_diff",
    "conf_tourney_win_pct_diff",
    "conf_tourney_champ_diff",
    "conf_tourney_last_day_diff",
    "conf_tourney_elo_diff",
    "recent_win_pct_diff",
    "recent_margin_diff",
    "recent_elo_diff",
    "adj_net_eff_diff",
    "seed_strength_diff",
    "favorite_flag",
    "common_opp_margin_diff",
    "common_opp_win_diff",
    "common_opp_count",
    "head_to_head_margin",
    "head_to_head_games",
    "massey_ratio",
    "massey_gap_sq",
    "top_rank_flag_diff",
    "massey_mean_diff",
    "massey_best_diff",
    "massey_count_diff",
    "massey_median_rank_diff",
    "massey_mean_rank_diff",
    "massey_best_rank_diff",
    "massey_top10_count_diff",
    "massey_top25_count_diff",
    "seed_efficiency_interaction",
    "seed_recent_interaction",
    "recent_rank_interaction",
    "head_to_head_common_opp_interaction",
    "efficiency_rank_interaction",
    "loc_elo_diff",
    "neutral_win_pct_diff",
    "neutral_margin_diff",
    "away_win_pct_diff",
    "away_margin_diff",
    "site_bias_diff",
    "neutral_games_diff",
    "coach_tenure_diff",
    "coach_change_diff",
    "quality_win_pct_diff",
    "quality_margin_diff",
    "quality_games_diff",
    "bad_loss_rate_diff",
    "best_win_elo_diff",
    "margin_std_diff",
    "close_win_pct_diff",
    "close_games_diff",
    "blowout_rate_diff",
]


@dataclass
class TeamSeasonProfile:
    games: int = 0
    wins: int = 0
    losses: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    scoring_margin: float = 0.0
    elo: float = 1500.0
    win_pct: float = 0.5
    off_eff: float = 0.0
    def_eff: float = 0.0
    net_eff: float = 0.0
    efg: float = 0.0
    tov_rate: float = 0.0
    orb_rate: float = 0.0
    ftr: float = 0.0
    seed: int | None = None
    massey: float | None = None
    massey_mean: float | None = None
    massey_best: float | None = None
    massey_count: float = 0.0
    massey_median_rank: float | None = None
    massey_mean_rank: float | None = None
    massey_best_rank: float | None = None
    massey_top10_count: float = 0.0
    massey_top25_count: float = 0.0
    sos_elo: float = 1500.0
    sos_margin: float = 0.0
    conference: str | None = None
    conf_elo: float = 1500.0
    conf_margin: float = 0.0
    conf_tourney_wins: float = 0.0
    conf_tourney_games: float = 0.0
    conf_tourney_win_pct: float = 0.0
    conf_tourney_champ: float = 0.0
    conf_tourney_last_day: float = 0.0
    conf_tourney_elo: float = 1500.0
    recent_win_pct: float = 0.5
    recent_margin: float = 0.0
    recent_elo: float = 1500.0
    adj_net_eff: float = 0.0
    loc_elo: float = 1500.0
    neutral_win_pct: float = 0.5
    neutral_margin: float = 0.0
    away_win_pct: float = 0.5
    away_margin: float = 0.0
    site_bias: float = 0.0
    neutral_games: float = 0.0
    coach_tenure: float = 0.0
    coach_change: float = 0.0
    quality_win_pct: float = 0.0
    quality_margin: float = 0.0
    quality_games: float = 0.0
    bad_loss_rate: float = 0.0
    best_win_elo: float = 1500.0
    margin_std: float = 0.0
    close_win_pct: float = 0.5
    close_games: float = 0.0
    blowout_rate: float = 0.0
    opponent_results: dict[int, tuple[float, float, float]] | None = None

    def __post_init__(self) -> None:
        if self.opponent_results is None:
            self.opponent_results = {}


@dataclass
class TrainingExample:
    season: int
    gender: str
    team_a: int
    team_b: int
    label: float
    features: dict[str, float]


@dataclass
class Stump:
    feature_index: int
    threshold: float
    left_value: float
    right_value: float


@dataclass
class LogisticModel:
    means: list[float]
    stds: list[float]
    weights: list[float]
    intercept: float
    feature_names: list[str]

    def predict_prob(self, features: dict[str, float]) -> float:
        total = self.intercept
        for index, name in enumerate(self.feature_names):
            value = features.get(name, 0.0)
            total += self.weights[index] * ((value - self.means[index]) / self.stds[index])
        return 1.0 / (1.0 + math.exp(-max(-25.0, min(25.0, total))))


@dataclass
class BoostedStumpModel:
    base_prob: float
    learning_rate: float
    stumps: list[Stump]
    feature_names: list[str]

    def predict_prob(self, features: dict[str, float]) -> float:
        pred = self.base_prob
        values = [features.get(name, 0.0) for name in self.feature_names]
        for stump in self.stumps:
            if values[stump.feature_index] <= stump.threshold:
                pred += self.learning_rate * stump.left_value
            else:
                pred += self.learning_rate * stump.right_value
        return min(0.999, max(0.001, pred))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def _safe_div(numerator: float, denominator: float) -> float:
    if abs(denominator) < EPSILON:
        return 0.0
    return numerator / denominator


def _parse_seed(seed: str) -> int:
    digits = "".join(char for char in seed if char.isdigit())
    return int(digits[:2])


def _load_seeds(path: Path) -> dict[int, dict[int, int]]:
    by_season: dict[int, dict[int, int]] = {}
    for row in _read_csv(path):
        season = int(row["Season"])
        by_season.setdefault(season, {})[int(row["TeamID"])] = _parse_seed(row["Seed"])
    return by_season


def _load_compact_profiles(path: Path) -> dict[int, dict[int, dict[str, float]]]:
    season_teams: dict[int, dict[int, dict[str, float]]] = {}
    season_elos: dict[int, dict[int, float]] = {}
    season_loc_elos: dict[int, dict[int, float]] = {}
    opponents: dict[int, dict[int, list[int]]] = {}
    recent_windows: dict[int, dict[int, list[tuple[float, float, float]]]] = {}
    for row in _read_csv(path):
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        winner_score = float(row["WScore"])
        loser_score = float(row["LScore"])
        wloc = row["WLoc"]
        margin = winner_score - loser_score
        teams = season_teams.setdefault(season, {})
        winner_stats = teams.setdefault(
            winner,
            {
                "games": 0.0,
                "wins": 0.0,
                "points_for": 0.0,
                "points_against": 0.0,
                "margin_sq_sum": 0.0,
                "close_games": 0.0,
                "close_wins": 0.0,
                "blowout_wins": 0.0,
                "blowout_losses": 0.0,
                "home_games": 0.0,
                "home_wins": 0.0,
                "home_margin": 0.0,
                "away_games": 0.0,
                "away_wins": 0.0,
                "away_margin": 0.0,
                "neutral_games": 0.0,
                "neutral_wins": 0.0,
                "neutral_margin": 0.0,
                "opponent_results": {},
            },
        )
        loser_stats = teams.setdefault(
            loser,
            {
                "games": 0.0,
                "wins": 0.0,
                "points_for": 0.0,
                "points_against": 0.0,
                "margin_sq_sum": 0.0,
                "close_games": 0.0,
                "close_wins": 0.0,
                "blowout_wins": 0.0,
                "blowout_losses": 0.0,
                "home_games": 0.0,
                "home_wins": 0.0,
                "home_margin": 0.0,
                "away_games": 0.0,
                "away_wins": 0.0,
                "away_margin": 0.0,
                "neutral_games": 0.0,
                "neutral_wins": 0.0,
                "neutral_margin": 0.0,
                "opponent_results": {},
            },
        )
        winner_stats["games"] += 1
        winner_stats["wins"] += 1
        winner_stats["points_for"] += winner_score
        winner_stats["points_against"] += loser_score
        winner_stats["margin_sq_sum"] += margin * margin

        loser_stats["games"] += 1
        loser_stats["points_for"] += loser_score
        loser_stats["points_against"] += winner_score
        loser_stats["margin_sq_sum"] += margin * margin
        winner_vs = winner_stats["opponent_results"].setdefault(loser, {"games": 0.0, "wins": 0.0, "margin": 0.0})
        loser_vs = loser_stats["opponent_results"].setdefault(winner, {"games": 0.0, "wins": 0.0, "margin": 0.0})
        winner_vs["games"] += 1
        winner_vs["wins"] += 1
        winner_vs["margin"] += margin
        loser_vs["games"] += 1
        loser_vs["margin"] -= margin
        if margin <= 5.0:
            winner_stats["close_games"] += 1
            winner_stats["close_wins"] += 1
            loser_stats["close_games"] += 1
        if margin >= 15.0:
            winner_stats["blowout_wins"] += 1
            loser_stats["blowout_losses"] += 1
        if wloc == "H":
            winner_stats["home_games"] += 1
            winner_stats["home_wins"] += 1
            winner_stats["home_margin"] += margin
            loser_stats["away_games"] += 1
            loser_stats["away_margin"] -= margin
        elif wloc == "A":
            winner_stats["away_games"] += 1
            winner_stats["away_wins"] += 1
            winner_stats["away_margin"] += margin
            loser_stats["home_games"] += 1
            loser_stats["home_margin"] -= margin
        else:
            winner_stats["neutral_games"] += 1
            winner_stats["neutral_wins"] += 1
            winner_stats["neutral_margin"] += margin
            loser_stats["neutral_games"] += 1
            loser_stats["neutral_margin"] -= margin
        season_opponents = opponents.setdefault(season, {})
        season_opponents.setdefault(winner, []).append(loser)
        season_opponents.setdefault(loser, []).append(winner)

        season_elo = season_elos.setdefault(season, {})
        ra = season_elo.get(winner, 1500.0)
        rb = season_elo.get(loser, 1500.0)
        expected = _elo_expected(ra, rb)
        season_elo[winner] = ra + 20.0 * (1.0 - expected)
        season_elo[loser] = rb - 20.0 * (1.0 - expected)
        season_loc_elo = season_loc_elos.setdefault(season, {})
        rla = season_loc_elo.get(winner, 1500.0)
        rlb = season_loc_elo.get(loser, 1500.0)
        home_edge = 70.0
        if wloc == "H":
            loc_expected = _elo_expected(rla + home_edge, rlb)
        elif wloc == "A":
            loc_expected = _elo_expected(rla, rlb + home_edge)
        else:
            loc_expected = _elo_expected(rla, rlb)
        season_loc_elo[winner] = rla + 20.0 * (1.0 - loc_expected)
        season_loc_elo[loser] = rlb - 20.0 * (1.0 - loc_expected)
        recent = recent_windows.setdefault(season, {})
        recent.setdefault(winner, []).append((1.0, winner_score - loser_score, season_elo[winner]))
        recent.setdefault(loser, []).append((0.0, loser_score - winner_score, season_elo[loser]))
        if len(recent[winner]) > 10:
            recent[winner] = recent[winner][-10:]
        if len(recent[loser]) > 10:
            recent[loser] = recent[loser][-10:]

    for season, teams in season_teams.items():
        season_elo = season_elos.get(season, {})
        season_loc_elo = season_loc_elos.get(season, {})
        ordered_elos = sorted(season_elo.values())
        if ordered_elos:
            strong_cutoff = ordered_elos[min(len(ordered_elos) - 1, int(0.75 * (len(ordered_elos) - 1)))]
            weak_cutoff = ordered_elos[min(len(ordered_elos) - 1, int(0.25 * (len(ordered_elos) - 1)))]
        else:
            strong_cutoff = 1550.0
            weak_cutoff = 1450.0
        for team_id, stats in teams.items():
            stats["elo"] = season_elo.get(team_id, 1500.0)
            stats["loc_elo"] = season_loc_elo.get(team_id, 1500.0)
            team_opponents = opponents.get(season, {}).get(team_id, [])
            if team_opponents:
                stats["sos_elo"] = sum(season_elo.get(opp_id, 1500.0) for opp_id in team_opponents) / len(team_opponents)
                stats["sos_margin"] = sum(
                    _safe_div(
                        season_teams[season].get(opp_id, {}).get("points_for", 0.0)
                        - season_teams[season].get(opp_id, {}).get("points_against", 0.0),
                        max(season_teams[season].get(opp_id, {}).get("games", 1.0), 1.0),
                    )
                    for opp_id in team_opponents
                ) / len(team_opponents)
            else:
                stats["sos_elo"] = 1500.0
                stats["sos_margin"] = 0.0
            recent = recent_windows.get(season, {}).get(team_id, [])
            if recent:
                stats["recent_win_pct"] = sum(item[0] for item in recent) / len(recent)
                stats["recent_margin"] = sum(item[1] for item in recent) / len(recent)
                stats["recent_elo"] = recent[-1][2]
            else:
                stats["recent_win_pct"] = _safe_div(stats.get("wins", 0.0), max(stats.get("games", 1.0), 1.0))
                stats["recent_margin"] = _safe_div(
                    stats.get("points_for", 0.0) - stats.get("points_against", 0.0),
                    max(stats.get("games", 1.0), 1.0),
                )
                stats["recent_elo"] = stats["elo"]
            neutral_games = max(stats.get("neutral_games", 0.0), 0.0)
            away_games = max(stats.get("away_games", 0.0), 0.0)
            home_games = max(stats.get("home_games", 0.0), 0.0)
            home_margin = _safe_div(stats.get("home_margin", 0.0), max(home_games, 1.0))
            away_margin = _safe_div(stats.get("away_margin", 0.0), max(away_games, 1.0))
            neutral_margin = _safe_div(stats.get("neutral_margin", 0.0), max(neutral_games, 1.0))
            stats["neutral_win_pct"] = _safe_div(stats.get("neutral_wins", 0.0), max(neutral_games, 1.0)) if neutral_games else _safe_div(stats.get("wins", 0.0), max(stats.get("games", 1.0), 1.0))
            stats["neutral_margin_avg"] = neutral_margin if neutral_games else _safe_div(
                stats.get("points_for", 0.0) - stats.get("points_against", 0.0),
                max(stats.get("games", 1.0), 1.0),
            )
            stats["away_win_pct"] = _safe_div(stats.get("away_wins", 0.0), max(away_games, 1.0)) if away_games else _safe_div(stats.get("wins", 0.0), max(stats.get("games", 1.0), 1.0))
            stats["away_margin_avg"] = away_margin if away_games else _safe_div(
                stats.get("points_for", 0.0) - stats.get("points_against", 0.0),
                max(stats.get("games", 1.0), 1.0),
            )
            stats["site_bias"] = home_margin - away_margin if home_games and away_games else home_margin - stats["away_margin_avg"]
            strong_games = 0.0
            strong_wins = 0.0
            strong_margin = 0.0
            weak_games = 0.0
            weak_losses = 0.0
            best_win_elo = 1500.0
            for opp_id, bucket in stats.get("opponent_results", {}).items():
                opp_elo = season_elo.get(opp_id, 1500.0)
                games_vs = float(bucket.get("games", 0.0))
                wins_vs = float(bucket.get("wins", 0.0))
                margin_vs = float(bucket.get("margin", 0.0))
                if opp_elo >= strong_cutoff:
                    strong_games += games_vs
                    strong_wins += wins_vs
                    strong_margin += margin_vs
                if opp_elo <= weak_cutoff:
                    weak_games += games_vs
                    weak_losses += max(games_vs - wins_vs, 0.0)
                if wins_vs > 0.0:
                    best_win_elo = max(best_win_elo, opp_elo)
            stats["quality_games"] = strong_games
            stats["quality_win_pct"] = _safe_div(strong_wins, max(strong_games, 1.0))
            stats["quality_margin"] = _safe_div(strong_margin, max(strong_games, 1.0))
            stats["bad_loss_rate"] = _safe_div(weak_losses, max(weak_games, 1.0))
            stats["best_win_elo"] = best_win_elo
            mean_margin = _safe_div(stats.get("points_for", 0.0) - stats.get("points_against", 0.0), max(stats.get("games", 1.0), 1.0))
            mean_sq_margin = _safe_div(stats.get("margin_sq_sum", 0.0), max(stats.get("games", 1.0), 1.0))
            stats["margin_std"] = math.sqrt(max(mean_sq_margin - mean_margin * mean_margin, 0.0))
            close_games_count = max(stats.get("close_games", 0.0), 0.0)
            stats["close_win_pct"] = _safe_div(stats.get("close_wins", 0.0), max(close_games_count, 1.0)) if close_games_count else _safe_div(
                stats.get("wins", 0.0), max(stats.get("games", 1.0), 1.0)
            )
            stats["blowout_rate"] = _safe_div(
                stats.get("blowout_wins", 0.0) - stats.get("blowout_losses", 0.0),
                max(stats.get("games", 1.0), 1.0),
            )
    return season_teams


def _load_team_conferences(path: Path) -> dict[int, dict[int, str]]:
    season_teams: dict[int, dict[int, str]] = {}
    for row in _read_csv(path):
        season = int(row["Season"])
        season_teams.setdefault(season, {})[int(row["TeamID"])] = row["ConfAbbrev"]
    return season_teams


def _load_active_coaches(path: Path, target_day: int = 133) -> dict[int, dict[int, dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in _read_csv(path):
        grouped.setdefault((int(row["Season"]), int(row["TeamID"])), []).append(row)

    season_teams: dict[int, dict[int, dict[str, Any]]] = {}
    for (season, team_id), rows in grouped.items():
        sorted_rows = sorted(rows, key=lambda row: (int(row["FirstDayNum"]), int(row["LastDayNum"])))
        active_row = None
        for row in sorted_rows:
            first = int(row["FirstDayNum"])
            last = int(row["LastDayNum"])
            if first <= target_day <= last:
                active_row = row
                break
        if active_row is None:
            prior_rows = [row for row in sorted_rows if int(row["FirstDayNum"]) <= target_day]
            active_row = prior_rows[-1] if prior_rows else sorted_rows[-1]
        season_teams.setdefault(season, {})[team_id] = {
            "coach": active_row["CoachName"],
            "coach_change": 1.0 if len(sorted_rows) > 1 or int(active_row["FirstDayNum"]) > 0 else 0.0,
        }
    return season_teams


def _build_coach_tenure(active_coaches: dict[int, dict[int, dict[str, Any]]]) -> dict[int, dict[int, float]]:
    tenure: dict[int, dict[int, float]] = {}
    history: dict[int, tuple[str, float]] = {}
    for season in sorted(active_coaches):
        tenure[season] = {}
        for team_id, payload in active_coaches[season].items():
            coach = str(payload.get("coach", ""))
            prev_coach, prev_tenure = history.get(team_id, ("", 0.0))
            current_tenure = prev_tenure + 1.0 if coach and coach == prev_coach else 1.0
            tenure[season][team_id] = current_tenure if coach else 0.0
            history[team_id] = (coach, current_tenure)
    return tenure


def _load_conference_tourney_stats(path: Path) -> dict[int, dict[int, dict[str, float]]]:
    season_teams: dict[int, dict[int, dict[str, float]]] = {}
    season_elos: dict[int, dict[int, float]] = {}
    rows_by_season_conf: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in _read_csv(path):
        season = int(row["Season"])
        conf = row["ConfAbbrev"]
        rows_by_season_conf.setdefault((season, conf), []).append(row)

    for (season, conf), rows in rows_by_season_conf.items():
        sorted_rows = sorted(rows, key=lambda item: int(item["DayNum"]))
        season_elo = season_elos.setdefault(season, {})
        champion_team_id = int(sorted_rows[-1]["WTeamID"])
        champion_day = float(sorted_rows[-1]["DayNum"])
        for row in sorted_rows:
            winner = int(row["WTeamID"])
            loser = int(row["LTeamID"])
            day_num = float(row["DayNum"])
            winner_stats = season_teams.setdefault(season, {}).setdefault(
                winner,
                {
                    "games": 0.0,
                    "wins": 0.0,
                    "last_day": 0.0,
                    "champ": 0.0,
                    "elo": 1500.0,
                },
            )
            loser_stats = season_teams.setdefault(season, {}).setdefault(
                loser,
                {
                    "games": 0.0,
                    "wins": 0.0,
                    "last_day": 0.0,
                    "champ": 0.0,
                    "elo": 1500.0,
                },
            )
            winner_stats["games"] += 1
            winner_stats["wins"] += 1
            winner_stats["last_day"] = max(winner_stats["last_day"], day_num)
            loser_stats["games"] += 1
            loser_stats["last_day"] = max(loser_stats["last_day"], day_num)

            ra = season_elo.get(winner, 1500.0)
            rb = season_elo.get(loser, 1500.0)
            expected = _elo_expected(ra, rb)
            season_elo[winner] = ra + 24.0 * (1.0 - expected)
            season_elo[loser] = rb - 24.0 * (1.0 - expected)
            winner_stats["elo"] = season_elo[winner]
            loser_stats["elo"] = season_elo[loser]

        champion_stats = season_teams.setdefault(season, {}).setdefault(
            champion_team_id,
            {
                "games": 0.0,
                "wins": 0.0,
                "last_day": champion_day,
                "champ": 0.0,
                "elo": season_elo.get(champion_team_id, 1500.0),
            },
        )
        champion_stats["champ"] = 1.0
        champion_stats["last_day"] = max(champion_stats["last_day"], champion_day)

    for teams in season_teams.values():
        for stats in teams.values():
            stats["win_pct"] = _safe_div(stats["wins"], max(stats["games"], 1.0))
    return season_teams


def _build_conference_strengths(
    compact: dict[int, dict[int, dict[str, float]]],
    team_conferences: dict[int, dict[int, str]],
) -> dict[int, dict[str, dict[str, float]]]:
    strengths: dict[int, dict[str, dict[str, float]]] = {}
    for season, teams in compact.items():
        conf_stats: dict[str, dict[str, float]] = {}
        conf_counts: dict[str, int] = {}
        for team_id, stats in teams.items():
            conference = team_conferences.get(season, {}).get(team_id)
            if not conference:
                continue
            bucket = conf_stats.setdefault(conference, {"elo": 0.0, "margin": 0.0})
            bucket["elo"] += stats.get("elo", 1500.0)
            bucket["margin"] += _safe_div(
                stats.get("points_for", 0.0) - stats.get("points_against", 0.0),
                max(stats.get("games", 1.0), 1.0),
            )
            conf_counts[conference] = conf_counts.get(conference, 0) + 1
        for conference, bucket in conf_stats.items():
            count = max(conf_counts.get(conference, 1), 1)
            bucket["elo"] /= count
            bucket["margin"] /= count
        strengths[season] = conf_stats
    return strengths


def _load_detailed_profiles(path: Path) -> dict[int, dict[int, dict[str, float]]]:
    season_teams: dict[int, dict[int, dict[str, float]]] = {}
    for row in _read_csv(path):
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        teams = season_teams.setdefault(season, {})
        w = teams.setdefault(
            winner,
            {
                "poss": 0.0,
                "opp_poss": 0.0,
                "points_for": 0.0,
                "points_against": 0.0,
                "fgm": 0.0,
                "fga": 0.0,
                "fgm3": 0.0,
                "fta": 0.0,
                "to": 0.0,
                "or": 0.0,
                "opp_dr": 0.0,
            },
        )
        l = teams.setdefault(
            loser,
            {
                "poss": 0.0,
                "opp_poss": 0.0,
                "points_for": 0.0,
                "points_against": 0.0,
                "fgm": 0.0,
                "fga": 0.0,
                "fgm3": 0.0,
                "fta": 0.0,
                "to": 0.0,
                "or": 0.0,
                "opp_dr": 0.0,
            },
        )

        w_poss = float(row["WFGA"]) - float(row["WOR"]) + float(row["WTO"]) + 0.475 * float(row["WFTA"])
        l_poss = float(row["LFGA"]) - float(row["LOR"]) + float(row["LTO"]) + 0.475 * float(row["LFTA"])

        w["poss"] += w_poss
        w["opp_poss"] += l_poss
        w["points_for"] += float(row["WScore"])
        w["points_against"] += float(row["LScore"])
        w["fgm"] += float(row["WFGM"])
        w["fga"] += float(row["WFGA"])
        w["fgm3"] += float(row["WFGM3"])
        w["fta"] += float(row["WFTA"])
        w["to"] += float(row["WTO"])
        w["or"] += float(row["WOR"])
        w["opp_dr"] += float(row["LDR"])

        l["poss"] += l_poss
        l["opp_poss"] += w_poss
        l["points_for"] += float(row["LScore"])
        l["points_against"] += float(row["WScore"])
        l["fgm"] += float(row["LFGM"])
        l["fga"] += float(row["LFGA"])
        l["fgm3"] += float(row["LFGM3"])
        l["fta"] += float(row["LFTA"])
        l["to"] += float(row["LTO"])
        l["or"] += float(row["LOR"])
        l["opp_dr"] += float(row["WDR"])
    return season_teams


def _load_massey_scores(path: Path) -> dict[int, dict[int, dict[str, float]]]:
    latest_day_by_system: dict[tuple[int, str], int] = {}
    for row in _read_csv(path):
        season = int(row["Season"])
        day = int(row["RankingDayNum"])
        if day > 133:
            continue
        system = row["SystemName"]
        key = (season, system)
        latest_day_by_system[key] = max(day, latest_day_by_system.get(key, -1))

    values: dict[tuple[int, int], list[float]] = {}
    raw_ranks: dict[tuple[int, int], list[float]] = {}
    for row in _read_csv(path):
        season = int(row["Season"])
        day = int(row["RankingDayNum"])
        system = row["SystemName"]
        key = (season, system)
        if latest_day_by_system.get(key) != day:
            continue
        team_id = int(row["TeamID"])
        rank = float(row["OrdinalRank"])
        normalized = 1.0 / math.sqrt(rank)
        values.setdefault((season, team_id), []).append(normalized)
        raw_ranks.setdefault((season, team_id), []).append(rank)

    scores: dict[int, dict[int, dict[str, float]]] = {}
    for (season, team_id), items in values.items():
        ranks = sorted(raw_ranks.get((season, team_id), []))
        ordered = sorted(items)
        middle = len(ordered) // 2
        if len(ordered) % 2 == 1:
            median = ordered[middle]
        else:
            median = 0.5 * (ordered[middle - 1] + ordered[middle])
        top_quartile = ordered[max(0, len(ordered) * 3 // 4)]
        rank_middle = len(ranks) // 2
        if len(ranks) % 2 == 1:
            median_rank = ranks[rank_middle]
        else:
            median_rank = 0.5 * (ranks[rank_middle - 1] + ranks[rank_middle])
        scores.setdefault(season, {})[team_id] = {
            "score": 0.7 * median + 0.3 * top_quartile,
            "mean": sum(ordered) / len(ordered),
            "best": ordered[-1],
            "count": float(len(ordered)),
            "median_rank": median_rank,
            "mean_rank": sum(ranks) / len(ranks),
            "best_rank": ranks[0],
            "top10_count": float(sum(1 for rank in ranks if rank <= 10)),
            "top25_count": float(sum(1 for rank in ranks if rank <= 25)),
        }
    return scores


def build_profiles(data_dir: Path) -> dict[str, dict[int, dict[int, TeamSeasonProfile]]]:
    compact_m = _load_compact_profiles(data_dir / "MRegularSeasonCompactResults.csv")
    compact_w = _load_compact_profiles(data_dir / "WRegularSeasonCompactResults.csv")
    detailed_m = _load_detailed_profiles(data_dir / "MRegularSeasonDetailedResults.csv")
    detailed_w = _load_detailed_profiles(data_dir / "WRegularSeasonDetailedResults.csv")
    seeds_m = _load_seeds(data_dir / "MNCAATourneySeeds.csv")
    seeds_w = _load_seeds(data_dir / "WNCAATourneySeeds.csv")
    massey_m = _load_massey_scores(data_dir / "MMasseyOrdinals.csv")
    men_conferences = _load_team_conferences(data_dir / "MTeamConferences.csv")
    women_conferences = _load_team_conferences(data_dir / "WTeamConferences.csv")
    men_active_coaches = _load_active_coaches(data_dir / "MTeamCoaches.csv")
    men_coach_tenure = _build_coach_tenure(men_active_coaches)
    men_conf_tourney_stats = _load_conference_tourney_stats(data_dir / "MConferenceTourneyGames.csv")
    women_conf_tourney_stats = _load_conference_tourney_stats(data_dir / "WConferenceTourneyGames.csv")
    men_conf_strengths = _build_conference_strengths(compact_m, men_conferences)
    women_conf_strengths = _build_conference_strengths(compact_w, women_conferences)

    def merge(
        compact: dict[int, dict[int, dict[str, float]]],
        detailed: dict[int, dict[int, dict[str, float]]],
        seeds: dict[int, dict[int, int]],
        massey: dict[int, dict[int, dict[str, float]]] | None,
        conferences: dict[int, dict[int, str]],
        conf_strengths: dict[int, dict[str, dict[str, float]]],
        conf_tourney_stats: dict[int, dict[int, dict[str, float]]],
        active_coaches: dict[int, dict[int, dict[str, Any]]] | None,
        coach_tenure: dict[int, dict[int, float]] | None,
    ) -> dict[int, dict[int, TeamSeasonProfile]]:
        profiles: dict[int, dict[int, TeamSeasonProfile]] = {}
        all_seasons = set(compact) | set(detailed) | set(seeds)
        for season in all_seasons:
            profiles[season] = {}
            team_ids = set(compact.get(season, {})) | set(detailed.get(season, {})) | set(seeds.get(season, {}))
            for team_id in team_ids:
                c = compact.get(season, {}).get(team_id, {})
                d = detailed.get(season, {}).get(team_id, {})
                games = int(c.get("games", 0.0))
                wins = int(c.get("wins", 0.0))
                losses = max(0, games - wins)
                points_for = _safe_div(c.get("points_for", 0.0), max(games, 1))
                points_against = _safe_div(c.get("points_against", 0.0), max(games, 1))
                off_eff = 100.0 * _safe_div(d.get("points_for", 0.0), d.get("poss", 0.0))
                def_eff = 100.0 * _safe_div(d.get("points_against", 0.0), d.get("opp_poss", 0.0))
                conference = conferences.get(season, {}).get(team_id)
                conference_strength = conf_strengths.get(season, {}).get(conference or "", {})
                massey_stats = massey.get(season, {}).get(team_id) if massey is not None else None
                profiles[season][team_id] = TeamSeasonProfile(
                    games=games,
                    wins=wins,
                    losses=losses,
                    points_for=points_for,
                    points_against=points_against,
                    scoring_margin=points_for - points_against,
                    elo=c.get("elo", 1500.0),
                    win_pct=_safe_div(wins, max(games, 1)),
                    off_eff=off_eff,
                    def_eff=def_eff,
                    net_eff=off_eff - def_eff,
                    efg=_safe_div(d.get("fgm", 0.0) + 0.5 * d.get("fgm3", 0.0), d.get("fga", 0.0)),
                    tov_rate=_safe_div(d.get("to", 0.0), d.get("poss", 0.0)),
                    orb_rate=_safe_div(d.get("or", 0.0), d.get("or", 0.0) + d.get("opp_dr", 0.0)),
                    ftr=_safe_div(d.get("fta", 0.0), d.get("fga", 0.0)),
                    seed=seeds.get(season, {}).get(team_id),
                    massey=massey_stats.get("score") if massey_stats is not None else None,
                    massey_mean=massey_stats.get("mean") if massey_stats is not None else None,
                    massey_best=massey_stats.get("best") if massey_stats is not None else None,
                    massey_count=massey_stats.get("count", 0.0) if massey_stats is not None else 0.0,
                    massey_median_rank=massey_stats.get("median_rank") if massey_stats is not None else None,
                    massey_mean_rank=massey_stats.get("mean_rank") if massey_stats is not None else None,
                    massey_best_rank=massey_stats.get("best_rank") if massey_stats is not None else None,
                    massey_top10_count=massey_stats.get("top10_count", 0.0) if massey_stats is not None else 0.0,
                    massey_top25_count=massey_stats.get("top25_count", 0.0) if massey_stats is not None else 0.0,
                    sos_elo=c.get("sos_elo", 1500.0),
                    sos_margin=c.get("sos_margin", 0.0),
                    conference=conference,
                    conf_elo=conference_strength.get("elo", 1500.0),
                    conf_margin=conference_strength.get("margin", 0.0),
                    conf_tourney_wins=float(conf_tourney_stats.get(season, {}).get(team_id, {}).get("wins", 0.0)),
                    conf_tourney_games=float(conf_tourney_stats.get(season, {}).get(team_id, {}).get("games", 0.0)),
                    conf_tourney_win_pct=float(conf_tourney_stats.get(season, {}).get(team_id, {}).get("win_pct", 0.0)),
                    conf_tourney_champ=float(conf_tourney_stats.get(season, {}).get(team_id, {}).get("champ", 0.0)),
                    conf_tourney_last_day=float(conf_tourney_stats.get(season, {}).get(team_id, {}).get("last_day", 0.0)),
                    conf_tourney_elo=float(conf_tourney_stats.get(season, {}).get(team_id, {}).get("elo", 1500.0)),
                    recent_win_pct=c.get("recent_win_pct", _safe_div(wins, max(games, 1))),
                    recent_margin=c.get("recent_margin", points_for - points_against),
                    recent_elo=c.get("recent_elo", c.get("elo", 1500.0)),
                    adj_net_eff=(off_eff - def_eff) + 0.04 * (c.get("sos_elo", 1500.0) - 1500.0) + 0.6 * c.get("sos_margin", 0.0),
                    loc_elo=c.get("loc_elo", c.get("elo", 1500.0)),
                    neutral_win_pct=c.get("neutral_win_pct", _safe_div(wins, max(games, 1))),
                    neutral_margin=c.get("neutral_margin_avg", points_for - points_against),
                    away_win_pct=c.get("away_win_pct", _safe_div(wins, max(games, 1))),
                    away_margin=c.get("away_margin_avg", points_for - points_against),
                    site_bias=c.get("site_bias", 0.0),
                    neutral_games=float(c.get("neutral_games", 0.0)),
                    coach_tenure=coach_tenure.get(season, {}).get(team_id, 0.0) if coach_tenure is not None else 0.0,
                    coach_change=active_coaches.get(season, {}).get(team_id, {}).get("coach_change", 0.0) if active_coaches is not None else 0.0,
                    quality_win_pct=float(c.get("quality_win_pct", 0.0)),
                    quality_margin=float(c.get("quality_margin", 0.0)),
                    quality_games=float(c.get("quality_games", 0.0)),
                    bad_loss_rate=float(c.get("bad_loss_rate", 0.0)),
                    best_win_elo=float(c.get("best_win_elo", 1500.0)),
                    margin_std=float(c.get("margin_std", 0.0)),
                    close_win_pct=float(c.get("close_win_pct", _safe_div(wins, max(games, 1)))),
                    close_games=float(c.get("close_games", 0.0)),
                    blowout_rate=float(c.get("blowout_rate", 0.0)),
                    opponent_results={
                        opp_id: (
                            _safe_div(bucket.get("wins", 0.0), max(bucket.get("games", 1.0), 1.0)),
                            _safe_div(bucket.get("margin", 0.0), max(bucket.get("games", 1.0), 1.0)),
                            float(bucket.get("games", 0.0)),
                        )
                        for opp_id, bucket in c.get("opponent_results", {}).items()
                    },
                )
        return profiles

    return {
        "men": merge(
            compact_m,
            detailed_m,
            seeds_m,
            massey_m,
            men_conferences,
            men_conf_strengths,
            men_conf_tourney_stats,
            men_active_coaches,
            men_coach_tenure,
        ),
        "women": merge(
            compact_w,
            detailed_w,
            seeds_w,
            None,
            women_conferences,
            women_conf_strengths,
            women_conf_tourney_stats,
            None,
            None,
        ),
    }


def build_matchup_features(
    profiles: dict[str, dict[int, dict[int, TeamSeasonProfile]]],
    gender: str,
    season: int,
    team_a: int,
    team_b: int,
) -> dict[str, float]:
    season_profiles = profiles[gender].get(season, {})
    a = season_profiles.get(team_a, TeamSeasonProfile())
    b = season_profiles.get(team_b, TeamSeasonProfile())

    seed_diff = 0.0
    seed_known = 0.0
    if a.seed is not None and b.seed is not None:
        seed_diff = float(b.seed - a.seed)
        seed_known = 1.0

    massey_diff = 0.0
    massey_known = 0.0
    massey_ratio = 0.0
    massey_gap_sq = 0.0
    top_rank_flag_diff = 0.0
    massey_mean_diff = 0.0
    massey_best_diff = 0.0
    massey_count_diff = 0.0
    massey_median_rank_diff = 0.0
    massey_mean_rank_diff = 0.0
    massey_best_rank_diff = 0.0
    massey_top10_count_diff = 0.0
    massey_top25_count_diff = 0.0
    if a.massey is not None and b.massey is not None:
        massey_diff = a.massey - b.massey
        massey_known = 1.0
        massey_ratio = _safe_div(a.massey, max(b.massey, EPSILON))
        massey_gap_sq = massey_diff * massey_diff
        top_rank_flag_diff = (1.0 if a.massey > 0.2 else 0.0) - (1.0 if b.massey > 0.2 else 0.0)
    if a.massey_mean is not None and b.massey_mean is not None:
        massey_mean_diff = a.massey_mean - b.massey_mean
    if a.massey_best is not None and b.massey_best is not None:
        massey_best_diff = a.massey_best - b.massey_best
    massey_count_diff = a.massey_count - b.massey_count
    if a.massey_median_rank is not None and b.massey_median_rank is not None:
        massey_median_rank_diff = b.massey_median_rank - a.massey_median_rank
    if a.massey_mean_rank is not None and b.massey_mean_rank is not None:
        massey_mean_rank_diff = b.massey_mean_rank - a.massey_mean_rank
    if a.massey_best_rank is not None and b.massey_best_rank is not None:
        massey_best_rank_diff = b.massey_best_rank - a.massey_best_rank
    massey_top10_count_diff = a.massey_top10_count - b.massey_top10_count
    massey_top25_count_diff = a.massey_top25_count - b.massey_top25_count
    seed_strength_diff = 0.0
    favorite_flag = 0.0
    if a.seed is not None and b.seed is not None:
        seed_strength_diff = _safe_div(1.0, a.seed) - _safe_div(1.0, b.seed)
        favorite_flag = 1.0 if a.seed < b.seed else 0.0
    common_opp_margin_diff = 0.0
    common_opp_win_diff = 0.0
    common_opp_count = 0.0
    head_to_head_margin = 0.0
    head_to_head_games = 0.0
    common_opponents = set(a.opponent_results or {}).intersection(set(b.opponent_results or {}))
    if common_opponents:
        common_opp_count = float(len(common_opponents))
        common_opp_margin_diff = sum(
            (a.opponent_results or {}).get(opp_id, (0.0, 0.0, 0.0))[1]
            - (b.opponent_results or {}).get(opp_id, (0.0, 0.0, 0.0))[1]
            for opp_id in common_opponents
        ) / len(common_opponents)
        common_opp_win_diff = sum(
            (a.opponent_results or {}).get(opp_id, (0.0, 0.0, 0.0))[0]
            - (b.opponent_results or {}).get(opp_id, (0.0, 0.0, 0.0))[0]
            for opp_id in common_opponents
        ) / len(common_opponents)
    head_to_head = (a.opponent_results or {}).get(team_b)
    if head_to_head is not None:
        head_to_head_margin = head_to_head[1]
        head_to_head_games = head_to_head[2]
    seed_efficiency_interaction = seed_strength_diff * (a.adj_net_eff - b.adj_net_eff)
    seed_recent_interaction = seed_strength_diff * (a.recent_margin - b.recent_margin)
    recent_rank_interaction = (a.recent_elo - b.recent_elo) * massey_diff
    head_to_head_common_opp_interaction = head_to_head_margin * common_opp_margin_diff
    efficiency_rank_interaction = (a.adj_net_eff - b.adj_net_eff) * massey_diff

    return {
        "elo_diff": a.elo - b.elo,
        "win_pct_diff": a.win_pct - b.win_pct,
        "scoring_margin_diff": a.scoring_margin - b.scoring_margin,
        "points_for_diff": a.points_for - b.points_for,
        "points_against_diff": a.points_against - b.points_against,
        "off_eff_diff": a.off_eff - b.off_eff,
        "def_eff_diff": b.def_eff - a.def_eff,
        "net_eff_diff": a.net_eff - b.net_eff,
        "efg_diff": a.efg - b.efg,
        "tov_rate_diff": b.tov_rate - a.tov_rate,
        "orb_rate_diff": a.orb_rate - b.orb_rate,
        "ftr_diff": a.ftr - b.ftr,
        "seed_diff": seed_diff,
        "seed_known": seed_known,
        "massey_diff": massey_diff,
        "massey_known": massey_known,
        "massey_ratio": massey_ratio,
        "massey_gap_sq": massey_gap_sq,
        "top_rank_flag_diff": top_rank_flag_diff,
        "massey_mean_diff": massey_mean_diff,
        "massey_best_diff": massey_best_diff,
        "massey_count_diff": massey_count_diff,
        "massey_median_rank_diff": massey_median_rank_diff,
        "massey_mean_rank_diff": massey_mean_rank_diff,
        "massey_best_rank_diff": massey_best_rank_diff,
        "massey_top10_count_diff": massey_top10_count_diff,
        "massey_top25_count_diff": massey_top25_count_diff,
        "seed_efficiency_interaction": seed_efficiency_interaction,
        "seed_recent_interaction": seed_recent_interaction,
        "recent_rank_interaction": recent_rank_interaction,
        "head_to_head_common_opp_interaction": head_to_head_common_opp_interaction,
        "efficiency_rank_interaction": efficiency_rank_interaction,
        "loc_elo_diff": a.loc_elo - b.loc_elo,
        "neutral_win_pct_diff": a.neutral_win_pct - b.neutral_win_pct,
        "neutral_margin_diff": a.neutral_margin - b.neutral_margin,
        "away_win_pct_diff": a.away_win_pct - b.away_win_pct,
        "away_margin_diff": a.away_margin - b.away_margin,
        "site_bias_diff": a.site_bias - b.site_bias,
        "neutral_games_diff": a.neutral_games - b.neutral_games,
        "coach_tenure_diff": a.coach_tenure - b.coach_tenure,
        "coach_change_diff": a.coach_change - b.coach_change,
        "quality_win_pct_diff": a.quality_win_pct - b.quality_win_pct,
        "quality_margin_diff": a.quality_margin - b.quality_margin,
        "quality_games_diff": a.quality_games - b.quality_games,
        "bad_loss_rate_diff": b.bad_loss_rate - a.bad_loss_rate,
        "best_win_elo_diff": a.best_win_elo - b.best_win_elo,
        "margin_std_diff": b.margin_std - a.margin_std,
        "close_win_pct_diff": a.close_win_pct - b.close_win_pct,
        "close_games_diff": a.close_games - b.close_games,
        "blowout_rate_diff": a.blowout_rate - b.blowout_rate,
        "games_diff": float(a.games - b.games),
        "sos_elo_diff": a.sos_elo - b.sos_elo,
        "sos_margin_diff": a.sos_margin - b.sos_margin,
        "conf_elo_diff": a.conf_elo - b.conf_elo,
        "conf_margin_diff": a.conf_margin - b.conf_margin,
        "conf_tourney_win_diff": a.conf_tourney_wins - b.conf_tourney_wins,
        "conf_tourney_games_diff": a.conf_tourney_games - b.conf_tourney_games,
        "conf_tourney_win_pct_diff": a.conf_tourney_win_pct - b.conf_tourney_win_pct,
        "conf_tourney_champ_diff": a.conf_tourney_champ - b.conf_tourney_champ,
        "conf_tourney_last_day_diff": a.conf_tourney_last_day - b.conf_tourney_last_day,
        "conf_tourney_elo_diff": a.conf_tourney_elo - b.conf_tourney_elo,
        "recent_win_pct_diff": a.recent_win_pct - b.recent_win_pct,
        "recent_margin_diff": a.recent_margin - b.recent_margin,
        "recent_elo_diff": a.recent_elo - b.recent_elo,
        "adj_net_eff_diff": a.adj_net_eff - b.adj_net_eff,
        "seed_strength_diff": seed_strength_diff,
        "favorite_flag": favorite_flag,
        "common_opp_margin_diff": common_opp_margin_diff,
        "common_opp_win_diff": common_opp_win_diff,
        "common_opp_count": common_opp_count,
        "head_to_head_margin": head_to_head_margin,
        "head_to_head_games": head_to_head_games,
    }


def _build_training_examples_for_path(
    path: Path,
    gender: str,
    profiles: dict[str, dict[int, dict[int, TeamSeasonProfile]]],
) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for row in _read_csv(path):
        season = int(row["Season"])
        winner = int(row["WTeamID"])
        loser = int(row["LTeamID"])
        team_a = min(winner, loser)
        team_b = max(winner, loser)
        label = 1.0 if winner == team_a else 0.0
        features = build_matchup_features(profiles, gender, season, team_a, team_b)
        examples.append(
            TrainingExample(
                season=season,
                gender=gender,
                team_a=team_a,
                team_b=team_b,
                label=label,
                features=features,
            )
        )
    return examples


def build_training_examples(
    data_dir: Path,
    profiles: dict[str, dict[int, dict[int, TeamSeasonProfile]]] | None = None,
) -> list[TrainingExample]:
    resolved_profiles = profiles or build_profiles(data_dir)
    examples = _build_training_examples_for_path(
        data_dir / "MNCAATourneyCompactResults.csv", "men", resolved_profiles
    )
    examples.extend(
        _build_training_examples_for_path(
            data_dir / "WNCAATourneyCompactResults.csv", "women", resolved_profiles
        )
    )
    return examples


def read_submission_rows(path: Path) -> list[tuple[str, int, int, str, int]]:
    rows: list[tuple[str, int, int, str, int]] = []
    for row in _read_csv(path):
        game_id = row["ID"]
        season_text, team_a_text, team_b_text = game_id.split("_")
        team_a = int(team_a_text)
        team_b = int(team_b_text)
        gender = "men" if 1000 <= team_a <= 1999 else "women"
        rows.append((game_id, int(season_text), team_a, team_b, gender))
    return rows


def train_logistic_model(
    examples: list[TrainingExample],
    feature_names: list[str] | None = None,
    epochs: int = 250,
    learning_rate: float = 0.05,
    l2: float = 0.001,
) -> LogisticModel:
    names = feature_names or FEATURE_NAMES
    rows = [[example.features[name] for name in names] for example in examples]
    targets = [example.label for example in examples]

    means = []
    stds = []
    for index in range(len(names)):
        column = [row[index] for row in rows]
        mean = sum(column) / max(len(column), 1)
        variance = sum((value - mean) ** 2 for value in column) / max(len(column), 1)
        std = math.sqrt(variance) if variance > EPSILON else 1.0
        means.append(mean)
        stds.append(std)

    weights = [0.0] * len(names)
    intercept = 0.0
    for _ in range(epochs):
        grad_intercept = 0.0
        grad_weights = [0.0] * len(names)
        for row, target in zip(rows, targets):
            total = intercept
            normalized = []
            for index, value in enumerate(row):
                norm_value = (value - means[index]) / stds[index]
                normalized.append(norm_value)
                total += weights[index] * norm_value
            pred = 1.0 / (1.0 + math.exp(-max(-25.0, min(25.0, total))))
            error = pred - target
            grad_intercept += error
            for index, norm_value in enumerate(normalized):
                grad_weights[index] += error * norm_value

        scale = 1.0 / max(len(rows), 1)
        intercept -= learning_rate * grad_intercept * scale
        for index in range(len(weights)):
            penalty = l2 * weights[index]
            weights[index] -= learning_rate * (grad_weights[index] * scale + penalty)

    return LogisticModel(means=means, stds=stds, weights=weights, intercept=intercept, feature_names=names)


def _candidate_thresholds(values: list[float], max_candidates: int = 16) -> list[float]:
    unique = sorted(set(values))
    if len(unique) <= 1:
        return unique
    if len(unique) <= max_candidates:
        return [(unique[i] + unique[i + 1]) * 0.5 for i in range(len(unique) - 1)]
    thresholds = []
    for index in range(1, max_candidates):
        position = int(index * len(unique) / max_candidates)
        position = min(max(position, 1), len(unique) - 1)
        thresholds.append((unique[position - 1] + unique[position]) * 0.5)
    return sorted(set(thresholds))


def train_boosted_stump_model(
    examples: list[TrainingExample],
    feature_names: list[str] | None = None,
    n_estimators: int = 32,
    learning_rate: float = 0.08,
    sample_features: int | None = 8,
    seed: int = 2026,
) -> BoostedStumpModel:
    names = feature_names or FEATURE_NAMES
    matrix = [[example.features[name] for name in names] for example in examples]
    targets = [example.label for example in examples]
    base_prob = sum(targets) / max(len(targets), 1)
    predictions = [base_prob] * len(examples)
    stumps: list[Stump] = []
    rng = random.Random(seed)

    for _ in range(n_estimators):
        residuals = [target - pred for target, pred in zip(targets, predictions)]
        feature_indexes = list(range(len(names)))
        if sample_features is not None and sample_features < len(feature_indexes):
            feature_indexes = rng.sample(feature_indexes, sample_features)

        best_loss = float("inf")
        best_stump: Stump | None = None
        for feature_index in feature_indexes:
            values = [row[feature_index] for row in matrix]
            for threshold in _candidate_thresholds(values):
                left = [residuals[i] for i, value in enumerate(values) if value <= threshold]
                right = [residuals[i] for i, value in enumerate(values) if value > threshold]
                if not left or not right:
                    continue
                left_value = sum(left) / len(left)
                right_value = sum(right) / len(right)
                loss = 0.0
                for value, residual in zip(values, residuals):
                    pred = left_value if value <= threshold else right_value
                    loss += (residual - pred) ** 2
                if loss < best_loss:
                    best_loss = loss
                    best_stump = Stump(
                        feature_index=feature_index,
                        threshold=threshold,
                        left_value=left_value,
                        right_value=right_value,
                    )

        if best_stump is None:
            break
        stumps.append(best_stump)
        for index, row in enumerate(matrix):
            if row[best_stump.feature_index] <= best_stump.threshold:
                predictions[index] += learning_rate * best_stump.left_value
            else:
                predictions[index] += learning_rate * best_stump.right_value
            predictions[index] = min(0.999, max(0.001, predictions[index]))

    return BoostedStumpModel(
        base_prob=base_prob,
        learning_rate=learning_rate,
        stumps=stumps,
        feature_names=names,
    )


def brier_score(predictions: list[float], targets: list[float]) -> float:
    if not predictions:
        return 0.0
    return sum((pred - target) ** 2 for pred, target in zip(predictions, targets)) / len(predictions)


def evaluate_by_season(
    examples: list[TrainingExample],
    trainer_name: str,
    feature_names: list[str] | None = None,
    min_test_season: int | None = None,
) -> dict[str, Any]:
    names = feature_names or FEATURE_NAMES
    seasons = sorted({example.season for example in examples})
    rows: list[dict[str, Any]] = []
    overall_predictions: list[float] = []
    overall_targets: list[float] = []

    for season in seasons:
        if min_test_season is not None and season < min_test_season:
            continue
        train_rows = [example for example in examples if example.season < season]
        test_rows = [example for example in examples if example.season == season]
        if not train_rows or not test_rows:
            continue
        if trainer_name == "logistic":
            model = train_logistic_model(train_rows, feature_names=names)
        elif trainer_name == "boosted_stumps":
            model = train_boosted_stump_model(train_rows, feature_names=names)
        else:
            raise ValueError(f"Unknown trainer: {trainer_name}")
        preds = [model.predict_prob(example.features) for example in test_rows]
        targets = [example.label for example in test_rows]
        rows.append(
            {
                "season": season,
                "examples": len(test_rows),
                "brier": round(brier_score(preds, targets), 6),
            }
        )
        overall_predictions.extend(preds)
        overall_targets.extend(targets)

    return {
        "seasons": rows,
        "overall_brier": round(brier_score(overall_predictions, overall_targets), 6),
        "feature_names": names,
    }


def write_submission(path: Path, predictions: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Pred"])
        for game_id, pred in predictions:
            writer.writerow([game_id, f"{pred:.6f}"])


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
