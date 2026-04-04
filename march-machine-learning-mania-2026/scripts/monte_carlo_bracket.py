#!/usr/bin/env python3
"""Monte Carlo tournament simulation backed by live Polymarket market data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_kaggle_seeds(data_dir: Path) -> dict[int, dict[int, int]]:
    """Load historical tournament seeds from Kaggle data."""
    seeds_by_season: dict[int, dict[int, int]] = defaultdict(dict)
    with (data_dir / "MNCAATourneySeeds.csv").open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            season = int(row["Season"])
            seed = row["Seed"]
            team_id = int(row["TeamID"])
            seed_num = int("".join(char for char in seed if char.isdigit())[:2])
            seeds_by_season[season][team_id] = seed_num
    return seeds_by_season


def load_tournament_results(data_dir: Path) -> list[dict[str, int]]:
    """Load historical tournament results."""
    results: list[dict[str, int]] = []
    with (data_dir / "MNCAATourneyCompactResults.csv").open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            results.append(
                {
                    "season": int(row["Season"]),
                    "wteam": int(row["WTeamID"]),
                    "lteam": int(row["LTeamID"]),
                    "day": int(row["DayNum"]),
                }
            )
    return results


def calculate_seed_upset_probs(
    results: list[dict[str, int]], seeds: dict[int, dict[int, int]]
) -> dict[tuple[int, int], float]:
    """Return probability that the favorite (lower seed number) wins."""
    matchup_results: dict[tuple[int, int], dict[str, int]] = defaultdict(
        lambda: {"favorite_wins": 0, "underdog_wins": 0, "total": 0}
    )

    for result in results:
        season = result["season"]
        season_seeds = seeds.get(season)
        if not season_seeds:
            continue
        wteam = result["wteam"]
        lteam = result["lteam"]
        if wteam not in season_seeds or lteam not in season_seeds:
            continue

        wseed = season_seeds[wteam]
        lseed = season_seeds[lteam]
        if wseed == lseed:
            continue

        key = (min(wseed, lseed), max(wseed, lseed))
        matchup_results[key]["total"] += 1
        if wseed < lseed:
            matchup_results[key]["favorite_wins"] += 1
        else:
            matchup_results[key]["underdog_wins"] += 1

    return {
        matchup: counts["favorite_wins"] / counts["total"]
        for matchup, counts in matchup_results.items()
        if counts["total"] > 0
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_polymarket_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected Polymarket payload in {path}")
    return payload


def build_market_strengths(payload: dict[str, Any]) -> dict[str, float]:
    futures_index = payload.get("futures_index", {})
    champion = futures_index.get("champion", {})
    one_seed = futures_index.get("number_1_seed", {})

    teams = set(champion) | set(one_seed)
    strengths: dict[str, float] = {}
    for team in teams:
        champion_prob = _safe_float(champion.get(team, {}).get("yes_prob"), 0.0)
        one_seed_prob = _safe_float(one_seed.get(team, {}).get("yes_prob"), 0.0)
        # Championship futures carry more signal; #1 seed futures provide useful shape early.
        score = champion_prob + 0.35 * one_seed_prob
        if score > 0:
            strengths[team] = score
    return strengths


def build_pairwise_implied_strengths(pairwise_market_probs: dict[tuple[str, str], float]) -> dict[str, float]:
    """Approximate team strength from available pairwise probabilities."""
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for (team1, team2), prob in pairwise_market_probs.items():
        if not team1 or not team2:
            continue
        p = max(0.0, min(1.0, prob))
        totals[team1] += p
        counts[team1] += 1
        totals[team2] += 1.0 - p
        counts[team2] += 1
    return {
        team: totals[team] / counts[team]
        for team in totals
        if counts[team] > 0
    }


def build_pairwise_market_probs(payload: dict[str, Any]) -> dict[tuple[str, str], float]:
    pairwise_index = payload.get("pairwise_index", {})
    market_probs: dict[tuple[str, str], float] = {}
    for value in pairwise_index.values():
        team1 = str(value.get("team1", "")).strip()
        team2 = str(value.get("team2", "")).strip()
        if not team1 or not team2:
            continue
        market_probs[(team1, team2)] = _safe_float(value.get("team1_prob"), 0.0)
    return market_probs


def approximate_seeds_from_strengths(strengths: dict[str, float]) -> dict[str, int]:
    ordered = sorted(strengths.items(), key=lambda item: item[1], reverse=True)
    seeds: dict[str, int] = {}
    for index, (team, _) in enumerate(ordered):
        if index < 4:
            seeds[team] = 1
        elif index < 8:
            seeds[team] = 2
        elif index < 16:
            seeds[team] = 3
        elif index < 24:
            seeds[team] = 4
        elif index < 32:
            seeds[team] = 5
        elif index < 40:
            seeds[team] = 6
        elif index < 48:
            seeds[team] = 7
        elif index < 56:
            seeds[team] = 8
        else:
            seeds[team] = 9
    return seeds


def _blend(prob_a: float, prob_b: float, weight_a: float) -> float:
    return weight_a * prob_a + (1.0 - weight_a) * prob_b


def _relative_strength_prob(team: str, opponent: str, strengths: dict[str, float]) -> float:
    team_strength = strengths.get(team, 0.0)
    opponent_strength = strengths.get(opponent, 0.0)
    total = team_strength + opponent_strength
    if total <= 0:
        return 0.5
    return team_strength / total


def get_team_win_probability(
    team: str,
    opponent: str,
    pairwise_market_probs: dict[tuple[str, str], float],
    market_strengths: dict[str, float],
    seed_probs: dict[tuple[int, int], float],
    team_seed: int,
    opponent_seed: int,
) -> float:
    if (team, opponent) in pairwise_market_probs:
        market_prob = pairwise_market_probs[(team, opponent)]
    elif (opponent, team) in pairwise_market_probs:
        market_prob = 1.0 - pairwise_market_probs[(opponent, team)]
    else:
        market_prob = _relative_strength_prob(team, opponent, market_strengths)

    seed_adjustment = 0.5
    if team_seed and opponent_seed and team_seed != opponent_seed:
        seed_key = (min(team_seed, opponent_seed), max(team_seed, opponent_seed))
        favorite_prob = seed_probs.get(seed_key, 0.5)
        if team_seed < opponent_seed:
            seed_adjustment = favorite_prob
        else:
            seed_adjustment = 1.0 - favorite_prob

    # Keep market as primary signal; historical seed rates only nudge it.
    final_prob = _blend(market_prob, seed_adjustment, 0.88)
    return max(0.01, min(0.99, final_prob))


def simulate_game(
    team1: str,
    team2: str,
    team1_seed: int,
    team2_seed: int,
    pairwise_market_probs: dict[tuple[str, str], float],
    market_strengths: dict[str, float],
    seed_probs: dict[tuple[int, int], float],
) -> str:
    prob1 = get_team_win_probability(
        team1,
        team2,
        pairwise_market_probs,
        market_strengths,
        seed_probs,
        team1_seed,
        team2_seed,
    )
    return team1 if random.random() < prob1 else team2


def simulate_bracket(
    teams: list[str],
    pairwise_market_probs: dict[tuple[str, str], float],
    market_strengths: dict[str, float],
    seed_probs: dict[tuple[int, int], float],
    team_seeds: dict[str, int],
    num_simulations: int,
) -> dict[str, int]:
    win_counts: dict[str, int] = defaultdict(int)
    if len(teams) < 2:
        return win_counts

    for _ in range(num_simulations):
        shuffled = teams.copy()
        random.shuffle(shuffled)
        if len(shuffled) < 64:
            shuffled = (shuffled * ((64 // len(shuffled)) + 1))[:64]

        current_round = shuffled[:64]
        while len(current_round) > 1:
            next_round: list[str] = []
            for index in range(0, len(current_round) - 1, 2):
                team1 = current_round[index]
                team2 = current_round[index + 1]
                winner = simulate_game(
                    team1,
                    team2,
                    team_seeds.get(team1, 8),
                    team_seeds.get(team2, 8),
                    pairwise_market_probs,
                    market_strengths,
                    seed_probs,
                )
                next_round.append(winner)
            current_round = next_round

        if current_round:
            win_counts[current_round[0]] += 1

    return win_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulation using live Polymarket data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--polymarket-json",
        type=Path,
        default=Path("reports/polymarket_ncaab_matchups.json"),
        help="Normalized Polymarket payload generated by scrape_polymarket_ncaab.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/monte_carlo_results.json"),
    )
    parser.add_argument("--num-simulations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("March Madness 2026 Monte Carlo Simulation")
    print("Using Polymarket live matchup + futures data")
    print("=" * 60)

    print("\n[1] Loading Kaggle historical data...")
    seeds = load_kaggle_seeds(args.data_dir)
    results = load_tournament_results(args.data_dir)
    seasons = [row["season"] for row in results]
    print(f"    Loaded {len(results)} tournament games from {min(seasons)}-{max(seasons)}")

    print("\n[2] Calculating historical seed upset probabilities...")
    seed_probs = calculate_seed_upset_probs(results, seeds)

    print("\n[3] Loading Polymarket market data...")
    payload = load_polymarket_payload(args.polymarket_json)
    pairwise_market_probs = build_pairwise_market_probs(payload)
    market_strengths = build_market_strengths(payload)
    if not market_strengths:
        market_strengths = build_pairwise_implied_strengths(pairwise_market_probs)
        if not market_strengths:
            raise ValueError(
                f"No usable matchup or futures markets were found in {args.polymarket_json}. Re-run scrape_polymarket_ncaab.py."
            )
        print("    Futures markets unavailable; using pairwise-implied strengths.")

    print(f"    Direct matchup markets: {len(pairwise_market_probs)}")
    print(f"    Teams with futures-derived strength: {len(market_strengths)}")

    team_seeds = approximate_seeds_from_strengths(market_strengths)
    teams = [team for team, _ in sorted(market_strengths.items(), key=lambda item: item[1], reverse=True)]

    print("\n[4] Running Monte Carlo simulation...")
    win_counts = simulate_bracket(
        teams=teams,
        pairwise_market_probs=pairwise_market_probs,
        market_strengths=market_strengths,
        seed_probs=seed_probs,
        team_seeds=team_seeds,
        num_simulations=args.num_simulations,
    )

    total_sims = sum(win_counts.values())
    sorted_wins = sorted(win_counts.items(), key=lambda item: item[1], reverse=True)

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS - Championship Win Probability")
    print("=" * 60)
    print(f"{'Team':<22} {'Sim Win %':>10} {'Market Str':>10} {'Seed':>6}")
    print("-" * 60)
    for team, wins in sorted_wins[:20]:
        sim_pct = wins / total_sims * 100 if total_sims else 0.0
        print(
            f"{team:<22} {sim_pct:>9.2f}% {market_strengths.get(team, 0.0):>10.4f} {team_seeds.get(team, 8):>6}"
        )

    output = {
        "simulation_iterations": total_sims,
        "polymarket_source": str(args.polymarket_json),
        "market_strengths": market_strengths,
        "pairwise_market_count": len(pairwise_market_probs),
        "approximate_team_seeds": team_seeds,
        "win_probabilities": {
            team: wins / total_sims for team, wins in sorted_wins if total_sims > 0
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
