#!/usr/bin/env python3
"""Scan Polymarket NCAAB markets and normalize them into matchup probabilities.

This script uses Polymarket's public Gamma API rather than scraping HTML.
It discovers active college basketball events, extracts head-to-head winner
markets, resolves team names to Kaggle team IDs, and writes normalized outputs
for downstream Monte Carlo usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_LIMIT = 100
YES_TOKENS = {"yes", "y"}
NO_TOKENS = {"no", "n"}
HOME_AWAY_TOKENS = {"home", "away"}
SKIP_MARKET_KEYWORDS = (
    "spread",
    "total",
    "over/under",
    "moneyline spread",
    "rebounds",
    "assists",
    "player",
    "parlay",
    "points",
    "3-pointers",
    "three-pointers",
    "margin",
    "halftime",
    "1st half",
    "2nd half",
    "quarter",
    "period",
)
COLLEGE_BASKETBALL_HINTS = (
    "ncaab",
    "college basketball",
    "ncaa basketball",
    "march madness",
    "ncaa tournament",
)
SPORT_NAME_HINTS = (
    "ncaab",
    "ncaa basketball",
    "college basketball",
)
COLLEGE_TAG_HINTS = (
    "college-basketball",
    "college basketball",
    "cbb",
    "march-madness",
    "march madness",
    "ncaa-basketball",
    "ncaa basketball",
    "ncaa-cbb",
    "ncaa tournament",
)
TEAM_NAME_OVERRIDES = {
    "uconn": "Connecticut",
    "u conn": "Connecticut",
    "u-conn": "Connecticut",
    "unc": "North Carolina",
    "unc chapel hill": "North Carolina",
    "unc wilmington": "UNC Wilmington",
    "ole miss": "Mississippi",
    "lsu": "Louisiana St",
    "byu": "BYU",
    "smu": "SMU",
    "vcu": "VCU",
    "usc": "USC",
    "ucla": "UCLA",
    "uc san diego": "UC San Diego",
    "uc irvine": "UC Irvine",
    "saint marys": "Saint Mary's CA",
    "saint mary's": "Saint Mary's CA",
    "st marys": "Saint Mary's CA",
    "st mary's": "Saint Mary's CA",
    "st johns": "St John's",
    "st john's": "St John's",
    "pitt": "Pittsburgh",
    "penn": "Penn",
    "miami fl": "Miami FL",
    "miami florida": "Miami FL",
    "miami hurricanes": "Miami FL",
    "southern utah thunderbirds": "Southern Utah",
    "texas am": "Texas A&M",
    "texas a&m": "Texas A&M",
    "texas a and m": "Texas A&M",
}


@dataclass
class MatchupMarket:
    event_id: str
    market_id: str
    event_title: str
    market_question: str
    market_slug: str
    start_date: str | None
    end_date: str | None
    team1: str
    team1_id: int | None
    team1_prob: float
    team2: str
    team2_id: int | None
    team2_prob: float
    team1_raw: str
    team2_raw: str
    source_outcomes: list[str]
    source_prices: list[float]
    gamma_source_prices: list[float]
    clob_token_ids: list[str]
    clob_source_prices: list[float]
    price_source: str
    market_liquidity: float | None
    category: str | None
    sport: str | None
    league: str | None
    event_slug: str | None


@dataclass
class FutureMarket:
    event_id: str
    market_id: str
    event_title: str
    market_question: str
    market_slug: str
    future_type: str
    start_date: str | None
    end_date: str | None
    team: str
    team_id: int | None
    yes_prob: float
    no_prob: float
    team_raw: str
    source_outcomes: list[str]
    source_prices: list[float]
    gamma_source_prices: list[float]
    clob_token_ids: list[str]
    clob_source_prices: list[float]
    price_source: str
    market_liquidity: float | None
    category: str | None
    sport: str | None
    league: str | None
    event_slug: str | None


MATCHUP_FIELDNAMES = [field.name for field in fields(MatchupMarket)]
FUTURE_FIELDNAMES = [field.name for field in fields(FutureMarket)]
GENERIC_TAG_IDS = {1}
FUTURE_TYPE_PATTERNS = [
    ("champion", re.compile(r"\bwin\b.*\b(?:tournament|march madness|championship)\b", re.I)),
    ("number_1_seed", re.compile(r"\bnumber 1 seed\b|\b#1 seed\b", re.I)),
    ("final_four", re.compile(r"\bfinal four\b", re.I)),
    ("elite_eight", re.compile(r"\belite 8\b|\belite eight\b", re.I)),
    ("sweet_sixteen", re.compile(r"\bsweet 16\b|\bsweet sixteen\b", re.I)),
    ("make_tournament", re.compile(r"\bmake\b.*\bncaa\b.*\btournament\b", re.I)),
]


def _normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = value.replace("&", " and ")
    value = re.sub(r"\bmen'?s\b", " ", value)
    value = re.sub(r"\bwomen'?s\b", " ", value)
    value = re.sub(r"\bno\.\s*\d+\b", " ", value)
    value = re.sub(r"\(\d+\)", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _clean_team_fragment(value: str) -> str:
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\b(?:men'?s|women'?s|college basketball|ncaab|ncaa)\b", " ", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" -:|?")
    return value


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_token_ids(value: Any) -> list[str]:
    return [str(item).strip() for item in _parse_json_array(value) if str(item).strip()]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _market_liquidity(market: dict[str, Any]) -> float | None:
    return _safe_float(
        _first_non_empty(
            market.get("liquidityClob"),
            market.get("liquidityNum"),
            market.get("liquidity"),
        )
    )


class GammaClient:
    def __init__(self, base_url: str = GAMMA_API_BASE, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None}, doseq=True)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "march-machine-learning-mania-2026/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"Gamma API request failed: {exc.code} {exc.reason} for {url}") from exc
        except URLError as exc:
            raise RuntimeError(f"Gamma API request failed: {exc.reason} for {url}") from exc


class ClobClient:
    def __init__(self, base_url: str = CLOB_API_BASE, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urlencode({k: v for k, v in (params or {}).items() if v is not None}, doseq=True)
        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "march-machine-learning-mania-2026/1.0",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            raise RuntimeError(f"CLOB API request failed: {exc.code} {exc.reason} for {url}") from exc
        except URLError as exc:
            raise RuntimeError(f"CLOB API request failed: {exc.reason} for {url}") from exc

    def get_midpoints(self, token_ids: list[str]) -> dict[str, float]:
        midpoints: dict[str, float] = {}
        for token_id in token_ids:
            try:
                response = self.get_json("/midpoint", {"token_id": token_id})
            except RuntimeError:
                continue
            if not isinstance(response, dict):
                continue
            parsed = _safe_float(response.get("mid"))
            if parsed is not None:
                midpoints[str(token_id)] = parsed
        return midpoints


def _pick_market_prices(
    market: dict[str, Any],
    outcomes: list[str],
    clob_midpoints: dict[str, float] | None,
) -> tuple[list[float], list[float], list[str], list[float], str]:
    gamma_prices = [_safe_float(item) for item in _parse_json_array(market.get("outcomePrices"))]
    if len(outcomes) != 2 or len(gamma_prices) != 2 or any(price is None for price in gamma_prices):
        return [], [], [], [], "unavailable"

    token_ids = _parse_token_ids(market.get("clobTokenIds"))[:len(outcomes)]
    clob_prices = [clob_midpoints.get(token_id) for token_id in token_ids] if clob_midpoints else []
    if len(clob_prices) == len(outcomes) and all(price is not None for price in clob_prices):
        return (
            [float(price) for price in clob_prices],
            [float(price) for price in gamma_prices],
            token_ids,
            [float(price) for price in clob_prices],
            "clob_midpoints",
        )

    return (
        [float(price) for price in gamma_prices],
        [float(price) for price in gamma_prices],
        token_ids,
        [float(price) for price in clob_prices if price is not None],
        "gamma_outcome_prices",
    )


class TeamResolver:
    def __init__(self, data_dir: Path) -> None:
        self._teams_by_norm: dict[str, tuple[str, int]] = {}
        self._token_index: dict[str, set[str]] = {}
        self._load_teams(data_dir)

    def _load_teams(self, data_dir: Path) -> None:
        teams_path = data_dir / "MTeams.csv"
        spellings_path = data_dir / "MTeamSpellings.csv"
        canonical_names: dict[int, str] = {}

        with teams_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                team_id = int(row["TeamID"])
                team_name = row["TeamName"]
                canonical_names[team_id] = team_name
                self._register_alias(team_name, team_name, team_id)

        with spellings_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                team_id = int(row["TeamID"])
                team_name = canonical_names.get(team_id)
                if not team_name:
                    continue
                self._register_alias(row["TeamNameSpelling"], team_name, team_id)

        for alias, canonical in TEAM_NAME_OVERRIDES.items():
            resolved = self.resolve(canonical)
            if resolved is None:
                continue
            self._register_alias(alias, resolved[0], resolved[1])

    def _register_alias(self, alias: str, canonical_name: str, team_id: int) -> None:
        norm = _normalize_name(alias)
        if not norm:
            return
        self._teams_by_norm[norm] = (canonical_name, team_id)
        for token in norm.split():
            if len(token) < 3:
                continue
            self._token_index.setdefault(token, set()).add(norm)

    def resolve(self, raw_name: str) -> tuple[str, int] | None:
        cleaned = _clean_team_fragment(raw_name)
        norm = _normalize_name(cleaned)
        if not norm:
            return None
        direct = self._teams_by_norm.get(norm)
        if direct is not None:
            return direct

        # Prefer the longest contained alias when outcomes add mascot text,
        # such as "Florida State Seminoles" or "Kansas State Wildcats".
        contained_matches = [
            (key, value)
            for key, value in self._teams_by_norm.items()
            if key and (norm.startswith(key) or norm.endswith(key) or f" {key} " in f" {norm} ")
        ]
        if contained_matches:
            max_tokens = max(len(key.split()) for key, _ in contained_matches)
            best_matches = [value for key, value in contained_matches if len(key.split()) == max_tokens]
            unique_best = _unique_value(best_matches)
            if unique_best is not None:
                return unique_best

        candidates: set[str] = set()
        for token in norm.split():
            candidates.update(self._token_index.get(token, set()))
        if not candidates:
            return None

        ranked = sorted(
            candidates,
            key=lambda candidate: _token_overlap_score(norm, candidate),
            reverse=True,
        )
        if not ranked:
            return None
        best = ranked[0]
        best_score = _token_overlap_score(norm, best)
        if best_score < 0.75:
            return None
        if len(ranked) > 1 and best_score - _token_overlap_score(norm, ranked[1]) < 0.15:
            return None
        return self._teams_by_norm[best]


def _token_overlap_score(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    return intersection / max(len(left_tokens), len(right_tokens))


def _unique_value(values: list[tuple[str, int]]) -> tuple[str, int] | None:
    if not values:
        return None
    first = values[0]
    if all(value == first for value in values[1:]):
        return first
    return None


def discover_ncaab_tags(client: GammaClient) -> list[int]:
    sports = client.get_json("/sports")
    tag_ids: set[int] = set()
    for item in sports:
        sport = str(item.get("sport", "")).lower()
        if not any(hint in sport for hint in SPORT_NAME_HINTS):
            continue
        raw_tags = str(item.get("tags", "")).strip()
        for part in raw_tags.split(","):
            part = part.strip()
            if part.isdigit() and int(part) not in GENERIC_TAG_IDS:
                tag_ids.add(int(part))
    return sorted(tag_ids)


def discover_ncaab_series_ids(client: GammaClient) -> list[int]:
    sports = client.get_json("/sports")
    series_ids: set[int] = set()
    for item in sports:
        sport = str(item.get("sport", "")).lower()
        if not any(hint in sport for hint in SPORT_NAME_HINTS):
            continue
        raw_series = str(item.get("series", "")).strip()
        for part in raw_series.split(","):
            part = part.strip()
            if part.isdigit():
                series_ids.add(int(part))
    return sorted(series_ids)


def fetch_events(
    client: GammaClient,
    *,
    max_pages: int,
    series_id: int | None = None,
    tag_id: int | None = None,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    offset = 0
    for _ in range(max_pages):
        params: dict[str, Any] = {
            "closed": False,
            "limit": DEFAULT_LIMIT,
            "offset": offset,
        }
        if series_id is not None:
            params["series_id"] = series_id
            params["active"] = True
            params["order"] = "start_date"
            params["ascending"] = True
        elif tag_id is not None:
            params["tag_id"] = tag_id
            params["related_tags"] = True
            params["active"] = True
            params["order"] = "id"
            params["ascending"] = False
        else:
            break

        try:
            page = client.get_json("/events", params)
        except RuntimeError as exc:
            # Gamma can reject some order fields or filter combinations with 422.
            # Retry once with the minimal supported filter set.
            if "422" not in str(exc):
                raise
            retry_params = {
                key: value
                for key, value in params.items()
                if key not in {"order", "ascending", "active"}
            }
            page = client.get_json("/events", retry_params)
        if not isinstance(page, list) or not page:
            break
        events.extend(page)
        if len(page) < DEFAULT_LIMIT:
            break
        offset += DEFAULT_LIMIT
    return events


def collect_ncaab_events(
    client: GammaClient,
    explicit_tags: list[int],
    explicit_series_ids: list[int],
    max_pages: int,
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    series_ids = explicit_series_ids or discover_ncaab_series_ids(client)
    tag_ids = explicit_tags or discover_ncaab_tags(client)
    if not series_ids and not tag_ids:
        raise RuntimeError("Could not discover any NCAAB Polymarket series IDs or tag IDs from /sports.")

    deduped: dict[str, dict[str, Any]] = {}
    for series_id in series_ids:
        for event in fetch_events(client, series_id=series_id, max_pages=max_pages):
            event_id = str(event.get("id", ""))
            if event_id:
                deduped[event_id] = event
    for tag_id in sorted(set(tag_ids)):
        for event in fetch_events(client, tag_id=tag_id, max_pages=max_pages):
            event_id = str(event.get("id", ""))
            if event_id:
                deduped[event_id] = event
    return list(deduped.values()), {"series_ids": series_ids, "tag_ids": tag_ids}


def event_looks_like_college_basketball(
    event: dict[str, Any],
    expected_tag_ids: set[int] | None = None,
) -> bool:
    if _event_sport_looks_like_college_basketball(event):
        return True
    haystack = " ".join(
        str(event.get(key, ""))
        for key in ("title", "slug", "category", "sub_title", "seriesSlug")
    ).lower()
    if any(hint in haystack for hint in COLLEGE_BASKETBALL_HINTS):
        return True
    if _event_has_college_tag_text(event):
        return True
    # Do not trust tag-only matches unless sport/category text also resembles CBB.
    if expected_tag_ids and _event_has_any_tag(event, expected_tag_ids) and "basketball" in haystack:
        return True
    return any(_market_looks_like_college_basketball(market) for market in event.get("markets", []) or [])


def _event_has_any_tag(event: dict[str, Any], expected_tag_ids: set[int]) -> bool:
    for tag in event.get("tags", []) or []:
        raw_id = tag.get("id") if isinstance(tag, dict) else tag
        try:
            tag_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if tag_id in expected_tag_ids:
            return True
    return False


def _event_has_college_tag_text(event: dict[str, Any]) -> bool:
    for tag in event.get("tags", []) or []:
        if isinstance(tag, dict):
            haystack = " ".join(str(tag.get(key, "")) for key in ("label", "slug")).lower()
        else:
            haystack = str(tag).lower()
        if any(hint in haystack for hint in COLLEGE_TAG_HINTS):
            return True
    return False


def market_looks_like_target_context(
    event: dict[str, Any],
    market: dict[str, Any],
    expected_tag_ids: set[int] | None,
) -> bool:
    if _market_looks_like_college_basketball(market):
        return True
    if _event_sport_looks_like_college_basketball(event):
        return True
    if _event_has_college_tag_text(event):
        return True
    if expected_tag_ids and _event_has_any_tag(event, expected_tag_ids):
        event_haystack = " ".join(
            str(event.get(key, ""))
            for key in ("title", "slug", "category", "sub_title", "seriesSlug")
        ).lower()
        if "basketball" in event_haystack:
            return True
    return False


def _event_sport_looks_like_college_basketball(event: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(event.get(key, ""))
        for key in ("sport", "category", "seriesSlug", "sub_title")
    ).lower()
    if any(hint in haystack for hint in SPORT_NAME_HINTS):
        return True
    return False


def _future_type(question: str, event_title: str) -> str | None:
    haystack = f"{question} {event_title}"
    for name, pattern in FUTURE_TYPE_PATTERNS:
        if pattern.search(haystack):
            return name
    return None


def _extract_future_team(question: str, market: dict[str, Any]) -> str | None:
    group_title = _maybe_str(market.get("groupItemTitle"))
    if group_title:
        return group_title
    match = re.search(r"will\s+(.+?)\s+(?:be|win|make|reach)\b", question, re.I)
    if match:
        return _clean_team_fragment(match.group(1))
    return None


def resolve_future_market(
    event: dict[str, Any],
    market: dict[str, Any],
    resolver: TeamResolver,
    clob_midpoints: dict[str, float] | None = None,
) -> FutureMarket | None:
    outcomes = [str(item).strip() for item in _parse_json_array(market.get("outcomes"))]
    prices, gamma_prices, token_ids, clob_prices, price_source = _pick_market_prices(market, outcomes, clob_midpoints)
    if len(prices) != 2:
        return None

    normalized_outcomes = [_normalize_name(outcome) for outcome in outcomes]
    if set(normalized_outcomes) != {"yes", "no"}:
        return None

    question = str(market.get("question", ""))
    future_type = _future_type(question, str(event.get("title", "")))
    if future_type is None:
        return None

    raw_team = _extract_future_team(question, market)
    if raw_team is None:
        return None
    resolved = resolver.resolve(raw_team)
    if resolved is None:
        return None

    yes_index = normalized_outcomes.index("yes")
    no_index = normalized_outcomes.index("no")
    yes_prob = float(prices[yes_index])
    no_prob = float(prices[no_index])
    total = yes_prob + no_prob
    if total <= 0:
        return None
    yes_prob /= total
    no_prob /= total

    return FutureMarket(
        event_id=str(event.get("id", "")),
        market_id=str(market.get("id", "")),
        event_title=str(event.get("title", "")),
        market_question=question,
        market_slug=str(market.get("slug", "")),
        future_type=future_type,
        start_date=_first_non_empty(market.get("startDate"), market.get("startDateIso"), event.get("startDate")),
        end_date=_first_non_empty(market.get("endDate"), market.get("endDateIso"), event.get("endDate")),
        team=resolved[0],
        team_id=resolved[1],
        yes_prob=round(yes_prob, 6),
        no_prob=round(no_prob, 6),
        team_raw=raw_team,
        source_outcomes=outcomes,
        source_prices=[round(float(prices[0]), 6), round(float(prices[1]), 6)],
        gamma_source_prices=[round(float(gamma_prices[0]), 6), round(float(gamma_prices[1]), 6)],
        clob_token_ids=token_ids,
        clob_source_prices=[round(float(price), 6) for price in clob_prices],
        price_source=price_source,
        market_liquidity=_market_liquidity(market),
        category=_maybe_str(event.get("category")),
        sport=_maybe_str(event.get("sport")),
        league=_maybe_str(market.get("sportsMarketType")),
        event_slug=_maybe_str(event.get("slug")),
    )


def _market_looks_like_college_basketball(market: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(market.get(key, ""))
        for key in ("question", "slug", "sportsMarketType", "groupItemTitle", "subtitle")
    ).lower()
    return any(hint in haystack for hint in COLLEGE_BASKETBALL_HINTS)


def market_is_supported(market: dict[str, Any]) -> bool:
    if not market.get("active", True) or market.get("closed", False):
        return False

    outcomes = [str(item).strip() for item in _parse_json_array(market.get("outcomes"))]
    prices = [_safe_float(item) for item in _parse_json_array(market.get("outcomePrices"))]
    if len(outcomes) != 2 or len(prices) != 2 or any(price is None for price in prices):
        return False

    haystack = " ".join(
        str(market.get(key, ""))
        for key in ("question", "slug", "sportsMarketType", "groupItemTitle")
    ).lower()
    if any(keyword in haystack for keyword in SKIP_MARKET_KEYWORDS):
        return False
    return True


def extract_matchup_from_text(text: str) -> tuple[str, str] | None:
    cleaned = re.sub(r"\s+", " ", text).strip()
    patterns = [
        re.compile(r"(.+?)\s+(?:vs\.?|v\.?|@|at)\s+(.+?)(?:\?|$)", re.I),
        re.compile(r"will\s+(.+?)\s+beat\s+(.+?)(?:\?|$)", re.I),
        re.compile(r"who\s+will\s+win[: ]+(.+?)\s+(?:vs\.?|v\.?)\s+(.+?)(?:\?|$)", re.I),
    ]
    for pattern in patterns:
        match = pattern.search(cleaned)
        if not match:
            continue
        return _clean_team_fragment(match.group(1)), _clean_team_fragment(match.group(2))
    return None


def resolve_market_matchup(
    event: dict[str, Any],
    market: dict[str, Any],
    resolver: TeamResolver,
    clob_midpoints: dict[str, float] | None = None,
) -> MatchupMarket | None:
    outcomes = [str(item).strip() for item in _parse_json_array(market.get("outcomes"))]
    prices, gamma_prices, token_ids, clob_prices, price_source = _pick_market_prices(market, outcomes, clob_midpoints)
    if len(prices) != 2:
        return None

    team1_raw: str | None = None
    team2_raw: str | None = None
    team1_prob: float | None = None
    team2_prob: float | None = None

    normalized_outcomes = [_normalize_name(outcome) for outcome in outcomes]
    question = str(market.get("question") or event.get("title") or "")
    matchup_from_text = extract_matchup_from_text(question)
    if matchup_from_text is None:
        matchup_from_text = extract_matchup_from_text(str(event.get("title", "")))

    if all(outcome not in YES_TOKENS | NO_TOKENS | HOME_AWAY_TOKENS for outcome in normalized_outcomes):
        team1_raw, team2_raw = outcomes
        team1_prob = float(prices[0])
        team2_prob = float(prices[1])
    elif matchup_from_text is not None:
        left_team, right_team = matchup_from_text
        if normalized_outcomes == ["yes", "no"]:
            team1_raw, team2_raw = left_team, right_team
            team1_prob = float(prices[0])
            team2_prob = float(prices[1])
        elif normalized_outcomes == ["no", "yes"]:
            team1_raw, team2_raw = left_team, right_team
            team1_prob = float(prices[1])
            team2_prob = float(prices[0])
        elif normalized_outcomes == ["home", "away"]:
            team1_raw, team2_raw = left_team, right_team
            team1_prob = float(prices[0])
            team2_prob = float(prices[1])
        elif normalized_outcomes == ["away", "home"]:
            team1_raw, team2_raw = left_team, right_team
            team1_prob = float(prices[1])
            team2_prob = float(prices[0])

    if team1_raw is None or team2_raw is None or team1_prob is None or team2_prob is None:
        return None

    resolved1 = resolver.resolve(team1_raw)
    resolved2 = resolver.resolve(team2_raw)
    if resolved1 is None or resolved2 is None:
        return None
    if resolved1[1] == resolved2[1]:
        return None

    prob_sum = team1_prob + team2_prob
    if prob_sum <= 0:
        return None

    team1_prob = team1_prob / prob_sum
    team2_prob = team2_prob / prob_sum

    return MatchupMarket(
        event_id=str(event.get("id", "")),
        market_id=str(market.get("id", "")),
        event_title=str(event.get("title", "")),
        market_question=str(market.get("question", "")),
        market_slug=str(market.get("slug", "")),
        start_date=_first_non_empty(market.get("startDate"), market.get("startDateIso"), event.get("startDate")),
        end_date=_first_non_empty(market.get("endDate"), market.get("endDateIso"), event.get("endDate")),
        team1=resolved1[0],
        team1_id=resolved1[1],
        team1_prob=round(team1_prob, 6),
        team2=resolved2[0],
        team2_id=resolved2[1],
        team2_prob=round(team2_prob, 6),
        team1_raw=team1_raw,
        team2_raw=team2_raw,
        source_outcomes=outcomes,
        source_prices=[round(float(prices[0]), 6), round(float(prices[1]), 6)],
        gamma_source_prices=[round(float(gamma_prices[0]), 6), round(float(gamma_prices[1]), 6)],
        clob_token_ids=token_ids,
        clob_source_prices=[round(float(price), 6) for price in clob_prices],
        price_source=price_source,
        market_liquidity=_market_liquidity(market),
        category=_maybe_str(event.get("category")),
        sport=_maybe_str(event.get("sport")),
        league=_maybe_str(market.get("sportsMarketType")),
        event_slug=_maybe_str(event.get("slug")),
    )


def _maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_pairwise_index(markets: list[MatchupMarket]) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for market in markets:
        key = f"{market.team1}__vs__{market.team2}"
        reverse_key = f"{market.team2}__vs__{market.team1}"
        payload = {
            "event_id": market.event_id,
            "market_id": market.market_id,
            "event_title": market.event_title,
            "market_question": market.market_question,
            "team1": market.team1,
            "team1_id": market.team1_id,
            "team1_prob": market.team1_prob,
            "team2": market.team2,
            "team2_id": market.team2_id,
            "team2_prob": market.team2_prob,
            "start_date": market.start_date,
            "end_date": market.end_date,
            "price_source": market.price_source,
            "source_prices": market.source_prices,
            "gamma_source_prices": market.gamma_source_prices,
            "clob_source_prices": market.clob_source_prices,
            "clob_token_ids": market.clob_token_ids,
            "market_liquidity": market.market_liquidity,
        }
        pairs[key] = payload
        pairs[reverse_key] = {
            **payload,
            "team1": market.team2,
            "team1_id": market.team2_id,
            "team1_prob": market.team2_prob,
            "team2": market.team1,
            "team2_id": market.team1_id,
            "team2_prob": market.team1_prob,
        }
    return pairs


def build_futures_index(markets: list[FutureMarket]) -> dict[str, dict[str, dict[str, Any]]]:
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for market in markets:
        index.setdefault(market.future_type, {})[market.team] = {
            "event_id": market.event_id,
            "market_id": market.market_id,
            "event_title": market.event_title,
            "market_question": market.market_question,
            "team": market.team,
            "team_id": market.team_id,
            "yes_prob": market.yes_prob,
            "no_prob": market.no_prob,
            "start_date": market.start_date,
            "end_date": market.end_date,
            "price_source": market.price_source,
            "source_prices": market.source_prices,
            "gamma_source_prices": market.gamma_source_prices,
            "clob_source_prices": market.clob_source_prices,
            "clob_token_ids": market.clob_token_ids,
            "market_liquidity": market.market_liquidity,
        }
    return index


def write_csv(path: Path, markets: list[MatchupMarket]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(market) for market in markets]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MATCHUP_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_futures_csv(path: Path, markets: list[FutureMarket]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(market) for market in markets]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FUTURE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan Polymarket college basketball markets and normalize matchup odds."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing MTeams.csv and MTeamSpellings.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/polymarket_ncaab_matchups.json"),
        help="Path for normalized JSON output.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("reports/polymarket_ncaab_matchups.csv"),
        help="Path for normalized CSV output.",
    )
    parser.add_argument(
        "--futures-output-csv",
        type=Path,
        default=Path("reports/polymarket_ncaab_futures.csv"),
        help="Path for normalized futures CSV output.",
    )
    parser.add_argument(
        "--raw-json",
        type=Path,
        default=Path("reports/polymarket_ncaab_events_raw.json"),
        help="Path for the raw event payload snapshot.",
    )
    parser.add_argument(
        "--input-raw-json",
        type=Path,
        default=None,
        help="Optional pre-fetched raw Polymarket events JSON. If provided, no network requests are made.",
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        action="append",
        default=[],
        help="Explicit Polymarket tag ID. Repeat to provide more than one.",
    )
    parser.add_argument(
        "--series-id",
        type=int,
        action="append",
        default=[],
        help="Explicit Polymarket series ID. Repeat to provide more than one.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum event pages to fetch per tag ID.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout in seconds for Gamma API calls.",
    )
    parser.add_argument(
        "--disable-clob",
        action="store_true",
        help="Use Gamma outcomePrices only and skip CLOB midpoint lookups.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    resolver = TeamResolver(args.data_dir)
    if args.input_raw_json is not None:
        events = json.loads(args.input_raw_json.read_text(encoding="utf-8"))
        source_ids = {"series_ids": args.series_id, "tag_ids": args.tag_id}
    else:
        client = GammaClient(timeout=args.timeout)
        events, source_ids = collect_ncaab_events(
            client,
            explicit_tags=args.tag_id,
            explicit_series_ids=args.series_id,
            max_pages=args.max_pages,
        )
    expected_tag_ids = set(source_ids["tag_ids"]) - GENERIC_TAG_IDS
    filtered_events = [
        event
        for event in events
        if event_looks_like_college_basketball(event, expected_tag_ids=expected_tag_ids)
    ]

    clob_midpoints: dict[str, float] = {}
    if not args.disable_clob:
        token_ids: list[str] = []
        seen_token_ids: set[str] = set()
        for event in filtered_events:
            for market in event.get("markets", []) or []:
                if not market_is_supported(market):
                    continue
                if not market_looks_like_target_context(event, market, expected_tag_ids):
                    continue
                for token_id in _parse_token_ids(market.get("clobTokenIds")):
                    if token_id in seen_token_ids:
                        continue
                    seen_token_ids.add(token_id)
                    token_ids.append(token_id)
        if token_ids:
            try:
                clob_midpoints = ClobClient(timeout=args.timeout).get_midpoints(token_ids)
            except RuntimeError as exc:
                print(f"Warning: {exc}", file=sys.stderr)
                clob_midpoints = {}

    markets: list[MatchupMarket] = []
    futures: list[FutureMarket] = []
    unmatched_markets: list[dict[str, Any]] = []
    for event in filtered_events:
        for market in event.get("markets", []) or []:
            if not market_is_supported(market):
                continue
            if not market_looks_like_target_context(event, market, expected_tag_ids):
                continue
            resolved_future = resolve_future_market(event, market, resolver, clob_midpoints)
            if resolved_future is not None:
                futures.append(resolved_future)
                continue
            resolved = resolve_market_matchup(event, market, resolver, clob_midpoints)
            if resolved is None:
                unmatched_markets.append(
                    {
                        "event_id": event.get("id"),
                        "event_title": event.get("title"),
                        "market_id": market.get("id"),
                        "market_question": market.get("question"),
                        "outcomes": _parse_json_array(market.get("outcomes")),
                        "outcomePrices": _parse_json_array(market.get("outcomePrices")),
                    }
                )
                continue
            markets.append(resolved)

    payload = {
        "fetched_at_utc": _now_utc_iso(),
        "source": {
            "gamma_base_url": GAMMA_API_BASE,
            "clob_base_url": CLOB_API_BASE,
            "series_ids": source_ids["series_ids"],
            "tag_ids": source_ids["tag_ids"],
            "event_count": len(filtered_events),
            "market_count": len(markets),
            "futures_count": len(futures),
            "clob_midpoints_count": len(clob_midpoints),
            "unmatched_supported_markets": len(unmatched_markets),
        },
        "matchups": [asdict(market) for market in markets],
        "pairwise_index": build_pairwise_index(markets),
        "futures": [asdict(market) for market in futures],
        "futures_index": build_futures_index(futures),
        "unmatched_markets": unmatched_markets,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.output_csv, markets)
    write_futures_csv(args.futures_output_csv, futures)
    if args.input_raw_json is None:
        args.raw_json.parent.mkdir(parents=True, exist_ok=True)
        args.raw_json.write_text(json.dumps(filtered_events, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "fetched_at_utc": payload["fetched_at_utc"],
                "series_ids": source_ids["series_ids"],
                "tag_ids": source_ids["tag_ids"],
                "events_scanned": len(filtered_events),
                "matchups_resolved": len(markets),
                "futures_resolved": len(futures),
                "clob_midpoints_fetched": len(clob_midpoints),
                "unmatched_supported_markets": len(unmatched_markets),
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
