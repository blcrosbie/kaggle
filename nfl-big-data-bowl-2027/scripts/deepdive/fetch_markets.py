"""Fetch public betting / prediction-market data for the 2023-2025 NFL seasons into data/markets/*.parquet.

Sources (all public, read-only, no auth):
  nflverse   closing lines           github.com/nflverse/nfldata games.csv
  Kalshi     markets/trades/candles  api.elections.kalshi.com/trade-api/v2/historical/*
  Polymarket events/markets          gamma-api.polymarket.com/events
             hourly prices           clob.polymarket.com/prices-history
             trades with wallets     data-api.polymarket.com/v2/trades
No sportsbook sites are scraped.

Raw responses are cached (gzipped JSON) under data/markets/raw/, so re-runs only fetch what is missing.
Run from the project root:
  python -I -X utf8 scripts/deepdive/fetch_markets.py all            # everything, then build
  python -I -X utf8 scripts/deepdive/fetch_markets.py kalshi-trades  # one stage (see --help)
"""
from __future__ import annotations

import argparse
import base64
import collections
import datetime as dt
import gzip
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit

import polars as pl
import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/parquet"
OUT = ROOT / "data/markets"
RAW = OUT / "raw"

NFLVERSE_GAMES = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
PMDATA = "https://data-api.polymarket.com"

# game dates kept in the market tables (scope Sep 2023 - Feb 2026, plus the Aug preseason games in games.parquet)
DATE_MIN, DATE_MAX = dt.date(2023, 8, 1), dt.date(2026, 2, 28)
# heavy fetches (trades / prices / candles) only for these season types
HEAVY_SEASON_TYPES = ("REG", "POST")

# ---------------------------------------------------------------- Kalshi series in scope
K_GAME = {"KXNFLGAME": "game_winner", "KXNFLSPREAD": "spread", "KXNFLTOTAL": "total", "KXNFLTEAMTOTAL": "team_total"}
# championship-winner series whose finalist markets double as game-winner markets: 2024 conference title games + SB LIX
# (KXAFC/KXNFC/KXSB-25, listed 3 days before), and 2025's conference title games + SB LX, which had no (traded)
# KXNFLGAME market and only exist as season-long futures (KXNFLAFCCHAMP-25 / KXNFLNFCCHAMP-25 / KXSB-26)
K_TITLE = {"KXAFC": "title_game_winner", "KXNFC": "title_game_winner", "KXSB": "title_game_winner",
           "KXNFLAFCCHAMP": "title_game_winner", "KXNFLNFCCHAMP": "title_game_winner"}
K_SEASON_FUTURES = ("KXSB-26-", "KXNFLAFCCHAMP-", "KXNFLNFCCHAMP-")  # candles only, no trades
K_COMBO = ["KXNFLCOMBO", "KXNFLPREPACK", "KXNFLPREPACK1HFT", "KXNFLPREPACK1Q1H", "KXNFLPREPACK2ML",
           "KXNFLPREPACK3ML", "KXNFLPREPACKSGP", "KXNFLPREPACKSGPSPREAD", "KXMVENFLMULTIGAME"]
# user-built parlays (RFQ combos): millions of markets, so only a sample is listed: for every 2025 REG/POST game date,
# the N pages (1000 markets each) created just before that date's first kickoff (listing is sorted by created_time desc)
K_MVE_SAMPLED = {"KXMVENFLSINGLEGAME": 2, "KXMVENFLMULTIGAMEEXTENDED": 1}
K_OTHER = {"KXNFLCOMBINE40": "combine", "KXNFLGAMETD": "game_prop"}
# player series without a per-game event ticker that still settle on one game (SB MVP)
K_PROPS_EXTRA = ["KXNFLSBMVP"]
# player series that are not per-game (career/retirement/rankings): never treated as props
K_NOT_PROPS = {"KXNFLCOMPETE", "KXNFLRETIRE", "KXNFLT100TOP", "KXNFLCOMBO"}
# player-prop series are discovered (football_player strike + per-game event ticker); these are the expected ones
K_PROPS_EXPECTED = ["KXNFLANYTD", "KXNFLRECYDS", "KXNFLREC", "KXNFLRSHYDS", "KXNFLPASSYDS", "KXNFLINT", "KXNFLFIRSTTD",
                    "KXNFLTEAMFIRSTTD", "KXNFLLADDERRECYDS", "KXNFLLADDERREC", "KXNFLLONGREC", "KXNFLGAMETD"]

KALSHI_TEAMS = {"ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAC",
                "JAX", "KC", "LA", "LAR", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA",
                "SF", "TB", "TEN", "WAS", "WSH"}
TEAM_ALIASES = {"LAR": "LA", "STL": "LA", "WSH": "WAS", "OAK": "LV", "LVR": "LV", "LAS": "LV", "JAC": "JAX",
                "SD": "LAC"}
NICKNAMES = {"cardinals": "ARI", "falcons": "ATL", "ravens": "BAL", "bills": "BUF", "panthers": "CAR", "bears": "CHI",
             "bengals": "CIN", "browns": "CLE", "cowboys": "DAL", "broncos": "DEN", "lions": "DET", "packers": "GB",
             "texans": "HOU", "colts": "IND", "jaguars": "JAX", "chiefs": "KC", "rams": "LA", "chargers": "LAC",
             "raiders": "LV", "dolphins": "MIA", "vikings": "MIN", "patriots": "NE", "saints": "NO", "giants": "NYG",
             "jets": "NYJ", "eagles": "PHI", "steelers": "PIT", "seahawks": "SEA", "49ers": "SF", "niners": "SF",
             "buccaneers": "TB", "bucs": "TB", "titans": "TEN", "commanders": "WAS"}
MONTHS = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def team(code: str | None) -> str | None:
    if not code:
        return None
    c = code.upper()
    return TEAM_ALIASES.get(c, c)


# ---------------------------------------------------------------- http + cache
class Http:
    """requests wrapper: <= rps requests/second per host (shared across threads), retries 429/5xx with backoff."""

    def __init__(self, rps: float = 5.0):
        self.gap = 1.0 / rps
        self.lock = threading.Lock()
        self.next_t: dict[str, float] = {}
        self.local = threading.local()
        self.calls = collections.Counter()
        self.retries = collections.Counter()

    def _session(self) -> requests.Session:
        s = getattr(self.local, "s", None)
        if s is None:
            s = self.local.s = requests.Session()
            s.headers["User-Agent"] = "nfl-bdb-2027-research (read-only)"
        return s

    def _wait(self, host: str) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t.get(host, 0.0))
            self.next_t[host] = t + self.gap
        if t > now:
            time.sleep(t - now)

    def get(self, url: str, params: dict | None = None, tries: int = 9, as_text: bool = False):
        host = urlsplit(url).netloc
        err = ""
        for k in range(tries):
            self._wait(host)
            try:
                r = self._session().get(url, params=params, timeout=90)
                self.calls[host] += 1
                if r.status_code == 200:
                    return r.text if as_text else r.json()
                if r.status_code == 404:
                    return None
                err = f"HTTP {r.status_code} {r.text[:200]}"
                if r.status_code not in (408, 425, 429, 500, 502, 503, 504):
                    raise RuntimeError(f"{url} {params}: {err}")
                delay = float(r.headers.get("Retry-After") or 0) or min(120, 2 ** k)
            except (requests.RequestException, ValueError) as e:  # network error or bad JSON
                err, delay = repr(e), min(120, 2 ** k)
            self.retries[host] += 1
            time.sleep(delay)
        raise RuntimeError(f"giving up on {url} {params}: {err}")


HTTP = Http()


def jload(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def jdump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def utc(s) -> dt.datetime | None:
    """ISO string / unix seconds -> naive UTC datetime."""
    if s is None or s == "":
        return None
    if isinstance(s, (int, float)):
        return dt.datetime.fromtimestamp(s, dt.timezone.utc).replace(tzinfo=None)
    s = str(s).strip().replace(" ", "T")
    if s.endswith("+00"):
        s += ":00"
    try:
        d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d.astimezone(dt.timezone.utc).replace(tzinfo=None) if d.tzinfo else d


def to_ts(d: dt.datetime) -> int:
    return int(d.replace(tzinfo=dt.timezone.utc).timestamp())


def fnum(x) -> float | None:
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- reference tables (games, cohort)
def norm_name(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
    s = s.replace("’", "'")
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_games() -> pl.DataFrame:
    return (pl.read_parquet(CACHE / "games.parquet")
            .with_columns(game_date=pl.col("game_id").cast(pl.Utf8).str.slice(0, 8).str.to_date("%Y%m%d"),
                          home=pl.col("home_team_abbr").replace(TEAM_ALIASES),
                          away=pl.col("visitor_team_abbr").replace(TEAM_ALIASES)))


class GameIndex:
    """(date +-tol days, unordered team pair) -> game row."""

    def __init__(self):
        self.g = load_games()
        self.by_key = {}
        for r in self.g.iter_rows(named=True):
            self.by_key[(r["game_date"], frozenset((r["home"], r["away"])))] = r
        self.by_date_team = collections.defaultdict(list)
        for r in self.g.iter_rows(named=True):
            for t in (r["home"], r["away"]):
                self.by_date_team[(r["game_date"], t)].append(r)
        self.by_id = {r["game_id"]: r for r in self.g.iter_rows(named=True)}

    def find(self, d: dt.date | None, t1: str | None, t2: str | None = None, tol: int = 1):
        if d is None or t1 is None:
            return None
        for off in sorted(range(-tol, tol + 1), key=abs):
            dd = d + dt.timedelta(days=off)
            if t2:
                r = self.by_key.get((dd, frozenset((team(t1), team(t2)))))
            else:
                rs = self.by_date_team.get((dd, team(t1)), [])
                r = rs[0] if len(rs) == 1 else None
            if r:
                return r
        return None


NAME_ALIASES = {"mitchell tinsley": "mitch tinsley"}  # market spelling -> players.display_name (normalized)


class Cohort:
    """Name matching against players.parquet, disambiguated with the team the player suited up for (player_play)."""

    def __init__(self):
        p = pl.read_parquet(CACHE / "players.parquet")
        self.names = {r["nfl_id"]: r["display_name"] for r in p.iter_rows(named=True)}
        self.by_norm = collections.defaultdict(list)
        for nid, nm in self.names.items():
            self.by_norm[norm_name(nm)].append(nid)
        pp = (pl.scan_parquet(CACHE / "player_play.parquet").select("game_id", "nfl_id", "team_abbr").unique()
              .with_columns(team_abbr=pl.col("team_abbr").replace(TEAM_ALIASES)).collect())
        self.team_in_game = {(r["game_id"], r["nfl_id"]): r["team_abbr"] for r in pp.iter_rows(named=True)}
        g = load_games().select("game_id", "season")
        ps = pp.join(g, on="game_id").group_by("nfl_id", "season", "team_abbr").len()
        self.season_teams = collections.defaultdict(set)
        for r in ps.iter_rows(named=True):
            self.season_teams[(r["nfl_id"], r["season"])].add(r["team_abbr"])

    def match(self, name: str | None, game: dict | None, team_code: str | None = None):
        """-> (nfl_id, how, candidate_id); how in exact / team_disambiguated / name_only / ambiguous / team_mismatch.
        nfl_id is only set when the name is unique in the cohort (or made unique by team) and the team agrees."""
        nn = norm_name(name)
        cands = self.by_norm.get(NAME_ALIASES.get(nn, nn), [])
        if not cands:
            return None, None, None
        gid = game["game_id"] if game else None
        season = game["season"] if game else None
        tc = team(team_code)
        game_teams = {game["home"], game["away"]} if game else set()

        def team_of(nid):
            t = self.team_in_game.get((gid, nid))
            if t:
                return {t}
            return self.season_teams.get((nid, season), set())

        ok = []
        for nid in cands:
            ts = team_of(nid)
            if tc:
                good = tc in ts
            elif game_teams:
                good = bool(ts & game_teams)
            else:
                good = True
            ok.append((nid, good, bool(ts)))
        good = [o for o in ok if o[1]]
        if len(good) == 1:
            return good[0][0], "exact" if len(cands) == 1 else "team_disambiguated", good[0][0]
        if len(good) > 1:
            return None, "ambiguous", None
        # no team evidence: accept a unique name only if we have no team info at all for that season
        if len(cands) == 1 and not ok[0][2]:
            return cands[0], "name_only", cands[0]
        return None, "team_mismatch", cands[0] if len(cands) == 1 else None


# ---------------------------------------------------------------- nflverse
def fetch_nflverse() -> Path:
    f = RAW / "nflverse/games.csv"
    if not f.exists():
        txt = HTTP.get(NFLVERSE_GAMES, as_text=True)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(txt, encoding="utf-8")
    return f


def build_nflverse() -> pl.DataFrame:
    f = fetch_nflverse()
    g = pl.read_csv(f, infer_schema_length=50000).filter(pl.col("season").is_between(2023, 2025))
    ours = load_games()
    # kickoff: gameday + gametime are US/Eastern local
    et = (pl.col("gameday") + " " + pl.col("gametime")).str.to_datetime("%Y-%m-%d %H:%M")
    g = g.with_columns(kickoff_utc=et.dt.replace_time_zone("America/New_York").dt.convert_time_zone("UTC")
                       .dt.replace_time_zone(None),
                       gameday=pl.col("gameday").str.to_date(),
                       home_c=pl.col("home_team").replace(TEAM_ALIASES), away_c=pl.col("away_team").replace(TEAM_ALIASES))
    # primary key: old_game_id == our game_id (and gsis == game_key); fall back to date + teams
    g = g.rename({"game_id": "nflverse_game_id"})
    m1 = g.join(ours.select(pl.col("game_id").alias("our_id"), "home", "away"), left_on="old_game_id",
                right_on="our_id", how="left", coalesce=False)
    ok = (m1["our_id"].is_not_null() & (m1["home"] == m1["home_c"]) & (m1["away"] == m1["away_c"])).to_list()
    gi = GameIndex()
    ids, how = [], []
    for r, good in zip(m1.iter_rows(named=True), ok):
        if good:
            ids.append(r["our_id"]); how.append("old_game_id")
            continue
        hit = gi.find(r["gameday"], r["home_c"], r["away_c"])
        ids.append(hit["game_id"] if hit else None); how.append("date_teams" if hit else None)
    base = g.with_columns(
        pl.Series("game_id", ids, dtype=pl.Int64), pl.Series("map_method", how, dtype=pl.Utf8))
    out = base.select("game_id", "map_method", "nflverse_game_id", pl.col("old_game_id").alias("nflverse_old_game_id"),
                      pl.col("gsis").alias("game_key"), "season", "game_type", "week", "gameday", "weekday", "gametime",
                      "kickoff_utc", "away_team", "home_team", "away_score", "home_score", "result", "total", "overtime",
                      "location", "spread_line", "away_spread_odds", "home_spread_odds", "away_moneyline",
                      "home_moneyline", "total_line", "under_odds", "over_odds", "div_game", "roof", "surface", "temp",
                      "wind", "away_qb_name", "home_qb_name", "stadium")
    out.write_parquet(OUT / "nflverse_lines.parquet")
    n_reg = ours.filter(pl.col("season_type") != "PRE").height
    log(f"nflverse_lines: {out.height} rows, mapped {out['game_id'].is_not_null().sum()} "
        f"({out['game_id'].n_unique() - (1 if out['game_id'].null_count() else 0)} of our {n_reg} REG/POST games)")
    return out


def kickoffs() -> dict[int, dt.datetime]:
    f = OUT / "nflverse_lines.parquet"
    if not f.exists():
        build_nflverse()
    k = pl.read_parquet(f).filter(pl.col("game_id").is_not_null())
    return dict(zip(k["game_id"].to_list(), k["kickoff_utc"].to_list()))


# ---------------------------------------------------------------- Kalshi
def kalshi_walk(path: str, params: dict, key: str, cache_dir: Path, max_pages: int | None = None):
    """Walk a cursor-paginated Kalshi endpoint, one cached file per page. -> (items, complete)."""
    items, cur, i = [], None, 0
    while True:
        f = cache_dir / f"p{i:05d}.json.gz"
        if f.exists():
            page = jload(f)
        else:
            q = dict(params)
            if cur:
                q["cursor"] = cur
            page = HTTP.get(KALSHI + path, q) or {}
            page["_request"] = {k: v for k, v in q.items()}
            jdump(f, page)
        got = page.get(key) or []
        items += got
        cur = page.get("cursor")
        i += 1
        if not cur or not got:
            return items, True
        if max_pages and i >= max_pages:
            return items, False


def kalshi_series_all() -> list[dict]:
    f = RAW / "kalshi/series_all.json.gz"
    if not f.exists():
        jdump(f, HTTP.get(KALSHI + "/series") or {})
    return jload(f).get("series") or []


def kalshi_nfl_series(series: list[dict]) -> list[str]:
    out = []
    for s in series:
        t = s["ticker"]
        if s.get("category") not in ("Sports", "Exotics"):
            continue
        if "NFL" in t or t in K_TITLE or t in ("KXTEAMSINSB", "KXNFCAFCSB"):
            out.append(t)
    return sorted(set(out))


EVT_RE = re.compile(r"^(\d{2})([A-Z]{3})(\d{2})([A-Z]{4,6})([A-Z0-9]*)$")


def parse_kalshi_event(event_ticker: str):
    """'KXNFLGAME-25OCT19NOCHI' -> (date, away, home) using the Kalshi team-code list to split the team string.
    Trailing qualifiers are allowed ('...SEANETD4', '...SEANESPREAD')."""
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return None, None, None
    m = EVT_RE.match(parts[1])
    if not m or m.group(2) not in MONTHS:
        return None, None, None
    try:
        d = dt.date(2000 + int(m.group(1)), MONTHS[m.group(2)], int(m.group(3)))
    except ValueError:
        return None, None, None
    s = m.group(4) + m.group(5)
    splits = [(s[:i], s[i:i + j], s[i + j:]) for i in (2, 3) for j in (2, 3)]
    splits = [x for x in splits if x[0] in KALSHI_TEAMS and x[1] in KALSHI_TEAMS]
    if splits:
        a, h, _ = min(splits, key=lambda x: len(x[2]))  # prefer the split that consumes the whole string
        return d, team(a), team(h)
    return d, None, None


def kalshi_discover() -> pl.DataFrame:
    """First listing page of every NFL-related series -> census of what each series looks like."""
    rows = []
    names = kalshi_nfl_series(kalshi_series_all())
    log(f"kalshi discover: {len(names)} NFL-related series")
    titles = {s["ticker"]: s.get("title") for s in kalshi_series_all()}
    for s in names:
        f = RAW / f"kalshi/discover/{safe_name(s)}.json.gz"
        if not f.exists():
            jdump(f, HTTP.get(KALSHI + "/historical/markets", {"series_ticker": s, "limit": 200}) or {})
        page = jload(f)
        ms = page.get("markets") or []
        per_game = sum(1 for m in ms if parse_kalshi_event(m["event_ticker"])[1])
        player = sum(1 for m in ms if "football_player" in (m.get("custom_strike") or {}))
        rows.append(dict(series_ticker=s, title=titles.get(s), first_page_markets=len(ms), more=bool(page.get("cursor")),
                         per_game_markets=per_game, player_markets=player,
                         mve=sum(1 for m in ms if m.get("mve_selected_legs")),
                         min_close=min((m["close_time"] for m in ms), default=None),
                         max_close=max((m["close_time"] for m in ms), default=None)))
    df = pl.DataFrame(rows)
    df.write_parquet(RAW / "kalshi/discover_census.parquet")
    return df


def kalshi_prop_series() -> list[str]:
    f = RAW / "kalshi/discover_census.parquet"
    df = pl.read_parquet(f) if f.exists() else kalshi_discover()
    props = df.filter((pl.col("player_markets") > 0) & (pl.col("per_game_markets") > 0))["series_ticker"].to_list()
    have = set(df.filter(pl.col("first_page_markets") > 0)["series_ticker"].to_list())
    props = set(props) | {s for s in K_PROPS_EXPECTED + K_PROPS_EXTRA if s in have}
    return sorted(props - K_NOT_PROPS - set(K_OTHER) - set(K_GAME))


def kalshi_series_plan() -> dict[str, tuple[str, int | None]]:
    """series -> (market_kind, max_pages)."""
    plan = {s: (k, None) for s, k in K_GAME.items()}
    plan.update({s: (k, None) for s, k in K_TITLE.items()})
    plan.update({s: ("player_prop", None) for s in kalshi_prop_series()})
    plan.update({s: ("combo", None) for s in K_COMBO})
    plan.update({s: ("combo", None) for s in K_MVE_SAMPLED})
    plan.update({s: (k, None) for s, k in K_OTHER.items()})
    return plan


def _varint(n: int) -> bytes:
    out = b""
    while True:
        b, n = n & 0x7F, n >> 7
        out += bytes([b | 0x80]) if n else bytes([b])
        if not n:
            return out


def kalshi_created_cursor(ts: int) -> str:
    """Cursor for /historical/markets positioned at created_time == ts (protobuf {1: Timestamp, 2: ticker}).
    Undocumented, reverse-engineered from returned cursors; used only to sample the huge MVE parlay series."""
    t = b"\x08" + _varint(ts) + b"\x10\x00"
    msg = b"\x0a" + _varint(len(t)) + t + b"\x12\x01~"
    return base64.urlsafe_b64encode(msg).decode().rstrip("=")


def kalshi_mve_sample(series: str, pages_per_date: int) -> tuple[list[dict], bool]:
    ko = kickoffs()
    g = load_games().filter(pl.col("season") == 2025, pl.col("season_type").is_in(HEAVY_SEASON_TYPES))
    first = {}
    for gid, d in zip(g["game_id"].to_list(), g["game_date"].to_list()):
        k = ko.get(gid)
        if k and (d not in first or k < first[d]):
            first[d] = k
    items = []
    for d, k in sorted(first.items()):
        got, _ = kalshi_walk("/historical/markets", {"series_ticker": series, "limit": 1000,
                                                     "cursor": kalshi_created_cursor(to_ts(k))}, "markets",
                             RAW / f"kalshi/markets/{safe_name(series)}/d{d:%Y%m%d}", pages_per_date)
        for m in got:
            m["_sample_date"] = d.isoformat()
        items += got
    return items, False


def kalshi_fetch_markets() -> dict[str, dict]:
    plan = kalshi_series_plan()
    status = {}
    for s, (kind, max_pages) in plan.items():
        params = {"series_ticker": s, "limit": 1000}
        if s in K_MVE_SAMPLED:
            ms, complete = kalshi_mve_sample(s, K_MVE_SAMPLED[s])
        else:
            ms, complete = kalshi_walk("/historical/markets", params, "markets", RAW / f"kalshi/markets/{safe_name(s)}",
                                       max_pages)
        status[s] = dict(kind=kind, markets=len(ms), complete=complete,
                         min_close=min((m["close_time"] for m in ms), default=None),
                         max_close=max((m["close_time"] for m in ms), default=None))
        log(f"kalshi markets {s:28s} {kind:18s} {len(ms):>8,} {'complete' if complete else 'TRUNCATED'}")
    jdump(RAW / "kalshi/markets_status.json.gz", status)
    return status


def kalshi_load_markets() -> list[dict]:
    out = []
    for s, (kind, max_pages) in kalshi_series_plan().items():
        d = RAW / f"kalshi/markets/{safe_name(s)}"
        if not d.exists():
            continue
        for f in sorted(d.rglob("p*.json.gz")):
            sd = f.parent.name
            sd = f"{sd[1:5]}-{sd[5:7]}-{sd[7:9]}" if re.fullmatch(r"d\d{8}", sd) else None
            for m in jload(f).get("markets") or []:
                m["_series"], m["_kind"], m["_sample_date"] = s, kind, sd
                out.append(m)
    return out


PROP_STAT = {"KXNFLANYTD": "anytime_td", "KXNFLRECYDS": "receiving_yards", "KXNFLREC": "receptions",
             "KXNFLRSHYDS": "rushing_yards", "KXNFLPASSYDS": "passing_yards", "KXNFLINT": "interceptions",
             "KXNFLFIRSTTD": "first_td", "KXNFLTEAMFIRSTTD": "team_first_td", "KXNFLLADDERRECYDS": "receiving_yards",
             "KXNFLLADDERREC": "receptions", "KXNFLLADDERRSHYDS": "rushing_yards", "KXNFLLONGREC": "longest_reception",
             "KXNFLLONGRSH": "longest_rush", "KXNFLGAMETD": "touchdowns", "KXNFL2TD": "two_plus_td",
             "KXNFLPASSTDS": "passing_tds", "KXNFLPASSATT": "pass_attempts", "KXNFLPASSCOMP": "pass_completions",
             "KXNFLPASSINT": "passing_interceptions", "KXNFLRSHATT": "rushing_attempts", "KXNFLRRYDS": "rush_rec_yards",
             "KXNFLSACK": "sacks", "KXNFLTKL": "tackles", "KXNFLTD": "touchdowns", "KXNFLNEXTTD": "next_td_ingame",
             "KXNFLMOSTRECYDS": "most_receiving_yards_in_game", "KXNFLMOSTRSHYDS": "most_rushing_yards_in_game",
             "KXNFLFANTASYMOST": "most_fantasy_points_in_game", "KXNFLSBMVP": "game_mvp"}


def kalshi_parse(m: dict, gi: GameIndex, cohort: Cohort) -> dict:
    s, kind = m["_series"], m["_kind"]
    ev = m["event_ticker"]
    d, away, home = parse_kalshi_event(ev)
    cs = m.get("custom_strike") or {}
    legs = m.get("mve_selected_legs") or []
    if not d and m.get("mve_collection_ticker"):
        d, away, home = parse_kalshi_event(m["mve_collection_ticker"])
    suffix = m["ticker"][len(ev) + 1:] if m["ticker"].startswith(ev + "-") else ""
    game = gi.find(d, away, home) if (d and away and home) else None
    leg_events = [lg.get("event_ticker") for lg in legs] or [x.strip() for x in (cs.get("Associated Events") or "")
                                                            .split(",") if x.strip()]
    leg_games = set()
    if kind == "combo" and game is None:
        for le in leg_events:
            ld, la, lh = parse_kalshi_event(le)
            lgm = gi.find(ld, la, lh) if (ld and la and lh) else None
            leg_games.add(lgm["game_id"] if lgm else None)
        if len(leg_games) == 1 and None not in leg_games:
            game = gi.by_id[next(iter(leg_games))]
        elif not leg_events:  # e.g. KXNFLCOMBO-26FEB08SB261: date only; map when it is the only game that day
            m2 = re.match(r"^(\d{2})([A-Z]{3})(\d{2})", ev.split("-")[1] if "-" in ev else "")
            if m2 and m2.group(2) in MONTHS:
                dd = dt.date(2000 + int(m2.group(1)), MONTHS[m2.group(2)], int(m2.group(3)))
                one = [r for r in gi.by_id.values() if r["game_date"] == dd]
                if len(one) == 1:
                    game, d = one[0], dd
        if game:
            d, away, home = game["game_date"], game["away"], game["home"]
    mteam = None
    if kind in ("game_winner", "title_game_winner", "team_total", "spread") and suffix:
        tok = re.match(r"^([A-Z]+)", suffix)
        mteam = team(tok.group(1)) if tok and tok.group(1) in KALSHI_TEAMS else None
    if kind == "title_game_winner" and mteam:
        # finalists only: map the team's market to the title game it would have to win
        title_game = {("KXAFC", "25"): ("2025-01-26", "AFC"), ("KXNFC", "25"): ("2025-01-26", "NFC"),
                      ("KXSB", "25"): ("2025-02-09", "SB"), ("KXSB", "26"): ("2026-02-08", "SB"),
                      ("KXNFLAFCCHAMP", "25"): ("2026-01-25", "AFC"), ("KXNFLNFCCHAMP", "25"): ("2026-01-25", "NFC")}
        key = title_game.get((s, ev.split("-")[1] if "-" in ev else ""))
        if key:
            game = gi.find(dt.date.fromisoformat(key[0]), mteam, tol=0)
            if game:
                d, away, home = game["game_date"], game["away"], game["home"]
    if s == "KXNFLSBMVP":
        sb = {"25": dt.date(2025, 2, 9), "26": dt.date(2026, 2, 8)}.get(ev.split("-")[1] if "-" in ev else "")
        rs = [r for r in gi.by_id.values() if r["game_date"] == sb and r["season_type"] == "POST"] if sb else []
        if rs:
            game = rs[0]
            d, away, home = game["game_date"], game["away"], game["home"]
    player = stat = thr = None
    nfl_id = how = cand = None
    if kind == "player_prop":
        stat = PROP_STAT.get(s, s.replace("KXNFL", "").lower())
        sub = m.get("yes_sub_title") or ""
        player = re.split(r":\s", sub)[0].strip() if sub else None
        if player and re.match(r"^(no |none|no touchdown)", player, re.I):
            player = None
        if home and away:  # market suffix = team code + initial + surname + jersey [-threshold]
            for t in sorted({home, away} | {k for k, v in TEAM_ALIASES.items() if v in (home, away)}, key=len,
                            reverse=True):
                if suffix.startswith(t):
                    mteam = team(t)
                    break
        last = suffix.rsplit("-", 1)
        if fnum(m.get("floor_strike")) is not None:
            thr = fnum(m.get("floor_strike"))  # "5+" markets carry floor_strike 4.5
        elif len(last) == 2 and re.fullmatch(r"\d+(\.\d+)?", last[1]):
            thr = float(last[1]) - 0.5
        else:
            mm = re.search(r"(\d+(?:\.\d+)?)\+", (m.get("yes_sub_title") or "") + " " + (m.get("title") or ""))
            thr = float(mm.group(1)) if mm else None
        if player:
            nfl_id, how, cand = cohort.match(player, game, mteam)
    return dict(
        ticker=m["ticker"], event_ticker=ev, series_ticker=s, market_kind=kind,
        title=m.get("title"), yes_sub_title=m.get("yes_sub_title"), no_sub_title=m.get("no_sub_title"),
        rules_primary=m.get("rules_primary"), market_type=m.get("market_type"), strike_type=m.get("strike_type"),
        custom_strike=json.dumps(cs) if cs else None, floor_strike=fnum(m.get("floor_strike")),
        cap_strike=fnum(m.get("cap_strike")), status=m.get("status"), result=m.get("result"),
        expiration_value=m.get("expiration_value"), settlement_value=fnum(m.get("settlement_value_dollars")),
        settlement_ts=utc(m.get("settlement_ts")), created_time=utc(m.get("created_time")),
        open_time=utc(m.get("open_time")), close_time=utc(m.get("close_time")),
        expected_expiration_time=utc(m.get("expected_expiration_time")),
        occurrence_datetime=utc(m.get("occurrence_datetime")),
        volume=fnum(m.get("volume_fp", m.get("volume"))), volume_24h=fnum(m.get("volume_24h_fp", m.get("volume_24h"))),
        open_interest=fnum(m.get("open_interest_fp", m.get("open_interest"))),
        liquidity=fnum(m.get("liquidity_dollars")), notional_value=fnum(m.get("notional_value_dollars")),
        last_price=fnum(m.get("last_price_dollars")), previous_price=fnum(m.get("previous_price_dollars")),
        yes_bid=fnum(m.get("yes_bid_dollars")), yes_ask=fnum(m.get("yes_ask_dollars")),
        no_bid=fnum(m.get("no_bid_dollars")), no_ask=fnum(m.get("no_ask_dollars")),
        previous_yes_bid=fnum(m.get("previous_yes_bid_dollars")),
        previous_yes_ask=fnum(m.get("previous_yes_ask_dollars")),
        price_level_structure=m.get("price_level_structure"),
        mve_collection_ticker=m.get("mve_collection_ticker"), n_legs=len(legs) or None,
        mve_sample_date=m.get("_sample_date"), n_leg_games=len(leg_games) or None,
        event_date=d, away_team=away, home_team=home, market_team=mteam,
        game_id=game["game_id"] if game else None, season=game["season"] if game else None,
        season_type=game["season_type"] if game else None, week=game["week"] if game else None,
        player_name=player, kalshi_player_id=cs.get("football_player"), stat=stat, threshold=thr,
        nfl_id=nfl_id, name_match=how, cohort_candidate_id=cand)


def kalshi_markets_table() -> tuple[pl.DataFrame, pl.DataFrame]:
    gi, cohort = GameIndex(), Cohort()
    raw = kalshi_load_markets()
    rows, legs = [], []
    seen = set()
    for m in raw:
        if m["ticker"] in seen:
            continue
        seen.add(m["ticker"])
        r = kalshi_parse(m, gi, cohort)
        d = r["event_date"] or (r["close_time"].date() if r["close_time"] else None)
        if r["market_kind"] == "combine":
            pass
        elif r["market_kind"] == "title_game_winner" and r["game_id"] is None:
            continue  # non-finalist championship futures are out of scope
        elif r["game_id"] is None and not (d and DATE_MIN <= d <= DATE_MAX):
            continue
        rows.append(r)
        lgs = m.get("mve_selected_legs") or []
        cs = m.get("custom_strike") or {}
        if not lgs and cs.get("Associated Markets"):  # Kalshi pre-packs describe legs in custom_strike
            split = lambda k: [x.strip() for x in (cs.get(k) or "").split(",")]
            ev_, mk_, sd_ = split("Associated Events"), split("Associated Markets"), split("Associated Market Sides")
            lgs = [dict(event_ticker=ev_[i] if i < len(ev_) else None, market_ticker=mk_[i],
                        side=(sd_[i] if i < len(sd_) else "").lower() or None) for i in range(len(mk_))]
        for i, lg in enumerate(lgs):
            legs.append(dict(ticker=m["ticker"], leg_index=i, leg_event_ticker=lg.get("event_ticker"),
                             leg_market_ticker=lg.get("market_ticker"), leg_side=lg.get("side")))
    df = pl.DataFrame(rows, infer_schema_length=None)
    lg = pl.DataFrame(legs, schema={"ticker": pl.Utf8, "leg_index": pl.Int64, "leg_event_ticker": pl.Utf8,
                                    "leg_market_ticker": pl.Utf8, "leg_side": pl.Utf8})
    return df, lg


def kalshi_trade_targets(mk: pl.DataFrame) -> pl.DataFrame:
    heavy = pl.col("season_type").is_in(HEAVY_SEASON_TYPES)
    game = mk.filter((pl.col("market_kind") == "game_winner") & heavy)
    # 2024-25 conference / SB title markets were listed days before the games; KXSB-26 is a season-long future
    # (hundreds of millions of contracts) so it gets candles only
    title = mk.filter((pl.col("market_kind") == "title_game_winner") & heavy & pl.col("game_id").is_not_null()
                      & ~pl.any_horizontal([pl.col("ticker").str.starts_with(x) for x in K_SEASON_FUTURES]))
    props = mk.filter((pl.col("market_kind") == "player_prop") & pl.col("nfl_id").is_not_null() & heavy)
    return pl.concat([props.with_columns(target=pl.lit("cohort_prop")),
                      game.with_columns(target=pl.lit("game_winner")),
                      title.with_columns(target=pl.lit("title_game_winner"))], how="diagonal_relaxed")


def kalshi_fetch_trades_one(ticker: str, series: str, max_ts: int | None = None) -> dict:
    tag = "" if max_ts is None else f"__max{max_ts}"
    f = RAW / f"kalshi/trades/{safe_name(series)}/{safe_name(ticker)}{tag}.json.gz"
    if f.exists():
        return jload(f)
    trades, cur = [], None
    while True:
        q = {"ticker": ticker, "limit": 1000}
        if max_ts is not None:
            q["max_ts"] = max_ts
        if cur:
            q["cursor"] = cur
        page = HTTP.get(KALSHI + "/historical/trades", q) or {}
        got = page.get("trades") or []
        trades += got
        cur = page.get("cursor")
        if not cur or not got:
            break
    obj = {"ticker": ticker, "max_ts": max_ts, "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "trades": trades}
    jdump(f, obj)
    return obj


def kalshi_fetch_trades(max_trades: int, max_minutes: float) -> None:
    """Trades in priority tiers until the trade / time budget is spent (manifest records what each market got):
      1 cohort-player props + 2024-25 title-game markets: full history (always)
      2 KXNFLGAME home-team market of each game: trades before scheduled kickoff
      3 KXNFLGAME away-team market: trades before kickoff
      4/5 KXNFLGAME home / away market: full history incl. in-game
    Everything not reached keeps hourly candles only."""
    mk, _ = kalshi_markets_table()
    tg = kalshi_trade_targets(mk).sort("event_date", "ticker")
    ko = kickoffs()
    mf = RAW / "kalshi/trades_manifest.json.gz"
    manifest = jload(mf) if mf.exists() else {}
    jobs = []
    for r in tg.select("ticker", "series_ticker", "target", "game_id", "market_team", "home_team").to_dicts():
        if r["target"] != "game_winner":
            jobs.append((1, r, "full"))
            continue
        home = r["market_team"] == r["home_team"]
        jobs += [(2 if home else 3, r, "pregame"), (4 if home else 5, r, "full")]
    jobs.sort(key=lambda j: j[0])
    total = sum(v["n"] for v in manifest.values() if v["mode"] in ("full", "pregame"))
    log(f"kalshi trades: {len(tg)} target markets {tg['target'].value_counts().to_dicts()}; {total:,} cached trades")
    t0 = time.time()

    def work(job):
        tier, r, mode = job
        k = ko.get(r["game_id"])
        max_ts = to_ts(k) if mode == "pregame" and k else None
        if mode == "pregame" and max_ts is None:
            mode = "full"
        return job, mode, len(kalshi_fetch_trades_one(r["ticker"], r["series_ticker"], max_ts)["trades"])

    stopped_at = None
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs, it = {}, iter(jobs)
        while True:
            while len(futs) < 10 and stopped_at is None:
                job = next(it, None)
                if job is None:
                    break
                tier, r, mode = job
                have = manifest.get(r["ticker"], {}).get("mode")
                if have == "full" or (have == "pregame" and mode == "pregame"):
                    continue
                if tier > 1 and (total >= max_trades or (time.time() - t0) / 60 >= max_minutes):
                    stopped_at = tier
                    break
                futs[ex.submit(work, job)] = job
            if not futs:
                break
            done = next(as_completed(futs))
            futs.pop(done)
            (tier, r, _), mode, n = done.result()
            prev = manifest.get(r["ticker"], {})
            total += n - (prev.get("n", 0) if prev.get("mode") in ("full", "pregame") else 0)
            manifest[r["ticker"]] = dict(mode=mode, n=n, target=r["target"], tier=tier)
            if len(manifest) % 100 == 0:
                jdump(mf, manifest)
                log(f"  trades {total:,}  markets {len(manifest)}  tier {tier}  ({(time.time() - t0) / 60:.1f} min)")
    for tier, r, mode in jobs:  # document what was not reached
        manifest.setdefault(r["ticker"], dict(mode="candles_only", n=0, target=r["target"], tier=tier))
    jdump(mf, manifest)
    log(f"kalshi trades done: {total:,} trades; stopped at tier {stopped_at}; "
        f"modes={collections.Counter(v['mode'] for v in manifest.values())}")


def kalshi_fetch_candles() -> None:
    """Hourly candlesticks for every game-winner / title market (REG/POST), plus cohort props not fully traded."""
    mk, _ = kalshi_markets_table()
    heavy = pl.col("season_type").is_in(HEAVY_SEASON_TYPES)
    tg = mk.filter(pl.col("market_kind").is_in(["game_winner", "title_game_winner"]) & pl.col("game_id").is_not_null()
                   & heavy)
    rows = tg.select("ticker", "series_ticker", "open_time", "close_time").to_dicts()
    log(f"kalshi candles: {len(rows)} markets")

    def work(r):
        f = RAW / f"kalshi/candles/{safe_name(r['series_ticker'])}/{safe_name(r['ticker'])}.json.gz"
        if f.exists():
            return 0
        s, e = to_ts(r["open_time"]), to_ts(r["close_time"]) + 3600
        out, a = [], s
        while a < e:  # 1500-hour windows
            b = min(e, a + 1500 * 3600)
            page = HTTP.get(f"{KALSHI}/historical/markets/{r['ticker']}/candlesticks",
                            {"start_ts": a, "end_ts": b, "period_interval": 60}) or {}
            out += page.get("candlesticks") or []
            a = b
        jdump(f, {"ticker": r["ticker"], "period_interval": 60, "candlesticks": out})
        return len(out)

    with ThreadPoolExecutor(max_workers=5) as ex:
        n = sum(ex.map(work, rows))
    log(f"kalshi candles fetched: {n:,} new candles")


# ---------------------------------------------------------------- Polymarket
def gamma_walk(params: dict, name: str) -> list[dict]:
    """Offset-paginated /events (limit 100), one cached file per page."""
    out, off = [], 0
    d = RAW / f"polymarket/events/{name}"
    while True:
        f = d / f"o{off:06d}.json.gz"
        if f.exists():
            page = jload(f)
        else:
            page = HTTP.get(GAMMA + "/events", dict(params, limit=100, offset=off, order="id", ascending="true")) or []
            jdump(f, page)
        out += page
        if len(page) < 100:
            return out
        off += 100


PM_GAME_SLUG = re.compile(r"^nfl-([a-z0-9]+)-([a-z0-9]+)-(\d{4}-\d{2}-\d{2})$")
PM_DAILIES = re.compile(r"^nfl-dailies-(\d{4}-\d{2}-\d{2})$")
PM_EXTRA_SLUGS = ["super-bowl-winner-2024"]  # SB LVIII has no game event; finalist futures stand in


def poly_fetch_events() -> list[dict]:
    evs = {}
    for e in gamma_walk({"tag_id": 450, "closed": "true"}, "tag450_closed"):
        evs[e["id"]] = e
    for sid in (1, 10186, 10187):
        for e in gamma_walk({"series_id": sid}, f"series{sid}"):
            evs[e["id"]] = e
    for slug in PM_EXTRA_SLUGS:
        f = RAW / f"polymarket/events/slug/{slug}.json.gz"
        if not f.exists():
            jdump(f, HTTP.get(GAMMA + "/events", {"slug": slug}) or [])
        for e in jload(f):
            evs[e["id"]] = e
    log(f"polymarket events: {len(evs)} unique")
    return list(evs.values())


GAME_LEVEL_TYPES = {"moneyline", "spreads", "totals", "team_totals", "first_half_moneyline", "first_half_spreads",
                    "first_half_totals", "first_quarter_moneyline", "first_quarter_spreads", "first_quarter_totals",
                    "second_half_moneyline", "both_teams_to_score"}
PROP_PATTERNS = [
    (re.compile(r"^(?P<p>.+?):\s*Anytime Touchdown\??$", re.I), "anytime_td"),
    (re.compile(r"^(?P<p>.+?)\s+Anytime Touchdown\??$", re.I), "anytime_td"),
    (re.compile(r"^First Touchdown:\s*(?P<p>.+)$", re.I), "first_td"),
    (re.compile(r"^(?P<p>.+?):\s*First Touchdown$", re.I), "first_td"),
    (re.compile(r"^(?P<p>.+?):\s*2\+ Touchdowns$", re.I), "two_plus_td"),
    (re.compile(r"^(?P<p>.+?):\s*(?P<s>[A-Za-z +/]+?)\s+O/U\s+(?P<t>[\d.]+)$", re.I), None),
    (re.compile(r"^(?P<p>.+?)\s+O/U\s+(?P<t>[\d.]+)\s+(?P<s>[A-Za-z ]+?)\??$", re.I), None),
    (re.compile(r"^Will (?P<p>.+?) score the first (?:touchdown|TD)", re.I), "first_td"),
    (re.compile(r"^Will (?P<p>.+?) score a rushing (?:touchdown|TD)", re.I), "rushing_td"),
    (re.compile(r"^Will (?P<p>.+?) (?:score|run for) an? (?:touchdown|TD)", re.I), "anytime_td"),
    (re.compile(r"^Will (?P<p>.+?) rush for (?P<t>\d+) or more yards", re.I), "rushing_yards"),
    (re.compile(r"^Will (?P<p>.+?) record (?P<t>\d+) or more (?P<s>receptions|receiving yards|passing yards|rushing yards)",
                re.I), None),
    (re.compile(r"^Will (?P<p>.+?) throw (?:an|a) interception", re.I), "interception_thrown"),
    (re.compile(r"^Will (?P<p>.+?) throw for over (?P<t>\d+) yards", re.I), "passing_yards"),
    (re.compile(r"^(?P<p>.+?) Rushing/Receiving TD\??$", re.I), "rush_rec_td"),
]


def parse_pm_prop(q: str):
    q = (q or "").strip()
    for rx, stat in PROP_PATTERNS:
        m = rx.match(q)
        if not m:
            continue
        p = m.group("p").strip()
        if re.search(r"\b(team|total|spread|vs|game|every|either|or)\b", p, re.I):
            continue
        if not re.fullmatch(r"[A-Z][\w.'’-]*(?: [A-Z][\w.'’-]*){1,3}", p):
            continue  # must look like a player name ("a lineman", "the first score" are not)
        gs = m.groupdict()
        s = stat or re.sub(r"\s+", "_", (gs.get("s") or "").strip().lower())
        if not re.search(r"yard|reception|touchdown|td|attempt|completion|interception|sack|tackle", s):
            continue
        t = fnum(gs.get("t"))
        if t is not None and stat in ("rushing_yards", "passing_yards") or (t is not None and "or more" in q):
            t = t - 0.5  # "N or more" == over N-0.5
        return p, s, t
    return None, None, None


def nick_teams(text: str) -> list[str]:
    found = []
    for w in re.findall(r"[a-z0-9]+", (text or "").lower()):
        c = NICKNAMES.get(w)
        if c and c not in found:
            found.append(c)
    return found


_QB = {}


def qb_game(gi: GameIndex, name: str, d: dt.date, tol: int = 3):
    """Game within +-tol days of d whose nflverse starting QB (home or away) is `name`."""
    if not _QB:
        f = OUT / "nflverse_lines.parquet"
        if not f.exists():
            build_nflverse()
        for r in pl.read_parquet(f).select("game_id", "gameday", "home_qb_name", "away_qb_name").iter_rows(named=True):
            for q in (r["home_qb_name"], r["away_qb_name"]):
                _QB.setdefault(norm_name(q), []).append((r["gameday"], r["game_id"]))
    hits = [gid for gd, gid in _QB.get(norm_name(name), []) if abs((gd - d).days) <= tol]
    return gi.by_id.get(hits[0]) if len(hits) == 1 else None


def poly_markets_table() -> pl.DataFrame:
    gi, cohort = GameIndex(), Cohort()
    rows = []
    for e in poly_fetch_events():
        slug = e.get("slug") or ""
        etitle = e.get("title") or ""
        end = utc(e.get("endDate"))
        ev_kind, ev_game = None, None
        m = PM_GAME_SLUG.match(slug)
        if m and m.group(1) not in ("dailies",):
            d = dt.date.fromisoformat(m.group(3))
            ev_kind = "game"
            ev_game = gi.find(d, m.group(1), m.group(2))
        elif PM_DAILIES.match(slug):
            ev_kind = "dailies"
        elif slug in PM_EXTRA_SLUGS:
            ev_kind = "sb_futures"
        else:
            if not end or not (DATE_MIN <= end.date() <= DATE_MAX + dt.timedelta(days=14)):
                continue
            ev_kind = "other"
            low = slug + " " + etitle.lower()
            if "super-bowl-lix" in low or "super bowl lix" in low or "super-box-lix" in low:
                ev_game = gi.by_id.get(2025020900)
            elif "super-bowl-lx" in low or "pro-football-championship" in low:
                ev_game = gi.by_id.get(2026020800)
            else:
                ts = nick_teams(slug + " " + etitle)
                if len(ts) == 2 and end:
                    ev_game = gi.find(end.date(), ts[0], ts[1], tol=3)
        for mk in e.get("markets") or []:
            q = mk.get("question") or ""
            smt = mk.get("sportsMarketType")
            outcomes = json.loads(mk.get("outcomes") or "[]")
            toks = json.loads(mk.get("clobTokenIds") or "[]")
            prices = [fnum(x) for x in json.loads(mk.get("outcomePrices") or "[]")]
            game = ev_game
            kind = None
            player = stat = thr = None
            if ev_kind == "dailies":
                mm = PM_GAME_SLUG.match(mk.get("slug") or "")
                if mm:
                    game = gi.find(dt.date.fromisoformat(mm.group(3)), mm.group(1), mm.group(2))
                kind = "moneyline"
            elif ev_kind == "game":
                kind = smt or ("moneyline" if len(e.get("markets") or []) == 1 or re.fullmatch(
                    r"[\w .']+ vs\.? [\w .']+", q) else None)
                if kind is None:
                    kind = "unknown"
            elif ev_kind == "sb_futures":
                ts = nick_teams(q)
                if not (ts and ts[0] in ("KC", "SF")):
                    continue  # non-finalist futures are out of scope
                game = gi.find(dt.date(2024, 2, 11), ts[0], tol=0)
                kind = "sb_futures_finalist"
            if kind is None or kind not in GAME_LEVEL_TYPES | {"moneyline", "sb_futures", "sb_futures_finalist",
                                                               "unknown"}:
                player, stat, thr = parse_pm_prop(mk.get("groupItemTitle") if smt and smt not in GAME_LEVEL_TYPES
                                                  and ":" in (mk.get("groupItemTitle") or "") else q)
                if player is None:
                    player, stat, thr = parse_pm_prop(q)
                if player:
                    kind = "player_prop"
                    if smt and smt not in GAME_LEVEL_TYPES:
                        stat = smt
                    if thr is None:
                        thr = fnum(mk.get("line"))
                elif kind is None:
                    kind = "other"
            if ev_kind == "other" and kind != "player_prop":
                continue  # keep only per-game player props from the non-game events
            if ev_kind == "other" and game is None and player and end:
                game = qb_game(gi, player, end.date())  # standalone QB passing-yards events carry no team names
            if ev_kind == "other" and game is None:
                kind = "player_prop_unmapped_game"
            gdate = game["game_date"] if game else None
            if game is None and ev_kind in ("game", "dailies"):
                mm = PM_GAME_SLUG.match(slug if ev_kind == "game" else (mk.get("slug") or ""))
                d = dt.date.fromisoformat(mm.group(3)) if mm else None
                if d is None or not (DATE_MIN <= d <= DATE_MAX):
                    continue
            nfl_id = how = cand = None
            if kind == "player_prop" and player:
                nfl_id, how, cand = cohort.match(player, game)
            win = None
            if mk.get("closed") and prices and max(p or 0 for p in prices) >= 0.99:
                win = max(range(len(prices)), key=lambda i: prices[i] or 0)
            oteams = [None, None]
            if kind in ("moneyline", "first_half_moneyline", "spreads", "first_half_spreads") and len(outcomes) == 2:
                oteams = [(nick_teams(o) or [team(o.strip()) if o.strip().upper() in KALSHI_TEAMS else None])[0]
                          for o in outcomes]
            rows.append(dict(
                market_id=mk.get("id"), condition_id=mk.get("conditionId"), question_id=mk.get("questionID"),
                slug=mk.get("slug"), question=q, group_item_title=mk.get("groupItemTitle"),
                event_id=e.get("id"), event_slug=slug, event_title=etitle, event_kind=ev_kind, market_kind=kind,
                sports_market_type=smt, outcomes=outcomes, token_ids=toks, final_outcome_prices=prices,
                winning_outcome_index=win, winning_outcome=outcomes[win] if win is not None and win < len(outcomes) else None,
                winning_token_id=toks[win] if win is not None and win < len(toks) else None,
                outcome0_team=oteams[0], outcome1_team=oteams[1],
                line=fnum(mk.get("line")), volume=fnum(mk.get("volumeNum", mk.get("volume"))),
                volume_clob=fnum(mk.get("volumeClob")), volume_amm=fnum(mk.get("volumeAmm")),
                liquidity=fnum(mk.get("liquidityNum", mk.get("liquidity"))),
                last_trade_price=fnum(mk.get("lastTradePrice")), best_bid=fnum(mk.get("bestBid")),
                best_ask=fnum(mk.get("bestAsk")), created_at=utc(mk.get("createdAt")),
                start_date=utc(mk.get("startDate")), end_date=utc(mk.get("endDate")),
                closed_time=utc(mk.get("closedTime")), game_start_time=utc(mk.get("gameStartTime")),
                accepting_orders_ts=utc(mk.get("acceptingOrdersTimestamp")), closed=mk.get("closed"),
                uma_resolution_status=mk.get("umaResolutionStatus"), neg_risk=mk.get("negRisk"),
                order_min_tick=fnum(mk.get("orderPriceMinTickSize")), market_maker_address=mk.get("marketMakerAddress"),
                game_id=game["game_id"] if game else None, season=game["season"] if game else None,
                season_type=game["season_type"] if game else None, week=game["week"] if game else None,
                game_date=gdate, home_team=game["home"] if game else None, away_team=game["away"] if game else None,
                player_name=player, stat=stat, threshold=thr, nfl_id=nfl_id, name_match=how,
                cohort_candidate_id=cand))
    df = pl.DataFrame(rows, infer_schema_length=None).unique("condition_id", keep="first", maintain_order=True)
    return df


def poly_targets(pm: pl.DataFrame) -> pl.DataFrame:
    heavy = pl.col("season_type").is_in(HEAVY_SEASON_TYPES)
    win = pm.filter(pl.col("market_kind").is_in(["moneyline", "sb_futures_finalist"]) & pl.col("game_id").is_not_null()
                    & heavy)
    props = pm.filter((pl.col("market_kind") == "player_prop") & pl.col("nfl_id").is_not_null() & heavy)
    return pl.concat([win.with_columns(target=pl.lit("game_winner")), props.with_columns(target=pl.lit("cohort_prop"))])


def poly_fetch_prices() -> None:
    pm = poly_markets_table()
    tg = poly_targets(pm)
    jobs = []
    for r in tg.iter_rows(named=True):
        a = r["accepting_orders_ts"] or r["created_at"] or r["start_date"]
        b = (r["closed_time"] or r["end_date"] or a) + dt.timedelta(days=1)
        if a is None:
            continue
        a = max(a, b - dt.timedelta(days=120))  # pre-season-long listings: keep the last 120 days
        for tok in r["token_ids"]:
            jobs.append((tok, to_ts(a), to_ts(b)))
    log(f"polymarket prices: {len(jobs)} tokens")

    def work(job):
        tok, a, b = job
        f = RAW / f"polymarket/prices/{tok}.json.gz"
        if f.exists():
            return 0
        pts, s = [], a
        while s < b:  # 14-day windows, hourly fidelity
            e = min(b, s + 14 * 86400)
            page = HTTP.get(CLOB + "/prices-history", {"market": tok, "startTs": s, "endTs": e, "fidelity": 60}) or {}
            pts += page.get("history") or []
            s = e
        jdump(f, {"token_id": tok, "start_ts": a, "end_ts": b, "fidelity": 60, "history": pts})
        return len(pts)

    with ThreadPoolExecutor(max_workers=5) as ex:
        n = 0
        for i, k in enumerate(ex.map(work, jobs)):
            n += k
            if i % 500 == 0:
                log(f"  prices {i}/{len(jobs)}")
    log(f"polymarket prices fetched: {n:,} new points")


def poly_fetch_trades_one(cid: str, taker_only: bool = False) -> int:
    """All fills of one market. taker_only=False returns both the taker and the maker row(s) of every fill;
    the taker_only=True pass is fetched separately only to flag which rows are the taker side."""
    f = RAW / f"polymarket/{'trades_taker' if taker_only else 'trades'}/{cid}.json.gz"
    if f.exists():
        return 0
    rows, cur = [], None
    keep = ("proxy_wallet", "side", "token_id", "condition_id", "size", "price", "timestamp", "outcome",
            "outcome_index", "transaction_hash", "name", "pseudonym")
    while True:
        q = {"condition": cid, "taker_only": "true" if taker_only else "false"}
        q.update({"cursor": cur} if cur else {"limit": 1000})
        page = HTTP.get(PMDATA + "/v2/trades", q) or {}
        got = page.get("data") or []
        rows += [{k: x.get(k) for k in keep} for x in got]  # drop per-row title/icon/bio/profile-image fields
        cur = (page.get("pagination") or {}).get("next_cursor")
        if not cur or not got:
            break
    jdump(f, {"condition_id": cid, "taker_only": taker_only, "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "trades": rows})
    return len(rows)


def poly_fetch_trades(taker_only: bool = False) -> None:
    pm = poly_markets_table()
    tg = poly_targets(pm)
    cids = tg["condition_id"].unique().to_list()
    log(f"polymarket trades (taker_only={taker_only}): {len(cids)} markets")
    n = 0
    with ThreadPoolExecutor(max_workers=5) as ex:
        for i, k in enumerate(ex.map(lambda c: poly_fetch_trades_one(c, taker_only), cids)):
            n += k
            if i % 100 == 0:
                log(f"  trades markets {i}/{len(cids)} rows {n:,}")
    log(f"polymarket trades fetched: {n:,} new rows")


# ---------------------------------------------------------------- build tidy tables
def build_kalshi() -> None:
    mk, legs = kalshi_markets_table()
    ko = kickoffs()
    mk = mk.with_columns(kickoff_utc=pl.col("game_id").replace_strict(ko, default=None, return_dtype=pl.Datetime("us")))
    # trades
    manifest = jload(RAW / "kalshi/trades_manifest.json.gz") if (RAW / "kalshi/trades_manifest.json.gz").exists() else {}
    tmeta = {r["ticker"]: r for r in mk.select("ticker", "series_ticker", "game_id", "nfl_id", "kickoff_utc").to_dicts()}
    frames = []
    for tk, st in manifest.items():
        if st["mode"] not in ("full", "pregame") or tk not in tmeta:
            continue
        r = tmeta[tk]
        k = ko.get(r["game_id"])
        tag = "" if st["mode"] == "full" or k is None else f"__max{to_ts(k)}"
        f = RAW / f"kalshi/trades/{safe_name(r['series_ticker'])}/{safe_name(tk)}{tag}.json.gz"
        if not f.exists():
            continue
        tr = jload(f)["trades"]
        if not tr:
            continue
        frames.append(pl.DataFrame({
            "trade_id": [t["trade_id"] for t in tr], "ticker": tk,
            "created_time": [utc(t["created_time"]) for t in tr],
            "yes_price": [fnum(t.get("yes_price_dollars")) for t in tr],
            "no_price": [fnum(t.get("no_price_dollars")) for t in tr],
            "count": [fnum(t.get("count_fp", t.get("count"))) for t in tr],
            "taker_side": [t.get("taker_outcome_side") or t.get("taker_side") for t in tr],
            "taker_book_side": [t.get("taker_book_side") for t in tr],
            "is_block_trade": [t.get("is_block_trade") for t in tr],
            "fetch_mode": st["mode"]}, schema_overrides={"created_time": pl.Datetime("us")}))
    tr = (pl.concat(frames) if frames else pl.DataFrame())
    if tr.height:
        tr = (tr.join(mk.select("ticker", "series_ticker", "event_ticker", "market_kind", "game_id", "nfl_id",
                                "kickoff_utc"), on="ticker", how="left")
              .with_columns(pregame=pl.col("created_time") < pl.col("kickoff_utc"))
              .sort("ticker", "created_time"))
        tr.drop("kickoff_utc").write_parquet(OUT / "kalshi_trades.parquet")
    # candles
    cframes = []
    for f in sorted((RAW / "kalshi/candles").rglob("*.json.gz")):
        o = jload(f)
        cs = o["candlesticks"]
        if not cs:
            continue

        def g(c, a, b):
            return fnum((c.get(a) or {}).get(b))
        cframes.append(pl.DataFrame({
            "ticker": o["ticker"], "end_period_ts": [utc(c["end_period_ts"]) for c in cs],
            "price_open": [g(c, "price", "open") for c in cs], "price_high": [g(c, "price", "high") for c in cs],
            "price_low": [g(c, "price", "low") for c in cs], "price_close": [g(c, "price", "close") for c in cs],
            "price_mean": [g(c, "price", "mean") for c in cs], "price_previous": [g(c, "price", "previous") for c in cs],
            "yes_bid_open": [g(c, "yes_bid", "open") for c in cs], "yes_bid_close": [g(c, "yes_bid", "close") for c in cs],
            "yes_bid_high": [g(c, "yes_bid", "high") for c in cs], "yes_bid_low": [g(c, "yes_bid", "low") for c in cs],
            "yes_ask_open": [g(c, "yes_ask", "open") for c in cs], "yes_ask_close": [g(c, "yes_ask", "close") for c in cs],
            "yes_ask_high": [g(c, "yes_ask", "high") for c in cs], "yes_ask_low": [g(c, "yes_ask", "low") for c in cs],
            "volume": [fnum(c.get("volume_fp", c.get("volume"))) for c in cs],
            "open_interest": [fnum(c.get("open_interest_fp", c.get("open_interest"))) for c in cs]},
            schema_overrides={"end_period_ts": pl.Datetime("us")}))
    cd = pl.concat(cframes) if cframes else pl.DataFrame()
    if cd.height:
        cd = (cd.join(mk.select("ticker", "series_ticker", "event_ticker", "market_kind", "game_id", "kickoff_utc"),
                      on="ticker", how="left")
              .with_columns(pregame=pl.col("end_period_ts") <= pl.col("kickoff_utc")).sort("ticker", "end_period_ts"))
        cd.drop("kickoff_utc").write_parquet(OUT / "kalshi_candles.parquet")
    # pre-game price per market: last trade before kickoff, else close of last pre-kickoff hourly candle
    pre = []
    if tr.height:
        pre.append(tr.filter(pl.col("pregame")).group_by("ticker").agg(
            pregame_last_price=pl.col("yes_price").sort_by("created_time").last(),
            pregame_last_trade_time=pl.col("created_time").max(),
            pregame_volume=pl.col("count").sum(), n_trades_fetched=pl.len()).with_columns(pregame_price_source=pl.lit("trades")))
    if cd.height:
        cc = (cd.filter(pl.col("pregame") & pl.col("price_close").is_not_null()).group_by("ticker").agg(
            pregame_last_price=pl.col("price_close").sort_by("end_period_ts").last(),
            pregame_last_trade_time=pl.col("end_period_ts").max(), pregame_volume=pl.col("volume").sum())
            .with_columns(pregame_price_source=pl.lit("candles_1h")))
        if pre:
            cc = cc.join(pre[0].select("ticker"), on="ticker", how="anti")
        pre.append(cc)
    if pre:
        pr = pl.concat(pre, how="diagonal_relaxed")
        mk = mk.join(pr, on="ticker", how="left")
    mk.write_parquet(OUT / "kalshi_markets.parquet")
    legs = legs.join(mk.select(pl.col("ticker").alias("leg_market_ticker"), pl.col("series_ticker").alias("leg_series"),
                               pl.col("market_kind").alias("leg_kind"), pl.col("game_id").alias("leg_game_id"),
                               pl.col("player_name").alias("leg_player_name"), pl.col("stat").alias("leg_stat"),
                               pl.col("threshold").alias("leg_threshold"), pl.col("nfl_id").alias("leg_nfl_id"),
                               pl.col("result").alias("leg_result")),
                     on="leg_market_ticker", how="left")
    legs.write_parquet(OUT / "kalshi_mve_legs.parquet")
    log(f"kalshi_markets {mk.height:,}  legs {legs.height:,}  trades {tr.height:,}  candles {cd.height:,}")


def build_polymarket() -> None:
    pm = poly_markets_table()
    ko = kickoffs()
    pm = pm.with_columns(kickoff_utc=pl.col("game_id").replace_strict(ko, default=None, return_dtype=pl.Datetime("us")))
    pm = pm.with_columns(kickoff_utc=pl.coalesce("kickoff_utc", "game_start_time"))
    tok_meta = {}
    for r in pm.select("condition_id", "token_ids", "outcomes", "game_id", "market_kind", "nfl_id").to_dicts():
        for i, t in enumerate(r["token_ids"] or []):
            tok_meta[t] = (r["condition_id"], i, (r["outcomes"] or [None] * 9)[i] if i < len(r["outcomes"] or []) else None)
    # prices
    frames = []
    for f in sorted((RAW / "polymarket/prices").glob("*.json.gz")):
        o = jload(f)
        h = o["history"]
        if not h:
            continue
        cid, idx, outc = tok_meta.get(o["token_id"], (None, None, None))
        frames.append(pl.DataFrame({"token_id": o["token_id"], "condition_id": cid, "outcome_index": idx,
                                    "outcome": outc, "t": [utc(p["t"]) for p in h], "price": [fnum(p["p"]) for p in h]},
                                   schema_overrides={"t": pl.Datetime("us"), "outcome_index": pl.Int64,
                                                     "condition_id": pl.Utf8, "outcome": pl.Utf8}))
    px = pl.concat(frames).unique(["token_id", "t"]).sort("token_id", "t") if frames else pl.DataFrame()
    if px.height:
        px = (px.join(pm.select("condition_id", "market_kind", "game_id", "nfl_id", "kickoff_utc"), on="condition_id",
                      how="left").with_columns(pregame=pl.col("t") < pl.col("kickoff_utc")))
        px.drop("kickoff_utc").write_parquet(OUT / "polymarket_prices.parquet")
    # trades
    frames = []
    for f in sorted((RAW / "polymarket/trades").glob("*.json.gz")):
        o = jload(f)
        tr = o["trades"]
        if not tr:
            continue
        frames.append(pl.DataFrame(tr, schema={"proxy_wallet": pl.Utf8, "side": pl.Utf8, "token_id": pl.Utf8,
                                               "condition_id": pl.Utf8, "size": pl.Float64, "price": pl.Float64,
                                               "timestamp": pl.Int64, "outcome": pl.Utf8, "outcome_index": pl.Int64,
                                               "transaction_hash": pl.Utf8, "name": pl.Utf8, "pseudonym": pl.Utf8}))
    td = pl.concat(frames) if frames else pl.DataFrame()
    key = ["transaction_hash", "proxy_wallet", "token_id", "side", "size", "price"]
    tk = []
    for f in sorted((RAW / "polymarket/trades_taker").glob("*.json.gz")):
        rows = jload(f)["trades"]
        if rows:
            tk.append(pl.DataFrame([{k: r.get(k) for k in key} for r in rows],
                                   schema={"transaction_hash": pl.Utf8, "proxy_wallet": pl.Utf8, "token_id": pl.Utf8,
                                           "side": pl.Utf8, "size": pl.Float64, "price": pl.Float64}))
    taker_cids = {f.name.split(".")[0] for f in (RAW / "polymarket/trades_taker").glob("*.json.gz")}
    if td.height:
        if tk:
            tk = pl.concat(tk).unique().with_columns(_taker=pl.lit(True))
            td = td.join(tk, on=key, how="left")
        else:
            td = td.with_columns(_taker=pl.lit(None, pl.Boolean))
        # taker flag only where the taker pass exists for that market (null otherwise)
        td = td.with_columns(is_taker=pl.when(pl.col("condition_id").is_in(list(taker_cids)))
                             .then(pl.col("_taker").fill_null(False)).otherwise(None)).drop("_taker")
        td = (td.with_columns(ts=pl.from_epoch("timestamp", time_unit="s"), usdc=pl.col("size") * pl.col("price"))
              .join(pm.select("condition_id", "market_kind", "game_id", "nfl_id", "kickoff_utc", "winning_outcome_index"),
                    on="condition_id", how="left")
              .with_columns(pregame=pl.col("ts") < pl.col("kickoff_utc"),
                            outcome_won=pl.col("outcome_index") == pl.col("winning_outcome_index"))
              .drop("kickoff_utc", "winning_outcome_index").sort("condition_id", "timestamp"))
        td.write_parquet(OUT / "polymarket_trades.parquet")
        # pre-game price: last trade before kickoff on outcome 0
        pre = (td.filter(pl.col("pregame") & (pl.col("outcome_index") == 0)).group_by("condition_id")
               .agg(pregame_last_price0=pl.col("price").sort_by("timestamp").last(),
                    pregame_last_trade_time=pl.col("ts").max()))
        pm = pm.join(pre, on="condition_id", how="left")
        n_tr = td.group_by("condition_id").agg(n_trade_rows=pl.len(), first_trade=pl.col("ts").min())
        pm = pm.join(n_tr, on="condition_id", how="left")
    pm.write_parquet(OUT / "polymarket_markets.parquet")
    log(f"polymarket_markets {pm.height:,}  prices {px.height:,}  trades {td.height:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["nflverse", "kalshi-discover", "kalshi-markets", "kalshi-trades", "kalshi-candles",
                                      "kalshi", "poly-events", "poly-prices", "poly-trades", "poly-trades-taker",
                                      "polymarket", "build", "all"])
    ap.add_argument("--max-trades", type=int, default=3_000_000, help="Kalshi trade budget (rows) for full histories")
    ap.add_argument("--max-minutes", type=float, default=90, help="Kalshi trade fetch time budget")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    st = a.stage
    if st in ("nflverse", "all"):
        build_nflverse()
    if st in ("kalshi-discover", "kalshi", "all"):
        kalshi_discover()
    if st in ("kalshi-markets", "kalshi", "all"):
        kalshi_fetch_markets()
    if st in ("kalshi-trades", "kalshi", "all"):
        kalshi_fetch_trades(a.max_trades, a.max_minutes)
    if st in ("kalshi-candles", "kalshi", "all"):
        kalshi_fetch_candles()
    if st in ("poly-events", "polymarket", "all"):
        poly_fetch_events()
    if st in ("poly-prices", "polymarket", "all"):
        poly_fetch_prices()
    if st in ("poly-trades", "polymarket", "all"):
        poly_fetch_trades()
    if st in ("poly-trades-taker", "polymarket", "all"):
        poly_fetch_trades(taker_only=True)
    if st in ("build", "kalshi", "all"):
        build_kalshi()
    if st in ("build", "polymarket", "all"):
        build_polymarket()
    log("http calls", dict(HTTP.calls), "retries", dict(HTTP.retries))


if __name__ == "__main__":
    main()
