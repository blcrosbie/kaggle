"""Shared paths and helpers for the deep-dive scripts (run from the project root or scripts/deepdive)."""
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/parquet"           # built by notebooks/00_build_cache.ipynb (STARTER.md §4)
OUT = ROOT / "data/deepdive"            # derived tables / models (gitignored)
FIG = ROOT / "docs/deepdive"            # figures referenced by DEEPDIVE.md (tracked; reports/ is gitignored)
MARKETS = ROOT / "data/markets"         # betting-market tables (scripts/deepdive/fetch_markets.py)
for p in (OUT, FIG):
    p.mkdir(parents=True, exist_ok=True)

DT = 0.1  # 10 Hz
# Savitzky-Golay (window 7, quadratic) derivative kernels, centered: sum_j c_j * x[i+j], j = -3..3
SG_D1 = [-3, -2, -1, 0, 1, 2, 3]
SG_D1_NORM = 28 * DT
SG_D2 = [5, 0, -3, -4, -3, 0, 5]
SG_D2_NORM = 42 * DT * DT

POS_GROUPS = ["DB", "DL", "OL", "TE", "WR"]


def sg(col: str, kernel: list[int], norm: float, over: list[str]) -> pl.Expr:
    """Centered Savitzky-Golay filter applied within groups (nulls at the 3 edge frames)."""
    return sum(c * pl.col(col).shift(-j).over(over) for j, c in zip(range(-3, 4), kernel) if c != 0) / norm


def kinematics(lf: pl.LazyFrame, over: list[str]) -> pl.LazyFrame:
    """Add v, a_lon, a_lat, a_tot, omega from x/y (yards, 10 Hz). Window must not span a time gap."""
    valid = ((pl.col("time").shift(-3).over(over) - pl.col("time").shift(3).over(over)).dt.total_milliseconds() == 600)
    return (lf.with_columns(vx=sg("x", SG_D1, SG_D1_NORM, over), vy=sg("y", SG_D1, SG_D1_NORM, over),
                            ax=sg("x", SG_D2, SG_D2_NORM, over), ay=sg("y", SG_D2, SG_D2_NORM, over), _ok=valid)
              .with_columns(v=(pl.col("vx") ** 2 + pl.col("vy") ** 2).sqrt())
              .with_columns(a_lon=(pl.col("vx") * pl.col("ax") + pl.col("vy") * pl.col("ay")) / pl.col("v").clip(0.3),
                            # + = turning left (counter-clockwise); flip sign when mirroring
                            a_lat=(pl.col("vx") * pl.col("ay") - pl.col("vy") * pl.col("ax")) / pl.col("v").clip(0.3),
                            a_tot=(pl.col("ax") ** 2 + pl.col("ay") ** 2).sqrt())
              .with_columns(omega=pl.col("a_lat") / pl.col("v").clip(0.5))
              .with_columns([pl.when(pl.col("_ok")).then(pl.col(c)).otherwise(None).alias(c)
                             for c in ["v", "a_lon", "a_lat", "a_tot", "omega"]])
              .drop("vx", "vy", "ax", "ay", "_ok"))


def players() -> pl.DataFrame:
    return (pl.read_parquet(CACHE / "players.parquet")
              .join(pl.read_parquet(CACHE / "combine_results.parquet"), on=["nfl_id", "draft_year"], how="left")
              .join(pl.read_parquet(CACHE / "player_career_successes.parquet"), on="nfl_id", how="left"))
