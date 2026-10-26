"""Step 1: frame-level kinematics (v, a_lon, a_lat, omega) for combine reps and game player-plays, same filter in both."""
import time

import polars as pl

from common import CACHE, OUT, kinematics

t0 = time.time()

# --- combine: player rows only, normalized rep frame from the cache
reps = (pl.scan_parquet(CACHE / "combine_reps.parquet")
          .filter(pl.col("entity_type") == "PLAYER", pl.col("theta").is_not_null())
          .sort("event_id", "time"))
kc = (kinematics(reps, ["event_id"])
        .select("event_id", "nfl_id", "draft_year", "drill_type", "drill_name", "attempt", "time", "t_rep",
                "x_rep", "y_rep", "s", "a", "v", "a_lon", "a_lat", "a_tot", "omega")
        .collect())
kc.write_parquet(OUT / "kin_combine.parquet")
print(f"combine frames {kc.height:,}  reps {kc['event_id'].n_unique():,}  {time.time() - t0:.0f}s")

# --- game: aligned frames, -1 s .. +10 s around the snap, plays we can align
gt = (pl.scan_parquet(CACHE / "game_tracking.parquet")
        .filter(pl.col("play_direction").is_not_null(), pl.col("t_snap").is_between(-1.5, 10.5))
        .with_columns(x=pl.col("x_std"), y=pl.col("y_std"))
        .sort("game_id", "play_id", "nfl_id", "time"))
key = ["game_id", "play_id", "nfl_id"]
kg = (kinematics(gt, key)
        .filter(pl.col("t_snap").is_between(-1.0, 10.0))
        .select("game_id", "play_id", "nfl_id", "season", "time", "t_snap", "frame_snap", "event",
                "x_std", "y_std", "x_los", "dir_std", "o_std", "s", "a", "v", "a_lon", "a_lat", "a_tot", "omega")
        .collect())
kg.write_parquet(OUT / "kin_game.parquet")
print(f"game frames {kg.height:,}  player-plays {kg.select(pl.struct(key).n_unique()).item():,}  {time.time() - t0:.0f}s")

# --- consistency of our derivatives with the provided NGS columns, per domain
for name, df in (("combine", kc), ("game", kg)):
    m = df.filter(pl.col("v") > 3).select((pl.col("v") / pl.col("s")).median().alias("v/s"),
                                          (pl.col("a_tot") / pl.col("a").clip(0.1)).median().alias("a_tot/a"),
                                          pl.col("v").null_count().alias("null_v"))
    print(name, m.rows(named=True))
