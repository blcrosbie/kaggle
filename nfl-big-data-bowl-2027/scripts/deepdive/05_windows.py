"""Step 5: fixed-length movement windows for the deep encoder (rotation/translation invariant channels).

Channels: v, a_lon, a_lat, omega (clipped). Window = 32 frames (3.2 s).
- combine: sliding windows (stride 8) over each rep, max v >= 1
- game: frames 0-31 and 32-63 after the snap of each player-play, max v >= 1
Output: data/deepdive/windows.npz (X float16 [N, 4, 32]) + windows_meta.parquet (row-aligned).
"""
import numpy as np
import polars as pl

from common import CACHE, OUT

L, STRIDE = 32, 8
CH = ["v", "a_lon", "a_lat", "omega"]
CLIP = {"v": (0, 13), "a_lon": (-12, 12), "a_lat": (-12, 12), "omega": (-6, 6)}


def clipped():
    return [pl.col(c).clip(*CLIP[c]) for c in CH]


# --- combine
kc = (pl.read_parquet(OUT / "kin_combine.parquet").sort("event_id", "time")
        .filter(pl.col("v").is_not_null()).with_columns(clipped()))
cx, cmeta = [], []
for (eid, nid, drill), g in kc.group_by("event_id", "nfl_id", "drill_name"):
    arr = g.select(CH).to_numpy().T                                       # [4, T]
    for s in range(0, arr.shape[1] - L + 1, STRIDE):
        w = arr[:, s:s + L]
        if w[0].max() >= 1.0:
            cx.append(w)
            cmeta.append((nid, eid, drill, s))
cx = np.stack(cx).astype(np.float16)
cmeta = pl.DataFrame(cmeta, schema=["nfl_id", "event_id", "drill_name", "start"], orient="row").with_columns(
    domain=pl.lit("combine"))
print(f"combine windows {len(cx):,} from {cmeta['event_id'].n_unique():,} reps, {cmeta['nfl_id'].n_unique()} players")

# --- game
pp = pl.read_parquet(CACHE / "player_play.parquet",
                     columns=["game_id", "play_id", "nfl_id", "lined_up_position", "route_ran", "pass_result",
                              "coverage_assignment", "team_abbr"])
kg = (pl.scan_parquet(OUT / "kin_game.parquet")
        .filter(pl.col("frame_snap").is_between(0, 2 * L - 1))
        .with_columns(clipped(), w=pl.col("frame_snap") // L)
        .sort("game_id", "play_id", "nfl_id", "frame_snap")
        .group_by("game_id", "play_id", "nfl_id", "season", "w")
        .agg([pl.col(c) for c in CH] + [pl.col("frame_snap").n_unique().alias("n")])
        .filter(pl.col("n") == L, pl.col("v").list.max() >= 1.0)
        .filter(~pl.any_horizontal([pl.col(c).list.eval(pl.element().is_null()).list.any() for c in CH]))
        .collect())
gx = np.stack([kg[c].list.to_array(L).to_numpy() for c in CH], axis=1).astype(np.float16)
gmeta = (kg.select("nfl_id", "game_id", "play_id", "season", "w")
           .join(pp, on=["game_id", "play_id", "nfl_id"], how="left")
           .with_columns(domain=pl.lit("game")))
print(f"game windows {len(gx):,} from {kg.select(pl.struct('game_id', 'play_id', 'nfl_id').n_unique()).item():,} player-plays")

X = np.concatenate([cx, gx])
meta = pl.concat([cmeta, gmeta], how="diagonal_relaxed").with_row_index("row")
np.savez(OUT / "windows.npz", X=X)
meta.write_parquet(OUT / "windows_meta.parquet")
print("X", X.shape, X.dtype, " channel means", X.astype(np.float32).mean(axis=(0, 2)).round(2),
      " stds", X.astype(np.float32).std(axis=(0, 2)).round(2))
