"""Step 9: one row per player-game with sensor-derived behavior scores (input to the market/integrity layer).

fp_z      fingerprint drift: cosine similarity of this game's (context-centered) movement prototype to the
          player's leave-this-game-out prototype, z-scored within player (negative = moved unlike himself)
eff_*_z   effort residuals: mean over the game's plays of (actual - expected) from the context-only play model
          (vmax_5s, burst, dist_3s), z-scored within player
box       targets, receptions, rec_yards, rec_td, sacks, tackles, pressures allowed, snaps
Output: data/deepdive/player_games.parquet
"""
import numpy as np
import polars as pl

from common import CACHE, OUT

E = np.load(OUT / "emb_oof.npy")
wm = pl.read_parquet(OUT / "windows_meta.parquet")
g = (wm["domain"] == "game").to_numpy()
gm = (wm.filter(pl.col("domain") == "game")
        .with_columns(ctx=pl.concat_str([pl.col("lined_up_position").fill_null("?"), pl.col("route_ran").fill_null("-"),
                                         pl.col("pass_result").is_null().cast(pl.String), pl.col("w").cast(pl.String)], separator="|")))
Eg = E[g].copy()
ctx = gm["ctx"].to_numpy()
for c in np.unique(ctx):                       # context-centering over all game windows (unsupervised)
    m = ctx == c
    Eg[m] -= Eg[m].mean(0)

# per player-game sums of embeddings -> leave-one-game-out prototypes
key = gm.select("nfl_id", "game_id").with_row_index("i")
pg = key.group_by("nfl_id", "game_id").agg(pl.col("i")).sort("nfl_id", "game_id")
rows = []
for (nid,), grp in pg.group_by("nfl_id", maintain_order=True):
    if grp.height < 5:
        continue
    sums = np.stack([Eg[np.array(ix)].sum(0) for ix in grp["i"]])
    cnt = np.array([len(ix) for ix in grp["i"]])
    tot = sums.sum(0)
    for k in range(grp.height):
        this = sums[k] / cnt[k]
        rest = (tot - sums[k]) / (cnt.sum() - cnt[k])
        sim = this @ rest / (np.linalg.norm(this) * np.linalg.norm(rest) + 1e-9)
        rows.append((nid, grp["game_id"][k], int(cnt[k]), float(sim)))
fp = (pl.DataFrame(rows, schema=["nfl_id", "game_id", "fp_windows", "fp_sim"], orient="row")
        .with_columns(fp_z=((pl.col("fp_sim") - pl.col("fp_sim").mean().over("nfl_id")) / pl.col("fp_sim").std().over("nfl_id"))))

# effort residuals from the play-level model (out-of-fold predictions)
oof = pl.read_parquet(OUT / "play_oof.parquet").filter(pl.col("target").is_in(["vmax_5s", "burst", "dist_3s"]))
eff = (oof.with_columns(r=pl.col("y") - pl.col("pred_ctx"))
          .group_by("nfl_id", "game_id", "target").agg(pl.col("r").mean(), pl.len())
          .with_columns(z=((pl.col("r") - pl.col("r").mean().over("nfl_id", "target")) / pl.col("r").std().over("nfl_id", "target")))
          .pivot(on="target", index=["nfl_id", "game_id"], values=["z", "len"])
          .rename({"z_vmax_5s": "eff_vmax_z", "z_burst": "eff_burst_z", "z_dist_3s": "eff_dist_z", "len_vmax_5s": "eff_plays"})
          .select("nfl_id", "game_id", "eff_vmax_z", "eff_burst_z", "eff_dist_z", "eff_plays"))

pp = pl.read_parquet(CACHE / "player_play.parquet")
box = (pp.with_columns(rec=(pl.col("target") & (pl.col("pass_result") == "C")).fill_null(False),
                       td=pl.col("play_description").str.contains("TOUCHDOWN") & pl.col("play_nullified_by_penalty").ne("Y").fill_null(True))
         .group_by("nfl_id", "game_id", "team_abbr")
         .agg(snaps=pl.len(),
              targets=pl.col("target").fill_null(False).sum(),
              receptions=pl.col("rec").sum(),
              rec_yards=pl.col("rec_yards").fill_null(0).sum(),
              rec_td=(pl.col("rec") & pl.col("td")).sum(),
              sacks=pl.col("sack").fill_null(0).sum(),
              tackles=pl.col("tackle").fill_null(0).sum(),
              pressures_allowed=pl.col("pressure_allowed").fill_null(False).sum(),
              sacks_allowed=pl.col("sack_allowed").fill_null(0).sum(),
              mean_sep=pl.col("separation_at_pass_forward").mean()))
games = pl.read_parquet(CACHE / "games.parquet")
out = (box.join(fp, on=["nfl_id", "game_id"], how="left").join(eff, on=["nfl_id", "game_id"], how="left")
          .join(games.select("game_id", "season", "season_type", "week", "home_team_abbr", "visitor_team_abbr"), on="game_id"))
out.write_parquet(OUT / "player_games.parquet")
print(out.height, "player-games;", out["fp_z"].is_not_null().sum(), "with fingerprint z;", out["eff_vmax_z"].is_not_null().sum(), "with effort z")
print(out.select("fp_z", "eff_vmax_z", "eff_burst_z", "eff_dist_z").drop_nulls().fill_nan(None).drop_nulls().corr())
