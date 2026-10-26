"""Step 8: play-level conditional transfer. Given play context, does a player's combine TRACKING improve
predictions of what he does on each play, for draft classes the model has never seen?

Targets (per player-play):  vmax_5s (max speed 0-5 s after snap), burst (max a_lon 0-1.5 s), dist_3s (yards in 0-3 s),
                            separation (WR/TE/RB: separation_at_pass_forward), get_off (DL: player_get_off),
                            pressure_allowed (OL, binary)
Feature sets (nested):      ctx      play context only (+ combine_position)
                            official ctx + draft pick + official measurables
                            tracking ctx + combine-tracking envelope, sprint profile, cut scores, DL combine prototype (PCA-8)
                            all      ctx + official + tracking
CV: leave-one-draft-class-out (players in the test class are never seen). Metrics on held-out plays: R^2 (AUC for
binary) and "player-effect R^2" = how much of the held-out players' mean residual (vs the ctx model) each set explains.
Outputs: data/deepdive/play_table.parquet, play_cv.parquet, play_oof.parquet (OOF predictions for the anomaly layer)
"""
import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from common import CACHE, OUT, players

# ---------------- play table
kin = (pl.scan_parquet(OUT / "kin_game.parquet").filter(pl.col("t_snap").is_between(0, 5))
         .group_by("game_id", "play_id", "nfl_id")
         .agg(vmax_5s=pl.col("v").max(),
              burst=pl.col("a_lon").filter(pl.col("t_snap") <= 1.5).max(),
              dist_3s=(pl.col("v").filter(pl.col("t_snap") <= 3) * 0.1).sum(),
              x_los_snap=pl.col("x_los").filter(pl.col("frame_snap") == 0).first(),
              n=pl.len())
         .filter(pl.col("n") >= 30).drop("n").collect())
pp = pl.read_parquet(CACHE / "player_play.parquet")
games = pl.read_parquet(CACHE / "games.parquet").select("game_id", "season", "season_type", "week", "home_team_abbr")
ctx_cat = ["lined_up_position", "route_ran", "offense_formation", "receiver_alignment", "team_coverage_type",
           "team_coverage_man_zone", "coverage_assignment", "season_type", "combine_position"]
t = (pp.join(games, on="game_id", how="left")
       .with_columns(is_off=pl.col("team_abbr") == pl.col("possession_team"),
                     is_pass=pl.col("pass_result").is_not_null(),
                     score_diff=pl.when(pl.col("possession_team") == pl.col("home_team_abbr"))
                                  .then(pl.col("pre_snap_home_score") - pl.col("pre_snap_visitor_score"))
                                  .otherwise(pl.col("pre_snap_visitor_score") - pl.col("pre_snap_home_score")),
                     clock_s=pl.col("game_clock").str.split(":").list.eval(pl.element().cast(pl.Int32)).list.first() * 60
                             + pl.col("game_clock").str.split(":").list.eval(pl.element().cast(pl.Int32)).list.last())
       .select("game_id", "play_id", "nfl_id", "team_abbr", "season", "week", "season_type", "lined_up_position", "route_ran",
               "offense_formation", "receiver_alignment", "team_coverage_type", "team_coverage_man_zone",
               "coverage_assignment", "in_motion_at_ball_snap", "is_off", "is_pass", "down", "yards_to_go",
               "quarter", "clock_s", "score_diff", "yardline_number", "separation_at_pass_forward", "player_get_off",
               "pressure_allowed", "target", "cushion")
       .join(kin, on=["game_id", "play_id", "nfl_id"], how="inner"))

# ---------------- player features
pinfo = players()
OFFICIAL = ["draft_overall_pick", "combine_height", "combine_weight", "hand_size", "arm_length", "forty", "ten_yd_split",
            "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps"]
env = pl.read_parquet(OUT / "envelopes.parquet")
TRACK_ENV = ["c_vmax", "c_brake", "c_lat_4_6", "c_lat_6_8", "c_acc_low", "c_omega_4_6", "c_as_A0", "c_as_V0",
             "sp_vmax", "sp_tau", "sp_A0"]
cutp = pl.read_parquet(OUT / "cut_players.parquet").select("nfl_id", "c_cut_ret", "c_cut_reacc")
# DL combine prototype from out-of-fold embeddings (encoder never saw the player's class)
E = np.load(OUT / "emb_oof.npy")
wm = pl.read_parquet(OUT / "windows_meta.parquet")
cm = (wm["domain"] == "combine").to_numpy()
ids_c = wm["nfl_id"].to_numpy()[cm]
u = np.unique(ids_c)
proto = np.stack([E[cm][ids_c == i].mean(0) for i in u])
pf = (pinfo.select(["nfl_id", "draft_year", "combine_position"] + OFFICIAL)
          .join(env.select(["nfl_id"] + TRACK_ENV), on="nfl_id", how="left")
          .join(cutp, on="nfl_id", how="left"))
t = t.join(pf, on="nfl_id", how="inner")
t.write_parquet(OUT / "play_table.parquet")
print(f"play table {t.height:,} rows, {t['nfl_id'].n_unique()} players")

CTX = ctx_cat + ["in_motion_at_ball_snap", "is_off", "is_pass", "down", "yards_to_go", "quarter", "clock_s",
                 "score_diff", "yardline_number", "season", "week", "x_los_snap"]
DLF = [f"dl_pc{i}" for i in range(8)]
TRACK = TRACK_ENV + ["c_cut_ret", "c_cut_reacc"] + DLF
SETS = {"ctx": CTX, "official": CTX + OFFICIAL, "tracking": CTX + TRACK, "all": CTX + OFFICIAL + TRACK}
TARGETS = {"vmax_5s": (None, "reg"), "burst": (None, "reg"), "dist_3s": (None, "reg"),
           "separation_at_pass_forward": (["WR", "SLOT_WR", "TE", "RB", "FB"], "reg"),
           "player_get_off": (["EDGE", "INTERIOR_LINE"], "reg"),
           "pressure_allowed": (["T", "G", "C"], "bin")}
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=14)


def to_pandas(df, cols):
    out = df.select(cols).with_columns(pl.col(pl.Boolean).cast(pl.Float64)).to_pandas()
    for c in cols:
        if c in ctx_cat:
            out[c] = out[c].astype("category")
    return out


rows, oof_parts = [], []
for tgt, (positions, kind) in TARGETS.items():
    d = t.filter(pl.col(tgt).is_not_null())
    if positions:
        d = d.filter(pl.col("lined_up_position").is_in(positions))
    if kind == "bin":
        d = d.with_columns(pl.col(tgt).cast(pl.Int8))
    preds = {s: np.full(d.height, np.nan) for s in SETS}
    for yr in (2023, 2024, 2025):
        tr_ids = pinfo.filter(pl.col("draft_year") != yr)["nfl_id"]
        pca = PCA(8, random_state=0).fit(proto[np.isin(u, tr_ids.to_numpy())])     # fit on training players only
        dl = pl.DataFrame({"nfl_id": u, **{f"dl_pc{i}": pca.transform(proto)[:, i] for i in range(8)}})
        dd = d.join(dl, on="nfl_id", how="left")
        te = (dd["draft_year"] == yr).to_numpy()
        y = dd[tgt].to_numpy().astype(float)
        for s, cols in SETS.items():
            Xp = to_pandas(dd, cols)
            obj = dict(objective="binary") if kind == "bin" else dict(objective="regression")
            m = lgb.train({**PARAMS, **obj}, lgb.Dataset(Xp[~te], y[~te], categorical_feature=[c for c in cols if c in ctx_cat]),
                          num_boost_round=600)
            preds[s][te] = m.predict(Xp[te])
    y = d[tgt].to_numpy().astype(float)
    pid = d["nfl_id"].to_numpy()
    resid_ctx = y - preds["ctx"]
    pe = pl.DataFrame({"nfl_id": pid, "r": resid_ctx}).group_by("nfl_id").agg(pl.col("r").mean(), pl.len()).filter(pl.col("len") >= 30)
    for s in SETS:
        if kind == "bin":
            score = roc_auc_score(y, preds[s])
        else:
            score = 1 - np.nanmean((y - preds[s]) ** 2) / np.nanvar(y)
        # player-effect R^2: how much of held-out players' mean ctx-residual is explained by this set's improvement
        imp = pl.DataFrame({"nfl_id": pid, "d": preds[s] - preds["ctx"]}).group_by("nfl_id").agg(pl.col("d").mean())
        j = pe.join(imp, on="nfl_id")
        pe_r2 = 1 - ((j["r"] - j["d"]) ** 2).sum() / ((j["r"] - j["r"].mean()) ** 2).sum()
        rows.append(dict(target=tgt, set=s, n_plays=d.height, n_players=j.height, score=score, player_effect_r2=pe_r2))
    oof_parts.append(d.select("game_id", "play_id", "nfl_id").with_columns(
        target=pl.lit(tgt), y=pl.Series(y), pred_ctx=pl.Series(preds["ctx"]), pred_all=pl.Series(preds["all"])))
    print(pl.DataFrame([r for r in rows if r["target"] == tgt]).with_columns(pl.selectors.float().round(4)))

pl.DataFrame(rows).write_parquet(OUT / "play_cv.parquet")
pl.concat(oof_parts).write_parquet(OUT / "play_oof.parquet")
