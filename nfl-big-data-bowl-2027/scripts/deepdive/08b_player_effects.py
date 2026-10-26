"""Step 8b: two-stage version of the play-level transfer test (the single-stage LightGBM in 08 overfit: with ~340
training players, player-level columns act as an ID and memorize player quirks).

Stage 1 (from 08): context-only model, out-of-fold by draft class -> residual per play.
Stage 2: player effect = mean residual, empirical-Bayes shrunk; reliability = Spearman-Brown of odd/even-game halves.
         Ridge (one row per player, within-position z-scored features + NA flags) predicts the effect from
         pre-draft features; leave-one-draft-class-out, scored within held-out class x position.
Output: data/deepdive/player_effects.parquet, player_effects_cv.parquet
"""
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, players

pinfo = players()
oof = pl.read_parquet(OUT / "play_oof.parquet").with_columns(r=pl.col("y") - pl.col("pred_ctx"))
OFFICIAL = ["draft_overall_pick", "combine_height", "combine_weight", "hand_size", "arm_length", "forty", "ten_yd_split",
            "vertical", "broad_jump", "three_cone", "short_shuttle", "bench_reps"]
ENV = ["c_vmax", "c_brake", "c_lat_4_6", "c_lat_6_8", "c_acc_low", "c_omega_4_6", "c_as_A0", "c_as_V0", "sp_vmax", "sp_tau", "sp_A0"]
env = pl.read_parquet(OUT / "envelopes.parquet").select(["nfl_id"] + ENV)
cut = pl.read_parquet(OUT / "cut_players.parquet").select("nfl_id", "c_cut_ret", "c_cut_reacc")
E = np.load(OUT / "emb_oof.npy")
wm = pl.read_parquet(OUT / "windows_meta.parquet")
cm = (wm["domain"] == "combine").to_numpy()
ids_c = wm["nfl_id"].to_numpy()[cm]
u = np.unique(ids_c)
proto = np.stack([E[cm][ids_c == i].mean(0) for i in u])
TRACK = ENV + ["c_cut_ret", "c_cut_reacc"]

effects, rows = [], []
for tgt in oof["target"].unique(maintain_order=True):
    d = oof.filter(pl.col("target") == tgt)
    pe = (d.group_by("nfl_id").agg(m=pl.col("r").mean(), n=pl.len(), v=pl.col("r").var(),
                                   a=pl.col("r").filter(pl.col("game_id") % 2 == 0).mean(),
                                   b=pl.col("r").filter(pl.col("game_id") % 2 == 1).mean())
            .filter(pl.col("n") >= 30))
    s2 = pe["v"].median()
    tau2 = max(pe["m"].var() - (s2 / pe["n"]).mean(), 1e-12)
    pe = pe.with_columns(effect=pl.col("m") * tau2 / (tau2 + s2 / pl.col("n")), target=pl.lit(tgt))
    h = pe.drop_nulls(["a", "b"])
    r_half = spearmanr(h["a"], h["b"]).statistic
    rel = 2 * r_half / (1 + r_half)
    effects.append(pe.select("target", "nfl_id", "n", "m", "effect"))
    dd = (pe.join(pinfo.select(["nfl_id", "draft_year", "combine_position"] + OFFICIAL), on="nfl_id")
            .join(env, on="nfl_id", how="left").join(cut, on="nfl_id", how="left"))
    feats = OFFICIAL + TRACK
    dd = dd.with_columns([((pl.col(c) - pl.col(c).mean().over("combine_position")) / pl.col(c).std().over("combine_position")).fill_nan(None).alias(c)
                          for c in feats])
    dd = dd.with_columns([pl.col(c).is_null().cast(pl.Float64).alias(f"{c}_na") for c in feats]).with_columns(
        [pl.col(c).fill_null(0.0) for c in feats])
    y = dd["effect"].to_numpy()
    for name in ("official", "tracking", "tracking+dl", "all"):
        pred = np.full(len(y), np.nan)
        for yr in (2023, 2024, 2025):
            te = (dd["draft_year"] == yr).to_numpy()
            pca = PCA(8, random_state=0).fit(proto[np.isin(u, pinfo.filter(pl.col("draft_year") != yr)["nfl_id"].to_numpy())])
            pcs = pl.DataFrame({"nfl_id": u, **{f"dl{i}": pca.transform(proto)[:, i] for i in range(8)}})
            ddd = dd.join(pcs, on="nfl_id", how="left").fill_null(0)
            na = lambda cs: [f"{c}_na" for c in cs]
            cols = {"official": OFFICIAL + na(OFFICIAL), "tracking": TRACK + na(TRACK),
                    "tracking+dl": TRACK + na(TRACK) + [f"dl{i}" for i in range(8)],
                    "all": OFFICIAL + na(OFFICIAL) + TRACK + na(TRACK) + [f"dl{i}" for i in range(8)]}[name]
            X = ddd.select(cols).to_numpy()
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 40))).fit(X[~te], y[~te])
            pred[te] = m.predict(X[te])
        # score within held-out class x position (class is confounded with experience/season, so pooled
        # metrics would mostly measure class-level offsets, not who is better within a cohort)
        sc = (dd.select("draft_year", "combine_position").with_columns(y=pl.Series(y), p=pl.Series(pred))
                .with_columns(yc=pl.col("y") - pl.col("y").mean().over("draft_year", "combine_position"),
                              pc=pl.col("p") - pl.col("p").mean().over("draft_year", "combine_position")))
        rows.append(dict(target=tgt, features=name, n_players=len(y), effect_reliability=rel,
                         cv_r2=1 - ((sc["yc"] - sc["pc"]) ** 2).sum() / (sc["yc"] ** 2).sum(),
                         spearman=spearmanr(sc["yc"], sc["pc"]).statistic))
pl.concat(effects).write_parquet(OUT / "player_effects.parquet")
res = pl.DataFrame(rows)
res.write_parquet(OUT / "player_effects_cv.parquet")
pl.Config.set_tbl_rows(40)
print(res.with_columns(pl.selectors.float().round(3)))
