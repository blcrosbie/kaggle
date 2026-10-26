"""Step 11: pre-draft only. Does combine tracking predict early-career success beyond draft slot + stopwatch?

Target: start_rate = career_games_started / (17 * seasons available)   (2023 class: 3, 2024: 2, 2025: 1)
        snap_rate  = (off + def snaps) / seasons available, z-scored within position
Feature sets: draft (log pick), +official, +tracking (envelope, sprint, cuts, DL combine prototype PCA-8), all.
Leave-one-draft-class-out ridge, within-position z-scored targets/features.
"""
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, POS_GROUPS, players

p = players().filter(pl.col("combine_position").is_in(POS_GROUPS))
env = pl.read_parquet(OUT / "envelopes.parquet")
cut = pl.read_parquet(OUT / "cut_players.parquet").select("nfl_id", "c_cut_ret", "c_cut_reacc")
E = np.load(OUT / "emb_oof.npy")
wm = pl.read_parquet(OUT / "windows_meta.parquet")
cm = (wm["domain"] == "combine").to_numpy()
ids_c = wm["nfl_id"].to_numpy()[cm]
u = np.unique(ids_c)
proto = np.stack([E[cm][ids_c == i].mean(0) for i in u])

OFFICIAL = ["combine_height", "combine_weight", "hand_size", "arm_length", "forty", "ten_yd_split", "vertical",
            "broad_jump", "three_cone", "short_shuttle", "bench_reps"]
TRACK = ["c_vmax", "c_brake", "c_lat_4_6", "c_lat_6_8", "c_acc_low", "c_omega_4_6", "c_as_A0", "c_as_V0", "sp_vmax",
         "sp_tau", "sp_A0", "c_cut_ret", "c_cut_reacc"]
d = (p.join(env.select(["nfl_id"] + [c for c in TRACK if c in env.columns]), on="nfl_id", how="left")
      .join(cut, on="nfl_id", how="left")
      .with_columns(seasons=2026 - pl.col("draft_year"), log_pick=pl.col("draft_overall_pick").log())
      .with_columns(start_rate=pl.col("career_games_started") / (17 * pl.col("seasons")),
                    snap_rate=(pl.col("career_offensive_snaps") + pl.col("career_defensive_snaps")) / pl.col("seasons")))


def z(df, cols):
    df = df.with_columns([((pl.col(c) - pl.col(c).mean().over("combine_position")) / pl.col(c).std().over("combine_position")).alias(c)
                          for c in cols])
    return df.with_columns([pl.col(c).is_null().cast(pl.Float64).alias(f"{c}_na") for c in cols]).with_columns(
        [pl.col(c).fill_null(0.0) for c in cols])


rows = []
for tgt in ("start_rate", "snap_rate"):
    dd = z(d, OFFICIAL + TRACK + ["log_pick", tgt])
    y = dd[tgt].to_numpy()
    for name in ("draft", "draft+official", "draft+tracking", "draft+all", "tracking_only"):
        pred = np.full(len(y), np.nan)
        for yr in (2023, 2024, 2025):
            te = (dd["draft_year"] == yr).to_numpy()
            pca = PCA(8, random_state=0).fit(proto[np.isin(u, dd.filter(pl.col("draft_year") != yr)["nfl_id"].to_numpy())])
            pcs = pl.DataFrame({"nfl_id": u, **{f"dl{i}": pca.transform(proto)[:, i] for i in range(8)}})
            ddd = dd.join(pcs, on="nfl_id", how="left").fill_null(0)
            cols = {"draft": ["log_pick"], "draft+official": ["log_pick"] + OFFICIAL + [f"{c}_na" for c in OFFICIAL],
                    "draft+tracking": ["log_pick"] + TRACK + [f"dl{i}" for i in range(8)],
                    "draft+all": ["log_pick"] + OFFICIAL + [f"{c}_na" for c in OFFICIAL] + TRACK + [f"dl{i}" for i in range(8)],
                    "tracking_only": TRACK + [f"dl{i}" for i in range(8)]}[name]
            X = ddd.select(cols).to_numpy()
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 40))).fit(X[~te], y[~te])
            pred[te] = m.predict(X[te])
        rows.append(dict(target=tgt, features=name, cv_r2=1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(),
                         n=len(y)))
res = pl.DataFrame(rows).pivot(on="features", index="target", values="cv_r2").with_columns(pl.selectors.float().round(3))
res.write_parquet(OUT / "player_outcomes_cv.parquet")
print(res)
