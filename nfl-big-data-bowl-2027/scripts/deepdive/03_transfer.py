"""Step 3: does combine movement capacity show up in games? And does tracking beat the stopwatch at predicting it?

(a) Spearman r, combine feature vs the same feature in games, within position group (bootstrap 95% CI)
(b) Leave-one-draft-class-out ridge: predict each game capacity from
    - official measurables (40, 10-yd split, 3-cone, shuttle, vertical, broad, height, weight)
    - combine-tracking envelope (A-S profile, sprint profile, braking, lateral, speed ceiling)
    - both
    Features/targets z-scored within position group, so R^2 is about individual differences, not OL-vs-WR.
"""
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, POS_GROUPS

rng = np.random.default_rng(0)
env = pl.read_parquet(OUT / "envelopes.parquet").filter(pl.col("g_plays") >= 50)

PAIRS = [("c_vmax", "g_vmax"), ("sp_vmax", "g_vmax"), ("c_as_A0", "g_as_A0"), ("sp_A0", "g_as_A0"),
         ("c_brake", "g_brake"), ("c_lat_4_6", "g_lat_4_6"), ("c_lat_6_8", "g_lat_6_8"), ("c_acc_low", "g_acc_low"),
         ("forty", "g_vmax"), ("ten_yd_split", "g_as_A0"), ("three_cone", "g_lat_4_6"), ("short_shuttle", "g_brake")]


def boot_r(x, y, n=1000):
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 15:
        return len(x), np.nan, np.nan, np.nan
    r = spearmanr(x, y).statistic
    bs = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return len(x), r, *np.nanpercentile(bs, [2.5, 97.5])


rows = []
for pos in POS_GROUPS + ["ALL"]:
    d = env if pos == "ALL" else env.filter(pl.col("combine_position") == pos)
    for a, b in PAIRS:
        n, r, lo, hi = boot_r(d[a].cast(pl.Float64).to_numpy(), d[b].cast(pl.Float64).to_numpy())
        rows.append(dict(pos=pos, combine=a, game=b, n=n, r=r, lo=lo, hi=hi))
tr = pl.DataFrame(rows)
tr.write_parquet(OUT / "transfer_pairs.parquet")
print(tr.filter(pl.col("pos") != "ALL").pivot(on="pos", index=["combine", "game"], values="r").with_columns(pl.selectors.float().round(2)))

# ---- (b) stopwatch vs tracking, within-position z-scores, leave-one-class-out
OFFICIAL = ["forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump", "combine_height", "combine_weight"]
TRACK = ["c_vmax", "c_as_A0", "c_as_V0", "sp_vmax", "sp_tau", "sp_A0", "c_brake", "c_lat_4_6", "c_lat_6_8", "c_acc_low", "c_omega_4_6"]
TARGETS = ["g_vmax", "g_as_A0", "g_as_V0", "g_brake", "g_lat_4_6", "g_lat_6_8", "g_acc_low"]


def zpos(df, cols):
    """z-score within position group; impute missing with 0 (= position mean) + missingness flags."""
    out = df.with_columns([((pl.col(c) - pl.col(c).mean().over("combine_position")) /
                            pl.col(c).std().over("combine_position")).alias(c) for c in cols])
    flags = [pl.col(c).is_null().cast(pl.Float64).alias(f"{c}_na") for c in cols if df[c].null_count() > 0]
    return out.with_columns(flags).with_columns([pl.col(c).fill_null(0.0) for c in cols])


d = env.filter(pl.col("combine_position").is_in(POS_GROUPS))
d = zpos(d, OFFICIAL + TRACK + TARGETS)
na_cols = [c for c in d.columns if c.endswith("_na")]
sets = {"official": OFFICIAL + [c for c in na_cols if c[:-3] in OFFICIAL],
        "tracking": TRACK + [c for c in na_cols if c[:-3] in TRACK],
        "both": OFFICIAL + TRACK + na_cols}
res = []
for tgt in TARGETS:
    y_all = d[tgt].to_numpy()
    for name, cols in sets.items():
        X = d.select(cols).to_numpy()
        pred = np.full(len(y_all), np.nan)
        for yr in (2023, 2024, 2025):
            te = (d["draft_year"] == yr).to_numpy()
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))).fit(X[~te], y_all[~te])
            pred[te] = m.predict(X[te])
        r2 = 1 - ((y_all - pred) ** 2).sum() / ((y_all - y_all.mean()) ** 2).sum()
        res.append(dict(target=tgt, features=name, cv_r2=r2))
cv = pl.DataFrame(res).pivot(on="features", index="target", values="cv_r2").with_columns(pl.selectors.float().round(3))
cv.write_parquet(OUT / "transfer_cv.parquet")
print("\nLeave-one-draft-class-out R^2 (within-position z-scores):")
print(cv)
