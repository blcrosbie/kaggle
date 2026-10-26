"""Step 12: split-half reliability of each capacity feature in each domain, and disattenuated transfer.

combine halves: alternate reps per player (by rep order); game halves: odd vs even game_id.
rel = Spearman-Brown(r_half) = 2r / (1 + r);  r_true ~= r_obs / sqrt(rel_c * rel_g)
Output: data/deepdive/reliability.parquet
"""
import importlib

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from common import OUT, POS_GROUPS, players

capacity = importlib.import_module("02_envelopes").capacity
pos = players().select("nfl_id", "combine_position")
FEATS = ["vmax", "brake", "lat_4_6", "lat_6_8", "acc_low", "omega_4_6"]

kc = (pl.scan_parquet(OUT / "kin_combine.parquet")
        .with_columns(half=pl.col("event_id").rank("dense").over("nfl_id") % 2))
kg = (pl.scan_parquet(OUT / "kin_game.parquet").filter(pl.col("t_snap").is_between(0, 8))
        .with_columns(half=pl.col("game_id") % 2))
nplays = pl.read_parquet(OUT / "envelopes.parquet").select("nfl_id", "g_plays")
ok = nplays.filter(pl.col("g_plays") >= 100)["nfl_id"].implode()

halves = {}
for dom, lf, q in (("c", kc, 1.0), ("g", kg.filter(pl.col("nfl_id").is_in(ok)), 0.995)):
    a = capacity(lf.filter(pl.col("half") == 0), q).rename(lambda c: c if c == "nfl_id" else f"{c}_a")
    b = capacity(lf.filter(pl.col("half") == 1), q).rename(lambda c: c if c == "nfl_id" else f"{c}_b")
    halves[dom] = a.join(b, on="nfl_id").join(pos, on="nfl_id")

env = pl.read_parquet(OUT / "envelopes.parquet").filter(pl.col("g_plays") >= 100)
tr = pl.read_parquet(OUT / "transfer_pairs.parquet")
rows = []
for p in POS_GROUPS:
    for f in FEATS:
        rel = {}
        for dom in ("c", "g"):
            h = halves[dom].filter(pl.col("combine_position") == p).drop_nulls([f"{f}_a", f"{f}_b"])
            r = spearmanr(h[f"{f}_a"], h[f"{f}_b"]).statistic if h.height >= 15 else np.nan
            rel[dom] = 2 * r / (1 + r) if r > 0 else np.nan
        e = env.filter(pl.col("combine_position") == p).drop_nulls([f"c_{f}", f"g_{f}"])
        r_obs = spearmanr(e[f"c_{f}"], e[f"g_{f}"]).statistic if e.height >= 15 else np.nan
        rows.append(dict(pos=p, feature=f, rel_combine=rel["c"], rel_game=rel["g"], r_obs=r_obs,
                         r_disatt=r_obs / np.sqrt(rel["c"] * rel["g"]) if rel["c"] and rel["g"] else np.nan, n=e.height))
res = pl.DataFrame(rows)
res.write_parquet(OUT / "reliability.parquet")
pl.Config.set_tbl_rows(40)
print(res.with_columns(pl.selectors.float().round(2)))
