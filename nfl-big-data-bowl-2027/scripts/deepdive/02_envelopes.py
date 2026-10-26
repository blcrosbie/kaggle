"""Step 2: movement capacity envelopes per player, measured the same way in both domains.

- A-S profile (Morin et al. in-situ method): top-2 a_lon per 0.25 yd/s speed bin (v >= 3), linear fit a = A0 - (A0/V0) v
- braking capacity, lateral (turning) capacity at speed, speed ceiling
- 40-yd sprint mechanical profile: v(t) = Vmax (1 - exp(-(t - t0)/tau)), A0_sprint = Vmax / tau
Output: data/deepdive/envelopes.parquet (one row per player; prefixes c_ = combine, g_ = game).
"""
import warnings

import numpy as np
import polars as pl
from scipy.optimize import curve_fit

from common import OUT, players

warnings.filterwarnings("ignore")
MIN_GAME_PLAYS = 50


def as_profile(lf: pl.LazyFrame) -> pl.DataFrame:
    top = (lf.filter(pl.col("v") >= 3, pl.col("a_lon").is_not_null())
             .with_columns(vbin=(pl.col("v") / 0.25).floor())
             .group_by("nfl_id", "vbin").agg(pl.col("a_lon").top_k(2), pl.col("v").top_k_by("a_lon", 2))
             .explode("a_lon", "v").collect())
    rows = []
    for (nid,), g in top.group_by("nfl_id"):
        v, a = g["v"].to_numpy(), g["a_lon"].to_numpy()
        keep = a > 0
        if keep.sum() < 8:
            continue
        v, a = v[keep], a[keep]
        for _ in range(2):                       # fit, drop |resid| > 2 sd, refit
            b, c = np.polyfit(v, a, 1)
            r = a - (b * v + c)
            ok = np.abs(r) <= 2 * r.std() + 1e-9
            v, a = v[ok], a[ok]
        b, c = np.polyfit(v, a, 1)
        if b >= 0:
            continue
        rows.append(dict(nfl_id=nid, A0=c, V0=-c / b, r2=1 - ((a - (b * v + c)) ** 2).sum() / ((a - a.mean()) ** 2).sum()))
    return pl.DataFrame(rows)


def capacity(lf: pl.LazyFrame, q: float) -> pl.DataFrame:
    """Quantile-based ceilings (q = 1.0 -> max)."""
    qq = (lambda c: c.max()) if q >= 1 else (lambda c: c.quantile(q))
    return (lf.group_by("nfl_id")
              .agg(vmax=qq(pl.col("v")),
                   brake=qq(-pl.col("a_lon").filter(pl.col("v").is_between(3, 9))),
                   lat_4_6=qq(pl.col("a_lat").abs().filter(pl.col("v").is_between(4, 6))),
                   lat_6_8=qq(pl.col("a_lat").abs().filter(pl.col("v").is_between(6, 8))),
                   acc_low=qq(pl.col("a_lon").filter(pl.col("v").is_between(1, 3))),
                   omega_4_6=qq(pl.col("omega").abs().filter(pl.col("v").is_between(4, 6))),
                   n_frames=pl.len())
              .collect())


def sprint_profile(kc: pl.LazyFrame) -> pl.DataFrame:
    f = (kc.filter(pl.col("drill_name") == "FORTY_YARD_DASH").select("event_id", "nfl_id", "t_rep", "s", "x_rep")
           .sort("event_id", "t_rep").collect())
    model = lambda t, vmax, tau, t0: np.where(t > t0, vmax * (1 - np.exp(-(t - t0) / tau)), 0.0)
    rows = []
    for (eid, nid), g in f.group_by("event_id", "nfl_id"):
        t, v, x = g["t_rep"].to_numpy(), g["s"].to_numpy(), g["x_rep"].to_numpy()
        if len(t) < 20 or v.max() < 6:
            continue
        stop = min(np.argmax(v) + 1, np.searchsorted(x, 40.0) + 1, len(t))   # rising part, up to 40 yd
        try:
            (vmax, tau, t0), _ = curve_fit(model, t[:stop], v[:stop], p0=[v.max(), 1.2, 0.3],
                                           bounds=([4, 0.3, -1], [13, 3.5, 3]), maxfev=5000)
        except RuntimeError:
            continue
        resid = v[:stop] - model(t[:stop], vmax, tau, t0)
        rows.append(dict(event_id=eid, nfl_id=nid, sp_vmax=vmax, sp_tau=tau, sp_A0=vmax / tau,
                         sp_rmse=float(np.sqrt((resid ** 2).mean()))))
    sp = pl.DataFrame(rows)
    return sp.sort("sp_rmse").group_by("nfl_id").agg(pl.all().sort_by("sp_vmax", descending=True).first()).drop("event_id")


if __name__ == "__main__":
    kc = pl.scan_parquet(OUT / "kin_combine.parquet")
    kg = (pl.scan_parquet(OUT / "kin_game.parquet").filter(pl.col("t_snap").is_between(0, 8)))
    nplays = kg.group_by("nfl_id").agg(g_plays=pl.struct("game_id", "play_id").n_unique()).collect()
    ok_ids = nplays.filter(pl.col("g_plays") >= MIN_GAME_PLAYS)["nfl_id"]
    kg = kg.filter(pl.col("nfl_id").is_in(ok_ids.implode()))

    c_cap = capacity(kc, 1.0).rename(lambda c: c if c == "nfl_id" else f"c_{c}")
    g_cap = capacity(kg, 0.995).rename(lambda c: c if c == "nfl_id" else f"g_{c}")
    c_as = as_profile(kc).rename(lambda c: c if c == "nfl_id" else f"c_as_{c}")
    g_as = as_profile(kg).rename(lambda c: c if c == "nfl_id" else f"g_as_{c}")
    sp = sprint_profile(kc)
    env = (players().select("nfl_id", "display_name", "draft_year", "combine_position", "nfl_position", "draft_overall_pick",
                            "forty", "ten_yd_split", "three_cone", "short_shuttle", "vertical", "broad_jump",
                            "combine_weight", "combine_height", "ngs_athleticism_score")
             .join(c_cap, on="nfl_id", how="left").join(c_as, on="nfl_id", how="left").join(sp, on="nfl_id", how="left")
             .join(nplays, on="nfl_id", how="left").join(g_cap, on="nfl_id", how="left").join(g_as, on="nfl_id", how="left"))
    env.write_parquet(OUT / "envelopes.parquet")
    print(env.select(pl.len(), pl.col("c_as_A0").is_not_null().sum().alias("c_as"), pl.col("sp_A0").is_not_null().sum().alias("sprint"),
                     pl.col("g_vmax").is_not_null().sum().alias("game"), pl.col("g_as_A0").is_not_null().sum().alias("g_as")))
    print(env.select("c_as_A0", "c_as_V0", "c_as_r2", "sp_vmax", "sp_tau", "sp_A0", "sp_rmse", "g_as_A0", "g_as_V0", "g_as_r2",
                     "c_vmax", "g_vmax", "c_brake", "g_brake", "c_lat_4_6", "g_lat_4_6").describe())
