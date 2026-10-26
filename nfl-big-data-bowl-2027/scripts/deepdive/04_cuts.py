"""Step 4: change-of-direction ("cut") efficiency, detected identically in combine reps and game plays.

Cut event at frame i: |omega| is a local max (+-5 frames), entry speed v_in = max v over [i-10, i-3] >= 2.5 yd/s (combine cuts are mostly plant-and-go),
turn angle = |sum omega dt| over [i-5, i+5] >= 30 deg. Measures: retention = v_min / v_in (v_min over [i-5, i+5]),
re-acceleration = (v_out - v_min) with v_out = max v over [i+3, i+10].
Population curve per domain (retention ~ angle, v_in, interactions); player cut efficiency = mean residual,
empirical-Bayes shrunk. Output: data/deepdive/cuts.parquet (events), cut_players.parquet (one row per player).
"""
import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from common import OUT, players


def detect(lf: pl.LazyFrame, over: list[str]) -> pl.LazyFrame:
    w = lambda e: e.over(over)
    aom = pl.col("omega").abs()
    return (lf.with_columns(v_in=w(pl.col("v").rolling_max(8).shift(3)),
                            v_out=w(pl.col("v").rolling_max(8).shift(-10)),
                            v_min=w(pl.col("v").rolling_min(11, center=True)),
                            turn=w((pl.col("omega") * 0.1).rolling_sum(11, center=True)).abs().degrees(),
                            peak=w(aom.rolling_max(11, center=True)) == aom)
              .filter(pl.col("peak"), pl.col("v_in") >= 2.5, pl.col("turn") >= 30, pl.col("v_min").is_not_null(),
                      pl.col("v_out").is_not_null())
              .with_columns(retention=pl.col("v_min") / pl.col("v_in"), reaccel=pl.col("v_out") - pl.col("v_min")))


kc = pl.scan_parquet(OUT / "kin_combine.parquet").sort("event_id", "time")
cc = (detect(kc, ["event_id"]).with_columns(domain=pl.lit("combine"))
        .select("domain", "nfl_id", "drill_name", "event_id", "t_rep", "v_in", "v_min", "v_out", "turn", "retention", "reaccel")
        .collect())
kg = pl.scan_parquet(OUT / "kin_game.parquet").filter(pl.col("t_snap").is_between(-0.5, 9.5)).sort("game_id", "play_id", "nfl_id", "time")
cg = (detect(kg, ["game_id", "play_id", "nfl_id"]).filter(pl.col("t_snap") >= 0).with_columns(domain=pl.lit("game"))
        .select("domain", "nfl_id", "game_id", "play_id", "t_snap", "v_in", "v_min", "v_out", "turn", "retention", "reaccel")
        .collect())
print(f"cuts: combine {cc.height:,} (players {cc['nfl_id'].n_unique()}), game {cg.height:,} (players {cg['nfl_id'].n_unique()})")


def player_scores(ev: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Residual of retention / reaccel vs a population model of (turn, v_in), averaged per player with EB shrinkage."""
    X = ev.select("turn", "v_in").to_numpy()
    out = ev
    for tgt in ("retention", "reaccel"):
        y = ev[tgt].to_numpy()
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15).fit(X, y)
        out = out.with_columns(pl.Series(f"res_{tgt}", y - m.predict(X)))
    agg = out.group_by("nfl_id").agg(n=pl.len(), m_ret=pl.col("res_retention").mean(), m_re=pl.col("res_reaccel").mean(),
                                     turn_med=pl.col("turn").median())
    rows = {"nfl_id": agg["nfl_id"], f"{prefix}_cuts": agg["n"], f"{prefix}_turn_med": agg["turn_med"]}
    for col, name in (("m_ret", "cut_ret"), ("m_re", "cut_reacc")):
        tgt = "res_retention" if name == "cut_ret" else "res_reaccel"
        s2_within = out.group_by("nfl_id").agg(pl.col(tgt).var())[tgt].drop_nulls().median()
        tau2 = max(agg[col].var() - s2_within * (1 / agg["n"]).mean(), 1e-9)
        k = tau2 / (tau2 + s2_within / agg["n"])                      # EB shrinkage toward 0 (population mean)
        rows[f"{prefix}_{name}"] = agg[col] * k
    return pl.DataFrame(rows)


cuts = pl.concat([cc.select("domain", "nfl_id", "v_in", "v_min", "v_out", "turn", "retention", "reaccel"),
                  cg.select("domain", "nfl_id", "v_in", "v_min", "v_out", "turn", "retention", "reaccel")])
cuts.write_parquet(OUT / "cuts.parquet")
cg.select("nfl_id", "game_id", "play_id", "t_snap", "v_in", "v_min", "v_out", "turn", "retention", "reaccel").write_parquet(OUT / "cuts_game.parquet")
cp = (players().select("nfl_id", "combine_position", "draft_year")
        .join(player_scores(cc, "c"), on="nfl_id", how="left")
        .join(player_scores(cg, "g"), on="nfl_id", how="left"))
cp.write_parquet(OUT / "cut_players.parquet")

from scipy.stats import spearmanr  # noqa: E402
print("\ncut efficiency transfer (Spearman, players with >=8 combine cuts and >=100 game cuts):")
d = cp.filter(pl.col("c_cuts") >= 8, pl.col("g_cuts") >= 100)
for pos in ["DB", "DL", "OL", "TE", "WR"]:
    p = d.filter(pl.col("combine_position") == pos)
    if p.height < 15:
        continue
    r1 = spearmanr(p["c_cut_ret"], p["g_cut_ret"]).statistic
    r2 = spearmanr(p["c_cut_reacc"], p["g_cut_reacc"]).statistic
    print(f"  {pos}: n={p.height:3d}  retention r={r1:+.2f}  re-accel r={r2:+.2f}")
