"""Step 14: prediction-market layer (2025 season props; 2024-25 Polymarket moneylines) x sensor behavior.

P1  Prop calibration for cohort players (Kalshi): final-hour VWAP vs result, by fame tier (draft rounds 1-2 vs later)
    and by attention (player-game prop volume tercile). Logistic recalibration with player-game clustered SEs.
P2  Parlay exposure: number of sampled user-parlay legs on each prop (all 'yes' side) vs (result - price).
P3  Large late longshot trades on cohort props: taker buys at <= 25c, in the last 2 h vs earlier, big vs small.
P4  Polymarket wallets: pregame longshot positions (net long, avg buy price <= 0.30) on game-winner markets 2024-25.
    Per-wallet excess wins z-score; BH-FDR discoveries vs a null that re-simulates GAME outcomes from closing prices
    (keeps the correlation between wallets that bet the same upsets). Plus game-level: does late taker flow toward the
    underdog predict the upset beyond the closing price?
P5  Sensor link: do props that resolve on the unlikely side (priced >= 65% yes, settled no) coincide with effort /
    fingerprint anomalies, or with late money on that side, more than chance?
Outputs: data/deepdive/market_props_results.json, prop_markets.parquet, wallet_scores.parquet (local only)
"""
import json

import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy.stats import norm

from common import MARKETS, OUT, players

rng = np.random.default_rng(0)
R = {}
pinfo = players().select("nfl_id", "draft_round", "combine_position")
logit = lambda p: np.log(p / (1 - p))

# ---------------------------------------------------------------- P1 prop calibration
km = pl.read_parquet(MARKETS / "kalshi_markets.parquet")
props = km.filter(pl.col("nfl_id").is_not_null(), pl.col("market_kind") == "player_prop",
                  pl.col("series_ticker").is_in(["KXNFLRECYDS", "KXNFLREC", "KXNFLANYTD", "KXNFLFIRSTTD", "KXNFL2TD"]),
                  pl.col("result").is_in(["yes", "no"]))
kt = pl.read_parquet(MARKETS / "kalshi_trades.parquet").filter(pl.col("nfl_id").is_not_null())
kt = kt.join(props.select("ticker", "kickoff_utc", "result"), on="ticker", how="inner").filter(pl.col("created_time") < pl.col("kickoff_utc"))
vwap = (kt.filter(pl.col("created_time") >= pl.col("kickoff_utc") - pl.duration(hours=1))
          .group_by("ticker").agg(p_vwap=(pl.col("yes_price") * pl.col("count")).sum() / pl.col("count").sum(),
                                  vol_last_hour=pl.col("count").sum()))
pm = (props.join(vwap, on="ticker", how="left")
           .with_columns(p=pl.coalesce("p_vwap", "pregame_last_price").clip(0.01, 0.99), y=(pl.col("result") == "yes").cast(pl.Int8))
           .filter(pl.col("p").is_not_null(), pl.col("pregame_volume") >= 100)
           .join(pinfo, on="nfl_id", how="left")
           .with_columns(early=(pl.col("draft_round").fill_null(99) <= 2).cast(pl.Int8), pg=pl.concat_str(["nfl_id", "game_id"], separator="_")))
att = pm.group_by("pg").agg(att=pl.col("pregame_volume").sum())
pm = pm.join(att, on="pg").with_columns(att_tier=pl.col("att").rank("ordinal").truediv(pl.len()).mul(3).ceil().cast(pl.Int8))
pm = pm.with_columns(surprise=pl.col("y") - pl.col("p"))


def cluster_mean(df, col, by="pg", n=2000):
    g = df.group_by(by).agg(pl.col(col).sum().alias("s"), pl.len().alias("n"))
    s, c = g["s"].to_numpy(), g["n"].to_numpy()
    idx = rng.integers(0, len(g), (n, len(g)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


p1 = dict(n_markets=pm.height, n_player_games=pm["pg"].n_unique(), n_players=pm["nfl_id"].n_unique(),
          mean_price=float(pm["p"].mean()), hit_rate=float(pm["y"].mean()), surprise_all=cluster_mean(pm, "surprise"))
for name, f in (("early_round", pl.col("early") == 1), ("later_round", pl.col("early") == 0)):
    p1[f"surprise_{name}"] = cluster_mean(pm.filter(f), "surprise")
for t in (1, 2, 3):
    p1[f"surprise_attention_t{t}"] = cluster_mean(pm.filter(pl.col("att_tier") == t), "surprise")
for s in pm["series_ticker"].unique().sort():
    p1[f"surprise_{s}"] = cluster_mean(pm.filter(pl.col("series_ticker") == s), "surprise")
X = pm.select(lp=pl.col("p").map_elements(lambda v: float(logit(v)), return_dtype=pl.Float64), early=pl.col("early").cast(pl.Float64))
X = X.with_columns(lp_x_early=pl.col("lp") * pl.col("early")).to_pandas()
pd_groups = pm["pg"].cast(pl.Categorical).to_physical().to_numpy()
glm = sm.GLM(pm["y"].to_numpy(), sm.add_constant(X), family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd_groups})
p1["recal"] = {k: dict(coef=float(glm.params[k]), p=float(glm.pvalues[k])) for k in glm.params.index}
cal = (pm.with_columns(b=(pl.col("p") * 10).floor() / 10).group_by("b").agg(pl.len(), exp=pl.col("p").mean(), obs=pl.col("y").mean()).sort("b"))
p1["calibration"] = cal.to_dicts()
R["P1"] = p1
print("P1", {k: v for k, v in p1.items() if k not in ("calibration",)})

# ---------------------------------------------------------------- P2 parlay exposure
legs = pl.read_parquet(MARKETS / "kalshi_mve_legs.parquet").filter(pl.col("leg_nfl_id").is_not_null(), pl.col("leg_side") == "yes")
exp_ = legs.group_by("leg_market_ticker").agg(parlay_legs=pl.len()).rename({"leg_market_ticker": "ticker"})
pm = pm.join(exp_, on="ticker", how="left").with_columns(pl.col("parlay_legs").fill_null(0))
X2 = pm.select(lp=pl.col("p").map_elements(lambda v: float(logit(v)), return_dtype=pl.Float64),
               lpl=pl.col("parlay_legs").log1p(), early=pl.col("early").cast(pl.Float64)).to_pandas()
glm2 = sm.GLM(pm["y"].to_numpy(), sm.add_constant(X2), family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd_groups})
R["P2"] = dict(markets_with_parlay_legs=int((pm["parlay_legs"] > 0).sum()), median_legs=float(pm.filter(pl.col("parlay_legs") > 0)["parlay_legs"].median()),
               coef={k: dict(coef=float(glm2.params[k]), p=float(glm2.pvalues[k])) for k in glm2.params.index},
               surprise_top_decile_parlayed=cluster_mean(pm.filter(pl.col("parlay_legs") >= pm["parlay_legs"].quantile(0.9)), "surprise"),
               surprise_no_parlay=cluster_mean(pm.filter(pl.col("parlay_legs") == 0), "surprise"))
print("P2", R["P2"])
pm.write_parquet(OUT / "prop_markets.parquet")

# ---------------------------------------------------------------- P3 large late longshot trades on cohort props
tr = (kt.with_columns(paid=pl.when(pl.col("taker_side") == "yes").then(pl.col("yes_price")).otherwise(pl.col("no_price")),
                      won=(pl.col("taker_side") == pl.col("result")).cast(pl.Int8),
                      late=(pl.col("created_time") >= pl.col("kickoff_utc") - pl.duration(hours=2)))
        .with_columns(usd=pl.col("paid") * pl.col("count"))
        .filter(pl.col("paid") <= 0.25, pl.col("paid") > 0))
big_cut = float(tr["usd"].quantile(0.99))
tr = tr.with_columns(big=pl.col("usd") >= big_cut, edge=pl.col("won") - pl.col("paid"))
p3 = dict(longshot_trades=tr.height, big_cut_usd=big_cut)
for lab, f in (("big_late", pl.col("big") & pl.col("late")), ("big_early", pl.col("big") & ~pl.col("late")),
               ("small_late", ~pl.col("big") & pl.col("late")), ("small_early", ~pl.col("big") & ~pl.col("late"))):
    d = tr.filter(f)
    p3[lab] = dict(n=d.height, mean_paid=float(d["paid"].mean()), win_rate=float(d["won"].mean()),
                   edge=cluster_mean(d, "edge", by="ticker"))
R["P3"] = p3
print("P3", p3)

# ---------------------------------------------------------------- P4 Polymarket wallets + game-level late flow
pmk = pl.read_parquet(MARKETS / "polymarket_markets.parquet").filter(pl.col("market_kind") == "moneyline", pl.col("season") >= 2024,
                                                                   pl.col("season_type").is_in(["REG", "POST"]),
                                                                   pl.col("winning_outcome_index").is_not_null())
ptr = (pl.read_parquet(MARKETS / "polymarket_trades.parquet")
         .filter(pl.col("market_kind") == "moneyline", pl.col("pregame"))
         .join(pmk.select("condition_id", "kickoff_utc", "winning_outcome_index", "season"), on="condition_id", how="inner"))
# closing price per outcome: last pregame taker price for outcome 0 (outcome 1 = 1 - p0)
close = (ptr.filter(pl.col("is_taker")).sort("ts")
            .with_columns(p0=pl.when(pl.col("outcome_index") == 0).then(pl.col("price")).otherwise(1 - pl.col("price")))
            .group_by("condition_id").agg(p0_close=pl.col("p0").last()))
pos = (ptr.group_by("proxy_wallet", "condition_id", "outcome_index")
          .agg(buy_sh=pl.col("size").filter(pl.col("side") == "BUY").sum(), buy_usd=pl.col("usdc").filter(pl.col("side") == "BUY").sum(),
               sell_sh=pl.col("size").filter(pl.col("side") == "SELL").sum(), won=pl.col("outcome_won").first())
          .with_columns(net=pl.col("buy_sh") - pl.col("sell_sh"), avg_px=pl.col("buy_usd") / pl.col("buy_sh"))
          .filter(pl.col("net") > 0, pl.col("buy_sh") > 0))
# keep one-sided positions only (wallets that bought both sides of a game are hedging / market making)
two = pos.group_by("proxy_wallet", "condition_id").agg(k=pl.len()).filter(pl.col("k") > 1)
pos = pos.join(two, on=["proxy_wallet", "condition_id"], how="anti")
ls = pos.filter(pl.col("avg_px") <= 0.30, pl.col("avg_px") >= 0.02).join(close, on="condition_id").with_columns(
    p_close=pl.when(pl.col("outcome_index") == 0).then(pl.col("p0_close")).otherwise(1 - pl.col("p0_close")))
MIN_N = 10


def wallet_scores(df, won_col):
    w = (df.group_by("proxy_wallet").agg(n=pl.len(), wins=pl.col(won_col).cast(pl.Int32).sum(), e=pl.col("avg_px").sum(),
                                         v=(pl.col("avg_px") * (1 - pl.col("avg_px"))).sum(), usd=pl.col("buy_usd").sum())
           .filter(pl.col("n") >= MIN_N)
           .with_columns(z=(pl.col("wins") - pl.col("e") - 0.5) / pl.col("v").sqrt()))
    p = norm.sf(w["z"].to_numpy())
    order = np.argsort(p)
    m = len(p)
    q = np.empty(m)
    q[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return w.with_columns(p=pl.Series(p), q=pl.Series(np.minimum(q, 1)))


ws = wallet_scores(ls, "won")
obs_disc = {a: int((ws["q"] <= a).sum()) for a in (0.05, 0.10)}
# null: re-simulate each GAME's winner from the closing price (shared across wallets), recompute discoveries
games_c = close.join(pmk.select("condition_id"), on="condition_id")
null_disc, null_excess, null_z = [], [], []
for it in range(200):
    sim = games_c.with_columns(w0=pl.Series(rng.random(games_c.height)) < pl.col("p0_close"))
    d = ls.join(sim.select("condition_id", "w0"), on="condition_id").with_columns(
        won_sim=pl.when(pl.col("outcome_index") == 0).then(pl.col("w0")).otherwise(~pl.col("w0")))
    wsn = wallet_scores(d, "won_sim")
    null_disc.append(int((wsn["q"] <= 0.10).sum()))
    null_excess.append(float((wsn["wins"] - wsn["e"]).sum()))
    if it < 20:
        null_z.append(wsn["z"].to_numpy())
null_disc = np.array(null_disc)
top = ws.sort("p").head(20)
R["P4_wallets"] = dict(longshot_positions=ls.height, wallets_total=ls["proxy_wallet"].n_unique(), wallets_scored=ws.height,
                       min_positions=MIN_N, total_excess_wins=float((ws["wins"] - ws["e"]).sum()),
                       observed_discoveries=obs_disc, null_discoveries_q10_mean=float(null_disc.mean()),
                       null_discoveries_q10_p95=float(np.percentile(null_disc, 95)),
                       p_value_vs_null=float((null_disc >= obs_disc[0.10]).mean()),
                       top20_min_q=float(top["q"].min()), top20_median_n=float(top["n"].median()),
                       scored_positions=int(ws["n"].sum()), scored_expected_wins=float(ws["e"].sum()),
                       null_excess_mean=float(np.mean(null_excess)), null_excess_p95=float(np.percentile(null_excess, 95)),
                       excess_p_vs_null=float((np.array(null_excess) >= float((ws["wins"] - ws["e"]).sum())).mean()))
np.savez(OUT / "wallet_z.npz", obs=ws["z"].to_numpy(), null=np.concatenate(null_z))
ws.write_parquet(OUT / "wallet_scores.parquet")
print("P4 wallets", R["P4_wallets"])

# game-level: late taker flow toward the underdog (last 3 h) vs closing price
fl = (ptr.filter(pl.col("is_taker")).join(close, on="condition_id")
         .with_columns(dog_idx=pl.when(pl.col("p0_close") < 0.5).then(0).otherwise(1),
                       hrs=(pl.col("kickoff_utc") - pl.col("ts")).dt.total_seconds() / 3600)
         .with_columns(to_dog=((pl.col("outcome_index") == pl.col("dog_idx")) == (pl.col("side") == "BUY")))
         .with_columns(window=pl.when(pl.col("hrs") <= 3).then(pl.lit("late")).when(pl.col("hrs") <= 24).then(pl.lit("day")).otherwise(None))
         .filter(pl.col("window").is_not_null())
         .group_by("condition_id", "window").agg(dog_share=(pl.col("usdc") * pl.col("to_dog")).sum() / pl.col("usdc").sum(),
                                                 usd=pl.col("usdc").sum(),
                                                 whale_dog=(pl.col("usdc") * pl.col("to_dog") * (pl.col("usdc") >= 5000)).sum())
         .pivot(on="window", index="condition_id", values=["dog_share", "usd", "whale_dog"]))
gm = (fl.join(close, on="condition_id").join(pmk.select("condition_id", "winning_outcome_index", "season"), on="condition_id")
        .with_columns(p_dog=pl.min_horizontal("p0_close", 1 - pl.col("p0_close")),
                      dog_won=(pl.when(pl.col("p0_close") < 0.5).then(pl.col("winning_outcome_index") == 0)
                                 .otherwise(pl.col("winning_outcome_index") == 1)).cast(pl.Int8))
        .drop_nulls(["dog_share_late", "dog_share_day"]).filter(pl.col("p_dog").is_between(0.03, 0.5)))
Xg = gm.select(lp=pl.col("p_dog").map_elements(lambda v: float(logit(v)), return_dtype=pl.Float64),
               late_shift=pl.col("dog_share_late") - pl.col("dog_share_day"),
               whale=(pl.col("whale_dog_late") > 0).cast(pl.Float64)).to_pandas()
g4 = sm.GLM(gm["dog_won"].to_numpy(), sm.add_constant(Xg), family=sm.families.Binomial()).fit()
R["P4_games"] = dict(n_games=gm.height, dog_wins=int(gm["dog_won"].sum()), expected=float(gm["p_dog"].sum()),
                     coef={k: dict(coef=float(g4.params[k]), p=float(g4.pvalues[k])) for k in g4.params.index},
                     games_with_late_whale_dog=int((gm["whale_dog_late"] > 0).sum()),
                     whale_dog_win_rate=float(gm.filter(pl.col("whale_dog_late") > 0)["dog_won"].mean()),
                     whale_dog_mean_p=float(gm.filter(pl.col("whale_dog_late") > 0)["p_dog"].mean()))
print("P4 games", R["P4_games"])

# ---------------------------------------------------------------- P5 sensor link
pg = pl.read_parquet(OUT / "player_games.parquet").with_columns(effort=pl.mean_horizontal("eff_vmax_z", "eff_burst_z", "eff_dist_z"))
late_flow = (kt.filter(pl.col("created_time") >= pl.col("kickoff_utc") - pl.duration(hours=2))
               .group_by("ticker").agg(no_share_late=(pl.col("count") * (pl.col("taker_side") == "no")).sum() / pl.col("count").sum()))
early_flow = (kt.filter(pl.col("created_time") < pl.col("kickoff_utc") - pl.duration(hours=2))
                .group_by("ticker").agg(no_share_early=(pl.col("count") * (pl.col("taker_side") == "no")).sum() / pl.col("count").sum()))
over = (pm.filter(pl.col("series_ticker").is_in(["KXNFLRECYDS", "KXNFLREC", "KXNFLANYTD"]))
          .join(late_flow, on="ticker", how="left").join(early_flow, on="ticker", how="left")
          .with_columns(unlikely_under=((pl.col("p") >= 0.65) & (pl.col("y") == 0)),
                        no_surge=(pl.col("no_share_late") - pl.col("no_share_early")) >= 0.25))
pgx = (over.group_by("nfl_id", "game_id")
           .agg(unlikely_under=pl.col("unlikely_under").any(), no_surge=(pl.col("no_surge") & pl.col("unlikely_under")).any(),
                likely_markets=(pl.col("p") >= 0.65).sum(), surprise=pl.col("surprise").mean())
           .join(pg.select("nfl_id", "game_id", "effort", "fp_z", "snaps"), on=["nfl_id", "game_id"], how="inner")
           .fill_nan(None).drop_nulls(["effort", "fp_z"])
           .with_columns(anomaly=(pl.col("effort") <= -1) | (pl.col("fp_z") <= -1.5)))
base = pgx.filter(pl.col("likely_markets") > 0)
obs = base.filter(pl.col("unlikely_under"))["anomaly"].mean()
nul = []
an = base["anomaly"].to_numpy()
uu = base["unlikely_under"].to_numpy()
for _ in range(5000):
    nul.append(an[rng.permutation(uu)].mean())
nul = np.array(nul)
triple = base.filter(pl.col("unlikely_under"), pl.col("no_surge"), pl.col("anomaly")).height
R["P5"] = dict(player_games_with_likely_over=base.height, unlikely_under_games=int(uu.sum()),
               anomaly_rate_unlikely_under=float(obs), anomaly_rate_all=float(an.mean()),
               perm_p_higher=float((nul >= obs).mean()),
               corr_effort_surprise=float(np.corrcoef(pgx["effort"], pgx["surprise"])[0, 1]),
               corr_fp_surprise=float(np.corrcoef(pgx["fp_z"], pgx["surprise"])[0, 1]),
               unlikely_under_with_late_no_surge=int(base.filter(pl.col("unlikely_under"), pl.col("no_surge")).height),
               triple_coincidences=triple)
# expected triples under independence of anomaly from (under & surge)
us = base.filter(pl.col("unlikely_under"), pl.col("no_surge")).height
R["P5"]["triple_expected"] = float(us * an.mean())
print("P5", R["P5"])
json.dump(R, open(OUT / "market_props_results.json", "w"), indent=1, default=float)
