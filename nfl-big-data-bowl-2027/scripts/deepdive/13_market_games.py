"""Step 13: game-level betting lines x sensor behavior (2023-2025 REG/POST, nflverse closing lines).

G1  Calibration: de-vigged closing moneyline win probability vs outcome; upsets observed vs expected by bucket.
G2  "Collapse" check: do cohort players on heavy favorites that LOSE show effort deficits / out-of-character
    movement vs their own baseline? OLS of player-game effort index and fingerprint z on game outcome x pregame
    probability, SEs clustered by game; plus a permutation test that shuffles outcomes across games within
    pregame-probability buckets.
Output: data/deepdive/market_games.parquet, market_games_results.json
"""
import json

import numpy as np
import polars as pl
import statsmodels.formula.api as smf

from common import CACHE, MARKETS, OUT, players


def implied(ml):
    return pl.when(ml > 0).then(100 / (ml + 100)).otherwise(-ml / (-ml + 100))


lines = (pl.read_parquet(MARKETS / "nflverse_lines.parquet")
           .filter(pl.col("home_moneyline").is_not_null(), pl.col("away_moneyline").is_not_null())
           .with_columns(ph=implied(pl.col("home_moneyline")), pa=implied(pl.col("away_moneyline")))
           .with_columns(p_home=pl.col("ph") / (pl.col("ph") + pl.col("pa")), vig=pl.col("ph") + pl.col("pa") - 1,
                         home_won=(pl.col("home_score") > pl.col("away_score")).cast(pl.Int8),
                         tie=pl.col("home_score") == pl.col("away_score"))
           .filter(~pl.col("tie")))
fav = lines.with_columns(p_fav=pl.max_horizontal("p_home", 1 - pl.col("p_home")),
                         fav_won=pl.when(pl.col("p_home") >= 0.5).then(pl.col("home_won")).otherwise(1 - pl.col("home_won")))
cal = (fav.with_columns(bucket=(pl.col("p_fav") * 20).floor() / 20)
          .group_by("bucket").agg(n=pl.len(), expected=pl.col("p_fav").mean(), observed=pl.col("fav_won").mean())
          .sort("bucket"))
brier = ((fav["p_fav"] - fav["fav_won"]) ** 2).mean()
dog = fav.filter(pl.col("p_fav") >= 0.75)
res = dict(n_games=fav.height, brier=brier, mean_vig=fav["vig"].mean(),
           big_dog_games=dog.height, big_dog_wins=int((1 - dog["fav_won"]).sum()), big_dog_expected=float((1 - dog["p_fav"]).sum()))
print("calibration (favorite win prob bucket):")
print(cal.with_columns(pl.selectors.float().round(3)))
print(res)

# ---- G2: player-game behavior vs outcome x expectation
games = pl.read_parquet(CACHE / "games.parquet").select("game_id", "home_team_abbr")
pg = (pl.read_parquet(OUT / "player_games.parquet")
        .join(lines.select("game_id", "p_home", "home_won", "spread_line"), on="game_id", how="inner")
        .join(players().select("nfl_id", "combine_position", "draft_round"), on="nfl_id", how="left")
        .with_columns(is_home=pl.col("team_abbr") == pl.col("home_team_abbr"))
        .with_columns(p_team=pl.when(pl.col("is_home")).then(pl.col("p_home")).otherwise(1 - pl.col("p_home")),
                      won=pl.when(pl.col("is_home")).then(pl.col("home_won")).otherwise(1 - pl.col("home_won")))
        .with_columns(effort=pl.mean_horizontal("eff_vmax_z", "eff_burst_z", "eff_dist_z"),
                      fav_collapse=((pl.col("p_team") >= 0.70) & (pl.col("won") == 0)).cast(pl.Int8),
                      dog_upset=((pl.col("p_team") <= 0.30) & (pl.col("won") == 1)).cast(pl.Int8))
        .filter(pl.col("eff_plays") >= 10)
        .fill_nan(None).drop_nulls(["effort", "fp_z", "p_team"]))
pg.write_parquet(OUT / "market_games.parquet")
df = pg.select("effort", "fp_z", "won", "p_team", "fav_collapse", "dog_upset", "game_id", "combine_position").to_pandas()
models = {}
for y in ("effort", "fp_z"):
    m = smf.ols(f"{y} ~ won * p_team + fav_collapse + dog_upset + C(combine_position)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["game_id"]})
    models[y] = {k: dict(coef=float(m.params[k]), se=float(m.bse[k]), p=float(m.pvalues[k]))
                 for k in ("won", "p_team", "won:p_team", "fav_collapse", "dog_upset")}
    print(f"\n{y}: n={int(m.nobs)}")
    for k, v in models[y].items():
        print(f"   {k:14s} coef {v['coef']:+.3f}  se {v['se']:.3f}  p {v['p']:.3f}")

# permutation: within p_team deciles, shuffle game outcomes at the game level; statistic = mean effort in collapses
rng = np.random.default_rng(0)
gl = pg.group_by("game_id", "team_abbr").agg(pl.col("effort").mean(), pl.col("p_team").first(), pl.col("won").first(),
                                             pl.col("fp_z").mean())
gl = gl.with_columns(dec=(pl.col("p_team") * 10).floor())
obs = gl.filter((pl.col("p_team") >= 0.7) & (pl.col("won") == 0)).select(pl.col("effort").mean(), pl.col("fp_z").mean()).row(0)
null = []
for _ in range(2000):
    sh = gl.with_columns(won=pl.col("won").shuffle(seed=int(rng.integers(1e9))).over("dec"))
    null.append(sh.filter((pl.col("p_team") >= 0.7) & (pl.col("won") == 0)).select(pl.col("effort").mean(), pl.col("fp_z").mean()).row(0))
null = np.array(null)
perm = dict(collapse_team_games=int(gl.filter((pl.col("p_team") >= 0.7) & (pl.col("won") == 0)).height),
            effort_obs=obs[0], effort_p_low=float((null[:, 0] <= obs[0]).mean()),
            fp_obs=obs[1], fp_p_low=float((null[:, 1] <= obs[1]).mean()))
print("\npermutation (favorite collapses, team-game level):", perm)
json.dump(dict(calibration=cal.to_dicts(), summary=res, ols=models, permutation=perm), open(OUT / "market_games_results.json", "w"), indent=1, default=float)

# ---- G3: betting-attention proxy (primetime/standalone games carry the biggest handle) x fame (draft round <= 2)
pt = (pl.read_parquet(MARKETS / "nflverse_lines.parquet")
        .select("game_id", "weekday", "gametime")
        .with_columns(primetime=(pl.col("weekday").is_in(["Monday", "Thursday"]) |
                                 ((pl.col("weekday") == "Sunday") & (pl.col("gametime") >= "19:00"))).cast(pl.Int8)))
g3 = (pg.join(pt, on="game_id", how="left")
        .with_columns(famous=(pl.col("draft_round") <= 2).cast(pl.Int8)))
d3 = g3.select("effort", "fp_z", "primetime", "famous", "combine_position", "game_id").drop_nulls().to_pandas()
g3res = {}
for y in ("effort", "fp_z"):
    m = smf.ols(f"{y} ~ primetime * famous + C(combine_position)", data=d3).fit(cov_type="cluster", cov_kwds={"groups": d3["game_id"]})
    g3res[y] = {k: dict(coef=float(m.params[k]), se=float(m.bse[k]), p=float(m.pvalues[k])) for k in ("primetime", "famous", "primetime:famous")}
    print(f"\nG3 {y}: n={int(m.nobs)} primetime games share={d3['primetime'].mean():.2f}")
    for k, v in g3res[y].items():
        print(f"   {k:18s} coef {v['coef']:+.3f}  se {v['se']:.3f}  p {v['p']:.3f}")
out = json.load(open(OUT / "market_games_results.json"))
out["primetime_fame"] = g3res
json.dump(out, open(OUT / "market_games_results.json", "w"), indent=1, default=float)

# ---- G3b: is the primetime effect betting attention or short rest? split Thursday (short rest) from SNF/MNF
g3b = g3.with_columns(thu=(pl.col("weekday") == "Thursday").cast(pl.Int8),
                      snf_mnf=((pl.col("primetime") == 1) & (pl.col("weekday") != "Thursday")).cast(pl.Int8))
d3b = g3b.select("effort", "eff_vmax_z", "eff_burst_z", "eff_dist_z", "thu", "snf_mnf", "famous", "combine_position", "game_id").drop_nulls().to_pandas()
g3bres = {}
for y in ("effort", "eff_vmax_z", "eff_burst_z", "eff_dist_z"):
    m = smf.ols(f"{y} ~ thu + snf_mnf + famous + C(combine_position)", data=d3b).fit(cov_type="cluster", cov_kwds={"groups": d3b["game_id"]})
    g3bres[y] = {k: dict(coef=float(m.params[k]), se=float(m.bse[k]), p=float(m.pvalues[k])) for k in ("thu", "snf_mnf")}
    print(f"G3b {y:12s} Thursday {m.params['thu']:+.3f} (p {m.pvalues['thu']:.3f})   SNF/MNF {m.params['snf_mnf']:+.3f} (p {m.pvalues['snf_mnf']:.3f})")
print("player-games: Thursday", int(d3b["thu"].sum()), " SNF/MNF", int(d3b["snf_mnf"].sum()))
out = json.load(open(OUT / "market_games_results.json"))
out["thursday_vs_night"] = g3bres
json.dump(out, open(OUT / "market_games_results.json", "w"), indent=1, default=float)

# ---- G3c: robustness of the Thursday effect: control week + home/away + season type; drop week-1 & Thanksgiving Thursdays
lines_meta = pl.read_parquet(MARKETS / "nflverse_lines.parquet").select("game_id", "gameday")
g3c = (g3b.join(lines_meta, on="game_id", how="left")
          .with_columns(thanksgiving=(pl.col("gameday").dt.month() == 11) & (pl.col("gameday").dt.weekday() == 4)
                                     & (pl.col("gameday").dt.day() >= 22),
                        is_home_i=pl.col("is_home").cast(pl.Int8), post=(pl.col("season_type") == "POST").cast(pl.Int8)))
rob = {}
for label, frame in (("controls", g3c), ("drop_wk1_thanksgiving", g3c.filter(~((pl.col("thu") == 1) & ((pl.col("week") == 1) | pl.col("thanksgiving")))))):
    dfc = frame.select("eff_dist_z", "eff_burst_z", "effort", "thu", "snf_mnf", "week", "is_home_i", "post", "famous",
                       "combine_position", "game_id").drop_nulls().to_pandas()
    for y in ("effort", "eff_dist_z", "eff_burst_z"):
        m = smf.ols(f"{y} ~ thu + snf_mnf + week + is_home_i + post + famous + C(combine_position)", data=dfc).fit(
            cov_type="cluster", cov_kwds={"groups": dfc["game_id"]})
        rob[f"{label}:{y}"] = dict(thu=float(m.params["thu"]), thu_p=float(m.pvalues["thu"]), snf_mnf=float(m.params["snf_mnf"]),
                                  snf_mnf_p=float(m.pvalues["snf_mnf"]), week=float(m.params["week"]), week_p=float(m.pvalues["week"]),
                                  n_thu=int(dfc["thu"].sum()))
        r_ = rob[f"{label}:{y}"]
        print(f"G3c {label:22s} {y:11s} Thu {r_['thu']:+.3f} (p {r_['thu_p']:.3f}, n_thu {r_['n_thu']})  "
              f"SNF/MNF {r_['snf_mnf']:+.3f} (p {r_['snf_mnf_p']:.3f})  week {r_['week']:+.4f} (p {r_['week_p']:.3f})")
out = json.load(open(OUT / "market_games_results.json"))
out["thursday_robustness"] = rob
json.dump(out, open(OUT / "market_games_results.json", "w"), indent=1, default=float)

# ---- G3d: does temperature / wind explain the SNF/MNF dip? (domes/closed roofs -> 70F, 0 wind)
wx = (pl.read_parquet(MARKETS / "nflverse_lines.parquet").select("game_id", "temp", "wind", "roof")
        .with_columns(indoor=pl.col("roof").is_in(["dome", "closed"]))
        .with_columns(temp_f=pl.when(pl.col("indoor")).then(70).otherwise(pl.col("temp")).cast(pl.Float64),
                      wind_mph=pl.when(pl.col("indoor")).then(0).otherwise(pl.col("wind")).cast(pl.Float64)))
g3d = g3c.join(wx, on="game_id", how="left")
dfw = g3d.select("eff_dist_z", "eff_burst_z", "effort", "thu", "snf_mnf", "week", "is_home_i", "post", "famous", "temp_f",
                 "wind_mph", "combine_position", "game_id").drop_nulls().to_pandas()
wxres = {}
for y in ("effort", "eff_dist_z", "eff_burst_z"):
    m = smf.ols(f"{y} ~ thu + snf_mnf + temp_f + wind_mph + week + is_home_i + post + famous + C(combine_position)",
                data=dfw).fit(cov_type="cluster", cov_kwds={"groups": dfw["game_id"]})
    wxres[y] = {k: dict(coef=float(m.params[k]), p=float(m.pvalues[k])) for k in ("thu", "snf_mnf", "temp_f", "wind_mph")}
    print(f"G3d {y:11s} n={int(m.nobs)} " + "  ".join(f"{k} {v['coef']:+.4f} (p {v['p']:.3f})" for k, v in wxres[y].items()))
out = json.load(open(OUT / "market_games_results.json"))
out["weather"] = wxres
json.dump(out, open(OUT / "market_games_results.json", "w"), indent=1, default=float)
