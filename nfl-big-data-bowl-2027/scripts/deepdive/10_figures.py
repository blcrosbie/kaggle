"""Figures for DEEPDIVE.md (PNG, docs/deepdive/). Each figure is skipped if its inputs don't exist yet.

Palette: dataviz reference instance (light mode): categorical blue/orange/aqua; diverging blue<->red, gray midpoint.
"""
import re
import sys

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from common import FIG, MARKETS, OUT, POS_GROUPS, players  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DIVERGE = LinearSegmentedColormap.from_list("bdiv", ["#d03b3b", "#ec9a99", "#f0efec", "#86b6ef", "#1c5cab"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "legend.frameon": False, "lines.linewidth": 2})
ONLY = set(sys.argv[1:])


def want(name):
    return not ONLY or name in ONLY


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


# ---- F1: speed-acceleration envelopes, combine capacity vs in-game usage, per position group
if want("envelopes") and (OUT / "kin_game.parquet").exists():
    pos = players().select("nfl_id", "combine_position")
    bins = (pl.col("v") / 0.5).floor() * 0.5 + 0.25

    def env(lf):
        return (lf.filter(pl.col("v").is_between(0.5, 11), pl.col("a_lon").is_not_null())
                  .join(pos.lazy(), on="nfl_id").with_columns(vb=bins)
                  .group_by("combine_position", "vb")
                  .agg(hi=pl.col("a_lon").quantile(0.99), lo=pl.col("a_lon").quantile(0.01), n=pl.len())
                  .filter(pl.col("n") >= 200).sort("vb").collect())

    ec = env(pl.scan_parquet(OUT / "kin_combine.parquet"))
    eg = env(pl.scan_parquet(OUT / "kin_game.parquet").filter(pl.col("t_snap").is_between(0, 8)))
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), sharey=True)
    for ax, p in zip(axes, POS_GROUPS):
        for df, col, lab in ((ec, BLUE, "Combine (capacity)"), (eg, ORANGE, "Games (usage)")):
            d = df.filter(pl.col("combine_position") == p)
            ax.plot(d["vb"], d["hi"], color=col, label=lab)
            ax.plot(d["vb"], d["lo"], color=col)
            ax.fill_between(d["vb"], d["lo"], d["hi"], color=col, alpha=0.10, linewidth=0)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(p)
        ax.set_xlabel("speed (yd/s)")
    axes[0].set_ylabel("acceleration along path (yd/s²)\n99th / 1st percentile")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.06))
    fig.suptitle("Speed–acceleration envelope: what players can do at the combine vs what they use on Sundays",
                 x=0.01, y=1.06, ha="left", fontweight="bold")
    save(fig, "f1_envelopes")

# ---- F2: transfer heatmap (combine feature vs same feature in games), sign-flipped for timed drills
if want("transfer") and (OUT / "transfer_pairs.parquet").exists():
    tr = pl.read_parquet(OUT / "transfer_pairs.parquet").filter(pl.col("pos").is_in(POS_GROUPS))
    timed = ["forty", "ten_yd_split", "three_cone", "short_shuttle"]
    tr = tr.with_columns(r=pl.when(pl.col("combine").is_in(timed)).then(-pl.col("r")).otherwise(pl.col("r")),
                         label=pl.concat_str([pl.col("combine"), pl.lit(" → "), pl.col("game")]))
    labels = tr["label"].unique(maintain_order=True).to_list()
    M = np.array([[tr.filter(pl.col("label") == lab, pl.col("pos") == p)["r"].item() for p in POS_GROUPS] for lab in labels])
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.imshow(M, cmap=DIVERGE, vmin=-0.75, vmax=0.75, aspect="auto")
    ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "n<15" if np.isnan(M[i, j]) else f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=9,
                    color=INK2 if np.isnan(M[i, j]) else INK)
    ax.set_xticks(range(len(POS_GROUPS)), POS_GROUPS)
    pretty = lambda x: re.sub(r"^c_", "combine ", re.sub(r"^g_", "game ", re.sub(r"^sp_", "sprint-fit ", x)))
    ax.set_yticks(range(len(labels)), [" → ".join(pretty(part) for part in lab.split(" → ")) for lab in labels])
    ax.axhline(7.5, color=INK, linewidth=1.2)
    ax.text(5.6, 3.5, "combine\ntracking", fontsize=9, color=INK2, va="center")
    ax.text(5.6, 9.5, "official\nstopwatch\n(sign flipped:\nfaster = +)", fontsize=9, color=INK2, va="center")
    ax.set_title("Does it transfer? Spearman r between combine measure and in-game measure,\nwithin position group",
                 loc="left")
    save(fig, "f2_transfer")

# ---- F3: lineup test
if want("lineup") and (OUT / "lineup_variants.parquet").exists():
    lv = pl.read_parquet(OUT / "lineup_variants.parquet")
    order = [("split_half_game", "Game vs game (split-half)", ORANGE), ("split_half_combine", "Combine vs combine (split-half)", ORANGE),
             ("envelope", "Combine → game: capacity envelope", BLUE), ("fused", "Combine → game: envelope + deep encoder", BLUE),
             ("dl_raw", "Combine → game: deep encoder", BLUE), ("dl_centered", "Combine → game: deep encoder, context-centered", BLUE)]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for k, (v, lab, col) in enumerate(order):
        d = lv.filter(pl.col("variant") == v)
        ax.barh(k, d["within_pct"].mean() - 0.5, left=0.5, color=col, height=0.6)
        ax.scatter(d["within_pct"], [k] * d.height, color=INK, s=14, zorder=3)
        ax.text(max(d["within_pct"].max(), d["within_pct"].mean()) + 0.012, k, f"{d['within_pct'].mean():.3f}", va="center",
                fontsize=9, color=INK)
    ax.set_yticks(range(len(order)), [o[1] for o in order])
    ax.invert_yaxis()
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle="--")
    ax.text(0.502, len(order) - 0.45, "chance", fontsize=8, color=INK2)
    ax.set_xlim(0.45, 0.92)
    ax.set_xlabel("mean percentile rank of the true player among his position group (1 = always first, 0.5 = chance)")
    ax.set_title("The lineup test on unseen draft classes (bars = mean of 3 held-out classes, dots = each class)", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="reliability ceiling (same domain)"), Patch(color=BLUE, label="cross-domain match")],
              loc="lower right")
    save(fig, "f3_lineup")

# ---- F4: play-level conditional transfer (two-stage: context model -> shrunk player effects -> pre-draft ridge)
if want("play") and (OUT / "player_effects_cv.parquet").exists():
    cv = pl.read_parquet(OUT / "player_effects_cv.parquet")
    tg = cv["target"].unique(maintain_order=True).to_list()
    sets = [("official", "official measurables + draft pick", YELLOW), ("tracking", "combine tracking (envelope, sprint, cuts)", BLUE),
            ("tracking+dl", "combine tracking + deep encoder", AQUA)]
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.26
    for k, (s_, lab, col) in enumerate(sets):
        vals = [cv.filter(pl.col("target") == t_, pl.col("features") == s_)["spearman"].item() for t_ in tg]
        ax.bar(np.arange(len(tg)) + (k - 1) * w, vals, width=w - 0.02, color=col, label=lab)
    rel = [cv.filter(pl.col("target") == t_)["effect_reliability"][0] for t_ in tg]
    ax.scatter(range(len(tg)), np.sqrt(rel), marker="_", s=900, color=INK, linewidths=2, label="ceiling: √reliability of player effect", zorder=3)
    ax.axhline(0, color=INK2, linewidth=0.8)
    names = {"vmax_5s": "top speed\n(0–5 s)", "burst": "burst\n(0–1.5 s)", "dist_3s": "distance\n(0–3 s)",
             "separation_at_pass_forward": "separation\n(WR/TE)", "player_get_off": "get-off\n(DL)",
             "pressure_allowed": "pressure allowed\n(OL)"}
    ax.set_xticks(range(len(tg)), [names.get(t_, t_) for t_ in tg])
    ax.set_ylabel("Spearman r, predicted vs actual player effect\n(held-out class, within position)")
    ax.set_ylim(-0.3, 1.0)
    ax.set_title("Pre-draft data vs the stable part of each player's in-game behavior (after play context)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, fontsize=9)
    save(fig, "f4_player_effects")

# ---- F5: closing-line calibration + Thursday / night-game effort effects
if want("lines") and (OUT / "market_games_results.json").exists():
    import json
    r = json.load(open(OUT / "market_games_results.json"))
    cal = pl.DataFrame(r["calibration"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})
    a1.plot([0.5, 1], [0.5, 1], color=INK2, linewidth=1, linestyle="--")
    a1.scatter(cal["expected"], cal["observed"], s=np.maximum(cal["n"].to_numpy() * 1.5, 50), color=BLUE, edgecolor=SURF, linewidth=2, zorder=3)
    for row in cal.iter_rows(named=True):
        a1.text(row["expected"] + 0.012, row["observed"] - 0.012, f"n={row['n']}", fontsize=8, color=INK2)
    a1.set_xlabel("closing-line favorite win probability (de-vigged)")
    a1.set_ylabel("favorite actually won")
    a1.set_title("Closing lines 2023–25 are calibrated\n(854 games; big dogs won 38 vs 33.8 expected)", loc="left")
    tv = r["thursday_vs_night"]
    comps = [("effort", "effort index"), ("eff_dist_z", "distance 0–3 s"), ("eff_burst_z", "burst 0–1.5 s"), ("eff_vmax_z", "top speed")]
    for k, (slot, lab, col) in enumerate((("thu", "Thursday (short rest)", ORANGE), ("snf_mnf", "Sunday/Monday night", BLUE))):
        y = np.arange(len(comps)) + (0.18 if k else -0.18)
        c = np.array([tv[c_][slot]["coef"] for c_, _ in comps])
        se = np.array([tv[c_][slot]["se"] for c_, _ in comps])
        a2.errorbar(c, y, xerr=1.96 * se, fmt="o", color=col, markersize=8, capsize=0, linewidth=2, label=lab)
    a2.axvline(0, color=INK2, linewidth=1)
    a2.set_yticks(range(len(comps)), [lab for _, lab in comps])
    a2.invert_yaxis()
    a2.set_xlabel("difference vs other games (within-player SD, 95% CI)")
    a2.set_title("Rookies are less explosive in primetime — mostly on short rest", loc="left")
    a2.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    save(fig, "f5_lines_primetime")

# ---- F6: prop calibration by fame + wallet longshot z-scores vs game-level null
if want("markets") and (OUT / "prop_markets.parquet").exists() and (OUT / "wallet_z.npz").exists():
    pm = pl.read_parquet(OUT / "prop_markets.parquet")
    wz = np.load(OUT / "wallet_z.npz")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot([0, 0.9], [0, 0.9], color=INK2, linewidth=1, linestyle="--")
    for k, (lab, f, col) in enumerate((("drafted rounds 1–2", pl.col("early") == 1, BLUE), ("rounds 3+ / undrafted", pl.col("early") == 0, ORANGE))):
        c = (pm.filter(f).with_columns(b=(pl.col("p") * 10).floor() / 10).group_by("b")
               .agg(n=pl.len(), e=pl.col("p").mean(), o=pl.col("y").mean()).filter(pl.col("n") >= 20).sort("b"))
        a1.plot(c["e"], c["o"], color=col, marker="o", markersize=8, markeredgecolor=SURF, markeredgewidth=2,
                label=f"{lab} ({pm.filter(f).height:,} props)")
    a1.set_xlabel("final-hour Kalshi price (implied probability)")
    a1.set_ylabel("share that hit")
    a1.set_title("Cohort player props (2025): priced as accurately for\nlesser-known players as for early-round picks", loc="left")
    a1.legend(loc="upper left")
    obs, null = np.sort(wz["obs"]), np.sort(wz["null"])
    qs = (np.arange(len(obs)) + 0.5) / len(obs)
    a2.plot([-4, 4], [-4, 4], color=INK2, linewidth=1, linestyle="--")
    a2.scatter(np.quantile(null, qs), obs, s=16, color=BLUE, zorder=3)
    a2.set_xlim(-3.5, 4)
    a2.set_ylim(-3.5, 4)
    a2.set_xlabel("expected z under null (game outcomes re-simulated from closing prices)")
    a2.set_ylabel("observed wallet z (excess longshot wins)")
    a2.set_title(f"{len(obs)} Polymarket wallets with ≥ 10 longshot bets:\nno tail of implausibly lucky wallets", loc="left")
    save(fig, "f6_markets")
