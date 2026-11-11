"""Monte Carlo visuals for pitch decks and screen capture (matplotlib, runs locally).

    python scripts/animate_simulation.py                          # DC2 power: GIF + final-frame PNG
    python scripts/animate_simulation.py --site DC5 --resource cooling
    python scripts/animate_simulation.py --show                   # live window (for screen recording)
    python scripts/animate_simulation.py --curves                 # static 6-site probability chart
    python scripts/animate_simulation.py --scenario high_growth --site DC3
    python scripts/animate_simulation.py --light                  # light theme (default: Campbell dark)

Outputs land in ./outputs/. Synthetic data - demonstration only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from dcsim import config as C  # noqa: E402
from dcsim import scenario as S  # noqa: E402
from dcsim import theme as T  # noqa: E402
from dcsim.data import load_planned_expansions, load_sites  # noqa: E402
from dcsim.forecast import MONTHS, run_forecast  # noqa: E402
from dcsim.montecarlo import MCSettings, run_monte_carlo  # noqa: E402

SCENARIOS = {
    "baseline": S.baseline,
    "high_growth": S.high_growth,
    "efficiency": S.efficiency_optimization,
    "no_planned": lambda: S.baseline(include_planned=False),
}
OUT_DIR = ROOT / "outputs"
SUFFIX = ""  # "_light" when rendering the light theme
FOOTER = "Synthetic data — demonstration only  ·  Six-Site Data Center Capacity Simulator"


def _fmt_month(m) -> str:
    return "not within 36 mo" if m is None or not np.isfinite(m) else f"month {int(m)}"


def animate_site(site_id: str, resource: str, scenario_key: str, draws: int, seed: int,
                 show: bool, fps: int) -> None:
    T.apply_matplotlib_theme()
    sites, planned = load_sites(), load_planned_expansions()
    sc = SCENARIOS[scenario_key]()
    settings = MCSettings(n_draws=draws, seed=seed)
    mc = run_monte_carlo(sites, planned, sc, settings)
    det = run_forecast(sites, planned, sc)

    i, j = mc.site_ids.index(site_id), C.RESOURCES.index(resource)
    unit = C.RESOURCE_UNITS[resource]
    name = mc.site_names[i]
    demand = mc.demand[:, i, j, :]         # (N, M)
    exhaust = mc.exhaust[:, i, j]          # (N,)
    det_demand, det_cap = det.demand[i, j], det.capacity[i, j]
    det_ex = det.site_resource.query("site_id == @site_id and resource == @resource")["exhaustion_month"].iloc[0]
    det_exhaust = None if pd.isna(det_ex) else int(det_ex)

    fig = plt.figure(figsize=(13.33, 7.5), dpi=100)  # 16:9
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 1], left=0.06, right=0.97, top=0.82, bottom=0.12, wspace=0.18)
    ax = fig.add_subplot(gs[0])
    right = gs[1].subgridspec(2, 1, height_ratios=[1.15, 2], hspace=0.35)
    ax_stat, axh = fig.add_subplot(right[0]), fig.add_subplot(right[1])
    ax_stat.axis("off")

    fig.text(0.06, 0.93, f"{name} ({site_id}) · {C.RESOURCE_LABELS[resource]}: {draws:,} simulated futures",
             fontsize=19, weight="semibold", color=T.TEXT_PRIMARY)
    fig.text(0.06, 0.885,
             f"Scenario: {sc.name}  ·  growth uncertainty ±{settings.growth_sd * 100:.0f} pts/yr  ·  expansion slip "
             f"0–{settings.slip_max:.0f} months  ·  red = demand above that future's capacity",
             fontsize=11.5, color=T.TEXT_SECONDARY)
    fig.text(0.06, 0.035, FOOTER, fontsize=9.5, color=T.TEXT_MUTED)
    label_box = dict(boxstyle="square,pad=0.25", facecolor=T.SURFACE, edgecolor="none", alpha=0.85)

    # --- left: demand paths vs capacity ---
    ymax = max(np.quantile(demand[:, -1], 0.98), det_cap.max()) * 1.12
    ax.set_xlim(0, C.MAX_HORIZON)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(0, C.MAX_HORIZON + 1, 6))
    ax.set_xlabel("Months from baseline (Oct 2026)")
    ax.set_ylabel(f"Demand ({unit})")

    site_exps = [e for e in det.expansions if e.site_id == site_id and e.resource == resource]
    for e in site_exps:
        ax.axvspan(e.activation_month, min(e.activation_month + settings.slip_max, C.MAX_HORIZON),
                   color=T.TEXT_SECONDARY, alpha=0.07, lw=0)
        ax.text(e.activation_month + 0.3, ymax * 0.97, f"+{e.amount:g} {unit} expansion\nslip window",
                fontsize=9, color=T.TEXT_MUTED, va="top")
    ax.step(MONTHS, det_cap, where="post", color=T.TEXT_SECONDARY, lw=2, zorder=5)
    ax.text(C.MAX_HORIZON - 0.3, det_cap[-1] + ymax * 0.015, "Capacity (as scheduled)",
            ha="right", va="bottom", fontsize=10, color=T.TEXT_SECONDARY, bbox=label_box, zorder=8)

    det_line, = ax.plot([], [], color=T.TEXT_PRIMARY, lw=2.2, zorder=6)
    det_label = ax.text(0, 0, "", fontsize=10, color=T.TEXT_PRIMARY, ha="left", va="center", bbox=label_box, zorder=8)
    counter = ax.text(0.985, 0.03, "", transform=ax.transAxes, fontsize=11, color=T.TEXT_SECONDARY, va="bottom", ha="right")

    # --- right: exhaustion-month histogram + headline stat ---
    bins = np.arange(1, C.MAX_HORIZON + 1)
    bars = axh.bar(bins, np.zeros_like(bins, dtype=float), width=0.75, color=T.STATUS["critical"], lw=0)
    final_counts = np.array([(exhaust == b).sum() for b in bins])
    axh.set_xlim(0, C.MAX_HORIZON + 1)
    axh.set_ylim(0, max(final_counts.max(), 1) / draws * 100 * 1.25)
    axh.set_xticks(range(0, C.MAX_HORIZON + 1, 12))
    axh.set_xlabel("Month capacity is exhausted")
    axh.set_ylabel("Share of simulations (%)")
    headline = ax_stat.text(0.0, 1.0, "", transform=ax_stat.transAxes, fontsize=34, weight="bold",
                            color=T.TEXT_PRIMARY, va="top")
    ax_stat.text(0.0, 0.58, "chance of exhaustion within 36 months", transform=ax_stat.transAxes,
                 fontsize=11, color=T.TEXT_SECONDARY, va="top")
    detail = ax_stat.text(0.0, 0.40, "", transform=ax_stat.transAxes, fontsize=10, color=T.TEXT_SECONDARY,
                          va="top", linespacing=1.55)

    # Frame schedule: few paths at first (so the eye follows them), then accelerate.
    counts = np.unique(np.round(np.geomspace(1, draws, 75)).astype(int))
    hold = int(fps * 3)
    frames = list(counts) + [draws] * hold
    drawn = {"n": 0}

    def path_segments(k0, k1):
        blue, red = [], []
        for k in range(k0, k1):
            pts = np.column_stack([MONTHS, demand[k]])
            e = exhaust[k]
            if np.isfinite(e):
                cut = int(e)
                blue.append(pts[: cut + 1] if cut > 0 else pts[:1])
                red.append(pts[cut - 1:] if cut > 0 else pts)
            else:
                blue.append(pts)
        return blue, red

    alpha = float(np.clip(18 / draws ** 0.75, 0.05, 0.5))

    def update(n):
        if n > drawn["n"]:
            blue, red = path_segments(drawn["n"], n)
            ax.add_collection(LineCollection(blue, colors=T.ACCENT, linewidths=1.0, alpha=alpha, zorder=3))
            if red:
                ax.add_collection(LineCollection(red, colors=T.STATUS["critical"], linewidths=1.0,
                                                 alpha=min(alpha * 1.8, 0.7), zorder=4))
            drawn["n"] = n
        x = exhaust[:n]
        for b, rect in zip(bins, bars):
            rect.set_height((x == b).sum() / draws * 100)
        p36 = np.mean(x <= 36)
        headline.set_text(f"{p36:.0%}")
        p10 = np.quantile(x, 0.10, method="inverted_cdf")
        p50 = np.quantile(x, 0.50, method="inverted_cdf")
        detail.set_text(
            f"Within 12 / 24 months:  {np.mean(x <= 12):.0%} / {np.mean(x <= 24):.0%}\n"
            f"Early case (P10):  {_fmt_month(p10)}\n"
            f"Median:  {_fmt_month(p50)}\n"
            f"Deterministic forecast:  {_fmt_month(det_exhaust)}"
        )
        counter.set_text(f"Simulations run: {n:,} / {draws:,}")
        if n >= draws:
            det_line.set_data(MONTHS, det_demand)
            det_label.set_position((24.3, det_demand[24] - ymax * 0.045))
            det_label.set_text("Deterministic forecast")
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False, repeat=False)
    OUT_DIR.mkdir(exist_ok=True)
    stem = f"mc_{site_id}_{resource}_{scenario_key}"
    if show:
        plt.show()
        return
    gif = OUT_DIR / f"{stem}{SUFFIX}.gif"
    anim.save(gif, writer=PillowWriter(fps=fps))
    update(draws)
    png = OUT_DIR / f"{stem}{SUFFIX}.png"
    fig.savefig(png, dpi=144)  # 1920x1080 for slides
    print(f"Saved {gif}\nSaved {png}")


def probability_curves(scenario_key: str, draws: int, seed: int, show: bool) -> None:
    T.apply_matplotlib_theme()
    sites, planned = load_sites(), load_planned_expansions()
    sc = SCENARIOS[scenario_key]()
    mc = run_monte_carlo(sites, planned, sc, MCSettings(n_draws=draws, seed=seed))
    curve = mc.exhaustion_curve()

    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=100)
    fig.subplots_adjust(left=0.07, right=0.80, top=0.80, bottom=0.12)
    fig.text(0.07, 0.93, "Probability each site has exhausted a resource, by month",
             fontsize=19, weight="semibold", color=T.TEXT_PRIMARY)
    fig.text(0.07, 0.885, f"Scenario: {sc.name}.  {draws:,} Monte Carlo simulations; any of power, cooling or rack space.",
             fontsize=11.5, color=T.TEXT_SECONDARY)
    fig.text(0.07, 0.035, FOOTER, fontsize=9.5, color=T.TEXT_MUTED)

    ends = []
    for sid, name in zip(mc.site_ids, mc.site_names):
        g = curve[curve.site_id == sid]
        ax.plot(g.month, g.probability * 100, color=T.SITE_COLORS[sid], lw=2.2, label=f"{sid} {name}")
        ends.append((g.probability.iloc[-1] * 100, sid, name))
    # Direct-label the line ends, nudging apart so labels never collide.
    ends.sort()
    placed = []
    for y, sid, name in ends:
        y_lab = max(y, (placed[-1] + 4.5) if placed else y)
        placed.append(y_lab)
        ax.text(C.MAX_HORIZON + 0.6, y_lab, f"{sid} {name}  {y:.0f}%", va="center", fontsize=10,
                color=T.TEXT_SECONDARY)
        ax.plot([C.MAX_HORIZON], [y], "o", ms=6, color=T.SITE_COLORS[sid], mec=T.SURFACE, mew=2, clip_on=False)
    ax.set_xlim(0, C.MAX_HORIZON)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(0, C.MAX_HORIZON + 1, 6))
    ax.set_xlabel("Months from baseline (Oct 2026)")
    ax.set_ylabel("Probability of exhaustion (%)")
    ax.legend(loc="center left", bbox_to_anchor=(0.30, 0.62), fontsize=10, ncols=2)

    OUT_DIR.mkdir(exist_ok=True)
    if show:
        plt.show()
        return
    png = OUT_DIR / f"mc_exhaustion_curves_{scenario_key}{SUFFIX}.png"
    fig.savefig(png, dpi=144)
    print(f"Saved {png}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--site", default="DC2", choices=[f"DC{i}" for i in range(1, 7)])
    ap.add_argument("--resource", default="power", choices=C.RESOURCES)
    ap.add_argument("--scenario", default="baseline", choices=SCENARIOS)
    ap.add_argument("--draws", type=int, default=MCSettings().n_draws,
                    help="simulations (default matches the app, so headline numbers agree)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--show", action="store_true", help="open a live window instead of saving files")
    ap.add_argument("--curves", action="store_true", help="static probability-of-exhaustion chart for all six sites")
    ap.add_argument("--light", action="store_true", help="light theme instead of Campbell dark")
    args = ap.parse_args(argv)
    global SUFFIX
    T.set_mode("light" if args.light else "dark")
    SUFFIX = "_light" if args.light else ""
    if not args.show:
        matplotlib.use("Agg")
    if args.curves:
        probability_curves(args.scenario, max(args.draws, 2000), args.seed, args.show)
    else:
        animate_site(args.site, args.resource, args.scenario, args.draws, args.seed, args.show, args.fps)


if __name__ == "__main__":
    main()
