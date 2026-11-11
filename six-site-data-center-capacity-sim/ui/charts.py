"""Plotly figures: boxed plot area, monospace ticks, single y-axis, thin marks,
fixed site colours plus dash patterns, hover on every chart.

Rendered with Streamlit's Plotly theme (theme="streamlit"), which supplies the
background and text colours of whichever Light / Dark theme is active. Neutrals
here are semi-transparent grays so they read on navy and on white alike."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from dcsim import config as C
from dcsim import theme as T
from dcsim.forecast import ForecastResult, period_label

FONT = ", ".join(T.MONO_FAMILY)
GRIDC = "rgba(138,138,138,0.28)"   # gridlines on either background
LINEC = "rgba(138,138,138,0.75)"   # plot frame and ticks
CHART_KW = dict(theme="streamlit", config={"displaylogo": False})


def _layout(fig: go.Figure, title: str | None = None, height: int = 380, ytitle: str = "",
            yformat: str | None = None) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15), x=0, xanchor="left") if title else None,
        height=height,
        margin=dict(l=8, r=8, t=48 if title else 16, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font=dict(family=FONT)),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRIDC, linecolor=LINEC, showline=True, mirror=True, ticks="outside",
                     tickcolor=LINEC, zeroline=False,
                     showspikes=True, spikecolor=LINEC, spikethickness=1, spikedash="solid",
                     automargin=True)
    fig.update_yaxes(gridcolor=GRIDC, linecolor=LINEC, showline=True, mirror=True, ticks="outside",
                     tickcolor=LINEC, zeroline=False,
                     title=dict(text=ytitle, font=dict(size=12), standoff=8), tickformat=yformat,
                     rangemode="tozero", automargin=True)
    return fig


def _legend_below(fig: go.Figure, entrywidth: int = 150) -> None:
    # Fixed entry widths: Plotly measures text before the web font loads, which overlaps entries.
    fig.update_layout(legend=dict(orientation="h", y=-0.12, yanchor="top", x=0, xanchor="left",
                                  entrywidthmode="pixels", entrywidth=entrywidth))


def _threshold_key(fig: go.Figure, name: str, color: str) -> None:
    """Legend entry for a threshold line (reference lines carry no legend of their own)."""
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=name, hoverinfo="skip",
                             line=dict(color=color, width=1.2, dash="dot")))


def _dates(periods) -> pd.Series:
    return pd.to_datetime(pd.Series(periods), format="%Y-%m")


def portfolio_forecast(result: ForecastResult, resource: str = "power") -> go.Figure:
    """Portfolio demand vs capacity, with the earliest site-level exhaustion marked."""
    sc = result.scenario
    pm = result.portfolio_monthly[result.portfolio_monthly["resource"] == resource]
    x = _dates(pm["period"])
    unit = C.RESOURCE_UNITS[resource]
    fmt = ",.0f" if resource == "space" else ",.1f"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pm["capacity"], name="Capacity", mode="lines", line_shape="hv",
                             line=dict(color=T.NEUTRAL, width=2),
                             hovertemplate=f"%{{y:{fmt}}} {unit}<extra>Capacity</extra>"))
    fig.add_trace(go.Scatter(x=x, y=pm["capacity"] * sc.warning_threshold, mode="lines", line_shape="hv",
                             name=f"{sc.warning_threshold:.0%} warning",
                             line=dict(color=T.STATUS["warning"], width=1.2, dash="dot"),
                             hovertemplate=f"%{{y:{fmt}}} {unit}<extra>Warning threshold</extra>"))
    fig.add_trace(go.Scatter(x=x, y=pm["demand"], name="Projected demand", mode="lines",
                             line=dict(color=T.ACCENT, width=2, dash="dash"),
                             hovertemplate=f"%{{y:{fmt}}} {unit}<extra>Demand</extra>"))
    fig.add_trace(go.Scatter(x=x[:1], y=pm["demand"].iloc[:1], name="Actual (Oct 2026)", mode="markers",
                             marker=dict(size=10, symbol="square", color=T.ACCENT),
                             hovertemplate=f"%{{y:{fmt}}} {unit}<extra>Actual</extra>"))

    ec = result.kpis["earliest_constraint"]
    first = result.site_resource[result.site_resource["resource"] == resource].dropna(subset=["exhaustion_month"])
    if len(first):
        r = first.sort_values("exhaustion_month").iloc[0]
        m = int(r["exhaustion_month"])
        if m <= sc.horizon:
            xd = pd.to_datetime(period_label(m, "%Y-%m"), format="%Y-%m")
            fig.add_vline(x=xd, line=dict(color=T.STATUS["critical"], width=1.5, dash="dot"))
            fig.add_annotation(x=xd, y=1, yref="paper", yanchor="bottom", showarrow=False, xanchor="left",
                               text=f"■ {r['site_id']} exhausts {C.RESOURCE_LABELS[resource].lower()} ({period_label(m)})",
                               font=dict(size=11), bordercolor=T.STATUS["critical"], borderwidth=1, xshift=4)
    _layout(fig, None, 400, f"{C.RESOURCE_LABELS[resource]} ({unit})")
    _legend_below(fig, 135)
    fig.update_layout(margin=dict(t=28))
    fig.update_xaxes(hoverformat="%b %Y")
    return fig


def site_utilization(result: ForecastResult, resource: str = "power") -> go.Figure:
    """Utilization by site: now (bar) vs end of horizon (marker), with threshold lines."""
    sc, h = result.scenario, result.horizon
    sr = result.site_resource[result.site_resource["resource"] == resource].iloc[::-1]  # DC1 on top
    labels = sr["site_id"] + " " + sr["site_name"]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=sr["current_utilization"], orientation="h", name="Now (actual)",
                         marker=dict(color=T.ACCENT, line=dict(width=0)), width=0.5,
                         hovertemplate="%{x:.0%}<extra>Now</extra>"))
    fig.add_trace(go.Scatter(y=labels, x=sr["horizon_utilization"], mode="markers", name=f"M{h} (proj.)",
                             marker=dict(symbol="diamond", size=12, color=T.NEUTRAL),
                             hovertemplate="%{x:.0%}<extra>Month " + str(h) + "</extra>"))
    for thr, color, name in ((sc.warning_threshold, T.STATUS["warning"], "Warning"),
                             (sc.critical_threshold, T.STATUS["serious"], "Critical"),
                             (1.0, T.STATUS["critical"], "Capacity")):
        fig.add_vline(x=thr, line=dict(color=color, width=1.2, dash="dot"))
        _threshold_key(fig, f"{name} {thr:.0%}", color)
    xmax = max(1.1, float(sr["horizon_utilization"].max()) * 1.08)
    _layout(fig, None, 400)
    _legend_below(fig, 150)
    fig.update_layout(hovermode="y unified", margin=dict(t=16), bargap=0.4)
    fig.update_xaxes(tickformat=".0%", range=[0, xmax], showspikes=False)
    fig.update_yaxes(showgrid=False, rangemode=None)
    return fig


def site_utilization_forecast(result: ForecastResult, site_ids: list[str], resource: str = "power") -> go.Figure:
    """Utilization over time for the selected sites (common % scale), with thresholds."""
    sc = result.scenario
    m = result.monthly[(result.monthly["resource"] == resource) & (result.monthly["site_id"].isin(site_ids))]
    fig = go.Figure()
    for sid in site_ids:
        g = m[m["site_id"] == sid]
        fig.add_trace(go.Scatter(
            x=_dates(g["period"]), y=g["utilization"], mode="lines", name=f"{sid} {g['site_name'].iloc[0]}",
            line=dict(color=T.SITE_COLORS[sid], width=2, dash=T.SITE_DASH[sid]),
            hovertemplate="%{y:.0%}<extra>" + sid + "</extra>",
        ))
    for thr, color, name in ((sc.warning_threshold, T.STATUS["warning"], "Warning"),
                             (sc.critical_threshold, T.STATUS["serious"], "Critical"),
                             (1.0, T.STATUS["critical"], "Capacity (100%)")):
        fig.add_hline(y=thr, line=dict(color=color, width=1.2, dash="dot"),
                      annotation=dict(text=f"{name} {thr:.0%}" if thr < 1 else name, font=dict(size=10),
                                      xanchor="left", x=0, yanchor="bottom"))
    _layout(fig, None, 400, f"{C.RESOURCE_LABELS[resource]} utilization", ".0%")
    _legend_below(fig, 165)
    fig.update_layout(margin=dict(t=16))
    fig.update_xaxes(hoverformat="%b %Y")
    return fig
