"""Reusable HTML components: plain bordered panels with title bars, filled status
tags, site cards with bar indicators, and insight rows. Square corners, 1px rules,
Ubuntu / Ubuntu Mono type.

Neutral colours are tints of `currentColor` (the active Streamlit theme's text
colour), so every component follows the Light / Dark switch live, without Python
needing to know which theme is showing. Data and status colours are fixed and
were chosen to read on both backgrounds (see dcsim/theme.py)."""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from dcsim import config as C
from dcsim import theme as T
from dcsim.forecast import period_label

MONO = ", ".join(f"'{f}'" if " " in f else f for f in T.MONO_FAMILY)
BAR_MAX = 1.2  # bar indicators span 0-120% so overloads stay visible

def _mix(pct: int) -> str:
    """A tint of the active theme's text colour, so components follow the light/dark switch."""
    return f"color-mix(in srgb, currentColor {pct}%, transparent)"


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Ubuntu+Mono:wght@400;700&family=Ubuntu:wght@400;500;700&display=swap');
:root {{ --mono: {MONO}; --accent: {T.ACCENT}; }}
.block-container {{ padding-top: 3.2rem; max-width: 1560px; }}
h1 {{
  font-size: 1.6rem !important; font-weight: 700 !important;
  padding-bottom: .3rem !important; border-bottom: 2px solid currentColor;
}}
[data-testid="stHeading"] h3 {{
  font-size: 1.05rem !important; font-weight: 700 !important;
  border-bottom: 1px solid {_mix(40)}; padding: 0 0 4px 0 !important; margin: .6rem 0 .6rem 0;
}}
[data-testid="stSidebar"] h4 {{ font-size: .95rem !important; font-weight: 700 !important; }}
[data-testid="stCaptionContainer"] {{ font-family: var(--mono); font-size: .86rem; }}
.notice {{
  border: 1px solid {T.STATUS['warning']}; background: color-mix(in srgb, {T.STATUS['warning']} 16%, transparent);
  padding: 5px 12px; font-size: .9rem; margin-bottom: .9rem;
}}
.notice b {{ font-weight: 700; }}
.statusbar {{
  display: flex; flex-wrap: wrap; border: 1px solid {_mix(40)}; background: {_mix(5)};
  font-family: var(--mono); font-size: .9rem; margin: -.2rem 0 1.1rem 0;
}}
.statusbar .cell {{ padding: 3px 10px; border-right: 1px solid {_mix(40)}; white-space: nowrap; }}
.statusbar .cell .k {{ opacity: .7; margin-right: 4px; }}
.panel {{ border: 1px solid {_mix(40)}; background: {_mix(3)}; height: 100%; }}
.panel > .ph {{
  background: {_mix(11)}; border-bottom: 1px solid {_mix(40)}; padding: 3px 10px;
  font-size: .82rem; font-weight: 500;
  display: flex; justify-content: space-between; gap: 8px; align-items: center;
}}
.panel > .pb {{ padding: 8px 12px 10px; }}
.kpi.flagged {{ border-left: 5px solid var(--kpi-accent); }}
.kpi .value {{
  font-family: var(--mono); font-size: 1.9rem; font-weight: 700; line-height: 1.15;
  font-variant-numeric: tabular-nums;
}}
.kpi .value .unit {{ font-size: 1rem; font-weight: 400; opacity: .8; margin-left: 4px; }}
.kpi .sub {{ font-size: .86rem; opacity: .72; margin-top: 4px; font-family: var(--mono); line-height: 1.25; }}
.tag {{
  display: inline-flex; align-items: center; gap: 5px; padding: 0 7px; font-family: var(--mono);
  font-size: .86rem; font-weight: 700; white-space: nowrap; vertical-align: middle;
  background: var(--tag-bg); color: var(--tag-fg); border: 1px solid var(--tag-bg);
}}
.tag.info {{ background: transparent; color: inherit; border-color: var(--accent); }}
.callout {{ border: 1px solid {_mix(40)}; border-left: 6px solid var(--co); background: {_mix(3)};
  margin: .2rem 0 1rem 0; padding: 9px 14px 11px; }}
.callout .eyebrow {{ font-size: .86rem; opacity: .85; display: flex; gap: 10px; align-items: center; }}
.callout .headline {{ font-size: 1.3rem; font-weight: 700; margin: 4px 0 3px; }}
.callout .detail {{ font-size: .95rem; opacity: .85; }}
.sc .meta {{ opacity: .72; font-size: .86rem; font-family: var(--mono); margin-bottom: 6px; }}
.sc .swatch {{ display: inline-block; width: 11px; height: 11px; background: var(--site); margin-right: 7px;
  vertical-align: -1px; }}
.sc .ind {{ display: grid; grid-template-columns: 74px 1fr 100px; gap: 8px; align-items: center;
  font-size: .9rem; padding: 3px 0; font-family: var(--mono); }}
.sc .ind .lbl {{ opacity: .8; }}
.sc .ind .val {{ text-align: right; font-variant-numeric: tabular-nums; }}
.sc .ind.limiting .lbl {{ opacity: 1; font-weight: 700; }}
.bar {{ position: relative; height: 12px; border: 1px solid {_mix(45)}; }}
.bar .fill {{ position: absolute; left: 0; top: 0; bottom: 0; background: var(--fill); }}
.bar .tick {{ position: absolute; top: -3px; bottom: -3px; width: 1px; background: var(--tick); }}
.bar .hz {{ position: absolute; top: -4px; bottom: -4px; width: 3px; margin-left: -1px; background: currentColor; }}
.sc .kv {{ display: flex; justify-content: space-between; font-size: .9rem;
  border-top: 1px solid {_mix(18)}; margin-top: 7px; padding-top: 5px; font-family: var(--mono); }}
.sc .kv span {{ opacity: .75; }}
.sc .reason {{ min-height: 2.6em; font-size: .84rem; opacity: .72; margin-top: 4px; line-height: 1.3; }}
.ins {{ display: grid; grid-template-columns: 110px 1fr; border: 1px solid {_mix(40)}; margin-bottom: 8px; }}
.ins .pri {{ border-right: 1px solid {_mix(40)}; background: {_mix(6)}; padding: 8px; display: flex;
  flex-direction: column; gap: 6px; font-family: var(--mono); font-size: .86rem; }}
.ins .body {{ padding: 8px 12px; }}
.ins .title {{ font-weight: 700; }}
.ins .detail {{ opacity: .85; font-size: .94rem; margin-top: 2px; }}
.ins .action {{ font-size: .94rem; margin-top: 6px; }}
.ins .action::before {{ content: "> "; font-family: var(--mono); font-weight: 700; color: var(--accent); }}
.ins .basis {{ opacity: .68; font-size: .84rem; margin-top: 5px; font-family: var(--mono); }}
.legend-note {{ opacity: .7; font-size: .84rem; font-family: var(--mono); }}
</style>
"""

INFO_COLOR = T.ACCENT
SEVERITY_COLORS = {**T.RISK_COLORS, "Info": INFO_COLOR}
SEVERITY_ICONS = {**T.RISK_ICONS, "Info": "○"}


def esc(x) -> str:
    return html.escape(str(x))


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def synthetic_banner() -> None:
    st.markdown(
        '<div class="notice" role="note"><b>Synthetic data — demonstration only.</b> '
        "All facilities, names and values are fictional.</div>",
        unsafe_allow_html=True,
    )


def page_header(title: str, ctx) -> None:
    sc = ctx.scenario
    cells = [
        ("scenario", sc.name),
        ("horizon", f"{sc.horizon} mo"),
        ("baseline", f"{C.AS_OF_PERIOD.strftime('%b %Y')} (actual)"),
        ("thresholds", f"{sc.warning_threshold:.0%} / {sc.critical_threshold:.0%}"),
        ("planned expansions", "included" if sc.include_planned else "excluded"),
    ]
    if sc.portfolio_growth is not None:
        cells.append(("growth", f"{sc.portfolio_growth:.0%}/yr all sites"))
    if sc.efficiency:
        cells.append(("efficiency", f"-{sc.efficiency:.0%}"))
    if sc.budget_usd is not None:
        cells.append(("budget", f"${sc.budget_usd / 1e6:,.0f}M"))
    if sc.added_expansions:
        cells.append(("scenario expansions", str(len(sc.added_expansions))))
    st.title(title)
    st.markdown('<div class="statusbar">' + "".join(
        f'<div class="cell"><span class="k">{k}:</span>{esc(v)}</div>' for k, v in cells) + "</div>",
        unsafe_allow_html=True)


def chip(status: str) -> str:
    """Filled status tag with a shape icon; Info is outlined."""
    icon = SEVERITY_ICONS.get(status, "●")
    if status not in T.RISK_TAG:
        return f'<span class="tag info"><span aria-hidden="true">{icon}</span>{esc(status)}</span>'
    bg, fg = T.RISK_TAG[status]
    return (f'<span class="tag" style="--tag-bg:{bg};--tag-fg:{fg}"><span aria-hidden="true">{icon}</span>'
            f"{esc(status)}</span>")


def panel(title: str, body: str, right: str = "", cls: str = "", style: str = "") -> str:
    return (f'<div class="panel {cls}" style="{style}"><div class="ph"><span>{title}</span>{right}</div>'
            f'<div class="pb">{body}</div></div>')


def kpi(label: str, value: str, unit: str = "", sub: str = "", accent: str | None = None, badge: str = "") -> str:
    unit_html = f'<span class="unit">{esc(unit)}</span>' if unit else ""
    body = f'<div class="value">{esc(value)}{unit_html} {badge}</div><div class="sub">{sub}</div>'
    return panel(esc(label), body, cls="kpi flagged" if accent else "kpi",
                 style=f"--kpi-accent:{accent}" if accent else "")


def kpi_row(cards: list[str]) -> None:
    cols = st.columns(len(cards), gap="small")
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def util_status(u: float, warn: float, crit: float) -> str:
    return "Critical" if u > 1 else "High" if u >= crit else "Medium" if u >= warn else "Low"


def fmt_month(m) -> str:
    if m is None or (isinstance(m, float) and not np.isfinite(m)) or pd.isna(m):
        return f"Not within {C.MAX_HORIZON} mo"
    return f"{period_label(int(m))} (M{int(m)})"


def fmt_pct(x: float) -> str:
    return f"{x:.0%}"


def fmt_usd_m(x: float) -> str:
    return f"${x / 1e6:,.1f}M"


def hero(eyebrow: str, headline: str, detail: str, status: str) -> None:
    color = SEVERITY_COLORS.get(status, T.ACCENT)
    st.markdown(
        f'<div class="callout" style="--co:{color}"><div class="eyebrow">{chip(status)}<span>{esc(eyebrow)}</span></div>'
        f'<div class="headline">{esc(headline)}</div><div class="detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def bar_indicator(now: float, later: float, warn: float, crit: float) -> str:
    """Bar: fill = utilization today (status-coloured), black mark = end of horizon, ticks = thresholds."""
    pct = lambda x: f"{min(max(x, 0), BAR_MAX) / BAR_MAX * 100:.1f}%"  # noqa: E731
    fill = T.RISK_COLORS[util_status(now, warn, crit)]
    ticks = "".join(f'<div class="tick" style="left:{pct(v)};--tick:{c}"></div>'
                    for v, c in ((warn, T.STATUS["warning"]), (crit, T.STATUS["serious"]), (1.0, T.STATUS["critical"])))
    return (f'<div class="bar" title="now {now:.0%} → horizon {later:.0%}">'
            f'<div class="fill" style="width:{pct(now)};--fill:{fill}"></div>{ticks}'
            f'<div class="hz" style="left:{pct(later)}"></div></div>')


def site_card(row, sr: pd.DataFrame, warn: float, crit: float) -> str:
    """Site card: one bar per resource plus first exhaustion and the rule that set the risk level."""
    inds = []
    for res in C.RESOURCES:
        r = sr.loc[(row.site_id, res)]
        lim = res == row.limiting_resource
        inds.append(f'<div class="ind{" limiting" if lim else ""}"><span class="lbl">{res}{" ◂" if lim else ""}</span>'
                    f"{bar_indicator(r.current_utilization, r.horizon_utilization, warn, crit)}"
                    f'<span class="val">{r.current_utilization:.0%} → {r.horizon_utilization:.0%}</span></div>')
    ex = fmt_month(row.earliest_exhaustion_month)
    body = (f'<div class="meta">{esc(row.region)} · growth {row.annual_growth:.1%}/yr</div>'
            + "".join(inds)
            + f'<div class="kv"><span>first exhaustion</span><b>{esc(ex)}</b></div>'
            f'<div class="reason">Why: {esc(row.risk_reason)}</div>')
    title = (f'<span class="swatch" style="--site:{T.SITE_COLORS[row.site_id]}"></span>'
             f"<b>{esc(row.site_id)}</b> · {esc(row.site_name)}")
    return panel(title, body, right=chip(row.risk_status), cls="sc")


def insight_card(ins) -> None:
    action = f'<div class="action">{esc(ins.action)}</div>' if ins.action else ""
    site = esc(ins.site_id) if ins.site_id else "portfolio"
    st.markdown(
        f'<div class="ins"><div class="pri">{chip(ins.severity)}<span>{site}</span></div>'
        f'<div class="body"><div class="title">{esc(ins.title)}</div><div class="detail">{esc(ins.detail)}</div>'
        f'{action}<div class="basis">Based on: {esc(ins.basis)}</div></div></div>',
        unsafe_allow_html=True,
    )


# Streamlit stores the Light / Dark / System choice in localStorage under this key
# (frontend constant: `stActiveTheme-${pathname}-v2`). With no stored choice it
# follows the OS setting; we want Campbell dark as the first-visit default.
_DEFAULT_DARK_JS = """<script>
(function () {
  try {
    var w = window.parent, ls = w.localStorage, FLAG = "dcsim-dark-default-v1";
    if (ls.getItem(FLAG)) return;                     // applied once per browser; later choices are respected
    ls.setItem(FLAG, "1");
    // Streamlit may already have stored "System" on load, so overwrite unless the viewer chose Light/Dark.
    var key = "stActiveTheme-" + w.location.pathname + "-v2", cur = ls.getItem(key);
    if (cur === null || cur === JSON.stringify("System")) {
      ls.setItem(key, JSON.stringify("Dark"));
      // Reload from the app window's own realm (the iframe sandbox can't navigate its parent).
      w.setTimeout(w.Function("location.reload()"), 0);
    }
  } catch (e) { /* storage blocked: fall back to the OS setting */ }
})();
</script>"""


def default_dark_theme() -> None:
    """Make the Campbell dark theme the first-visit default; the app menu still offers Light / Dark / System."""
    st.iframe(_DEFAULT_DARK_JS, height=1)
