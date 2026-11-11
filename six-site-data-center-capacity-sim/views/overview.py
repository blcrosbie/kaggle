"""Executive Overview - portfolio KPIs, earliest constraint, site comparison (SOW Modules 1 & 2)."""

import pandas as pd
import streamlit as st

from dcsim import config as C
from dcsim import theme as T
from dcsim.forecast import period_label
from dcsim.rules import fmt_amount, overview_insights
from ui import charts, state
from ui import components as ui

ctx = state.context()
r, sc = ctx.result, ctx.scenario
k = r.kpis
warn, crit, h = sc.warning_threshold, sc.critical_threshold, sc.horizon
summary = r.site_summary
sr_idx = r.site_resource.set_index(["site_id", "resource"])
risk_of = summary.set_index("site_id")["risk_status"]

ui.page_header("Executive Overview", ctx)

# --- Earliest site-level constraint (always prominent) ------------------------
ec = k["earliest_constraint"]
if ec:
    res, m = ec["resource"], ec["month"]
    pe = k["portfolio_exhaustion_month"][res]
    j = C.RESOURCES.index(res)
    pu = r.demand[:, j, 0].sum() / r.capacity[:, j, 0].sum()
    p_txt = "never runs out within 36 months" if pe is None else f"runs out in {period_label(pe)}"
    if ec["kind"] == "exhaustion":
        headline = f"{ec['site_name']} ({ec['site_id']}) runs out of {C.RESOURCE_LABELS[res].lower()} in {ec['period']} — month {m}"
        eyebrow = "Earliest site-level capacity exhaustion"
        status = risk_of[ec["site_id"]]
    else:
        headline = f"{ec['site_name']} ({ec['site_id']}) reaches the {warn:.0%} {C.RESOURCE_LABELS[res].lower()} warning threshold in {ec['period']}"
        eyebrow = "No site exhausts capacity within 36 months — earliest warning"
        status = "Medium"
    beyond = f" This is beyond the selected {h}-month horizon." if m > h else ""
    ui.hero(eyebrow, headline,
            f"For comparison, portfolio-wide {C.RESOURCE_LABELS[res].lower()} is {pu:.0%} utilized and {p_txt} — "
            f"totals across sites would hide this constraint.{beyond}", status)
else:
    ui.hero("Capacity outlook", f"No site reaches the {warn:.0%} warning threshold within 36 months", "", "Low")

# --- KPI cards -------------------------------------------------------------
u_p = k["power_utilization"]
ui.kpi_row([
    ui.kpi("Electrical capacity", f"{k['power_capacity_mw']:,.1f}", "MW", "Installed across 6 facilities"),
    ui.kpi("Current electrical demand", f"{k['power_demand_mw']:,.1f}", "MW", "Actual (synthetic baseline)"),
    ui.kpi("Electrical utilization", ui.fmt_pct(u_p), "", f"Warning at {warn:.0%} · critical at {crit:.0%}",
           badge=ui.chip(ui.util_status(u_p, warn, crit))),
    ui.kpi("Electrical headroom", f"{k['power_headroom_mw']:,.1f}", "MW",
           f"{k['power_headroom_mw'] / k['power_capacity_mw']:.0%} of capacity unused"),
])
st.write("")
n_app = k["sites_approaching_constraint"]
ui.kpi_row([
    ui.kpi("Cooling utilization", ui.fmt_pct(k["cooling_utilization"]), "",
           f"{k['cooling_load_mwt']:,.1f} of {k['cooling_capacity_mwt']:,.1f} MWt "
           f"({k['cooling_capacity_mwt'] * C.MWT_TO_TONS:,.0f} tons)",
           badge=ui.chip(ui.util_status(k["cooling_utilization"], warn, crit))),
    ui.kpi("Racks occupied", f"{k['racks_occupied']:,.0f}", f"/ {k['rack_capacity']:,.0f}",
           f"{k['rack_utilization']:.0%} of rack positions occupied"),
    ui.kpi("Sites approaching constraints", f"{n_app}", "of 6",
           (f"Reach {warn:.0%} on any resource within {h} months: {', '.join(k['sites_approaching_ids'])}"
            if n_app else f"None reach {warn:.0%} within {h} months")),
    ui.kpi("Portfolio capacity risk", f"{T.RISK_ICONS[k['portfolio_risk']]} {k['portfolio_risk']}", "",
           f"Worst site status (conservative roll-up): "
           f"{', '.join(summary.loc[summary['risk_status'] == k['portfolio_risk'], 'site_id'])}",
           accent=T.RISK_COLORS[k["portfolio_risk"]]),
])
st.write("")
d0 = k["power_demand_mw"]
pe = k["portfolio_power_exhaustion_month"]
ui.kpi_row([
    ui.kpi(f"Forecast demand · {mo} mo", f"{k[f'power_demand_{mo}m_mw']:,.1f}", "MW",
           f"{k[f'power_demand_{mo}m_mw'] / d0 - 1:+.0%} vs today · {period_label(mo)} (projected)")
    for mo in (12, 24, 36)
] + [
    ui.kpi("Portfolio-level power exhaustion", "None" if pe is None else period_label(pe), "",
           "Σ demand vs Σ capacity — see the site-level constraint above" if pe is None
           else f"Month {pe}; site-level constraints arrive earlier"),
])

# --- Charts ------------------------------------------------------------------
st.write("")
c1, c2 = st.columns([3, 2], gap="large")
with c1:
    st.subheader("Portfolio demand vs capacity")
    res = st.segmented_control("Resource", options=list(C.RESOURCES), default="power", key="ov_res",
                               format_func=C.RESOURCE_LABELS.get, label_visibility="collapsed") or "power"
    st.plotly_chart(charts.portfolio_forecast(r, res), **charts.CHART_KW)
    st.caption("Square = actual synthetic baseline (Oct 2026). Dashed line = projected demand. "
               "Red marker = first site to exhaust this resource.")
with c2:
    st.subheader("Power utilization by site")
    st.plotly_chart(charts.site_utilization(r, "power"), **charts.CHART_KW)
    st.caption(f"Bar = today's utilization (actual). Diamond = projected at month {h}.")

# --- Capacity risk indicators -------------------------------------------------
st.subheader("Capacity risk indicators")
rows = list(summary.itertuples(index=False))
for chunk in (rows[:3], rows[3:]):
    cols = st.columns(3)
    for col, row in zip(cols, chunk):
        col.markdown(ui.site_card(row, sr_idx, warn, crit), unsafe_allow_html=True)
    st.write("")
st.caption(f"Bar fill = utilization today (green < warn, yellow ≥ warn, orange ≥ crit, red > 100%). Black mark = month {h}. "
           "Ticks = warn / crit / 100%. Scale 0–120%. ◂ = first limiting resource.")

# --- Six-site comparison table -------------------------------------------------
st.subheader("Six-site comparison")
table = pd.DataFrame({
    "Site": summary["site_id"],
    "Facility": summary["site_name"],
    "Region": summary["region"],
    "Risk": [f"{T.RISK_ICONS[s]} {s}" for s in summary["risk_status"]],
    "Power": summary["power_utilization"] * 100,
    "Cooling": summary["cooling_utilization"] * 100,
    "Racks": summary["space_utilization"] * 100,
    "Power headroom (MW)": [sr_idx.loc[(s, "power"), "current_headroom"] for s in summary["site_id"]],
    "Growth / yr": summary["annual_growth"] * 100,
    "First limiting resource": [C.RESOURCE_LABELS[x] for x in summary["limiting_resource"]],
    "First exhaustion": [ui.fmt_month(x) for x in summary["earliest_exhaustion_month"]],
})
util_col = lambda label: st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.0f%%", color=T.ACCENT)  # noqa: E731
st.dataframe(
    table, hide_index=True,
    column_config={
        "Region": None,
        "Power": util_col("Power util."), "Cooling": util_col("Cooling util."), "Racks": util_col("Rack util."),
        "Power headroom (MW)": st.column_config.NumberColumn("Power headroom", format="%.1f MW"),
        "Growth / yr": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

# --- Executive insights -------------------------------------------------------
st.subheader("Executive insights & recommended actions")
insights = overview_insights(r, ctx.needs)
for ins in insights:
    ui.insight_card(ins)
st.caption("Rules-based and deterministic: each insight cites the modeled metric it rests on. "
           "Full decision support is on the Executive Decisions page.")

# --- Facility comparison (SOW Module 2) --------------------------------------
st.divider()
st.subheader("Facility comparison")
names = summary.set_index("site_id")["site_name"]
default_sel = summary.loc[summary["risk_status"] != "Low", "site_id"].tolist() or r.site_ids[:3]
f1, f2 = st.columns([3, 2])
sel = f1.multiselect("Facilities to compare", options=r.site_ids, default=default_sel,
                     format_func=lambda s: f"{s} · {names[s]}", key="cmp_sites")
cres = f2.segmented_control("Resource", options=list(C.RESOURCES), default="power", key="cmp_res",
                            format_func=C.RESOURCE_LABELS.get) or "power"
if not sel:
    st.info("Select at least one facility to compare.")
else:
    st.plotly_chart(charts.site_utilization_forecast(r, sel, cres), **charts.CHART_KW)
    needs = ctx.needs.set_index(["site_id", "resource"]) if len(ctx.needs) else None
    comp = []
    for sid in sel:
        for res_ in C.RESOURCES:
            x = sr_idx.loc[(sid, res_)]
            need = needs.loc[(sid, res_)] if needs is not None and (sid, res_) in needs.index else None
            comp.append({
                "Site": f"{sid} · {names[sid]}",
                "Resource": C.RESOURCE_LABELS[res_],
                "Now": x.current_utilization * 100,
                f"Month {h}": x.horizon_utilization * 100,
                "Headroom now": fmt_amount(x.current_headroom, res_),
                f"Warning ({warn:.0%}) reached": ui.fmt_month(x.warning_month),
                "Exhaustion": ui.fmt_month(x.exhaustion_month),
                "Expansion needed": "—" if need is None else fmt_amount(need.capacity_added, res_),
                "Est. cost": "—" if need is None else ui.fmt_usd_m(need.cost_usd),
                "Commit by": "—" if need is None else ("Now" if need.commit_by_month <= 0 else period_label(need.commit_by_month)),
            })
    st.dataframe(pd.DataFrame(comp), hide_index=True, row_height=32, height=32 * (len(comp) + 1) + 3,
                 column_config={"Now": util_col("Util. now"), f"Month {h}": util_col(f"Util. month {h}")})
    st.caption(f"Expansion needed = smallest addition keeping utilization ≤ {warn:.0%} through month {h}, "
               "rounded to build increments; cost from configurable unit costs (illustrative, not an engineering estimate).")
