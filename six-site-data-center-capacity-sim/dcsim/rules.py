"""Deterministic, rules-based executive insights.

Every insight cites the modeled metrics it is based on, so a reader can trace
it back to the forecast tables. No AI is involved.

Milestone 2 covers the Executive Overview insights; Milestone 4 extends this
module into the full Executive Decisions panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config as C
from .forecast import ForecastResult, period_label

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Info": 3, "Low": 4}
LOW_UTILIZATION = 0.50  # below this, a site is a candidate to absorb migrated load
HIGH_PUE = 1.50


@dataclass(frozen=True)
class Insight:
    severity: str          # Critical | High | Medium | Info | Low
    title: str
    detail: str
    action: str | None
    basis: str             # which modeled metrics / rules the insight rests on
    site_id: str | None = None


def _unit(res: str) -> str:
    return C.RESOURCE_UNITS[res]


def _res(res: str) -> str:
    return C.RESOURCE_LABELS[res].lower()


def fmt_amount(x: float, res: str) -> str:
    return f"{x:,.0f} racks" if res == "space" else f"{x:,.1f} {_unit(res)}"


def project_action(p: pd.Series) -> str:
    amount = fmt_amount(p.capacity_added, p.resource)
    cost = f"~${p.cost_usd / 1e6:,.1f}M"
    if p.commit_by_month <= 0:
        when = "Commit now"
    else:
        when = f"Commit by {period_label(p.commit_by_month)}"
    text = f"{when}: add {amount} of {_res(p.resource)} ({cost}), live {period_label(p.activation_month)}."
    if p.arrives_late:
        text += (f" Lead time ({p.lead_time_months} mo) means it lands after the need date "
                 f"({period_label(p.need_month)}) — bridge with load placement or deferral of new deployments.")
    return text


def overview_insights(result: ForecastResult, projects: pd.DataFrame) -> list[Insight]:
    """Headline insights for the Executive Overview.

    `projects` is capital.size_projects(result): the expansion each site/resource
    needs to stay at or below the warning threshold through the horizon.
    """
    sc = result.scenario
    out: list[Insight] = []
    ss = result.site_summary.set_index("site_id")
    sr = result.site_resource.set_index(["site_id", "resource"])
    k = result.kpis

    def action_for(sid, res):
        if projects.empty:
            return None
        match = projects[(projects["site_id"] == sid) & (projects["resource"] == res)]
        return project_action(match.iloc[0]) if len(match) else None

    # 1. Earliest site-level constraint.
    ec = k["earliest_constraint"]
    if ec and ec["kind"] == "exhaustion":
        sid, res, m = ec["site_id"], ec["resource"], ec["month"]
        row = sr.loc[(sid, res)]
        d_m, c_m = result.demand[result.site_ids.index(sid), C.RESOURCES.index(res), m], \
            result.capacity[result.site_ids.index(sid), C.RESOURCES.index(res), m]
        out.append(Insight(
            severity=ss.loc[sid, "risk_status"],
            title=f"{ec['site_name']} ({sid}) runs out of {_res(res)} in {ec['period']}",
            detail=(f"{row.current_utilization:.0%} utilized today, demand growing "
                    f"{ss.loc[sid, 'annual_growth']:.1%}/yr. In month {m}, modeled demand reaches "
                    f"{fmt_amount(d_m, res)} against {fmt_amount(c_m, res)} of capacity."),
            action=action_for(sid, res),
            basis="Earliest month where forecast demand exceeds available capacity (incl. planned expansions).",
            site_id=sid,
        ))

        # 2. Portfolio totals mask the site constraint.
        pe = k["portfolio_exhaustion_month"][res]
        if pe is None or pe > m:
            n_sites = int(((sr["exhaustion_month"].notna()) &
                           (sr.index.get_level_values("resource") == res)).sum())
            dem0 = result.demand[:, C.RESOURCES.index(res), 0].sum()
            cap0 = result.capacity[:, C.RESOURCES.index(res), 0].sum()
            when = "does not run out within 36 months" if pe is None else f"runs out only in {period_label(pe)}"
            out.append(Insight(
                severity="Info",
                title=f"Portfolio totals hide site-level {_res(res)} constraints",
                detail=(f"Portfolio {_res(res)} is {dem0 / cap0:.0%} utilized and {when}, yet {n_sites} "
                        f"site(s) exhaust {_res(res)} earlier. Spare capacity at one facility does not "
                        f"serve load at another without migration."),
                action=None,
                basis="Portfolio utilization = sum of demand / sum of capacity; compared with site-level exhaustion.",
            ))

    # 3. Other sites at High / Critical risk.
    flagged = ss[ss["risk_status"].isin(["High", "Critical"])]
    for sid, r in flagged.iterrows():
        if ec and sid == ec["site_id"]:
            continue
        res = r["limiting_resource"]
        out.append(Insight(
            severity=r["risk_status"],
            title=f"{r['site_name']} ({sid}) is {r['risk_status'].lower()} risk — {_res(res)} constrained",
            detail=f"Rule triggered: {r['risk_reason']}.",
            action=action_for(sid, res),
            basis=f"Risk rules with warning {sc.warning_threshold:.0%} / critical {sc.critical_threshold:.0%} thresholds.",
            site_id=sid,
        ))

    # 4. Optimization: under-used sites could absorb load from constrained ones.
    if len(flagged):
        p = C.RESOURCES.index("power")
        for sid, r in ss.iterrows():
            if r["risk_status"] == "Low" and r["max_current_utilization"] < LOW_UTILIZATION:
                i = result.site_ids.index(sid)
                head_h = result.capacity[i, p, sc.horizon] - result.demand[i, p, sc.horizon]
                out.append(Insight(
                    severity="Info",
                    title=f"{r['site_name']} ({sid}) has room to absorb migrated load",
                    detail=(f"Peak current utilization is {r['max_current_utilization']:.0%}; "
                            f"{head_h:,.1f} MW power headroom remains at month {sc.horizon}. Placing new "
                            f"deployments here can defer spend at {', '.join(flagged.index)}."),
                    action=None,
                    basis=f"Low-risk sites with all resources below {LOW_UTILIZATION:.0%} utilization today.",
                    site_id=sid,
                ))

    # 5. Efficiency opportunity at high-PUE sites.
    high_pue = result.sites[result.sites["pue"] >= HIGH_PUE]
    if len(high_pue) and sc.efficiency == 0:
        names = ", ".join(f"{r.site_id} (PUE {r.pue:.2f})" for r in high_pue.itertuples())
        out.append(Insight(
            severity="Info",
            title="Efficiency upgrades are an alternative to building",
            detail=(f"{names} run well above the portfolio's best PUE of {result.sites['pue'].min():.2f}. "
                    "Try the Efficiency Optimization scenario to see the deferral effect."),
            action=None,
            basis=f"Sites with PUE >= {HIGH_PUE:.2f}.",
        ))

    return sorted(out, key=lambda x: SEVERITY_ORDER.get(x.severity, 9))
