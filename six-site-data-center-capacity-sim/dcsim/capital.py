"""Simplified capital planning: size, price, rank and budget expansion projects.

Sizing:   for each site and resource, the smallest capacity addition (rounded up
          to build increments) that keeps utilisation at or below the warning
          threshold in every month of the planning horizon.
Timing:   needed by the first month utilisation reaches the warning threshold;
          must be committed `lead_time` months earlier. Earliest possible
          activation is month `lead_time` (if you commit today).
Pricing:  capacity added x configurable unit cost. Order-of-magnitude only.
Ranking:  priority = 0.5 x urgency + 0.3 x risk + 0.2 x cost score (each 0-100):
            urgency    = 100 x (1 - slack / 24), clipped to 0-100, where
                         slack = need month - lead time (months left to commit)
            risk       = site risk status: Low 0, Medium 33, High 67, Critical 100
            cost score = 100 x (cheapest candidate cost / this project's cost)
Budget:   fund projects in priority order while they fit in the remaining budget
          (a cheaper lower-ranked project may still fit after a larger one does not).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .forecast import ForecastResult, period_label, run_forecast
from .scenario import Expansion, Scenario

PROJECT_COLUMNS = [
    "rank", "site_id", "site_name", "resource", "unit", "risk_status",
    "capacity_added", "cost_usd", "need_month", "need_period", "lead_time_months",
    "commit_by_month", "activation_month", "activation_period", "arrives_late",
    "exhaustion_if_not_built", "months_over_capacity_if_not_built",
    "urgency_score", "risk_score", "cost_score", "priority_score",
    "funded", "cumulative_funded_usd", "budget_remaining_usd",
]


@dataclass
class CapitalPlan:
    scenario: Scenario
    projects: pd.DataFrame
    budget_usd: float | None
    before: ForecastResult
    after: ForecastResult  # forecast with funded projects applied

    @property
    def proposed_total(self) -> float:
        return float(self.projects["cost_usd"].sum())

    @property
    def funded_total(self) -> float:
        return float(self.projects.loc[self.projects["funded"], "cost_usd"].sum())

    @property
    def unfunded(self) -> pd.DataFrame:
        return self.projects[~self.projects["funded"]]

    @property
    def committed_planned_total(self) -> float:
        """Already-planned expansion capex (outside the new-project budget)."""
        f = self.before.expansions_frame()
        return float(f.loc[f["source"] == "planned", "cost_usd"].sum())


def size_projects(result: ForecastResult) -> pd.DataFrame:
    sc, h = result.scenario, result.horizon
    target = sc.warning_threshold
    risk = result.site_summary.set_index("site_id")["risk_status"]
    sr = result.site_resource.set_index(["site_id", "resource"])
    rows = []
    for i, sid in enumerate(result.site_ids):
        for j, res in enumerate(C.RESOURCES):
            d = result.demand[i, j, : h + 1]
            cap = result.capacity[i, j, : h + 1]
            shortfall = float(np.max(d / target - cap))
            if shortfall <= 1e-9:
                continue
            inc = C.SIZING_INCREMENT[res]
            amount = math.ceil(shortfall / inc - 1e-9) * inc
            need = int(np.argmax(d >= target * cap))
            lead = int(sc.lead_time[res])
            activation = max(need, lead, 1)
            ex = sr.loc[(sid, res), "exhaustion_month"]
            rows.append({
                "site_id": sid,
                "site_name": result.sites["site_name"].iat[i],
                "resource": res,
                "unit": C.RESOURCE_UNITS[res],
                "risk_status": risk[sid],
                "capacity_added": amount,
                "cost_usd": amount * sc.unit_cost(res),
                "need_month": need,
                "need_period": period_label(need),
                "lead_time_months": lead,
                "commit_by_month": need - lead,
                "activation_month": activation,
                "activation_period": period_label(activation),
                "arrives_late": need < lead,
                "exhaustion_if_not_built": ex,
                "months_over_capacity_if_not_built": int(np.sum(d > cap)),
            })
    return pd.DataFrame(rows)


def score_projects(projects: pd.DataFrame) -> pd.DataFrame:
    if projects.empty:
        return projects.assign(urgency_score=[], risk_score=[], cost_score=[], priority_score=[], rank=[])
    p = projects.copy()
    slack = p["commit_by_month"].astype(float)
    p["urgency_score"] = (100 * (1 - slack / C.URGENCY_SLACK_MONTHS)).clip(0, 100)
    p["risk_score"] = p["risk_status"].map(C.RISK_SCORE).astype(float)
    min_cost = p["cost_usd"].min()
    p["cost_score"] = np.where(p["cost_usd"] > 0, 100 * min_cost / p["cost_usd"].where(p["cost_usd"] > 0, 1), 100)
    w = C.PRIORITY_WEIGHTS
    p["priority_score"] = w["urgency"] * p["urgency_score"] + w["risk"] * p["risk_score"] + w["cost"] * p["cost_score"]
    p = p.sort_values(["priority_score", "need_month"], ascending=[False, True]).reset_index(drop=True)
    p["rank"] = np.arange(1, len(p) + 1)
    return p


def apply_budget(projects: pd.DataFrame, budget_usd: float | None) -> pd.DataFrame:
    p = projects.copy()
    funded, cum, remaining = [], [], []
    left = math.inf if budget_usd is None else float(budget_usd)
    total = 0.0
    for cost in p["cost_usd"]:
        ok = cost <= left + 1e-6
        if ok:
            left -= cost
            total += cost
        funded.append(ok)
        cum.append(total)
        remaining.append(None if budget_usd is None else left)
    p["funded"] = funded
    p["cumulative_funded_usd"] = cum
    p["budget_remaining_usd"] = remaining
    return p


def projects_to_expansions(projects: pd.DataFrame) -> list[Expansion]:
    return [
        Expansion(
            site_id=r.site_id, resource=r.resource, amount=float(r.capacity_added),
            activation_month=int(r.activation_month), cost_usd=float(r.cost_usd),
            label=f"Recommended {C.RESOURCE_LABELS[r.resource].lower()} expansion",
            source="capital plan",
        )
        for r in projects.itertuples(index=False)
    ]


def plan_capital(sites: pd.DataFrame, planned: pd.DataFrame | None, scenario: Scenario) -> CapitalPlan:
    before = run_forecast(sites, planned, scenario)
    projects = apply_budget(score_projects(size_projects(before)), scenario.budget_usd)
    if not projects.empty:
        projects = projects[PROJECT_COLUMNS]
    funded = projects[projects["funded"]] if not projects.empty else projects
    after_scenario = scenario.with_expansions(*projects_to_expansions(funded))
    after = run_forecast(sites, planned, after_scenario, validate=False)

    if not projects.empty:
        a = after.site_resource.set_index(["site_id", "resource"])
        keys = list(zip(projects["site_id"], projects["resource"]))
        projects = projects.assign(
            horizon_headroom_after=[a.loc[k, "horizon_headroom"] for k in keys],
            horizon_utilization_after=[a.loc[k, "horizon_utilization"] for k in keys],
            exhaustion_after=[a.loc[k, "exhaustion_month"] for k in keys],
        )
    return CapitalPlan(scenario, projects, scenario.budget_usd, before, after)


def budget_constrained_forecast(sites, planned, scenario: Scenario) -> tuple[ForecastResult, CapitalPlan]:
    """The forecast a user sees under the Budget-Constrained scenario: funded projects applied."""
    plan = plan_capital(sites, planned, scenario)
    return plan.after, plan
