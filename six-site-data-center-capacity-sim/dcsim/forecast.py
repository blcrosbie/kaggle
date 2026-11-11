"""Monthly capacity forecasting engine.

One vectorised core (`project`) serves both the deterministic forecast (a single
draw with no uncertainty) and the Monte Carlo simulation (thousands of draws),
so the two can never disagree about the underlying arithmetic.

Array convention: (draws N, sites S, resources R, months M) with months 0..36,
where month 0 is the synthetic "actual" baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import config as C
from .data import baseline_arrays, expansions_from_frame, validate_sites
from .scenario import Expansion, Scenario

R_IDX = {r: i for i, r in enumerate(C.RESOURCES)}
MONTHS = np.arange(C.MAX_HORIZON + 1)


def period_label(month: int | float | None, fmt: str = "%b %Y") -> str | None:
    if month is None or not np.isfinite(month):
        return None
    return (C.AS_OF_PERIOD + int(month)).strftime(fmt)


# --- Core arithmetic --------------------------------------------------------

def growth_factor(annual_growth, months, density_growth: float = 0.0):
    """Compound growth: (1 + g) ^ (months / 12), optionally net of rack densification.

    For rack space, more kW per rack means fewer new racks per MW of new load, so
    rack demand grows at (1 + g) / (1 + d) - 1 per year.
    """
    g = np.asarray(annual_growth, dtype=float)
    base = (1.0 + g) / (1.0 + density_growth)
    return base[..., None] ** (np.asarray(months, dtype=float) / 12.0)


def project(
    demand0: np.ndarray,
    capacity0: np.ndarray,
    growth: np.ndarray,
    exp_site: np.ndarray,
    exp_res: np.ndarray,
    exp_amount: np.ndarray,
    exp_month: np.ndarray,
    efficiency: np.ndarray,
    density_growth: float,
    months: np.ndarray = MONTHS,
) -> tuple[np.ndarray, np.ndarray]:
    """Project demand and capacity for every draw, site, resource and month.

    demand0, capacity0 : (S, R) baseline values
    growth             : (N, S) annual growth rate per draw and site
    exp_site, exp_res, exp_amount : (E,) expansion site index, resource index, amount
    exp_month          : (N, E) activation month per draw (allows schedule slip)
    efficiency         : (N,) fractional reduction of power and cooling demand
    Returns (demand, capacity), each (N, S, R, M).
    """
    n, s = growth.shape
    r, m = demand0.shape[1], len(months)

    load = growth_factor(growth, months)  # (N, S, M)
    racks = growth_factor(growth, months, density_growth)
    # Efficiency programmes act on forecast months only; month 0 stays the actual baseline.
    eff = np.where(months[None, :] >= 1, 1.0 - efficiency[:, None], 1.0)  # (N, M)

    factor = np.empty((n, s, r, m))
    factor[:, :, R_IDX["power"]] = load * eff[:, None, :]
    factor[:, :, R_IDX["cooling"]] = load * eff[:, None, :]
    factor[:, :, R_IDX["space"]] = racks
    demand = demand0[None, :, :, None] * factor

    capacity = np.broadcast_to(capacity0[None, :, :, None], (n, s, r, m)).copy()
    for e in range(len(exp_amount)):
        active = months[None, :] >= exp_month[:, e][:, None]  # (N, M)
        capacity[:, exp_site[e], exp_res[e], :] += exp_amount[e] * active
    return demand, capacity


def first_month(mask: np.ndarray) -> np.ndarray:
    """Index of the first True along the last axis, or +inf where there is none."""
    hit = mask.any(axis=-1)
    return np.where(hit, mask.argmax(axis=-1).astype(float), np.inf)


def exhaustion_month(demand: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """First month where demand exceeds available capacity (SOW definition)."""
    return first_month(demand > capacity)


def threshold_month(demand: np.ndarray, capacity: np.ndarray, threshold: float) -> np.ndarray:
    """First month where utilisation reaches `threshold` (month 0 included)."""
    return first_month(demand >= threshold * capacity)


def classify_risk(
    exhaust: np.ndarray, warning: np.ndarray, current_util: np.ndarray, critical_threshold: float
) -> tuple[str, str]:
    """Rules-based site risk status from per-resource indicators (arrays of length R).

    Critical: any resource exhausted within 6 months
    High:     exhausted within 12 months, or currently at/above the critical threshold
    Medium:   exhausted within 24 months, or warning threshold reached within 12 months
    Low:      otherwise
    """
    names = [C.RESOURCE_LABELS[r].lower() for r in C.RESOURCES]
    e_min = float(np.min(exhaust))
    e_res = names[int(np.argmin(exhaust))]
    u_max = float(np.max(current_util))
    u_res = names[int(np.argmax(current_util))]
    w_min = float(np.min(warning))
    w_res = names[int(np.argmin(warning))]

    if e_min <= C.RISK_CRITICAL_MONTHS:
        return "Critical", f"{e_res} exhausted in month {e_min:.0f} (<= {C.RISK_CRITICAL_MONTHS})"
    if e_min <= C.RISK_HIGH_MONTHS:
        return "High", f"{e_res} exhausted in month {e_min:.0f} (<= {C.RISK_HIGH_MONTHS})"
    if u_max >= critical_threshold:
        return "High", f"{u_res} already at {u_max:.0%} (critical threshold {critical_threshold:.0%})"
    if e_min <= C.RISK_MEDIUM_MONTHS:
        return "Medium", f"{e_res} exhausted in month {e_min:.0f} (<= {C.RISK_MEDIUM_MONTHS})"
    if w_min <= C.RISK_WARNING_WINDOW:
        when = "now" if w_min == 0 else f"in month {w_min:.0f}"
        return "Medium", f"{w_res} reaches warning threshold {when}"
    return "Low", f"no resource exhausted within {C.MAX_HORIZON} months; warning threshold not reached within {C.RISK_WARNING_WINDOW}"


# --- Scenario assembly ------------------------------------------------------

def resolve_growth(sites: pd.DataFrame, scenario: Scenario) -> np.ndarray:
    g = sites["annual_growth_pct"].to_numpy(float) / 100.0
    if scenario.portfolio_growth is not None:
        g = np.full_like(g, scenario.portfolio_growth)
    ids = sites["site_id"].tolist()
    for sid, val in scenario.site_growth.items():
        g[ids.index(sid)] = val
    return g


def scenario_expansions(planned: pd.DataFrame | None, scenario: Scenario) -> list[Expansion]:
    exps = expansions_from_frame(planned) if (planned is not None and scenario.include_planned) else []
    exps += list(scenario.added_expansions)
    return [
        e if e.cost_usd is not None else replace(e, cost_usd=e.amount * scenario.unit_cost(e.resource))
        for e in exps
    ]


def expansion_arrays(exps: list[Expansion], site_ids: list[str]):
    site = np.array([site_ids.index(e.site_id) for e in exps], dtype=int)
    res = np.array([R_IDX[e.resource] for e in exps], dtype=int)
    amount = np.array([e.amount for e in exps], dtype=float)
    month = np.array([e.activation_month for e in exps], dtype=float)
    return site, res, amount, month


# --- Deterministic forecast -------------------------------------------------

@dataclass
class ForecastResult:
    scenario: Scenario
    sites: pd.DataFrame
    expansions: list[Expansion]
    growth: np.ndarray            # (S,)
    demand: np.ndarray            # (S, R, M)
    capacity: np.ndarray          # (S, R, M)
    site_resource: pd.DataFrame = field(init=False)
    site_summary: pd.DataFrame = field(init=False)
    monthly: pd.DataFrame = field(init=False)
    portfolio_monthly: pd.DataFrame = field(init=False)
    kpis: dict = field(init=False)

    def __post_init__(self):
        self.site_resource = self._site_resource()
        self.site_summary = self._site_summary()
        self.monthly = self._monthly()
        self.portfolio_monthly = self._portfolio_monthly()
        self.kpis = self._kpis()

    @property
    def site_ids(self) -> list[str]:
        return self.sites["site_id"].tolist()

    @property
    def horizon(self) -> int:
        return self.scenario.horizon

    def expansions_frame(self) -> pd.DataFrame:
        rows = [
            {
                "site_id": e.site_id, "project": e.label, "source": e.source, "resource": e.resource,
                "capacity_added": e.amount, "unit": C.RESOURCE_UNITS[e.resource],
                "activation_month": e.activation_month, "activation_period": period_label(e.activation_month),
                "cost_usd": e.cost_usd,
            }
            for e in self.expansions
        ]
        return pd.DataFrame(rows, columns=[
            "site_id", "project", "source", "resource", "capacity_added", "unit",
            "activation_month", "activation_period", "cost_usd",
        ])

    # -- tables --
    def _site_resource(self) -> pd.DataFrame:
        sc, h = self.scenario, self.horizon
        util = self.demand / self.capacity
        exhaust = exhaustion_month(self.demand, self.capacity)
        warn = threshold_month(self.demand, self.capacity, sc.warning_threshold)
        crit = threshold_month(self.demand, self.capacity, sc.critical_threshold)
        rows = []
        for i, sid in enumerate(self.site_ids):
            for j, res in enumerate(C.RESOURCES):
                rows.append({
                    "site_id": sid,
                    "site_name": self.sites["site_name"].iat[i],
                    "resource": res,
                    "unit": C.RESOURCE_UNITS[res],
                    "current_capacity": self.capacity[i, j, 0],
                    "current_demand": self.demand[i, j, 0],
                    "current_utilization": util[i, j, 0],
                    "current_headroom": self.capacity[i, j, 0] - self.demand[i, j, 0],
                    "horizon_capacity": self.capacity[i, j, h],
                    "horizon_demand": self.demand[i, j, h],
                    "horizon_utilization": util[i, j, h],
                    "horizon_headroom": self.capacity[i, j, h] - self.demand[i, j, h],
                    "peak_utilization": util[i, j, : h + 1].max(),
                    "warning_month": warn[i, j],
                    "critical_month": crit[i, j],
                    "exhaustion_month": exhaust[i, j],
                    "exhaustion_period": period_label(exhaust[i, j]),
                    "exhausted_within_horizon": bool(exhaust[i, j] <= h),
                })
        df = pd.DataFrame(rows)
        for c in ("warning_month", "critical_month", "exhaustion_month"):
            df[c] = df[c].replace(np.inf, np.nan).astype("Int64")
        return df

    def _site_summary(self) -> pd.DataFrame:
        sc, h = self.scenario, self.horizon
        exhaust = exhaustion_month(self.demand, self.capacity)
        warn = threshold_month(self.demand, self.capacity, sc.warning_threshold)
        util = self.demand / self.capacity
        rows = []
        for i, sid in enumerate(self.site_ids):
            status, reason = classify_risk(exhaust[i], warn[i], util[i, :, 0], sc.critical_threshold)
            limiting, basis = limiting_resource(exhaust[i], warn[i], util[i, :, h])
            e_min = float(exhaust[i].min())
            w_min = float(warn[i].min())
            rows.append({
                "site_id": sid,
                "site_name": self.sites["site_name"].iat[i],
                "region": self.sites["region"].iat[i],
                "annual_growth": self.growth[i],
                "power_utilization": util[i, R_IDX["power"], 0],
                "cooling_utilization": util[i, R_IDX["cooling"], 0],
                "space_utilization": util[i, R_IDX["space"], 0],
                "max_current_utilization": util[i, :, 0].max(),
                "limiting_resource": limiting,
                "limiting_basis": basis,
                "earliest_warning_month": w_min,
                "earliest_exhaustion_month": e_min,
                "earliest_exhaustion_period": period_label(e_min),
                "exhausted_within_horizon": e_min <= h,
                "risk_status": status,
                "risk_reason": reason,
            })
        df = pd.DataFrame(rows)
        for c in ("earliest_warning_month", "earliest_exhaustion_month"):
            df[c] = df[c].replace(np.inf, np.nan).astype("Int64")
        return df

    def _monthly(self) -> pd.DataFrame:
        h = self.horizon
        s, r, _ = self.demand.shape
        months = MONTHS[: h + 1]
        si, ri, mi = np.meshgrid(np.arange(s), np.arange(r), months, indexing="ij")
        si, ri, mi = si.ravel(), ri.ravel(), mi.ravel()
        demand = self.demand[si, ri, mi]
        capacity = self.capacity[si, ri, mi]
        ids = np.array(self.site_ids)
        names = self.sites["site_name"].to_numpy()
        res = np.array(C.RESOURCES)
        return pd.DataFrame({
            "site_id": ids[si],
            "site_name": names[si],
            "resource": res[ri],
            "unit": [C.RESOURCE_UNITS[x] for x in res[ri]],
            "month": mi,
            "period": [period_label(m, "%Y-%m") for m in mi],
            "data_type": np.where(mi == 0, "Actual (synthetic baseline)", "Projected"),
            "demand": demand,
            "capacity": capacity,
            "headroom": capacity - demand,
            "utilization": demand / capacity,
        })

    def _portfolio_monthly(self) -> pd.DataFrame:
        h = self.horizon
        dem = self.demand.sum(axis=0)  # (R, M)
        cap = self.capacity.sum(axis=0)
        rows = []
        for j, res in enumerate(C.RESOURCES):
            for m in range(h + 1):
                rows.append({
                    "resource": res, "unit": C.RESOURCE_UNITS[res], "month": m,
                    "period": period_label(m, "%Y-%m"),
                    "data_type": "Actual (synthetic baseline)" if m == 0 else "Projected",
                    "demand": dem[j, m], "capacity": cap[j, m],
                    "headroom": cap[j, m] - dem[j, m], "utilization": dem[j, m] / cap[j, m],
                })
        return pd.DataFrame(rows)

    def _kpis(self) -> dict:
        h = self.horizon
        p, c, sp = R_IDX["power"], R_IDX["cooling"], R_IDX["space"]
        dem0 = self.demand[:, :, 0].sum(axis=0)
        cap0 = self.capacity[:, :, 0].sum(axis=0)
        sr = self.site_resource
        approaching = sr.loc[sr["warning_month"].fillna(10**6) <= h, "site_id"].unique()
        portfolio_exhaust = exhaustion_month(self.demand.sum(axis=0), self.capacity.sum(axis=0))
        worst = max(self.site_summary["risk_status"], key=C.RISK_LEVELS.index)
        return {
            "power_capacity_mw": cap0[p],
            "power_demand_mw": dem0[p],
            "power_utilization": dem0[p] / cap0[p],
            "power_headroom_mw": cap0[p] - dem0[p],
            "cooling_capacity_mwt": cap0[c],
            "cooling_load_mwt": dem0[c],
            "cooling_utilization": dem0[c] / cap0[c],
            "rack_capacity": cap0[sp],
            "racks_occupied": dem0[sp],
            "rack_utilization": dem0[sp] / cap0[sp],
            "power_demand_12m_mw": self.demand[:, p, 12].sum(),
            "power_demand_24m_mw": self.demand[:, p, 24].sum(),
            "power_demand_36m_mw": self.demand[:, p, 36].sum(),
            "sites_approaching_constraint": len(approaching),
            "sites_approaching_ids": sorted(approaching),
            "earliest_constraint": earliest_constraint(sr),
            "portfolio_power_exhaustion_month": _finite_or_none(portfolio_exhaust[p]),
            "portfolio_exhaustion_month": {r: _finite_or_none(portfolio_exhaust[k]) for k, r in enumerate(C.RESOURCES)},
            "portfolio_risk": worst,
        }


def _finite_or_none(x: float) -> int | None:
    return int(x) if np.isfinite(x) else None


def limiting_resource(exhaust: np.ndarray, warning: np.ndarray, util_h: np.ndarray) -> tuple[str, str]:
    """First limiting resource for a site.

    Earliest exhaustion; if nothing exhausts, earliest warning breach; otherwise the
    resource with the highest utilisation at the horizon. Ties go to the higher
    utilisation at the horizon.
    """
    for arr, basis in ((exhaust, "exhaustion"), (warning, "warning threshold")):
        if np.isfinite(arr).any():
            cand = np.flatnonzero(arr == arr.min())
            j = cand[np.argmax(util_h[cand])]
            return C.RESOURCES[j], basis
    return C.RESOURCES[int(np.argmax(util_h))], "highest utilization"


def earliest_constraint(site_resource: pd.DataFrame) -> dict | None:
    """Earliest site-level exhaustion (or warning breach if none) across the portfolio."""
    for col, kind in (("exhaustion_month", "exhaustion"), ("warning_month", "warning threshold")):
        hits = site_resource.dropna(subset=[col])
        if len(hits):
            row = hits.sort_values([col, "horizon_utilization"], ascending=[True, False]).iloc[0]
            m = int(row[col])
            return {
                "kind": kind, "site_id": row["site_id"], "site_name": row["site_name"],
                "resource": row["resource"], "month": m, "period": period_label(m),
            }
    return None


def run_forecast(
    sites: pd.DataFrame, planned: pd.DataFrame | None, scenario: Scenario, validate: bool = True
) -> ForecastResult:
    site_ids = sites["site_id"].tolist()
    if validate:
        validate_sites(sites)
        scenario.validate(site_ids)
    growth = resolve_growth(sites, scenario)
    exps = scenario_expansions(planned, scenario)
    demand0, capacity0 = baseline_arrays(sites)
    e_site, e_res, e_amt, e_month = expansion_arrays(exps, site_ids)
    demand, capacity = project(
        demand0, capacity0, growth[None, :], e_site, e_res, e_amt, e_month[None, :],
        np.array([scenario.efficiency]), scenario.density_growth,
    )
    return ForecastResult(scenario, sites, exps, growth, demand[0], capacity[0])
