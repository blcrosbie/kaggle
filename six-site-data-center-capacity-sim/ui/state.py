"""Session state: scenario levers, predefined scenarios, and the computed context.

Levers live in st.session_state under the keys in DEFAULTS. Every page builds
the same Scenario from them and runs the same engine, so KPIs stay consistent
across pages.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from dcsim import config as C
from dcsim import scenario as S
from dcsim.capital import CapitalPlan, plan_capital, size_projects
from dcsim.data import load_planned_expansions, load_sites
from dcsim.forecast import ForecastResult
from dcsim.scenario import Expansion, Scenario
from dcsim.validation import ValidationError

PRESETS = ("Baseline", "High Growth", "Infrastructure Expansion", "Efficiency Optimization", "Budget-Constrained")

DEFAULTS = {
    "preset": "Baseline",
    # planning assumptions (view settings - presets keep these)
    "horizon": C.MAX_HORIZON,
    "warn_pct": int(C.DEFAULT_WARNING_THRESHOLD * 100),
    "crit_pct": int(C.DEFAULT_CRITICAL_THRESHOLD * 100),
    # scenario levers
    "pg_on": False,
    "pg_pct": 15.0,
    "eff_pct": 0,
    "include_planned": True,
    "budget_on": False,
    "budget_m": 60.0,
    "exp_site": "DC3",
    "exp_power": 0.0,
    "exp_cooling": 0.0,
    "exp_racks": 0,
    "exp_month": 12,
}
VIEW_KEYS = ("horizon", "warn_pct", "crit_pct")


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    sites = load_sites()
    return sites, load_planned_expansions(site_ids=sites["site_id"].tolist())


def ensure_state() -> None:
    """Initialise defaults and keep lever values alive across page switches.

    Re-assigning each key detaches it from widget clean-up, so a lever set on the
    Scenario Planning page survives a visit to the Overview page.
    """
    ss = st.session_state
    for k, v in DEFAULTS.items():
        ss[k] = ss[k] if k in ss else v


def _reset_levers(keep_view: bool = True) -> None:
    ss = st.session_state
    for k, v in DEFAULTS.items():
        if keep_view and k in VIEW_KEYS:
            continue
        ss[k] = v


def apply_preset(name: str) -> None:
    """Callback for the predefined scenario buttons."""
    ss = st.session_state
    _reset_levers(keep_view=True)
    ss["preset"] = name
    if name == "High Growth":
        ss["pg_on"], ss["pg_pct"] = True, 25.0
    elif name == "Efficiency Optimization":
        ss["eff_pct"] = 10
    elif name == "Budget-Constrained":
        ss["budget_on"], ss["budget_m"] = True, 60.0
    elif name == "Infrastructure Expansion":
        _prefill_expansion()


def _prefill_expansion() -> None:
    """Pre-size the expansion for the site with the earliest constraint (user can edit)."""
    ss = st.session_state
    sites, planned = load_data()
    base = plan_capital(sites, planned, _build_scenario(name="Baseline"))
    ec = base.before.kpis["earliest_constraint"]
    sid = ec["site_id"] if ec else "DC3"
    p = base.projects[base.projects["site_id"] == sid].set_index("resource")
    ss["exp_site"] = sid
    ss["exp_power"] = float(p.loc["power", "capacity_added"]) if "power" in p.index else 0.0
    ss["exp_cooling"] = float(p.loc["cooling", "capacity_added"]) if "cooling" in p.index else 0.0
    ss["exp_racks"] = int(p.loc["space", "capacity_added"]) if "space" in p.index else 0
    ss["exp_month"] = int(p["activation_month"].max()) if len(p) else 12


def reset_to_baseline() -> None:
    _reset_levers(keep_view=False)


def mark_custom() -> None:
    """on_change for scenario levers: the scenario no longer matches a preset."""
    st.session_state["preset"] = "Custom"


def _build_scenario(name: str | None = None) -> Scenario:
    ss = st.session_state
    adds = tuple(
        Expansion(ss["exp_site"], res, float(amt), int(ss["exp_month"]),
                  label=f"{ss['exp_site']} {C.RESOURCE_LABELS[res].lower()} expansion (scenario)")
        for res, amt in (("power", ss["exp_power"]), ("cooling", ss["exp_cooling"]), ("space", ss["exp_racks"]))
        if amt and amt > 0
    )
    return Scenario(
        name=name or ss["preset"],
        horizon=int(ss["horizon"]),
        portfolio_growth=ss["pg_pct"] / 100 if ss["pg_on"] else None,
        efficiency=ss["eff_pct"] / 100,
        include_planned=bool(ss["include_planned"]),
        added_expansions=adds,
        budget_usd=ss["budget_m"] * 1e6 if ss["budget_on"] else None,
        warning_threshold=ss["warn_pct"] / 100,
        critical_threshold=ss["crit_pct"] / 100,
    )


@dataclass
class Context:
    scenario: Scenario
    sites: pd.DataFrame
    planned: pd.DataFrame
    result: ForecastResult     # what every page displays
    plan: CapitalPlan          # capital plan (budget applied if set)
    needs: pd.DataFrame        # expansion each site/resource still needs, sized on `result`


def context() -> Context:
    """Build the scenario and run the engine. Invalid inputs stop the page with clear messages."""
    sites, planned = load_data()
    sc = _build_scenario()
    try:
        plan = plan_capital(sites, planned, sc)
    except ValidationError as e:
        st.error("**These inputs can't be modeled:**\n\n" + "\n".join(f"- {m}" for m in e.errors))
        st.stop()
    result = plan.after if sc.budget_usd is not None else plan.before
    return Context(sc, sites, planned, result, plan, size_projects(result))
