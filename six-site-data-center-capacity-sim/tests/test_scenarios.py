"""Predefined scenarios and capital planning behave as specified."""

import numpy as np
import pytest

from conftest import make_sites, no_expansions
from dcsim import config as C
from dcsim import scenario as S
from dcsim.capital import plan_capital
from dcsim.forecast import run_forecast

P = C.RESOURCES.index("power")


def test_reset_to_baseline_is_default_scenario():
    assert S.baseline() == S.Scenario()


def test_high_growth_sets_every_site_to_25pct(real_sites, real_planned):
    r = run_forecast(real_sites, real_planned, S.high_growth())
    np.testing.assert_allclose(r.growth, 0.25)


def test_site_override_beats_portfolio_growth(real_sites, real_planned):
    r = run_forecast(real_sites, real_planned, S.high_growth(site_growth={"DC1": 0.05}))
    assert r.growth[0] == 0.05
    np.testing.assert_allclose(r.growth[1:], 0.25)


def test_high_growth_is_worse_than_baseline(real_sites, real_planned):
    base = run_forecast(real_sites, real_planned, S.baseline())
    high = run_forecast(real_sites, real_planned, S.high_growth())
    assert high.kpis["power_demand_36m_mw"] > base.kpis["power_demand_36m_mw"]
    order = C.RISK_LEVELS.index
    assert order(high.kpis["portfolio_risk"]) >= order(base.kpis["portfolio_risk"])


def test_efficiency_scenario_reduces_modeled_demand_10pct(real_sites, real_planned):
    base = run_forecast(real_sites, real_planned, S.baseline())
    eff = run_forecast(real_sites, real_planned, S.efficiency_optimization())
    assert eff.kpis["power_demand_12m_mw"] == pytest.approx(0.9 * base.kpis["power_demand_12m_mw"])


def test_infrastructure_expansion_adds_capacity_at_selected_site(real_sites, real_planned):
    sc = S.infrastructure_expansion("DC3", power_mw=8, cooling_mwt=4, racks=200, activation_month=6)
    base = run_forecast(real_sites, real_planned, S.baseline())
    exp = run_forecast(real_sites, real_planned, sc)
    i = exp.site_ids.index("DC3")
    assert exp.capacity[i, P, 6] - base.capacity[i, P, 6] == 8
    assert exp.capacity[i, P, 5] == base.capacity[i, P, 5]
    # other sites untouched
    np.testing.assert_allclose(np.delete(exp.capacity, i, 0), np.delete(base.capacity, i, 0))
    # and the cost is priced from unit costs
    f = exp.expansions_frame().query("source == 'scenario'")
    assert f["cost_usd"].sum() == pytest.approx(8 * sc.cost_per_mw + 4 * sc.cost_per_mwt + 200 * sc.cost_per_rack)


def test_excluding_planned_expansions(real_sites, real_planned):
    without = run_forecast(real_sites, real_planned, S.baseline(include_planned=False))
    assert without.expansions == []


@pytest.mark.parametrize("budget", [0, 10e6, 60e6, 150e6])
def test_budget_never_exceeded(real_sites, real_planned, budget):
    plan = plan_capital(real_sites, real_planned, S.budget_constrained(budget))
    assert plan.funded_total <= budget + 1e-6
    if budget == 0:
        assert not plan.projects["funded"].any()


def test_unlimited_budget_restores_headroom_by_horizon(real_sites, real_planned):
    plan = plan_capital(real_sites, real_planned, S.baseline())  # budget None = fund everything
    assert plan.projects["funded"].all()
    after = plan.after.site_resource
    assert (after["horizon_utilization"] <= plan.scenario.warning_threshold + 1e-9).all()


def test_project_sizing_hand_case():
    """8 MW on 10 MW, 25%/yr, 12-month horizon, 80% target:
    peak demand month 12 = 10 MW -> need 12.5 MW -> +2.5 MW, priced at $3.5M/MW."""
    sites = make_sites(annual_growth_pct=25.0)
    plan = plan_capital(sites, no_expansions(), S.baseline(horizon=12, density_growth=0.0))
    p = plan.projects.query("site_id == 'DC1' and resource == 'power'").iloc[0]
    assert p.capacity_added == 2.5
    assert p.cost_usd == pytest.approx(2.5 * 3_500_000)
    assert p.need_month == 0  # already at the 80% target
    assert p.activation_month == C.DEFAULT_LEAD_TIME["power"]
    assert bool(p.arrives_late)


def test_ranking_is_ordered_and_scores_bounded(real_sites, real_planned):
    p = plan_capital(real_sites, real_planned, S.budget_constrained(60e6)).projects
    assert p["priority_score"].is_monotonic_decreasing
    assert p["rank"].tolist() == list(range(1, len(p) + 1))
    for c in ("urgency_score", "risk_score", "cost_score", "priority_score"):
        assert p[c].between(0, 100).all()
