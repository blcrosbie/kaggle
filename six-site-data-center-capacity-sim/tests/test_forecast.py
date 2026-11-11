"""Forecast engine against the SOW formulas and documented hand-worked cases."""

import numpy as np
import pytest

from conftest import make_sites, no_expansions
from dcsim import config as C
from dcsim.forecast import classify_risk, growth_factor, run_forecast
from dcsim.scenario import Expansion, Scenario, baseline

P, CL, SP = (C.RESOURCES.index(r) for r in ("power", "cooling", "space"))


def test_case_1_compound_growth_formula():
    """TC-1: 10 MW at 12%/yr -> month 24 = 10 x 1.12^2 = 12.544 MW; month 6 = 10 x 1.12^0.5."""
    sites = make_sites(power_capacity_mw=20.0, power_demand_mw=10.0, annual_growth_pct=12.0)
    r = run_forecast(sites, no_expansions(), Scenario(density_growth=0.0))
    assert r.demand[0, P, 24] == pytest.approx(12.544)
    assert r.demand[0, P, 6] == pytest.approx(10 * 1.12 ** 0.5)
    assert r.demand[0, P, 0] == pytest.approx(10.0)


def test_case_2_utilization_and_headroom():
    """TC-2: Utilization = demand / capacity; Headroom = capacity - demand."""
    r = run_forecast(make_sites(), no_expansions(), baseline())
    row = r.site_resource.query("site_id == 'DC1' and resource == 'power'").iloc[0]
    assert row.current_utilization == pytest.approx(0.8)
    assert row.current_headroom == pytest.approx(2.0)


def test_case_3_exhaustion_is_first_month_demand_exceeds_capacity():
    """TC-3: 8 MW on 10 MW at 25%/yr. Month 12 demand == 10.0 exactly (not exceeding),
    so exhaustion is month 13, the first month demand is strictly greater."""
    sites = make_sites(annual_growth_pct=25.0)
    r = run_forecast(sites, no_expansions(), baseline())
    assert r.demand[0, P, 12] == pytest.approx(10.0)
    sr = r.site_resource.query("site_id == 'DC1' and resource == 'power'").iloc[0]
    assert sr.exhaustion_month == 13
    assert sr.exhaustion_period == "Nov 2027"


def test_no_exhaustion_reported_as_missing():
    r = run_forecast(make_sites(), no_expansions(), baseline())
    assert r.site_resource["exhaustion_month"].isna().all()


def test_case_4_planned_expansion_activates_on_its_month():
    """TC-4: +5 MW live from month 10 -> capacity 10 in month 9, 15 in month 10; exhaustion pushed out."""
    sites = make_sites(annual_growth_pct=25.0)
    sc = baseline(added_expansions=(Expansion("DC1", "power", 5.0, 10),))
    r = run_forecast(sites, no_expansions(), sc)
    assert r.capacity[0, P, 9] == 10.0
    assert r.capacity[0, P, 10] == 15.0
    # 8 x 1.25^(m/12) > 15  ->  m > 12 x ln(1.875)/ln(1.25) = 33.8 -> month 34
    assert r.site_resource.query("site_id=='DC1' and resource=='power'").exhaustion_month.iloc[0] == 34
    assert r.site_resource.query("site_id=='DC2' and resource=='power'").exhaustion_month.iloc[0] == 13


def test_efficiency_reduces_power_and_cooling_but_not_racks_or_baseline():
    sites = make_sites(annual_growth_pct=10.0)
    base = run_forecast(sites, no_expansions(), baseline())
    eff = run_forecast(sites, no_expansions(), baseline(efficiency=0.10))
    assert eff.demand[0, P, 0] == base.demand[0, P, 0]  # actual baseline untouched
    assert eff.demand[0, P, 12] == pytest.approx(0.9 * base.demand[0, P, 12])
    assert eff.demand[0, CL, 12] == pytest.approx(0.9 * base.demand[0, CL, 12])
    np.testing.assert_allclose(eff.demand[:, SP], base.demand[:, SP])


def test_rack_densification():
    """Racks grow at (1+g)/(1+d) - 1: 50 racks, g=10%, d=10% -> flat."""
    sites = make_sites(annual_growth_pct=10.0)
    r = run_forecast(sites, no_expansions(), baseline(density_growth=0.10))
    np.testing.assert_allclose(r.demand[0, SP], 50.0)
    np.testing.assert_allclose(growth_factor(0.10, [12], 0.0), [[1.10]][0])


def test_portfolio_utilization_is_sum_over_sum():
    sites = make_sites(power_capacity_mw=[10, 20, 30, 40, 50, 60], power_demand_mw=[5, 5, 5, 5, 5, 5])
    r = run_forecast(sites, no_expansions(), baseline())
    assert r.kpis["power_utilization"] == pytest.approx(30 / 210)
    assert r.kpis["power_headroom_mw"] == pytest.approx(180)


def test_site_constraint_not_hidden_by_portfolio(real_sites, real_planned):
    """The portfolio never exhausts power in the baseline, but DC3 does - and must be surfaced."""
    r = run_forecast(real_sites, real_planned, baseline())
    assert r.kpis["portfolio_power_exhaustion_month"] is None
    ec = r.kpis["earliest_constraint"]
    assert (ec["site_id"], ec["resource"], ec["kind"]) == ("DC3", "power", "exhaustion")
    assert ec["month"] == r.site_resource["exhaustion_month"].min()


def test_first_limiting_resource():
    # Cooling at 95% exhausts before power at 80% (both growing 10%).
    sites = make_sites(annual_growth_pct=10.0, cooling_load_mwt=[9.5] + [5.0] * 5)
    r = run_forecast(sites, no_expansions(), baseline())
    s = r.site_summary.set_index("site_id")
    assert s.loc["DC1", "limiting_resource"] == "cooling"
    assert s.loc["DC1", "limiting_basis"] == "exhaustion"
    assert s.loc["DC2", "limiting_resource"] == "power"


@pytest.mark.parametrize(
    "exhaust, warning, util, expected",
    [
        ([5, np.inf, np.inf], [0, 0, 0], [0.5, 0.5, 0.5], "Critical"),
        ([12, np.inf, np.inf], [0, 0, 0], [0.5, 0.5, 0.5], "High"),
        ([np.inf] * 3, [np.inf] * 3, [0.95, 0.5, 0.5], "High"),
        ([24, np.inf, np.inf], [np.inf] * 3, [0.5, 0.5, 0.5], "Medium"),
        ([np.inf] * 3, [12, np.inf, np.inf], [0.5, 0.5, 0.5], "Medium"),
        ([np.inf] * 3, [13, np.inf, np.inf], [0.5, 0.5, 0.5], "Low"),
        ([25, np.inf, np.inf], [np.inf] * 3, [0.5, 0.5, 0.5], "Low"),
    ],
)
def test_risk_rules(exhaust, warning, util, expected):
    status, reason = classify_risk(np.array(exhaust, float), np.array(warning, float), np.array(util), 0.90)
    assert status == expected
    assert reason


def test_horizon_changes_display_not_risk(real_sites, real_planned):
    short = run_forecast(real_sites, real_planned, baseline(horizon=6))
    full = run_forecast(real_sites, real_planned, baseline(horizon=36))
    assert short.site_summary["risk_status"].tolist() == full.site_summary["risk_status"].tolist()
    assert short.monthly["month"].max() == 6
    assert full.monthly["month"].max() == 36


def test_actual_vs_projected_labelled(real_sites, real_planned):
    m = run_forecast(real_sites, real_planned, baseline()).monthly
    assert set(m.loc[m.month == 0, "data_type"]) == {"Actual (synthetic baseline)"}
    assert set(m.loc[m.month > 0, "data_type"]) == {"Projected"}
