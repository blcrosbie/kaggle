"""Invalid or physically inconsistent inputs must be rejected with clear messages."""

import pytest

from conftest import make_sites, no_expansions
from dcsim.data import validate_sites
from dcsim.forecast import run_forecast
from dcsim.montecarlo import MCSettings
from dcsim.scenario import Expansion, baseline
from dcsim.validation import ValidationError


@pytest.mark.parametrize(
    "overrides, fragment",
    [
        ({"power_capacity_mw": [-5.0] + [10.0] * 5}, "cannot be negative"),
        ({"power_demand_mw": [12.0] + [8.0] * 5}, "exceeds installed capacity"),
        ({"racks_occupied": [150] + [50] * 5}, "exceeds installed capacity"),
        ({"cooling_capacity_mwt": [0.0] + [10.0] * 5}, "greater than zero"),
        ({"pue": [0.8] + [1.5] * 5}, "PUE"),
        ({"annual_growth_pct": [250.0] + [0.0] * 5}, "Annual growth"),
        ({"site_id": ["DC1"] * 6}, "Duplicate"),
    ],
)
def test_bad_site_data_rejected(overrides, fragment):
    with pytest.raises(ValidationError) as exc:
        validate_sites(make_sites(**overrides))
    assert any(fragment in e for e in exc.value.errors)


def test_wrong_number_of_sites():
    with pytest.raises(ValidationError, match="Expected 6"):
        validate_sites(make_sites().iloc[:5])


@pytest.mark.parametrize(
    "scenario, fragment",
    [
        (baseline(horizon=7), "horizon"),
        (baseline(efficiency=0.6), "Efficiency"),
        (baseline(portfolio_growth=1.5), "growth"),
        (baseline(warning_threshold=0.95, critical_threshold=0.9), "Thresholds"),
        (baseline(budget_usd=-1.0), "budget"),
        (baseline(cost_per_mw=-1.0), "cost per MW"),
        (baseline(site_growth={"DC9": 0.1}), "unknown site"),
        (baseline(added_expansions=(Expansion("DC1", "power", -2.0, 6),)), "positive"),
        (baseline(added_expansions=(Expansion("DC1", "power", 2.0, 0),)), "activation month"),
        (baseline(added_expansions=(Expansion("DC1", "water", 2.0, 6),)), "resource"),
    ],
)
def test_bad_scenarios_rejected(scenario, fragment):
    with pytest.raises(ValidationError) as exc:
        run_forecast(make_sites(), no_expansions(), scenario)
    assert any(fragment.lower() in e.lower() for e in exc.value.errors)


def test_all_errors_reported_together():
    with pytest.raises(ValidationError) as exc:
        baseline(horizon=7, efficiency=0.9, budget_usd=-1).validate()
    assert len(exc.value.errors) == 3


def test_bad_mc_settings():
    with pytest.raises(ValidationError):
        MCSettings(n_draws=10).validate()
    with pytest.raises(ValidationError):
        MCSettings(slip_mode=10, slip_max=5).validate()
