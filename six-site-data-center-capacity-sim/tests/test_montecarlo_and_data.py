"""Monte Carlo consistency with the deterministic engine, and dataset reproducibility."""

import numpy as np
import pandas as pd
import pytest

from dcsim import config as C
from dcsim.data import SITES_CSV, EXPANSIONS_CSV
from dcsim.forecast import run_forecast
from dcsim.generate import DEFAULT_SEED, generate
from dcsim.montecarlo import MCSettings, run_monte_carlo
from dcsim.scenario import baseline, efficiency_optimization

NO_UNCERTAINTY = MCSettings(n_draws=100, growth_sd=0.0, slip_mode=0.0, slip_max=0.0, eff_realization_low=1.0)


@pytest.mark.parametrize("scenario", [baseline(), efficiency_optimization()])
def test_zero_uncertainty_reproduces_deterministic(real_sites, real_planned, scenario):
    det = run_forecast(real_sites, real_planned, scenario)
    mc = run_monte_carlo(real_sites, real_planned, scenario, NO_UNCERTAINTY)
    np.testing.assert_allclose(mc.demand, np.broadcast_to(det.demand, mc.demand.shape))
    np.testing.assert_allclose(mc.capacity, np.broadcast_to(det.capacity, mc.capacity.shape))
    s = mc.exhaustion_summary()
    pd.testing.assert_series_equal(s["p50_month"], s["deterministic_month"], check_names=False)


def test_seeded_runs_are_reproducible(real_sites, real_planned):
    a = run_monte_carlo(real_sites, real_planned, baseline(), MCSettings(n_draws=500, seed=7))
    b = run_monte_carlo(real_sites, real_planned, baseline(), MCSettings(n_draws=500, seed=7))
    np.testing.assert_array_equal(a.exhaust, b.exhaust)


def test_exhaustion_curve_is_monotone(real_sites, real_planned):
    curve = run_monte_carlo(real_sites, real_planned, baseline(), MCSettings(n_draws=500)).exhaustion_curve()
    for _, g in curve.groupby("site_id"):
        assert g.sort_values("month")["probability"].is_monotonic_increasing


def test_schedule_slip_only_adds_risk(real_sites, real_planned):
    """Projects never land early, so slip can only raise the chance of exhaustion."""
    no_slip = run_monte_carlo(real_sites, real_planned, baseline(), MCSettings(n_draws=2000, slip_max=0, slip_mode=0))
    slip = run_monte_carlo(real_sites, real_planned, baseline(), MCSettings(n_draws=2000))
    assert (slip.site_summary()["prob_within_36m"] >= no_slip.site_summary()["prob_within_36m"] - 1e-12).all()


def test_generator_reproducible_and_matches_committed_csv():
    s1, e1 = generate(DEFAULT_SEED)
    s2, _ = generate(DEFAULT_SEED)
    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(s1, pd.read_csv(SITES_CSV), check_dtype=False)
    pd.testing.assert_frame_equal(e1, pd.read_csv(EXPANSIONS_CSV), check_dtype=False)


def test_generated_values_physically_consistent():
    sites, _ = generate(DEFAULT_SEED)
    it_load = sites["power_demand_mw"] / sites["pue"]
    np.testing.assert_allclose(sites["cooling_load_mwt"], it_load * C.COOLING_HEAT_FACTOR, atol=0.01)
    assert sites["site_id"].tolist() == [f"DC{i}" for i in range(1, 7)]
