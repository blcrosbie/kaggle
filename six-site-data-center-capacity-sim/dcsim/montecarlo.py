"""Monte Carlo layer: the same projection engine, run across thousands of plausible futures.

The deterministic forecast answers "when does DC3 run out of power if every
assumption is exactly right?". This module answers "how likely is it, and how
early could it happen?" by sampling three uncertainties:

1. Demand growth - each site's annual growth is drawn around its assumption.
   Sites share a common regional/market shock (correlation `growth_corr`).
2. Expansion schedule slip - each project's activation month slips by a
   triangular(0, mode, max) number of months (projects rarely land early).
3. Efficiency realisation - only a fraction of a targeted efficiency gain is
   achieved, drawn uniformly between `eff_realization_low` and 1.

The deterministic numbers remain the official SOW results; Monte Carlo output is
an additional risk view and is always labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .data import baseline_arrays
from .forecast import (
    MONTHS, exhaustion_month, expansion_arrays, period_label, project,
    resolve_growth, run_forecast, scenario_expansions,
)
from .scenario import Scenario


@dataclass(frozen=True)
class MCSettings:
    n_draws: int = 2000
    seed: int = 42
    growth_sd: float = 0.04          # std-dev of annual growth, absolute (4 pp)
    growth_corr: float = 0.5         # correlation of growth shocks between sites
    slip_mode: float = 1.0           # most likely schedule slip (months)
    slip_max: float = 9.0            # worst-case slip (months)
    eff_realization_low: float = 0.5  # share of the efficiency target achieved, worst case

    def validate(self) -> None:
        from .validation import ValidationError

        errs = []
        if not 100 <= self.n_draws <= 50_000:
            errs.append("Number of simulations must be between 100 and 50,000.")
        if not 0 <= self.growth_sd <= 0.25:
            errs.append("Growth uncertainty must be between 0 and 25 percentage points.")
        if not 0 <= self.growth_corr <= 1:
            errs.append("Growth correlation must be between 0 and 1.")
        if not 0 <= self.slip_mode <= self.slip_max <= 24:
            errs.append("Schedule slip must satisfy 0 <= most likely <= worst case <= 24 months.")
        if not 0 <= self.eff_realization_low <= 1:
            errs.append("Efficiency realisation must be between 0% and 100%.")
        if errs:
            raise ValidationError(errs)


@dataclass
class MCResult:
    scenario: Scenario
    settings: MCSettings
    site_ids: list[str]
    site_names: list[str]
    growth: np.ndarray       # (N, S) sampled growth
    demand: np.ndarray       # (N, S, R, M)
    capacity: np.ndarray     # (N, S, R, M)
    exhaust: np.ndarray      # (N, S, R) first exhaustion month, inf if none
    deterministic: pd.DataFrame = field(repr=False)  # site_resource table from run_forecast

    @property
    def site_exhaust(self) -> np.ndarray:
        """(N, S) earliest exhaustion across resources for each draw."""
        return self.exhaust.min(axis=2)

    def exhaustion_summary(self) -> pd.DataFrame:
        """P10 / P50 / P90 exhaustion month and probability of exhaustion, per site-resource."""
        rows = []
        det = self.deterministic.set_index(["site_id", "resource"])["exhaustion_month"]
        for i, sid in enumerate(self.site_ids):
            for j, res in enumerate(C.RESOURCES):
                x = self.exhaust[:, i, j]
                q10, q50, q90 = _quantiles(x)
                rows.append({
                    "site_id": sid, "site_name": self.site_names[i], "resource": res,
                    "deterministic_month": det.loc[(sid, res)],
                    "p10_month": q10, "p50_month": q50, "p90_month": q90,
                    "prob_within_12m": float(np.mean(x <= 12)),
                    "prob_within_24m": float(np.mean(x <= 24)),
                    "prob_within_36m": float(np.mean(x <= 36)),
                })
        df = pd.DataFrame(rows)
        for c in ("deterministic_month", "p10_month", "p50_month", "p90_month"):
            df[c] = df[c].astype("Int64")
        return df

    def site_summary(self) -> pd.DataFrame:
        """Probability that *any* resource at each site is exhausted, by horizon."""
        x = self.site_exhaust
        rows = []
        for i, sid in enumerate(self.site_ids):
            q10, q50, q90 = _quantiles(x[:, i])
            rows.append({
                "site_id": sid, "site_name": self.site_names[i],
                "p10_month": q10, "p50_month": q50, "p90_month": q90,
                **{f"prob_within_{h}m": float(np.mean(x[:, i] <= h)) for h in (6, 12, 18, 24, 30, 36)},
            })
        df = pd.DataFrame(rows)
        for c in ("p10_month", "p50_month", "p90_month"):
            df[c] = df[c].astype("Int64")
        return df

    def exhaustion_curve(self) -> pd.DataFrame:
        """Cumulative probability each site has hit any exhaustion by month m (long format)."""
        x = self.site_exhaust
        m = MONTHS
        probs = (x[:, :, None] <= m[None, None, :]).mean(axis=0)  # (S, M)
        return pd.DataFrame({
            "site_id": np.repeat(self.site_ids, len(m)),
            "month": np.tile(m, len(self.site_ids)),
            "probability": probs.ravel(),
        })

    def demand_bands(self, resource: str = "power", site_id: str | None = None,
                     quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
        """Fan-chart bands of demand (and median capacity) per month, for a site or the portfolio."""
        j = C.RESOURCES.index(resource)
        if site_id is None:
            d = self.demand[:, :, j, :].sum(axis=1)
            cap = self.capacity[:, :, j, :].sum(axis=1)
        else:
            i = self.site_ids.index(site_id)
            d, cap = self.demand[:, i, j, :], self.capacity[:, i, j, :]
        out = pd.DataFrame({"month": MONTHS, "period": [period_label(m, "%Y-%m") for m in MONTHS]})
        for q in quantiles:
            out[f"demand_p{int(q * 100):02d}"] = np.quantile(d, q, axis=0)
        out["capacity_p50"] = np.quantile(cap, 0.5, axis=0)
        out["capacity_p10"] = np.quantile(cap, 0.1, axis=0)  # capacity is lower when projects slip
        return out


def _quantiles(x: np.ndarray, qs=(0.1, 0.5, 0.9)) -> list:
    """Empirical quantiles that tolerate +inf ('not exhausted within 36 months') -> NA."""
    vals = np.quantile(x, qs, method="inverted_cdf")
    return [int(v) if np.isfinite(v) else pd.NA for v in vals]


def sample_inputs(scenario: Scenario, sites: pd.DataFrame, planned: pd.DataFrame | None, settings: MCSettings):
    rng = np.random.default_rng(settings.seed)
    n, s = settings.n_draws, len(sites)
    base_g = resolve_growth(sites, scenario)
    rho = settings.growth_corr
    z_common = rng.standard_normal((n, 1))
    z_site = rng.standard_normal((n, s))
    shock = np.sqrt(rho) * z_common + np.sqrt(1 - rho) * z_site
    growth = np.clip(base_g[None, :] + settings.growth_sd * shock, C.GROWTH_BOUNDS[0], C.GROWTH_BOUNDS[1])

    exps = scenario_expansions(planned, scenario)
    e_site, e_res, e_amt, e_month = expansion_arrays(exps, sites["site_id"].tolist())
    if len(exps) and settings.slip_max > 0:
        slip = rng.triangular(0.0, settings.slip_mode, settings.slip_max, size=(n, len(exps)))
        months = e_month[None, :] + np.round(slip)
    else:
        months = np.broadcast_to(e_month[None, :], (n, len(exps))).astype(float)

    realization = rng.uniform(settings.eff_realization_low, 1.0, size=n)
    efficiency = scenario.efficiency * realization
    return growth, (e_site, e_res, e_amt, months), efficiency


def run_monte_carlo(
    sites: pd.DataFrame, planned: pd.DataFrame | None, scenario: Scenario, settings: MCSettings = MCSettings()
) -> MCResult:
    scenario.validate(sites["site_id"].tolist())
    settings.validate()
    det = run_forecast(sites, planned, scenario, validate=False)
    growth, (e_site, e_res, e_amt, e_month), eff = sample_inputs(scenario, sites, planned, settings)
    demand0, capacity0 = baseline_arrays(sites)
    demand, capacity = project(demand0, capacity0, growth, e_site, e_res, e_amt, e_month, eff, scenario.density_growth)
    return MCResult(
        scenario=scenario, settings=settings,
        site_ids=sites["site_id"].tolist(), site_names=sites["site_name"].tolist(),
        growth=growth, demand=demand, capacity=capacity,
        exhaust=exhaustion_month(demand, capacity),
        deterministic=det.site_resource,
    )
