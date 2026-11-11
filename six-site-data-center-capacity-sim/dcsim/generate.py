"""Reproducible synthetic dataset for six fictional data centers.

    python -m dcsim.generate            # writes data/sites.csv + data/planned_expansions.csv
    python -m dcsim.generate --seed 7   # a different but equally reproducible draw

Each site starts from a hand-designed archetype so the portfolio tells a useful
story (one power-bound AI site, one cooling-bound legacy site, one space-bound
enterprise site, an expansion that has to land on time, and an under-used new
campus). A seeded RNG then perturbs utilisation and growth slightly so the
numbers look measured rather than typed. All names and values are invented.

Internal consistency is enforced by construction:
    power demand (MW)  = IT load x PUE
    cooling load (MWt) = IT load x COOLING_HEAT_FACTOR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import DATA_DIR, EXPANSION_COLUMNS, SITE_COLUMNS, validate_expansions, validate_sites

DEFAULT_SEED = 2026

# Archetype targets before jitter. util_* are target utilisations of each resource.
ARCHETYPES = [
    dict(site_id="DC1", site_name="Basalt Ridge", region="Pacific Northwest", commissioned_year=2016,
         power_capacity_mw=24.0, util_power=0.62, pue=1.22, util_cooling=0.66,
         rack_capacity=2400, util_space=0.63, growth=0.09, grid=0.09, wue=0.4),
    dict(site_id="DC2", site_name="Copper Mesa", region="Desert Southwest", commissioned_year=2014,
         power_capacity_mw=30.0, util_power=0.82, pue=1.42, util_cooling=0.62,
         rack_capacity=2600, util_space=0.70, growth=0.15, grid=0.38, wue=1.8),
    dict(site_id="DC3", site_name="Prairie Junction", region="Upper Midwest", commissioned_year=2019,
         power_capacity_mw=20.0, util_power=0.85, pue=1.30, util_cooling=0.70,
         rack_capacity=1600, util_space=0.66, growth=0.28, grid=0.45, wue=0.9),
    dict(site_id="DC4", site_name="Harbor Point", region="Mid-Atlantic", commissioned_year=2009,
         power_capacity_mw=16.0, util_power=0.74, pue=1.55, util_cooling=0.68,
         rack_capacity=2000, util_space=0.91, growth=0.08, grid=0.32, wue=1.1),
    dict(site_id="DC5", site_name="Piedmont Hollow", region="Southeast", commissioned_year=2011,
         power_capacity_mw=22.0, util_power=0.68, pue=1.60, util_cooling=0.87,
         rack_capacity=1800, util_space=0.71, growth=0.11, grid=0.41, wue=1.4),
    dict(site_id="DC6", site_name="Granite Pass", region="Mountain West", commissioned_year=2023,
         power_capacity_mw=36.0, util_power=0.34, pue=1.18, util_cooling=0.38,
         rack_capacity=3200, util_space=0.30, growth=0.20, grid=0.28, wue=0.2),
]

PLANNED_EXPANSIONS = [
    ("EXP-201", "DC2", "Phase 3 electrical: new 10 MW utility feed + UPS block", "power", 10.0, 34_000_000, 12, "Approved"),
    ("EXP-202", "DC2", "Phase 3 cooling: two 3 MWt air-cooled chillers", "cooling", 6.0, 11_000_000, 12, "Approved"),
    ("EXP-501", "DC5", "Chiller plant replacement (+4 MWt net)", "cooling", 4.0, 7_500_000, 9, "Approved"),
    ("EXP-601", "DC6", "Phase 2 shell: 12 MW electrical", "power", 12.0, 40_000_000, 24, "Planned"),
    ("EXP-602", "DC6", "Phase 2 shell: 10 MWt cooling", "cooling", 10.0, 18_000_000, 24, "Planned"),
    ("EXP-603", "DC6", "Phase 2 white space: 800 racks", "space", 800, 16_000_000, 24, "Planned"),
]


def generate(seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    for a in ARCHETYPES:
        jit = lambda sd: rng.normal(0.0, sd)  # noqa: E731
        util_p = float(np.clip(a["util_power"] + jit(0.015), 0.05, 0.97))
        util_c = float(np.clip(a["util_cooling"] + jit(0.015), 0.05, 0.97))
        util_s = float(np.clip(a["util_space"] + jit(0.015), 0.05, 0.97))
        pue = round(a["pue"] + jit(0.02), 2)

        power_cap = a["power_capacity_mw"]
        power_dem = round(power_cap * util_p, 2)
        it_load = power_dem / pue
        cooling_load = round(it_load * C.COOLING_HEAT_FACTOR, 2)
        cooling_cap = round(cooling_load / util_c * 2) / 2  # installed plant in 0.5 MWt steps
        racks_occ = int(round(a["rack_capacity"] * util_s))

        rows.append({
            "site_id": a["site_id"],
            "site_name": a["site_name"],
            "region": a["region"],
            "commissioned_year": a["commissioned_year"],
            "power_capacity_mw": power_cap,
            "power_demand_mw": power_dem,
            "pue": pue,
            "cooling_capacity_mwt": cooling_cap,
            "cooling_load_mwt": cooling_load,
            "rack_capacity": a["rack_capacity"],
            "racks_occupied": racks_occ,
            "annual_growth_pct": round(100 * (a["growth"] + jit(0.01)), 1),
            "grid_carbon_t_per_mwh": a["grid"],
            "wue_l_per_kwh": a["wue"],
        })
    sites = pd.DataFrame(rows, columns=SITE_COLUMNS)
    expansions = pd.DataFrame(PLANNED_EXPANSIONS, columns=EXPANSION_COLUMNS)
    validate_sites(sites)
    validate_expansions(expansions, sites["site_id"].tolist())
    return sites, expansions


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", type=Path, default=DATA_DIR)
    args = ap.parse_args(argv)
    sites, expansions = generate(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    sites.to_csv(args.out / "sites.csv", index=False)
    expansions.to_csv(args.out / "planned_expansions.csv", index=False)
    print(f"Wrote {len(sites)} sites and {len(expansions)} planned expansions to {args.out} (seed {args.seed})")
    print(sites.to_string(index=False))


if __name__ == "__main__":
    main()
