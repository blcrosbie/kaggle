import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dcsim.data import EXPANSION_COLUMNS, load_planned_expansions, load_sites  # noqa: E402


def make_sites(**overrides) -> pd.DataFrame:
    """Six identical, hand-checkable sites. Override columns with lists or scalars.

    Defaults: 10 MW power capacity / 8 MW demand (80%), 10 MWt cooling / 5 MWt load,
    100 racks / 50 occupied, 0% growth, PUE 1.5.
    """
    base = {
        "site_id": [f"DC{i}" for i in range(1, 7)],
        "site_name": [f"Test Site {i}" for i in range(1, 7)],
        "region": ["Test"] * 6,
        "commissioned_year": [2020] * 6,
        "power_capacity_mw": [10.0] * 6,
        "power_demand_mw": [8.0] * 6,
        "pue": [1.5] * 6,
        "cooling_capacity_mwt": [10.0] * 6,
        "cooling_load_mwt": [5.0] * 6,
        "rack_capacity": [100] * 6,
        "racks_occupied": [50] * 6,
        "annual_growth_pct": [0.0] * 6,
        "grid_carbon_t_per_mwh": [0.3] * 6,
        "wue_l_per_kwh": [1.0] * 6,
    }
    for k, v in overrides.items():
        base[k] = v if isinstance(v, list) else [v] * 6
    return pd.DataFrame(base)


def no_expansions() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPANSION_COLUMNS)


@pytest.fixture
def real_sites():
    return load_sites()


@pytest.fixture
def real_planned(real_sites):
    return load_planned_expansions(site_ids=real_sites["site_id"].tolist())
