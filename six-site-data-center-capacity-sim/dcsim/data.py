"""Loading and validating the synthetic site dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .scenario import Expansion, validate_expansion
from .validation import ValidationError

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SITES_CSV = DATA_DIR / "sites.csv"
EXPANSIONS_CSV = DATA_DIR / "planned_expansions.csv"

SITE_COLUMNS = [
    "site_id", "site_name", "region", "commissioned_year",
    "power_capacity_mw", "power_demand_mw", "pue",
    "cooling_capacity_mwt", "cooling_load_mwt",
    "rack_capacity", "racks_occupied",
    "annual_growth_pct", "grid_carbon_t_per_mwh", "wue_l_per_kwh",
]
EXPANSION_COLUMNS = [
    "expansion_id", "site_id", "description", "resource",
    "capacity_added", "cost_usd", "activation_month", "status",
]

# Resource -> (demand column, capacity column) in sites.csv
RESOURCE_COLUMNS = {
    "power": ("power_demand_mw", "power_capacity_mw"),
    "cooling": ("cooling_load_mwt", "cooling_capacity_mwt"),
    "space": ("racks_occupied", "rack_capacity"),
}


def load_sites(path: str | Path = SITES_CSV) -> pd.DataFrame:
    sites = pd.read_csv(path)
    validate_sites(sites)
    return sites


def load_planned_expansions(path: str | Path = EXPANSIONS_CSV, site_ids: list[str] | None = None) -> pd.DataFrame:
    exp = pd.read_csv(path)
    validate_expansions(exp, site_ids)
    return exp


def validate_sites(sites: pd.DataFrame) -> None:
    errors: list[str] = []
    missing = [c for c in SITE_COLUMNS if c not in sites.columns]
    if missing:
        raise ValidationError([f"Site data is missing column(s): {missing}."])

    ids = sites["site_id"].astype(str)
    if ids.duplicated().any():
        errors.append(f"Duplicate site IDs: {sorted(ids[ids.duplicated()].unique())}.")
    if len(sites) != 6:
        errors.append(f"Expected 6 facilities (DC1-DC6), found {len(sites)}.")

    numeric = [c for c in SITE_COLUMNS if c not in ("site_id", "site_name", "region")]
    for col in numeric:
        vals = pd.to_numeric(sites[col], errors="coerce")
        if vals.isna().any():
            errors.append(f"Column '{col}' has missing or non-numeric values.")
            continue
        if col != "annual_growth_pct" and (vals < 0).any():
            bad = ids[vals < 0].tolist()
            errors.append(f"'{col}' cannot be negative (sites {bad}).")
    if errors:
        raise ValidationError(errors)

    for res, (dcol, ccol) in RESOURCE_COLUMNS.items():
        zero_cap = ids[sites[ccol] <= 0].tolist()
        if zero_cap:
            errors.append(f"{C.RESOURCE_LABELS[res]} capacity must be greater than zero (sites {zero_cap}).")
        over = ids[sites[dcol] > sites[ccol]].tolist()
        if over:
            errors.append(
                f"Current {C.RESOURCE_LABELS[res].lower()} demand exceeds installed capacity (sites {over}); "
                "a facility cannot draw more than it has."
            )
    lo, hi = C.PUE_BOUNDS
    bad_pue = ids[(sites["pue"] < lo) | (sites["pue"] > hi)].tolist()
    if bad_pue:
        errors.append(f"PUE must be between {lo} and {hi} (sites {bad_pue}).")
    glo, ghi = C.GROWTH_BOUNDS
    g = sites["annual_growth_pct"] / 100
    bad_g = ids[(g < glo) | (g > ghi)].tolist()
    if bad_g:
        errors.append(f"Annual growth must be between {glo:.0%} and {ghi:.0%} (sites {bad_g}).")
    if errors:
        raise ValidationError(errors)


def validate_expansions(exp: pd.DataFrame, site_ids: list[str] | None = None) -> None:
    missing = [c for c in EXPANSION_COLUMNS if c not in exp.columns]
    if missing:
        raise ValidationError([f"Expansion data is missing column(s): {missing}."])
    errors: list[str] = []
    for e in expansions_from_frame(exp):
        errors.extend(validate_expansion(e, site_ids))
    if errors:
        raise ValidationError(errors)


def expansions_from_frame(exp: pd.DataFrame) -> list[Expansion]:
    return [
        Expansion(
            site_id=str(r.site_id),
            resource=str(r.resource),
            amount=float(r.capacity_added),
            activation_month=int(r.activation_month),
            cost_usd=float(r.cost_usd),
            label=str(r.description),
            source="planned",
        )
        for r in exp.itertuples(index=False)
    ]


def baseline_arrays(sites: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (demand0, capacity0), each shaped (n_sites, n_resources)."""
    demand0 = np.column_stack([sites[RESOURCE_COLUMNS[r][0]].to_numpy(float) for r in C.RESOURCES])
    capacity0 = np.column_stack([sites[RESOURCE_COLUMNS[r][1]].to_numpy(float) for r in C.RESOURCES])
    return demand0, capacity0
