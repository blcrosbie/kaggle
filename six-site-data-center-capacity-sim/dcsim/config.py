"""Model constants and default assumptions.

Every number that shapes a result lives here so it can be documented in one
place (see docs/METHODOLOGY.md) and overridden through a Scenario.
"""

from __future__ import annotations

import pandas as pd

# --- Time ------------------------------------------------------------------
# Baseline ("actual") data is as of the end of this month; forecast month 1 is
# the following calendar month.
AS_OF_PERIOD = pd.Period("2026-10", freq="M")
MAX_HORIZON = 36  # months; the engine always projects this far (risk lookahead)
HORIZON_CHOICES = (6, 12, 18, 24, 30, 36)

# --- Resources -------------------------------------------------------------
RESOURCES = ("power", "cooling", "space")
RESOURCE_LABELS = {"power": "Electrical power", "cooling": "Cooling", "space": "Rack space"}
RESOURCE_UNITS = {"power": "MW", "cooling": "MWt", "space": "racks"}

# Heat rejected to the cooling plant per MW of IT load. Almost all IT power
# becomes heat; the extra ~4% covers in-room losses (PDUs, lighting, fans).
COOLING_HEAT_FACTOR = 1.04
MWT_TO_TONS = 284.345  # 1 MW thermal = 284.345 tons of refrigeration

# --- Defaults for scenario levers -------------------------------------------
DEFAULT_WARNING_THRESHOLD = 0.80   # planning threshold: start expansion work
DEFAULT_CRITICAL_THRESHOLD = 0.90  # operating risk threshold
DEFAULT_DENSITY_GROWTH = 0.03      # annual growth in kW per rack (densification)

# Illustrative unit costs (USD). Order-of-magnitude only - not engineering estimates.
DEFAULT_COST_PER_MW = 3_500_000     # electrical: utility, switchgear, UPS, generators
DEFAULT_COST_PER_MWT = 1_800_000    # cooling: chillers/CRAHs, heat rejection, piping
DEFAULT_COST_PER_RACK = 20_000      # white-space fit-out: rack, PDUs, containment, cabling

# Procurement + construction lead time from commitment to activation (months).
DEFAULT_LEAD_TIME = {"power": 12, "cooling": 9, "space": 3}

# Expansion sizing increments (projects are built in modules, not fractions).
SIZING_INCREMENT = {"power": 0.5, "cooling": 0.5, "space": 10}

# --- Risk classification ----------------------------------------------------
# Months-to-exhaustion cut-offs for site risk status (see classify_risk).
RISK_CRITICAL_MONTHS = 6
RISK_HIGH_MONTHS = 12
RISK_MEDIUM_MONTHS = 24
RISK_WARNING_WINDOW = 12  # warning threshold breached within this many months -> Medium
RISK_LEVELS = ("Low", "Medium", "High", "Critical")
RISK_SCORE = {"Low": 0, "Medium": 33, "High": 67, "Critical": 100}

# --- Capital prioritisation -------------------------------------------------
# Priority = weighted sum of three 0-100 scores (see capital.py).
PRIORITY_WEIGHTS = {"urgency": 0.5, "risk": 0.3, "cost": 0.2}
URGENCY_SLACK_MONTHS = 24  # slack >= this many months scores 0 urgency

# --- Input validation bounds ------------------------------------------------
GROWTH_BOUNDS = (-0.50, 1.00)
EFFICIENCY_BOUNDS = (0.0, 0.50)
PUE_BOUNDS = (1.0, 3.0)
