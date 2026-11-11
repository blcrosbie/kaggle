# Data Dictionary

> **Synthetic data — demonstration only.** Every facility, name and value is invented. Regenerate with
> `python -m dcsim.generate` (seed 2026). The test suite asserts the committed CSVs match the generator output.

## `data/sites.csv` — one row per facility (6 rows)

| Column | Type | Unit | Description | Validation |
|---|---|---|---|---|
| `site_id` | str | — | DC1–DC6 | unique, exactly 6 |
| `site_name` | str | — | Fictional facility name | |
| `region` | str | — | Generic region label | |
| `commissioned_year` | int | year | Year the facility opened | ≥ 0 |
| `power_capacity_mw` | float | MW | Usable facility electrical capacity | > 0 |
| `power_demand_mw` | float | MW | Current total facility draw (IT × PUE) | 0 ≤ demand ≤ capacity |
| `pue` | float | ratio | Power Usage Effectiveness | 1.0–3.0 |
| `cooling_capacity_mwt` | float | MWt | Installed cooling plant capacity | > 0 |
| `cooling_load_mwt` | float | MWt | Current heat rejection load ≈ IT load × 1.04 | 0 ≤ load ≤ capacity |
| `rack_capacity` | int | racks | Installed rack positions | > 0 |
| `racks_occupied` | int | racks | Occupied rack positions | 0 ≤ occupied ≤ capacity |
| `annual_growth_pct` | float | % / yr | Site demand growth assumption | −50 to 100 |
| `grid_carbon_t_per_mwh` | float | tCO₂e/MWh | Grid emissions intensity (sustainability view) | ≥ 0 |
| `wue_l_per_kwh` | float | L/kWh | Water Usage Effectiveness (sustainability view) | ≥ 0 |

## `data/planned_expansions.csv` — committed / planned projects

| Column | Type | Unit | Description | Validation |
|---|---|---|---|---|
| `expansion_id` | str | — | Project identifier | |
| `site_id` | str | — | Site the capacity is added to | must exist |
| `description` | str | — | Project description | |
| `resource` | str | — | `power`, `cooling` or `space` | one of three |
| `capacity_added` | float | MW / MWt / racks | Capacity added | > 0 |
| `cost_usd` | float | USD | Estimated project cost | ≥ 0 |
| `activation_month` | int | month | Forecast month the capacity becomes usable | 1–36 |
| `status` | str | — | `Approved` or `Planned` | |

## Derived fields (computed, not stored)

| Field | Where | Definition |
|---|---|---|
| `utilization` | all tables | demand / capacity |
| `headroom` | all tables | capacity − demand |
| `data_type` | monthly tables | `Actual (synthetic baseline)` for month 0, else `Projected` |
| `exhaustion_month` | site × resource | first month demand > capacity; blank = not within 36 months |
| `warning_month`, `critical_month` | site × resource | first month utilization ≥ threshold |
| `limiting_resource`, `limiting_basis` | site summary | see Methodology §4 |
| `risk_status`, `risk_reason` | site summary | Low / Medium / High / Critical with the rule that fired (Methodology §5) |
| `priority_score`, `rank`, `funded` | capital plan | Methodology §6 |
| `p10_month`, `p50_month`, `p90_month`, `prob_within_Nm` | Monte Carlo | Methodology §7 |

## Site archetypes (the story the synthetic data is designed to tell)

| Site | Archetype | What the model should surface |
|---|---|---|
| DC1 Basalt Ridge | Efficient hydro-powered site, moderate growth | Low risk; low-carbon candidate for new load |
| DC2 Copper Mesa | Desert site at 82% power, approved Phase 3 at month 12 | Safe *if* Phase 3 lands on time; Monte Carlo shows slip risk |
| DC3 Prairie Junction | AI growth (~28%/yr) on a 20 MW site, nothing planned | Earliest constraint: power exhausted ~month 8 |
| DC4 Harbor Point | Legacy enterprise site, 92% racks, PUE 1.55 | Rack-space bound; efficiency opportunity |
| DC5 Piedmont Hollow | Legacy cooling at ~87%, chiller replacement at month 9 | Cooling-bound; depends on chiller timing |
| DC6 Granite Pass | New campus at ~35% with Phase 2 planned for month 24 | Spare capacity; Phase 2 deferral candidate |
