"""Scenario definition: the levers a user can pull, plus the predefined scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from . import config as C
from .validation import ValidationError


@dataclass(frozen=True)
class Expansion:
    """A capacity addition to one resource at one site, live from `activation_month`."""

    site_id: str
    resource: str  # one of config.RESOURCES
    amount: float  # MW, MWt or racks depending on resource
    activation_month: int  # forecast month the capacity becomes usable (>= 1)
    cost_usd: float | None = None  # None -> priced from the scenario's unit costs
    label: str = "Scenario expansion"
    source: str = "scenario"  # "planned" | "scenario" | "capital plan"


@dataclass(frozen=True)
class Scenario:
    name: str = "Baseline"
    horizon: int = C.MAX_HORIZON
    portfolio_growth: float | None = None  # overrides every site's growth when set
    site_growth: dict[str, float] = field(default_factory=dict)  # per-site override (wins)
    efficiency: float = 0.0  # fractional reduction in power + cooling demand
    include_planned: bool = True
    added_expansions: tuple[Expansion, ...] = ()
    budget_usd: float | None = None
    warning_threshold: float = C.DEFAULT_WARNING_THRESHOLD
    critical_threshold: float = C.DEFAULT_CRITICAL_THRESHOLD
    density_growth: float = C.DEFAULT_DENSITY_GROWTH
    cost_per_mw: float = C.DEFAULT_COST_PER_MW
    cost_per_mwt: float = C.DEFAULT_COST_PER_MWT
    cost_per_rack: float = C.DEFAULT_COST_PER_RACK
    lead_time: dict[str, int] = field(default_factory=lambda: dict(C.DEFAULT_LEAD_TIME))

    def unit_cost(self, resource: str) -> float:
        return {"power": self.cost_per_mw, "cooling": self.cost_per_mwt, "space": self.cost_per_rack}[resource]

    def with_expansions(self, *expansions: Expansion, name: str | None = None) -> "Scenario":
        return replace(self, added_expansions=self.added_expansions + tuple(expansions), name=name or self.name)

    def validate(self, site_ids: list[str] | None = None) -> None:
        errors: list[str] = []
        if self.horizon not in C.HORIZON_CHOICES:
            errors.append(f"Forecast horizon must be one of {C.HORIZON_CHOICES} months (got {self.horizon}).")
        lo, hi = C.GROWTH_BOUNDS
        growths = {"portfolio": self.portfolio_growth} | dict(self.site_growth)
        for who, g in growths.items():
            if g is not None and not lo <= g <= hi:
                errors.append(f"Annual growth for {who} must be between {lo:.0%} and {hi:.0%} (got {g:.1%}).")
        if site_ids is not None:
            unknown = set(self.site_growth) - set(site_ids)
            if unknown:
                errors.append(f"Growth override for unknown site(s): {sorted(unknown)}.")
        elo, ehi = C.EFFICIENCY_BOUNDS
        if not elo <= self.efficiency <= ehi:
            errors.append(f"Efficiency improvement must be between {elo:.0%} and {ehi:.0%} (got {self.efficiency:.1%}).")
        if not 0 < self.warning_threshold < self.critical_threshold <= 1.0:
            errors.append(
                "Thresholds must satisfy 0 < warning < critical <= 100% "
                f"(got warning {self.warning_threshold:.0%}, critical {self.critical_threshold:.0%})."
            )
        if self.budget_usd is not None and self.budget_usd < 0:
            errors.append("Capital budget cannot be negative.")
        if not 0 <= self.density_growth <= 0.5:
            errors.append("Rack density growth must be between 0% and 50% per year.")
        for name, v in (("cost per MW", self.cost_per_mw), ("cost per MWt", self.cost_per_mwt), ("cost per rack", self.cost_per_rack)):
            if v < 0:
                errors.append(f"Unit {name} cannot be negative.")
        for r, lt in self.lead_time.items():
            if lt < 0:
                errors.append(f"Lead time for {r} cannot be negative.")
        for e in self.added_expansions:
            errors.extend(validate_expansion(e, site_ids))
        if errors:
            raise ValidationError(errors)


def validate_expansion(e: Expansion, site_ids: list[str] | None = None) -> list[str]:
    errors = []
    tag = f"Expansion '{e.label}' at {e.site_id}"
    if site_ids is not None and e.site_id not in site_ids:
        errors.append(f"{tag}: unknown site.")
    if e.resource not in C.RESOURCES:
        errors.append(f"{tag}: resource must be one of {C.RESOURCES} (got '{e.resource}').")
    if not e.amount > 0:
        errors.append(f"{tag}: capacity added must be positive (got {e.amount}).")
    if not 1 <= e.activation_month <= C.MAX_HORIZON:
        errors.append(f"{tag}: activation month must be 1-{C.MAX_HORIZON} (got {e.activation_month}).")
    if e.cost_usd is not None and e.cost_usd < 0:
        errors.append(f"{tag}: cost cannot be negative.")
    return errors


# --- Predefined scenarios (SOW Module 4) ------------------------------------

def baseline(**overrides) -> Scenario:
    return replace(Scenario(name="Baseline"), **overrides)


def high_growth(rate: float = 0.25, **overrides) -> Scenario:
    return replace(Scenario(name="High Growth", portfolio_growth=rate), **overrides)


def infrastructure_expansion(
    site_id: str,
    power_mw: float = 0.0,
    cooling_mwt: float = 0.0,
    racks: float = 0.0,
    activation_month: int = 12,
    **overrides,
) -> Scenario:
    adds = [
        Expansion(site_id, res, amt, activation_month, label=f"{site_id} {C.RESOURCE_LABELS[res].lower()} expansion")
        for res, amt in (("power", power_mw), ("cooling", cooling_mwt), ("space", racks))
        if amt > 0
    ]
    return replace(Scenario(name="Infrastructure Expansion", added_expansions=tuple(adds)), **overrides)


def efficiency_optimization(reduction: float = 0.10, **overrides) -> Scenario:
    return replace(Scenario(name="Efficiency Optimization", efficiency=reduction), **overrides)


def budget_constrained(budget_usd: float = 60_000_000, **overrides) -> Scenario:
    """Budget is applied by capital.plan_capital(); funded projects become expansions."""
    return replace(Scenario(name="Budget-Constrained", budget_usd=budget_usd), **overrides)
