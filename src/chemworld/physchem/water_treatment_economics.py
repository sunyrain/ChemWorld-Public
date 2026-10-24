"""Read-only engineering estimates from committed device throughput receipts.

Currencies are authored nominal units at an explicit price base year. This is
neither an exchange-rate service nor the benchmark's action-cost ledger.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import expm1, fsum, isclose, isfinite, log1p
from types import MappingProxyType
from typing import Any

from chemworld.physchem.zero_order_treatment_specs import ZeroOrderTreatmentSpec, finite_number

SCALARS = (
    "elapsed_s",
    "feed_volume_m3",
    "product_volume_m3",
    "pump_work_J",
    "hydraulic_work_J",
    "additional_electrical_work_J",
    "motor_loss_J",
    "bath_heat_into_liquid_J",
)
MAPPINGS = ("reject_volumes_m3", "stage_inlet_volumes_m3")


def empty_treatment_receipt(spec: ZeroOrderTreatmentSpec) -> dict[str, Any]:
    return {
        "schema": "native-treatment-throughput-v1",
        **dict.fromkeys(SCALARS, 0.0),
        **{key: {s.stage_id: 0.0 for s in spec.stages} for key in MAPPINGS},
    }


def validate_treatment_receipt(data: Mapping[str, Any]) -> dict[str, Any]:
    if set(data) != {"schema", *SCALARS, *MAPPINGS} or (
        data.get("schema") != "native-treatment-throughput-v1"
    ):
        raise ValueError("unknown or incomplete actual treatment receipt")
    result = deepcopy(dict(data))
    for key in SCALARS:
        value = result[key]
        if key == "bath_heat_into_liquid_J":
            if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
                raise ValueError("receipt bath heat must be finite signed energy")
        else:
            finite_number(value, key)
    for key in MAPPINGS:
        if not isinstance(result[key], dict) or not result[key]:
            raise ValueError("receipt requires all stage quantities")
        if any(not isinstance(name, str) or not name for name in result[key]):
            raise ValueError("receipt stage identity must be explicit")
        for value in result[key].values():
            finite_number(value, key)
    if set(result[MAPPINGS[0]]) != set(result[MAPPINGS[1]]):
        raise ValueError("treatment receipt stage identities disagree")
    if not isclose(
        result["feed_volume_m3"],
        result["product_volume_m3"] + fsum(result["reject_volumes_m3"].values()),
        rel_tol=3e-7,
        abs_tol=1e-12,
    ):
        raise ValueError("treatment receipt lacks the actual collected volume balance")
    if not isclose(
        result["pump_work_J"],
        result["hydraulic_work_J"] + result["motor_loss_J"],
        rel_tol=3e-7,
        abs_tol=1e-9,
    ):
        raise ValueError("treatment pump receipt energy disagrees")
    if result["elapsed_s"] == 0 and (
        any(result[k] != 0 for k in SCALARS if k != "elapsed_s")
        or any(value != 0 for key in MAPPINGS for value in result[key].values())
    ):
        raise ValueError("zero-time treatment cannot produce volume or consume utility")
    return result


def accumulate_treatment_receipts(
    previous: Mapping[str, Any],
    increment: Mapping[str, Any],
) -> dict[str, Any]:
    left, right = validate_treatment_receipt(previous), validate_treatment_receipt(increment)
    if set(left["reject_volumes_m3"]) != set(right["reject_volumes_m3"]):
        raise ValueError("cannot mix receipts from different treatment chains")
    return validate_treatment_receipt(
        {
            "schema": left["schema"],
            **{key: left[key] + right[key] for key in SCALARS},
            **{
                key: {stage: left[key][stage] + right[key][stage] for stage in left[key]}
                for key in MAPPINGS
            },
        }
    )


@dataclass(frozen=True)
class WaterTreatmentEconomicsSpec:
    currency: str
    price_base_year: int
    installed_capital: Mapping[str, float]
    lifetime_years: float
    discount_rate_per_year: float
    maintenance_fraction_per_year: float
    fixed_labor_per_year: float
    electricity_price_per_kWh: float
    concentrate_handling_price_per_m3: float
    operating_hours_per_year: float
    source_note: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.currency, str)
            or len(self.currency) != 3
            or not self.currency.isascii()
            or not self.currency.isalpha()
            or self.currency != self.currency.upper()
            or not self.source_note.strip()
        ):
            raise ValueError(
                "economics requires an explicit uppercase currency code and price source"
            )
        if type(self.price_base_year) is not int or not 1 <= self.price_base_year <= 9999:
            raise ValueError("economics requires an explicit integer price base year")
        if not self.installed_capital or any(not key for key in self.installed_capital):
            raise ValueError("installed capital must identify every empirical stage")
        capital = {
            k: finite_number(v, "installed capital") for k, v in self.installed_capital.items()
        }
        object.__setattr__(self, "installed_capital", MappingProxyType(capital))
        for key in ("lifetime_years", "operating_hours_per_year"):
            finite_number(getattr(self, key), key, positive=True)
        if self.operating_hours_per_year > 8760:
            raise ValueError("annual operating hours exceed the declared 365-day year")
        for key in (
            "discount_rate_per_year",
            "maintenance_fraction_per_year",
            "fixed_labor_per_year",
            "electricity_price_per_kWh",
            "concentrate_handling_price_per_m3",
        ):
            finite_number(getattr(self, key), key)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "installed_capital": dict(self.installed_capital)}


def evaluate_water_treatment_economics(
    receipt: Mapping[str, Any],
    policy: WaterTreatmentEconomicsSpec,
) -> dict[str, Any]:
    data = validate_treatment_receipt(receipt)
    if set(data["stage_inlet_volumes_m3"]) != set(policy.installed_capital):
        raise ValueError("cost policy capital items must match the actual treatment stages")
    electricity = (data["pump_work_J"] + data["additional_electrical_work_J"]) / 3.6e6
    concentrate = fsum(data["reject_volumes_m3"].values())
    electricity_cost = electricity * policy.electricity_price_per_kWh
    handling_cost = concentrate * policy.concentrate_handling_price_per_m3
    variable = electricity_cost + handling_cost
    capital = fsum(policy.installed_capital.values())
    r, years = policy.discount_rate_per_year, policy.lifetime_years
    crf = 1 / years if r == 0 else r / -expm1(-years * log1p(r))
    capital_charge = capital * crf
    fixed = capital * policy.maintenance_fraction_per_year + policy.fixed_labor_per_year
    annual_seconds = policy.operating_hours_per_year * 3600
    factor = annual_seconds / data["elapsed_s"] if data["elapsed_s"] else 0.0
    annual_product = factor * data["product_volume_m3"]
    annual_variable = factor * variable
    total_annual = capital_charge + fixed + annual_variable
    return {
        "schema": "native-water-treatment-economics-v1",
        "currency": policy.currency,
        "price_base_year": policy.price_base_year,
        "policy": policy.to_dict(),
        "actual_observed_receipt": data,
        "observed_electricity_kWh": electricity,
        "observed_electricity_cost": electricity_cost,
        "observed_concentrate_handling_estimate": handling_cost,
        "observed_variable_cost_estimate": variable,
        "installed_capital": capital,
        "capital_recovery_factor_per_year": crf,
        "annual_capital_charge": capital_charge,
        "annual_fixed_operating_cost": fixed,
        "projected_annual_variable_cost": annual_variable,
        "projected_annual_product_m3": annual_product,
        "projected_annual_total_cost": total_annual,
        "levelized_product_cost_per_m3": total_annual / annual_product if annual_product else None,
        "annualization_scope": (
            "observed production/utility profile at declared annual operating hours"
        ),
        "product_basis": "newly collected treated liquid solution volume, excluding initial stock",
        "boundary": "engineering estimate; no payment, disposal, inflation or currency conversion",
    }
