"""Executable physical constitution for ChemWorld environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Any, ClassVar

from chemworld.foundation.constitution_preconditions import check_operation_preconditions
from chemworld.foundation.constitution_reports import CheckResult, ConstitutionReport
from chemworld.foundation.constitution_state_checks import (
    check_ledger_single_source,
    check_nonnegative,
    check_risk_range,
    check_species_registry,
    check_typed_ledgers,
    check_units,
    check_vessel_bounds,
)
from chemworld.foundation.ontology import Instrument, Substance, Vessel
from chemworld.foundation.public_leakage import audit_public_payload
from chemworld.foundation.state import Observation, WorldState
from chemworld.foundation.units import canonical_unit


class PhysicalConstitution:
    """Executable constraints that every environment state must satisfy."""

    hidden_state_keys: ClassVar[set[str]] = {
        "species_amounts",
        "phase_ledger",
        "rate_constants",
        "theta",
        "temperature_internal",
        "true_yield",
        "true_selectivity",
        "true_conversion",
    }

    primary_reactor_metadata_keys: ClassVar[set[str]] = {
        "catalyst",
        "solvent",
        "stirring_speed_rpm",
    }

    primary_phase_metadata_keys: ClassVar[set[str]] = {
        "phase_system",
        "phase_settled",
        "selected_phase",
    }

    primary_vessel_metadata_keys: ClassVar[set[str]] = {
        "max_volume_L",
        "max_temperature_K",
        "max_pressure_Pa",
    }

    primary_instrument_metadata_keys: ClassVar[set[str]] = {
        "final_assay_done",
        "final_assay_time_s",
    }

    primary_crystallizer_metadata_keys: ClassVar[set[str]] = {
        "crystal_seeded",
        "crystal_seed_mass_g",
    }

    primary_crystallization_output_metadata_keys: ClassVar[set[str]] = {
        "crystallization_active",
        "crystal_product_mol",
        "crystal_impurity_mol",
    }

    primary_distillation_output_metadata_keys: ClassVar[set[str]] = {
        "distillation_active",
        "distillate_product_mol",
        "distillate_impurity_mol",
    }

    primary_downstream_operation_metadata_keys: ClassVar[set[str]] = {
        "crystals_filtered",
        "distillation_model",
        "distillation_kernel",
        "fraction_collected",
    }

    primary_process_metric_metadata_keys: ClassVar[set[str]] = {
        "last_observation",
        "last_observed_mask",
        "pre_separation_product_mol",
        "purity",
        "recovery",
        "solvent_loss",
        "crystal_yield",
        "crystal_purity",
        "crystal_size",
        "distillate_purity",
        "distillate_recovery",
        "flow_conversion",
        "flow_campaign_time_s",
        "flow_throughput_mL",
        "electrochemical_model",
        "selective_product_yield",
        "electrochemical_conversion",
        "electrochemical_selectivity",
        "faradaic_efficiency",
        "energy_efficiency",
        "equilibrium_potential_V",
        "measured_potential_V",
        "interfacial_potential_V",
        "overpotential_V",
        "kinetic_current_A",
        "actual_current_A",
        "charge_C",
        "faradaic_charge_C",
        "electrical_work_J",
        "interfacial_work_J",
        "ohmic_loss_J",
        "electrolyte_resistance_ohm",
        "contact_resistance_ohm",
        "total_resistance_ohm",
        "uncompensated_voltage_drop_V",
        "voltage_window_exceeded",
    }

    required_state_units: ClassVar[dict[str, str]] = {
        "volume_L": "L",
        "temperature_K": "K",
        "pressure_Pa": "Pa",
        "ledger.time_s": "s",
        "ledger.cost": "currency",
        "ledger.risk": "risk",
        "ledger.sample_consumed_L": "L",
    }

    def __init__(
        self,
        *,
        substances: dict[str, Substance],
        vessel: Vessel,
        instruments: dict[str, Instrument],
        max_yield: float = 1.0,
        tolerance: float = 1.0e-8,
    ) -> None:
        self.substances = substances
        self.vessel = vessel
        self.instruments = instruments
        self.max_yield = max_yield
        self.tolerance = tolerance

    def check_state(self, state: WorldState) -> ConstitutionReport:
        checks = [
            *check_nonnegative(self, state),
            check_species_registry(self, state),
            *check_units(self),
            *check_vessel_bounds(self, state),
            *check_typed_ledgers(self, state),
            *check_ledger_single_source(self, state),
            check_risk_range(self, state),
        ]
        from chemworld.foundation.state_helpers import equipment_settings

        controller = equipment_settings(state.equipment, "process_controller")
        if controller:
            from chemworld.foundation.gas import GasBoundary

            try:
                gas = GasBoundary.from_dict(controller["gas"])
                gas_valid = abs(state.pressure_Pa - gas.pressure_pa) <= 1e-6
            except (KeyError, TypeError, ValueError):
                gas_valid = False
            checks.append(
                CheckResult(
                    "gas_boundary_balance",
                    gas_valid,
                    "plenum component and energy accounting; measured pressure",
                )
            )
        for name, local in sorted(state.inactive_vessels.items()):
            for check in sorted(self.check_state(local).checks, key=lambda item: item.name):
                from dataclasses import replace

                checks.append(replace(check, name=f"vessel:{name}:{check.name}"))
        return ConstitutionReport(checks)

    def check_observation(
        self,
        observation: Observation,
        *,
        debug_truth: bool = False,
    ) -> ConstitutionReport:
        checks: list[CheckResult] = []
        leakage = audit_public_payload(
            observation.to_dict(),
            hidden_species_ids=set(self.substances),
            allow_debug_truth=debug_truth,
        )
        checks.append(
            CheckResult(
                "observation_non_omniscient",
                not leakage,
                ""
                if not leakage
                else "Public observation contains hidden or private payload fields.",
            )
        )
        normalized_values = all(
            _is_missing_or_normalized(value) for value in observation.values.values()
        )
        checks.append(
            CheckResult(
                "observation_values_finite_and_bounded",
                normalized_values,
                "Public scalar observations must be missing or finite values in [0, 1].",
            )
        )
        mask_consistent = all(
            isinstance(key, str) and isinstance(observed, bool)
            for key, observed in observation.observed_mask.items()
        ) and all(
            not observed or observation.values.get(key) is not None
            for key, observed in observation.observed_mask.items()
        )
        checks.append(
            CheckResult(
                "observation_mask_consistent",
                mask_consistent,
                "Observed mask entries must be boolean and observed values cannot be missing.",
            )
        )
        processed_valid = all(
            _is_missing_or_normalized(value) for value in observation.processed_estimate.values()
        )
        uncertainty_valid = all(
            _is_finite_nonnegative(value) for value in observation.uncertainty.values()
        )
        raw_signal_valid = _json_numeric_tree_is_finite(observation.raw_signal)
        checks.extend(
            (
                CheckResult(
                    "observation_processed_estimate_finite_and_bounded",
                    processed_valid,
                    "Processed estimates must be missing or finite values in [0, 1].",
                ),
                CheckResult(
                    "observation_uncertainty_finite_nonnegative",
                    uncertainty_valid,
                    "Observation uncertainties must be finite and nonnegative.",
                ),
                CheckResult(
                    "observation_raw_signal_finite",
                    raw_signal_valid,
                    "Raw public signal payloads must be JSON-like and numerically finite.",
                ),
                CheckResult(
                    "observation_accounting_finite_nonnegative",
                    _is_finite_nonnegative(observation.cost)
                    and _is_finite_nonnegative(observation.sample_consumed_L),
                    "Observation cost and sample accounting must be finite and nonnegative.",
                ),
            )
        )
        checks.append(
            CheckResult(
                "measurement_has_cost",
                observation.instrument_id is None
                or observation.cost > 0.0
                or observation.sample_consumed_L > 0.0,
                "Instrument observations must consume cost or sample.",
            )
        )
        for key, unit in observation.units.items():
            checks.append(
                CheckResult(
                    f"observation_unit:{key}",
                    _is_canonical_unit(unit),
                    str(unit),
                )
            )
        return ConstitutionReport(checks)

    def check_preconditions(
        self,
        operation_type: str,
        state: WorldState,
        payload: dict[str, Any],
    ) -> dict[str, bool]:
        return check_operation_preconditions(self, operation_type, state, payload)

    def check_material_conservation(
        self,
        before: WorldState,
        after: WorldState,
        *,
        allowed_element_delta: dict[str, float] | None = None,
    ) -> CheckResult:
        allowed_element_delta = allowed_element_delta or {}
        unknown_species = (
            set(before.species_amounts)
            | set(after.species_amounts)
            | set(before.samples.total_amounts_mol())
            | set(after.samples.total_amounts_mol())
            | {s for local in before.inactive_vessels.values() for s in local.species_amounts}
            | {s for local in after.inactive_vessels.values() for s in local.species_amounts}
        ) - set(self.substances)
        if unknown_species:
            return CheckResult(
                "material_conservation",
                False,
                "Material conservation cannot be established for unregistered species.",
                value=float(len(unknown_species)),
                tolerance=0.0,
            )
        before_amounts = before.species_amounts.copy()
        after_amounts = after.species_amounts.copy()
        for amounts, state in ((before_amounts, before), (after_amounts, after)):
            for species, amount in state.samples.total_amounts_mol().items():
                amounts[species] = amounts.get(species, 0.0) + amount
            for local in state.inactive_vessels.values():
                for species, amount in local.species_amounts.items():
                    amounts[species] = amounts.get(species, 0.0) + amount
        before_elements = self.element_totals(before_amounts)
        after_elements = self.element_totals(after_amounts)
        errors: list[float] = []
        elements = before_elements.keys() | after_elements.keys() | allowed_element_delta.keys()
        for element in sorted(elements):
            expected = before_elements.get(element, 0.0) + allowed_element_delta.get(element, 0.0)
            errors.append(abs(after_elements.get(element, 0.0) - expected))
        max_error = max(errors) if errors else 0.0
        return CheckResult(
            "material_conservation",
            max_error <= self.tolerance,
            f"max element-balance error={max_error:.3g}",
            value=max_error,
            tolerance=self.tolerance,
        )

    def check_yield_upper_bound(self, product_amount: float, limiting_amount: float) -> CheckResult:
        yield_value = 0.0 if limiting_amount <= self.tolerance else product_amount / limiting_amount
        return CheckResult(
            "yield_upper_bound",
            yield_value <= self.max_yield + self.tolerance,
            f"yield={yield_value:.3g}",
            value=yield_value,
            tolerance=self.tolerance,
        )

    def element_totals(self, species_amounts: dict[str, float]) -> dict[str, float]:
        totals: dict[str, float] = {}
        for species_id, amount in species_amounts.items():
            substance = self.substances.get(species_id)
            if substance is None:
                continue
            for element, count in substance.formula.items():
                totals[element] = totals.get(element, 0.0) + amount * count
        return totals


def _is_missing_or_normalized(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    resolved = float(value)
    return isfinite(resolved) and 0.0 <= resolved <= 1.0


def _is_canonical_unit(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        canonical_unit(value)
    except ValueError:
        return False
    return True


def _is_finite_nonnegative(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    resolved = float(value)
    return isfinite(resolved) and resolved >= 0.0


def _json_numeric_tree_is_finite(value: Any) -> bool:
    """Validate the JSON-like public signal tree without repairing bad numbers."""

    if value is None or isinstance(value, str | bool):
        return True
    if isinstance(value, Real):
        return isfinite(float(value))
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _json_numeric_tree_is_finite(child)
            for key, child in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return all(_json_numeric_tree_is_finite(child) for child in value)
    return False


__all__ = ["CheckResult", "ConstitutionReport", "PhysicalConstitution"]
