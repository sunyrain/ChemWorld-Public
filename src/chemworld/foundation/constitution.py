"""Executable physical constitution for ChemWorld environments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Any, ClassVar

from chemworld.foundation.balances import TransitionBalance
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
from chemworld.foundation.solvents import SOLVENT_IDS, total_solvents
from chemworld.foundation.state import Observation, WorldState
from chemworld.foundation.units import unit_spec


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
        native_errors = observation.errors()
        checks.append(CheckResult(
            "observation_native_channels", not native_errors, "; ".join(native_errors),
        ))
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
                "Legacy Gym projections must be missing or finite values in [0, 1].",
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
        processed_valid = _json_numeric_tree_is_finite(observation.processed_estimate)
        uncertainty_valid = all(
            _is_finite_nonnegative(value) for value in observation.uncertainty.values()
        )
        raw_signal_valid = _json_numeric_tree_is_finite(observation.raw_signal)
        checks.extend(
            (
                CheckResult(
                    "observation_processed_estimate_finite_and_bounded",
                    processed_valid,
                    "Native processed estimates must be missing or finite in their declared units.",
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
                    _is_supported_observation_unit(unit),
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
        allowed_owned_delta: dict[tuple[str,str],float] | None = None,
    ) -> CheckResult:
        allowed_element_delta = allowed_element_delta or {}
        allowed_owned_delta = allowed_owned_delta or {}
        before_amounts = self.material_inventory(before)
        after_amounts = self.material_inventory(after)
        unknown_species = (before_amounts.keys() | after_amounts.keys()) - self.substances.keys()
        if unknown_species:
            return CheckResult(
                "material_conservation",
                False,
                "Material conservation cannot be established for unregistered species.",
                value=float(len(unknown_species)),
                tolerance=0.0,
            )
        before_elements = self.element_totals(before_amounts)
        after_elements = self.element_totals(after_amounts)
        if not all(isfinite(value) for values in (before_elements, after_elements,
                                                  allowed_element_delta)
                   for value in values.values()):
            return CheckResult("material_conservation", False, "nonfinite material balance")
        errors: list[float] = []
        elements = before_elements.keys() | after_elements.keys() | allowed_element_delta.keys()
        for element in sorted(elements):
            expected = before_elements.get(element, 0.0) + allowed_element_delta.get(element, 0.0)
            errors.append(abs(after_elements.get(element, 0.0) - expected))
        max_error = max(errors) if errors else 0.0
        # Declared entity quantities are closed unless an installed producer
        # supplies an explicit typed external packet (for example disposable kg).
        owned_before = self.owned_inventory(before)
        owned_after = self.owned_inventory(after)
        if any(not isfinite(v) for values in (owned_before,owned_after,allowed_owned_delta)
               for v in values.values()):
            return CheckResult("material_conservation",False,"nonfinite entity balance")
        owned_keys = owned_before.keys() | owned_after.keys() | allowed_owned_delta.keys()
        changed = [key for key in owned_keys
                   if abs(owned_after.get(key, 0) - owned_before.get(key, 0)
                          - allowed_owned_delta.get(key,0))
                   > 1e-12 + 1e-9 * abs(owned_before.get(key, 0))]
        return CheckResult(
            "material_conservation",
            max_error <= self.tolerance and not changed,
            f"max element-balance error={max_error:.3g}; "
            f"changed entity inventory count={len(changed)}",
            value=max_error,
            tolerance=self.tolerance,
        )

    @staticmethod
    def material_inventory(state: WorldState) -> dict[str, float]:
        """Registered species across every vessel and retained sample, once each."""
        amounts = state.species_amounts.copy()
        inventories = [state.samples.total_amounts_mol(), state.state_slots.species_amounts()]
        inventories.extend(
            PhysicalConstitution.material_inventory(state.inactive_vessels[name])
            for name in sorted(state.inactive_vessels)
        )
        for inventory in inventories:
            for species, amount in inventory.items():
                amounts[species] = amounts.get(species, 0.0) + amount
        return amounts

    @staticmethod
    def owned_inventory(state: WorldState) -> dict[tuple[str, str], float]:
        inventories = [state.state_slots.conserved()]
        inventories.extend(sample.contents.state_slots.conserved()
                           for sample in state.samples.samples.values() if sample.contents)
        inventories.extend(PhysicalConstitution.owned_inventory(local)
                           for local in state.inactive_vessels.values())
        result: dict[tuple[str, str], float] = {}
        for inventory in inventories:
            for key, value in inventory.items():
                result[key] = result.get(key, 0) + value
        return result

    def check_transition(
        self, before: WorldState, after: WorldState, balance: TransitionBalance
    ) -> CheckResult:
        from chemworld.foundation.discrete_inventory import integer_element_residual
        from chemworld.foundation.empirical_inventory import check_empirical_transitions

        try:
            check_empirical_transitions(before, after, balance)
            atoms = integer_element_residual(before, after, balance, self.substances)
        except (ValueError, KeyError) as exc:
            return CheckResult("material_conservation", False, str(exc), tolerance=0)
        if atoms is not None and atoms != 0:
            return CheckResult("material_conservation", False,
                               "integer atom balance does not close", value=atoms, tolerance=0)
        delta: dict[str, float] = {}
        entity_delta = {}
        for exchange in balance.exchanges:
            if exchange.inventory_kind == "entity":
                known = self.owned_inventory(before).keys() | self.owned_inventory(after).keys()
                for quantity,amount in exchange.amounts:
                    key = (quantity,exchange.unit)
                    if key not in known:
                        return CheckResult("material_conservation",False,
                                           "undeclared entity quantity/unit in boundary exchange")
                    entity_delta[key] = entity_delta.get(key,0)+(
                        amount if exchange.direction == "in" else -amount)
            if exchange.inventory_kind != "species":
                continue
            if any(key not in self.substances for key, _ in exchange.amounts):
                return CheckResult("material_conservation", False,
                                   "unregistered species in boundary exchange")
            sign = 1.0 if exchange.direction == "in" else -1.0
            for element, amount in self.element_totals(dict(exchange.amounts)).items():
                delta[element] = delta.get(element, 0.0) + sign * amount
        return self.check_material_conservation(before, after, allowed_element_delta=delta,
                                               allowed_owned_delta=entity_delta)

    def check_carrier_transition(
        self, before: WorldState, after: WorldState, balance: TransitionBalance
    ) -> CheckResult:
        delta = dict.fromkeys(SOLVENT_IDS, 0.0)
        for exchange in balance.exchanges:
            if exchange.inventory_kind != "carrier":
                continue
            for key, amount in exchange.amounts:
                if key not in delta:
                    return CheckResult("carrier_conservation", False, "unregistered carrier")
                delta[key] += amount if exchange.direction == "in" else -amount
        residual = max(abs(new - old - delta[key]) for key, old, new in zip(
            SOLVENT_IDS, total_solvents(before).volumes_L, total_solvents(after).volumes_L,
            strict=True,
        ))
        return CheckResult("carrier_conservation", residual <= 1e-10,
                           "finite carrier component volumes close", value=residual,
                           tolerance=1e-10)

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


def _is_supported_observation_unit(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        unit_spec(value)
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
