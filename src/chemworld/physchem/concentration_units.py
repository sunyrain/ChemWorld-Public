"""Energy-limited differential batch vacuum concentration with explicit ledgers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, log
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelCard,
    ValidationEvidence,
)

CONCENTRATION_MODEL_ID = "chemworld_vacuum_concentration_vnext"
IDAES_COMMIT = "4275c45bfa76cd5b05926beaa8eee58f7b0b05e8"
IDAES_FLASH_PATH = "idaes/models/unit_models/flash.py"
_SOLVER_RTOL = 1.0e-9
_SOLVER_ATOL = 1.0e-12


def _finite_nonnegative(value: float, label: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return resolved


def _finite_positive(value: float, label: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return resolved


def _closed_fraction(value: float, label: str) -> float:
    resolved = float(value)
    if not isfinite(resolved) or not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{label} must lie in [0, 1]")
    return resolved


def _amounts(values: Mapping[str, float]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for component_id, raw_value in values.items():
        key = str(component_id).strip()
        if not key:
            raise ValueError("feed component ids cannot be empty")
        if key in resolved:
            raise ValueError("feed component ids cannot collide after normalization")
        resolved[key] = _finite_nonnegative(raw_value, f"feed_amounts_mol[{key!r}]")
    if not resolved or sum(resolved.values()) <= 0.0:
        raise ValueError("feed_amounts_mol must contain positive total material")
    return dict(sorted(resolved.items()))


@dataclass(frozen=True)
class ConcentrationComponentSpec:
    """Property values already evaluated at the declared operating condition."""

    component_id: str
    vapor_pressure_Pa: float
    activity_coefficient: float
    latent_heat_J_mol: float
    liquid_heat_capacity_J_mol_K: float
    liquid_molar_volume_L_mol: float
    evaluation_temperature_K: float
    thermal_limit_K: float
    provenance_id: str

    def __post_init__(self) -> None:
        component_id = self.component_id.strip()
        if not component_id or not self.provenance_id.strip():
            raise ValueError("component_id and provenance_id cannot be empty")
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(
            self,
            "vapor_pressure_Pa",
            _finite_nonnegative(self.vapor_pressure_Pa, "vapor_pressure_Pa"),
        )
        for field_name in (
            "activity_coefficient",
            "latent_heat_J_mol",
            "liquid_heat_capacity_J_mol_K",
            "evaluation_temperature_K",
            "thermal_limit_K",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_positive(getattr(self, field_name), field_name),
            )
        # A dilute tracer may have zero apparent volume; the request still
        # requires a positive total liquid volume from its carrier components.
        object.__setattr__(
            self,
            "liquid_molar_volume_L_mol",
            _finite_nonnegative(self.liquid_molar_volume_L_mol, "liquid_molar_volume_L_mol"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "vapor_pressure_Pa": self.vapor_pressure_Pa,
            "activity_coefficient": self.activity_coefficient,
            "latent_heat_J_mol": self.latent_heat_J_mol,
            "liquid_heat_capacity_J_mol_K": self.liquid_heat_capacity_J_mol_K,
            "liquid_molar_volume_L_mol": self.liquid_molar_volume_L_mol,
            "evaluation_temperature_K": self.evaluation_temperature_K,
            "thermal_limit_K": self.thermal_limit_K,
            "provenance_id": self.provenance_id,
        }


@dataclass(frozen=True)
class VacuumConcentratorSpec:
    equipment_id: str
    max_working_volume_L: float
    minimum_residual_volume_L: float
    max_heater_power_W: float
    max_evaporation_rate_mol_s: float
    condenser_recovery_fraction: float
    minimum_pressure_Pa: float
    maximum_temperature_K: float
    maximum_duration_s: float
    provenance_id: str

    def __post_init__(self) -> None:
        if not self.equipment_id.strip() or not self.provenance_id.strip():
            raise ValueError("equipment_id and provenance_id cannot be empty")
        max_volume = _finite_positive(
            self.max_working_volume_L,
            "max_working_volume_L",
        )
        minimum_volume = _finite_nonnegative(
            self.minimum_residual_volume_L,
            "minimum_residual_volume_L",
        )
        if minimum_volume >= max_volume:
            raise ValueError("minimum residual volume must be below maximum working volume")
        object.__setattr__(self, "max_working_volume_L", max_volume)
        object.__setattr__(self, "minimum_residual_volume_L", minimum_volume)
        object.__setattr__(
            self,
            "max_heater_power_W",
            _finite_nonnegative(self.max_heater_power_W, "max_heater_power_W"),
        )
        object.__setattr__(
            self,
            "max_evaporation_rate_mol_s",
            _finite_nonnegative(
                self.max_evaporation_rate_mol_s,
                "max_evaporation_rate_mol_s",
            ),
        )
        object.__setattr__(
            self,
            "condenser_recovery_fraction",
            _closed_fraction(
                self.condenser_recovery_fraction,
                "condenser_recovery_fraction",
            ),
        )
        for field_name in (
            "minimum_pressure_Pa",
            "maximum_temperature_K",
            "maximum_duration_s",
        ):
            object.__setattr__(
                self,
                field_name,
                _finite_positive(getattr(self, field_name), field_name),
            )


@dataclass(frozen=True)
class VacuumConcentrationRequest:
    feed_amounts_mol: Mapping[str, float]
    component_specs: Mapping[str, ConcentrationComponentSpec]
    target_component_id: str
    solvent_component_ids: tuple[str, ...]
    initial_temperature_K: float
    operating_temperature_K: float
    pressure_Pa: float
    duration_s: float
    heater_power_W: float
    equipment: VacuumConcentratorSpec
    target_solvent_remaining_fraction: float = 0.20
    minimum_target_recovery: float = 0.95
    balance_tolerance: float = 1.0e-9

    def __post_init__(self) -> None:
        feed = _amounts(self.feed_amounts_mol)
        target_id = self.target_component_id.strip()
        if not target_id or target_id not in feed or feed[target_id] <= 0.0:
            raise ValueError("target_component_id must have positive feed inventory")
        solvent_ids = tuple(value.strip() for value in self.solvent_component_ids)
        if not solvent_ids or any(not value for value in solvent_ids):
            raise ValueError("solvent_component_ids must contain non-empty ids")
        if len(set(solvent_ids)) != len(solvent_ids):
            raise ValueError("solvent_component_ids cannot contain duplicates")
        if target_id in solvent_ids:
            raise ValueError("target_component_id cannot also be a solvent component")
        missing_solvents = sorted(set(solvent_ids) - set(feed))
        if missing_solvents:
            raise ValueError(f"solvent components are absent from feed: {missing_solvents}")
        if sum(feed[key] for key in solvent_ids) <= 0.0:
            raise ValueError("solvent components must have positive total inventory")

        specs = {str(key).strip(): value for key, value in self.component_specs.items()}
        if set(specs) != set(feed):
            raise ValueError("component_specs keys must exactly match feed components")
        for key, spec in specs.items():
            if not isinstance(spec, ConcentrationComponentSpec):
                raise ValueError("component_specs must contain ConcentrationComponentSpec values")
            if key != spec.component_id:
                raise ValueError("component_specs keys must match embedded component_id values")
        initial_temperature = _finite_positive(
            self.initial_temperature_K,
            "initial_temperature_K",
        )
        operating_temperature = _finite_positive(
            self.operating_temperature_K,
            "operating_temperature_K",
        )
        if operating_temperature < initial_temperature:
            raise ValueError("operating_temperature_K cannot be below initial_temperature_K")
        mismatched_property_temperatures = [
            key
            for key, spec in specs.items()
            if abs(spec.evaluation_temperature_K - operating_temperature) > 1.0e-8
        ]
        if mismatched_property_temperatures:
            raise ValueError(
                "component property profiles must be evaluated at "
                "operating_temperature_K: "
                f"{mismatched_property_temperatures}"
            )
        pressure = _finite_positive(self.pressure_Pa, "pressure_Pa")
        duration = _finite_nonnegative(self.duration_s, "duration_s")
        heater_power = _finite_nonnegative(self.heater_power_W, "heater_power_W")
        endpoint = _closed_fraction(
            self.target_solvent_remaining_fraction,
            "target_solvent_remaining_fraction",
        )
        minimum_recovery = _closed_fraction(
            self.minimum_target_recovery,
            "minimum_target_recovery",
        )
        tolerance = _finite_positive(self.balance_tolerance, "balance_tolerance")
        if pressure + tolerance < self.equipment.minimum_pressure_Pa:
            raise ValueError("pressure_Pa is below the equipment vacuum limit")
        if operating_temperature > self.equipment.maximum_temperature_K + tolerance:
            raise ValueError("operating temperature exceeds the equipment maximum")
        if duration > self.equipment.maximum_duration_s + tolerance:
            raise ValueError("duration_s exceeds the equipment maximum")
        if heater_power > self.equipment.max_heater_power_W + tolerance:
            raise ValueError("heater_power_W exceeds the equipment maximum")
        exceeded_thermal_limits = [
            key
            for key, spec in specs.items()
            if feed[key] > 0.0 and operating_temperature > spec.thermal_limit_K + tolerance
        ]
        if exceeded_thermal_limits:
            raise ValueError(
                f"operating temperature exceeds component thermal limits: {exceeded_thermal_limits}"
            )
        initial_volume = sum(feed[key] * specs[key].liquid_molar_volume_L_mol for key in feed)
        if initial_volume > self.equipment.max_working_volume_L + tolerance:
            raise ValueError("feed equivalent liquid volume exceeds equipment capacity")
        if initial_volume + tolerance < self.equipment.minimum_residual_volume_L:
            raise ValueError("feed volume is below the equipment minimum residual volume")
        object.__setattr__(self, "feed_amounts_mol", feed)
        object.__setattr__(self, "component_specs", dict(sorted(specs.items())))
        object.__setattr__(self, "target_component_id", target_id)
        object.__setattr__(self, "solvent_component_ids", solvent_ids)
        object.__setattr__(self, "initial_temperature_K", initial_temperature)
        object.__setattr__(self, "operating_temperature_K", operating_temperature)
        object.__setattr__(self, "pressure_Pa", pressure)
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "heater_power_W", heater_power)
        object.__setattr__(self, "target_solvent_remaining_fraction", endpoint)
        object.__setattr__(self, "minimum_target_recovery", minimum_recovery)
        object.__setattr__(self, "balance_tolerance", tolerance)


@dataclass(frozen=True)
class _VaporState:
    bubble_pressure_Pa: float
    vapor_composition: dict[str, float]
    mixture_latent_heat_J_mol: float


def _vapor_state(
    amounts_mol: Mapping[str, float],
    specs: Mapping[str, ConcentrationComponentSpec],
) -> _VaporState:
    total = sum(max(float(value), 0.0) for value in amounts_mol.values())
    if total <= 0.0:
        return _VaporState(0.0, dict.fromkeys(amounts_mol, 0.0), 0.0)
    contributions = {
        key: (
            max(float(amounts_mol[key]), 0.0)
            / total
            * specs[key].activity_coefficient
            * specs[key].vapor_pressure_Pa
        )
        for key in amounts_mol
    }
    bubble_pressure = sum(contributions.values())
    if bubble_pressure <= 0.0:
        vapor = dict.fromkeys(amounts_mol, 0.0)
        latent = 0.0
    else:
        vapor = {key: contribution / bubble_pressure for key, contribution in contributions.items()}
        latent = sum(vapor[key] * specs[key].latent_heat_J_mol for key in vapor)
    return _VaporState(bubble_pressure, vapor, latent)


def _equivalent_liquid_volume_l(
    amounts_mol: Mapping[str, float],
    specs: Mapping[str, ConcentrationComponentSpec],
) -> float:
    return sum(
        max(float(amounts_mol.get(key, 0.0)), 0.0) * spec.liquid_molar_volume_L_mol
        for key, spec in specs.items()
    )


@dataclass(frozen=True)
class VacuumConcentrationResult:
    model_id: str
    equipment_id: str
    feed_amounts_mol: dict[str, float]
    liquid_amounts_mol: dict[str, float]
    condensate_amounts_mol: dict[str, float]
    vent_amounts_mol: dict[str, float]
    evaporated_amounts_mol: dict[str, float]
    initial_equivalent_liquid_volume_L: float
    final_equivalent_liquid_volume_L: float
    condensate_equivalent_liquid_volume_L: float
    vent_equivalent_liquid_volume_L: float
    initial_temperature_K: float
    final_temperature_K: float
    operating_temperature_K: float
    pressure_Pa: float
    initial_bubble_pressure_Pa: float
    final_bubble_pressure_Pa: float
    requested_duration_s: float
    elapsed_time_s: float
    heating_time_s: float
    boiling_time_s: float
    sensible_energy_J: float
    latent_energy_J: float
    heat_duty_J: float
    available_heater_energy_J: float
    heater_energy_utilization: float
    average_evaporation_rate_mol_s: float
    initial_solvent_amount_mol: float
    final_solvent_amount_mol: float
    solvent_remaining_fraction: float
    target_solvent_remaining_fraction: float
    endpoint_met: bool
    target_recovery: float
    minimum_target_recovery: float
    target_recovery_constraint_met: bool
    target_concentration_factor: float
    minimum_thermal_margin_K: float
    termination_reason: str
    solver_success: bool
    solver_steps: int
    component_balance_error_mol: dict[str, float]
    material_balance_error_mol: float
    volume_balance_error_L: float
    energy_balance_error_J: float
    warnings: tuple[str, ...]
    provenance: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "equipment_id": self.equipment_id,
            "feed_amounts_mol": dict(self.feed_amounts_mol),
            "liquid_amounts_mol": dict(self.liquid_amounts_mol),
            "condensate_amounts_mol": dict(self.condensate_amounts_mol),
            "vent_amounts_mol": dict(self.vent_amounts_mol),
            "evaporated_amounts_mol": dict(self.evaporated_amounts_mol),
            "initial_equivalent_liquid_volume_L": self.initial_equivalent_liquid_volume_L,
            "final_equivalent_liquid_volume_L": self.final_equivalent_liquid_volume_L,
            "condensate_equivalent_liquid_volume_L": (self.condensate_equivalent_liquid_volume_L),
            "vent_equivalent_liquid_volume_L": self.vent_equivalent_liquid_volume_L,
            "initial_temperature_K": self.initial_temperature_K,
            "final_temperature_K": self.final_temperature_K,
            "operating_temperature_K": self.operating_temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "initial_bubble_pressure_Pa": self.initial_bubble_pressure_Pa,
            "final_bubble_pressure_Pa": self.final_bubble_pressure_Pa,
            "requested_duration_s": self.requested_duration_s,
            "elapsed_time_s": self.elapsed_time_s,
            "heating_time_s": self.heating_time_s,
            "boiling_time_s": self.boiling_time_s,
            "sensible_energy_J": self.sensible_energy_J,
            "latent_energy_J": self.latent_energy_J,
            "heat_duty_J": self.heat_duty_J,
            "available_heater_energy_J": self.available_heater_energy_J,
            "heater_energy_utilization": self.heater_energy_utilization,
            "average_evaporation_rate_mol_s": self.average_evaporation_rate_mol_s,
            "initial_solvent_amount_mol": self.initial_solvent_amount_mol,
            "final_solvent_amount_mol": self.final_solvent_amount_mol,
            "solvent_remaining_fraction": self.solvent_remaining_fraction,
            "target_solvent_remaining_fraction": (self.target_solvent_remaining_fraction),
            "endpoint_met": self.endpoint_met,
            "target_recovery": self.target_recovery,
            "minimum_target_recovery": self.minimum_target_recovery,
            "target_recovery_constraint_met": self.target_recovery_constraint_met,
            "target_concentration_factor": self.target_concentration_factor,
            "minimum_thermal_margin_K": self.minimum_thermal_margin_K,
            "termination_reason": self.termination_reason,
            "solver_success": self.solver_success,
            "solver_steps": self.solver_steps,
            "component_balance_error_mol": dict(self.component_balance_error_mol),
            "material_balance_error_mol": self.material_balance_error_mol,
            "volume_balance_error_L": self.volume_balance_error_L,
            "energy_balance_error_J": self.energy_balance_error_J,
            "warnings": list(self.warnings),
            "provenance": list(self.provenance),
        }


def simulate_vacuum_concentration(
    request: VacuumConcentrationRequest,
) -> VacuumConcentrationResult:
    """Run a controlled differential batch evaporation at fixed pressure and target T."""

    component_ids = tuple(request.feed_amounts_mol)
    specs = request.component_specs
    feed = dict(request.feed_amounts_mol)
    equipment = request.equipment
    initial_volume = _equivalent_liquid_volume_l(feed, specs)
    initial_solvent = sum(feed[key] for key in request.solvent_component_ids)
    initial_vapor = _vapor_state(feed, specs)
    heat_capacity = sum(
        feed[key] * specs[key].liquid_heat_capacity_J_mol_K for key in component_ids
    )
    sensible_required = heat_capacity * (
        request.operating_temperature_K - request.initial_temperature_K
    )
    final = dict(feed)
    final_temperature = request.initial_temperature_K
    sensible_energy = 0.0
    heating_time = 0.0
    boiling_time = 0.0
    elapsed_time = 0.0
    solver_success = True
    solver_steps = 0
    termination_reason = "duration_exhausted"
    warnings: list[str] = []

    initial_endpoint_met = (
        request.target_solvent_remaining_fraction + request.balance_tolerance >= 1.0
    )
    target_is_volatile = specs[request.target_component_id].vapor_pressure_Pa > 0.0
    target_constraint_initially_binding = (
        target_is_volatile and request.minimum_target_recovery >= 1.0 - request.balance_tolerance
    )
    minimum_volume_initially_binding = (
        initial_volume <= equipment.minimum_residual_volume_L + request.balance_tolerance
    )
    if request.duration_s <= request.balance_tolerance:
        termination_reason = "zero_duration"
    elif initial_endpoint_met:
        termination_reason = "solvent_endpoint"
    elif target_constraint_initially_binding:
        termination_reason = "target_recovery_limit"
    elif minimum_volume_initially_binding:
        termination_reason = "minimum_residual_volume"
    elif request.heater_power_W <= request.balance_tolerance:
        termination_reason = "zero_heater_power"
        elapsed_time = request.duration_s
        warnings.append("zero heater power prevents heating and evaporation")
    else:
        available_energy = request.heater_power_W * request.duration_s
        if sensible_required > available_energy + request.balance_tolerance:
            sensible_energy = available_energy
            final_temperature = request.initial_temperature_K + sensible_energy / heat_capacity
            heating_time = request.duration_s
            elapsed_time = request.duration_s
            termination_reason = "heating_incomplete"
            warnings.append("operating temperature was not reached within the duration")
        else:
            sensible_energy = sensible_required
            heating_time = (
                0.0
                if sensible_required <= request.balance_tolerance
                else sensible_required / request.heater_power_W
            )
            final_temperature = request.operating_temperature_K
            available_boiling_time = max(request.duration_s - heating_time, 0.0)
            elapsed_time = request.duration_s
            if (
                available_boiling_time <= request.balance_tolerance
                or equipment.max_evaporation_rate_mol_s <= request.balance_tolerance
            ):
                termination_reason = (
                    "zero_evaporation_capacity"
                    if available_boiling_time > request.balance_tolerance
                    else "duration_exhausted"
                )
                if equipment.max_evaporation_rate_mol_s <= request.balance_tolerance:
                    warnings.append("equipment evaporation-rate capacity is zero")
            elif initial_vapor.bubble_pressure_Pa <= (
                request.pressure_Pa + request.balance_tolerance
            ):
                termination_reason = "equilibrium_pressure"
                warnings.append("bubble pressure does not exceed operating pressure")
            else:
                indices = {key: index for index, key in enumerate(component_ids)}
                initial_vector = np.array(
                    [feed[key] for key in component_ids],
                    dtype=float,
                )

                def vector_map(values: np.ndarray) -> dict[str, float]:
                    return {key: max(float(values[index]), 0.0) for key, index in indices.items()}

                def derivative(_time: float, values: np.ndarray) -> np.ndarray:
                    state = _vapor_state(vector_map(values), specs)
                    if (
                        state.bubble_pressure_Pa <= request.pressure_Pa
                        or state.mixture_latent_heat_J_mol <= 0.0
                    ):
                        return np.zeros_like(values)
                    rate = min(
                        request.heater_power_W / state.mixture_latent_heat_J_mol,
                        equipment.max_evaporation_rate_mol_s,
                    )
                    return np.array(
                        [-rate * state.vapor_composition[key] for key in component_ids],
                        dtype=float,
                    )

                event_names: list[str] = []
                events: list[Any] = []

                def add_event(name: str, event: Any) -> None:
                    event.terminal = True
                    event.direction = -1.0
                    event_names.append(name)
                    events.append(event)

                def solvent_event(_time: float, values: np.ndarray) -> float:
                    state = vector_map(values)
                    return (
                        sum(state[key] for key in request.solvent_component_ids) / initial_solvent
                        - request.target_solvent_remaining_fraction
                    )

                add_event("solvent_endpoint", solvent_event)

                if (
                    target_is_volatile
                    and request.minimum_target_recovery > request.balance_tolerance
                ):

                    def target_event(_time: float, values: np.ndarray) -> float:
                        return (
                            max(float(values[indices[request.target_component_id]]), 0.0)
                            / feed[request.target_component_id]
                            - request.minimum_target_recovery
                        )

                    add_event("target_recovery_limit", target_event)

                if equipment.minimum_residual_volume_L > request.balance_tolerance:

                    def volume_event(_time: float, values: np.ndarray) -> float:
                        return (
                            _equivalent_liquid_volume_l(vector_map(values), specs)
                            - equipment.minimum_residual_volume_L
                        )

                    add_event("minimum_residual_volume", volume_event)

                def pressure_event(_time: float, values: np.ndarray) -> float:
                    return (
                        _vapor_state(vector_map(values), specs).bubble_pressure_Pa
                        - request.pressure_Pa
                    )

                add_event("equilibrium_pressure", pressure_event)
                solution = solve_ivp(
                    derivative,
                    (0.0, available_boiling_time),
                    initial_vector,
                    method="RK45",
                    events=events,
                    rtol=_SOLVER_RTOL,
                    atol=_SOLVER_ATOL,
                    max_step=max(available_boiling_time / 128.0, 1.0e-6),
                )
                solver_success = bool(solution.success)
                solver_steps = len(solution.t)
                if not solver_success:
                    raise RuntimeError(f"vacuum concentration solver failed: {solution.message}")
                final = vector_map(solution.y[:, -1])
                boiling_time = float(solution.t[-1])
                elapsed_time = heating_time + boiling_time
                termination_reason = "duration_exhausted"
                for name, event_times in zip(event_names, solution.t_events, strict=True):
                    if len(event_times):
                        termination_reason = name
                        break

    evaporated = {key: max(feed[key] - final[key], 0.0) for key in component_ids}
    condenser_fraction = equipment.condenser_recovery_fraction
    condensate = {key: evaporated[key] * condenser_fraction for key in component_ids}
    vent = {key: evaporated[key] - condensate[key] for key in component_ids}
    final_volume = _equivalent_liquid_volume_l(final, specs)
    condensate_volume = _equivalent_liquid_volume_l(condensate, specs)
    vent_volume = _equivalent_liquid_volume_l(vent, specs)
    latent_energy = sum(evaporated[key] * specs[key].latent_heat_J_mol for key in component_ids)
    heat_duty = sensible_energy + latent_energy
    available_heater_energy = request.heater_power_W * request.duration_s
    if heat_duty > available_heater_energy + max(
        request.balance_tolerance,
        available_heater_energy * 1.0e-8,
    ):
        raise RuntimeError("concentration heat duty exceeded available heater energy")
    energy_utilization = (
        heat_duty / available_heater_energy
        if available_heater_energy > request.balance_tolerance
        else 0.0
    )
    evaporated_total = sum(evaporated.values())
    average_evaporation_rate = (
        evaporated_total / boiling_time if boiling_time > request.balance_tolerance else 0.0
    )
    final_solvent = sum(final[key] for key in request.solvent_component_ids)
    solvent_remaining = final_solvent / initial_solvent
    endpoint_met = (
        solvent_remaining <= request.target_solvent_remaining_fraction + request.balance_tolerance
    )
    target_recovery = final[request.target_component_id] / feed[request.target_component_id]
    target_constraint_met = (
        target_recovery + request.balance_tolerance >= request.minimum_target_recovery
    )
    initial_target_concentration = feed[request.target_component_id] / initial_volume
    final_target_concentration = final[request.target_component_id] / max(
        final_volume,
        1.0e-30,
    )
    concentration_factor = final_target_concentration / initial_target_concentration
    final_vapor = _vapor_state(final, specs)
    minimum_thermal_margin = min(
        specs[key].thermal_limit_K - request.operating_temperature_K
        for key in component_ids
        if feed[key] > 0.0
    )

    component_errors = {
        key: abs(feed[key] - final[key] - condensate[key] - vent[key]) for key in component_ids
    }
    material_error = sum(component_errors.values())
    volume_error = abs(initial_volume - final_volume - condensate_volume - vent_volume)
    energy_error = abs(heat_duty - sensible_energy - latent_energy)
    if material_error > request.balance_tolerance or volume_error > request.balance_tolerance:
        raise RuntimeError(
            "vacuum concentration control volume failed closure: "
            f"material={material_error}, volume={volume_error}"
        )
    if energy_error > request.balance_tolerance:
        raise RuntimeError(f"vacuum concentration energy ledger failed closure: {energy_error}")

    if not endpoint_met:
        warnings.append("declared solvent-removal endpoint was not met")
    if not target_constraint_met:
        warnings.append("minimum target recovery constraint was violated")
    if target_recovery < 1.0 - request.balance_tolerance:
        warnings.append("volatile target loss is explicit in condensate and vent ledgers")
    if condenser_fraction < 1.0 and evaporated_total > request.balance_tolerance:
        warnings.append("unrecovered vapor is explicit in the vent-loss ledger")
    if minimum_thermal_margin < 5.0:
        warnings.append("operating temperature is within 5 K of a component thermal limit")
    if termination_reason == "minimum_residual_volume":
        warnings.append("evaporation stopped at the equipment minimum residual volume")
    if termination_reason == "target_recovery_limit":
        warnings.append("evaporation stopped at the minimum target recovery")
    if termination_reason == "duration_exhausted" and not endpoint_met:
        warnings.append("available processing duration was exhausted before the endpoint")

    return VacuumConcentrationResult(
        model_id=CONCENTRATION_MODEL_ID,
        equipment_id=equipment.equipment_id,
        feed_amounts_mol=feed,
        liquid_amounts_mol=final,
        condensate_amounts_mol=condensate,
        vent_amounts_mol=vent,
        evaporated_amounts_mol=evaporated,
        initial_equivalent_liquid_volume_L=initial_volume,
        final_equivalent_liquid_volume_L=final_volume,
        condensate_equivalent_liquid_volume_L=condensate_volume,
        vent_equivalent_liquid_volume_L=vent_volume,
        initial_temperature_K=request.initial_temperature_K,
        final_temperature_K=final_temperature,
        operating_temperature_K=request.operating_temperature_K,
        pressure_Pa=request.pressure_Pa,
        initial_bubble_pressure_Pa=initial_vapor.bubble_pressure_Pa,
        final_bubble_pressure_Pa=final_vapor.bubble_pressure_Pa,
        requested_duration_s=request.duration_s,
        elapsed_time_s=elapsed_time,
        heating_time_s=heating_time,
        boiling_time_s=boiling_time,
        sensible_energy_J=sensible_energy,
        latent_energy_J=latent_energy,
        heat_duty_J=heat_duty,
        available_heater_energy_J=available_heater_energy,
        heater_energy_utilization=energy_utilization,
        average_evaporation_rate_mol_s=average_evaporation_rate,
        initial_solvent_amount_mol=initial_solvent,
        final_solvent_amount_mol=final_solvent,
        solvent_remaining_fraction=solvent_remaining,
        target_solvent_remaining_fraction=request.target_solvent_remaining_fraction,
        endpoint_met=endpoint_met,
        target_recovery=target_recovery,
        minimum_target_recovery=request.minimum_target_recovery,
        target_recovery_constraint_met=target_constraint_met,
        target_concentration_factor=concentration_factor,
        minimum_thermal_margin_K=minimum_thermal_margin,
        termination_reason=termination_reason,
        solver_success=solver_success,
        solver_steps=solver_steps,
        component_balance_error_mol=component_errors,
        material_balance_error_mol=material_error,
        volume_balance_error_L=volume_error,
        energy_balance_error_J=energy_error,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=(
            "differential batch Rayleigh evaporation with fixed declared activity coefficients",
            "heater-power and equipment-rate constrained latent-energy ledger",
            "explicit recovered-condensate and unrecovered-vent material ledgers",
            (
                f"IDAES {IDAES_COMMIT}:{IDAES_FLASH_PATH} component material, "
                "energy, heat-duty, and pressure-balance convention boundary"
            ),
        ),
    )


def binary_rayleigh_residual(
    *,
    initial_total_mol: float,
    final_total_mol: float,
    initial_light_fraction: float,
    final_light_fraction: float,
    relative_volatility: float,
) -> float:
    """Return the closed-form ideal binary Rayleigh identity residual."""

    for value, label in (
        (initial_total_mol, "initial_total_mol"),
        (final_total_mol, "final_total_mol"),
    ):
        _finite_positive(value, label)
    for value, label in (
        (initial_light_fraction, "initial_light_fraction"),
        (final_light_fraction, "final_light_fraction"),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{label} must lie in (0, 1)")
    alpha = _finite_positive(relative_volatility, "relative_volatility")
    if abs(alpha - 1.0) <= 1.0e-12:
        raise ValueError("relative_volatility must differ from one")
    primitive_initial = (
        log(initial_light_fraction) - alpha * log(1.0 - initial_light_fraction)
    ) / (alpha - 1.0)
    primitive_final = (log(final_light_fraction) - alpha * log(1.0 - final_light_fraction)) / (
        alpha - 1.0
    )
    return abs(log(initial_total_mol / final_total_mol) - (primitive_initial - primitive_final))


def vacuum_concentration_model_card() -> ModelCard:
    return ModelCard(
        model_id=CONCENTRATION_MODEL_ID,
        module_id="separations",
        title="Energy-Limited Differential Batch Vacuum Concentration",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        summary=(
            "A fixed-pressure differential batch evaporator using declared "
            "gamma-Raoult volatility, sensible/latent energy, equipment limits, "
            "condenser recovery, and event-driven endpoint protection."
        ),
        equations=(
            "P_bubble = sum_i x_i gamma_i P_i^sat",
            "y_i = x_i gamma_i P_i^sat / P_bubble",
            "dn_i/dt = -min(Qdot/lambda_mix, r_max) y_i when P_bubble > P",
            "Q = sum_i n_i Cp_i DeltaT + sum_i n_i,evap lambda_i",
            "n_i,feed = n_i,liquid + n_i,condensate + n_i,vent",
            "binary ideal limit satisfies the closed-form Rayleigh identity",
        ),
        assumptions=(
            "one perfectly mixed liquid phase and equilibrium vapor leaving differentially",
            (
                "fixed operating pressure, target temperature, activity "
                "coefficients, and property values"
            ),
            "ideal vapor and no Poynting correction or pressure drop within the bounded slice",
            "heater throttles after an event and condenser recovery is component independent",
        ),
        validity_limits=(
            (
                "declared property profiles only; correlations must be evaluated "
                "upstream at the operating condition"
            ),
            (
                "no precipitation, foaming, bumping, reaction, degradation "
                "kinetics, or viscosity feedback"
            ),
            (
                "no falling-film geometry, mass-transfer resistance, vapor "
                "holdup, or condenser thermal model"
            ),
            "thermal limits are hard operating boundaries, not degradation-rate predictions",
        ),
        failure_modes=(
            (
                "invalid property, equipment, temperature, pressure, duration, "
                "capacity, or endpoint inputs are rejected"
            ),
            (
                "operation above component/equipment thermal limits or below the "
                "vacuum limit is rejected"
            ),
            (
                "ODE failure, heater-energy overspend, or material/volume/energy "
                "non-closure is a hard failure"
            ),
            (
                "unmet solvent endpoint, volatile-target loss, and condenser vent "
                "loss remain explicit warnings"
            ),
        ),
        units={
            "component amount": "mol",
            "equivalent liquid volume": "L",
            "temperature": "K",
            "pressure and vapor pressure": "Pa",
            "heater power": "W",
            "energy and heat duty": "J",
            "latent heat": "J/mol",
            "heat capacity": "J/(mol K)",
            "evaporation rate": "mol/s",
        },
        reference_reading=(
            (
                f"IDAES {IDAES_COMMIT}:{IDAES_FLASH_PATH} control-volume "
                "material/energy, heat-duty, and pressure conventions"
            ),
            "differential batch distillation and the binary Rayleigh equation",
            "gamma-Raoult bubble-pressure and equilibrium-vapor identities",
        ),
        validation_evidence=(
            ValidationEvidence(
                evidence_id="concentration-single-volatile-energy-limit",
                evidence_type="analytic_test",
                description=(
                    "A single volatile solvent with a nonvolatile target is checked "
                    "against the exact power-times-time divided by latent heat limit."
                ),
                status="implemented",
                reference_backend="closed-form sensible and latent energy identity",
                command_or_path="tests/test_concentration_units.py",
                tolerance="1e-9 mol and J",
            ),
            ValidationEvidence(
                evidence_id="concentration-binary-rayleigh",
                evidence_type="analytic_test",
                description=(
                    "The ideal binary differential trajectory is checked against "
                    "the closed-form constant-relative-volatility Rayleigh identity."
                ),
                status="implemented",
                reference_backend="closed-form binary Rayleigh equation",
                command_or_path="tests/test_concentration_units.py",
                tolerance="2e-7 dimensionless residual",
            ),
            ValidationEvidence(
                evidence_id="concentration-ledger-domain-sweep",
                evidence_type="invariant_test",
                description=(
                    "Deterministic multicomponent cases check component, equivalent-"
                    "volume, heater-energy, endpoint, and equipment invariants."
                ),
                status="implemented",
                reference_backend="analytic control-volume identities",
                command_or_path="tests/test_concentration_units.py",
                tolerance="1e-8 mol, L, and J",
            ),
        ),
        model_limit_notes=(
            (
                "Reference validation covers the bounded differential batch "
                "ledger, not evaporator scale-up or plant safety."
            ),
            (
                "This proposal does not replace the distillation operation and "
                "does not alter the v0.3 concentrate route."
            ),
        ),
        intended_use=(
            "World Law vNext concentrate-operation candidate with declared property cards",
            "agent trade-offs among vacuum, time, energy, solvent recovery, and product loss",
            "material, equivalent-volume, energy, endpoint, and condenser-ledger evaluation",
        ),
    )


__all__ = [
    "CONCENTRATION_MODEL_ID",
    "IDAES_COMMIT",
    "IDAES_FLASH_PATH",
    "ConcentrationComponentSpec",
    "VacuumConcentrationRequest",
    "VacuumConcentrationResult",
    "VacuumConcentratorSpec",
    "binary_rayleigh_residual",
    "simulate_vacuum_concentration",
    "vacuum_concentration_model_card",
]
