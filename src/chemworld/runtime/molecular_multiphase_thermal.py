"""Finite thermal devices with general PH/UV closure on the original world."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import fsum, isclose, isfinite
from typing import Any, cast

import numpy as np

from chemworld.foundation.state_ledgers import EquipmentLedger
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_multiphase import (
    MolecularMultiphaseResult,
    MolecularMultiphaseSeed,
    solve_molecular_multiphase,
)
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.molecular_multiphase_inventory import (
    MolecularMultiphaseInventorySpec,
    MolecularMultiphaseInventoryState,
    molecular_multiphase_inventory_state,
)
from chemworld.runtime.molecular_phase_inventory import _install_molecular_phase_result


@dataclass(frozen=True)
class MolecularMultiphaseThermalSpec:
    inventory: MolecularMultiphaseInventorySpec
    boundary: str
    boundary_value: float
    minimum_temperature_K: float
    stop_temperature_K: float
    support_heat_capacity_J_K: float = 0.0
    support_reference_temperature_K: float = 298.15
    loss_conductance_W_K: float = 0.0
    environment_temperature_K: float = 298.15
    ode_method: str = "DOP853"
    maximum_step_s: float = 20.0

    def __post_init__(self) -> None:
        if (self.boundary not in {"P", "V"} or isinstance(self.boundary_value, bool)
                or not isfinite(self.boundary_value) or self.boundary_value <= 0):
            raise ValueError("multiphase thermal boundary requires positive P in Pa or V in m3")
        low, high = self.inventory.equilibrium.temperature_bounds_K
        if not low < self.minimum_temperature_K < self.stop_temperature_K < high:
            raise ValueError("thermal operating temperatures must lie inside the EOS domain")
        if self.boundary == "P":
            low_p, high_p = self.inventory.equilibrium.pressure_bounds_Pa
            if not low_p < self.boundary_value < high_p:
                raise ValueError("thermal pressure must lie inside the equilibrium domain")
        if (any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in (
                self.support_heat_capacity_J_K, self.loss_conductance_W_K))
                or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (
                    self.support_reference_temperature_K, self.environment_temperature_K,
                    self.maximum_step_s))
                or self.ode_method not in {"BDF", "Radau", "DOP853"}):
            raise ValueError("thermal support, losses and numerical policy must be explicit")

    @property
    def coordinate(self) -> str:
        return "H" if self.boundary == "P" else "U"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "molecular_multiphase_thermal_v1",
            "inventory": self.inventory.to_dict(),
            **{key: getattr(self, key) for key in (
                "boundary", "boundary_value", "minimum_temperature_K", "stop_temperature_K",
                "support_heat_capacity_J_K", "support_reference_temperature_K",
                "loss_conductance_W_K", "environment_temperature_K", "ode_method",
                "maximum_step_s",
            )},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularMultiphaseThermalSpec:
        if data.get("model_type") != "molecular_multiphase_thermal_v1":
            raise ValueError("unknown molecular multiphase thermal declaration")
        values = {k: v for k, v in data.items() if k != "model_type"}
        values["inventory"] = MolecularMultiphaseInventorySpec.from_dict(data["inventory"])
        return cls(**values)


def _inventory(state: Any, spec: MolecularMultiphaseThermalSpec
               ) -> MolecularMultiphaseInventoryState:
    declared = {t.phase_id for t in spec.inventory.phase_targets}
    if state.state_slots.slots or not set(state.phases.phases) <= declared:
        raise ValueError("multiphase thermal vessel contains undeclared owned material")
    stock = molecular_multiphase_inventory_state(
        state.phases, spec.inventory, vessel_id=state.vessel_id,
        source_phase_ids=tuple(state.phases.phases), temperature_K=state.temperature_K,
        pressure_Pa=state.pressure_Pa,
    )
    if (set(stock.feed_amounts_mol) != set(state.species_amounts)
            or any(not isclose(n, state.species_amounts[k], rel_tol=2e-12, abs_tol=0)
                   for k, n in stock.feed_amounts_mol.items())
            or not isclose(stock.volume_m3 * 1000, state.volume_L, rel_tol=2e-12, abs_tol=0)):
        raise ValueError("vessel totals disagree with their actual molecular phases")
    actual = state.pressure_Pa if spec.boundary == "P" else stock.volume_m3
    if not isclose(actual, spec.boundary_value, rel_tol=2e-8, abs_tol=0):
        raise ValueError("actual inventory disagrees with its declared pressure/volume boundary")
    vessel = state.vessels.vessels[state.vessel_id]
    if (not spec.minimum_temperature_K - 1e-7 <= state.temperature_K
            <= min(spec.stop_temperature_K, vessel.max_temperature_K) + 1e-7
            or state.volume_L > vessel.max_volume_L + 1e-10
            or state.pressure_Pa > vessel.max_pressure_Pa * (1 + 1e-10)):
        raise ValueError("molecular inventory exceeds the finite thermal vessel limits")
    return stock


def molecular_multiphase_inventory_energy_j(state: Any, spec: MolecularMultiphaseThermalSpec
                                          ) -> float:
    stock = _inventory(state, spec)
    material = stock.enthalpy_J if spec.boundary == "P" else stock.internal_energy_J
    return material + spec.support_heat_capacity_J_K * (
        state.temperature_K - spec.support_reference_temperature_K
    )


def molecular_multiphase_inventory_enthalpy_j(state: Any, spec: MolecularMultiphaseThermalSpec
                                            ) -> float:
    return _inventory(state, spec).enthalpy_J + spec.support_heat_capacity_J_K * (
        state.temperature_K - spec.support_reference_temperature_K
    )


def _closure(spec: MolecularMultiphaseThermalSpec, feed: Mapping[str, float], value: float,
             seed: MolecularMultiphaseSeed) -> MolecularMultiphaseResult:
    return solve_molecular_multiphase(
        spec.inventory.equilibrium, feed,
        inputs={spec.boundary: spec.boundary_value, spec.coordinate: value}, seed=seed,
        support_heat_capacity_J_K=spec.support_heat_capacity_J_K,
        support_reference_temperature_K=spec.support_reference_temperature_K,
    )


def install_molecular_multiphase_thermal(
    state: Any, spec: MolecularMultiphaseThermalSpec, result: MolecularMultiphaseResult,
) -> Any:
    if not result.success or result.temperature_K is None or result.pressure_Pa is None:
        raise ValueError("cannot install an unresolved molecular multiphase result")
    if not result.stability_checked:
        raise ValueError("cannot install an unchecked molecular multiphase trial")
    installed = _install_molecular_phase_result(
        state.phases, spec.inventory.phase_declaration, result, vessel_id=state.vessel_id,
        source_phase_ids=tuple(state.phases.phases), temperature_K=result.temperature_K,
        pressure_Pa=result.pressure_Pa,
    )
    if not installed.success:
        raise ValueError("molecular phase commit failed: " + installed.failure_reason)
    volume = fsum(p.volume_L for p in installed.ledger.phases.values())
    vessel = state.vessels.vessels[state.vessel_id]
    if (volume > vessel.max_volume_L + 1e-10
            or result.pressure_Pa > vessel.max_pressure_Pa * (1 + 1e-10)
            or not spec.minimum_temperature_K - 1e-7 <= result.temperature_K
            <= min(spec.stop_temperature_K, vessel.max_temperature_K) + 1e-7):
        raise ValueError("molecular multiphase result exceeds finite vessel limits")
    return state.replace(
        phases=installed.ledger,
        species_amounts={k: fsum(p.component_amounts_mol[k] for p in result.phases)
                         for k in spec.inventory.equilibrium.model.component_ids},
        temperature_K=result.temperature_K, pressure_Pa=result.pressure_Pa, volume_L=volume,
        phase="liquid" if any(p.phase_type == "liquid"
                              for p in installed.ledger.phases.values()) else "gas",
        process=replace(state.process, last_observation_record=None),
    )


def validate_molecular_multiphase_thermal_state(
    state: Any, spec: MolecularMultiphaseThermalSpec,
) -> None:
    stock = _inventory(state, spec)
    result = _closure(spec, stock.feed_amounts_mol,
                      molecular_multiphase_inventory_energy_j(state, spec), stock.seed)
    if not result.success:
        raise ValueError("molecular multiphase equilibrium unresolved: " + result.failure_reason)
    candidate = install_molecular_multiphase_thermal(state, spec, result)
    if (not isclose(candidate.temperature_K, state.temperature_K, rel_tol=0, abs_tol=2e-5)
            or set(candidate.phases.phases) != set(state.phases.phases)
            or any(abs(candidate.phases.phases[key].species_amounts_mol[k] - n)
                   > 2e-7 * fsum(stock.feed_amounts_mol.values())
                   for key, phase in state.phases.phases.items()
                   for k, n in phase.species_amounts_mol.items())):
        raise ValueError("actual molecular phase allocation is not at the declared equilibrium")


def _thermal_energy_limit(
    spec: MolecularMultiphaseThermalSpec, stock: MolecularMultiphaseInventoryState,
    state: Any, initial: float, direction: int,
) -> tuple[float, str, MolecularMultiphaseResult, list[dict[str, Any]]]:
    """Enclose the current thermal direction with a real same-model boundary.

    These are finite branch roots, not an exhaustive search of nonmonotone
    pressure/volume paths. Irrelevant unreachable far limits do not block an
    operation. A missing required caloric enclosure is an explicit failure.
    """
    vessel = state.vessels.vessels[state.vessel_id]
    temperature = (min(spec.stop_temperature_K, vessel.max_temperature_K)
                   if direction > 0 else spec.minimum_temperature_K)
    attempts: list[dict[str, Any]] = []
    candidates: list[tuple[float, str, MolecularMultiphaseResult]] = []

    def evaluate(inputs: dict[str, float], name: str) -> MolecularMultiphaseResult | None:
        result = solve_molecular_multiphase(
            spec.inventory.equilibrium, stock.feed_amounts_mol, inputs=inputs, seed=stock.seed,
            support_heat_capacity_J_K=spec.support_heat_capacity_J_K,
            support_reference_temperature_K=spec.support_reference_temperature_K,
        )
        item = {"boundary": name, "inputs": inputs, "success": result.success,
                "failure_reason": result.failure_reason, "selected_candidate": False}
        attempts.append(item)
        if not result.success:
            return None
        energy = cast(float, result.enthalpy_J if spec.boundary == "P"
                      else result.internal_energy_J)
        t = cast(float, result.temperature_K)
        # A pressure bound beyond the requested temperature stop is irrelevant.
        if (spec.minimum_temperature_K-1e-7 <= t
                <= min(spec.stop_temperature_K, vessel.max_temperature_K)+1e-7
                and direction*(energy-initial) >= -1e-9):
            candidates.append((energy, name, result))
            item["selected_candidate"] = True
        return result

    thermal = evaluate({spec.boundary: spec.boundary_value, "T": temperature},
                       "temperature_limit" if direction > 0 else "minimum_temperature")
    if spec.boundary == "V":
        low, high = spec.inventory.equilibrium.pressure_bounds_Pa
        # If the T boundary cannot be reached within the EOS pressure domain,
        # an actual PV root must bound the energy before any provisional trial.
        if thermal is None:
            evaluate({"V": spec.boundary_value, "P": low}, "minimum_pressure_domain")
            evaluate({"V": spec.boundary_value, "P": high}, "pressure_domain_limit")
        if ((thermal is None or cast(float, thermal.pressure_Pa) > vessel.max_pressure_Pa)
                and low <= vessel.max_pressure_Pa <= high):
            evaluate({"V": spec.boundary_value, "P": vessel.max_pressure_Pa},
                     "pressure_limit")
    elif thermal is None:
        # A nearer hardware volume can still provide a valid finite enclosure.
        evaluate({"P": spec.boundary_value, "V": vessel.max_volume_L*1e-3},
                 "vessel_capacity")
    if not candidates:
        raise ValueError("molecular thermal energy boundary unresolved: " + str(attempts))
    energy, name, result = min(candidates, key=lambda candidate: direction*(candidate[0]-initial))
    return energy, name, result, attempts


class MolecularMultiphaseThermalRuntime:
    supported_operations = ("wait",)

    def __init__(self, composition: Any) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return (self.composition is not None and self.composition.physics is not None
                and isinstance(self.composition.physics.models_of("flash").get(state.vessel_id),
                               MolecularMultiphaseThermalSpec))

    def reconcile_pressure(self, state: Any) -> Any:
        spec = self.composition.physics.models_of("flash")[state.vessel_id]
        return state.replace(pressure_Pa=spec.boundary_value) if spec.boundary == "P" else state

    def advance(self, state: Any, action: Mapping[str, Any], *, heat: bool = False,
                jacket_program: Any = None) -> ModelProviderResult:
        spec: MolecularMultiphaseThermalSpec = self.composition.physics.models_of("flash")[
            state.vessel_id
        ]
        try:
            record, device = connected_device(state, required=False)
            if jacket_program is not None or (device is not None and device.mode != "finite_power"):
                raise ValueError("multiphase thermal control requires a finite-power device")
            if heat:
                if device is None:
                    raise ValueError("heat requires an attached original thermal device")
                device = replace(device, setpoint_K=float(action["target_temperature_K"]),
                                 enabled=True)
            bounds = spec.inventory.equilibrium.temperature_bounds_K
            if device is not None and not bounds[0] <= device.setpoint_K <= bounds[1]:
                raise ValueError("device setpoint leaves declared molecular domain")
            duration = float(action["duration_s"])
            if not isfinite(duration) or duration <= 0:
                raise ValueError("thermal duration must be positive and finite")
        except (ValueError, KeyError) as exc:
            return ModelProviderResult({}, success=False, outcome="rejected",
                failure_reason="invalid_thermal_control", diagnostics={"control_error": str(exc)})
        failed_flash: dict[str, Any] = {}
        try:
            validate_molecular_multiphase_thermal_state(state, spec)
            stock = _inventory(state, spec)
            initial = molecular_multiphase_inventory_energy_j(state, spec)
            initial_u = stock.internal_energy_J + spec.support_heat_capacity_J_K * (
                state.temperature_K - spec.support_reference_temperature_K
            )
            cache: dict[float, MolecularMultiphaseResult] = {}
            initial_power = 0.0 if device is None else device.delivered_power_w(state.temperature_K)
            initial_loss = spec.loss_conductance_W_K * (
                state.temperature_K-spec.environment_temperature_K)
            direction = int(initial_power > initial_loss)-int(initial_power < initial_loss)
            enclosure: tuple[float, str, MolecularMultiphaseResult,
                             list[dict[str, Any]]] | None = None
            provisional_failures: list[dict[str, Any]] = []

            def trial(value: float) -> MolecularMultiphaseResult:
                value = float(value)
                if value not in cache:
                    result = _closure(spec, stock.feed_amounts_mol, value, stock.seed)
                    if not result.success:
                        failed_flash.update(
                            inputs={spec.boundary: spec.boundary_value, spec.coordinate: value},
                            seed_temperature_K=stock.seed.temperature_K,
                            seed_pressure_Pa=stock.seed.pressure_Pa,
                            result=result.to_dict(),
                        )
                        raise ValueError(
                            "multiphase thermal trial failed: " + result.failure_reason
                        )
                    cache[value] = result
                return cache[value]

            def ode_trial(value: float) -> MolecularMultiphaseResult:
                nonlocal enclosure
                # Internal RK/Jacobian stages are numerical trials. Extending
                # the boundary response lets the integrator locate the earlier
                # event without asking the EOS to describe an out-of-domain
                # state. No such extension is eligible for installation.
                if enclosure is not None:
                    if direction*(value-enclosure[0]) > 0:
                        return enclosure[2]
                    if direction*(value-initial) < 0:
                        return trial(initial)
                try:
                    return trial(value)
                except ValueError:
                    if not direction or enclosure is not None:
                        raise
                    enclosure = _thermal_energy_limit(spec, stock, state, initial, direction)
                    if direction*(value-enclosure[0]) <= 0:
                        raise  # An unresolved interior state is never extrapolated.
                    provisional_failures.append(dict(failed_flash))
                    return enclosure[2]

            def rhs(time: float, vector: Any) -> Any:
                temperature = cast(float, ode_trial(vector[0]).temperature_K)
                power = 0.0 if device is None else device.delivered_power_w(temperature)
                loss = spec.loss_conductance_W_K * (temperature - spec.environment_temperature_K)
                return np.array([power - loss, power, loss])

            vessel = state.vessels.vessels[state.vessel_id]

            def energy_limit(time: float, vector: Any) -> float:
                return 1.0 if enclosure is None else direction*(enclosure[0]-float(vector[0]))

            def hot(time: float, vector: Any) -> float:
                return min(spec.stop_temperature_K, vessel.max_temperature_K) - cast(
                    float, ode_trial(vector[0]).temperature_K)

            def cold(time: float, vector: Any) -> float:
                return cast(float, ode_trial(vector[0]).temperature_K) - spec.minimum_temperature_K

            def boundary_limit(time: float, vector: Any) -> float:
                result = ode_trial(vector[0])
                return (vessel.max_volume_L - cast(float, result.volume_m3) * 1000
                        if spec.boundary == "P"
                        else vessel.max_pressure_Pa - cast(float, result.pressure_Pa))

            events = (energy_limit, hot, cold, boundary_limit)
            names = ("thermal_energy_limit", "temperature_limit", "minimum_temperature",
                     "vessel_capacity" if spec.boundary == "P" else "pressure_limit")
            for event in events:
                cast(Any, event).terminal, cast(Any, event).direction = True, -1
            report = solve_ode(
                rhs, [initial, 0., 0.], time_span_s=(0., duration),
                policy=ODESolverPolicy("molecular_multiphase_thermal_v1", spec.ode_method,
                    2e-9, 2e-11, max_step_s=min(spec.maximum_step_s, duration), event_names=names),
                events=events,
            )
            report.raise_for_failure("molecular multiphase finite thermal advance")
            final_time, vector = float(report.t[-1]), report.y[:, -1]
            stopping_event = None
            if report.terminal_point is not None:
                final_time, vector, index = report.terminal_point
                stopping_event = names[index]
                if index == 0 and enclosure is not None:
                    vector = vector.copy()
                    vector[0] = enclosure[0]
                    stopping_event = enclosure[1]
            # Resolve the actual event energy with the original PH/UV closure;
            # never install the numerical boundary extension from ode_trial.
            result = trial(float(vector[0]))
            candidate = install_molecular_multiphase_thermal(state, spec, result)
            work = (spec.boundary_value * (candidate.volume_L - state.volume_L) * 1e-3
                    if spec.boundary == "P" else 0.0)
            residual = (cast(float, result.internal_energy_J) - initial_u
                        - vector[1] + vector[2] + work)
            if abs(residual) > 2e-7 * max(1.0, abs(vector[1]), abs(vector[2])):
                raise ValueError("molecular multiphase thermal first law failed")
            equipment = dict(state.equipment.equipment)
            if record is not None:
                device = cast(ThermalDeviceSpec, device)
                equipment[record.equipment_id] = replace(
                    record, settings={**record.settings, "thermal_device": device.to_dict()})
            candidate = candidate.replace(
                equipment=EquipmentLedger(equipment),
                ledger=advance_ledger(state.ledger, final_time, energy_deltas={
                    "energy_jacket_J": float(vector[1]), "heat_loss_J": float(vector[2]),
                    "boundary_work_J": work,
                }),
            )
        except (ArithmeticError, ValueError, RuntimeError) as exc:
            return ModelProviderResult({}, success=False, outcome="numerical_failure",
                failure_reason="molecular_multiphase_advance_failed",
                diagnostics={"solver_error": str(exc), **(
                    {"failed_flash": failed_flash} if failed_flash else {})})
        return ModelProviderResult(
            {}, state=candidate, actual_duration_s=final_time,
            outcome="stopped" if stopping_event else "success",
            diagnostics={"solver": report.diagnostic.to_dict(), "flash": result.to_dict(),
                "caloric_trial_count": len(cache), "first_law_residual_J": float(residual),
                "boundary_work_J": work, "stopping_event": stopping_event,
                "event_energy_enclosure": [] if enclosure is None else enclosure[3],
                "provisional_out_of_domain_trials": provisional_failures,
                "energy_balance_basis": ("constant_pressure_enthalpy" if spec.boundary == "P"
                                         else "constant_volume_internal_energy")},
        )
