"""Original finite thermal device consuming simultaneous molecular PH closure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from math import fsum, isclose, isfinite
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger, PhaseRecord
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_caloric_flash import (
    MolecularCaloricFlashResult,
    MolecularCaloricFlashSeed,
    MolecularCaloricFlashSpec,
    solve_molecular_caloric_flash,
)
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.molecular_phase_inventory import MolecularPhaseTarget


@dataclass(frozen=True)
class MolecularFlashThermalSpec:
    equilibrium: MolecularCaloricFlashSpec
    phase_targets: tuple[MolecularPhaseTarget, ...]
    support_heat_capacity_J_K: float = 0.0
    support_reference_temperature_K: float = 298.15
    loss_conductance_W_K: float = 0.0
    environment_temperature_K: float = 298.15
    ode_method: str = "BDF"
    minimum_temperature_K: float = 285.0
    stop_temperature_K: float = 315.0
    stop_phase_fraction: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase_targets", tuple(self.phase_targets))
        if (
            len(self.phase_targets) != 2
            or len({p.phase_id for p in self.phase_targets}) != 2
            or tuple(p.basin for p in self.phase_targets) != self.equilibrium.phase_basins
            or any(p.phase_type not in {"liquid", "gas"} for p in self.phase_targets)
        ):
            raise ValueError("molecular thermal needs two ordered original liquid/gas phase IDs")
        if (
            any(
                not isfinite(v) or isinstance(v, bool) or v < 0
                for v in (self.support_heat_capacity_J_K, self.loss_conductance_W_K)
            )
            or any(
                not isfinite(v) or isinstance(v, bool) or v <= 0
                for v in (self.support_reference_temperature_K, self.environment_temperature_K)
            )
            or self.ode_method not in {"BDF", "Radau"}
        ):
            raise ValueError("thermal support/loss and solver method must be explicit and finite")
        if (
            not self.equilibrium.temperature_range_K[0]
            < self.minimum_temperature_K
            < self.stop_temperature_K
            < self.equilibrium.temperature_range_K[1]
            or not self.equilibrium.minimum_phase_fraction < self.stop_phase_fraction < 0.5
        ):
            raise ValueError(
                "thermal stopping range must lie inside the constitutive two-phase domain"
            )

    @property
    def pressure_Pa(self) -> float:  # noqa: N802
        return self.equilibrium.pressure_Pa

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "molecular_flash_thermal_v1",
            "equilibrium": self.equilibrium.to_dict(),
            "phase_targets": [asdict(p) for p in self.phase_targets],
            "support_heat_capacity_J_K": self.support_heat_capacity_J_K,
            "support_reference_temperature_K": self.support_reference_temperature_K,
            "loss_conductance_W_K": self.loss_conductance_W_K,
            "environment_temperature_K": self.environment_temperature_K,
            "ode_method": self.ode_method,
            "minimum_temperature_K": self.minimum_temperature_K,
            "stop_temperature_K": self.stop_temperature_K,
            "stop_phase_fraction": self.stop_phase_fraction,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularFlashThermalSpec:
        if data.get("model_type") != "molecular_flash_thermal_v1":
            raise ValueError("unknown molecular flash thermal declaration")
        values = {k: v for k, v in data.items() if k != "model_type"}
        values.update(
            equilibrium=MolecularCaloricFlashSpec.from_dict(data["equilibrium"]),
            phase_targets=tuple(MolecularPhaseTarget(**p) for p in data["phase_targets"]),
        )
        return cls(**values)


def _inventory(
    state: Any, spec: MolecularFlashThermalSpec
) -> tuple[dict[str, float], MolecularCaloricFlashSeed, list[Any]]:
    ids = spec.equilibrium.model.component_ids
    if (
        state.state_slots.slots
        or set(state.phases.phases) != {p.phase_id for p in spec.phase_targets}
        or set(state.species_amounts) != set(ids)
    ):
        raise ValueError("molecular thermal requires exactly its original two phase inventories")
    if (
        not spec.equilibrium.temperature_range_K[0]
        <= state.temperature_K
        <= (spec.equilibrium.temperature_range_K[1])
    ):
        raise ValueError("molecular thermal temperature outside declared branch")
    if (
        not spec.minimum_temperature_K - 1e-7
        <= state.temperature_K
        <= spec.stop_temperature_K + 1e-7
    ):
        raise ValueError("molecular thermal temperature outside declared operating range")
    inventories, densities, calorics = [], [], []
    feed = dict.fromkeys(ids, 0.0)
    for target in spec.phase_targets:
        p = state.phases.phases[target.phase_id]
        if (
            p.vessel_id != state.vessel_id
            or p.phase_type != target.phase_type
            or p.solvents.volume_L != 0
            or set(p.species_amounts_mol) != set(ids)
            or p.volume_L <= 0
            or any(not isfinite(v) or v < 0 for v in p.species_amounts_mol.values())
        ):
            raise ValueError("phase identity, solvent, density or component inventory mismatch")
        amounts = dict(p.species_amounts_mol)
        n = fsum(amounts.values())
        if n <= 0:
            raise ValueError("declared molecular two-phase branch cannot contain an empty phase")
        rho = n / (p.volume_L * 1e-3)
        basin = next(b for b in spec.equilibrium.basins if b.name == target.basin)
        if not basin.density_bounds_mol_m3[0] <= rho <= basin.density_bounds_mol_m3[1]:
            raise ValueError("actual phase density leaves declared basin")
        caloric = spec.equilibrium.model.state(
            state.temperature_K, rho, tuple(amounts[k] / n for k in ids)
        )
        if not isclose(caloric.pressure_Pa, spec.pressure_Pa, rel_tol=2e-7, abs_tol=1e-6):
            raise ValueError("actual phase pressure differs from constant-P boundary")
        inventories.append(amounts)
        densities.append(rho)
        calorics.append(caloric)
        for k in ids:
            feed[k] += amounts[k]
    if (
        any(not isclose(feed[k], state.species_amounts[k], rel_tol=2e-12, abs_tol=0.0) for k in ids)
        or not isclose(state.pressure_Pa, spec.pressure_Pa, rel_tol=1e-10, abs_tol=1e-6)
        or not isclose(
            state.volume_L,
            fsum(p.volume_L for p in state.phases.phases.values()),
            rel_tol=2e-12,
            abs_tol=0.0,
        )
    ):
        raise ValueError(
            "original total inventory, volume or pressure disagrees with actual phases"
        )
    vessel = state.vessels.vessels[state.vessel_id]
    if (
        state.volume_L > vessel.max_volume_L + 1e-10
        or state.temperature_K > vessel.max_temperature_K + 1e-7
        or state.pressure_Pa > vessel.max_pressure_Pa
    ):
        raise ValueError("molecular thermal state exceeds finite vessel limits")
    seed = MolecularCaloricFlashSeed(
        state.temperature_K, (inventories[0], inventories[1]), (densities[0], densities[1])
    )
    return feed, seed, calorics


def molecular_flash_inventory_enthalpy_j(state: Any, spec: MolecularFlashThermalSpec) -> float:
    _, seed, calorics = _inventory(state, spec)
    return fsum(
        fsum(n.values()) * s.enthalpy_J_mol
        for n, s in zip(seed.phase_amounts_mol, calorics, strict=True)
    ) + spec.support_heat_capacity_J_K * (
        state.temperature_K - spec.support_reference_temperature_K
    )


def _closure(
    spec: MolecularFlashThermalSpec,
    feed: Mapping[str, float],
    value: float,
    seed: MolecularCaloricFlashSeed,
    *,
    verify_stability: bool = True,
) -> MolecularCaloricFlashResult:
    return solve_molecular_caloric_flash(
        spec.equilibrium,
        feed,
        coordinate="H",
        value=value,
        seed=seed,
        support_heat_capacity_J_K=spec.support_heat_capacity_J_K,
        support_reference_temperature_K=spec.support_reference_temperature_K,
        verify_stability=verify_stability,
    )


def validate_molecular_flash_state(state: Any, spec: MolecularFlashThermalSpec) -> None:
    feed, seed, _ = _inventory(state, spec)
    result = _closure(spec, feed, molecular_flash_inventory_enthalpy_j(state, spec), seed)
    if not result.success:
        raise ValueError("molecular initial equilibrium unresolved: " + result.failure_reason)
    if abs(cast(float, result.temperature_K) - state.temperature_K) > 2e-5 or any(
        abs(n[k] - p.component_amounts_mol[k]) > 2e-7 * fsum(feed.values())
        for n, p in zip(seed.phase_amounts_mol, result.phases, strict=True)
        for k in feed
    ):
        raise ValueError("actual molecular inventory is not at its declared equilibrium")


def install_molecular_caloric_flash(
    state: Any, spec: MolecularFlashThermalSpec, result: MolecularCaloricFlashResult
) -> Any:
    if not result.success or not result.diagnostics.get("stability_checked"):
        raise ValueError("cannot install an unresolved or unchecked caloric flash trial")
    if (
        cast(float, result.volume_m3) * 1000
        > state.vessels.vessels[state.vessel_id].max_volume_L + 1e-10
    ):
        raise ValueError("molecular caloric flash exceeds finite vessel capacity")
    phases = {}
    for target, phase in zip(spec.phase_targets, result.phases, strict=True):
        previous = state.phases.phases.get(target.phase_id)
        phases[target.phase_id] = PhaseRecord(
            target.phase_id,
            state.vessel_id,
            target.phase_type,
            phase.volume_m3 * 1000,
            phase.component_amounts_mol,
            selected=target.selected,
            metadata={
                **({} if previous is None else previous.metadata),
                "carrier_representation": "species_inventory",
                "molecular_density_basin": target.basin,
            },
        )
    return state.replace(
        phases=PhaseLedger(phases),
        species_amounts={
            k: fsum(p.component_amounts_mol[k] for p in result.phases)
            for k in spec.equilibrium.model.component_ids
        },
        temperature_K=result.temperature_K,
        pressure_Pa=spec.pressure_Pa,
        volume_L=cast(float, result.volume_m3) * 1000,
        phase="liquid" if any(t.phase_type == "liquid" for t in spec.phase_targets) else "gas",
        process=replace(state.process, last_observation_record=None),
    )


class MolecularFlashThermalRuntime:
    supported_operations = ("wait",)

    def __init__(self, composition: Any) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return (
            self.composition is not None
            and self.composition.physics is not None
            and isinstance(
                self.composition.physics.models_of("flash").get(state.vessel_id),
                MolecularFlashThermalSpec,
            )
        )

    def validate_state(self, state: Any) -> None:
        validate_molecular_flash_state(
            state, self.composition.physics.models_of("flash")[state.vessel_id]
        )

    def reconcile_pressure(self, state: Any) -> Any:
        return state.replace(
            pressure_Pa=self.composition.physics.models_of("flash")[state.vessel_id].pressure_Pa
        )

    def advance(
        self,
        state: Any,
        action: Mapping[str, Any],
        *,
        heat: bool = False,
        jacket_program: Any = None,
    ) -> ModelProviderResult:
        spec = self.composition.physics.models_of("flash")[state.vessel_id]
        try:
            record, device = connected_device(state, required=False)
            if jacket_program is not None or (device is not None and device.mode != "finite_power"):
                raise ValueError("molecular caloric branch requires finite-power device control")
            if heat:
                if device is None:
                    raise ValueError("heat requires an attached original thermal device")
                device = replace(
                    device, setpoint_K=float(action["target_temperature_K"]), enabled=True
                )
            if (
                device is not None
                and not spec.equilibrium.temperature_range_K[0]
                <= device.setpoint_K
                <= (spec.equilibrium.temperature_range_K[1])
            ):
                raise ValueError("device setpoint leaves declared molecular domain")
            duration = float(action["duration_s"])
            if not isfinite(duration) or duration <= 0:
                raise ValueError("thermal duration must be positive and finite")
        except (ValueError, KeyError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="invalid_thermal_control",
                diagnostics={"control_error": str(exc)},
            )
        try:
            self.validate_state(state)
            feed, original_seed, calorics = _inventory(state, spec)
            initial_h = molecular_flash_inventory_enthalpy_j(state, spec)
            initial_u = fsum(
                fsum(n.values()) * s.internal_energy_J_mol
                for n, s in zip(original_seed.phase_amounts_mol, calorics, strict=True)
            )
            initial_u += spec.support_heat_capacity_J_K * (
                state.temperature_K - spec.support_reference_temperature_K
            )
            cache = {}
            calls = [0]

            def trial(h: float) -> MolecularCaloricFlashResult:
                h = float(h)
                if h not in cache:
                    result = _closure(spec, feed, h, original_seed, verify_stability=False)
                    calls[0] += 1
                    if not result.success:
                        raise ValueError(
                            "molecular PH numerical trial failed: " + result.failure_reason
                        )
                    cache[h] = result
                return cache[h]

            def rhs(time: float, vector: Any) -> NDArray[np.float64]:
                t = cast(float, trial(vector[0]).temperature_K)
                power = 0.0 if device is None else device.delivered_power_w(t)
                loss = spec.loss_conductance_W_K * (t - spec.environment_temperature_K)
                return np.array([power - loss, power, loss])

            vessel = state.vessels.vessels[state.vessel_id]

            def capacity(time: float, vector: Any) -> float:
                return vessel.max_volume_L - cast(float, trial(vector[0]).volume_m3) * 1000

            def hot(time: float, vector: Any) -> float:
                return min(spec.stop_temperature_K, vessel.max_temperature_K) - cast(
                    float, trial(vector[0]).temperature_K
                )

            def cold(time: float, vector: Any) -> float:
                return cast(float, trial(vector[0]).temperature_K) - spec.minimum_temperature_K

            def branch(time: float, vector: Any) -> float:
                return (
                    min(p.amount_mol for p in trial(vector[0]).phases) / fsum(feed.values())
                    - spec.stop_phase_fraction
                )

            events = (capacity, hot, cold, branch)
            names = (
                "vessel_capacity",
                "temperature_limit",
                "minimum_temperature",
                "two_phase_operating_limit",
            )
            for event_function in events:
                cast(Any, event_function).terminal, cast(Any, event_function).direction = True, -1
            report = solve_ode(
                rhs,
                [initial_h, 0.0, 0.0],
                time_span_s=(0.0, duration),
                policy=ODESolverPolicy(
                    "molecular_simultaneous_ph_thermal_v1",
                    spec.ode_method,
                    2e-9,
                    2e-11,
                    max_step_s=min(20.0, duration),
                    event_names=names,
                ),
                events=events,
            )
            report.raise_for_failure("molecular two-phase finite thermal advance")
            final_time, vector = float(report.t[-1]), report.y[:, -1]
            event = None
            if report.terminal_point is not None:
                final_time, vector, index = report.terminal_point
                event = names[index]
            endpoint = trial(vector[0])
            result = _closure(
                spec, feed, float(vector[0]), MolecularCaloricFlashSeed.from_phases(endpoint.phases)
            )
            candidate = install_molecular_caloric_flash(state, spec, result)
            equipment = dict(state.equipment.equipment)
            if record is not None:
                device = cast(ThermalDeviceSpec, device)
                equipment[record.equipment_id] = replace(
                    record, settings={**record.settings, "thermal_device": device.to_dict()}
                )
            work = spec.pressure_Pa * (candidate.volume_L - state.volume_L) * 1e-3
            net_heat = float(vector[1] - vector[2])
            residual = cast(float, result.internal_energy_J) - initial_u - net_heat + work
            if abs(residual) > 2e-7 * max(1.0, abs(net_heat)):
                raise ValueError("molecular thermal first law failed")
            candidate = candidate.replace(
                equipment=EquipmentLedger(equipment),
                ledger=advance_ledger(
                    state.ledger,
                    final_time,
                    energy_deltas={
                        "energy_jacket_J": float(vector[1]),
                        "heat_loss_J": float(vector[2]),
                        "boundary_work_J": work,
                    },
                ),
            )
        except (ArithmeticError, ValueError, RuntimeError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="molecular_caloric_advance_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=final_time,
            outcome="stopped" if event else "success",
            diagnostics={
                "solver": report.diagnostic.to_dict(),
                "flash": result.to_dict(),
                "caloric_trial_count": calls[0],
                "first_law_residual_J": residual,
                "boundary_work_J": work,
                "energy_balance_basis": "constant_pressure_enthalpy",
                "stopping_event": event,
                "boundary_policy": (
                    "physical_operating_events; reject_unresolved_constitutive_trials"
                ),
            },
        )
