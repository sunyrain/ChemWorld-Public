"""Molecular equilibrium contacting through the original finite transport path."""

from collections.abc import Mapping
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.molecular_cascade_inventory import (
    MolecularCascadeInventorySpec,
    advance_molecular_cascade,
)
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.molecular_tanks import (
    install_molecular_tank,
    molecular_enthalpy_slot,
    read_molecular_tank,
)

CASCADE_TYPE = "molecular_equilibrium_cascade"


def cascade_record(state: Any) -> Any:
    return attached_device_record(state, CASCADE_TYPE)


def cascade_enthalpy_slot(state: Any, phase_id: str, enthalpy: float) -> StateSlots:
    return molecular_enthalpy_slot(state, phase_id, enthalpy, "molecular_cascade_enthalpy_v1")


def install_cascade_tank(
    state: Any,
    spec: MolecularCascadeInventorySpec,
    index: int,
    amounts: Mapping[str, float],
    enthalpy: float,
) -> Any:
    closed = spec.at_ph(index, amounts, enthalpy)
    return install_molecular_tank(
        state,
        closed,
        amounts,
        enthalpy,
        phase_id="gas" if index % 2 == 0 else "liquid",
        pressure_Pa=spec.cascade.pressure_Pa,
        state_type="molecular_cascade_enthalpy_v1",
    )


def unpack_cascade(state: Any) -> tuple[Any, MolecularCascadeInventorySpec, TransportInventory]:
    record = cascade_record(state)
    if record is None:
        raise ValueError("no original molecular cascade apparatus is attached")
    spec = MolecularCascadeInventorySpec.from_dict(record.settings["physical_model"])
    if state.vessel_id != spec.vessel_ids[2]:
        raise ValueError("molecular cascade must be anchored to its gas receiver")
    amounts, enthalpies = [], []
    for index, name in enumerate(spec.vessel_ids):
        tank = state if name == state.vessel_id else state.inactive_vessels.get(name)
        if tank is None:
            raise ValueError("molecular cascade current state is missing a finite owner")
        n = dict(tank.species_amounts)
        total = spec.validate_amounts(n)
        phase_id = "gas" if index % 2 == 0 else "liquid"
        closed = spec.at_tp(index, n, tank.temperature_K)
        h = read_molecular_tank(
            tank,
            closed,
            phase_id=phase_id,
            pressure_Pa=spec.cascade.pressure_Pa,
            maximum_volume_L=spec.tank_capacities_L[index],
            state_type="molecular_cascade_enthalpy_v1",
        )
        if index < 2 and total < spec.source_reserves_mol[index] - 1e-9:
            raise ValueError("molecular source is below its declared reserve")
        if index >= 2 and total > spec.receiver_capacities_mol[index - 2] + 1e-9:
            raise ValueError("molecular receiver exceeds its declared amount capacity")
        amounts.append(n)
        enthalpies.append(h)
    return record, spec, TransportInventory(tuple(amounts), tuple(enthalpies), state.ledger.time_s)


class MolecularCascadeAdapter:
    supported_operations = ("run_flow", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return cascade_record(state) is not None

    def validate_state(self, state: Any) -> None:
        unpack_cascade(state)

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = MolecularCascadeInventorySpec.from_dict(
            cascade_record(state).settings["physical_model"]
        )
        names = spec.vessel_ids
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            names,
            connection_ids(
                self.composition, [(names[0], names[2]), (names[1], names[2]), (names[2], names[3])]
            ),
        )

    @property
    def preserved_source_vessels(self) -> set[str]:
        names: set[str] = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    names.update(
                        MolecularCascadeInventorySpec.from_dict(
                            cascade_record(state).settings["physical_model"]
                        ).vessel_ids
                    )
        return names

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def reconcile_initialization(self, transition: Any) -> Any:
        active = transition.state
        locals_ = {active.vessel_id: active, **active.inactive_vessels}
        for state in locals_.values():
            if self.supports(state):
                self.validate_state(
                    state.replace(
                        inactive_vessels={k: s for k, s in locals_.items() if k != state.vessel_id}
                    )
                )
        return transition

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        installed = {v for v, s in states.items() if self.supports(s)}
        declared = {c.vessel for c in self.composition.components if c.kind == "finite_transport"}
        if not installed <= declared:
            raise ValueError("molecular cascade requires the original finite_transport component")
        occupied: set[str] = set()
        for name in installed:
            state = states[name].replace(
                inactive_vessels={k: v for k, v in states.items() if k != name}
            )
            _, spec, _ = unpack_cascade(state)
            if occupied.intersection(spec.vessel_ids):
                raise ValueError("molecular cascade groups cannot share finite tank ownership")
            occupied.update(spec.vessel_ids)
            if self.composition.physics.network.reactions or set(
                self.composition.physics.network.species_ids
            ) != set(spec.properties.component_ids):
                raise ValueError("molecular cascade needs the same nonreacting binary network")
            self.shared_system(state, "finite-pump-transport")
            for i, tank_id in enumerate(spec.vessel_ids):
                hardware = states[tank_id].vessels.vessels[tank_id]
                if (
                    hardware.max_volume_L < spec.tank_capacities_L[i]
                    or hardware.max_pressure_Pa < spec.cascade.pressure_Pa
                    or hardware.max_temperature_K < spec.tank_temperature_bounds_K[i][1]
                ):
                    raise ValueError("molecular cascade declaration exceeds installed hardware")

    def configure_flow(self, state: Any, action: dict) -> Any:
        raise ValueError("molecular cascade uses two declared fixed molar feed rates")

    def advance(
        self, state: Any, action: dict, *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        if heat or jacket_program is not None or "target_temperature_K" in action:
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="molecular_cascade_has_declared_stage_temperatures",
            )
        try:
            record, spec, initial = unpack_cascade(state)
            result, diagnostic = advance_molecular_cascade(
                spec, initial, float(action["duration_s"])
            )
            final, dt = result.final_state, result.actual_duration_s
            inactive, ages = dict(state.inactive_vessels), dict(state.vessel_elapsed_s)
            for i, name in enumerate(spec.vessel_ids):
                if name == state.vessel_id:
                    continue
                tank = install_cascade_tank(
                    inactive[name], spec, i, final.amounts_mol[i], final.enthalpies_J[i]
                )
                inactive[name] = tank.replace(ledger=advance_ledger(tank.ledger, dt))
                ages[name] = ages.get(name, 0.0) + dt
            candidate = install_cascade_tank(
                state, spec, 2, final.amounts_mol[2], final.enthalpies_J[2]
            )
            energy = {
                "flow_heat_J": diagnostic["heat_J"],
                "boundary_work_J": diagnostic["pressure_work_J"],
            }
            candidate = candidate.replace(
                inactive_vessels=inactive,
                vessel_elapsed_s=ages,
                equipment=upsert_equipment_record(
                    candidate.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"last_molecular_cascade": diagnostic},
                ),
                ledger=advance_ledger(state.ledger, dt, energy_deltas=energy),
            )
            self.validate_state(candidate)
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="molecular_cascade_failed",
                diagnostics={"solver_error": str(exc), "cascade": getattr(exc, "diagnostics", {})},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            vessel_thermal_deltas={name: {} for name in spec.vessel_ids},
            diagnostics={
                "cascade": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
            },
        )
