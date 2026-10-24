"""Original finite transport consumer for a one-feed molecular flash separator."""

from collections.abc import Mapping
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.physchem.molecular_flash_separator_inventory import (
    MolecularFlashSeparatorInventorySpec,
    advance_molecular_flash_separator,
)
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.molecular_tanks import install_molecular_tank, read_molecular_tank

SEPARATOR_TYPE = "molecular_flash_separator"
SEPARATOR_STATE_TYPE = "molecular_flash_separator_enthalpy_v1"


def separator_record(state: Any) -> Any:
    return attached_device_record(state, SEPARATOR_TYPE)


def install_separator_tank(
    state: Any,
    spec: MolecularFlashSeparatorInventorySpec,
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
        phase_id=spec.phase_id(index),
        pressure_Pa=spec.separator.pressure_Pa,
        state_type=SEPARATOR_STATE_TYPE,
    )


def unpack_separator(
    state: Any,
) -> tuple[Any, MolecularFlashSeparatorInventorySpec, TransportInventory]:
    record = separator_record(state)
    if record is None:
        raise ValueError("no molecular flash separator is attached")
    spec = MolecularFlashSeparatorInventorySpec.from_dict(record.settings["physical_model"])
    if state.vessel_id != spec.vessel_ids[1]:
        raise ValueError("molecular separator must be anchored to its gas receiver")
    amounts, enthalpies = [], []
    for i, name in enumerate(spec.vessel_ids):
        tank = state if name == state.vessel_id else state.inactive_vessels.get(name)
        if tank is None:
            raise ValueError("molecular separator current group is missing a finite owner")
        n = dict(tank.species_amounts)
        total = spec.validate_amounts(n)
        closed = spec.at_tp(i, n, tank.temperature_K)
        h = read_molecular_tank(
            tank,
            closed,
            phase_id=spec.phase_id(i),
            pressure_Pa=spec.separator.pressure_Pa,
            maximum_volume_L=spec.tank_capacities_L[i],
            state_type=SEPARATOR_STATE_TYPE,
        )
        if i == 0 and total < spec.source_reserve_mol - 1e-9:
            raise ValueError("molecular separator source is below reserve")
        if i > 0 and total > spec.receiver_capacities_mol[i - 1] + 1e-9:
            raise ValueError("molecular separator receiver exceeds its amount capacity")
        amounts.append(n)
        enthalpies.append(h)
    return record, spec, TransportInventory(tuple(amounts), tuple(enthalpies), state.ledger.time_s)


class MolecularFlashSeparatorAdapter:
    supported_operations = ("run_flow", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return separator_record(state) is not None

    def validate_state(self, state: Any) -> None:
        unpack_separator(state)

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = MolecularFlashSeparatorInventorySpec.from_dict(
            separator_record(state).settings["physical_model"]
        )
        ns = spec.vessel_ids
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            ns,
            connection_ids(self.composition, [(ns[0], ns[1]), (ns[0], ns[2])]),
        )

    @property
    def preserved_source_vessels(self) -> set[str]:
        names: set[str] = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    names.update(
                        MolecularFlashSeparatorInventorySpec.from_dict(
                            separator_record(state).settings["physical_model"]
                        ).vessel_ids
                    )
        return names

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def reconcile_initialization(self, transition: Any) -> Any:
        active = transition.state
        states = {active.vessel_id: active, **active.inactive_vessels}
        for state in states.values():
            if self.supports(state):
                self.validate_state(
                    state.replace(
                        inactive_vessels={k: v for k, v in states.items() if k != state.vessel_id}
                    )
                )
        return transition

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        installed = {v for v, s in states.items() if self.supports(s)}
        declared = {c.vessel for c in self.composition.components if c.kind == "finite_transport"}
        if not installed <= declared:
            raise ValueError("molecular separator requires original finite_transport")
        occupied: set[str] = set()
        for name in installed:
            state = states[name].replace(
                inactive_vessels={k: v for k, v in states.items() if k != name}
            )
            _, spec, _ = unpack_separator(state)
            if occupied.intersection(spec.vessel_ids):
                raise ValueError("molecular separators cannot share finite tank ownership")
            occupied.update(spec.vessel_ids)
            if self.composition.physics.network.reactions or set(
                self.composition.physics.network.species_ids
            ) != set(spec.properties.component_ids):
                raise ValueError("separator requires the same original nonreacting binary network")
            self.shared_system(state, "finite-pump-transport")
            for i, tank_id in enumerate(spec.vessel_ids):
                hardware = states[tank_id].vessels.vessels[tank_id]
                if (
                    hardware.max_volume_L < spec.tank_capacities_L[i]
                    or hardware.max_pressure_Pa < spec.separator.pressure_Pa
                    or hardware.max_temperature_K < spec.tank_temperature_bounds_K[i][1]
                ):
                    raise ValueError("separator declaration exceeds installed tank hardware")

    def configure_flow(self, state: Any, action: dict) -> Any:
        raise ValueError("molecular separator uses its declared fixed molar feed rate")

    def advance(
        self, state: Any, action: dict, *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        if heat or jacket_program is not None or "target_temperature_K" in action:
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="separator_has_declared_temperature_control",
            )
        try:
            record, spec, initial = unpack_separator(state)
            result, diagnostic = advance_molecular_flash_separator(
                spec, initial, float(action["duration_s"])
            )
            final, dt = result.final_state, result.actual_duration_s
            participating = set(diagnostic["participating_tank_indices"])
            inactive, ages = dict(state.inactive_vessels), dict(state.vessel_elapsed_s)
            for i, name in enumerate(spec.vessel_ids):
                if name == state.vessel_id:
                    continue
                tank = inactive[name]
                if i in participating:
                    tank = install_separator_tank(
                        tank, spec, i, final.amounts_mol[i], final.enthalpies_J[i]
                    )
                inactive[name] = tank.replace(ledger=advance_ledger(tank.ledger, dt))
                ages[name] = ages.get(name, 0.0) + dt
            candidate = (
                install_separator_tank(state, spec, 1, final.amounts_mol[1], final.enthalpies_J[1])
                if 1 in participating
                else state
            )
            candidate = candidate.replace(
                inactive_vessels=inactive,
                vessel_elapsed_s=ages,
                equipment=upsert_equipment_record(
                    candidate.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"last_molecular_separator": diagnostic},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    energy_deltas={
                        "flow_heat_J": diagnostic["heat_J"],
                        "boundary_work_J": diagnostic["pressure_work_J"],
                    },
                ),
            )
            self.validate_state(candidate)
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="molecular_flash_separator_failed",
                diagnostics={
                    "solver_error": str(exc),
                    "separator": getattr(exc, "diagnostics", {}),
                },
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            vessel_thermal_deltas={name: {} for name in spec.vessel_ids},
            diagnostics={
                "separator": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
            },
        )
