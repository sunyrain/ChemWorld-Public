"""DCMD adapter for the original membrane strategy and shared transaction."""

from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger, PhaseRecord
from chemworld.physchem.membrane_distillation import (
    MembraneDistillationSpec,
    advance_membrane_distillation,
    membrane_distillation_point,
    validate_membrane_thermal_devices,
)
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.transport_devices import enthalpy_slots

MEMBRANE_DISTILLATION_TYPE = "finite_direct_contact_membrane_distillation"


def membrane_distillation_record(state: Any) -> Any:
    return attached_device_record(state, MEMBRANE_DISTILLATION_TYPE)


def membrane_distillation_vessels(state: Any, spec: MembraneDistillationSpec) -> tuple[str, str]:
    if state.vessel_id == spec.cold_vessel_id:
        raise ValueError("thermal membrane needs two distinct finite vessels")
    return state.vessel_id, spec.cold_vessel_id


def install_membrane_liquid(
    state: Any, spec: MembraneDistillationSpec, index: int, amounts: Any, enthalpy: float
) -> Any:
    temperature = spec.at_ph(index, amounts, enthalpy)
    pressure = spec.pressures_Pa[index]
    volume = 1000 * spec.properties.volume(amounts, temperature, pressure)
    fluid_h = spec.properties.enthalpy(amounts, temperature, pressure)
    return state.replace(
        species_amounts=dict(amounts),
        temperature_K=temperature,
        pressure_Pa=pressure,
        volume_L=volume,
        phases=PhaseLedger(
            {
                "liquid": PhaseRecord(
                    "liquid",
                    state.vessel_id,
                    "liquid",
                    volume,
                    dict(amounts),
                    settled=True,
                    selected=True,
                    metadata={"carrier_representation": "species_inventory"},
                )
            }
        ),
        state_slots=enthalpy_slots(state, fluid_h, enthalpy - fluid_h),
        process=replace(state.process, last_observation_record=None),
    )


def unpack_membrane_distillation(
    state: Any,
) -> tuple[Any, MembraneDistillationSpec, TransportInventory]:
    record = membrane_distillation_record(state)
    if record is None:
        raise ValueError("no direct-contact thermal membrane is installed")
    spec = MembraneDistillationSpec.from_dict(record.settings["physical_model"])
    names = membrane_distillation_vessels(state, spec)
    amounts, enthalpies, devices = [], [], []
    for i, name in enumerate(names):
        tank = state if i == 0 else state.inactive_vessels[name]
        n = dict(tank.species_amounts)
        spec.properties.validate_amounts(n)
        fluid_h = spec.properties.enthalpy(n, tank.temperature_K, spec.pressures_Pa[i])
        h = spec.enthalpy(i, n, tank.temperature_K)
        slots = enthalpy_slots(tank, fluid_h, h - fluid_h)
        actual = {s.spec.slot_id: s for s in tank.state_slots.slots}
        if (
            not isclose(tank.pressure_Pa, spec.pressures_Pa[i], abs_tol=1e-5)
            or not isclose(
                tank.volume_L,
                1000 * spec.properties.volume(n, tank.temperature_K, spec.pressures_Pa[i]),
                rel_tol=1e-8,
                abs_tol=1e-10,
            )
            or set(tank.phases.phases) != {"liquid"}
            or dict(tank.phases.phases["liquid"].species_amounts_mol) != n
            or set(actual) != {s.spec.slot_id for s in slots.slots}
        ):
            raise ValueError("thermal membrane requires matching original liquid n/H/T/P/V")
        for expected in slots.slots:
            owned = actual[expected.spec.slot_id]
            if owned.spec != expected.spec or not isclose(
                owned.values["enthalpy_J"],
                expected.values["enthalpy_J"],
                rel_tol=1e-8,
                abs_tol=1e-6,
            ):
                raise ValueError("thermal membrane owned H differs from declared aqueous calorics")
        amounts.append(n)
        enthalpies.append(sum(s.values["enthalpy_J"] for s in actual.values()))
        devices.append(connected_device(tank)[1])
    validate_membrane_thermal_devices(*devices)
    inventory = TransportInventory(tuple(amounts), tuple(enthalpies), state.ledger.time_s)
    membrane_distillation_point(
        spec,
        inventory.amounts_mol,
        tuple(
            spec.at_ph(i, n, h) for i, (n, h) in enumerate(zip(amounts, enthalpies, strict=True))
        ),
    )
    return record, spec, inventory


def read_membrane_distillation(state: Any) -> dict[str, Any]:
    """Pure constitutive query, not an instrument or a paid measurement."""
    _, spec, inventory = unpack_membrane_distillation(state)
    temperatures = tuple(
        spec.at_ph(i, n, h)
        for i, (n, h) in enumerate(zip(inventory.amounts_mol, inventory.enthalpies_J, strict=True))
    )
    return {
        **membrane_distillation_point(spec, inventory.amounts_mol, temperatures),
        "quantity_kind": "pure_constitutive_query",
        "instrument_id": None,
    }


class MembraneDistillationRuntime:
    supported_operations = ("heat", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return membrane_distillation_record(state) is not None

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = MembraneDistillationSpec.from_dict(
            membrane_distillation_record(state).settings["physical_model"]
        )
        names = membrane_distillation_vessels(state, spec)
        return SharedAdvanceSystem(
            state.vessel_id, strategy_id, names, connection_ids(self.composition, [names])
        )

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return set()

    @property
    def preserved_source_vessels(self) -> set[str]:
        result: set[str] = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    spec = MembraneDistillationSpec.from_dict(
                        membrane_distillation_record(state).settings["physical_model"]
                    )
                    result.update(membrane_distillation_vessels(state, spec))
        return result

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        components = {c.id: c for c in self.composition.components}
        occupied: set[str] = set()
        for vessel, original in states.items():
            if not self.supports(original):
                continue
            if not any(c.kind == "ro_membrane" and c.vessel == vessel for c in components.values()):
                raise ValueError("thermal membrane requires original membrane strategy component")
            state = original.replace(
                inactive_vessels={v: s for v, s in states.items() if v != vessel}
            )
            _, spec, _ = unpack_membrane_distillation(state)
            names = membrane_distillation_vessels(state, spec)
            if occupied.intersection(names) or self.composition.physics.network.reactions:
                raise ValueError("thermal membrane excludes shared reservoirs and reactions")
            occupied.update(names)
            network = {
                s.species_id: (dict(s.composition), s.charge, s.phase)
                for s in self.composition.physics.network.species
            }
            if any(
                network.get(s.species_id) != (dict(s.composition), s.charge, "liquid")
                for s in spec.properties.database.species
            ):
                raise ValueError("thermal membrane database and network species/charge must agree")
            if any(
                capacity > states[name].vessels.vessels[name].max_volume_L
                for name, capacity in zip(names, spec.capacities_L, strict=True)
            ):
                raise ValueError("thermal membrane model exceeds actual finite vessel capacity")
            if not any(
                components[c["source"]["component"]].vessel == names[0]
                and components[c["target"]["component"]].vessel == names[1]
                for c in self.composition.connections
            ):
                raise ValueError("thermal membrane requires its declared two-liquid connection")

    def configure_flow(self, state: Any, action: dict[str, Any]) -> Any:
        raise ValueError("direct-contact thermal membrane has no declared pump/flow control")

    def advance(
        self, state: Any, action: dict[str, Any], *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        if (
            jacket_program is not None
            or action.get("operation") in {"run_flow", "set_flow_rate"}
            or (not heat and "target_temperature_K" in action)
        ):
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="thermal_membrane_control_not_supported",
            )
        try:
            record, spec, initial = unpack_membrane_distillation(state)
            hot_record, heater = connected_device(state)
            cold = state.inactive_vessels[spec.cold_vessel_id]
            _, cooler = connected_device(cold)
            if heat:
                heater = replace(
                    heater, enabled=True, setpoint_K=float(action["target_temperature_K"])
                )
        except (KeyError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="invalid_thermal_membrane_control",
                diagnostics={"control_error": str(exc)},
            )
        try:
            result, diagnostic = advance_membrane_distillation(
                spec, initial, heater=heater, cooler=cooler, duration_s=float(action["duration_s"])
            )
            final, dt = result.final_state, result.actual_duration_s
            cold = install_membrane_liquid(
                cold, spec, 1, final.amounts_mol[1], final.enthalpies_J[1]
            )
            cold = cold.replace(
                ledger=advance_ledger(
                    cold.ledger,
                    dt,
                    energy_deltas={
                        "heat_loss_J": diagnostic["cooler_energy_out_of_cold_J"],
                    },
                )
            )
            candidate = install_membrane_liquid(
                state, spec, 0, final.amounts_mol[0], final.enthalpies_J[0]
            )
            equipment = dict(candidate.equipment.equipment)
            equipment[hot_record.equipment_id] = replace(
                hot_record,
                settings={
                    **hot_record.settings,
                    "thermal_device": heater.to_dict(),
                },
            )
            ledger = upsert_equipment_record(
                EquipmentLedger(equipment),
                equipment_id=record.equipment_id,
                equipment_type=record.equipment_type,
                attached_vessel_id=state.vessel_id,
                batch_reset="preserve",
                settings={"last_membrane_distillation": diagnostic},
            )
            candidate = candidate.replace(
                equipment=ledger,
                inactive_vessels={**state.inactive_vessels, spec.cold_vessel_id: cold},
                vessel_elapsed_s={
                    **state.vessel_elapsed_s,
                    spec.cold_vessel_id: state.vessel_elapsed_s.get(spec.cold_vessel_id, 0) + dt,
                },
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    energy_deltas={
                        "energy_jacket_J": diagnostic["heater_energy_into_hot_J"],
                        "heat_loss_J": diagnostic["cooler_energy_out_of_cold_J"],
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="thermal_membrane_solver_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            vessel_thermal_deltas={
                state.vessel_id: {"energy_jacket_J": diagnostic["heater_energy_into_hot_J"]},
                spec.cold_vessel_id: {"heat_loss_J": diagnostic["cooler_energy_out_of_cold_J"]},
            },
            diagnostics={
                "membrane_distillation": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
            },
        )
