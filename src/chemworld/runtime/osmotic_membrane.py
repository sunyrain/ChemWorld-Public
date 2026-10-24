"""Two finite osmotic tanks on the original membrane transaction strategy."""

from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.osmotic_membrane import (
    OsmoticMembraneSpec,
    advance_osmotic_membrane,
    osmotic_membrane_inventory,
    osmotic_membrane_point,
)
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.transport_devices import enthalpy_slots

OSMOTIC_MEMBRANE_TYPE = "finite_isothermal_osmotic_membrane"


def osmotic_membrane_record(state: Any) -> Any:
    return attached_device_record(state, OSMOTIC_MEMBRANE_TYPE)


def osmotic_membrane_vessels(state: Any, spec: OsmoticMembraneSpec) -> tuple[str, str]:
    if state.vessel_id == spec.right_vessel_id:
        raise ValueError("osmotic membrane requires two distinct original finite tanks")
    return state.vessel_id, spec.right_vessel_id


def install_osmotic_liquid(
    state: Any,
    spec: OsmoticMembraneSpec,
    index: int,
    amounts: Any,
    enthalpy: float,
) -> Any:
    pressure = spec.pressures_Pa[index]
    volume = 1000 * spec.properties.volume(amounts, spec.temperature_K, pressure)
    return state.replace(
        species_amounts=dict(amounts),
        temperature_K=spec.temperature_K,
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
        state_slots=enthalpy_slots(state, enthalpy, 0.0),
        process=replace(state.process, last_observation_record=None),
    )


def unpack_osmotic_membrane(state: Any) -> tuple[Any, OsmoticMembraneSpec, TransportInventory]:
    record = osmotic_membrane_record(state)
    if record is None:
        raise ValueError("no finite isothermal osmotic membrane is installed")
    spec = OsmoticMembraneSpec.from_dict(record.settings["physical_model"])
    amounts, enthalpies = [], []
    for i, name in enumerate(osmotic_membrane_vessels(state, spec)):
        tank = state if i == 0 else state.inactive_vessels[name]
        n = dict(tank.species_amounts)
        spec.properties.validate_amounts(n)
        expected = spec.enthalpy(i, n)
        slots = enthalpy_slots(tank, expected, 0.0)
        actual = {slot.spec.slot_id: slot for slot in tank.state_slots.slots}
        phase = tank.phases.phases.get("liquid")
        volume = 1000 * spec.properties.volume(n, spec.temperature_K, spec.pressures_Pa[i])
        if (
            not isclose(tank.temperature_K, spec.temperature_K, rel_tol=0, abs_tol=1e-9)
            or not isclose(tank.pressure_Pa, spec.pressures_Pa[i], rel_tol=0, abs_tol=1e-5)
            or not isclose(tank.volume_L, volume, rel_tol=1e-8, abs_tol=1e-10)
            or set(tank.phases.phases) != {"liquid"}
            or phase is None
            or phase.phase_id != "liquid"
            or phase.vessel_id != name
            or phase.phase_type != "liquid"
            or not phase.settled
            or not phase.selected
            or phase.metadata.get("carrier_representation") != "species_inventory"
            or phase.solvents.volume_L != 0
            or dict(phase.species_amounts_mol) != n
            or not isclose(phase.volume_L, volume, rel_tol=1e-8, abs_tol=1e-10)
            or set(actual) != {slot.spec.slot_id for slot in slots.slots}
        ):
            raise ValueError("osmotic membrane needs matching original liquid n/H/T/P/V/phase")
        for wanted in slots.slots:
            owned = actual[wanted.spec.slot_id]
            if (
                owned.spec != wanted.spec
                or owned.location_id != name
                or not isclose(
                    owned.values["enthalpy_J"],
                    wanted.values["enthalpy_J"],
                    rel_tol=1e-8,
                    abs_tol=3e-6,
                )
            ):
                raise ValueError(
                    "osmotic original H differs from declared common-temperature calorics"
                )
        amounts.append(n)
        # Keep the original material slot; never reconstruct persistent H from T.
        enthalpies.append(actual[f"{name}:material:enthalpy"].values["enthalpy_J"])
    osmotic_membrane_inventory(spec, *amounts)
    return record, spec, TransportInventory(tuple(amounts), tuple(enthalpies), state.ledger.time_s)


def read_osmotic_membrane(state: Any) -> dict[str, Any]:
    """Pure constitutive query; this is not an instrument observation."""
    _, spec, inventory = unpack_osmotic_membrane(state)
    return {
        **osmotic_membrane_point(spec, inventory.amounts_mol),
        "quantity_kind": "pure_constitutive_query",
        "instrument_id": None,
    }


class OsmoticMembraneRuntime:
    supported_operations = ("wait",)

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return osmotic_membrane_record(state) is not None

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = OsmoticMembraneSpec.from_dict(
            osmotic_membrane_record(state).settings["physical_model"]
        )
        names = osmotic_membrane_vessels(state, spec)
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            names,
            connection_ids(self.composition, [names]),
        )

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    @property
    def preserved_source_vessels(self) -> set[str]:
        names: set[str] = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    spec = OsmoticMembraneSpec.from_dict(
                        osmotic_membrane_record(state).settings["physical_model"],
                    )
                    names.update(osmotic_membrane_vessels(state, spec))
        return names

    def validate_state(self, state: Any) -> None:
        record, _, _ = unpack_osmotic_membrane(state)
        if self.composition is None or self.composition.physics is None:
            raise ValueError("osmotic membrane needs its original composition")
        initial = self.composition.physics.initial_states.get(state.vessel_id)
        original = osmotic_membrane_record(initial) if initial is not None else None
        if (
            original is None
            or record.settings["physical_model"] != original.settings["physical_model"]
        ):
            raise ValueError("osmotic physical_model differs from original composition identity")

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        components = {component.id: component for component in self.composition.components}
        occupied: set[str] = set()
        for name, original in states.items():
            if not self.supports(original):
                continue
            if not any(c.kind == "ro_membrane" and c.vessel == name for c in components.values()):
                raise ValueError(
                    "osmotic membrane requires the original membrane strategy component"
                )
            state = original.replace(
                inactive_vessels={n: s for n, s in states.items() if n != name}
            )
            self.validate_state(state)
            _, spec, _ = unpack_osmotic_membrane(state)
            names = osmotic_membrane_vessels(state, spec)
            if occupied.intersection(names) or self.composition.physics.network.reactions:
                raise ValueError("osmotic membrane excludes overlapping reservoirs and reactions")
            occupied.update(names)
            network = {
                s.species_id: (dict(s.composition), s.charge, s.phase)
                for s in self.composition.physics.network.species
            }
            if any(
                network.get(s.species_id) != (dict(s.composition), s.charge, "liquid")
                for s in spec.properties.database.species
            ):
                raise ValueError("osmotic database and original network species/charge must agree")
            if any(
                capacity > states[vessel].vessels.vessels[vessel].max_volume_L
                for vessel, capacity in zip(names, spec.capacities_L, strict=True)
            ):
                raise ValueError("osmotic model exceeds actual finite vessel capacity")
            if not any(
                components[connection["source"]["component"]].vessel == names[0]
                and components[connection["target"]["component"]].vessel == names[1]
                for connection in self.composition.connections
            ):
                raise ValueError("osmotic membrane requires its declared two-liquid connection")

    def configure_flow(self, state: Any, action: dict[str, Any]) -> Any:
        raise ValueError("osmotic membrane has no declared pump/flow control")

    def advance(
        self,
        state: Any,
        action: dict[str, Any],
        *,
        heat: bool = False,
        jacket_program: Any = None,
    ) -> ModelProviderResult:
        if heat or jacket_program is not None or action.get("operation") != "wait":
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                diagnostics={},
                failure_reason="osmotic_membrane_control_not_supported",
            )
        try:
            self.validate_state(state)
            record, spec, initial = unpack_osmotic_membrane(state)
            result, diagnostic = advance_osmotic_membrane(
                spec, initial, duration_s=float(action["duration_s"])
            )
            final, dt = result.final_state, result.actual_duration_s
            right = state.inactive_vessels[spec.right_vessel_id]
            right = install_osmotic_liquid(
                right, spec, 1, final.amounts_mol[1], final.enthalpies_J[1]
            )
            right = right.replace(ledger=advance_ledger(right.ledger, dt))
            candidate = install_osmotic_liquid(
                state, spec, 0, final.amounts_mol[0], final.enthalpies_J[0]
            )
            equipment = upsert_equipment_record(
                candidate.equipment,
                equipment_id=record.equipment_id,
                equipment_type=record.equipment_type,
                attached_vessel_id=state.vessel_id,
                batch_reset="preserve",
                settings={"last_osmotic_membrane": diagnostic},
            )
            candidate = candidate.replace(
                equipment=equipment,
                inactive_vessels={**state.inactive_vessels, spec.right_vessel_id: right},
                vessel_elapsed_s={
                    **state.vessel_elapsed_s,
                    spec.right_vessel_id: state.vessel_elapsed_s.get(spec.right_vessel_id, 0) + dt,
                },
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    energy_deltas={
                        "energy_jacket_J": diagnostic["bath_heating_J"],
                        "heat_loss_J": diagnostic["bath_cooling_J"],
                        "boundary_work_J": diagnostic["boundary_work_J"],
                    },
                ),
            )
        except (ArithmeticError, KeyError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="osmotic_membrane_solver_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            # The membrane's compensation bath belongs to its anchor equipment.
            # Attribute that single boundary to the anchor's existing thermal
            # ledger; do not fabricate a second tank heater or duplicate heat.
            vessel_thermal_deltas={
                state.vessel_id: {
                    "energy_jacket_J": diagnostic["bath_heating_J"],
                    "heat_loss_J": diagnostic["bath_cooling_J"],
                },
                spec.right_vessel_id: {},
            },
            diagnostics={
                "osmotic_membrane": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
            },
        )
