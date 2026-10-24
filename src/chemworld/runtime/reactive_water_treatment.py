"""Original finite-transport adapter for a finite neutral reactive CSTR."""

from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.reactive_water_treatment import (
    ReactiveWaterTreatmentSpec,
    advance_reactive_water_treatment,
    reactive_water_treatment_inventory,
)
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

REACTIVE_TREATMENT_TYPE = "finite_reactive_water_treatment"


def reactive_treatment_record(state: Any) -> Any:
    return attached_device_record(state, REACTIVE_TREATMENT_TYPE)


def reactive_treatment_vessels(state: Any, spec: ReactiveWaterTreatmentSpec) -> tuple[str, ...]:
    names = (state.vessel_id, spec.reactor_vessel_id, spec.product_vessel_id)
    if len(set(names)) != 3:
        raise ValueError("reactive treatment requires three distinct original liquid tanks")
    return names


def install_reactive_treatment_liquid(
    state: Any, spec: ReactiveWaterTreatmentSpec, amounts: Any, enthalpy: float
) -> Any:
    from chemworld.runtime.transport_devices import enthalpy_slots

    volume = 1000 * spec.volume_m3(amounts)
    return state.replace(
        species_amounts=dict(amounts),
        temperature_K=spec.temperature_K,
        pressure_Pa=spec.properties.pressure_Pa,
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


def unpack_reactive_treatment(
    state: Any,
) -> tuple[Any, ReactiveWaterTreatmentSpec, TransportInventory]:
    from chemworld.runtime.transport_devices import enthalpy_slots

    record = reactive_treatment_record(state)
    if record is None:
        raise ValueError("no reactive water-treatment device is installed")
    spec = ReactiveWaterTreatmentSpec.from_dict(record.settings["physical_model"])
    amounts, enthalpies = [], []
    for name in reactive_treatment_vessels(state, spec):
        tank = state if name == state.vessel_id else state.inactive_vessels[name]
        n = dict(tank.species_amounts)
        expected_h = spec.properties.enthalpy(n, spec.temperature_K)
        volume = 1000 * spec.volume_m3(n)
        phase = tank.phases.phases.get("liquid")
        if (
            not isclose(tank.temperature_K, spec.temperature_K, rel_tol=0, abs_tol=1e-9)
            or not isclose(tank.pressure_Pa, spec.properties.pressure_Pa, rel_tol=0, abs_tol=1e-5)
            or not isclose(tank.volume_L, volume, rel_tol=1e-9, abs_tol=1e-10)
            or set(tank.phases.phases) != {"liquid"}
            or phase is None
            or phase.phase_id != "liquid"
            or phase.vessel_id != name
            or phase.phase_type != "liquid"
            or not phase.settled
            or not phase.selected
            or phase.solvents.volume_L != 0
            or dict(phase.species_amounts_mol) != n
            or phase.metadata.get("carrier_representation") != "species_inventory"
            or not isclose(phase.volume_L, volume, rel_tol=1e-9, abs_tol=1e-10)
        ):
            raise ValueError("reactive treatment original liquid n/H/T/P/V/phase disagree")
        wanted = {s.spec.slot_id: s for s in enthalpy_slots(tank, expected_h, 0).slots}
        owned = {s.spec.slot_id: s for s in tank.state_slots.slots}
        if set(wanted) != set(owned) or any(
            owned[k].spec != v.spec
            or owned[k].location_id != name
            or not isclose(
                owned[k].values["enthalpy_J"], v.values["enthalpy_J"], rel_tol=1e-9, abs_tol=1e-7
            )
            for k, v in wanted.items()
        ):
            raise ValueError("reactive treatment original H/slot ownership differs from properties")
        amounts.append(n)
        enthalpies.append(owned[f"{name}:material:enthalpy"].values["enthalpy_J"])
    reactive_water_treatment_inventory(spec, *amounts)
    return record, spec, TransportInventory(tuple(amounts), tuple(enthalpies), state.ledger.time_s)


class ReactiveWaterTreatmentAdapter:
    supported_operations = ("set_flow_rate", "run_flow", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return reactive_treatment_record(state) is not None

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = ReactiveWaterTreatmentSpec.from_dict(
            reactive_treatment_record(state).settings["physical_model"]
        )
        names = reactive_treatment_vessels(state, spec)
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            names,
            connection_ids(self.composition, [(names[0], names[1]), (names[1], names[2])]),
        )

    @property
    def preserved_source_vessels(self) -> set[str]:
        names: set[str] = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    spec = ReactiveWaterTreatmentSpec.from_dict(
                        reactive_treatment_record(state).settings["physical_model"]
                    )
                    names.update(reactive_treatment_vessels(state, spec))
        return names

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def validate_state(self, state: Any) -> None:
        record, _, _ = unpack_reactive_treatment(state)
        if self.composition is None or self.composition.physics is None:
            raise ValueError("reactive treatment requires its original composition")
        original = self.composition.physics.initial_states.get(state.vessel_id)
        source = reactive_treatment_record(original) if original is not None else None
        if source is None or source.settings["physical_model"] != record.settings["physical_model"]:
            raise ValueError("reactive treatment physical_model differs from original composition")

    def reconcile_initialization(self, transition: Any) -> Any:
        active = transition.state
        states = {active.vessel_id: active, **active.inactive_vessels}
        for name, state in states.items():
            if self.supports(state):
                self.validate_state(
                    state.replace(inactive_vessels={k: v for k, v in states.items() if k != name})
                )
        return transition

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        components = {c.id: c for c in self.composition.components}
        occupied: set[str] = set()
        for name, state in states.items():
            if not self.supports(state):
                continue
            if not any(
                c.kind == "finite_transport" and c.vessel == name for c in components.values()
            ):
                raise ValueError("reactive treatment needs its original finite_transport component")
            state = state.replace(inactive_vessels={k: v for k, v in states.items() if k != name})
            self.validate_state(state)
            _, spec, _ = unpack_reactive_treatment(state)
            names = reactive_treatment_vessels(state, spec)
            if occupied.intersection(names):
                raise ValueError("reactive treatment cannot share physical reservoirs")
            occupied.update(names)
            if self.composition.physics.network.to_dict() != spec.network.to_dict():
                raise ValueError(
                    "reactive treatment and original network must have identical kinetics"
                )
            for vessel, capacity in zip(names, spec.capacities_L, strict=True):
                if capacity > states[vessel].vessels.vessels[vessel].max_volume_L:
                    raise ValueError("reactive treatment exceeds original finite vessel capacity")
            edges = {
                (
                    components[e["source"]["component"]].vessel,
                    components[e["target"]["component"]].vessel,
                )
                for e in self.composition.connections
            }
            if not {(names[0], names[1]), (names[1], names[2])} <= edges:
                raise ValueError(
                    "reactive treatment requires its original two material connections"
                )

    def configure_flow(self, state: Any, action: dict[str, Any]) -> Any:
        from chemworld.runtime.transport_devices import configure_liquid_flow

        self.validate_state(state)
        return configure_liquid_flow(state, action, reactive_treatment_record(state))

    def advance(
        self, state: Any, action: dict[str, Any], *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        try:
            self.validate_state(state)
            record, spec, initial = unpack_reactive_treatment(state)
            if (
                heat
                or jacket_program is not None
                or action.get("operation") not in {"wait", "run_flow"}
                or action.get("target_temperature_K", spec.temperature_K) != spec.temperature_K
            ):
                return ModelProviderResult(
                    {},
                    success=False,
                    outcome="rejected",
                    diagnostics={},
                    failure_reason="reactive_treatment_requires_declared_isotherm",
                )
            flow = record.settings.get("flow_control", {}).get("requested_flow_m3_s", 0.0)
            result, diagnostic = advance_reactive_water_treatment(
                spec, initial, flow, float(action["duration_s"])
            )
            final, dt = result.final_state, result.actual_duration_s
            inactive, ages = dict(state.inactive_vessels), dict(state.vessel_elapsed_s)
            names = reactive_treatment_vessels(state, spec)
            for i, name in enumerate(names[1:], 1):
                tank = install_reactive_treatment_liquid(
                    inactive[name], spec, final.amounts_mol[i], final.enthalpies_J[i]
                )
                inactive[name] = tank.replace(ledger=advance_ledger(tank.ledger, dt))
                ages[name] = ages.get(name, 0) + dt
            candidate = install_reactive_treatment_liquid(
                state, spec, final.amounts_mol[0], final.enthalpies_J[0]
            )
            energy = {
                "flow_pump_work_J": diagnostic["pump_work_J"],
                "energy_jacket_J": diagnostic["bath_heating_J"],
                "heat_loss_J": diagnostic["motor_loss_J"] + diagnostic["bath_cooling_J"],
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
                    settings={"last_reactive_treatment": diagnostic},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    cost_delta=diagnostic["pump_work_J"] / 250000,
                    energy_deltas=energy,
                ),
            )
        except (ArithmeticError, KeyError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="reactive_treatment_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            vessel_thermal_deltas={
                names[0]: {
                    "energy_jacket_J": diagnostic["bath_heating_J"],
                    "heat_loss_J": diagnostic["motor_loss_J"] + diagnostic["bath_cooling_J"],
                },
                names[1]: {},
                names[2]: {},
            },
            diagnostics={
                "reactive_treatment": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "conservation_residuals": dict(result.conservation_residuals),
                "energy_balance_residual_J": result.energy_balance_residual_J,
            },
        )
