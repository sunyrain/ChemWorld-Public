"""Native finite RO strategy using common flow, transaction and lifecycle paths."""

from dataclasses import replace
from math import isclose

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.membranes import (
    MembraneSpec,
    advance_membrane,
    liquid_volume_m3,
    salt_concentration,
)
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.transport_devices import (
    configure_liquid_flow,
    enthalpy_slots,
)

MEMBRANE_TYPE = "dilute_ro_membrane"
FOULING_STATE = "membrane_bound_foulant_v1"


def membrane_record(state):
    return attached_device_record(state, MEMBRANE_TYPE)


def fouling_slot(vessel, device, amount=0.0):
    return StateSlot(
        StateSlotSpec(
            f"{device}:foulant",
            "equipment",
            device,
            FOULING_STATE,
            (StateFieldSpec("bound_mol", "mol", species_id="foulant"),),
            batch_reset="preserve",
        ),
        vessel,
        {"bound_mol": amount},
    )


def set_fouling_inventory(state, device, amount):
    previous = next(
        (s for s in state.state_slots.slots if s.spec.state_type == FOULING_STATE), None
    )
    slot = (
        replace(previous, values={"bound_mol": amount})
        if previous is not None
        else fouling_slot(state.vessel_id, device, amount)
    )
    return state.replace(
        state_slots=StateSlots(
            (
                *(s for s in state.state_slots.slots if s.spec.state_type != FOULING_STATE),
                slot,
            )
        )
    )


def membrane_liquid_ids(state, spec):
    return (
        state.vessel_id,
        spec.collector_vessel_id,
        *((spec.fouling.wash_vessel_id, spec.fouling.waste_vessel_id) if spec.fouling else ()),
    )


def install_liquid(state, amounts, temperature=298.15):
    volume = 1000 * liquid_volume_m3(amounts)
    return state.replace(
        species_amounts=amounts,
        volume_L=volume,
        temperature_K=temperature,
        state_slots=StateSlots(
            (
                *enthalpy_slots(state, 0.0, 0.0).slots,
                *(
                    s
                    for s in state.state_slots.slots
                    if s.spec.state_type != "liquid_transport_enthalpy_v1"
                ),
            )
        ),
        phases=PhaseLedger(
            {
                "liquid": PhaseRecord(
                    "liquid",
                    state.vessel_id,
                    "liquid",
                    volume,
                    amounts,
                    settled=True,
                    selected=True,
                    metadata={"carrier_representation": "species_inventory"},
                )
            }
        ),
        process=replace(state.process, last_observation_record=None),
    )


def unpack_membrane(state):
    record = membrane_record(state)
    if record is None:
        raise ValueError("no membrane is installed")
    spec = MembraneSpec.from_dict(record.settings["physical_model"])
    amounts = []
    expected_ids = spec.properties.component_ids
    bound = [s for s in state.state_slots.slots if s.spec.state_type == FOULING_STATE]
    if spec.fouling:
        expected = fouling_slot(state.vessel_id, record.equipment_id)
        if (
            len(bound) != 1
            or bound[0].spec != expected.spec
            or bound[0].location_id != state.vessel_id
        ):
            raise ValueError("fouling inventory must belong to the installed membrane")
    elif bound:
        raise ValueError("clean-only membrane has undeclared bound inventory")
    for index, name in enumerate(membrane_liquid_ids(state, spec)):
        local = state if name == state.vessel_id else state.inactive_vessels[name]
        # Independently owned static instrument stocks are outside the liquid
        # membrane inventory and survive install_liquid unchanged. Dynamic or
        # thermal extra state still requires an explicit coupled model.
        passive = {slot.spec.slot_id for slot in local.state_slots.slots
                   if slot.spec.owner_kind == "equipment"
                   and slot.spec.owner_id != record.equipment_id
                   and slot.spec.time_evolution == "static"
                   and slot.spec.heat_capacity_field is None and not slot.spec.energy_interactions}
        if (
            set(local.species_amounts) != set(expected_ids)
            or set(local.phases.phases) != {"liquid"}
            or local.temperature_K != spec.temperature_K
        ):
            raise ValueError("membrane requires declared isothermal water/NaCl liquids")
        n = dict(local.species_amounts)
        if (
            not isclose(local.volume_L, liquid_volume_m3(n) * 1000, abs_tol=1e-10)
            or salt_concentration(n) > spec.concentration_limit_mol_m3 * (1 + 1e-8)
            or {
                s.spec.slot_id: s
                for s in local.state_slots.slots
                if s.spec.state_type != FOULING_STATE
                and s.spec.slot_id not in passive
            }
            != {s.spec.slot_id: s for s in enthalpy_slots(local, 0.0, 0.0).slots}
        ):
            raise ValueError("membrane liquid volume, domain or caloric ownership disagrees")
        if local.phases.phases["liquid"].species_amounts_mol != local.species_amounts:
            raise ValueError("membrane phase and bulk inventories disagree")
        amounts.append(n)
        if spec.fouling and (
            (
                index != 0
                and any(s.spec.state_type == FOULING_STATE for s in local.state_slots.slots)
            )
            or (index == 1 and n["foulant"] != 0)
            or (index == 2 and (n["foulant"] != 0 or n["nacl"] != 0))
        ):
            raise ValueError("declared rinse uses pure wash and the membrane rejects all foulant")
    if spec.fouling:
        amounts.append({"water": 0.0, "nacl": 0.0, "foulant": bound[0].values["bound_mol"]})
    return (
        record,
        spec,
        TransportInventory(tuple(amounts), (0.0,) * len(amounts), state.ledger.time_s),
    )


class MembraneRuntime:
    supported_operations = ("set_flow_rate", "run_flow", "wait")

    def __init__(self, composition=None):
        self.composition = composition

    def supports(self, state):
        return membrane_record(state) is not None

    def shared_system(self, state, strategy_id):
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = MembraneSpec.from_dict(membrane_record(state).settings["physical_model"])
        pairs = [(state.vessel_id, spec.collector_vessel_id)]
        if spec.fouling:
            pairs += [(spec.fouling.wash_vessel_id, state.vessel_id),
                      (state.vessel_id, spec.fouling.waste_vessel_id)]
        return SharedAdvanceSystem(
            state.vessel_id, strategy_id, membrane_liquid_ids(state, spec),
            connection_ids(self.composition, pairs),
        )

    @property
    def protected_lifecycle_vessels(self):
        result = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    spec = MembraneSpec.from_dict(membrane_record(state).settings["physical_model"])
                    if spec.fouling:
                        result.add(spec.fouling.wash_vessel_id)
        return result

    @property
    def preserved_source_vessels(self):
        if self.composition is None or self.composition.physics is None:
            return set()
        result = set()
        for state in self.composition.physics.initial_states.values():
            if self.supports(state):
                spec = MembraneSpec.from_dict(membrane_record(state).settings["physical_model"])
                result.update(membrane_liquid_ids(state, spec))
        return result

    def validate_world(self):
        states = self.composition.physics.initial_states
        installed = {v for v, s in states.items() if self.supports(s)}
        declared = {c.vessel for c in self.composition.components if c.kind == "ro_membrane"}
        if installed != declared:
            raise ValueError("declared membranes must match installed devices")
        collectors = set()
        for vessel in installed:
            state = states[vessel].replace(
                inactive_vessels={v: s for v, s in states.items() if v != vessel}
            )
            _, spec, _ = unpack_membrane(state)
            auxiliary = set(membrane_liquid_ids(state, spec)[1:])
            if auxiliary & (installed | collectors):
                raise ValueError("finite RO has no shared collectors or recirculation")
            collectors.update(auxiliary)
            if self.composition.physics.network.reactions:
                raise ValueError("this dilute membrane model has no reactions")
            components = {c.id: c for c in self.composition.components}
            if not any(
                components[c["source"]["component"]].vessel == vessel
                and components[c["target"]["component"]].vessel == spec.collector_vessel_id
                for c in self.composition.connections
            ):
                raise ValueError("membrane needs a declared finite collection port")
            if (
                spec.collector_capacity_L
                > states[spec.collector_vessel_id]
                .vessels.vessels[spec.collector_vessel_id]
                .max_volume_L
            ):
                raise ValueError("collector model exceeds physical capacity")
            if spec.fouling:
                waste = states[spec.fouling.waste_vessel_id]
                if (
                    spec.fouling.waste_capacity_L
                    > waste.vessels.vessels[waste.vessel_id].max_volume_L
                ):
                    raise ValueError("waste model exceeds physical capacity")
                connections = self.composition.connections
                for source, target in (
                    (spec.fouling.wash_vessel_id, vessel),
                    (vessel, spec.fouling.waste_vessel_id),
                ):
                    if not any(
                        components[c["source"]["component"]].vessel == source
                        and components[c["target"]["component"]].vessel == target
                        for c in connections
                    ):
                        raise ValueError("rinsing requires declared finite wash and waste ports")

    def configure_flow(self, state, action):
        record, spec, _ = unpack_membrane(state)
        return configure_liquid_flow(
            state,
            action,
            record,
            supported_paths=("process", "rinse") if spec.fouling else ("process",),
        )

    def advance(self, state, action, *, heat=False, jacket_program=None):
        record, spec, inventory = unpack_membrane(state)
        if (
            heat
            or jacket_program is not None
            or action.get("target_temperature_K", spec.temperature_K) != spec.temperature_K
        ):
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="membrane_requires_declared_isotherm",
            )
        flow = record.settings.get("flow_control", {}).get("requested_flow_m3_s", 0.0)
        path = record.settings.get("flow_control", {}).get("flow_path", "process")
        try:
            result, diagnostic = advance_membrane(
                spec, inventory, flow, float(action["duration_s"]), flow_path=path
            )
            final, dt = result.final_state, result.actual_duration_s
            inactive = dict(state.inactive_vessels)
            ages = dict(state.vessel_elapsed_s)
            for index, name in enumerate(membrane_liquid_ids(state, spec)[1:], start=1):
                liquid = install_liquid(inactive[name], final.amounts_mol[index])
                inactive[name] = liquid.replace(ledger=advance_ledger(liquid.ledger, dt))
                ages[name] = ages.get(name, 0) + dt
            candidate = install_liquid(state, final.amounts_mol[0])
            if spec.fouling:
                candidate = set_fouling_inventory(
                    candidate, record.equipment_id, final.amounts_mol[4]["foulant"]
                )
            candidate = candidate.replace(
                pressure_Pa=101325.0 + diagnostic["final"].get("membrane_pressure_Pa", 0),
                inactive_vessels=inactive,
                vessel_elapsed_s=ages,
                equipment=upsert_equipment_record(
                    state.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"last_membrane": diagnostic},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    cost_delta=diagnostic["shaft_work_J"] / 250000,
                    energy_deltas={
                        "flow_pump_work_J": diagnostic["shaft_work_J"],
                        "heat_loss_J": diagnostic["isothermal_heat_out_J"],
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="membrane_solver_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            diagnostics={
                "membrane": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
                "energy_scope": "isothermal ideal solution; swept hydraulic work and bath heat",
            },
        )
