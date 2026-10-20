"""Finite transport strategy; source and receiver retain their original ledgers."""

from dataclasses import replace
from math import isclose, isfinite

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.transport_devices import (
    FiniteTransportSpec,
    TransportInventory,
    advance_transport,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

TRANSPORT_TYPE = "finite_pump_path"


def transport_record(state):
    return attached_device_record(state, TRANSPORT_TYPE)


def configure_liquid_flow(state, action, record, *, supported_paths=("process",)):
    flow = float(action["flow_rate_mL_min"])
    if not isfinite(flow) or not 0 <= flow <= 20:
        raise ValueError("requested flow must be within 0-20 mL/min")
    path = action.get(
        "flow_path", record.settings.get("flow_control", {}).get("flow_path", "process")
    )
    if path not in supported_paths:
        raise ValueError("flow path is not supported by the installed apparatus")
    return state.replace(
        equipment=upsert_equipment_record(
            state.equipment, equipment_id=record.equipment_id,
            equipment_type=record.equipment_type, attached_vessel_id=state.vessel_id,
            batch_reset="preserve",
            settings={"flow_control": {"configured": True, "requested_flow_m3_s": flow / 6e7,
                                       "flow_path": path}},
        )
    )


def enthalpy_slots(state, fluid_h, body_h):
    return StateSlots(
        tuple(
            StateSlot(
                StateSlotSpec(
                    f"{state.vessel_id}:{owner}:enthalpy",
                    owner,
                    state.vessel_id,
                    "liquid_transport_enthalpy_v1",
                    (StateFieldSpec("enthalpy_J", "J", minimum=None),),
                    phase_id="liquid" if owner == "material" else None,
                    batch_reset="initial",
                ),
                state.vessel_id,
                {"enthalpy_J": value},
            )
            for owner, value in (("material", fluid_h), ("vessel", body_h))
        )
    )


def install_tank(state, spec, index, amounts, enthalpy):
    temperature = spec.at_ph(index, amounts, enthalpy)
    fluid_h = spec.properties.enthalpy(amounts, temperature)
    volume = (
        1000 * spec.properties.evaluate(amounts, temperature)[2] if sum(amounts.values()) else 0.0
    )
    return state.replace(
        species_amounts=amounts,
        temperature_K=temperature,
        volume_L=volume,
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
        state_slots=enthalpy_slots(state, fluid_h, enthalpy - fluid_h),
        process=replace(state.process, last_observation_record=None),
    )


def unpack_transport(state):
    record = transport_record(state)
    if record is None:
        raise ValueError("no pump path is installed")
    spec = FiniteTransportSpec.from_dict(record.settings["physical_model"])
    source = state.inactive_vessels[spec.source_vessel_id]
    amounts, enthalpies = [], []
    for index, tank in enumerate((source, state)):
        if set(tank.species_amounts) != set(spec.properties.component_ids) or set(
            tank.phases.phases
        ) != {"liquid"}:
            raise ValueError("transport requires declared single-liquid inventories")
        if not isclose(tank.pressure_Pa, spec.properties.pressure_Pa, abs_tol=1e-6):
            raise ValueError("connected reservoirs must share the declared pressure")
        n = dict(tank.species_amounts)
        volume = (
            1000 * spec.properties.evaluate(n, tank.temperature_K)[2] if sum(n.values()) else 0.0
        )
        if not isclose(volume, tank.volume_L, abs_tol=1e-10, rel_tol=1e-8):
            raise ValueError("tank volume differs from its declared liquid density")
        fluid_h = spec.properties.enthalpy(n, tank.temperature_K)
        h = spec.enthalpy(index, n, tank.temperature_K)
        expected = enthalpy_slots(tank, fluid_h, h - fluid_h)
        if len(tank.state_slots.slots) != 2:
            raise ValueError("transport cannot silently discard other owned state")
        actual = {s.spec.slot_id: s for s in tank.state_slots.slots}
        for slot in expected.slots:
            other = actual.get(slot.spec.slot_id)
            if (
                other is None
                or other.spec != slot.spec
                or not isclose(
                    other.values["enthalpy_J"],
                    slot.values["enthalpy_J"],
                    abs_tol=1e-7,
                    rel_tol=1e-8,
                )
            ):
                raise ValueError("tank enthalpy ownership or physical state disagrees")
        amounts.append(n)
        enthalpies.append(sum(s.values["enthalpy_J"] for s in tank.state_slots.slots))
    return record, spec, TransportInventory(tuple(amounts), tuple(enthalpies), state.ledger.time_s)


class TransportRuntime:
    supported_operations = ("set_flow_rate", "run_flow", "wait")

    def __init__(self, composition=None):
        self.composition = composition

    def supports(self, state):
        return transport_record(state) is not None

    @property
    def preserved_source_vessels(self):
        if self.composition is None or self.composition.physics is None:
            return set()
        return {
            FiniteTransportSpec.from_dict(
                transport_record(s).settings["physical_model"]
            ).source_vessel_id
            for s in self.composition.physics.initial_states.values()
            if self.supports(s)
        }

    def reconcile_initialization(self, transition):
        active = transition.state
        locals_ = {active.vessel_id: active.replace(inactive_vessels={}), **active.inactive_vessels}
        changed = {}
        for receiver, state in tuple(locals_.items()):
            if not self.supports(state):
                continue
            spec = FiniteTransportSpec.from_dict(transport_record(state).settings["physical_model"])
            for index, name in enumerate((spec.source_vessel_id, receiver)):
                if name not in transition.receipt.get("vessels", {}):
                    continue
                local = locals_[name]
                enthalpy = sum(s.values["enthalpy_J"] for s in local.state_slots.slots)
                locals_[name] = install_tank(
                    local, spec, index, dict(local.species_amounts), enthalpy
                )
                changed[name] = {
                    "total_enthalpy_J": enthalpy,
                    "temperature_K": locals_[name].temperature_K,
                }
        if not changed:
            return transition
        local = locals_.pop(active.vessel_id)
        return replace(
            transition,
            state=local.replace(inactive_vessels=locals_),
            receipt={**transition.receipt, "transport_equilibration": changed},
        )

    def validate_world(self):
        declared = {c.vessel for c in self.composition.components if c.kind == "finite_transport"}
        states = self.composition.physics.initial_states
        installed = {v for v, s in states.items() if self.supports(s)}
        if declared != installed:
            raise ValueError("finite transport components must match installed devices")
        sources = set()
        for vessel in installed:
            state = states[vessel].replace(
                inactive_vessels={v: s for v, s in states.items() if v != vessel}
            )
            _, spec, _ = unpack_transport(state)
            if spec.source_vessel_id in sources or spec.source_vessel_id in installed:
                raise ValueError("first finite network has no shared feeds or recirculation")
            sources.add(spec.source_vessel_id)
            if self.composition.physics.network.reactions:
                raise ValueError("finite tank transport does not yet combine reaction kinetics")
            components = {c.id: c for c in self.composition.components}
            if not any(
                components[c["source"]["component"]].vessel == spec.source_vessel_id
                and components[c["target"]["component"]].vessel == vessel
                for c in self.composition.connections
            ):
                raise ValueError("finite feed needs a declared source-to-receiver connection")
            if spec.receiver_capacity_L > states[vessel].vessels.vessels[vessel].max_volume_L:
                raise ValueError("transport capacity exceeds receiver hardware")

    def configure_flow(self, state, action):
        record, _, _ = unpack_transport(state)
        return configure_liquid_flow(state, action, record)

    def advance(self, state, action, *, heat=False, jacket_program=None):
        if heat or jacket_program is not None:
            return ModelProviderResult(
                {}, success=False, outcome="rejected", failure_reason="use_flow_utility_control"
            )
        try:
            record, spec, inventory = unpack_transport(state)
            flow = record.settings.get("flow_control", {}).get("requested_flow_m3_s", 0.0)
            if "target_temperature_K" in action:
                spec = replace(
                    spec,
                    path=replace(
                        spec.path, utility_temperature_K=float(action["target_temperature_K"])
                    ),
                )
        except ValueError as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="invalid_flow_configuration",
                diagnostics={"control_error": str(exc)},
            )
        try:
            result, diagnostic = advance_transport(
                spec, inventory, flow, float(action["duration_s"])
            )
            final, dt = result.final_state, result.actual_duration_s
            source = install_tank(
                state.inactive_vessels[spec.source_vessel_id],
                spec,
                0,
                final.amounts_mol[0],
                final.enthalpies_J[0],
            )
            source = source.replace(ledger=advance_ledger(source.ledger, dt))
            candidate = install_tank(state, spec, 1, final.amounts_mol[1], final.enthalpies_J[1])
            heat_j = diagnostic["utility_heat_J"]
            candidate = candidate.replace(
                inactive_vessels={**state.inactive_vessels, spec.source_vessel_id: source},
                vessel_elapsed_s={
                    **state.vessel_elapsed_s,
                    spec.source_vessel_id: state.vessel_elapsed_s.get(spec.source_vessel_id, 0.0)
                    + dt,
                },
                equipment=upsert_equipment_record(
                    candidate.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"physical_model": spec.to_dict(), "last_transport": diagnostic},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    cost_delta=diagnostic["shaft_work_J"] / 250000,
                    energy_deltas={
                        "flow_pump_work_J": diagnostic["shaft_work_J"],
                        "energy_jacket_J": max(heat_j, 0),
                        "heat_loss_J": max(-heat_j, 0) + diagnostic["external_loss_J"],
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="finite_transport_solver_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            diagnostics={
                "transport": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
                "energy_scope": (
                    "liquid sensible H; hydraulic dissipation; explicit utility and motor loss"
                ),
            },
        )
