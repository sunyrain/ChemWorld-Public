"""Complete ASM1 adapter of the original finite-transport transaction path."""

from __future__ import annotations

from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.activated_sludge import (
    WORK_FIELDS,
    ActivatedSludgeInventory,
    advance_activated_sludge,
    inventory_quality,
    validate_inventory,
)
from chemworld.physchem.activated_sludge_kinetics import (
    ASM1_COMPONENTS,
    ASM1_REACTIONS,
    COMPONENT_UNITS,
)
from chemworld.physchem.activated_sludge_specs import (
    ASM1_EQUIPMENT,
    ASM1_EXTENTS,
    ASM1_MATERIAL,
    ASM1_MECHANISM,
    ASM1_UTILITIES,
    UTILITY_FIELDS,
    ActivatedSludgeSpec,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

ASM1_WORK = "asm1_work_v1"
ASM1_CARRIER_H = "asm1_carrier_sensible_enthalpy_v1"


def activated_sludge_record(state: Any) -> Any:
    return attached_device_record(state, ASM1_EQUIPMENT)


def slot_declarations(spec: ActivatedSludgeSpec, index: int) -> tuple[StateSlotSpec, ...]:
    name = spec.member_vessel_ids[index]
    result = [
        StateSlotSpec(
            f"{name}:asm1:quality",
            "material",
            f"{name}:asm1:mixture",
            ASM1_MATERIAL,
            tuple(StateFieldSpec(k, COMPONENT_UNITS[k]) for k in ASM1_COMPONENTS),
            mechanism_id=ASM1_MECHANISM,
            phase_id="liquid",
            batch_reset="initial",
        ),
        StateSlotSpec(
            f"{name}:water:enthalpy",
            "material",
            f"{name}:water",
            ASM1_CARRIER_H,
            (StateFieldSpec("enthalpy_J", "J", minimum=None),),
            phase_id="liquid",
            batch_reset="initial",
        ),
    ]
    if name in spec.reactor_vessel_ids:
        result.append(
            StateSlotSpec(
                f"{name}:asm1:extents",
                "equipment",
                f"asm1:{name}",
                ASM1_EXTENTS,
                tuple(StateFieldSpec(r, "kg") for r in ASM1_REACTIONS),
                mechanism_id=ASM1_MECHANISM,
                batch_reset="initial",
            )
        )
    if index == 0:
        result.extend(
            (
                StateSlotSpec(
                    f"{name}:asm1:utilities",
                    "material",
                    "asm1:finite-utility-stock",
                    ASM1_UTILITIES,
                    tuple(StateFieldSpec(k, "kg") for k in UTILITY_FIELDS),
                    mechanism_id=ASM1_MECHANISM,
                    batch_reset="initial",
                ),
                StateSlotSpec(
                    f"{name}:asm1:work",
                    "equipment",
                    "asm1_treatment",
                    ASM1_WORK,
                    tuple(
                        StateFieldSpec(k, "m^3" if k == "throughput_m3" else "J")
                        for k in WORK_FIELDS
                    ),
                    batch_reset="initial",
                ),
            )
        )
    return tuple(result)


def install_activated_sludge_tank(
    state: Any, spec: ActivatedSludgeSpec, index: int, inventory: ActivatedSludgeInventory
) -> Any:
    water = inventory.water_mol[index]
    volume = water * spec.water_molar_volume_m3 * 1000
    values = {
        ASM1_MATERIAL: inventory.components[index],
        ASM1_CARRIER_H: {
            "enthalpy_J": spec.properties.enthalpy({"H2O": water}, spec.temperature_K)
        },
    }
    if 0 < index <= len(spec.reactor_vessel_ids):
        values[ASM1_EXTENTS] = inventory.extents[index - 1]
    if index == 0:
        values.update({ASM1_UTILITIES: inventory.utilities, ASM1_WORK: inventory.work})
    old = {slot.spec.slot_id: slot for slot in state.state_slots.slots}
    slots = StateSlots(
        tuple(
            StateSlot(
                declaration,
                state.vessel_id,
                values[declaration.state_type],
                initial_values=old[declaration.slot_id].initial_values
                if declaration.slot_id in old
                else None,
            )
            for declaration in slot_declarations(spec, index)
        )
    )
    return state.replace(
        species_amounts={"H2O": water},
        volume_L=volume,
        temperature_K=spec.temperature_K,
        pressure_Pa=spec.pressure_Pa,
        phases=PhaseLedger(
            {
                "liquid": PhaseRecord(
                    "liquid",
                    state.vessel_id,
                    "liquid",
                    volume,
                    {"H2O": water},
                    settled=True,
                    selected=True,
                    metadata={"carrier_representation": "species_inventory"},
                )
            }
        ),
        state_slots=slots,
        process=replace(state.process, last_observation_record=None),
    )


def unpack_activated_sludge(
    state: Any,
) -> tuple[Any, ActivatedSludgeSpec, ActivatedSludgeInventory]:
    record = activated_sludge_record(state)
    if record is None:
        raise ValueError("no complete ASM1 treatment system installed")
    spec = ActivatedSludgeSpec.from_dict(record.settings["physical_model"])
    if (
        state.vessel_id != spec.member_vessel_ids[0]
        or record.settings.get("empirical_reaction_balance") != spec.balance_declaration()
    ):
        raise ValueError("ASM1 original apparatus/balance declaration differs from source model")
    members = {state.vessel_id: state, **state.inactive_vessels}
    water, components, extents = [], [], []
    utility = work = None
    for i, name in enumerate(spec.member_vessel_ids):
        local = members[name]
        declarations = {s.slot_id: s for s in slot_declarations(spec, i)}
        slots = {s.spec.slot_id: s for s in local.state_slots.slots}
        if (
            set(slots) != set(declarations)
            or any(s.spec != declarations[k] for k, s in slots.items())
            or set(local.species_amounts) != {"H2O"}
            or local.samples.samples
            or local.temperature_K != spec.temperature_K
            or local.pressure_Pa != spec.pressure_Pa
            or local.phase != "liquid"
            or set(local.phases.phases) != {"liquid"}
        ):
            raise ValueError(
                "ASM1 original slot ownership, actual carrier or fixed boundary differs"
            )
        phase = local.phases.phases["liquid"]
        volume = local.species_amounts["H2O"] * spec.water_molar_volume_m3 * 1000
        if (
            phase.phase_id != "liquid"
            or phase.vessel_id != name
            or phase.phase_type != "liquid"
            or phase.species_amounts_mol != local.species_amounts
            or phase.solvents.volume_L != 0
            or not phase.settled
            or not phase.selected
            or phase.metadata.get("carrier_representation") != "species_inventory"
            or not isclose(local.volume_L, volume, rel_tol=1e-9, abs_tol=1e-12)
            or not isclose(phase.volume_L, volume, rel_tol=1e-9, abs_tol=1e-12)
        ):
            raise ValueError("ASM1 carrier phase must remain the only water/volume inventory")
        typed = {slot.spec.state_type: slot for slot in slots.values()}
        expected_h = spec.properties.enthalpy(local.species_amounts, spec.temperature_K)
        if not isclose(
            typed[ASM1_CARRIER_H].values["enthalpy_J"], expected_h, rel_tol=1e-9, abs_tol=1e-7
        ):
            raise ValueError("ASM1 carrier sensible enthalpy differs from its original water stock")
        water.append(local.species_amounts["H2O"])
        components.append(dict(typed[ASM1_MATERIAL].values))
        if ASM1_EXTENTS in typed:
            extents.append(dict(typed[ASM1_EXTENTS].values))
        if i == 0:
            utility, work = dict(typed[ASM1_UTILITIES].values), dict(typed[ASM1_WORK].values)
        if local.ledger.time_s != state.ledger.time_s:
            raise ValueError("ASM1 member clocks disagree with the original shared advance")
    assert utility is not None and work is not None
    inventory = ActivatedSludgeInventory(
        tuple(water), tuple(components), tuple(extents), utility, work, state.ledger.time_s
    )
    validate_inventory(spec, inventory)
    return record, spec, inventory


def _totals(inventory: ActivatedSludgeInventory) -> tuple[dict[str, float], dict[str, float]]:
    totals = {k: sum(row[k] for row in inventory.components) for k in ASM1_COMPONENTS}
    totals["S_O"] += (
        inventory.utilities["oxygen_supply_kg"] + inventory.utilities["oxygen_offgas_kg"]
    )
    totals["N2_N"] = inventory.utilities["nitrogen_offgas_kg"]
    return totals, {r: sum(row[r] for row in inventory.extents) for r in ASM1_REACTIONS}


class ActivatedSludgeAdapter:
    supported_operations = ("set_flow_rate", "run_flow", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return activated_sludge_record(state) is not None

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = ActivatedSludgeSpec.from_dict(
            activated_sludge_record(state).settings["physical_model"]
        )
        names = spec.member_vessel_ids
        edges = [
            (names[0], names[1]),
            *((names[i], names[i + 1]) for i in range(1, len(spec.reactor_vessel_ids))),
            (names[-3], names[-2]),
            (names[-3], names[-1]),
        ]
        if len(spec.reactor_vessel_ids) > 1:
            edges.append((names[-3], names[1]))
        return SharedAdvanceSystem(
            state.vessel_id, strategy_id, names, connection_ids(self.composition, edges)
        )

    @property
    def preserved_source_vessels(self) -> set[str]:
        if self.composition is None or self.composition.physics is None:
            return set()
        return {
            name
            for state in self.composition.physics.initial_states.values()
            if self.supports(state)
            for name in ActivatedSludgeSpec.from_dict(
                activated_sludge_record(state).settings["physical_model"]
            ).member_vessel_ids
        }

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def validate_state(self, state: Any) -> None:
        from chemworld.foundation.empirical_inventory import (
            validate_empirical_transition,
        )

        record, spec, inventory = unpack_activated_sludge(state)
        if self.composition is None or self.composition.physics is None:
            raise ValueError("ASM1 requires its source composition")
        originals = self.composition.physics.initial_states
        original = originals[state.vessel_id]
        source = activated_sludge_record(original)
        if (
            source is None
            or source.settings["physical_model"] != record.settings["physical_model"]
            or source.settings.get("empirical_reaction_balance") != spec.balance_declaration()
        ):
            raise ValueError("ASM1 source physical model or reaction basis has been changed")
        original = original.replace(
            inactive_vessels={k: v for k, v in originals.items() if k != state.vessel_id}
        )
        members = {state.vessel_id: state, **state.inactive_vessels}
        for name in spec.member_vessel_ids:
            initial_slots = {s.spec.slot_id: s for s in originals[name].state_slots.slots}
            for slot in members[name].state_slots.slots:
                if slot.initial_values != initial_slots[slot.spec.slot_id].initial_values:
                    raise ValueError("ASM1 original initial receipt cannot be rewritten")
        _, _, initial = unpack_activated_sludge(original)
        before, old_extents = _totals(initial)
        after, new_extents = _totals(inventory)
        validate_empirical_transition(
            before, after, old_extents, new_extents, spec.balance_declaration()
        )
        if not isclose(
            sum(initial.water_mol), sum(inventory.water_mol), rel_tol=1e-10, abs_tol=1e-9
        ):
            raise ValueError("ASM1 original finite carrier water has been created or lost")
        throughput = (initial.water_mol[0] - inventory.water_mol[0]) * spec.water_molar_volume_m3
        if not isclose(inventory.work["throughput_m3"], throughput, rel_tol=1e-8, abs_tol=1e-14):
            raise ValueError("ASM1 throughput receipt differs from its actual finite feed debit")
        if spec.electricity_mode == "calculated":
            electricity = inventory.time_s * sum(
                spec.oxygen_equilibrium_kg_m3 * kla * volume / 1.8 * 1000
                for kla, volume in zip(spec.kla_per_hour, spec.reactor_volumes_m3, strict=True)
            )
        elif spec.electricity_mode == "fixed":
            train_ratio = 1 / (
                (1 - spec.internal_recycle_fraction)
                * (
                    1
                    - (1 - spec.clarifier_soluble_effluent_fraction) * spec.sludge_recycle_fraction
                )
            )
            electricity = (
                len(spec.reactor_vessel_ids)
                * spec.fixed_electricity_kWh_m3
                * (inventory.work["throughput_m3"] * train_ratio)
                * 3.6e6
            )
        else:
            electricity = 0.0
        if not isclose(
            inventory.work["aeration_electricity_J"], electricity, rel_tol=1e-8, abs_tol=1e-10
        ):
            raise ValueError("ASM1 electricity receipt differs from actual source time/throughput")

    def validate_world(self) -> None:
        physics = self.composition.physics
        components = self.composition.components
        for name, state in physics.initial_states.items():
            if not self.supports(state):
                continue
            state = state.replace(
                inactive_vessels={k: v for k, v in physics.initial_states.items() if k != name}
            )
            self.validate_state(state)
            _, spec, _ = unpack_activated_sludge(state)
            if physics.network.to_dict() != spec.network.to_dict():
                raise ValueError("ASM1 chemical network must contain only its real carrier water")
            if not any(c.vessel == name and c.kind == "finite_transport" for c in components):
                raise ValueError("ASM1 requires the original finite_transport component")
            if any(
                c.vessel in spec.member_vessel_ids
                and c.kind not in {"finite_transport", "material_input", "lab_schedule"}
                for c in components
            ):
                raise ValueError(
                    "ASM1 empirical quality cannot silently enable molecular chemistry"
                )
            if any(
                v in bindings
                for bindings in physics.models.values()
                for v in spec.member_vessel_ids
            ):
                raise ValueError("ASM1 cannot share its material owner with another physical model")
            for vessel, capacity in zip(spec.member_vessel_ids, spec.capacities_m3, strict=True):
                if (
                    capacity * 1000
                    > physics.initial_states[vessel].vessels.vessels[vessel].max_volume_L
                ):
                    raise ValueError("ASM1 capacity exceeds its original vessel equipment")
            self.shared_system(state, "finite-pump-transport")

    def reconcile_initialization(self, transition: Any) -> Any:
        affected = set(transition.receipt.get("vessels", {}))
        if affected & self.protected_lifecycle_vessels:
            raise ValueError(
                "ASM1 partial material reset cannot erase cumulative reaction receipts"
            )
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

    def configure_flow(self, state: Any, action: dict[str, Any]) -> Any:
        from chemworld.runtime.transport_devices import configure_liquid_flow

        self.validate_state(state)
        return configure_liquid_flow(state, action, activated_sludge_record(state))

    def advance(
        self, state: Any, action: dict[str, Any], *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        if (
            heat
            or jacket_program is not None
            or action.get("operation") not in {"wait", "run_flow"}
            or state.quenched
            or state.terminated
        ):
            return ModelProviderResult(
                {},
                {},
                success=False,
                outcome="rejected",
                failure_reason="asm1_declared_fixed_boundary_only",
            )
        try:
            self.validate_state(state)
            record, spec, inventory = unpack_activated_sludge(state)
            if action.get("target_temperature_K", spec.temperature_K) != spec.temperature_K:
                raise ValueError("ASM1 source model is a declared fixed-temperature workflow")
            flow = record.settings.get("flow_control", {}).get("requested_flow_m3_s", 0.0)
            result = advance_activated_sludge(spec, inventory, flow, float(action["duration_s"]))
            dt, diagnostic = result.actual_duration_s, result.diagnostic
            inactive, ages = dict(state.inactive_vessels), dict(state.vessel_elapsed_s)
            for i, name in enumerate(spec.member_vessel_ids[1:], 1):
                local = install_activated_sludge_tank(inactive[name], spec, i, result.final)
                inactive[name] = local.replace(ledger=advance_ledger(local.ledger, dt))
                ages[name] = ages.get(name, 0.0) + dt
            candidate = install_activated_sludge_tank(state, spec, 0, result.final)
            electrical = diagnostic["aeration_electricity_J"]
            pump = diagnostic["pump_shaft_work_J"]
            loss = diagnostic["pump_motor_loss_J"] - diagnostic["carrier_bath_heat_J"]
            candidate = candidate.replace(
                inactive_vessels=inactive,
                vessel_elapsed_s=ages,
                equipment=upsert_equipment_record(
                    candidate.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"last_activated_sludge": diagnostic},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    cost_delta=(electrical + pump) / 250000,
                    energy_deltas={
                        "flow_electrical_work_J": electrical,
                        "flow_pump_work_J": pump,
                        "heat_loss_J": loss,
                    },
                ),
            )
            self.validate_state(candidate)
        except (ValueError, ArithmeticError, KeyError, RuntimeError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="asm1_advance_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            vessel_thermal_deltas={
                name: ({"heat_loss_J": loss} if i == 0 else {})
                for i, name in enumerate(spec.member_vessel_ids)
            },
            diagnostics={
                "activated_sludge": diagnostic,
                "stopping_event": result.stopping_event,
                "quality": inventory_quality(spec, result.final),
            },
        )
