"""Original finite-transport adapter with a persistent original resin owner."""

from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.cation_exchange import (
    CationExchangeInventory,
    CationExchangeSpec,
    advance_cation_exchange,
    cation_exchange_inventory,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

CATION_EXCHANGE_TYPE = "finite_cation_exchange"
MEDIA_TYPE = "finite_cation_exchange_media"


def cation_exchange_record(state: Any) -> Any:
    return attached_device_record(state, CATION_EXCHANGE_TYPE)


def cation_exchange_vessels(state: Any, spec: CationExchangeSpec) -> tuple[str, ...]:
    names = (
        state.vessel_id,
        spec.contact_vessel_id,
        spec.product_vessel_id,
        spec.regenerant_vessel_id,
        spec.rinse_vessel_id,
        spec.waste_vessel_id,
    )
    if len(set(names)) != 6:
        raise ValueError("cation exchange requires six distinct original tanks")
    return names


def install_exchange_liquid(
    state: Any, spec: CationExchangeSpec, amounts: Any, enthalpy: float, slot: Any = None
) -> Any:
    from chemworld.runtime.transport_devices import enthalpy_slots

    volume = 1000 * spec.volume_m3(amounts)
    slots = enthalpy_slots(state, enthalpy, 0.0)
    if slot is not None:
        spec.validate_slot(slot)
        if state.vessel_id != spec.contact_vessel_id:
            raise ValueError("the resin lot belongs only to the original contact vessel")
        slots = StateSlots((*slots.slots, slot))
    return state.replace(
        species_amounts=dict(amounts),
        volume_L=volume,
        temperature_K=spec.temperature_K,
        pressure_Pa=spec.pressure_Pa,
        state_slots=slots,
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
        process=replace(state.process, last_observation_record=None),
    )


def unpack_cation_exchange(
    state: Any,
) -> tuple[Any, CationExchangeSpec, CationExchangeInventory]:
    from chemworld.runtime.transport_devices import enthalpy_slots

    record = cation_exchange_record(state)
    if record is None:
        raise ValueError("no finite cation exchange device is installed")
    spec = CationExchangeSpec.from_dict(record.settings["physical_model"])
    amounts, enthalpies = [], []
    media_slot = None
    for i, name in enumerate(cation_exchange_vessels(state, spec)):
        tank = state if i == 0 else state.inactive_vessels[name]
        n = dict(tank.species_amounts)
        h = spec.enthalpy(n)
        volume = spec.volume_m3(n) * 1000
        phase = tank.phases.phases.get("liquid")
        if (
            not isclose(tank.temperature_K, spec.temperature_K, rel_tol=0, abs_tol=1e-9)
            or not isclose(tank.pressure_Pa, spec.pressure_Pa, rel_tol=0, abs_tol=1e-5)
            or not isclose(tank.volume_L, volume, rel_tol=1e-9, abs_tol=1e-10)
            or set(tank.phases.phases) != {"liquid"}
            or phase is None
            or phase.phase_id != "liquid"
            or phase.phase_type != "liquid"
            or phase.vessel_id != name
            or not phase.settled
            or not phase.selected
            or phase.solvents.volume_L != 0
            or dict(phase.species_amounts_mol) != n
            or phase.metadata.get("carrier_representation") != "species_inventory"
            or not isclose(phase.volume_L, volume, rel_tol=1e-9, abs_tol=1e-10)
        ):
            raise ValueError("original cation exchange liquid n/H/T/P/V/phase disagree")
        expected = {s.spec.slot_id: s for s in enthalpy_slots(tank, h, 0.0).slots}
        owned = {s.spec.slot_id: s for s in tank.state_slots.slots}
        if i == 1:
            media_slot = owned.pop(spec.initial_slot().spec.slot_id, None)
            if media_slot is None:
                raise ValueError("exchange contact requires its original persistent resin lot")
            spec.validate_slot(media_slot)
            media = tank.equipment.equipment.get(spec.media_equipment_id)
            if (
                media is None
                or media.equipment_type != MEDIA_TYPE
                or media.attached_vessel_id != name
                or media.batch_reset != "preserve"
                or media.settings != {"resin_model": "gaines-thomas-Na-Ca-v1"}
            ):
                raise ValueError("aqueous resin lot differs from its original equipment owner")
        if set(owned) != set(expected) or any(
            owned[k].spec != v.spec
            or owned[k].location_id != name
            or not isclose(
                owned[k].values["enthalpy_J"], v.values["enthalpy_J"], rel_tol=1e-9, abs_tol=1e-7
            )
            for k, v in expected.items()
        ):
            raise ValueError("original aqueous H/slot ownership differs from same-source calorics")
        amounts.append(n)
        enthalpies.append(owned[f"{name}:material:enthalpy"].values["enthalpy_J"])
    inventory = cation_exchange_inventory(
        spec, amounts, resin_slot=media_slot, time_s=state.ledger.time_s
    )
    return record, spec, replace(inventory, enthalpies_J=tuple(enthalpies))


class CationExchangeAdapter:
    supported_operations = ("set_flow_rate", "run_flow", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return cation_exchange_record(state) is not None

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = CationExchangeSpec.from_dict(
            cation_exchange_record(state).settings["physical_model"]
        )
        names = cation_exchange_vessels(state, spec)
        pairs = [
            (names[0], names[1]),
            (names[1], names[2]),
            (names[3], names[1]),
            (names[4], names[1]),
            (names[1], names[5]),
        ]
        return SharedAdvanceSystem(
            state.vessel_id, strategy_id, names, connection_ids(self.composition, pairs)
        )

    @property
    def preserved_source_vessels(self) -> set[str]:
        names: set[str] = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    spec = CationExchangeSpec.from_dict(
                        cation_exchange_record(state).settings["physical_model"]
                    )
                    names.update(cation_exchange_vessels(state, spec))
        return names

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def validate_state(self, state: Any) -> None:
        record, spec, inventory = unpack_cation_exchange(state)
        if self.composition is None or self.composition.physics is None:
            raise ValueError("cation exchange requires its original composition")
        states = self.composition.physics.initial_states
        original = states.get(state.vessel_id)
        source = cation_exchange_record(original) if original is not None else None
        if source is None or source.settings["physical_model"] != record.settings["physical_model"]:
            raise ValueError("cation exchange physical_model differs from original composition")
        original_slot = next(
            (
                s
                for s in states[spec.contact_vessel_id].state_slots.slots
                if s.spec.slot_id == inventory.resin_slot.spec.slot_id
            ),
            None,
        )
        if (
            original_slot is None
            or original_slot.initial_values != inventory.resin_slot.initial_values
        ):
            raise ValueError("original resin preloading provenance cannot be replaced")

    def reconcile_initialization(self, transition: Any) -> Any:
        active = transition.state
        states = {active.vessel_id: active.replace(inactive_vessels={}), **active.inactive_vessels}
        for name, state in states.items():
            if self.supports(state):
                self.validate_state(
                    state.replace(inactive_vessels={k: v for k, v in states.items() if k != name})
                )
        return transition

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        components = {c.id: c for c in self.composition.components}
        for name, original in states.items():
            if not self.supports(original):
                continue
            if not any(
                c.kind == "finite_transport" and c.vessel == name for c in components.values()
            ):
                raise ValueError("cation exchange requires its original finite_transport component")
            state = original.replace(
                inactive_vessels={k: v for k, v in states.items() if k != name}
            )
            self.validate_state(state)
            _, spec, _ = unpack_cation_exchange(state)
            names = cation_exchange_vessels(state, spec)
            network = self.composition.physics.network
            expected = {
                s.species_id: (dict(s.composition), s.charge, "liquid")
                for s in spec.database.species
            }
            if (
                network.reactions
                or {s.species_id: (s.composition, s.charge, s.phase) for s in network.species}
                != expected
            ):
                raise ValueError(
                    "cation exchange needs matching original charged species and no reactions"
                )
            if any(c.kind == "sorbent_contact" and c.vessel in names for c in components.values()):
                raise ValueError("one resin lot cannot have a second independent process driver")
            if any(
                c > states[n].vessels.vessels[n].max_volume_L
                for n, c in zip(names, spec.capacities_L, strict=True)
            ):
                raise ValueError("cation exchange exceeds installed finite tank capacity")
            pairs = {
                (
                    components[e["source"]["component"]].vessel,
                    components[e["target"]["component"]].vessel,
                )
                for e in self.composition.connections
            }
            if (
                not {
                    (names[0], names[1]),
                    (names[1], names[2]),
                    (names[3], names[1]),
                    (names[4], names[1]),
                    (names[1], names[5]),
                }
                <= pairs
            ):
                raise ValueError("cation exchange requires all three actual finite material routes")

    def configure_flow(self, state: Any, action: dict[str, Any]) -> Any:
        from chemworld.runtime.transport_devices import configure_liquid_flow

        self.validate_state(state)
        return configure_liquid_flow(
            state,
            action,
            cation_exchange_record(state),
            supported_paths=("process", "regenerate", "rinse"),
        )

    def advance(
        self, state: Any, action: dict[str, Any], *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        try:
            self.validate_state(state)
            record, spec, initial = unpack_cation_exchange(state)
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
                    failure_reason="cation_exchange_requires_declared_isotherm",
                )
            control = record.settings.get("flow_control", {})
            result, diagnostic = advance_cation_exchange(
                spec,
                initial,
                control.get("requested_flow_m3_s", 0.0),
                float(action["duration_s"]),
                flow_path=control.get("flow_path", "process"),
            )
            final, dt = result.final_state, result.actual_duration_s
            names = cation_exchange_vessels(state, spec)
            inactive, ages = dict(state.inactive_vessels), dict(state.vessel_elapsed_s)
            for i, name in enumerate(names[1:], 1):
                tank = install_exchange_liquid(
                    inactive[name],
                    spec,
                    final.amounts_mol[i],
                    final.enthalpies_J[i],
                    final.resin_slot if i == 1 else None,
                )
                inactive[name] = tank.replace(ledger=advance_ledger(tank.ledger, dt))
                ages[name] = ages.get(name, 0.0) + dt
            candidate = install_exchange_liquid(
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
                    settings={"last_cation_exchange": diagnostic},
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
                failure_reason="cation_exchange_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            vessel_thermal_deltas={
                name: (
                    {
                        "energy_jacket_J": diagnostic["bath_heating_J"],
                        "heat_loss_J": diagnostic["motor_loss_J"] + diagnostic["bath_cooling_J"],
                    }
                    if i == 0
                    else {}
                )
                for i, name in enumerate(names)
            },
            diagnostics={
                "cation_exchange": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "conservation_residuals": dict(result.conservation_residuals),
                "energy_balance_residual_J": result.energy_balance_residual_J,
            },
        )
