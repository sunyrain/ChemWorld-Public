"""Original finite-transport adapter for a finite measured-jar suspension treatment."""

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
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.suspension_treatment import (
    advance_suspension_treatment,
    suspension_enthalpy,
    suspension_volume,
    validate_amounts,
)
from chemworld.physchem.suspension_treatment_specs import SuspensionTreatmentSpec
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

SUSPENSION_TREATMENT_TYPE = "finite_suspension_treatment"


def suspension_treatment_record(state: Any) -> Any:
    return attached_device_record(state, SUSPENSION_TREATMENT_TYPE)


def suspension_treatment_vessels(state: Any, spec: SuspensionTreatmentSpec) -> tuple[str, ...]:
    names = (state.vessel_id, spec.product_vessel_id, spec.sludge_vessel_id)
    if len(set(names)) != 3:
        raise ValueError("suspension treatment requires three distinct original liquid tanks")
    return names


def suspension_enthalpy_slots(state: Any, enthalpy: float) -> StateSlots:
    return StateSlots(
        (
            StateSlot(
                StateSlotSpec(
                    f"{state.vessel_id}:material:enthalpy",
                    "material",
                    state.vessel_id,
                    "finite_suspension_enthalpy",
                    (StateFieldSpec("enthalpy_J", "J", minimum=None),),
                    phase_id=None,
                    batch_reset="initial",
                ),
                state.vessel_id,
                {"enthalpy_J": enthalpy},
            ),
        )
    )


def install_suspension_treatment_material(
    state: Any, spec: SuspensionTreatmentSpec, amounts: Any, enthalpy: float
) -> Any:
    particle = spec.material.particle.species_id
    liquid = {k: amounts["liquid:" + k] for k in spec.material.liquid.component_ids}
    phases = {}
    for name, n, kind in (
        ("liquid", liquid, "liquid"),
        ("dispersed", {particle: amounts["dispersed"]}, "solid"),
        ("floc", {particle: amounts["floc"]}, "solid"),
    ):
        volume = (
            sum(n[k] * spec.molar_volumes["liquid:" + k] for k in n)
            if name == "liquid"
            else amounts[name] * spec.material.particle.molar_volume_m3_mol
        ) * 1000
        phases[name] = PhaseRecord(
            name,
            state.vessel_id,
            kind,
            volume,
            n,
            settled=True,
            selected=(name == "liquid"),
            metadata={"carrier_representation": "species_inventory", "particle_group": name},
        )
    return state.replace(
        species_amounts={**liquid, particle: amounts["dispersed"] + amounts["floc"]},
        temperature_K=spec.temperature_K,
        pressure_Pa=spec.pressure_Pa,
        volume_L=1000 * suspension_volume(spec, amounts),
        phases=PhaseLedger(phases),
        state_slots=suspension_enthalpy_slots(state, enthalpy),
        process=replace(state.process, last_observation_record=None),
    )


def unpack_suspension_treatment(
    state: Any,
) -> tuple[Any, SuspensionTreatmentSpec, TransportInventory]:
    record = suspension_treatment_record(state)
    if record is None:
        raise ValueError("no suspension treatment device is installed")
    spec = SuspensionTreatmentSpec.from_dict(record.settings["physical_model"])
    amounts, enthalpies = [], []
    particle = spec.material.particle.species_id
    for name in suspension_treatment_vessels(state, spec):
        tank = state if name == state.vessel_id else state.inactive_vessels[name]
        if set(tank.phases.phases) != {"liquid", "dispersed", "floc"}:
            raise ValueError("suspension requires original liquid/dispersed/floc phases")
        n = {
            **{
                "liquid:" + k: v
                for k, v in tank.phases.phases["liquid"].species_amounts_mol.items()
            },
            "dispersed": tank.phases.phases["dispersed"].species_amounts_mol[particle],
            "floc": tank.phases.phases["floc"].species_amounts_mol[particle],
        }
        expected_h = suspension_enthalpy(spec, n)
        expected = install_suspension_treatment_material(tank, spec, n, expected_h)
        if (
            dict(tank.species_amounts) != dict(expected.species_amounts)
            or not isclose(tank.temperature_K, spec.temperature_K, rel_tol=0, abs_tol=1e-9)
            or not isclose(tank.pressure_Pa, spec.pressure_Pa, rel_tol=0, abs_tol=1e-5)
            or not isclose(tank.volume_L, expected.volume_L, rel_tol=1e-9, abs_tol=1e-10)
        ):
            raise ValueError("suspension original n/T/P/V disagrees with actual phase inventories")
        for key, phase in tank.phases.phases.items():
            wanted = expected.phases.phases[key]
            if (
                phase.phase_id != wanted.phase_id
                or phase.vessel_id != name
                or phase.phase_type != wanted.phase_type
                or not phase.settled
                or phase.selected != wanted.selected
                or phase.solvents.volume_L != 0
                or dict(phase.species_amounts_mol) != dict(wanted.species_amounts_mol)
                or dict(phase.metadata) != dict(wanted.metadata)
                or not isclose(phase.volume_L, wanted.volume_L, rel_tol=1e-9, abs_tol=1e-10)
            ):
                raise ValueError("suspension phase identities/amounts/volumes disagree")
        wanted = expected.state_slots.slots[0]
        if len(tank.state_slots.slots) != 1:
            raise ValueError("suspension requires one total-material H owner per tank")
        slot = tank.state_slots.slots[0]
        if (
            slot.spec != wanted.spec
            or slot.location_id != name
            or set(slot.values) != {"enthalpy_J"}
            or not isclose(slot.values["enthalpy_J"], expected_h, rel_tol=1e-9, abs_tol=1e-7)
        ):
            raise ValueError("suspension original total-material H ownership/properties disagree")
        amounts.append(n)
        enthalpies.append(slot.values["enthalpy_J"])
    validate_amounts(spec, amounts)
    return record, spec, TransportInventory(tuple(amounts), tuple(enthalpies), state.ledger.time_s)


class SuspensionTreatmentAdapter:
    supported_operations = ("set_flow_rate", "run_flow", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return suspension_treatment_record(state) is not None

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = SuspensionTreatmentSpec.from_dict(
            suspension_treatment_record(state).settings["physical_model"]
        )
        names = suspension_treatment_vessels(state, spec)
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            names,
            connection_ids(self.composition, [(names[0], names[1]), (names[0], names[2])]),
        )

    @property
    def preserved_source_vessels(self) -> set[str]:
        names: set[str] = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    spec = SuspensionTreatmentSpec.from_dict(
                        suspension_treatment_record(state).settings["physical_model"]
                    )
                    names.update(suspension_treatment_vessels(state, spec))
        return names

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def validate_state(self, state: Any) -> None:
        record, _, _ = unpack_suspension_treatment(state)
        if self.composition is None or self.composition.physics is None:
            raise ValueError("suspension treatment requires its original composition")
        original = self.composition.physics.initial_states.get(state.vessel_id)
        source = suspension_treatment_record(original) if original is not None else None
        if source is None or source.settings["physical_model"] != record.settings["physical_model"]:
            raise ValueError(
                "suspension treatment physical_model differs from original composition"
            )

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
                raise ValueError(
                    "suspension treatment needs its original finite_transport component"
                )
            state = state.replace(inactive_vessels={k: v for k, v in states.items() if k != name})
            self.validate_state(state)
            _, spec, _ = unpack_suspension_treatment(state)
            names = suspension_treatment_vessels(state, spec)
            network = self.composition.physics.network
            expected = {
                p.component.identifier: (dict(p.component.composition), 0, "liquid")
                for p in spec.material.liquid.packages
            }
            particle = ComponentSpec(
                spec.material.particle.species_id, spec.material.particle.formula
            )
            expected[particle.identifier] = (dict(particle.composition), 0, "solid")
            if (
                network.reactions
                or {s.species_id: (dict(s.composition), s.charge, s.phase) for s in network.species}
                != expected
            ):
                raise ValueError(
                    "suspension needs matching original phase identities and no reactions"
                )
            if occupied.intersection(names):
                raise ValueError("suspension treatment cannot share physical reservoirs")
            occupied.update(names)
            for vessel, capacity in zip(names, spec.capacities_L, strict=True):
                if capacity > states[vessel].vessels.vessels[vessel].max_volume_L:
                    raise ValueError("suspension treatment exceeds original finite vessel capacity")
            edges = {
                (
                    components[e["source"]["component"]].vessel,
                    components[e["target"]["component"]].vessel,
                )
                for e in self.composition.connections
            }
            if not {(names[0], names[1]), (names[0], names[2])} <= edges:
                raise ValueError(
                    "suspension treatment requires its original two material connections"
                )

    def configure_flow(self, state: Any, action: dict[str, Any]) -> Any:
        from chemworld.runtime.transport_devices import configure_liquid_flow

        self.validate_state(state)
        return configure_liquid_flow(state, action, suspension_treatment_record(state))

    def advance(
        self, state: Any, action: dict[str, Any], *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        try:
            self.validate_state(state)
            record, spec, initial = unpack_suspension_treatment(state)
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
                    failure_reason="suspension_treatment_requires_declared_isotherm",
                )
            flow = record.settings.get("flow_control", {}).get("requested_flow_m3_s", 0.0)
            result, diagnostic = advance_suspension_treatment(
                spec, initial, flow, float(action["duration_s"])
            )
            final, dt = result.final_state, result.actual_duration_s
            inactive, ages = dict(state.inactive_vessels), dict(state.vessel_elapsed_s)
            names = suspension_treatment_vessels(state, spec)
            for i, name in enumerate(names[1:], 1):
                tank = install_suspension_treatment_material(
                    inactive[name], spec, final.amounts_mol[i], final.enthalpies_J[i]
                )
                inactive[name] = tank.replace(ledger=advance_ledger(tank.ledger, dt))
                ages[name] = ages.get(name, 0) + dt
            candidate = install_suspension_treatment_material(
                state, spec, final.amounts_mol[0], final.enthalpies_J[0]
            )
            energy = {
                "flow_pump_work_J": diagnostic["pump_work_J"],
                "flow_shaft_work_J": diagnostic["mixing_work_J"],
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
                    settings={"last_suspension_treatment": diagnostic},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    cost_delta=(diagnostic["pump_work_J"] + diagnostic["mixing_work_J"]) / 250000,
                    energy_deltas=energy,
                ),
            )
        except (ArithmeticError, KeyError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="suspension_treatment_failed",
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
                "suspension_treatment": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "conservation_residuals": dict(result.conservation_residuals),
                "energy_balance_residual_J": result.energy_balance_residual_J,
            },
        )
