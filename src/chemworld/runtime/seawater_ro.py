"""Empirical TDS kg on original owned slots, water moles and finite membrane clock."""

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
from chemworld.physchem.seawater_ro import SeawaterROInventory, SeawaterROSpec, advance_seawater_ro
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.transport_devices import configure_liquid_flow, enthalpy_slots

SEAWATER_RO_TYPE = "declared_seawater_ro_mass"
WATER_KG_MOL = 0.018015  # original elemental composition, not source package's rounded MW


def seawater_ro_record(state: Any) -> Any:
    return attached_device_record(state, SEAWATER_RO_TYPE)


def seawater_tds_slot(state: Any, mass: float) -> StateSlot:
    return StateSlot(
        StateSlotSpec(
            f"{state.vessel_id}:material:seawater-tds",
            "material",
            state.vessel_id,
            "seawater_tds_mass_v1",
            (StateFieldSpec("tds_mass_kg", "kg", conserved_quantity="seawater:tds-mass"),),
            phase_id="liquid",
            batch_reset="initial",
        ),
        state.vessel_id,
        {"tds_mass_kg": mass},
    )


def install_seawater_liquid(
    state: Any, spec: SeawaterROSpec, masses: dict[str, float], enthalpy: float
) -> Any:
    spec.properties.validate_amounts(masses, empty=True)
    volume = spec.properties.volume(masses, spec.temperature_K, spec.reservoir_pressure_Pa) * 1000
    expected = spec.properties.enthalpy(masses, spec.temperature_K, spec.reservoir_pressure_Pa)
    if not isclose(enthalpy, expected, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError("seawater H must match its declared empirical composition")
    n = {"H2O": masses["H2O"] / WATER_KG_MOL}
    return state.replace(
        species_amounts=n,
        volume_L=volume,
        temperature_K=spec.temperature_K,
        pressure_Pa=spec.reservoir_pressure_Pa,
        phases=PhaseLedger(
            {
                "liquid": PhaseRecord(
                    "liquid",
                    state.vessel_id,
                    "liquid",
                    volume,
                    n,
                    settled=True,
                    selected=True,
                    metadata={
                        "carrier_representation": "species_inventory",
                        "material_boundary": (
                            "H2O molecular; TDS conserved kg, elemental/charge identity unresolved"
                        ),
                    },
                )
            }
        ),
        state_slots=StateSlots(
            (*enthalpy_slots(state, enthalpy, 0.0).slots, seawater_tds_slot(state, masses["TDS"]))
        ),
        process=replace(state.process, last_observation_record=None),
    )


def seawater_vessels(state: Any, spec: SeawaterROSpec) -> tuple[str, str, str]:
    names = state.vessel_id, spec.permeate_vessel_id, spec.retentate_vessel_id
    if len(set(names)) != 3:
        raise ValueError("seawater membrane needs three distinct finite vessels")
    return names


def unpack_seawater_ro(state: Any) -> tuple[Any, SeawaterROSpec, SeawaterROInventory]:
    record = seawater_ro_record(state)
    if record is None:
        raise ValueError("no declared seawater RO equipment")
    spec = SeawaterROSpec.from_dict(record.settings["physical_model"])
    masses, enthalpies = [], []
    for i, name in enumerate(seawater_vessels(state, spec)):
        tank = state if i == 0 else state.inactive_vessels[name]
        slots = {s.spec.slot_id: s for s in tank.state_slots.slots}
        template = seawater_tds_slot(tank, 0.0)
        owned = slots.get(template.spec.slot_id)
        if owned is None or owned.spec != template.spec or owned.location_id != name:
            raise ValueError("seawater TDS needs its actual conserved material kg owner")
        if set(tank.species_amounts) != {"H2O"} or set(tank.phases.phases) != {"liquid"}:
            raise ValueError("empirical seawater cannot invent TDS molecular species")
        n = {"H2O": tank.species_amounts["H2O"] * WATER_KG_MOL, "TDS": owned.values["tds_mass_kg"]}
        spec.properties.validate_amounts(n, empty=i > 0)
        h = spec.properties.enthalpy(n, spec.temperature_K, spec.reservoir_pressure_Pa)
        expected = install_seawater_liquid(tank, spec, n, h)
        phase = tank.phases.phases["liquid"]
        if (
            set(slots) != {s.spec.slot_id for s in expected.state_slots.slots}
            or phase.species_amounts_mol != tank.species_amounts
            or phase.metadata != expected.phases.phases["liquid"].metadata
            or not isclose(phase.volume_L, expected.volume_L, rel_tol=1e-9, abs_tol=1e-10)
            or not isclose(tank.temperature_K, spec.temperature_K, abs_tol=1e-8)
            or not isclose(tank.pressure_Pa, spec.reservoir_pressure_Pa, abs_tol=1e-5)
            or not isclose(tank.volume_L, expected.volume_L, rel_tol=1e-9, abs_tol=1e-10)
        ):
            raise ValueError("seawater state differs from its declared owned liquid")
        for target in expected.state_slots.slots:
            actual = slots[target.spec.slot_id]
            if actual.spec != target.spec or any(
                not isclose(actual.values[k], v, rel_tol=1e-9, abs_tol=1e-7)
                for k, v in target.values.items()
            ):
                raise ValueError("seawater owned H/TDS differs from physical inventory")
        masses.append(n)
        enthalpies.append(h)
    return record, spec, SeawaterROInventory(tuple(masses), tuple(enthalpies), state.ledger.time_s)


class SeawaterRORuntime:
    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return seawater_ro_record(state) is not None

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = SeawaterROSpec.from_dict(seawater_ro_record(state).settings["physical_model"])
        names = seawater_vessels(state, spec)
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            names,
            connection_ids(self.composition, [(names[0], n) for n in names[1:]]),
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
                    spec = SeawaterROSpec.from_dict(
                        seawater_ro_record(state).settings["physical_model"]
                    )
                    result.update(seawater_vessels(state, spec))
        return result

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        components = {c.id: c for c in self.composition.components}
        occupied: set[str] = set()
        for vessel, initial in states.items():
            if not self.supports(initial):
                continue
            state = initial.replace(
                inactive_vessels={v: s for v, s in states.items() if v != vessel}
            )
            _, spec, _ = unpack_seawater_ro(state)
            network = self.composition.physics.network
            if (
                network.reactions
                or len(network.species) != 1
                or network.species[0].species_id != "H2O"
                or network.species[0].composition != {"H": 2, "O": 1}
                or network.species[0].charge
            ):
                raise ValueError("empirical seawater world has real water and inert TDS mass only")
            for name, capacity in (
                (spec.permeate_vessel_id, spec.permeate_capacity_L),
                (spec.retentate_vessel_id, spec.retentate_capacity_L),
            ):
                if name in occupied or capacity > states[name].vessels.vessels[name].max_volume_L:
                    raise ValueError("seawater outlet ownership or capacity conflict")
                occupied.add(name)
                if not any(
                    components[c["source"]["component"]].vessel == vessel
                    and components[c["target"]["component"]].vessel == name
                    for c in self.composition.connections
                ):
                    raise ValueError("seawater RO needs actual finite outlet ports")

    def configure_flow(self, state: Any, action: dict[str, Any]) -> Any:
        record, _, _ = unpack_seawater_ro(state)
        return configure_liquid_flow(state, action, record)

    def advance(
        self, state: Any, action: dict[str, Any], *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        record, spec, inventory = unpack_seawater_ro(state)
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
                failure_reason="seawater_ro_requires_declared_isotherm",
            )
        try:
            flow = record.settings.get("flow_control", {}).get("requested_flow_m3_s", 0.0)
            final, diagnostic = advance_seawater_ro(
                spec, inventory, flow, float(action["duration_s"])
            )
            dt = diagnostic["actual_duration_s"]
            inactive, ages = dict(state.inactive_vessels), dict(state.vessel_elapsed_s)
            for i, name in enumerate(seawater_vessels(state, spec)[1:], start=1):
                tank = install_seawater_liquid(
                    inactive[name], spec, final.masses_kg[i], final.enthalpies_J[i]
                )
                inactive[name] = tank.replace(ledger=advance_ledger(tank.ledger, dt))
                ages[name] = ages.get(name, 0.0) + dt
            candidate = install_seawater_liquid(
                state, spec, final.masses_kg[0], final.enthalpies_J[0]
            )
            bath = diagnostic["bath_heat_into_liquid_J"]
            candidate = candidate.replace(
                inactive_vessels=inactive,
                vessel_elapsed_s=ages,
                equipment=upsert_equipment_record(
                    state.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"last_seawater_ro": diagnostic},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    cost_delta=diagnostic["shaft_work_J"] / 250000,
                    energy_deltas={
                        "flow_pump_work_J": diagnostic["shaft_work_J"],
                        "energy_jacket_J": max(bath, 0.0),
                        "heat_loss_J": diagnostic["motor_loss_J"] + max(-bath, 0.0),
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="seawater_ro_solver_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if diagnostic["stopping_event"] else "success",
            diagnostics={"seawater_ro": diagnostic, "stopping_event": diagnostic["stopping_event"]},
        )
