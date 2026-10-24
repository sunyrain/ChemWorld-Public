"""Distributed constitutive adapter for the existing membrane/clock transaction.

No persistent simulation state is owned here: original species, phase, H slots,
equipment settings, finite vessels and resource ledger remain authoritative.
"""

from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.distributed_ro import DistributedROSpec, advance_distributed_ro
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.transport_devices import configure_liquid_flow, enthalpy_slots

DISTRIBUTED_RO_TYPE = "declared_distributed_ro"


def distributed_ro_record(state: Any) -> Any:
    return attached_device_record(state, DISTRIBUTED_RO_TYPE)


def distributed_ro_vessels(state: Any, spec: DistributedROSpec) -> tuple[str, str, str]:
    names = (state.vessel_id, spec.permeate_vessel_id, spec.retentate_vessel_id)
    if len(set(names)) != 3:
        raise ValueError("distributed membrane requires three distinct finite vessels")
    return names


def install_ro_liquid(
    state: Any, spec: DistributedROSpec, amounts: dict[str, float], enthalpy: float
) -> Any:
    p, t = spec.reservoir_pressure_Pa, spec.temperature_K
    volume = 1000 * spec.properties.volume(amounts, t, p)
    fluid_h = spec.properties.enthalpy(amounts, t, p)
    if not isclose(fluid_h, enthalpy, rel_tol=1e-7, abs_tol=1e-5):
        raise ValueError("distributed membrane liquid H must match its declared isotherm")
    return state.replace(
        species_amounts=dict(amounts),
        volume_L=volume,
        temperature_K=t,
        pressure_Pa=p,
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


def unpack_distributed_ro(state: Any) -> tuple[Any, DistributedROSpec, TransportInventory]:
    record = distributed_ro_record(state)
    if record is None:
        raise ValueError("no declared distributed RO equipment is installed")
    spec = DistributedROSpec.from_dict(record.settings["physical_model"])
    names = distributed_ro_vessels(state, spec)
    amounts, enthalpies = [], []
    for index, name in enumerate(names):
        tank = state if index == 0 else state.inactive_vessels[name]
        n = dict(tank.species_amounts)
        spec.properties.validate_amounts(n, empty=index > 0)
        expected_h = spec.properties.enthalpy(n, spec.temperature_K, spec.reservoir_pressure_Pa)
        expected_v = (
            spec.properties.volume(n, spec.temperature_K, spec.reservoir_pressure_Pa) * 1000
        )
        if (
            not isclose(tank.temperature_K, spec.temperature_K, abs_tol=1e-8)
            or not isclose(tank.pressure_Pa, spec.reservoir_pressure_Pa, abs_tol=1e-5)
            or not isclose(tank.volume_L, expected_v, abs_tol=1e-10, rel_tol=1e-8)
            or set(tank.phases.phases) != {"liquid"}
        ):
            raise ValueError("distributed RO tank differs from declared single-liquid T/P/V")
        phase = tank.phases.phases["liquid"]
        if dict(phase.species_amounts_mol) != n:
            raise ValueError("distributed RO phase and original bulk species disagree")
        slots = enthalpy_slots(tank, expected_h, 0.0)
        actual = {s.spec.slot_id: s for s in tank.state_slots.slots}
        if set(actual) != {s.spec.slot_id for s in slots.slots}:
            raise ValueError("distributed RO cannot discard other owned state")
        for expected in slots.slots:
            value = actual[expected.spec.slot_id]
            if value.spec != expected.spec or not isclose(
                value.values["enthalpy_J"],
                expected.values["enthalpy_J"],
                rel_tol=1e-7,
                abs_tol=1e-5,
            ):
                raise ValueError("distributed RO owned enthalpy differs from physical liquid")
        amounts.append(n)
        enthalpies.append(sum(slot.values["enthalpy_J"] for slot in actual.values()))
    return record, spec, TransportInventory(tuple(amounts), tuple(enthalpies), state.ledger.time_s)


class DistributedRORuntime:
    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return distributed_ro_record(state) is not None

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = DistributedROSpec.from_dict(distributed_ro_record(state).settings["physical_model"])
        names = distributed_ro_vessels(state, spec)
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
                    spec = DistributedROSpec.from_dict(
                        distributed_ro_record(state).settings["physical_model"]
                    )
                    result.update(distributed_ro_vessels(state, spec))
        return result

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        installed = {v for v, s in states.items() if self.supports(s)}
        components = {c.id: c for c in self.composition.components}
        declared = {c.vessel for c in components.values() if c.kind == "ro_membrane"}
        if not installed <= declared:
            raise ValueError("distributed membrane devices need original ro_membrane components")
        occupied = set(installed)
        for vessel in installed:
            state = states[vessel].replace(
                inactive_vessels={v: s for v, s in states.items() if v != vessel}
            )
            _, spec, _ = unpack_distributed_ro(state)
            if self.composition.physics.network.reactions:
                raise ValueError("declared binary RO currently excludes reactions")
            declared_species = {
                s.species_id: (dict(s.composition), s.charge, s.phase)
                for s in spec.properties.database.species
            }
            original_species = {
                s.species_id: (dict(s.composition), s.charge, s.phase)
                for s in self.composition.physics.network.species
            }
            if set(declared_species) - set(original_species) or any(
                original_species[key] != (composition, charge, "liquid")
                for key, (composition, charge, _) in declared_species.items()
            ):
                raise ValueError("RO database and original network species/charge must agree")
            for name, capacity in (
                (spec.permeate_vessel_id, spec.permeate_capacity_L),
                (spec.retentate_vessel_id, spec.retentate_capacity_L),
            ):
                if name in occupied or capacity > states[name].vessels.vessels[name].max_volume_L:
                    raise ValueError("distributed membrane outlet ownership/capacity conflict")
                occupied.add(name)
                if not any(
                    components[c["source"]["component"]].vessel == vessel
                    and components[c["target"]["component"]].vessel == name
                    for c in self.composition.connections
                ):
                    raise ValueError(
                        "distributed membrane requires both actual finite outlet ports"
                    )

    def configure_flow(self, state: Any, action: dict[str, Any]) -> Any:
        record, _, _ = unpack_distributed_ro(state)
        return configure_liquid_flow(state, action, record)

    def advance(
        self, state: Any, action: dict[str, Any], *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        record, spec, inventory = unpack_distributed_ro(state)
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
                failure_reason="distributed_ro_requires_declared_isotherm",
            )
        try:
            flow = record.settings.get("flow_control", {}).get("requested_flow_m3_s", 0.0)
            result, diagnostic = advance_distributed_ro(
                spec, inventory, flow, float(action["duration_s"])
            )
            final, dt = result.final_state, result.actual_duration_s
            inactive, ages = dict(state.inactive_vessels), dict(state.vessel_elapsed_s)
            for i, name in enumerate(distributed_ro_vessels(state, spec)[1:], start=1):
                tank = install_ro_liquid(
                    inactive[name], spec, final.amounts_mol[i], final.enthalpies_J[i]
                )
                inactive[name] = tank.replace(ledger=advance_ledger(tank.ledger, dt))
                ages[name] = ages.get(name, 0) + dt
            candidate = install_ro_liquid(state, spec, final.amounts_mol[0], final.enthalpies_J[0])
            bath = diagnostic["bath_heat_into_liquid_J"]
            energies = {
                "flow_pump_work_J": diagnostic["shaft_work_J"],
                "energy_jacket_J": max(bath, 0.0),
                "heat_loss_J": diagnostic["motor_loss_J"] + max(-bath, 0.0),
            }
            if spec.energy_recovery is not None:
                energies["flow_shaft_work_J"] = -diagnostic["recovered_shaft_work_J"]
            candidate = candidate.replace(
                inactive_vessels=inactive,
                vessel_elapsed_s=ages,
                equipment=upsert_equipment_record(
                    state.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"last_distributed_ro": diagnostic},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    cost_delta=diagnostic["shaft_work_J"] / 250000,
                    energy_deltas=energies,
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="distributed_ro_solver_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            diagnostics={
                "distributed_ro": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
            },
        )
