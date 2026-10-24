"""Periodic ED adapter for the original membrane, electrical source and clock.

No persistent simulation state is owned here: original species, phase, H slots,
equipment settings, finite vessels and resource ledger remain authoritative.
"""

from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.electrodialysis import ElectrodialysisSpec, advance_electrodialysis
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.control_program import electrical_device
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.transport_devices import enthalpy_slots

ELECTRODIALYSIS_TYPE = "periodic_interior_electrodialysis"


def electrodialysis_record(state: Any) -> Any:
    return attached_device_record(state, ELECTRODIALYSIS_TYPE)


def electrodialysis_vessels(state: Any, spec: ElectrodialysisSpec) -> tuple[str, str]:
    names = (state.vessel_id, spec.concentrate_vessel_id)
    if len(set(names)) != 2:
        raise ValueError("periodic ED requires two distinct finite vessels")
    return names


def install_ed_liquid(
    state: Any, spec: ElectrodialysisSpec, amounts: dict[str, float], enthalpy: float
) -> Any:
    p, t = spec.reservoir_pressure_Pa, spec.temperature_K
    volume = 1000 * spec.properties.volume(amounts, t, p)
    fluid_h = spec.properties.enthalpy(amounts, t, p)
    if not isclose(fluid_h, enthalpy, rel_tol=1e-7, abs_tol=1e-5):
        raise ValueError("periodic ED liquid H must match its declared isotherm")
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


def unpack_electrodialysis(state: Any) -> tuple[Any, ElectrodialysisSpec, TransportInventory]:
    record = electrodialysis_record(state)
    if record is None:
        raise ValueError("no declared periodic ED equipment is installed")
    spec = ElectrodialysisSpec.from_dict(record.settings["physical_model"])
    if electrical_device(state).equipment_id != record.equipment_id:
        raise ValueError("periodic ED must own its unique connected electrical source")
    names = electrodialysis_vessels(state, spec)
    amounts, enthalpies = [], []
    for index, name in enumerate(names):
        tank = state if index == 0 else state.inactive_vessels[name]
        n = dict(tank.species_amounts)
        spec.properties.validate_amounts(n, empty=False)
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
            raise ValueError("periodic ED tank differs from declared single-liquid T/P/V")
        phase = tank.phases.phases["liquid"]
        if dict(phase.species_amounts_mol) != n:
            raise ValueError("periodic ED phase and original bulk species disagree")
        slots = enthalpy_slots(tank, expected_h, 0.0)
        actual = {s.spec.slot_id: s for s in tank.state_slots.slots}
        if set(actual) != {s.spec.slot_id for s in slots.slots}:
            raise ValueError("periodic ED cannot discard other owned state")
        for expected in slots.slots:
            value = actual[expected.spec.slot_id]
            if value.spec != expected.spec or not isclose(
                value.values["enthalpy_J"],
                expected.values["enthalpy_J"],
                rel_tol=1e-7,
                abs_tol=1e-5,
            ):
                raise ValueError("periodic ED owned enthalpy differs from physical liquid")
        amounts.append(n)
        enthalpies.append(sum(slot.values["enthalpy_J"] for slot in actual.values()))
    return record, spec, TransportInventory(tuple(amounts), tuple(enthalpies), state.ledger.time_s)


class ElectrodialysisRuntime:
    supported_operations = ("configure_electrical_control", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return electrodialysis_record(state) is not None

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = ElectrodialysisSpec.from_dict(
            electrodialysis_record(state).settings["physical_model"]
        )
        names = electrodialysis_vessels(state, spec)
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            names,
            connection_ids(self.composition, [(names[0], names[1])]),
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
                    spec = ElectrodialysisSpec.from_dict(
                        electrodialysis_record(state).settings["physical_model"]
                    )
                    result.update(electrodialysis_vessels(state, spec))
        return result

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        components = {c.id: c for c in self.composition.components}
        for vessel, original in states.items():
            if not self.supports(original):
                continue
            state = original.replace(
                inactive_vessels={v: s for v, s in states.items() if v != vessel}
            )
            record, spec, _ = unpack_electrodialysis(state)
            if self.composition.physics.network.reactions:
                raise ValueError("periodic ED currently excludes homogeneous reactions")
            native = {
                s.species_id: (dict(s.composition), s.charge, s.phase)
                for s in self.composition.physics.network.species
            }
            if any(
                native.get(s.species_id) != (dict(s.composition), s.charge, "liquid")
                for s in spec.properties.database.species
            ):
                raise ValueError("ED database and original network identities must agree")
            names = electrodialysis_vessels(state, spec)
            if any(
                capacity > states[name].vessels.vessels[name].max_volume_L
                for name, capacity in zip(names, spec.capacities_L, strict=True)
            ):
                raise ValueError("ED physical tank capacity is smaller than its declaration")
            if not any(
                components[c["source"]["component"]].vessel == vessel
                and components[c["target"]["component"]].vessel == spec.concentrate_vessel_id
                for c in self.composition.connections
            ):
                raise ValueError("periodic ED requires an actual finite concentrate connection")
            settings = record.settings["electrical_control"]
            if (
                settings["limits_V"] != [-spec.maximum_voltage_V, spec.maximum_voltage_V]
                or settings["limits_A"] != [-spec.maximum_current_A, spec.maximum_current_A]
                or settings.get("staged_control", False)
            ):
                raise ValueError("ED electrical source must match unstaged physical I/V limits")

    def configure_flow(self, state: Any, action: dict[str, Any]) -> Any:
        raise ValueError("periodic batch ED has no declared liquid pump")

    def advance(
        self, state: Any, action: dict[str, Any], *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        record, spec, inventory = unpack_electrodialysis(state)
        if heat or jacket_program is not None or action.get("operation", "wait") != "wait":
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="periodic_ed_requires_isothermal_wait",
            )
        try:
            program = ScalarControlProgram(**record.settings["electrical_control"]["program"])
            result, receipt = advance_electrodialysis(
                spec, inventory, program, float(action["duration_s"])
            )
            final, dt = result.final_state, result.actual_duration_s
            heat_out = -receipt["bath_heat_into_liquid_J"]
            name = spec.concentrate_vessel_id
            inactive = dict(state.inactive_vessels)
            tank = install_ed_liquid(
                inactive[name], spec, final.amounts_mol[1], final.enthalpies_J[1]
            )
            inactive[name] = tank.replace(
                ledger=advance_ledger(tank.ledger, dt, energy_deltas={"heat_loss_J": heat_out / 2})
            )
            candidate = install_ed_liquid(state, spec, final.amounts_mol[0], final.enthalpies_J[0])
            ages = dict(state.vessel_elapsed_s)
            ages[name] = ages.get(name, 0) + dt
            candidate = candidate.replace(
                inactive_vessels=inactive,
                vessel_elapsed_s=ages,
                equipment=upsert_equipment_record(
                    state.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"last_electrodialysis": receipt},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    cost_delta=receipt["electrical_work_J"] / 250000,
                    energy_deltas={
                        "flow_electrical_work_J": receipt["electrical_work_J"],
                        "heat_loss_J": heat_out,
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                diagnostics={"solver_error": str(exc)},
                success=False,
                outcome="numerical_failure",
                failure_reason="periodic_ed_solver_failed",
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            vessel_thermal_deltas={
                v: {"energy_jacket_J": 0.0, "heat_loss_J": heat_out / 2}
                for v in electrodialysis_vessels(state, spec)
            },
            diagnostics={
                "electrodialysis": receipt,
                "electrical_control": {
                    "unit": program.unit,
                    "current_A": receipt["final"]["current_A"],
                    "voltage_V": receipt["final"]["voltage_V"],
                },
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
            },
        )
