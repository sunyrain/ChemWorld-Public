"""Finite flowsheet adapter for the original transport and shared-clock runtime."""

from collections.abc import Mapping
from math import isfinite
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.physchem.molecular_tanks import validate_molecular_amounts
from chemworld.physchem.process_flowsheet_inventory import (
    ProcessFlowsheetInventory,
    ProcessFlowsheetInventorySpec,
    advance_process_flowsheet,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.molecular_tanks import install_molecular_tank, read_molecular_tank

FLOWSHEET_TYPE = "finite_process_flowsheet"
FLOWSHEET_STATE_TYPE = "process_flowsheet_enthalpy_v1"


def flowsheet_record(state: Any) -> Any:
    return attached_device_record(state, FLOWSHEET_TYPE)


def install_flowsheet_tank(
    state: Any, spec: ProcessFlowsheetInventorySpec, index: int,
    amounts: Mapping[str, float], enthalpy: float,
) -> Any:
    tank = spec.tanks[index]
    return install_molecular_tank(
        state, spec.at_ph(index, amounts, enthalpy), amounts, enthalpy,
        phase_id=tank.phase_id, pressure_Pa=tank.pressure_Pa, state_type=FLOWSHEET_STATE_TYPE,
    )


def unpack_flowsheet(state: Any) -> tuple[Any, ProcessFlowsheetInventorySpec,
                                          ProcessFlowsheetInventory]:
    record = flowsheet_record(state)
    if record is None:
        raise ValueError("no finite process flowsheet is attached")
    spec = ProcessFlowsheetInventorySpec.from_dict(record.settings["physical_model"])
    if state.vessel_id != spec.vessel_ids[0]:
        raise ValueError("finite flowsheet must be anchored to its declared first tank")
    amounts, enthalpies = [], []
    for i, declared in enumerate(spec.tanks):
        tank = state if i == 0 else state.inactive_vessels.get(declared.vessel_id)
        if tank is None:
            raise ValueError("finite flowsheet is missing a declared tank owner")
        n = dict(tank.species_amounts)
        total = validate_molecular_amounts(spec.properties, n)
        h = read_molecular_tank(
            tank, spec.at_tp(i, n, tank.temperature_K), phase_id=declared.phase_id,
            pressure_Pa=declared.pressure_Pa, maximum_volume_L=declared.capacity_L,
            state_type=FLOWSHEET_STATE_TYPE,
        )
        if not declared.reserve_mol-1e-9 <= total <= declared.capacity_mol+1e-9:
            raise ValueError("finite flowsheet inventory exceeds its declared limits")
        amounts.append(n)
        enthalpies.append(h)
    scale = record.settings.get("flow_control", {}).get("rate_scale", 1.0)
    if isinstance(scale, bool) or not isfinite(scale) or scale < 0:
        raise ValueError("flowsheet feed scale must be finite and nonnegative")
    return record, spec, ProcessFlowsheetInventory(tuple(amounts), tuple(enthalpies),
                                                   state.ledger.time_s)


def flowsheet_connections(spec: ProcessFlowsheetInventorySpec) -> tuple[tuple[str, str], ...]:
    # The whole algebraic graph is one material-transport owner. These edges
    # declare its finite boundary connectivity; no second runtime owns a unit.
    return tuple(sorted({(s, t) for s in spec.source_tanks.values()
                         for t in spec.outlet_tanks.values() if s != t}))


class ProcessFlowsheetAdapter:
    supported_operations = ("set_flow_rate", "run_flow", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return flowsheet_record(state) is not None

    def validate_state(self, state: Any) -> None:
        unpack_flowsheet(state)

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = ProcessFlowsheetInventorySpec.from_dict(
            flowsheet_record(state).settings["physical_model"])
        return SharedAdvanceSystem(state.vessel_id, strategy_id, spec.vessel_ids,
                                   connection_ids(self.composition, flowsheet_connections(spec)))

    @property
    def preserved_source_vessels(self) -> set[str]:
        names: set[str] = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    names.update(ProcessFlowsheetInventorySpec.from_dict(
                        flowsheet_record(state).settings["physical_model"]).vessel_ids)
        return names

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def reconcile_initialization(self, transition: Any) -> Any:
        active = transition.state
        states = {active.vessel_id: active, **active.inactive_vessels}
        for state in states.values():
            if self.supports(state):
                self.validate_state(state.replace(inactive_vessels={
                    k: v for k, v in states.items() if k != state.vessel_id}))
        return transition

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        installed = {v for v, s in states.items() if self.supports(s)}
        declared = {c.vessel for c in self.composition.components if c.kind == "finite_transport"}
        if not installed <= declared:
            raise ValueError("finite flowsheet requires original finite_transport")
        occupied: set[str] = set()
        for name in installed:
            state = states[name].replace(inactive_vessels={k: v for k, v in states.items()
                                                          if k != name})
            _, spec, _ = unpack_flowsheet(state)
            if occupied.intersection(spec.vessel_ids):
                raise ValueError("finite flowsheets cannot share tank ownership")
            occupied.update(spec.vessel_ids)
            if self.composition.physics.network.reactions or set(
                    self.composition.physics.network.species_ids
            ) != set(spec.properties.component_ids):
                raise ValueError("flowsheet requires the same original nonreacting network")
            self.shared_system(state, "finite-pump-transport")
            for tank in spec.tanks:
                hardware = states[tank.vessel_id].vessels.vessels[tank.vessel_id]
                if (hardware.max_volume_L < tank.capacity_L
                        or hardware.max_pressure_Pa < tank.pressure_Pa
                        or hardware.max_temperature_K < tank.domain.temperature_range_K[1]):
                    raise ValueError("flowsheet declaration exceeds installed tank hardware")

    def configure_flow(self, state: Any, action: dict) -> Any:
        record, spec, initial = unpack_flowsheet(state)
        flow = action["flow_rate_mL_min"]
        if (isinstance(flow, bool) or not isfinite(flow) or not 0 <= flow <= 20
                or action.get("flow_path", "process") != "process"):
            raise ValueError("flowsheet control requires total feed 0-20 mL/min on process path")
        indices = {name: i for i, name in enumerate(spec.vessel_ids)}
        reference = 0.0
        for source in spec.graph.sources:
            i = indices[spec.source_tanks[source.port_id]]
            phase = spec.at_ph(i, initial.amounts_mol[i], initial.enthalpies_J[i])
            reference += 6e7*source.stream.molar_rate_mol_s/phase.molar_density_mol_m3
        return state.replace(equipment=upsert_equipment_record(
            state.equipment, equipment_id=record.equipment_id, equipment_type=record.equipment_type,
            attached_vessel_id=state.vessel_id, batch_reset="preserve",
            settings={"flow_control": {"configured": True, "rate_scale": flow/reference,
                                        "requested_total_feed_mL_min": flow,
                                        "semantics": "fixed_molar_ratios_at_configuration"}}))

    def advance(self, state: Any, action: dict, *, heat: bool = False,
                jacket_program: Any = None) -> ModelProviderResult:
        if heat or jacket_program is not None or "target_temperature_K" in action:
            return ModelProviderResult({}, diagnostics={}, success=False, outcome="rejected",
                                       failure_reason="flowsheet_uses_declared_process_units")
        try:
            record, spec, initial = unpack_flowsheet(state)
            result, diagnostic = advance_process_flowsheet(
                spec, initial, float(action["duration_s"]),
                rate_scale=record.settings.get("flow_control", {}).get("rate_scale", 1.0))
            final, dt = result.final_state, result.actual_duration_s
            inactive, ages = dict(state.inactive_vessels), dict(state.vessel_elapsed_s)
            for i, name in enumerate(spec.vessel_ids[1:], 1):
                tank = install_flowsheet_tank(inactive[name], spec, i,
                                              final.amounts_mol[i], final.enthalpies_J[i])
                inactive[name] = tank.replace(ledger=advance_ledger(tank.ledger, dt))
                ages[name] = ages.get(name, 0.0)+dt
            candidate = install_flowsheet_tank(state, spec, 0, final.amounts_mol[0],
                                               final.enthalpies_J[0])
            previous = record.settings.get("utility_integrals", {})
            totals = {key: previous.get(key, 0.0)+diagnostic[key] for key in
                      ("heater_energy_J", "condenser_energy_J", "shaft_import_energy_J",
                       "shaft_export_energy_J")}
            candidate = candidate.replace(
                inactive_vessels=inactive, vessel_elapsed_s=ages,
                equipment=upsert_equipment_record(
                    candidate.equipment, equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type, attached_vessel_id=state.vessel_id,
                    batch_reset="preserve", settings={"last_flowsheet": diagnostic,
                                                      "utility_integrals": totals}),
                ledger=advance_ledger(state.ledger, dt, energy_deltas={
                    "flow_heat_J": diagnostic["heat_J"],
                    "flow_shaft_work_J": diagnostic["shaft_work_J"],
                    "boundary_work_J": diagnostic["pressure_work_J"]}))
            self.validate_state(candidate)
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult({}, success=False, outcome="numerical_failure",
                                       failure_reason="finite_process_flowsheet_failed",
                                       diagnostics={"solver_error": str(exc),
                                                    "flowsheet": getattr(exc, "diagnostics", {})})
        return ModelProviderResult(
            {}, state=candidate, actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            vessel_thermal_deltas={name: {} for name in spec.vessel_ids},
            diagnostics={"flowsheet": diagnostic, "solver": result.solver_diagnostic,
                         "stopping_event": result.stopping_event,
                         "component_residuals_mol": result.component_residuals_mol,
                         "enthalpy_residual_J": result.enthalpy_residual_J})
