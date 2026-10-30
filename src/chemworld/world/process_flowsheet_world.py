"""Build one programmable finite-tank world from a declared process graph."""

from collections.abc import Mapping, Sequence
from typing import Any

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.process_flowsheet_inventory import (
    ProcessFlowsheetInventorySpec,
    process_flowsheet_inventory,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.process_flowsheet import (
    FLOWSHEET_TYPE,
    flowsheet_connections,
    install_flowsheet_tank,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def build_process_flowsheet_world(
    spec: ProcessFlowsheetInventorySpec,
    amounts: Sequence[Mapping[str, float]],
    temperatures_K: Sequence[float],
    *,
    formulas: Mapping[str, str],
    shared_clock: bool = True,
    budget: int = 100,
) -> WorldCompositionSpec:
    inventory = process_flowsheet_inventory(spec, amounts, temperatures_K)
    anchor = spec.vessel_ids[0]
    if set(formulas) != set(spec.properties.component_ids):
        raise ValueError("finite flowsheet requires original component formulas")
    states = {}
    for i, tank in enumerate(spec.tanks):
        name = tank.vessel_id
        equipment = {} if i else {
            "flowsheet": EquipmentRecord(
                "flowsheet", FLOWSHEET_TYPE, name, batch_reset="preserve",
                settings={"physical_model": spec.to_dict(),
                          "flow_control": {"configured": True, "rate_scale": 1.0},
                          "utility_integrals": dict.fromkeys(
                              ("heater_energy_J", "condenser_energy_J",
                               "shaft_import_energy_J", "shaft_export_energy_J"), 0.0)})}
        closed = spec.at_tp(i, amounts[i], temperatures_K[i])
        state = WorldState(
            dict(amounts[i]), 1000*sum(amounts[i].values())/closed.molar_density_mol_m3,
            temperatures_K[i], tank.pressure_Pa, tank.phase_id, name,
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger({name: VesselRecord(
                name, "molecular-moving-volume-tank", tank.capacity_L,
                tank.domain.temperature_range_K[1], tank.pressure_Pa*1.1,
                (tank.phase_id,), temperatures_K[i], tank.pressure_Pa)}),
            units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa",
                   "time": "s", "cost": "currency", "risk": "risk"})
        states[name] = install_flowsheet_tank(state, spec, i, amounts[i], inventory.enthalpies_J[i])
    physics = CompositionPhysics(
        ReactionNetworkSpec("finite-process-flowsheet", tuple(
            SpeciesSpec(k, formulas[k], phase="liquid")
            for k in spec.properties.component_ids), ()),
        states, lab_clock=LabClockSpec(vessel_capacity=len(states)) if shared_clock else None)
    components: list[dict[str, Any]] = [
        {"id": t.vessel_id, "kind": "finite_transport" if i == 0 else "material_input",
         "vessel": t.vessel_id,
         "parameters": {"duration_range_s": [1e-6, 14400]} if i == 0 else {}}
        for i, t in enumerate(spec.tanks)]
    if shared_clock:
        components.append({"id": "clock", "kind": "lab_schedule", "vessel": anchor,
                           "parameters": {"duration_range_s": [1e-6, 14400]}})
    return WorldCompositionSpec.from_dict({
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "finite-process-flowsheet", "components": components,
        "vessels": [{"id": t.vessel_id, "capacity_L": t.capacity_L} for t in spec.tanks],
        "connections": [{"id": f"flowsheet:{i}", "unit": "mol",
                         "source": {"component": s, "port": "out"},
                         "target": {"component": t, "port": "in"}}
                        for i, (s, t) in enumerate(flowsheet_connections(spec))],
        "physics": physics.to_dict(),
        "task": {"objective": "explore finite molecular flowsheets and controlled operation",
                 "budget": budget, "operations": ["set_flow_rate", "wait", "terminate"]
                 if shared_clock else ["set_flow_rate", "run_flow", "wait", "terminate"],
                 "instruments": [], "termination": "explicit-stop-or-budget"}})
