"""Declare an empirical circuit without fictitious chemical bulk inventories."""

from chemworld.foundation import WorldState
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_thevenin import TheveninSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime.thevenin import (
    DEVICE_TYPE,
    METER,
    STRATEGY,
    initial_thevenin_slot,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def thevenin_world_spec(spec, *, budget=80):
    if not isinstance(spec, TheveninSpec):
        raise ValueError("provide a sourced empirical circuit, not a lithium-model alias")
    vessel, p = "circuit_workspace", spec.native_parameters
    record = EquipmentRecord("circuit", DEVICE_TYPE, vessel, batch_reset="preserve", settings={
        "physical_model": spec.to_dict(),
        "electrical_control": {
            "limits_A": [-p["maximum_current_A"], p["maximum_current_A"]],
            "limits_V": [p["minimum_voltage_V"], p["maximum_voltage_V"]],
            "staged_control": True,
            "program": ScalarControlProgram(((0., 0.),), "A").to_dict(),
        }})
    state = WorldState({}, 0., p["reference_frame_temperature_K"], 101325., "formal", vessel,
        liquid_heat_capacity_J_L_K=None,
        vessels=VesselLedger({vessel: VesselRecord(
            vessel, "formal-circuit-workspace", p["housing_volume_L"],
            p["maximum_temperature_K"], 5e5, (), p["reference_frame_temperature_K"], 101325.)}),
        equipment=EquipmentLedger({record.equipment_id: record}),
        state_slots=StateSlots((initial_thevenin_slot(record, spec),)),
        units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa", "time": "s"})
    physics = CompositionPhysics(ReactionNetworkSpec("empirical-thevenin", (), ()),
        {vessel: state}, lab_clock=LabClockSpec(), advance_bindings={vessel: STRATEGY})
    return WorldCompositionSpec.from_dict({
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "native-empirical-thevenin",
        "vessels": [{"id": vessel, "capacity_L": p["housing_volume_L"]}],
        "components": [{"kind": "battery_thevenin", "vessel": vessel, "parameters": {}},
                       {"kind": "lab_schedule", "vessel": vessel,
                        "parameters": {"duration_range_s": [1e-6, 14400.]}},
                       {"kind": "observation", "vessel": vessel,
                        "parameters": {"instruments": [METER]}}],
        "physics": physics.to_dict(),
        "task": {"objective": "characterize a sourced empirical circuit", "budget": budget,
                 "operations": ["configure_electrical_control", "queue_electrical_stage", "wait",
                                "measure", "terminate"], "instruments": [METER]},
    })
