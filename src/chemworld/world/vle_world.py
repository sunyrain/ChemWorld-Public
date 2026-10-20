"""A finite two-vessel apparatus for an explicitly supplied VLE model."""

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import (
    EquipmentLedger,
    EquipmentRecord,
    VesselLedger,
    VesselRecord,
)
from chemworld.physchem.flash_units import FlashSystemSpec
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.flash_thermal import install_flash
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def vle_world_spec(properties):
    """Constant-pressure vessels with finite liquid stock; no kinetic reactions."""
    model = FlashSystemSpec(properties, stop_temperature_K=420)
    ids = properties.component_ids
    network = ReactionNetworkSpec("authored-vle", tuple(
        SpeciesSpec(p.component.identifier, p.component.formula) for p in properties.packages), ())
    states, components = {}, []
    for vessel in ("reactor", "receiver", "supply"):
        amounts = dict.fromkeys(ids, .001 if vessel != "receiver" else .0002)
        temperature = 300 if vessel == "supply" else 350
        equipment = {} if vessel == "supply" else {
            f"{vessel}:heater": EquipmentRecord(
                f"{vessel}:heater", "thermal_device", vessel, batch_reset="preserve",
                settings={"thermal_device": ThermalDeviceSpec().to_dict()})}
        state = WorldState(
            amounts, .001, temperature, model.pressure_Pa, "liquid", vessel,
            units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa",
                   "time": "s", "cost": "currency", "risk": "risk"},
            equipment=EquipmentLedger(equipment),
            vessels=VesselLedger({vessel: VesselRecord(
                vessel, "constant-pressure", .1, 450, 5e5, (), temperature, model.pressure_Pa)}))
        states[vessel] = install_flash(state, model, model.at_tp(amounts, temperature))
        components.append({"id": f"{vessel}:material", "kind": "material_input",
                           "vessel": vessel, "parameters": {}})
        if vessel != "supply":
            components.extend((
                {"id": f"{vessel}:thermal", "kind": "thermal", "vessel": vessel,
                 "parameters": {"temperature_range_K": [250, 430], "duration_range_s": [1, 600]}},
                {"id": f"{vessel}:vle", "kind": "vapor_liquid_equilibrium", "vessel": vessel,
                 "parameters": {}},
                {"id": f"{vessel}:sensor", "kind": "observation", "vessel": vessel,
                 "parameters": {"instruments": ["phase_state"]}}))
    physics = CompositionPhysics(network, states, feed_sources={"mixture-feed": "supply"},
                                 flash_models={"reactor": model, "receiver": model})
    return WorldCompositionSpec.from_dict({
        "schema_version": "chemworld-world-composition-0.1", "composition_id": "finite-vle",
        "components": components, "vessels": [{"id": name, "capacity_L": .1} for name in states],
        "connections": [{"id": "reactor-to-receiver", "unit": "mol",
                         "source": {"component": "reactor:material", "port": "out"},
                         "target": {"component": "receiver:material", "port": "in"}}],
        "physics": physics.to_dict(),
        "task": {"objective": "study chemical response", "budget": 40,
                 "operations": ["add_component", "heat", "wait", "pause_control", "resume_control",
                                "route_material", "select_vessel", "measure", "terminate"],
                 "instruments": ["phase_state"], "termination": "explicit-stop-or-budget"}})
