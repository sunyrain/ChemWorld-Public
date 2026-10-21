"""Finite isothermal liquid apparatus with explicitly supplied component identities."""

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import EquipmentLedger, VesselLedger, VesselRecord
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.runtime.liquid_equilibrium import install_liquids
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.physical_world import CompositionPhysics


def liquid_world_spec(model, components):
    if not model.wash_feed_id:
        raise ValueError("this finite-wash apparatus requires a named wash feed")
    ids = model.activity_model.component_ids
    if set(ids) != {c.identifier for c in components}:
        raise ValueError("liquid apparatus components must match the declared activity model")
    network = ReactionNetworkSpec("authored-liquid-equilibrium", tuple(
        SpeciesSpec(c.identifier, c.formula) for c in components), ())
    states, declarations = {}, []
    for vessel in ("reactor", "receiver", "wash_supply"):
        amounts = dict.fromkeys(ids, .001 if vessel == "reactor" else .0002)
        if vessel == "wash_supply":
            amounts = dict.fromkeys(ids, 0.0)
            amounts[ids[-1]] = .01
        state = WorldState(
            amounts, model.volume_l(amounts), model.temperature_K, model.pressure_Pa,
            "liquid", vessel, equipment=EquipmentLedger({}),
            units={"amount": "mol", "volume": "L", "temperature": "K", "pressure": "Pa",
                   "time": "s", "cost": "currency", "risk": "risk"},
            vessels=VesselLedger({vessel: VesselRecord(
                vessel, "isothermal-liquid", .1, 450, 5e5, (), model.temperature_K,
                model.pressure_Pa)}))
        states[vessel] = install_liquids(state, model, model.solve(amounts))
        declarations.append({"id": f"{vessel}:material", "kind": "material_input",
                             "vessel": vessel, "parameters": {}})
        if vessel != "wash_supply":
            declarations.extend((
                {"id": f"{vessel}:phase", "kind": "phase", "vessel": vessel, "parameters": {}},
                {"id": f"{vessel}:equilibrium", "kind": "liquid_equilibrium", "vessel": vessel,
                 "parameters": {}},
                {"id": f"{vessel}:observation", "kind": "observation", "vessel": vessel,
                 "parameters": {"instruments": ["phase_state"]}}))
    physics = CompositionPhysics(network, states,
        feed_sources={model.wash_feed_id: "wash_supply"},
        models={"liquid": {"reactor": model, "receiver": model}})
    return WorldCompositionSpec.from_dict({
        "schema_version": "chemworld-world-composition-0.1", "composition_id": "finite-liquid",
        "components": declarations, "vessels": [{"id": name, "capacity_L": .1} for name in states],
        "connections": [{"id": "reactor-to-receiver", "unit": "mol",
                         "source": {"component": "reactor:material", "port": "out"},
                         "target": {"component": "receiver:material", "port": "in"}}],
        "physics": physics.to_dict(),
        "task": {"objective": "study chemical response", "budget": 40,
                 "operations": ["add_component", "mix", "wash", "settle", "wait", "measure",
                                "route_material", "select_vessel", "terminate"],
                 "instruments": ["phase_state"], "termination": "explicit-stop-or-budget"}})
