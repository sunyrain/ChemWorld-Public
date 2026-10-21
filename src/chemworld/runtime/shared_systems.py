"""Resolve declared atomic systems, without another state store or time loop."""

from chemworld.foundation.lab_clock import SharedAdvanceSystem
from chemworld.runtime.advance_paths import select_advance_strategy


def connection_ids(composition, vessel_pairs):
    """One finite model path must map to one actual declared material port."""
    components = {component.id: component for component in composition.components}
    result = []
    for source, target in vessel_pairs:
        matches = [
            edge["id"] for edge in composition.connections
            if components[edge["source"]["component"]].vessel == source
            and components[edge["target"]["component"]].vessel == target
            and edge["source"]["port"] == "out" and edge["target"]["port"] == "in"
            and edge["unit"] == "mol"
            and edge.get("scheduling") != "manual"
        ]
        if len(matches) != 1:
            raise ValueError("shared finite path requires one unambiguous declared material port")
        result.extend(matches)
    return tuple(result)


def declared_shared_systems(composition):
    physics, modules = composition.physics, composition.modules
    strategies = modules.build_advance_strategies(composition)
    declarations = modules.advance_specs()
    systems, owned_vessels, owned_connections = [], set(), set()
    for vessel in sorted(physics.initial_states):
        state = physics.initial_states[vessel]
        try:
            selected = select_advance_strategy(
                strategies, state, bindings=physics.advance_bindings,
            )
        except ValueError as exc:
            raise ValueError(f"invalid shared-clock physical binding: {exc}") from exc
        if selected is None:
            continue
        key, strategy = selected
        if not declarations[key].shared_clock:
            raise ValueError(
                f"physical strategy {key!r} has not declared shared-clock participation"
            )
        describe = getattr(strategy, "shared_system", None)
        system = SharedAdvanceSystem(vessel, key, (vessel,)) if describe is None \
            else describe(state, key)
        if (not isinstance(system, SharedAdvanceSystem) or system.system_id != vessel
                or system.strategy_id != key):
            raise ValueError("shared system descriptor must match its declared physical owner")
        if set(system.vessels) - set(physics.initial_states):
            raise ValueError("shared system references an undeclared vessel")
        if owned_vessels & set(system.vessels) or owned_connections & set(system.connections):
            raise ValueError("shared systems cannot overlap physical ownership")
        systems.append(system)
        owned_vessels.update(system.vessels)
        owned_connections.update(system.connections)
    if owned_vessels != set(physics.initial_states):
        raise ValueError("every shared-clock vessel needs one declared physical system")
    manual = {edge["id"] for edge in composition.connections
              if edge.get("scheduling") == "manual"}
    if owned_connections & manual:
        raise ValueError("a shared connection cannot be both manual and model-driven")
    if owned_connections | manual != {edge["id"] for edge in composition.connections}:
        raise ValueError("shared clock has undeclared or unsupported connected paths")
    return tuple(systems)
