"""Fixed 35D authored cards; shared declarations, separate physical consumers."""

from chemworld.examples.gas_surface_network import example_inputs
from chemworld.physchem.gas_network_surface import GasSurfaceContext
from chemworld.physchem.gas_reactor_network import pack_gas_reactor_network_state

TIMES = (0.0, 0.025, 0.1, 0.3, 0.5)
NUMERIC = (
    "gas",
    "surface",
    "coverage",
    "temperature",
    "pressure",
    "energy",
    "bath",
    "transport",
    "enthalpy",
    "conservation",
    "work",
    "immutable",
)
WORLD = (
    "identity",
    "first_wait",
    "slot_evolution",
    "gas_only",
    "selection",
    "json",
    "second_wait",
    "exact_restore",
    "new_process",
    "site_stock",
    "initial_stock",
    "resource_once",
)
EVENT = (
    "stop_reason",
    "stop_time",
    "gas",
    "surface",
    "coverage",
    "temperature",
    "pressure",
    "energy",
    "bath",
    "transport",
    "enthalpy",
    "finite_work",
)
REJECT = (
    "missing_slot",
    "moved_slot",
    "capacity",
    "area",
    "duplicate_node",
    "unknown_node",
    "identity",
    "sensitivity",
    "missing_initial",
    "shared_species",
    "no_surface_slots",
    "rollback",
)


def case(index):
    size = (index + 1) // 2 if index <= 6 else 2
    thermal = index % 2 == 0
    document, spec, network, amounts, surface = example_inputs(
        size=size, thermal=thermal, both=index >= 7
    )
    slots = tuple(a.make_slot(surface[a.owner_id]) for a in spec.surfaces)
    context = GasSurfaceContext(spec, network, slots)
    state = pack_gas_reactor_network_state(spec, network, amounts, surface_slots=slots)
    energy = {e.vessel_id: e for e in spec.energy_nodes}
    card = {
        "document": document,
        "size": size,
        "thermal": thermal,
        "mass_flow": 6e-6,
        "conductance": 0.01 if thermal else 0.0,
        "nodes": [
            {
                "name": name,
                "volume": volume,
                "temperature": spec.initial_temperatures_K[name],
                "gas": amounts[name],
                "surface": surface.get(name + ":catalyst", {}),
                "support": (energy[name].support_heat_capacity_J_K if energy else 0.0)
                + (0.2 if name in context.slots else 0.0),
            }
            for name, volume in (("source", 0.001), ("receiver", 0.003))
        ],
    }
    return card, spec, network, slots, state, context
