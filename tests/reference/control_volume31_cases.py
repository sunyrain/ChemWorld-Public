"""Four prospectively fixed native worlds for parent 31 control-volume closure."""

from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
)
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.world.gas_reactor_network import gas_reactor_network_world
from reference.gas_network35_cases import document

R = 8.31446261815324
T = 700.0
NODES = ("feed-1", "feed-2", "control", "product-1", "product-2")
EDGES = (
    ("in-1", "feed-1", "control", 0.001),
    ("in-2", "feed-2", "control", 0.002),
    ("out-1", "control", "product-1", 0.00075),
    ("out-2", "control", "product-2", 0.00225),
)
CASES = tuple(
    {
        "name": f"{'cstr' if reactive else 'mixer'}-{inventory:.2f}mol",
        "inventory_mol": inventory,
        "rate_constant_s": 0.7 if reactive else 0.0,
    }
    for reactive in (False, True)
    for inventory in (0.03, 0.06)
)


def card(case):
    total = case["inventory_mol"]
    amounts = {
        "feed-1": {"A": 2.4, "B": 0.6},
        "feed-2": {"A": 0.6, "B": 2.4},
        "control": {"A": 0.9 * total, "B": 0.1 * total},
        "product-1": {"A": 0.01, "B": 0.01},
        "product-2": {"A": 0.01, "B": 0.01},
    }
    decay = 0.003 / total + case["rate_constant_s"]
    return {
        **case,
        "R": R,
        "temperature_K": T,
        "reference_temperature_K": 298.15,
        "cp_J_mol_K": {"A": 32.0, "B": 32.0},
        "h0_J_mol": {"A": 0.0, "B": -10000.0},
        "amounts": amounts,
        "volumes_L": {n: total * R * T / 100 if n == "control" else 100.0 for n in NODES},
        "edges": [list(e) for e in EDGES],
        "times_s": [v / decay for v in (0, 1, 5, 10, 20)],
        "minimum_amount_mol": 1e-8,
        "maximum_pressure_Pa": 1e6,
    }


def build(case):
    declared = card(case)
    network = convert_mechanism(document())
    spec = GasReactorNetworkSpec(
        tuple(
            GasReactorNodeSpec(
                n,
                declared["volumes_L"][n],
                chemistry_enabled=n == "control" and case["rate_constant_s"] > 0,
                minimum_amount_mol=1e-8,
                maximum_pressure_Pa=1e6,
            )
            for n in NODES
        ),
        tuple(
            GasFlowConnectionSpec(e, source, target, coefficient=flow)
            for e, source, target, flow in EDGES
        ),
        T,
        "control",
    )
    return declared, network, spec, gas_reactor_network_world(spec, network, declared["amounts"])
