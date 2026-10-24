"""Fixed synthetic declarations for the prospective 35A comparison block."""

from dataclasses import replace

from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
)

AMOUNTS = {"source": {"A": 0.02, "B": 0.01}, "receiver": {"A": 0.003, "B": 0.002}}


def document(*, second_order=False):
    return {
        "units": {"length": "m", "quantity": "kmol", "activation-energy": "J/kmol"},
        "phases": [
            {
                "name": "synthetic",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "elements": ["H"],
                "species": ["A", "B"],
            }
        ],
        "species": [
            {
                "name": name,
                "composition": {"H": 2},
                "thermo": {
                    "model": "constant-cp",
                    "T0": 298.15,
                    "h0": f"{h} J/mol",
                    "s0": "30 J/mol/K",
                    "cp0": "32 J/mol/K",
                    "T-min": 200,
                    "T-max": 3000,
                },
            }
            for name, h in (("A", 0), ("B", -10000))
        ],
        "reactions": [
            {
                "equation": "2 A => 2 B" if second_order else "A => B",
                "rate-constant": {"A": 0.7, "b": 0, "Ea": 0},
            }
        ],
    }


def network_spec(case="molar"):
    nodes = (GasReactorNodeSpec("source", 1), GasReactorNodeSpec("receiver", 3))
    mode = {"mass": "mass-flow", "valve": "pressure-valve", "closed": "pressure-valve"}.get(
        case, "molar-flow"
    )
    rate = {"mass": 6e-6, "valve": 2e-11, "closed": 2e-11}.get(case, 0.003)
    if case == "source-cutoff":
        nodes = (replace(nodes[0], minimum_amount_mol=0.015), nodes[1])
    elif case == "pressure-limit":
        nodes = (nodes[0], replace(nodes[1], maximum_pressure_Pa=20000))
    return GasReactorNetworkSpec(
        nodes,
        (
            GasFlowConnectionSpec(
                "feed", "source", "receiver", mode, rate, enabled=case != "closed"
            ),
        ),
        700,
        "source",
    )
