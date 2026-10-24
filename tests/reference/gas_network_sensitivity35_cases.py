"""Fixed 35C raw cards and thin native conversion; reference formulas are separate."""

from copy import deepcopy
from dataclasses import replace

from gas_network35_energy_cases import analytic, case, node

TIMES = (0.0, 0.1, 0.3, 0.7, 1.0)


def card(index):
    if index in (6, 7):
        result = analytic("two-body-conduction" if index == 6 else "newton-cooling")
    elif index == 8:
        result = case("loop")
    else:
        result = case()
        result["anchor"] = "source"
        result["nodes"] = [node("source", 700, 1, 1.2), node("receiver", 700, 3, 2.3)]
        result["amounts"] = {"source": {"A": 0.02, "B": 0.01}, "receiver": {"A": 0.003, "B": 0.002}}
        result["edges"][0]["target"] = "receiver"
        result["thermo"]["B"]["cp"] = 32.0
        result["kinetics"] = {"A": 0.7, "Ea": 0.0}
        result["isothermal"] = True
        if index == 1:
            result["edges"] = []
        if index in (2, 5):
            for item in result["nodes"]:
                item["chemistry"] = False
        if index == 4:
            result["dimer"] = True
            result["edges"][0].update(mode="mass-flow", coefficient=6e-6)
        if index == 5:
            result["edges"][0].update(mode="pressure-valve", coefficient=2e-11)
    result["id"] = f"sensitivity35-{index:02d}"
    result["unit_index"] = index
    return deepcopy(result)


def bindings(index):
    reaction = ("reaction", "reaction-001", "A")
    flow = ("flow", "feed", "coefficient")
    if index == 1:
        return (reaction,)
    if index in (2, 5):
        return (flow,)
    if index in (3, 4):
        return reaction, flow
    if index == 6:
        return (("heat_link", "wall", "conductance_W_K"),)
    if index == 7:
        return (("loss", "reactor", "loss_conductance_W_K"),)
    return reaction, flow, ("heat_link", "wall", "conductance_W_K")


def document(raw):
    return {
        "units": {"length": "m", "quantity": "kmol", "activation-energy": "J/kmol"},
        "phases": [
            {
                "name": raw["id"],
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "elements": ["H"],
                "species": ["A", "B"],
            }
        ],
        "species": [
            {
                "name": name,
                "composition": {"H": 1 if raw.get("dimer") and name == "A" else 2},
                "thermo": {
                    "model": "constant-cp",
                    "T0": 298.15,
                    "h0": f"{v['h']} J/mol",
                    "s0": "40 J/mol/K",
                    "cp0": f"{v['cp']} J/mol/K",
                    "T-min": 150,
                    "T-max": 2000,
                },
            }
            for name, v in raw["thermo"].items()
        ],
        "reactions": [
            {
                "equation": "2 A => B" if raw.get("dimer") else "A => B",
                "rate-constant": {
                    "A": raw["kinetics"]["A"],
                    "b": 0.0,
                    "Ea": raw["kinetics"]["Ea"] * 1000,
                },
            }
        ],
    }


def native(raw):
    from gas_network35_energy_cases import native as energy_native

    from chemworld.physchem.mechanism_import import convert_mechanism

    spec, _, device = energy_native(raw)
    if raw.get("isothermal"):
        spec = replace(spec, energy_nodes=(), heat_links=())
    return spec, convert_mechanism(document(raw)), device


NUMERICAL_SLOTS = (
    "baseline",
    "coarse",
    "fine",
    "step_convergence",
    "units",
    "nonzero_normalization",
    "zero_normalization",
    "element_closure",
    "first_law",
    "denominators",
    "work",
    "immutable",
)
CONTRACT_SLOTS = {
    9: (
        "zero_y",
        "zero_dy",
        "response_signs",
        "disabled_valve",
        "adverse_valve",
        "valve_kink",
        "zero_parameter",
        "unknown_parameter",
        "duplicate_parameter",
        "units",
        "no_filling",
        "immutable",
    ),
    10: (
        "analytic_event",
        "baseline_endpoint",
        "coarse_endpoints",
        "fine_endpoints",
        "pre_event",
        "post_event",
        "event_topology",
        "no_extrapolation",
        "denominator",
        "root_reason",
        "work",
        "immutable",
    ),
    11: (
        "current_time",
        "anchor",
        "baseline_n_T",
        "baseline_P_heat",
        "ledger",
        "snapshot",
        "json",
        "new_process",
        "connections",
        "no_live_charge",
        "work",
        "denominator",
    ),
    12: (
        "budget",
        "parameter_denominator",
        "unattempted",
        "nonfinite",
        "overflow",
        "spatial",
        "thermo_parameter",
        "time_grid",
        "step",
        "forward_failure",
        "strict_json",
        "immutable",
    ),
}
