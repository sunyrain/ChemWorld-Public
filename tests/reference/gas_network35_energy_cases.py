"""Fixed raw 35B inputs, shared by independent n/U reference and native adapter."""

from copy import deepcopy

R = 8.31446261815324
TIMES = (0.1, 0.4, 1.0)


def node(name, temperature, volume, support):
    return {
        "id": name,
        "temperature_K": temperature,
        "volume_L": volume,
        "support_J_K": support,
        "reference_K": 298.15,
        "environment_K": 300.0,
        "loss_W_K": 0.0,
        "minimum_K": 200.0,
        "maximum_K": 1500.0,
        "minimum_mol": 1e-6,
        "maximum_Pa": 1e8,
        "chemistry": True,
    }


def case(topology="serial", mode="adiabatic"):
    card = {
        "id": f"35B-{topology}-{mode}",
        "anchor": "reactor",
        "nodes": [node("source", 750, 1, 1.2), node("reactor", 450, 3, 2.3)],
        "amounts": {"source": {"A": 0.03, "B": 0.005}, "reactor": {"A": 0.01, "B": 0.005}},
        "edges": [
            {
                "id": "feed",
                "source": "source",
                "target": "reactor",
                "mode": "molar-flow",
                "coefficient": 0.003,
            }
        ],
        "heat_links": [],
        "thermo": {"A": {"cp": 32.0, "h": 0.0}, "B": {"cp": 38.0, "h": -10000.0}},
        "kinetics": {"A": 0.8, "Ea": 4000.0},
        "device": None,
    }
    if topology == "loop":
        card["nodes"][1]["volume_L"] = 2
        card["nodes"].append(node("collector", 400, 1.5, 1.8))
        card["amounts"]["collector"] = {"A": 0.008, "B": 0.002}
        card["edges"].extend(
            [
                {
                    "id": "product",
                    "source": "reactor",
                    "target": "collector",
                    "mode": "molar-flow",
                    "coefficient": 0.001,
                },
                {
                    "id": "recycle",
                    "source": "collector",
                    "target": "source",
                    "mode": "molar-flow",
                    "coefficient": 0.0005,
                },
            ]
        )
        card["heat_links"] = [
            {"id": "wall", "source": "reactor", "target": "collector", "conductance": 0.02}
        ]
    if mode == "inert":
        card["kinetics"]["A"] = 0.0
    elif mode in {"heating", "cooling"}:
        card["device"] = device(900 if mode == "heating" else 280)
        if mode == "cooling":
            for item in card["nodes"]:
                item["loss_W_K"] = 0.04
    elif mode != "adiabatic":
        raise ValueError(mode)
    return card


def device(setpoint=900, power=20, *, enabled=True):
    return {
        "conductance_W_K": 0.2,
        "maximum_power_W": power,
        "setpoint_K": setpoint,
        "calibration_factor": 1.0,
        "enabled": enabled,
        "mode": "finite_power",
        "minimum_setpoint_K": 200.0,
        "maximum_setpoint_K": 1500.0,
        "maximum_cooling_power_W": 10.0,
    }


def single():
    card = case(mode="inert")
    card["nodes"] = [node("reactor", 600, 1, 1.0)]
    card["amounts"] = {"reactor": {"A": 0.02, "B": 0.0}}
    card["edges"] = []
    card["thermo"]["B"]["cp"] = 32.0
    return card


def analytic(name):
    card = single()
    card["id"] = name
    if name == "constant-power":
        card["device"] = device(900, 5)
    elif name == "newton-cooling":
        card["nodes"][0]["loss_W_K"] = 0.2
    elif name == "two-body-conduction":
        card["nodes"].append(node("other", 300, 1, 2.0))
        card["amounts"]["other"] = {"A": 0.02, "B": 0.0}
        card["heat_links"] = [
            {"id": "wall", "source": "reactor", "target": "other", "conductance": 0.4}
        ]
    elif name == "closed-exotherm":
        card["nodes"][0]["temperature_K"] = 500.0
        card["nodes"][0]["support_J_K"] = 1.3
        card["kinetics"] = {"A": 0.7, "Ea": 0.0}
    else:
        raise ValueError(name)
    return card


def event(name):
    duration = 3.0
    if name == "inventory":
        card = case(mode="inert")
        card["nodes"][0]["minimum_mol"] = 0.03
    elif name == "high-T":
        card = analytic("closed-exotherm")
        card["nodes"][0]["maximum_K"] = 510.0
    elif name == "low-T":
        card = analytic("newton-cooling")
        card["nodes"][0].update({"minimum_K": 590.0, "loss_W_K": 1.0, "environment_K": 250.0})
    elif name == "pressure":
        card = analytic("constant-power")
        card["nodes"][0]["maximum_Pa"] = 0.02 * R * 600 * 1000 * 1.005
    else:
        raise ValueError(name)
    card["id"] = f"event-{name}"
    return card, duration


def flow_case(name):
    card = case()
    if name == "mass":
        card["edges"][0].update({"mode": "mass-flow", "coefficient": 0.003 * 0.002016})
    elif name in {"valve", "reverse-valve"}:
        card["edges"][0].update({"mode": "pressure-valve", "coefficient": 1e-11})
        if name == "reverse-valve":
            card["amounts"] = {
                "source": {"A": 0.003, "B": 0.002},
                "reactor": {"A": 0.08, "B": 0.02},
            }
    elif name == "reversed-T":
        card["nodes"][0]["temperature_K"], card["nodes"][1]["temperature_K"] = 350.0, 750.0
    else:
        raise ValueError(name)
    card["id"] = name
    return card


def native(card):
    from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec, GasThermalLinkSpec
    from chemworld.physchem.gas_reactor_network import (
        GasFlowConnectionSpec,
        GasReactorNetworkSpec,
        GasReactorNodeSpec,
    )
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
    from chemworld.physchem.mechanism_import import convert_mechanism

    document = {
        "units": {"length": "m", "quantity": "kmol", "activation-energy": "J/kmol"},
        "phases": [
            {
                "name": card["id"],
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
                    "h0": f"{values['h']} J/mol",
                    "s0": "40 J/mol/K",
                    "cp0": f"{values['cp']} J/mol/K",
                    "T-min": 150,
                    "T-max": 2000,
                },
            }
            for name, values in card["thermo"].items()
        ],
        "reactions": [
            {
                "equation": "A => B",
                "rate-constant": {
                    "A": card["kinetics"]["A"],
                    "b": 0.0,
                    "Ea": card["kinetics"]["Ea"] * 1000,
                },
            }
        ],
    }
    spec = GasReactorNetworkSpec(
        tuple(
            GasReactorNodeSpec(
                n["id"], n["volume_L"], n["chemistry"], n["minimum_mol"], n["maximum_Pa"]
            )
            for n in card["nodes"]
        ),
        tuple(
            GasFlowConnectionSpec(e["id"], e["source"], e["target"], e["mode"], e["coefficient"])
            for e in card["edges"]
        ),
        next(n["temperature_K"] for n in card["nodes"] if n["id"] == card["anchor"]),
        card["anchor"],
        tuple(
            GasNodeEnergySpec(
                n["id"],
                n["temperature_K"],
                n["support_J_K"],
                n["reference_K"],
                n["environment_K"],
                n["loss_W_K"],
                n["minimum_K"],
                n["maximum_K"],
            )
            for n in card["nodes"]
        ),
        tuple(
            GasThermalLinkSpec(e["id"], e["source"], e["target"], e["conductance"])
            for e in card["heat_links"]
        ),
    )
    return (
        spec,
        convert_mechanism(document),
        (None if card["device"] is None else ThermalDeviceSpec(**deepcopy(card["device"]))),
    )
