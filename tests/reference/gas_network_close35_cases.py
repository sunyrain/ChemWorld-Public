"""Fixed raw development qualification inputs for the remaining original 35 scope."""

from copy import deepcopy

R = 8.31446261815324
TIMES = (0.05, 0.2, 0.5)


def document(*, irreversible=False, ignition=False):
    return {
        "units": {"length": "m", "quantity": "kmol", "activation-energy": "J/kmol"},
        "phases": [
            {
                "name": "gas",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "species": ["A", "B"],
                "elements": ["H"],
            }
        ],
        "species": [
            {
                "name": s,
                "composition": {"H": 2},
                "thermo": {
                    "model": "constant-cp",
                    "T0": 298.15,
                    "h0": f"{h} J/mol",
                    "s0": "40 J/mol/K",
                    "cp0": "32 J/mol/K",
                    "T-min": 100,
                    "T-max": 3500,
                },
            }
            for s, h in (("A", 0), ("B", -40000 if ignition else -2000))
        ],
        "reactions": [
            {
                "equation": "A => B" if irreversible else "A <=> B",
                "rate-constant": {
                    "A": 1e8 if ignition else 2.0,
                    "b": 0,
                    "Ea": 8e7 if ignition else 0,
                },
            }
        ],
    }


def closed(thermal=False, pair=False):
    from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec, GasThermalLinkSpec
    from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec, GasReactorNodeSpec
    from chemworld.physchem.mechanism_import import convert_mechanism

    names = ("r", "s") if pair else ("r",)
    spec = GasReactorNetworkSpec(
        tuple(GasReactorNodeSpec(n, 1) for n in names),
        (),
        500,
        "r",
        tuple(
            GasNodeEnergySpec(
                n,
                500 if n == "r" else 700,
                0,
                minimum_temperature_K=150,
                maximum_temperature_K=3000,
            )
            for n in names
        )
        if thermal
        else (),
        (GasThermalLinkSpec("wall", "r", "s", 0.1),) if pair else (),
    )
    amounts = {n: {"A": 0.015, "B": 0.005} for n in names}
    raw = document()
    return raw, spec, convert_mechanism(raw), amounts


def stirred(temperature=300.0, *, environment=300.0, loss=0.03):
    from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec
    from chemworld.physchem.gas_reactor_network import (
        GasFlowConnectionSpec,
        GasReactorNetworkSpec,
        GasReactorNodeSpec,
    )
    from chemworld.physchem.mechanism_import import convert_mechanism

    # Real finite source and receiver. Only analysis explicitly holds them.
    # Their 10,000 L stocks also permit finite transient demonstrations.
    raw = document(irreversible=True, ignition=True)
    feed, valve, outlet_p, mw = 0.001, 2e-10, 10000.0, 0.002016
    n = (outlet_p + feed * mw / valve) / (R * temperature * 1000)
    k = 1e8 * __import__("math").exp(-80000 / (R * temperature))
    a = feed / (k + feed / n)
    spec = GasReactorNetworkSpec(
        (
            GasReactorNodeSpec("feed", 10000, False),
            GasReactorNodeSpec("r", 1),
            GasReactorNodeSpec("sink", 10000, False),
        ),
        (
            GasFlowConnectionSpec("in", "feed", "r", "molar-flow", feed),
            GasFlowConnectionSpec("out", "r", "sink", "pressure-valve", valve),
        ),
        temperature,
        "r",
        tuple(
            GasNodeEnergySpec(
                name,
                t,
                0,
                environment_temperature_K=environment,
                loss_conductance_W_K=loss if name == "r" else 0,
                minimum_temperature_K=150,
                maximum_temperature_K=3000,
            )
            for name, t in (("feed", 300), ("r", temperature), ("sink", 300))
        ),
    )
    amounts = {
        "feed": {"A": 800, "B": 0},
        "r": {"A": a, "B": n - a},
        "sink": {"A": 0, "B": outlet_p * 10000 / (R * 300 * 1000)},
    }
    return raw, spec, convert_mechanism(raw), amounts


def surface():
    from dataclasses import replace

    from chemworld.examples.gas_surface_network import example_inputs
    from chemworld.examples.native_imported_surface import SOURCE
    from chemworld.physchem.surface_mechanism_import import convert_surface_mechanism

    raw, spec, _network, amounts, initial = example_inputs()
    raw["reactions"][-1]["equation"] = "HB <=> B + Pt"
    raw["reactions"][-1].pop("orders")
    bundle = convert_surface_mechanism(
        raw,
        surface_phase="surface",
        source=SOURCE,
        minimum_temperature_K=300,
        maximum_temperature_K=1400,
    )
    attachment = replace(spec.surfaces[0], mechanism=bundle.surface, area_m2=1000)
    spec = replace(
        spec,
        nodes=tuple(n for n in spec.nodes if n.vessel_id == "source"),
        connections=(),
        surfaces=(attachment,),
    )
    slots = (attachment.make_slot({s: v * 10000 for s, v in initial[attachment.owner_id].items()}),)
    return deepcopy(raw), spec, bundle.gas_network, {"source": amounts["source"]}, slots
