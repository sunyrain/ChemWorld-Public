"""Fixed raw spatial gas cards; no generated evidence or external species data."""

from copy import deepcopy
from math import cos, pi, sin

R = 8.31446261815324
TIMES = (0.1, 0.4, 1.0)


def case(kind="non-equimolar", model="stefan-maxwell", cells=4, temperature=600, pressure=100000):
    species = {
        "A": {
            "atoms": {"H": 2},
            "mass": 0.002016,
            "cp": 30.0,
            "h": 0.0,
            "sigma": 2.9,
            "epsilon": 38.0,
        },
        "B": {
            "atoms": {"H": 4},
            "mass": 0.004032,
            "cp": 44.0,
            "h": -9000.0,
            "sigma": 3.5,
            "epsilon": 100.0,
        },
        "I": {
            "atoms": {"Ar": 1},
            "mass": 0.039948,
            "cp": 20.8,
            "h": 1500.0,
            "sigma": 3.33,
            "epsilon": 136.5,
        },
    }
    if kind in {"binary", "grid"}:
        species.pop("I")
    if kind == "equimolar":
        species["B"].update(atoms={"H": 2}, mass=0.002016)
    if kind == "pure":
        species = {"I": species["I"]}
    fractions = []
    for i in range(cells):
        wave = cos(pi * (i + 0.5) / cells)
        if kind == "pure":
            row = {"I": 1.0}
        elif kind in {"binary", "grid"}:
            amplitude = 0.25
            wave = (
                cells / pi * (sin(pi * (i + 1) / cells) - sin(pi * i / cells))
                if kind == "grid"
                else wave
            )
            row = {"A": 0.5 + amplitude * wave, "B": 0.5 - amplitude * wave}
        else:
            a = 0.45 + 0.2 * wave
            inert = 0.0 if i == 0 and kind == "ternary" else 0.15
            row = {"A": a, "B": 1 - a - inert, "I": inert}
        fractions.append(row)
    kf, kr, order = (0.3, 0.08, 1) if kind == "equimolar" else (0.005, 0.06, 2)
    if kind in {"binary", "ternary", "grid", "pure"}:
        kf = kr = 0.0
    return {
        "id": f"{kind}-{model}-{cells}-{temperature}-{pressure}",
        "kind": kind,
        "species": species,
        "cells": cells,
        "length": 0.02,
        "area": 0.01,
        "temperature": temperature,
        "pressure": pressure,
        "model": model,
        "maximum_pressure": 180000.0,
        "maximum_mach": 0.05,
        "fractions": fractions,
        "kf": kf,
        "kr": kr,
        "order": order,
        "Ea": 1200.0,
    }


def limit(name):
    card = case("pure" if name == "pure" else "non-equimolar")
    card["id"] = name
    card["Ea"] = 0.0
    if name == "association":
        card.update(kf=0.01, kr=0.0)
        card["fractions"] = [{"A": 1.0, "B": 0.0, "I": 0.0} for _ in range(4)]
    elif name in {"dissociation", "pressure-event"}:
        card.update(kf=0.0, kr=0.5)
        card["fractions"] = [{"A": 0.0, "B": 1.0, "I": 0.0} for _ in range(4)]
        if name == "pressure-event":
            card["maximum_pressure"] = 130000.0
    return card


def native(card):
    from chemworld.physchem.finite_volume import DiffusionMesh
    from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec, GasReactorNodeSpec
    from chemworld.physchem.gas_spatial_transport import GasSpatialTransportSpec
    from chemworld.physchem.gas_transport import gas_transport_from_network
    from chemworld.physchem.mechanism_import import convert_mechanism

    entries = []
    for name, values in card["species"].items():
        entries.append(
            {
                "name": name,
                "composition": values["atoms"],
                "transport": {
                    "model": "gas",
                    "geometry": "atom" if name == "I" else "linear",
                    "diameter": values["sigma"],
                    "well-depth": values["epsilon"],
                },
                "thermo": {
                    "model": "constant-cp",
                    "T0": 298.15,
                    "h0": f"{values['h']} J/mol",
                    "cp0": f"{values['cp']} J/mol/K",
                    "s0": "30 J/mol/K",
                    "T-min": 200.0,
                    "T-max": 2000.0,
                },
            }
        )
    reactions = []
    if "A" in card["species"]:
        left = "A" if card["order"] == 1 else "2 A"
        reactions = [
            {"equation": equation, "rate-constant": {"A": rate, "b": 0.0, "Ea": card["Ea"]}}
            for equation, rate in ((f"{left} => B", card["kf"]), (f"B => {left}", card["kr"]))
        ]
    doc = {
        "units": {"length": "m", "quantity": "mol", "activation-energy": "J/mol"},
        "phases": [
            {
                "name": card["id"],
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "elements": sorted({e for s in entries for e in s["composition"]}),
                "species": list(card["species"]),
            }
        ],
        "species": entries,
        "reactions": reactions,
    }
    if reactions:
        network = convert_mechanism(deepcopy(doc))
    else:
        # The mechanism importer deliberately requires a reaction. A pure inert
        # spatial gas uses the original species/network authoring API directly.
        from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec

        network = ReactionNetworkSpec(
            card["id"],
            tuple(
                SpeciesSpec(e["name"], "Ar", phase="gas", metadata={"source_declaration": e})
                for e in entries
            ),
            (),
            metadata={
                "gas_thermochemistry": {
                    e["name"]: {
                        "thermo": e["thermo"],
                        "composition": e["composition"],
                        "reference_pressure_Pa": 101325.0,
                        "units": {},
                    }
                    for e in entries
                }
            },
        )
    mesh = DiffusionMesh.planar(card["length"], card["cells"], area_m2=card["area"])
    names = tuple(f"cell-{i:03d}" for i in range(card["cells"]))
    spatial = GasSpatialTransportSpec(
        mesh,
        names,
        gas_transport_from_network(network, thermal_diffusion="off"),
        card["model"],
        card["maximum_mach"],
    )
    spec = GasReactorNetworkSpec(
        tuple(
            GasReactorNodeSpec(name, float(v * 1000), maximum_pressure_Pa=card["maximum_pressure"])
            for name, v in zip(names, mesh.cell_volumes_m3, strict=True)
        ),
        (),
        card["temperature"],
        names[0],
        spatial_transport=spatial,
    )
    amounts = {
        name: {
            s: x * card["pressure"] * float(v) / (R * card["temperature"]) for s, x in row.items()
        }
        for name, v, row in zip(names, mesh.cell_volumes_m3, card["fractions"], strict=True)
    }
    return spec, network, amounts
