"""Prospective eight-unit stationary Newman-Tobias EIS inputs."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
from battery_newman_tobias_cases import BASE, native_spec

F, R = 96485.33212, 8.314462618
FREQUENCIES = np.geomspace(1e-5, 1e1, 13)
GRIDS = {"coarse": (24, 12, 24), "fine": (48, 24, 48)}
REFUSALS = (
    "negative-lower",
    "negative-upper",
    "positive-lower",
    "positive-upper",
    "missing-Li",
    "missing-QW",
    "Faraday-inconsistent",
    "voltage-face",
    "zero-frequency",
    "nan-frequency",
)
NAMES = (
    "baseline",
    "low-ionic-conductivity",
    "low-solid-conductivity",
    "asymmetric-geometry",
    "high-conductivity",
    "SOC-temperature-electrolyte-kinetics",
    "flat-OCV",
    "domain-refusals",
)


def case_data(index):
    if index not in range(1, 9):
        raise ValueError("fixed NT EIS contains N01-N08")
    d = deepcopy(BASE)
    d.update(id=f"N{index:02d}", name=NAMES[index - 1], waveform=((1.0, 0.0, "A"),))
    if index == 2:
        d["conductivity_S_m"] = 0.15
    elif index == 3:
        d["solid_conductivity_S_m"] = (0.2, 0.1)
    elif index == 4:
        d.update(
            lengths_m=(1.4e-4, 4e-5, 8e-5), porosities=(0.35, 0.6, 0.45), bruggeman=(1.3, 1.8, 1.4)
        )
        d["electrodes"][0].update(active_volume_m3=7e-7, exchange_current_scale_A_m2=2.0)
        d["electrodes"][1].update(active_volume_m3=4e-7, exchange_current_scale_A_m2=7.0)
    elif index == 5:
        d.update(conductivity_S_m=100.0, solid_conductivity_S_m=(1e4, 1e4))
    elif index == 6:
        d.update(temperature_K=310.0, electrolyte_concentration_mol_m3=800.0)
        d["electrodes"][0].update(initial_fraction=0.45, exchange_current_scale_A_m2=2.0)
        d["electrodes"][1].update(initial_fraction=0.55, exchange_current_scale_A_m2=7.0)
    elif index == 7:
        d["electrodes"][0]["polynomial"] = (0.15,)
        d["electrodes"][1]["polynomial"] = (4.0,)
    return d


def binding(data, kind, cells, reason=None):
    return {
        "unit": data["id"],
        "inputs": json.loads(json.dumps(data)),
        "kind": kind,
        "cells": list(cells) if cells else None,
        "reason": reason,
    }


def ocv(e, derivative=False):
    x = e["initial_fraction"]
    return (
        sum(i * c * x ** (i - 1) for i, c in enumerate(e["polynomial"]) if i)
        if derivative
        else sum(c * x**i for i, c in enumerate(e["polynomial"]))
    )


def schema(cells=None):
    rows = [
        ("Z_ohm", "ohm", 0.002, 0.005),
        ("current", "A/A", 1e-8, 0.005),
        ("charge", "C/A", 1e-5, 0.005),
        ("work", "J/A", 1e-5, 0.005),
    ]
    for n in ("negative", "positive"):
        rows += [
            (n + ":amount", "mol/A", 1e-9, 0.005),
            (n + ":theta", "1/A", 5e-4, 0.005),
            (n + ":U", "V/A", 0.002, 0.005),
        ]
    if cells:
        for name, count in zip(("negative", "separator", "positive"), cells, strict=True):
            for key, size, unit, tol, rel in (
                ("phi_e", count, "V/A", 0.002, 0.005),
                ("ie", count + 1, "A/A", 0.002, 0.005),
            ):
                rows += [(f"{name}:{key}:{i}", unit, tol, rel) for i in range(size)]
            if name != "separator":
                for key, size, unit, tol, rel in (
                    ("phi_s", count, "V/A", 0.002, 0.005),
                    ("eta", count, "V/A", 0.002, 0.005),
                    ("j", count, "A/m^2/A", 0.02, 0.01),
                    ("q", count, "A/A", 0.0005, 0.005),
                    ("is", count + 1, "A/A", 0.002, 0.005),
                ):
                    rows += [(f"{name}:{key}:{i}", unit, tol, rel) for i in range(size)]
    return rows


def response_order(cells=None):
    return [r[0] for r in schema(cells)]


def response_units(cells=None):
    return [r[1] for r in schema(cells)]


def route_plan(index):
    if index < 8:
        return [(f"{k}_{g}", k, g) for k in ("native", "reference") for g in GRIDS] + [
            ("analytic", "analytic", None)
        ]
    return [(n, "refusal", "coarse") for n in REFUSALS]


def refusal_input(data, name, cells):
    from chemworld.physchem.battery_newman_tobias import newman_tobias_readout

    d = deepcopy(data)
    if name.split("-")[0] in ("negative", "positive"):
        for e in d["electrodes"]:
            e["initial_fraction"] = 0.4 if name.endswith("lower") else 0.6
    spec = native_spec(d, cells)
    state = spec.initial_state()
    amounts, coords = dict(state.amounts_mol), dict(state.coordinates)
    freq = [0.01]
    if name in REFUSALS[:4]:
        electrode, face = name.split("-")
        p = getattr(spec, electrode)
        theta = spec.surface_margin if face == "lower" else 1 - spec.surface_margin
        change = p.capacity_mol * theta - amounts[electrode + ":Li"]
        other = "positive" if electrode == "negative" else "negative"
        amounts[electrode + ":Li"] = p.capacity_mol * theta
        amounts[other + ":Li"] -= change
        coords["discharged_charge_C"] = -F * change if electrode == "negative" else F * change
        state = replace(state, amounts_mol=amounts, coordinates=coords)
    elif name == "missing-Li":
        del amounts["negative:Li"]
        state = replace(state, amounts_mol=amounts)
    elif name == "missing-QW":
        state = replace(state, coordinates={})
    elif name == "Faraday-inconsistent":
        amounts["negative:Li"] *= 0.9
        state = replace(state, amounts_mol=amounts)
    elif name == "voltage-face":
        spec = replace(spec, minimum_voltage_V=newman_tobias_readout(spec, state, 0.0).voltage_V)
    elif name == "zero-frequency":
        freq = [0.0]
    elif name == "nan-frequency":
        freq = [float("nan")]
    else:
        raise ValueError("unknown fixed refusal")
    return spec, state, freq
