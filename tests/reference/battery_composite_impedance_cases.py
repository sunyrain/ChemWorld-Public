"""Prospective finite composite rest-EIS inputs; no result-dependent selection."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
from battery_composite_cases import BASE
from battery_composite_cases import native_spec as original_native_spec

F, R = 96485.33212, 8.314462618
FREQUENCIES = np.geomspace(1e-5, 1e-1, 17)
GRIDS = {"coarse": 80, "fine": 160}
PHASES = ("host_a", "host_b", "cathode")
METRICS = ("q_A_A", "surface_A_inv", "average_A_inv", "amount_mol_A")
NAMES = (
    "same-material-SPM-limit",
    "equal-OCP-heterogeneous",
    "D-contrast",
    "j0-contrast",
    "volume-pivot-change",
    "phase-order-permutation",
    "series-resistance",
    "flat-OCP",
    "another-SOC-temperature",
    "radius-capacity-contrast",
    "nonstationary",
    "domain-refusals",
)
REFUSALS = (
    "empty-frequency",
    "zero-frequency",
    "negative-frequency",
    "nan-frequency",
    "missing-inventory",
    "missing-coordinates",
    "negative-inventory",
    "voltage-boundary",
)
ATOL = np.array([0.002, *([0.0005, 0.0005, 0.0005, 1e-5] * 3)])
RTOL = 0.005


def binding(data, kind, cells, single=False, reason=None):
    return {
        "unit": data["id"],
        "inputs": json.loads(json.dumps(data)),
        "kind": kind,
        "cells": cells,
        "single": single,
        "nonstationary": reason,
    }


def case_data(index):
    if index not in range(1, 13):
        raise ValueError("composite EIS fixes E01-E12")
    data = deepcopy(BASE)
    data.update(id=f"E{index:02d}", name=NAMES[index - 1], waveform=((1.0, 0.0, "A"),))
    a, b, p = data["phases"]
    b["initial_fraction"] = (0.38 - 0.15) / 0.35
    if index == 1:
        b["polynomial"], b["initial_fraction"] = a["polynomial"], a["initial_fraction"]
        data["same_material_limit"] = True
    elif index == 3:
        a["diffusivity_m2_s"], b["diffusivity_m2_s"] = 5e-13, 2e-14
    elif index == 4:
        a["exchange_current_scale_A_m2"], b["exchange_current_scale_A_m2"] = 0.8, 6.0
        p["exchange_current_scale_A_m2"] = 2.0
    elif index == 5:
        a["active_volume_m3"], b["active_volume_m3"] = 1e-7, 5e-7
    elif index == 6:
        data["phases"] = [b, a, p]
    elif index == 7:
        data["series_resistance_ohm"] = 0.15
    elif index == 8:
        a["polynomial"] = b["polynomial"] = (0.15,)
        p["polynomial"] = (3.92,)
    elif index == 9:
        data["temperature_K"] = 310.15
        a["initial_fraction"], b["initial_fraction"], p["initial_fraction"] = (
            0.4,
            (0.38 - 0.2) / 0.35,
            0.55,
        )
    elif index == 10:
        for phase, radius, cmax in zip(
            (a, b, p), (3e-6, 8e-6, 6e-6), (27000.0, 33000.0, 31000.0), strict=True
        ):
            phase.update(radius_m=radius, maximum_concentration_mol_m3=cmax)
    return data


def phase_values(data, single=False):
    if not single:
        return {p["phase_id"]: p for p in data["phases"]}
    if not data.get("same_material_limit"):
        raise ValueError("single-material source only for E01")
    a, b, p = deepcopy(data["phases"])
    a["active_volume_m3"] += b["active_volume_m3"]
    return {a["phase_id"]: a, p["phase_id"]: p}


def response_order(data, cells=0, *, single=False):
    phases = phase_values(data, single)
    identities = tuple(k for k in PHASES if k in phases)
    return [
        "Z_ohm",
        *(f"{k}:{v}" for k in identities for v in METRICS),
        *(f"{k}:profile:{i}:A_inv" for k in identities for i in range(cells)),
    ]


def native_spec(data, cells, *, single=False):
    spec = original_native_spec(data, cells)
    if not single:
        return replace(spec, provenance="45-COMPOSITE-EIS declared synthetic equal-OCP inputs")
    from chemworld.physchem.battery_models import BatterySPMSpec

    a, b = spec.negative_phases
    p = spec.positive_phase
    return BatterySPMSpec(
        negative=replace(
            a.particle, active_volume_m3=(a.particle.active_volume_m3 + b.particle.active_volume_m3)
        ),
        positive=p.particle,
        negative_ocv=a.ocv,
        positive_ocv=p.ocv,
        initial_negative_fraction=a.initial_fraction,
        initial_positive_fraction=p.initial_fraction,
        temperature_K=spec.temperature_K,
        surface_margin=spec.surface_margin,
        minimum_voltage_V=spec.minimum_voltage_V,
        maximum_voltage_V=spec.maximum_voltage_V,
        maximum_current_A=spec.maximum_current_A,
        series_resistance_ohm=spec.series_resistance_ohm,
        numerical_form="residual",
    )


def gradient_profile(cells, theta):
    edges = np.linspace(0.0, 1.0, cells + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    weights = np.diff(edges**3)
    return theta + 0.02 * (centers**2 - np.dot(weights, centers**2) / sum(weights))


def route_plan(index):
    if index <= 10:
        routes = [
            (f"{kind}_{grid}", kind, grid, False)
            for kind in ("native", "reference")
            for grid in GRIDS
        ]
        routes.append(("analytic", "analytic", None, False))
        if index == 1:
            routes.extend(
                (f"old_{kind}_{grid}", kind, grid, True)
                for kind in ("native", "reference")
                for grid in GRIDS
            )
        return routes
    if index == 11:
        return [
            (f"{kind}_{reason}", kind, "coarse", reason)
            for reason in ("internal-current", "radial-gradient")
            for kind in ("native", "reference")
        ]
    return [(name, "refusal", "coarse", False) for name in REFUSALS]
