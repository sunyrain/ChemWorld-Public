"""Prospective 10-unit/50-route finite-metal rest-EIS declarations."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
from battery_half_cell_cases import BASE, EXPONENTIAL, POLYNOMIAL, initial_metal, native_spec

F, R = 96485.33212, 8.314462618
FREQUENCIES = np.geomspace(1e-5, 1e-2, 17)
GRIDS = {"coarse": 80, "fine": 160}
RESPONSES = (
    "Z_ohm",
    "current_A_A",
    "surface_A_inv",
    "average_A_inv",
    "positive_mol_A",
    "metal_mol_A",
    "charge_C_A",
    "metal_eta_V_A",
    "working_eta_V_A",
    "metal_ohmic_V_A",
)
RESPONSE_UNITS = ("ohm", "A/A", "1/A", "1/A", "mol/A", "mol/A", "C/A", "V/A", "V/A", "V/A")
ATOL = np.array([0.002, 1e-8, 0.0005, 0.0005, 1e-9, 1e-9, 1e-5, 0.002, 0.002, 1e-8])
RTOL = 0.005
REFUSALS = (
    "empty-metal",
    "working-lower-face",
    "working-upper-face",
    "voltage-face",
    "missing-metal",
    "missing-QW",
    "Faraday-inconsistent",
    "zero-frequency",
)
NAMES = (
    "baseline",
    "another-SOC",
    "slow-diffusion",
    "slow-metal-kinetics",
    "slow-working-kinetics",
    "metal-resistance",
    "radius-volume",
    "finite-reservoir",
    "radial-gradient",
    "domain-refusals",
)


def case_data(index):
    if index not in range(1, 11):
        raise ValueError("half-cell EIS fixes H01-H10")
    data = deepcopy(BASE)
    data.update(id=f"H{index:02d}", name=NAMES[index - 1], waveform=((1.0, 0.0, "A"),))
    if index == 2:
        data["initial_positive_fraction"] = 0.65
    elif index == 3:
        data["diffusivity_m2_s"] = 2e-15
    elif index == 4:
        data["exchange_current_density_A_m2"] *= 0.25
    elif index == 5:
        data["exchange_current_scale_A_m2"] *= 0.25
    elif index == 6:
        data["conductivity_S_m"] = 0.2
    elif index == 7:
        data["radius_m"] = 8e-6
        data["active_volume_m3"] *= 1.5
    elif index == 8:
        data["partial_molar_volume_m3_mol"] *= 2
    return data


def binding(data, kind, cells, reason=None):
    return {
        "unit": data["id"],
        "inputs": json.loads(json.dumps(data)),
        "kind": kind,
        "cells": cells,
        "reason": reason,
    }


def response_order(cells=0):
    return [*RESPONSES, *(f"positive:profile:{i}:A_inv" for i in range(cells))]


def response_units(cells=0):
    return [*RESPONSE_UNITS, *(["1/A"] * cells)]


def ocv(theta, derivative=False):
    a, k, shift = EXPONENTIAL
    if derivative:
        return sum(
            i * c * theta ** (i - 1) for i, c in enumerate(POLYNOMIAL) if i
        ) + a * k * np.exp(k * (theta - shift))
    return sum(c * theta**i for i, c in enumerate(POLYNOMIAL)) + a * np.exp(k * (theta - shift))


def gradient_profile(cells, theta):
    edges = np.linspace(0.0, 1.0, cells + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    weights = np.diff(edges**3)
    return theta + 0.02 * (centers**2 - np.dot(weights, centers**2) / sum(weights))


def native_state(spec, *, gradient=False):
    state = spec.initial_state()
    if not gradient:
        return state
    p = spec.positive
    amounts = dict(state.amounts_mol)
    amounts.update(
        zip(
            p.keys("positive"),
            gradient_profile(p.cells, spec.initial_positive_fraction)
            * p.mesh.cell_volumes_m3
            * p.maximum_concentration_mol_m3,
            strict=True,
        )
    )
    return replace(state, amounts_mol=amounts)


def refusal_input(data, name, cells):
    """Authored physically consistent boundaries; no analysis call here."""
    from chemworld.physchem.battery_half_cell import half_cell_readout

    inputs = deepcopy(data)
    if name == "empty-metal":
        inputs["thickness_m"] = 2e-7
    spec = native_spec(inputs, cells)
    state = spec.initial_state()
    frequencies = [0.01]
    amounts, coordinates = dict(state.amounts_mol), dict(state.coordinates)
    if name in ("empty-metal", "working-lower-face", "working-upper-face"):
        p = spec.positive
        transferred = (
            initial_metal(inputs)
            if name == "empty-metal"
            else p.capacity_mol
            * (
                (spec.surface_margin if name == "working-lower-face" else 1 - spec.surface_margin)
                - spec.initial_positive_fraction
            )
        )
        theta = (
            spec.initial_positive_fraction + transferred / p.capacity_mol
            if name == "empty-metal"
            else spec.surface_margin
            if name == "working-lower-face"
            else 1 - spec.surface_margin
        )
        amounts["metal:Li"] = initial_metal(inputs) - transferred
        amounts.update(
            zip(
                p.keys("positive"),
                p.mesh.cell_volumes_m3 * p.maximum_concentration_mol_m3 * theta,
                strict=True,
            )
        )
        coordinates["discharged_charge_C"] = F * transferred
        state = replace(state, amounts_mol=amounts, coordinates=coordinates)
    elif name == "voltage-face":
        spec = replace(spec, minimum_voltage_V=half_cell_readout(spec, state, 0.0).voltage_V)
    elif name == "missing-metal":
        del amounts["metal:Li"]
        state = replace(state, amounts_mol=amounts)
    elif name == "missing-QW":
        state = replace(state, coordinates={})
    elif name == "Faraday-inconsistent":
        amounts["metal:Li"] *= 0.9
        state = replace(state, amounts_mol=amounts)
    elif name == "zero-frequency":
        frequencies = [0.0]
    else:
        raise ValueError("unknown fixed refusal")
    return spec, state, frequencies


def route_plan(index):
    if index <= 8:
        return [
            (f"{kind}_{grid}", kind, grid) for kind in ("native", "reference") for grid in GRIDS
        ] + [("analytic", "analytic", None)]
    if index == 9:
        return [(kind, kind, "coarse") for kind in ("native", "reference")]
    return [(name, "refusal", "coarse") for name in REFUSALS]
