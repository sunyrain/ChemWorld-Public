"""Prospectively fixed synthetic volume-consistent LOQS inputs; no production math."""

from copy import deepcopy

import numpy as np

F = 96485.33212
SOURCE_F = 96485.33212331001
R = 8.31446261815324
PRECISIONS = {"coarse": {"rtol": 1e-8, "atol": 1e-10}, "fine": {"rtol": 1e-10, "atol": 1e-12}}
SOURCE_ROOT_TOL = 1e-10
KEYS = (
    "negative:Pb",
    "negative:PbSO4",
    "positive:PbO2",
    "positive:PbSO4",
    "electrolyte:H2SO4",
    "electrolyte:H2O",
)
STOICHIOMETRY = np.array([-1.0, 1.0, -1.0, 1.0, -2.0, 2.0])
REFUSALS = (
    "incompatible-molar-volumes",
    "incompatible-initial-water",
    "negative-inventory",
    "invalid-reaction-area",
    "thermal",
    "chemical-energy",
    "EIS",
    "SOH",
)
NAMES = (
    "rest",
    "forward-CC",
    "reverse-CC",
    "temperature-kinetics-sequence",
    "bidirectional-interior-CV",
    "both-current-limits-CV",
    "acid-threshold",
    "finite-lead",
    "finite-lead-dioxide",
    "finite-negative-sulfate",
    "negative-porosity-threshold",
    "voltage-cutoff",
    "explicit-refusals",
)
RESPONSES = (
    "voltage_V",
    "current_A",
    "charge_C",
    "work_J",
    "acid_concentration_mol_m3",
    "water_concentration_mol_m3",
    "molality_mol_kg",
    "negative_porosity",
    "separator_porosity",
    "positive_porosity",
    "negative_ocp_V",
    "positive_ocp_V",
    "negative_overpotential_V",
    "positive_overpotential_V",
    "negative_j0_A_m2",
    "positive_j0_A_m2",
)
RESPONSE_ATOL = np.array(
    [1e-5, 1e-5, 1e-3, 3e-3, 1e-3, 1e-2, 2e-6, 2e-7, 2e-7, 2e-7, 1e-5, 1e-5, 1e-5, 1e-5, 1e-7, 1e-7]
)
RESPONSE_RTOL = np.full(len(RESPONSES), 2e-6)


def base_data():
    data = {
        "area_m2": 0.01,
        "lengths_m": (0.001, 0.001, 0.001),
        "inactive_solid_volumes_m3": (0.0, 0.0),
        "initial_porosity": (0.5, 0.6, 0.5),
        "reaction_area_densities_m_inv": (2e5, 2e6),
        "molar_volumes_m3_mol": (1.8e-5, 2.5e-5, 4.8e-5, 4.4e-5, 1.75e-5),
        "reference_acid_concentration_mol_m3": 4000.0,
        "negative_exchange_current_density_A_m2": 0.06,
        "positive_exchange_current_density_A_m2": 0.004,
        "water_molar_mass_kg_mol": 0.01801,
        "temperature_K": 298.15,
        "minimum_acid_concentration_mol_m3": 100.0,
        "minimum_water_concentration_mol_m3": 1.0,
        "minimum_porosity": 0.01,
        "maximum_porosity": 0.99,
        "minimum_voltage_V": 0.5,
        "maximum_voltage_V": 3.0,
        "maximum_current_A": 2.0,
        "samples_per_segment": 31,
        "expected": "complete",
        "source_stop_semantics": "actual_source_completion",
        "provenance": "Declared synthetic closed saturated-volume LOQS qualification; "
        "Bode1977 OCP and Sulzer2019 rate form, not default Sulzer material",
    }
    set_initial(data)
    return data


def set_initial(
    data, *, negative_sulfate=0.02, positive_sulfate=0.02, lead=None, lead_dioxide=None
):
    vp, vo, vs, ve, vw = data["molar_volumes_m3_mol"]
    volumes = np.asarray(data["lengths_m"]) * data["area_m2"]
    eps = data["initial_porosity"]
    inactive = list(data["inactive_solid_volumes_m3"])
    if lead is None:
        lead = (volumes[0] * (1 - eps[0]) - inactive[0] - vs * negative_sulfate) / vp
    else:
        inactive[0] = volumes[0] * (1 - eps[0]) - vp * lead - vs * negative_sulfate
    if lead_dioxide is None:
        lead_dioxide = (volumes[2] * (1 - eps[2]) - inactive[1] - vs * positive_sulfate) / vo
    else:
        inactive[1] = volumes[2] * (1 - eps[2]) - vo * lead_dioxide - vs * positive_sulfate
    pore = float(np.dot(volumes, eps))
    acid = pore * data["reference_acid_concentration_mol_m3"]
    water = (pore - ve * acid) / vw
    data["inactive_solid_volumes_m3"] = tuple(inactive)
    data["initial_amounts_mol"] = (
        lead,
        negative_sulfate,
        lead_dioxide,
        positive_sulfate,
        acid,
        water,
    )


def case_data(index):
    from battery_lead_acid_oracle import fields, stopping_time

    data = base_data()
    data["id"], data["name"] = f"LA{index:02d}", NAMES[index - 1]
    data["waveform"] = ((1200.0, 0.5, "A"),)
    if index == 1:
        data["waveform"] = ((1200.0, 0.0, "A"),)
    elif index == 3:
        data["waveform"] = ((1200.0, -0.25, "A"),)
    elif index == 4:
        data.update(
            temperature_K=310.0,
            negative_exchange_current_density_A_m2=0.04,
            positive_exchange_current_density_A_m2=0.006,
            reaction_area_densities_m_inv=(1.4e5, 1.5e6),
        )
        data["waveform"] = ((600.0, 0.5, "A"), (300.0, 0.0, "A"), (600.0, -0.5, "A"))
    elif index in (5, 6):
        equilibrium = fields(data, 0.0, 0.0)["voltage_V"]
        offset = 0.05 if index == 5 else 0.4
        data["waveform"] = ((600.0, equilibrium - offset, "V"), (600.0, equilibrium + offset, "V"))
        data["control_expectation"] = "interior" if index == 5 else "both_limits"
        if index == 6:
            data["maximum_current_A"] = 0.2
    elif index == 7:
        data.update(minimum_acid_concentration_mol_m3=500.0, expected="acid-threshold")
    elif index == 8:
        set_initial(data, lead=0.004)
        data["expected"] = "negative:Pb:depleted"
    elif index == 9:
        set_initial(data, lead_dioxide=0.004)
        data["expected"] = "positive:PbO2:depleted"
    elif index == 10:
        set_initial(data, negative_sulfate=0.003)
        data.update(expected="negative:PbSO4:depleted", waveform=((1200.0, -0.25, "A"),))
    elif index == 11:
        data["initial_porosity"] = (0.015, 0.6, 0.5)
        set_initial(data)
        data["expected"] = "negative:minimum-porosity"
    elif index == 12:
        data.update(
            minimum_voltage_V=1.95, expected="minimum-voltage", waveform=((6000.0, 0.5, "A"),)
        )
        data["source_stop_semantics"] = "actual_source_minimum_voltage_event"
    elif index == 13:
        data["expected"] = "refusals"
    if 7 <= index <= 12:
        time = stopping_time(data, data["waveform"][0][1])
        data["analytic_stop_time_s"] = time
        if index <= 11:
            data["waveform"] = ((1.1 * time, data["waveform"][0][1], "A"),)
            data["source_stop_semantics"] = (
                "author_imposed_analytic_valid_horizon_no_source_inventory_event"
            )
        data["observation_horizons_s"] = (time,)
    else:
        data["observation_horizons_s"] = tuple(segment[0] for segment in data["waveform"])
    return deepcopy(data)


def native_spec(data):
    from chemworld.physchem.battery_lead_acid import (
        BatteryLeadAcidSpec,
        LeadAcidGeometrySpec,
        LeadAcidMaterialSpec,
    )

    geometry = LeadAcidGeometrySpec(
        area_m2=data["area_m2"],
        lengths_m=data["lengths_m"],
        inactive_solid_volumes_m3=data["inactive_solid_volumes_m3"],
        separator_porosity=data["initial_porosity"][1],
        reaction_area_densities_m_inv=data["reaction_area_densities_m_inv"],
    )
    material = LeadAcidMaterialSpec(
        **{
            key: data[key]
            for key in (
                "molar_volumes_m3_mol",
                "reference_acid_concentration_mol_m3",
                "negative_exchange_current_density_A_m2",
                "positive_exchange_current_density_A_m2",
                "water_molar_mass_kg_mol",
                "provenance",
            )
        }
    )
    return BatteryLeadAcidSpec(
        geometry=geometry,
        material=material,
        **{
            key: data[key]
            for key in (
                "initial_amounts_mol",
                "temperature_K",
                "minimum_acid_concentration_mol_m3",
                "minimum_water_concentration_mol_m3",
                "minimum_porosity",
                "maximum_porosity",
                "minimum_voltage_V",
                "maximum_voltage_V",
                "maximum_current_A",
                "provenance",
            )
        },
    )
