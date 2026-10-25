"""Fixed authored inputs for the finite spatial composite-DFN development block."""

from copy import deepcopy

import numpy as np

F = 96485.33212
RTOL = 1e-8
ATOL = 1e-10
GRIDS = {
    "coarse": {"x": (6, 4, 6), "r": 20},
    "fine": {"x": (12, 8, 12), "r": 40},
}
OBSERVATION_COORDINATES = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
PHASES = ("host_a", "host_b", "cathode")
BASE = {
    "area_m2": 0.01,
    "lengths_m": (1e-4, 2.5e-5, 1e-4),
    "porosities": (0.4, 0.5, 0.4),
    "bruggeman": (1.5, 1.5, 1.5),
    "diffusivity_m2_s": 2e-10,
    "conductivity_S_m": 1.0,
    "solid_conductivity_S_m": (100.0, 100.0),
    "cation_transference": 0.4,
    "thermodynamic_factor": 1.0,
    "initial_concentration_mol_m3": 1000.0,
    "minimum_concentration_mol_m3": 1.0,
    "temperature_K": 298.15,
    "surface_margin": 0.01,
    "minimum_voltage_V": 2.5,
    "maximum_voltage_V": 4.5,
    "maximum_current_A": 2.0,
    "series_resistance_ohm": 0.0,
    "samples_per_segment": 31,
    "waveform": ((120.0, 0.3, "A"),),
    "expected": "complete",
    "phases": [
        {
            "phase_id": "host_a",
            "electrode": "negative",
            "radius_m": 5e-6,
            "diffusivity_m2_s": 1e-13,
            "maximum_concentration_mol_m3": 30000.0,
            "active_volume_m3": 4e-7,
            "exchange_current_scale_A_m2": 4.0,
            "initial_fraction": 0.6,
            "polynomial": (0.3, -0.25),
        },
        {
            "phase_id": "host_b",
            "electrode": "negative",
            "radius_m": 5e-6,
            "diffusivity_m2_s": 1e-13,
            "maximum_concentration_mol_m3": 30000.0,
            "active_volume_m3": 2e-7,
            "exchange_current_scale_A_m2": 4.0,
            "initial_fraction": 0.6,
            "polynomial": (0.38, -0.35),
        },
        {
            "phase_id": "cathode",
            "electrode": "positive",
            "radius_m": 5e-6,
            "diffusivity_m2_s": 1e-13,
            "maximum_concentration_mol_m3": 30000.0,
            "active_volume_m3": 6e-7,
            "exchange_current_scale_A_m2": 4.0,
            "initial_fraction": 0.35,
            "polynomial": (4.2, -0.8),
        },
    ],
}
NAMES = (
    "baseline",
    "low-ionic-conductivity",
    "low-salt-diffusivity",
    "heterogeneous-particle-kinetics-capacity",
    "identical-phase-merged-DFN",
    "high-transport-composite-SPM-limit",
    "zero-terminal-internal-exchange",
    "CC-rest-reverse",
    "reverse-CC",
    "interior-CV",
    "limited-CV",
    "individual-phase-cutoff",
    "voltage-cutoff",
    "domain-refusals",
)
REFUSALS = (
    "duplicate-phase-id",
    "wrong-topology",
    "nonmonotone-ocv",
    "wrong-ocv-family",
    "zero-particle-diffusivity",
    "zero-electrolyte-conductivity",
    "missing-particle-Li",
    "missing-salt",
    "EIS",
    "SOH",
    "chemical-energy",
    "thermal",
)


UNITS = {
    "values": ["V", "A", "C", "J"],
    "surface": "dimensionless",
    "average": "dimensionless",
    "theta": "dimensionless",
    "shell_amounts_mol": "mol",
    "q": "A",
    "j": "A/m^2",
    "j0": "A/m^2",
    "eta": "V",
    "salt_mol_m3": "mol/m^3",
    "salt_amounts_mol": "mol",
    "electrolyte_potential_V": "V",
    "solid_potential_V": "V",
    "electrolyte_face_current_A": "A",
    "solid_face_current_A": "A",
}


def case_data(index):
    if index not in range(1, 15):
        raise ValueError("fixed composite DFN units CD01-CD14")
    d = deepcopy(BASE)
    d.update(id=f"CD{index:02d}", name=NAMES[index - 1])
    a, b, p = d["phases"]
    if index == 2:
        d["conductivity_S_m"] = 0.15
    elif index == 3:
        d["diffusivity_m2_s"] = 2e-11
    elif index == 4:
        a.update(
            radius_m=4e-6,
            diffusivity_m2_s=3e-13,
            maximum_concentration_mol_m3=28000.0,
            active_volume_m3=3.5e-7,
            exchange_current_scale_A_m2=1.0,
        )
        b.update(
            radius_m=7e-6,
            diffusivity_m2_s=3e-14,
            maximum_concentration_mol_m3=36000.0,
            active_volume_m3=2.5e-7,
            exchange_current_scale_A_m2=6.0,
        )
        d["waveform"] = ((120.0, 0.25, "A"),)
    elif index == 5:
        b["polynomial"] = a["polynomial"]
        d["limit"] = "merged-dfn"
    elif index == 6:
        d.update(
            conductivity_S_m=100.0,
            diffusivity_m2_s=2e-7,
            solid_conductivity_S_m=(1e4, 1e4),
            limit="composite-spm",
        )
    elif index == 7:
        d.update(waveform=((120.0, 0.0, "A"),), internal_exchange=True)
    elif index == 8:
        d["waveform"] = ((60.0, 0.3, "A"), (30.0, 0.0, "A"), (60.0, -0.3, "A"), (30.0, 0.0, "A"))
    elif index == 9:
        a["initial_fraction"] = b["initial_fraction"] = 0.45
        p["initial_fraction"] = 0.55
        d["waveform"] = ((120.0, -0.3, "A"),)
    elif index == 10:
        d.update(waveform=((120.0, 3.74, "V"),), control_expectation="interior")
    elif index == 11:
        d.update(
            waveform=((120.0, 3.4, "V"),),
            maximum_current_A=0.2,
            control_expectation="positive_limit",
        )
    elif index == 12:
        a["polynomial"] = b["polynomial"] = (0.2,)
        a["initial_fraction"] = 0.03
        for phase in d["phases"]:
            phase["diffusivity_m2_s"] = 1e-12
        d.update(waveform=((300.0, 0.6, "A"),), expected="phase:host_a:depleted")
    elif index == 13:
        d.update(minimum_voltage_V=3.55, waveform=((600.0, 0.6, "A"),), expected="minimum-voltage")
    elif index == 14:
        d["expected"] = "refusals"
    return d


def route_plan(index):
    if index == 14:
        return [(name, "refusal", "coarse", None) for name in REFUSALS]
    routes = [
        (f"{kind}_{grid}", kind, grid, None) for kind in ("native", "reference") for grid in GRIDS
    ]
    limit = case_data(index).get("limit")
    if limit:
        routes.extend(
            (f"{kind}_{grid}_{limit}", kind, grid, limit)
            for kind in ("native", "reference")
            for grid in GRIDS
        )
    return routes


def declared_phases(data, limit=None):
    phases = deepcopy(data["phases"])
    if limit == "merged-dfn":
        if data.get("limit") != limit:
            raise ValueError("merged DFN is declared only for CD05")
        phases[0]["active_volume_m3"] += phases[1]["active_volume_m3"]
        phases = [phases[0], phases[2]]
    elif limit not in (None, "composite-spm"):
        raise ValueError("undeclared old-family limit")
    return phases


def source():
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    return InputSource(
        DataSourceProvenance(
            "composite-dfn-synthetic",
            0,
            "Explicit synthetic two-negative/one-positive material, geometry and transport inputs; "
            "actual PyBaMM25.4.2 modular DFN comparison, not Chen composite experimental material",
            "fixed-composite-dfn-block-1",
            "authored qualification inputs",
        ),
        "synthetic",
    )


def native_spec(data, grid, limit=None):
    from chemworld.physchem.battery_composite import (
        ActiveMaterialPhaseSpec,
        BatteryCompositeSPMSpec,
    )
    from chemworld.physchem.battery_dfn import BatteryDFNSpec
    from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
    from chemworld.physchem.battery_materials import ElectrodeOCV
    from chemworld.physchem.battery_models import ParticleSpec

    phases = []
    for p in declared_phases(data, limit):
        particle = ParticleSpec(
            **{
                k: p[k]
                for k in (
                    "radius_m",
                    "diffusivity_m2_s",
                    "maximum_concentration_mol_m3",
                    "active_volume_m3",
                    "exchange_current_scale_A_m2",
                )
            },
            cells=grid["r"],
            mesh_exponent=1.0,
        )
        curve = ElectrodeOCV(
            p["phase_id"],
            source(),
            p["polynomial"],
            supported_models=("DFN", "SPM"),
            stoichiometry_domain=(0.01, 0.99),
        )
        phases.append(
            ActiveMaterialPhaseSpec(p["phase_id"], particle, curve, p["initial_fraction"], source())
        )
    kwargs = {
        k: data[k]
        for k in (
            "temperature_K",
            "surface_margin",
            "minimum_voltage_V",
            "maximum_voltage_V",
            "maximum_current_A",
            "series_resistance_ohm",
        )
    }
    kwargs["provenance"] = "Declared composite DFN synthetic development qualification"
    if limit == "composite-spm":
        return BatteryCompositeSPMSpec(tuple(phases[:2]), phases[2], **kwargs)
    electrolyte = BatteryElectrolyteSpec(
        cells=grid["x"],
        **{
            k: data[k]
            for k in (
                "area_m2",
                "lengths_m",
                "porosities",
                "bruggeman",
                "diffusivity_m2_s",
                "conductivity_S_m",
                "solid_conductivity_S_m",
                "cation_transference",
                "thermodynamic_factor",
                "initial_concentration_mol_m3",
                "minimum_concentration_mol_m3",
            )
        },
    )
    if limit == "merged-dfn":
        return BatteryDFNSpec(
            negative=phases[0].particle,
            positive=phases[1].particle,
            negative_ocv=phases[0].ocv,
            positive_ocv=phases[1].ocv,
            initial_negative_fraction=phases[0].initial_fraction,
            initial_positive_fraction=phases[1].initial_fraction,
            electrolyte=electrolyte,
            **kwargs,
        )
    from chemworld.physchem.battery_dfn_composite import BatteryCompositeDFNSpec

    return BatteryCompositeDFNSpec(tuple(phases[:2]), phases[2], electrolyte, **kwargs)
