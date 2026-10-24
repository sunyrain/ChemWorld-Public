"""Fixed Xu2019-derived inputs; native and reference consume declaration data only."""

from copy import deepcopy
from math import exp, sqrt

import numpy as np

F = 96485.33212
GRIDS = {"native": (40, 80), "reference": (120, 200)}
RTOL, ATOL = 1e-9, 1e-11
POLYNOMIAL = (4.3452, -1.6518, 1.6225, -2.0843, 3.5146, -2.2166)
EXPONENTIAL = (-0.5623e-4, 109.451, 100.006 / 109.451)
AREA = .01 * .0154
BASE = {
    "area_m2": AREA, "thickness_m": .0007, "conductivity_S_m": 10776000.,
    "partial_molar_volume_m3_mol": 1.3e-5,
    "exchange_current_density_A_m2": 3.5e-8 * F * (1 / 1.3e-5)**.7 * 1000.**.3,
    "reference_temperature_K": 298.15, "temperature_K": 298.15,
    "electrolyte_concentration_mol_m3": 1000.,
    "radius_m": 5.3e-6, "diffusivity_m2_s": 1e-14,
    "maximum_concentration_mol_m3": 48230.,
    "active_volume_m3": AREA * 4.2e-5 * .518,
    "exchange_current_scale_A_m2": 5.76e-11 * F * sqrt(1000.) * 48230.,
    "initial_positive_fraction": .35, "surface_margin": .01,
    "minimum_voltage_V": 2.5, "maximum_voltage_V": 4.5,
    "maximum_current_A": .006, "samples_per_segment": 41,
    "waveform": ((600., .0024, "A"),), "expected": "complete",
}
UNITS = (
    {"name": "Xu2019-low-current", "initial_positive_fraction": 4631 / 48230,
     "waveform": ((600., .0008, "A"),)},
    {"name": "positive-discharge"},
    {"name": "negative-charge", "initial_positive_fraction": .65,
     "waveform": ((600., -.0024, "A"),)},
    {"name": "rest-and-reversal", "initial_positive_fraction": .5,
     "waveform": ((120., .0024, "A"), (120., 0., "A"),
                  (120., -.0018, "A"), (60., 0., "A"))},
    {"name": "reduced-kinetics", "initial_positive_fraction": .5,
     "exchange_current_density_A_m2": .7, "exchange_current_scale_A_m2": 1.},
    {"name": "synthetic-metal-resistance", "conductivity_S_m": .2},
    {"name": "interior-CV", "initial_positive_fraction": .5,
     "waveform": ((300., 3.8, "V"),), "control_expectation": "interior"},
    {"name": "current-limited-CV", "initial_positive_fraction": .5,
     "maximum_current_A": .0012, "waveform": ((300., 3.6, "V"),),
     "control_expectation": "positive_limit"},
    {"name": "finite-metal-depletion", "thickness_m": 2e-7,
     "waveform": ((300., .0024, "A"),), "expected": "metal_event"},
    {"name": "working-surface-saturation", "initial_positive_fraction": .8,
     "diffusivity_m2_s": 1e-12, "waveform": ((1200., .0036, "A"),),
     "expected": "surface_event"},
    {"name": "voltage-cutoff", "initial_positive_fraction": .5,
     "minimum_voltage_V": 3.75, "waveform": ((1200., .0024, "A"),),
     "expected": "voltage_event"},
    {"name": "domain-refusals", "expected": "refusals"},
)
REFUSALS = ("zero-area", "negative-conductivity", "temperature-mismatch",
            "concentration-mismatch", "missing-source-row", "EIS", "SOH", "energy")
RESPONSES = ("voltage_V", "current_A", "working_surface_fraction",
             "working_average_fraction", "metal_amount_mol", "charge_C",
             "metal_overpotential_V", "working_overpotential_V", "metal_ohmic_V")
RESPONSE_ATOL = np.array([1e-3, 2e-6, 3e-3, 3e-3, 2e-9, 1e-4, 1e-4, 1e-3, 1e-4])


def case_data(index):
    if index not in range(1, 13):
        raise ValueError("half-cell block fixes exactly HC01-HC12")
    return {**deepcopy(BASE), **deepcopy(UNITS[index - 1])}


def initial_metal(data):
    return data["area_m2"] * data["thickness_m"] / data["partial_molar_volume_m3_mol"]


def working_capacity(data):
    return data["active_volume_m3"] * data["maximum_concentration_mol_m3"]


def net_ocv(theta):
    a, k, shift = EXPONENTIAL
    return sum(c * theta**i for i, c in enumerate(POLYNOMIAL)) + a * exp(k * (theta - shift))


def source():
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    return InputSource(DataSourceProvenance(
        "halfcell-Xu2019-derived", 0,
        "PyBaMM25.4.2 Xu2019.py; explicitly authored geometry/kinetic/initial-state "
        "qualification variants, not an unmodified measured Xu2019 cell",
        "25.4.2 / fixed-halfcell-block-1", "BSD-3-Clause (source implementation)"), "synthetic")


def metal_table(data):
    from chemworld.data.scientific_inputs import (
        InputValue,
        QuantityColumn,
        ScientificRow,
        ScientificTable,
    )

    units = {"area_m2": "m^2", "thickness_m": "m", "conductivity_S_m": "S/m",
        "partial_molar_volume_m3_mol": "m^3/mol", "exchange_current_density_A_m2": "A/m^2",
        "reference_temperature_K": "K", "electrolyte_concentration_mol_m3": "mol/m^3"}
    return ScientificTable("halfcell-metal", "parameters",
        tuple(QuantityColumn(k, u, "parameter", (-1., 1e12), "Explicit fixed input")
              for k, u in units.items()), (source(),),
        (ScientificRow("metal", source().provenance.source_id, "halfcell", "metal", "parameters",
            {k: InputValue(data[k], u, None, "Fixed qualification input; no measured uncertainty")
             for k, u in units.items()}),))


def native_spec(data, cells):
    from chemworld.physchem.battery_half_cell import (
        BatteryHalfCellSPMSpec,
        LithiumMetalCounterElectrodeSpec,
    )
    from chemworld.physchem.battery_materials import ElectrodeOCV
    from chemworld.physchem.battery_models import ParticleSpec

    particle = ParticleSpec(**{k: data[k] for k in (
        "radius_m", "diffusivity_m2_s", "maximum_concentration_mol_m3", "active_volume_m3",
        "exchange_current_scale_A_m2")}, cells=cells, mesh_exponent=1.)
    curve = ElectrodeOCV("Xu2019-derived-NMC", source(), POLYNOMIAL,
        exponential_terms=(EXPONENTIAL,), supported_models=("SPM",),
        stoichiometry_domain=(data["surface_margin"], 1 - data["surface_margin"]))
    return BatteryHalfCellSPMSpec(
        metal=LithiumMetalCounterElectrodeSpec(metal_table(data), "metal"),
        positive=particle, positive_ocv=curve,
        **{k: data[k] for k in ("initial_positive_fraction", "temperature_K",
            "electrolyte_concentration_mol_m3", "surface_margin", "minimum_voltage_V",
            "maximum_voltage_V", "maximum_current_A")},
        provenance="Xu2019-derived fixed-geometry finite-metal development qualification")
