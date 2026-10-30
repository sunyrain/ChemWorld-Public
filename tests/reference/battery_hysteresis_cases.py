"""Authored fixed synthetic inputs; no native OCP evaluator is reused by references."""

from copy import deepcopy

import numpy as np

FREQUENCIES = np.geomspace(1e-4, 1e2, 13)
SPM_GRIDS = {"native": (32, 64), "reference": (96, 160)}
MPM_GRIDS = {"native": ((16, 32), (32, 64)), "reference": ((40, 96), (80, 160))}
WAVEFORM = ((120., .25, "A"), (60., 0., "A"), (120., -.2, "A"))
BASE = {
    "negative_mode": "Axen", "positive_mode": "Axen", "family": "SPM",
    "maximum_concentration_mol_m3": 30000., "active_volume_m3": 1e-6,
    "radius_m": 1e-5, "diffusivity_m2_s": 1e-13, "exchange_current_scale_A_m2": 1.,
    "initial_negative_fraction": .7, "initial_positive_fraction": .3,
    "temperature_K": 298.15, "surface_margin": .01,
    "minimum_voltage_V": 2.5, "maximum_voltage_V": 4.5, "maximum_current_A": 2.,
    "negative_equilibrium_polynomial_V": (.2, -.1),
    "positive_equilibrium_polynomial_V": (4.2, -.8),
    "negative_gap_V": .08, "positive_gap_V": .12,
    "negative_initial_h": .2, "positive_initial_h": -.3,
    "lithiation_decay": 20., "delithiation_decay": 40.,
    "current_density_scale_m2_A": 100., "effective_collector_area_m2": .02,
    "samples_per_segment": 41, "waveform": WAVEFORM,
    "expected": "trajectory", "size_distribution": "single",
}


def variant(name, **changes):
    return {"name": name, **changes}


UNITS = (
    (variant("single-compatibility", negative_mode="single", positive_mode="single"),),
    (variant("sigmoid-direction", negative_mode="current sigmoid",
             positive_mode="current sigmoid"),),
    (variant("sigmoid-zero-sweep", negative_mode="current sigmoid",
             positive_mode="current sigmoid", expected="static_sweep"),),
    (variant("small-area", negative_mode="current sigmoid", positive_mode="current sigmoid",
             effective_collector_area_m2=.01, expected="static_sweep"),
     variant("large-area", negative_mode="current sigmoid", positive_mode="current sigmoid",
             effective_collector_area_m2=.04, expected="static_sweep")),
    (variant("negative-Axen", positive_mode="single"),),
    (variant("positive-Axen", negative_mode="single"),),
    (variant("two-electrode-Axen"),),
    (variant("outward-endpoints", negative_initial_h=1., positive_initial_h=-1.),
     variant("inward-endpoints", negative_initial_h=-1., positive_initial_h=1.)),
    (variant("repeated-reversal", waveform=WAVEFORM+WAVEFORM),),
    (variant("finite-source-CV", maximum_current_A=.4, waveform=((120., 3.75, "V"),)),),
    (variant("voltage-cutoff", minimum_voltage_V=3.65,
             waveform=((1000., .25, "A"),), expected="voltage_event"),),
    (variant("MPM-single-radius", family="MPM"),),
    (variant("MPM-narrow-Axen", family="MPM", size_distribution="narrow"),),
    (variant("MPM-narrow-sigmoid", family="MPM", size_distribution="narrow",
             negative_mode="current sigmoid", positive_mode="current sigmoid"),),
    (variant("smooth-small-signal", negative_mode="current sigmoid",
             positive_mode="current sigmoid", expected="frequency"),
     variant("Axen-rest-nonsmooth", expected="nonsmooth_probe")),
    tuple(variant(name, expected="domain_refusal") for name in (
        "initial-h-outside", "negative-decay", "zero-collector-area",
        "thermal-combination", "SEI-combination", "equilibrium-SOH")),
)


def case_data(index, variant_index=0):
    if index not in range(1, 17):
        raise ValueError("hysteresis block fixes exactly H01-H16")
    return {**deepcopy(BASE), **deepcopy(UNITS[index-1][variant_index])}


def sweep_currents():
    # Identical physical currents for all areas, not identical density values.
    return .02/100*np.array([-4., -2., -1., -.5, 0., .5, 1., 2., 4.])


def branch_polynomial(data, electrode, lithiation):
    a, b = data[electrode+"_equilibrium_polynomial_V"]
    return a + (-.5 if lithiation else .5)*data[electrode+"_gap_V"], b


def analytic_h(data, electrode, initial_h, current, duration_s, faraday):
    signed = current if electrode == "negative" else -current
    if signed == 0:
        return initial_h
    sign = np.sign(signed)
    k = data["delithiation_decay" if signed > 0 else "lithiation_decay"]
    capacity = faraday*data["maximum_concentration_mol_m3"]*data["active_volume_m3"]
    return float(sign+(initial_h-sign)*np.exp(-k*abs(signed)*duration_s/capacity))


def response_names(data):
    names = ["voltage_V", "current_A", "negative_surface_fraction", "positive_surface_fraction",
             "negative_average_fraction", "positive_average_fraction", "charge_C"]
    names += [electrode+"_h" for electrode in ("negative", "positive")
              if data[electrode+"_mode"] == "Axen"]
    names += ["negative_surface_ocp_V", "positive_surface_ocp_V"]
    return names


def response_atol(data):
    values = [1e-3, 1e-3, 3e-3, 3e-3, 3e-3, 3e-3, 1e-4]
    values += [1e-5 for e in ("negative", "positive") if data[e+"_mode"] == "Axen"]
    values += [1e-3, 1e-3]
    return np.asarray(values)


def physical_spec(data, radial_cells, size_cells=None):
    from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec
    from chemworld.physchem.battery_mpm import BatteryMPMSpec
    from chemworld.physchem.battery_particles import lognormal_size_distribution

    particle = ParticleSpec(**{key: data[key] for key in (
        "radius_m", "diffusivity_m2_s", "maximum_concentration_mol_m3",
        "active_volume_m3", "exchange_current_scale_A_m2")}, cells=radial_cells, mesh_exponent=1.)
    parameters = {key: data[key] for key in (
        "initial_negative_fraction", "initial_positive_fraction", "temperature_K",
        "surface_margin", "minimum_voltage_V", "maximum_voltage_V", "maximum_current_A")}
    parameters.update(negative=particle, positive=particle, provenance="synthetic-hysteresis-v1")
    for name in ("negative", "positive"):
        a, b = data[name+"_equilibrium_polynomial_V"]
        parameters.update({name+"_ocv_intercept_V": a, name+"_ocv_slope_V": b})
    if data["family"] == "SPM":
        return BatterySPMSpec(**parameters)
    if data["size_distribution"] != "single":
        # This is declaration data only; the reference uses its own PDF and FV.
        from chemworld.data.scientific_inputs import InputSource
        from chemworld.physchem.data_conflicts import DataSourceProvenance

        source = InputSource(DataSourceProvenance("synthetic-hysteresis-PSD", 0,
            "Authored truncated lognormal area PDF; not measured material data",
            "1", "CC0-1.0"), "synthetic")
        distribution = lognormal_size_distribution(median_radius_m=1e-5,
            log_standard_deviation=.12, lower_radius_m=5e-6, upper_radius_m=17e-6,
            nodes=size_cells, source=source, basis="area")
        parameters.update(negative_distribution=distribution, positive_distribution=distribution)
    return BatteryMPMSpec(**parameters)


def native_case(data, radial_cells, size_cells=None):
    from dataclasses import replace

    from battery_thevenin_cases import table

    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.battery_materials import ElectrodeOCV
    from chemworld.physchem.battery_ocp import ElectrodeOCPSpec
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    source = InputSource(DataSourceProvenance("synthetic-hysteresis-branches", 0,
        "Authored affine net OCP branches; PyBaMM25.4.2 equation qualification",
        "1", "CC0-1.0"), "synthetic")
    values = {}
    for name in ("negative", "positive"):
        mode = {"single": "single", "Axen": "axen", "current sigmoid": "current_sigmoid"}[
            data[name+"_mode"]]
        if mode == "single":
            values[name+"_ocp"] = ElectrodeOCPSpec(mode="single")
            continue
        branches = {key: ElectrodeOCV(name+"-synthetic-hysteresis", source,
            polynomial_V=branch_polynomial(data, name, lith), supported_models=("SPM", "MPM"))
            for key, lith in (("lithiation", True), ("delithiation", False))}
        if mode == "axen":
            row = {"initial_h": data[name+"_initial_h"],
                   "lithiation_decay": data["lithiation_decay"],
                   "delithiation_decay": data["delithiation_decay"]}
            units = dict.fromkeys(row, "dimensionless")
        else:
            row = {"collector_area_m2": data["effective_collector_area_m2"],
                   "sigmoid_slope_m2_A": data["current_density_scale_m2_A"]}
            units = {"collector_area_m2": "m^2", "sigmoid_slope_m2_A": "m^2/A"}
        values[name+"_ocp"] = ElectrodeOCPSpec(mode=mode, **branches,
            parameters=table(name+"-OCP-parameters", units, [row]), parameter_row_id="0")
    return replace(physical_spec(data, radial_cells, size_cells), **values)
