"""Inputs for the prewritten nine-unit MPM block; no comparison is executed here."""

from dataclasses import replace

from battery_ocv_cases import released_functions, source_curve

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.battery_models import ParticleSpec
from chemworld.physchem.battery_mpm import BatteryMPMSpec
from chemworld.physchem.battery_particles import lognormal_size_distribution
from chemworld.physchem.data_conflicts import DataSourceProvenance

WAVEFORM = ((120.0, 0.25), (60.0, 0.0), (120.0, -0.2))
NATIVE_GRIDS = ((8, 64), (16, 64), (32, 64), (32, 16), (32, 32))
REFERENCE_GRIDS = ((40, 96), (80, 160))
IMPEDANCE_REFERENCE_GRIDS = ((40, 128), (80, 256))
FREQUENCIES_HZ = (1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.05, 0.1)


def distribution(electrode, narrow, nodes):
    source = InputSource(
        DataSourceProvenance(
            "synthetic-mpm-qualification",
            0,
            "Self-authored declared truncated lognormal fixture; not measured size data",
            "1",
            "CC0-1.0",
        ),
        "synthetic",
    )
    if narrow:
        median, sigma, low, high = 10e-6, 0.12, 5e-6, 17e-6
    elif electrode == "negative":
        median, sigma, low, high = 10e-6, 0.4, 2.5e-6, 25e-6
    else:
        median, sigma, low, high = 8e-6, 0.3, 2e-6, 20e-6
    return lognormal_size_distribution(
        median_radius_m=median,
        log_standard_deviation=sigma,
        lower_radius_m=low,
        upper_radius_m=high,
        nodes=nodes,
        source=source,
        basis="area",
    )


def case(index, *, size_nodes=32, radial_cells=64):
    if type(index) is not int or not 1 <= index <= 9:
        raise ValueError("the fixed MPM block contains G01 through G09")
    options = {
        "negative": ParticleSpec(cells=radial_cells), "positive": ParticleSpec(cells=radial_cells)
    }
    if index != 1:
        options.update(
            negative_distribution=distribution("negative", index == 2, size_nodes),
            positive_distribution=distribution("positive", index == 2, size_nodes),
        )
    if index == 4:
        options.update(
            negative=replace(
                options["negative"], diffusivity_m2_s=3e-15, exchange_current_scale_A_m2=1.5
            ),
            positive=replace(
                options["positive"], diffusivity_m2_s=2e-14, exchange_current_scale_A_m2=0.7
            ),
            initial_negative_fraction=0.8,
            initial_positive_fraction=0.2,
        )
    if index == 5:
        for name in ("negative", "positive"):
            curve = source_curve(name)[0]
            options[name + "_ocv"] = replace(
                curve, supported_models=(*curve.supported_models, "MPM")
            )
        options.update(initial_negative_fraction=0.8, initial_positive_fraction=0.3)
    if index == 6:
        options["maximum_current_A"] = 0.4
    if index == 7:
        options.update(
            initial_negative_fraction=0.1,
            initial_positive_fraction=0.5,
            minimum_voltage_V=2.0,
            maximum_voltage_V=5.0,
        )
    return BatteryMPMSpec(**options)


def ocp_functions(index, *, compensate_barrier=False):
    functions = (
        released_functions()
        if index == 5
        else {"negative": lambda x: 0.2 - 0.1 * x, "positive": lambda y: 4.2 - 0.8 * y}
    )
    if compensate_barrier:
        # G01 uses the released SPM adapter, which deliberately retains its old
        # default semantics. The MPM path already cancels the artificial term.
        return {
            name: (lambda theta, law=law: law(theta) - 1e-6 * (1 / theta + 1 / (theta - 1)))
            for name, law in functions.items()
        }
    return functions


def independent_area_pdfs(spec):
    import pybamm

    result = {}
    for name in ("negative", "positive"):
        distribution = getattr(spec, name + "_distribution")
        if distribution.source_basis != "area":
            raise ValueError("this released-model block fixes an area-PDF input")
        parameters = dict(distribution.distribution_parameters)
        median = parameters["median_radius"].to("m").value
        sigma = parameters["log_standard_deviation"].to("dimensionless").value
        result[name] = lambda radius, median=median, sigma=sigma: (
            pybamm.exp(-(pybamm.log(radius / median) ** 2) / (2 * sigma**2)) / radius
        )
    return result
