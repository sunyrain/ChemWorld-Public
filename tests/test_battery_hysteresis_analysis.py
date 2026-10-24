"""Operational OCP must use the real residual, not an equilibrium-analysis shortcut."""

from dataclasses import replace

import numpy as np
import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec
from chemworld.physchem.battery_mpm import BatteryMPMSpec
from chemworld.physchem.battery_ocp import ElectrodeOCPSpec
from chemworld.physchem.battery_particles import ParticleSizeDistribution
from chemworld.physchem.battery_soh import (
    ElectrodeSOHSpec,
    HalfCellSOHSpec,
    battery_spec_at_soc,
    soh_from_battery,
    solve_electrode_soh,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.residual_system import ResidualEvaluator, finite_jacobian


def source():
    return InputSource(
        DataSourceProvenance(
            "ocp-analysis-synthetic", 0, "Explicit analysis regression fixture", "1", "CC0-1.0"
        ),
        "synthetic",
    )


def branch(electrode, offset=0.0):
    intercept, slope = (0.2, -0.1) if electrode == "negative" else (4.2, -0.8)
    return ElectrodeOCV(
        electrode + "-analysis-branch",
        source(),
        (intercept + offset, slope),
        supported_models=("SPM", "MPM"),
    )


def operational(electrode, mode, *, initial_h=0.0):
    if mode == "single":
        return ElectrodeOCPSpec(mode=mode)
    values = (
        {"collector_area_m2": (0.02, "m^2"), "sigmoid_slope_m2_A": (100.0, "m^2/A")}
        if mode == "current_sigmoid"
        else {
            "initial_h": (initial_h, "dimensionless"),
            "lithiation_decay": (2.0, "dimensionless"),
            "delithiation_decay": (3.0, "dimensionless"),
        }
    )
    parameters = ScientificTable(
        "analysis-ocp-parameters",
        "parameters",
        tuple(
            QuantityColumn(key, unit, "parameter", (-1000, 1000), "Explicit synthetic input")
            for key, (_, unit) in values.items()
        ),
        (source(),),
        (
            ScientificRow(
                "row", source().provenance.source_id, "group", "sample", "parameters",
                {
                    key: InputValue(value, unit, None, "Fixed synthetic input")
                    for key, (value, unit) in values.items()
                },
            ),
        ),
    )
    gap = 0.08 if electrode == "negative" else 0.12
    return ElectrodeOCPSpec(
        mode=mode,
        lithiation=branch(electrode, -gap / 2),
        delithiation=branch(electrode, gap / 2),
        parameters=parameters,
        parameter_row_id="row",
    )


def model(family, mode="single", electrodes=("negative", "positive"), *, initial_h=0.0):
    kwargs = {
        "negative": ParticleSpec(cells=4),
        "positive": ParticleSpec(cells=4),
        "minimum_voltage_V": 3.38,
        "maximum_voltage_V": 3.92,
        "negative_ocv": branch("negative"),
        "positive_ocv": branch("positive"),
    }
    if family == "mpm":
        distribution = ParticleSizeDistribution.from_weights(
            (5e-6, 15e-6), (0.4, 0.6), basis="volume", source=source()
        )
        kwargs.update(negative_distribution=distribution, positive_distribution=distribution)
    kwargs.update(
        {name + "_ocp": operational(name, mode, initial_h=initial_h) for name in electrodes}
    )
    return (BatteryMPMSpec if family == "mpm" else BatterySPMSpec)(**kwargs)


def snapshot(state):
    return (
        dict(state.amounts_mol), dict(state.coordinates), dict(state.algebraic_values),
        state.time_s, state.temperature_K, state.volume_L,
    )


@pytest.mark.parametrize("family", ["spm", "mpm"])
@pytest.mark.parametrize("electrode", ["negative", "positive"])
@pytest.mark.parametrize("initial_h", [-1.0, 0.2, 1.0])
def test_axen_rest_rejects_before_constructing_an_arbitrary_frequency_derivative(
    family, electrode, initial_h
):
    spec = model(family, "axen", (electrode,), initial_h=initial_h)
    state = spec.initial_state()
    before = snapshot(state)
    for analyze in (
        lambda: battery_impedance_system(spec, state),
        lambda: battery_impedance(spec, [0.01, 1.0], state=state),
    ):
        with pytest.raises(ValueError, match=r"Axen.*two-sided.*derivative"):
            analyze()
        assert snapshot(state) == before


@pytest.mark.parametrize("family", ["spm", "mpm"])
@pytest.mark.parametrize("electrodes", [("negative",), ("positive",), ("negative", "positive")])
def test_sigmoid_direct_current_response_enters_same_residual_and_spectrum(family, electrodes):
    spec = model(family, "current_sigmoid", electrodes)
    state = spec.initial_state()
    before = snapshot(state)
    system, y = battery_impedance_system(spec, state)
    evaluator = ResidualEvaluator(system)
    zero = np.zeros(len(y))
    analytic = evaluator.derivative("state", 0, y, zero).toarray()
    numerical = finite_jacobian(
        lambda point: evaluator.value(0, point, zero), y, system.scales, relative_step=1e-7
    )
    np.testing.assert_allclose(
        analytic * system.scales, numerical * system.scales, rtol=5e-6, atol=2e-7
    )
    frequencies = [0.001, 0.03, 2.0]
    shifted = battery_impedance(spec, frequencies, state=state)
    baseline = battery_impedance(model(family), frequencies)
    assert shifted.success and baseline.success
    assert shifted.response.diagnostics["derivative_modes"]["state"] == "analytic"
    # Constant branch gaps preserve all theta derivatives. The only added port
    # response is -(dUp/dI - dUn/dI), exactly real and independent of frequency.
    expected = 0.5 * 100.0 / 0.02 * sum(
        0.08 if name == "negative" else 0.12 for name in electrodes
    )
    np.testing.assert_allclose(
        np.asarray(shifted.impedance_ohm) - np.asarray(baseline.impedance_ohm),
        expected,
        rtol=2e-8,
        atol=2e-7,
    )
    assert not y.flags.writeable
    assert snapshot(state) == before


@pytest.mark.parametrize("family", ["spm", "mpm"])
@pytest.mark.parametrize("mode", ["current_sigmoid", "axen"])
@pytest.mark.parametrize("electrode", ["negative", "positive"])
def test_operational_battery_soh_refuses_silent_equilibrium_branch_selection(
    family, mode, electrode
):
    spec = model(family, mode, (electrode,))
    state = spec.initial_state()
    before = snapshot(state)
    reference = soh_from_battery(model(family))
    with pytest.raises(ValueError, match=r"equilibrium SOH.*operational OCP"):
        soh_from_battery(spec, state)
    with pytest.raises(ValueError, match=r"equilibrium SOH.*operational OCP"):
        battery_spec_at_soc(spec, reference, 0.5)
    # Choosing a separate equilibrium-reference declaration is explicit and
    # available, but does not assert a hysteretic operating capacity/energy.
    independent = solve_electrode_soh(reference.spec)
    assert independent.to_dict() == reference.to_dict()
    assert independent.theoretical_energy_j() == reference.theoretical_energy_j()
    assert snapshot(state) == before


@pytest.mark.parametrize("mode", ["current_sigmoid", "axen"])
def test_equilibrium_soh_specs_do_not_accept_operational_ocp_objects(mode):
    ocp = operational("positive", mode)
    with pytest.raises(ValueError, match=r"equilibrium.*OCV"):
        HalfCellSOHSpec(ocp, 3600, 3.5, 4.0)
    with pytest.raises(ValueError, match=r"equilibrium.*OCV"):
        ElectrodeSOHSpec(branch("negative"), ocp, 3600, 3600, 3.38, 3.92, 3600)


@pytest.mark.parametrize("family", ["spm", "mpm"])
def test_explicit_single_mode_retains_legacy_impedance_and_soh(family):
    explicit = model(family)
    legacy = replace(explicit, negative_ocp=None, positive_ocp=None)
    state = explicit.initial_state()
    before = snapshot(state)
    np.testing.assert_allclose(
        battery_impedance(explicit, [0.001, 0.1], state=state).impedance_ohm,
        battery_impedance(legacy, [0.001, 0.1]).impedance_ohm,
        rtol=1e-12,
        atol=1e-12,
    )
    assert soh_from_battery(explicit).to_dict() == soh_from_battery(legacy).to_dict()
    assert snapshot(state) == before
