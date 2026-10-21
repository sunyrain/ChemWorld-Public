"""Nonlinear material consumption, parameter provenance and native lifetime behavior."""

import json
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
from chemworld.physchem.battery_degradation import SEISpec
from chemworld.physchem.battery_dfn import BatteryDFNSpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    BatteryThermalSpec,
    ParticleSpec,
    battery_chemical_energy_j,
    battery_chemical_potentials,
    battery_readout,
    battery_spec_from_dict,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.residual_system import ResidualEvaluator, finite_jacobian


def curve():
    source = InputSource(
        DataSourceProvenance(
            "synthetic-ocv", 0, "Self-authored test polynomial/exp/tanh", "1", "CC0-1.0"
        ),
        "synthetic",
    )
    return ElectrodeOCV(
        "test-electrode",
        source,
        (0.2, -0.1),
        exponential_terms=((0.01, -3.0, 0.0),),
        tanh_terms=((-0.02, 5.0, 0.4),),
    )


def model(family):
    options = {
        "negative": ParticleSpec(cells=8),
        "positive": ParticleSpec(cells=8),
        "negative_ocv": curve(),
    }
    if family != "SPM":
        options["electrolyte"] = BatteryElectrolyteSpec(cells=(3, 2, 3))
    return (BatteryDFNSpec if family == "DFN" else BatterySPMSpec)(**options)


def test_curve_value_chain_rule_integral_and_trial_domain():
    c = curve()
    x = 0.3
    u, d, dd = c.derivatives(x, order=2)
    from math import exp, tanh

    t = tanh(5 * (x - 0.4))
    assert u == pytest.approx(0.2 - 0.1 * x + 0.01 * exp(-3 * x) - 0.02 * t)
    assert d == pytest.approx(-0.1 - 0.03 * exp(-3 * x) - 0.1 * (1 - t * t))
    assert dd == pytest.approx(0.09 * exp(-3 * x) + (1 - t * t) * t)
    h = 1e-6
    assert (c.integral(0.01, x + h) - c.integral(0.01, x - h)) / (2 * h) == pytest.approx(
        u, abs=1e-10
    )
    assert c.integral(0.1, 0.9) == -c.integral(0.9, 0.1)
    with pytest.raises(ValueError, match="declared domain"):
        c.value(0.0)
    assert np.isfinite(c.value(0.0, trial=True))


def test_parameter_table_json_units_and_uncertainty_survive_actual_model():
    source = curve().source
    table = ScientificTable(
        "ocv",
        "parameters",
        (
            QuantityColumn("intercept", "mV", "parameter", (-10000, 10000), "same reference"),
            QuantityColumn("slope", "V", "parameter", (-10, 10), "same reference"),
        ),
        (source,),
        (
            ScientificRow(
                "row",
                "synthetic-ocv",
                "group",
                "sample",
                "parameters",
                {
                    "intercept": InputValue(200.0, "mV", 2.0, "test uncertainty"),
                    "slope": InputValue(-0.1, "V", None, "not supplied"),
                },
            ),
        ),
    )
    table = ScientificTable.from_json(json.dumps(table.to_dict()))
    c = ElectrodeOCV.from_parameter_row(
        table, "row", material_id="table-electrode", polynomial=("intercept", "slope")
    )
    assert c.polynomial_V == (0.2, -0.1)
    assert dict(c.parameter_inputs)["intercept"].uncertainty == 2.0
    assert dict(c.parameter_inputs)["intercept"].unit == "mV"
    spec = replace(model("SPM"), negative_ocv=c)
    restored = battery_spec_from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.negative_ocv.source_row_id == "row"
    expected = restored.ocv("positive", 0.3) - restored.ocv("negative", 0.7)
    assert battery_readout(restored, restored.initial_state(), 0).voltage_V == pytest.approx(
        expected
    )
    with pytest.raises(ValueError, match="missing OCV"):
        ElectrodeOCV.from_parameter_row(table, "row", material_id="bad", polynomial=("unknown",))
    with pytest.raises(ValueError):
        ElectrodeOCV.from_parameter_row(
            table, "row", material_id="bad", tanh_terms=(("intercept", "slope", "slope"),)
        )


@pytest.mark.parametrize(
    "change",
    [
        {"polynomial_V": (float("nan"),)},
        {"tanh_terms": ((1, 2),)},
        {"stoichiometry_domain": (1, 0)},
        {"supported_models": ("lead-acid",)},
        {"polynomial_V": (True,)},
        {"material_id": ""},
    ],
)
def test_invalid_materials_are_not_silently_evaluated(change):
    with pytest.raises(ValueError):
        replace(curve(), **change)


def test_incompatible_domains_models_and_couplings_reject():
    with pytest.raises(ValueError, match="stopping surfaces"):
        replace(model("SPM"), negative_ocv=replace(curve(), stoichiometry_domain=(0.1, 0.9)))
    with pytest.raises(ValueError, match="incompatible"):
        replace(model("SPM"), negative_ocv=replace(curve(), supported_models=("DFN",)))
    for changes in ({"thermal": BatteryThermalSpec()}, {"sei": SEISpec()}):
        with pytest.raises(ValueError, match="nonlinear OCV"):
            replace(model("SPM"), **changes)
    raw = curve().to_dict()
    raw["model_version"] = "unknown"
    with pytest.raises(ValueError, match="unsupported OCV"):
        ElectrodeOCV.from_dict(raw)


def test_dimensionless_exponential_and_transition_columns_bind():
    source = curve().source
    table = ScientificTable(
        "terms",
        "parameters",
        tuple(
            QuantityColumn(name, unit, "parameter", (-100, 100), "synthetic reference")
            for name, unit in (("a", "V"), ("k", "dimensionless"), ("s", "dimensionless"))
        ),
        (source,),
        (
            ScientificRow(
                "row",
                "synthetic-ocv",
                "group",
                "sample",
                "parameters",
                {
                    "a": InputValue(-0.02, "V", None, "not supplied"),
                    "k": InputValue(5.0, "dimensionless", None, "not supplied"),
                    "s": InputValue(0.4, "dimensionless", None, "not supplied"),
                },
            ),
        ),
    )
    c = ElectrodeOCV.from_parameter_row(
        table,
        "row",
        material_id="terms",
        exponential_terms=(("a", "k", "s"),),
        tanh_terms=(("a", "k", "s"),),
    )
    assert c.exponential_terms == c.tanh_terms == ((-0.02, 5.0, 0.4),)
    assert len(c.parameter_inputs) == 3


@pytest.mark.parametrize("family", ["SPM", "SPMe", "DFN"])
def test_nonlinear_voltage_energy_gradient_and_readonly_frequency(family):
    spec = model(family)
    initial = spec.initial_state()
    linear = replace(spec, negative_ocv=None)
    assert (
        abs(
            battery_readout(spec, initial, 0).voltage_V
            - battery_readout(linear, linear.initial_state(), 0).voltage_V
        )
        > 0.01
    )
    chemical = battery_chemical_potentials(spec, initial)
    key = next(iter(chemical))
    h = initial.amounts_mol[key] * 1e-5
    altered = [
        replace(initial, amounts_mol={**initial.amounts_mol, key: initial.amounts_mol[key] + d})
        for d in (-h, h)
    ]
    derivative = (
        battery_chemical_energy_j(spec, altered[1]) - battery_chemical_energy_j(spec, altered[0])
    ) / (2 * h)
    assert derivative == pytest.approx(chemical[key], rel=1e-6)
    answer = battery_impedance(spec, [0.001, 0.1])
    assert answer.success
    assert initial == spec.initial_state()
    assert answer.impedance_ohm != battery_impedance(linear, [0.001, 0.1]).impedance_ohm


def test_dfn_nonlinear_analytic_jacobian_matches_independent_differences():
    spec = model("DFN")
    system, y = battery_impedance_system(spec, spec.initial_state())
    ev = ResidualEvaluator(system)
    actual = ev.derivative("state", 0, y, np.zeros_like(y)).toarray()
    expected = finite_jacobian(lambda z: ev.value(0, z, np.zeros_like(z)), y, system.scales)
    actual_scaled = actual * system.scales[None, :] / system.residual_scales[:, None]
    expected_scaled = expected * system.scales[None, :] / system.residual_scales[:, None]
    np.testing.assert_allclose(actual_scaled, expected_scaled, rtol=5e-6, atol=2e-5)
    assert ev.diagnostics()["derivative_modes"]["state"] == "analytic"


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_real_nonlinear_observations_and_exact_replay(tmp_path, noise):
    from chemworld.examples.battery_spme import run

    result = run(tmp_path / (noise + ".jsonl"), spec=model("SPMe"), noise=noise)
    assert result["steps"] == result["replay_steps"] == 9


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
@pytest.mark.parametrize("family", ["SPMe", "DFN"])
def test_nonlinear_snapshot_continues_in_new_process(tmp_path, noise, family):
    from chemworld.examples.battery_spme import workflow
    from test_native_domains_end_to_end import test_snapshot_resume_in_middle_of_real_workflow

    test_snapshot_resume_in_middle_of_real_workflow(
        family, noise, tmp_path, recipe=workflow(model(family)), sort_keys=True
    )
