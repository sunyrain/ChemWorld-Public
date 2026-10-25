"""Released material identity, complete data and the actual SPM projection."""

import json
from dataclasses import FrozenInstanceError, replace
from math import exp, sqrt, tanh

import pytest

from chemworld.data.scientific_inputs import InputSource, InputValue, ScientificTable
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    battery_readout,
    battery_spec_from_dict,
)
from chemworld.physchem.battery_parameter_sets import (
    BatteryParameterBinding,
    BatteryParameterSet,
    battery_parameter_set,
)


def test_complete_released_catalog_has_units_sources_and_explicit_consumption():
    source = battery_parameter_set()
    assert isinstance(source.parameters, ScientificTable)
    assert len(source.fields) == 94
    assert len(source.scalar_inputs) == 86
    assert len(source.functions) == 6
    assert len({f.source_field for f in source.fields}) == 94
    assert source.metadata == {"chemistry": "lithium_ion", "citations": ("Chen2020",)}
    assert all(isinstance(s, InputSource) and s.data_kind == "literature" for s in source.sources)
    assert all(
        isinstance(v, InputValue)
        and v.quality == "valid"
        and v.uncertainty is None
        and v.uncertainty_note
        for v in source.scalar_inputs.values()
    )
    kinds = {f.source_field: (f.kind, f.usage) for f in source.fields}
    assert kinds["Negative particle diffusivity [m2.s-1]"] == ("number", "consumed")
    assert kinds["Current function [A]"] == ("number", "retained-not-consumed")
    assert kinds["Electrolyte conductivity [S.m-1]"] == ("function", "retained-not-consumed")
    assert kinds["SEI partial molar volume [m3.mol-1]"] == ("number", "retained-not-consumed")
    assert source.scalar("Nominal cell capacity [A.h]", "C") == 18000
    assert source.scalar("Negative current collector conductivity [S.m-1]", "S/m") == 58411000
    assert "SEI entries are example" in " ".join(source.to_dict()["notes"])


def test_source_catalog_is_immutable_and_codec_returns_defensive_copies():
    source = battery_parameter_set()
    with pytest.raises(FrozenInstanceError):
        source._payload_json = "{}"
    with pytest.raises(TypeError):
        source.scalar_inputs["Cell volume [m3]"] = InputValue(1, "m^3", None, "test")
    with pytest.raises(TypeError):
        source.metadata["chemistry"] = "lead_acid"
    with pytest.raises(FrozenInstanceError):
        source.fields[0].usage = "retained-not-consumed"
    raw = source.to_dict()
    restored = BatteryParameterSet.from_dict(json.loads(json.dumps(raw, sort_keys=True)))
    assert restored == source
    raw["metadata"]["citations"].append("other")
    assert source.metadata["citations"] == ("Chen2020",)


@pytest.mark.parametrize("change", ("missing", "unit", "value", "quality", "function", "source"))
def test_changed_released_catalog_does_not_keep_the_original_source_identity(change):
    raw = battery_parameter_set().to_dict()
    values = raw["parameters"]["rows"][0]["values"]
    if change == "missing":
        values.pop("Cell volume [m3]")
    elif change == "unit":
        values["Cell volume [m3]"]["unit"] = "V"
    elif change == "value":
        values["Cell volume [m3]"]["value"] *= 2
    elif change == "quality":
        values["Cell volume [m3]"]["quality"] = "invalid"
    elif change == "function":
        raw["functions"][0]["function_id"] = "__import__('os').system"
    else:
        raw["sources"][0]["provenance"]["citation"] = "unknown"
    with pytest.raises(ValueError, match="unknown or modified"):
        BatteryParameterSet.from_dict(raw)


@pytest.mark.parametrize("theta", (0.01, 0.1, 0.3, 0.5, 0.9, 0.99))
def test_released_ocp_numbers_and_independently_sourced_endpoint_are_both_consumed(theta):
    source = battery_parameter_set()
    negative = (
        1.9793 * exp(-39.3631 * theta)
        + 0.2482
        - 0.0909 * tanh(29.8538 * (theta - 0.1234))
        - 0.04478 * tanh(14.9159 * (theta - 0.2769))
        - 0.0205 * tanh(30.4444 * (theta - 0.6103))
    )
    positive = (
        -0.8090 * theta
        + 4.4875
        - 0.0428 * tanh(18.5138 * (theta - 0.5542))
        - 17.7326 * tanh(15.7890 * (theta - 0.3117))
        + 17.5842 * tanh(15.9308 * (theta - 0.3120))
    )
    barrier = 1e-6 * (1 / theta + 1 / (theta - 1))
    for electrode, expected in (("negative", negative), ("positive", positive)):
        raw = source.function(electrode.capitalize() + " electrode OCP [V]")
        curve = source.electrode_curve(electrode)
        assert raw.evaluate(sto=theta) == pytest.approx(expected, abs=1e-13)
        assert curve.value(theta) == pytest.approx(expected + barrier, abs=1e-13)
        assert curve.endpoint_policy.source != curve.source
        assert curve.source_row_id
        assert curve.parameter_inputs


@pytest.mark.parametrize(
    "electrode,m_ref,cmax,activation",
    (("negative", 6.48e-7, 33133, 35000), ("positive", 3.42e-6, 63104, 17800)),
)
def test_source_exchange_current_retains_all_inputs_and_compiles_fixed_conditions(
    electrode, m_ref, cmax, activation
):
    from chemworld.physchem.electrochemistry import R_J_PER_MOL_K

    source = battery_parameter_set()
    law = source.function(electrode.capitalize() + " electrode exchange-current density [A.m-2]")
    cs = 0.4 * cmax
    expected = m_ref * sqrt(1000) * sqrt(cs) * sqrt(cmax - cs)
    assert law.evaluate(c_e=1000, c_s_surf=cs, c_s_max=cmax, T=298.15) == pytest.approx(expected)
    warm = expected * exp(activation / R_J_PER_MOL_K * (1 / 298.15 - 1 / 310))
    assert law.evaluate(c_e=1000, c_s_surf=cs, c_s_max=cmax, T=310) == pytest.approx(warm)
    spec = source.build_spm(maximum_current_A=8, radial_cells=4)
    particle = getattr(spec, electrode)
    assert particle.exchange_current_scale_A_m2 == pytest.approx(m_ref * sqrt(1000) * cmax)
    assert particle.exchange_current_scale_A_m2 * sqrt(0.4 * 0.6) == pytest.approx(expected)
    derivation = next(
        d
        for d in spec.parameter_binding.derivations
        if d.target == electrode + ".exchange_current_scale_A_m2"
    )
    assert "Arrhenius factor=1" in derivation.formula
    assert any(
        name.endswith("::activation_energy") and value.value == activation
        for name, value in derivation.inputs
    )


def test_retained_nyman_laws_are_accessible_without_claiming_dynamic_spm_transport():
    source = battery_parameter_set()
    diffusivity = source.function("Electrolyte diffusivity [m2.s-1]")
    conductivity = source.function("Electrolyte conductivity [S.m-1]")
    assert diffusivity.evaluate(c_e=1000, T=298.15) == pytest.approx(1.7694e-10)
    assert conductivity.evaluate(c_e=1000, T=298.15) == pytest.approx(0.9487)
    assert conductivity.evaluate(c_e=1000, T=310) == conductivity.evaluate(c_e=1000, T=298.15)
    assert "no temperature dependence" in conductivity.validity_note
    with pytest.raises(ValueError, match="exact declared arguments"):
        conductivity.evaluate(c_e=1000)
    with pytest.raises(ValueError, match="T > 0"):
        conductivity.evaluate(c_e=1000, T=0)


def test_actual_projection_consumes_geometry_inventory_and_declared_device_capability():
    source = battery_parameter_set()
    spec = source.build_spm(maximum_current_A=8, radial_cells=6)
    assert type(spec) is BatterySPMSpec
    assert spec.parameter_binding.validate_projection(spec)
    assert spec.negative.active_volume_m3 == pytest.approx(0.065 * 1.58 * 8.52e-5 * 0.75)
    assert spec.positive.active_volume_m3 == pytest.approx(0.065 * 1.58 * 7.56e-5 * 0.665)
    assert spec.negative.maximum_concentration_mol_m3 == 33133
    assert spec.positive.maximum_concentration_mol_m3 == 63104
    assert spec.initial_negative_fraction == 29866 / 33133
    assert spec.initial_positive_fraction == 17038 / 63104
    assert spec.temperature_K == 298.15
    assert (spec.minimum_voltage_V, spec.maximum_voltage_V) == (2.5, 4.2)
    assert spec.maximum_current_A == 8  # Released default current 5 A is not Imax.
    assert spec.kinetics_stoichiometry_clip == 1e-8
    assert spec.cell_envelope_volume_m3 == pytest.approx(2.42e-5)
    assert spec.volume_l == pytest.approx(0.0242)
    assert spec.electrolyte is None and spec.thermal is None and spec.sei is None
    state = spec.initial_state()
    assert sum(state.amounts_mol[k] for k in spec.negative.keys("negative")) == pytest.approx(
        spec.negative.active_volume_m3 * 29866
    )
    assert sum(state.amounts_mol[k] for k in spec.positive.keys("positive")) == pytest.approx(
        spec.positive.active_volume_m3 * 17038
    )
    reading = battery_readout(spec, state, 0)
    assert reading.voltage_V == pytest.approx(
        spec.positive_ocv.value(spec.initial_positive_fraction)
        - spec.negative_ocv.value(spec.initial_negative_fraction)
    )


def test_nominal_current_is_distinct_from_unchanged_initial_stoichiometric_current():
    source = battery_parameter_set()
    spec = source.build_spm(maximum_current_A=10, radial_cells=4)
    assert source.source_nominal_capacity_Ah == 5
    assert source.current_at_nominal_c_rate(1) == 5
    assert source.current_at_nominal_c_rate(-0.5) == -2.5
    assert spec.parameter_binding.current_at_nominal_c_rate(1) == 5
    assert spec.current_at_c_rate(1) == pytest.approx(5.252997061422181)
    assert spec.current_at_c_rate(1) != source.current_at_nominal_c_rate(1)


@pytest.mark.parametrize("rate", (True, float("nan"), float("inf"), "1"))
def test_nominal_rate_refuses_nonfinite_or_nonnumeric_values(rate):
    with pytest.raises(ValueError, match="finite real"):
        battery_parameter_set().current_at_nominal_c_rate(rate)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"model": "DFN"},
        {"model": "SPMe"},
        {"temperature_K": 310},
        {"electrolyte_concentration_mol_m3": 900},
        {"radial_cells": True},
        {"radial_cells": 1},
        {"maximum_current_A": True},
        {"maximum_current_A": 0},
        {"surface_margin": 1e-10},
    ),
)
def test_source_builder_refuses_unsupported_physics_and_invalid_numerical_policy(kwargs):
    options = {"maximum_current_A": 10, **kwargs}
    with pytest.raises(ValueError):
        battery_parameter_set().build_spm(**options)


@pytest.mark.parametrize(
    "name,version", (("Chen2021", "PyBaMM-25.4.2"), ("Chen2020", "PyBaMM-25.4.1"))
)
def test_unknown_source_has_no_synthetic_fallback(name, version):
    with pytest.raises(ValueError, match="unknown battery parameter"):
        battery_parameter_set(name, version=version)


def test_binding_sorted_json_and_original_physical_codec_are_exact():
    spec = battery_parameter_set().build_spm(maximum_current_A=9, radial_cells=4)
    payload = json.loads(json.dumps(spec.to_dict(), sort_keys=True))
    restored = battery_spec_from_dict(payload)
    assert restored == spec
    assert restored.to_dict() == spec.to_dict()
    assert BatteryParameterBinding.from_dict(payload["parameter_binding"]) == spec.parameter_binding
    assert spec.parameter_binding.catalog_reference.endswith(
        "parameters/battery_chen2020_pybamm_25_4_2.json"
    )
    assert len(spec.parameter_binding.catalog_sha256) == 64


@pytest.mark.parametrize(
    "field",
    (
        "radius_m",
        "diffusivity_m2_s",
        "active_volume_m3",
        "maximum_concentration_mol_m3",
        "exchange_current_scale_A_m2",
    ),
)
def test_source_label_cannot_bind_a_changed_real_particle(field):
    spec = battery_parameter_set().build_spm(maximum_current_A=10, radial_cells=4)
    particle = replace(spec.negative, **{field: getattr(spec.negative, field) * 1.01})
    with pytest.raises(ValueError, match="projection mismatch"):
        replace(spec, negative=particle)


@pytest.mark.parametrize(
    "field,value",
    (
        ("initial_negative_fraction", 0.8),
        ("initial_positive_fraction", 0.3),
        ("temperature_K", 299),
        ("minimum_voltage_V", 2.6),
        ("maximum_voltage_V", 4.3),
        ("series_resistance_ohm", 0.01),
        ("cell_envelope_volume_m3", 3e-5),
        ("kinetics_stoichiometry_clip", 1e-7),
    ),
)
def test_source_label_cannot_bind_changed_initial_conditions_or_ports(field, value):
    spec = battery_parameter_set().build_spm(maximum_current_A=10, radial_cells=4)
    with pytest.raises(ValueError, match="projection mismatch"):
        replace(spec, **{field: value})


def test_binding_rejects_changed_curve_formula_and_false_derivation():
    spec = battery_parameter_set().build_spm(maximum_current_A=10, radial_cells=4)
    changed = replace(spec.negative_ocv, polynomial_V=(0.25,))
    with pytest.raises(ValueError, match="projection mismatch: negative_ocv"):
        replace(spec, negative_ocv=changed)
    payload = spec.parameter_binding.to_dict()
    payload["derivations"][0]["value"]["value"] *= 2
    with pytest.raises(ValueError, match="released source projection"):
        BatteryParameterBinding.from_dict(payload)
    payload = spec.parameter_binding.to_dict()
    payload["catalog_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="released source projection"):
        BatteryParameterBinding.from_dict(payload)


def test_binding_is_an_immutable_configuration_and_legacy_defaults_stay_unbound():
    binding = BatteryParameterBinding.from_parameter_set(battery_parameter_set())
    with pytest.raises(FrozenInstanceError):
        binding.projection_id = "DFN"
    raw = binding.to_dict()
    raw["derivations"][0]["inputs"].clear()
    assert binding.to_dict()["derivations"][0]["inputs"]
    legacy = BatterySPMSpec().to_dict()
    assert "parameter_binding" not in legacy
    assert "cell_envelope_volume_m3" not in legacy
    assert "kinetics_stoichiometry_clip" not in legacy
