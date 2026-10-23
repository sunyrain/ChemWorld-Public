"""Functional UNIFAC compilation, limits and analytic consumers."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.activity_thermodynamics import (
    active_activity_parameters,
    activity_infinite_dilution,
    activity_parameter_jacobian,
    activity_thermodynamic_report,
)
from chemworld.physchem.equilibrium import ActivityModelSpec, activity_coefficients
from chemworld.physchem.unifac import UNIFACSubgroup, unifac_activity_model, unifac_rq


def model(version=0):
    return unifac_activity_model(
        "hexane-butanone",
        {"hexane": {1: 2, 2: 4}, "butanone": {1: 1, 2: 1, 18: 1}},
        subgroups={
            1: UNIFACSubgroup(1, 1, 0.0 if version == 3 else 0.9011, 0.848),
            2: UNIFACSubgroup(2, 1, 0.0 if version == 3 else 0.6744, 0.540),
            18: UNIFACSubgroup(18, 9, 0.0 if version == 3 else 1.6724, 1.488),
        },
        interactions={(1, 9): (476.4, 0.0, 0.0), (9, 1): (26.76, 0.0, 0.0)},
        version=version,
        provenance={"citation": "UNIFAC public example"},
    )


def test_public_hexane_butanone_example_and_rq():
    spec = model()
    gamma = activity_coefficients(spec, {"hexane": 0.5, "butanone": 0.5}, temperature_K=333.15)
    assert list(gamma.values()) == pytest.approx([1.427602583562, 1.36465450101], rel=2e-10)
    assert unifac_rq(
        {1: 2, 2: 4},
        {1: UNIFACSubgroup(1, 1, 0.9011, 0.848), 2: UNIFACSubgroup(2, 1, 0.6744, 0.540)},
    ) == pytest.approx((4.4998, 3.856))


@pytest.mark.parametrize("version", range(6))
def test_variants_roundtrip_limits_and_shared_derivatives(version):
    spec = model(version)
    restored = ActivityModelSpec(
        **{**json.loads(json.dumps(spec.to_dict())), "component_ids": spec.component_ids}
    )
    report = activity_thermodynamic_report(
        restored, {"hexane": 0.2, "butanone": 0.8}, temperature_K=320
    )
    assert (
        report.to_dict()
        == activity_thermodynamic_report(
            spec, {"hexane": 0.2, "butanone": 0.8}, temperature_K=320
        ).to_dict()
    )
    assert report.enthalpy_excess_J_mol == pytest.approx(
        report.gibbs_excess_J_mol - 320 * report.dGE_dT
    )
    for key in spec.component_ids:
        pure = activity_thermodynamic_report(
            spec, {other: float(other == key) for other in spec.component_ids}, temperature_K=320
        )
        assert pure.gibbs_excess_J_mol == pytest.approx(0, abs=1e-10)
        assert pure.activity_coefficients[key] == pytest.approx(1)
    dilute = activity_infinite_dilution(
        spec, {"hexane": 0, "butanone": 1}, solute_id="hexane", temperature_K=320
    )
    trace = activity_coefficients(spec, {"hexane": 1e-10, "butanone": 1 - 1e-10}, temperature_K=320)
    assert dilute.activity_coefficients["hexane"] == pytest.approx(trace["hexane"], rel=1e-8)
    assert all(key.startswith("psi_") for key in active_activity_parameters(spec))
    with pytest.raises(ValueError):
        activity_parameter_jacobian(
            spec,
            {"hexane": 0.2, "butanone": 0.8},
            temperature_K=320,
            quantities=("GE",),
            parameters=("version",),
        )


def test_missing_pair_and_invalid_structural_data_rejected():
    spec = model()
    for key, value in (("psi_a:1|9", None), ("nu:hexane|1", 0.5), ("version", 6), ("Q:18", -1)):
        parameters = dict(spec.parameters)
        if value is None:
            parameters.pop(key)
        else:
            parameters[key] = value
        with pytest.raises(ValueError):
            replace(spec, parameters=parameters)
    with pytest.raises(ValueError, match="positive surface"):
        replace(
            spec,
            parameters={k: 0.0 if k.startswith("Q:") else v for k, v in spec.parameters.items()},
        )


def test_missing_directed_interaction_never_falls_back_to_ideal():
    with pytest.raises(ValueError, match="missing directed"):
        unifac_activity_model(
            "missing",
            {"A": {1: 1}, "B": {2: 1}},
            subgroups={1: UNIFACSubgroup(1, 1, 1, 1), 2: UNIFACSubgroup(2, 2, 1, 1)},
            interactions={(1, 2): (100.0, 0.0, 0.0)},
            version=0,
            provenance={"source": "authored"},
        )


def test_tiny_vtpr_interaction_does_not_create_zero_times_log_zero():
    spec = unifac_activity_model(
        "extreme",
        {"A": {1: 1}, "B": {2: 1}},
        subgroups={1: UNIFACSubgroup(1, 42, 0, 1), 2: UNIFACSubgroup(2, 151, 0, 1)},
        interactions={(42, 151): (605.58, -0.2551, 56.062), (151, 42): (0.0, 0.0, 0.0)},
        version=3,
        provenance={"source": "VTPR2012 row111"},
    )
    gamma = activity_coefficients(spec, {"A": 0.35, "B": 0.65}, temperature_K=320.0)
    assert all(value > 0 for value in gamma.values())
