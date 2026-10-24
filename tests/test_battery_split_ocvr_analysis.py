"""Independent port/SOH contracts for the direct-capacity two-stock model."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import F
from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery
from chemworld.physchem.battery_split_ocvr import BatterySplitOCVRSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEvaluator,
    finite_jacobian,
)


def curve(electrode, *, nonlinear=False):
    coefficients = (0.2, -0.1) if electrode == "negative" else (4.2, -0.8)
    if nonlinear:
        coefficients += (0.02 if electrode == "negative" else 0.05,)
    return ElectrodeOCV(
        "split-analysis-" + electrode,
        InputSource(
            DataSourceProvenance(
                "split-analysis-synthetic", 0, "Explicit analysis fixture", "1", "CC0-1.0"
            ),
            "synthetic",
        ),
        coefficients,
        stoichiometry_domain=(0.0, 1.0),
        supported_models=("SplitOCVR",),
    )


def model(*, resistance=0.08, nonlinear=False):
    # Qn != Qp and total electrode Li is 106 C, not either electrode capacity.
    return BatterySplitOCVRSpec(
        negative_capacity_C=100.0,
        positive_capacity_C=120.0,
        negative_ocv=curve("negative", nonlinear=nonlinear),
        positive_ocv=curve("positive", nonlinear=nonlinear),
        initial_negative_fraction=0.7,
        initial_positive_fraction=0.3,
        negative_minimum_fraction=0.1,
        negative_maximum_fraction=0.9,
        positive_minimum_fraction=0.15,
        positive_maximum_fraction=0.8,
        minimum_voltage_V=3.0,
        maximum_voltage_V=4.4,
        resistance_ohm=resistance,
    )


def snapshot(state):
    return (
        dict(state.amounts_mol), dict(state.coordinates), state.time_s,
        state.temperature_K, state.volume_L,
    )


@pytest.mark.parametrize("resistance", [0.0, 0.08])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_same_residual_matches_independent_two_capacity_port_impedance(resistance, nonlinear):
    spec = model(resistance=resistance, nonlinear=nonlinear)
    state = spec.initial_state()
    before, configuration = snapshot(state), spec.to_dict()
    system, initial = battery_impedance_system(spec, state)
    result = battery_impedance(spec, [0.001, 0.03, 3.0], state=state)
    assert result.success
    assert set(state.amounts_mol) == {"negative:Li", "positive:Li"}
    indices = {v.name: i for i, v in enumerate(system.variables)}
    assert set(indices) == {
        "negative:Li", "positive:Li", "discharged_charge_C", "delivered_work_J",
        "battery:current_A", "battery:voltage_V",
    }
    # Differentiate the declared fixture polynomials independently of OCV code.
    slope_n = -0.1 + (2 * 0.02 * 0.7 if nonlinear else 0)
    slope_p = -0.8 + (2 * 0.05 * 0.3 if nonlinear else 0)
    voltage = 4.2 - 0.8 * 0.3 - (0.2 - 0.1 * 0.7)
    voltage += 0.05 * 0.3**2 - 0.02 * 0.7**2 if nonlinear else 0
    for point, measured in zip(result.response.points, result.impedance_ohm, strict=True):
        iw = 2j * np.pi * point.frequency_Hz
        expected = resistance - (slope_n / 100.0 + slope_p / 120.0) / iw
        assert measured == pytest.approx(expected, rel=2e-10, abs=2e-12)
        response = point.response[:, 0]
        assert response[indices["battery:current_A"]] == pytest.approx(1)
        assert response[indices["negative:Li"]] == pytest.approx(-1 / (F * iw))
        assert response[indices["positive:Li"]] == pytest.approx(1 / (F * iw))
        assert response[indices["discharged_charge_C"]] == pytest.approx(1 / iw)
        assert response[indices["delivered_work_J"]] == pytest.approx(voltage / iw)
    assert result.response.diagnostics["derivative_modes"]["state"] == "analytic"
    assert snapshot(state) == before and spec.to_dict() == configuration
    assert not initial.flags.writeable
    json.dumps(result.to_dict(), allow_nan=False)


def test_analysis_jacobian_is_the_actual_nonlinear_inventory_residual():
    spec = model(nonlinear=True)
    system, initial = battery_impedance_system(spec, spec.initial_state())
    evaluator = ResidualEvaluator(system)
    rate = np.zeros(len(initial))
    actual = evaluator.derivative("state", 0, initial, rate).toarray()
    independent = finite_jacobian(
        lambda value: evaluator.value(0, value, rate), initial, system.scales
    )
    np.testing.assert_allclose(
        actual * system.scales, independent * system.scales, rtol=2e-6, atol=2e-8
    )


@pytest.mark.parametrize("electrode", ["negative", "positive"])
@pytest.mark.parametrize("side", ["minimum", "maximum"])
def test_eis_rejects_fraction_faces_even_when_directional_recovery_is_legal(electrode, side):
    spec = model()
    spec = replace(spec, **{
        "initial_" + electrode + "_fraction": getattr(spec, electrode + "_" + side + "_fraction")
    })
    state = spec.initial_state()
    before = snapshot(state)
    with pytest.raises(ResidualDomainError, match="stopping boundary"):
        battery_impedance(spec, [0.01], state=state)
    assert snapshot(state) == before


def test_analysis_does_not_add_missing_stock_or_accept_wrong_state_layout():
    spec = model()
    state = replace(spec.initial_state(), amounts_mol={"negative:Li": 0.0005})
    with pytest.raises(ValueError, match="inventory and coordinates"):
        battery_impedance(spec, [0.01], state=state)
    with pytest.raises(ValueError, match="inventory is missing"):
        soh_from_battery(spec, state)


def soh_model():
    # V(x)=4 - .8*(106-100*x)/120 + .1*x along the true Li constraint.
    return replace(
        model(),
        minimum_voltage_V=4.0 - 0.8 * (106.0 - 100.0 * 0.2) / 120.0 + 0.1 * 0.2,
        maximum_voltage_V=4.0 - 0.8 * (106.0 - 100.0 * 0.8) / 120.0 + 0.1 * 0.8,
    )


def test_soh_uses_actual_two_stocks_asymmetric_capacity_and_declared_ocv_domains():
    spec = soh_model()
    state = spec.initial_state()
    before = snapshot(state)
    result = soh_from_battery(spec, state)
    assert result.status == "unique"
    assert result.spec.negative_capacity_C == 100
    assert result.spec.positive_capacity_C == 120
    assert result.spec.known_capacity_C == pytest.approx(106)
    assert result.spec.negative_ocv.stoichiometry_domain == (0.1, 0.9)
    assert result.spec.positive_ocv.stoichiometry_domain == (0.15, 0.8)
    assert result.spec.negative_ocv.source == spec.negative_ocv.source
    assert result.spec.positive_ocv.polynomial_V == spec.positive_ocv.polynomial_V
    branch = result.branch()
    assert [branch[k] for k in ("x_0", "x_100", "y_0", "y_100")] == pytest.approx(
        [0.2, 0.8, 86.0 / 120, 26.0 / 120], abs=2e-9
    )
    assert branch["cell_capacity_C"] == pytest.approx(60)
    assert branch["cyclable_lithium_C"] == pytest.approx(106)
    assert branch["local_parameter_response"]["status"] == "local_full_rank"
    assert snapshot(state) == before
    json.dumps(result.to_dict(), allow_nan=False)


def test_soh_reads_live_inventory_without_restarting_or_using_charge_counter_as_capacity():
    spec = soh_model()
    original = spec.initial_state()
    q = 10.0
    moved = replace(
        original,
        time_s=13,
        amounts_mol={
            "negative:Li": original.amounts_mol["negative:Li"] - q / F,
            "positive:Li": original.amounts_mol["positive:Li"] + q / F,
        },
        coordinates={"discharged_charge_C": q, "delivered_work_J": 38.0},
    )
    before = snapshot(moved)
    first = soh_from_battery(spec, original, sensitivity=False)
    second = soh_from_battery(spec, moved, sensitivity=False)
    for key in ("x_0", "x_100", "y_0", "y_100", "cell_capacity_C", "cyclable_lithium_C"):
        assert second.branch()[key] == pytest.approx(first.branch()[key], abs=2e-10)
    assert snapshot(moved) == before
    lower_li = replace(original, amounts_mol={**original.amounts_mol, "negative:Li": 60.0 / F})
    assert soh_from_battery(spec, lower_li).spec.known_capacity_C == pytest.approx(96)


def test_soc_prepares_only_new_configuration_with_real_capacity_and_source_identity():
    spec = soh_model()
    state = spec.initial_state()
    before, configuration = snapshot(state), spec.to_dict()
    result = soh_from_battery(spec, state)
    prepared = battery_spec_at_soc(spec, result, 0.25)
    assert prepared.initial_negative_fraction == pytest.approx(0.35)
    assert prepared.initial_positive_fraction == pytest.approx(71.0 / 120)
    assert F * sum(prepared.initial_state().amounts_mol.values()) == pytest.approx(106)
    assert prepared.negative_ocv == spec.negative_ocv
    assert prepared.positive_ocv == spec.positive_ocv
    assert snapshot(state) == before and spec.to_dict() == configuration
    with pytest.raises(ValueError, match="capacities"):
        battery_spec_at_soc(replace(spec, negative_capacity_C=110), result, 0.5)
    with pytest.raises(ValueError, match="OCV/window"):
        battery_spec_at_soc(
            replace(spec, positive_ocv=replace(spec.positive_ocv, polynomial_V=(4.2, -0.7))),
            result, 0.5,
        )
    with pytest.raises(ValueError, match="OCV/window"):
        battery_spec_at_soc(replace(spec, negative_minimum_fraction=0.15), result, 0.5)


def test_soh_does_not_search_outside_physical_fraction_window_or_repair_invalid_stock():
    spec = model()
    # Both voltage cutoffs are outside the constrained physical working path.
    result = soh_from_battery(spec)
    assert result.status == "no_solution"
    assert not result.solutions
    invalid = replace(spec.initial_state(), amounts_mol={"negative:Li": 1.0, "positive:Li": 0.001})
    with pytest.raises(ValueError, match="physical capacity"):
        soh_from_battery(spec, invalid)


def test_native_split_mapping_retains_multiple_soh_branches_and_requires_selection():
    polynomial = np.polynomial.polynomial.polymul([0.0625, -0.5, 1], [0.5625, -1.5, 1]) * 16
    polynomial[0] += 3
    spec = replace(
        model(), negative_capacity_C=100, positive_capacity_C=100,
        initial_negative_fraction=0.8, initial_positive_fraction=0.2,
        negative_minimum_fraction=0.01, negative_maximum_fraction=0.99,
        positive_minimum_fraction=0.01, positive_maximum_fraction=0.99,
        negative_ocv=replace(curve("negative"), polynomial_V=(0.0,)),
        positive_ocv=replace(curve("positive"), polynomial_V=tuple(polynomial)),
        minimum_voltage_V=3.01, maximum_voltage_V=3.04,
    )
    result = soh_from_battery(spec, sensitivity=False)
    assert result.status == "multiple" and len(result.solutions) == 2
    with pytest.raises(ValueError, match="explicit branch"):
        battery_spec_at_soc(spec, result, 0.5)
    prepared = battery_spec_at_soc(spec, result, 0.5, branch=1)
    assert prepared.initial_negative_fraction + prepared.initial_positive_fraction == (
        pytest.approx(1)
    )
