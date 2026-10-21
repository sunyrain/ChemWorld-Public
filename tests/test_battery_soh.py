"""Public SOH analysis, explicit branch semantics and real native consumers."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.data.scientific_inputs import (
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_dfn import BatteryDFNSpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_models import BatterySPMSpec, F, ParticleSpec
from chemworld.physchem.battery_soh import (
    ElectrodeSOHSpec,
    HalfCellSOHSpec,
    battery_spec_at_soc,
    soh_from_battery,
    soh_spec_from_dict,
    solve_electrode_soh,
)
from test_battery_materials import curve


def linear():
    return ElectrodeSOHSpec(
        replace(curve(), polynomial_V=(0.2, -0.1), exponential_terms=(), tanh_terms=()),
        replace(curve(), polynomial_V=(4.2, -0.8), exponential_terms=(), tanh_terms=()),
        3600,
        3600,
        3.38,
        3.92,
        3600,
    )


def test_linear_window_soc_voltage_energy_and_sensitivity():
    spec = linear()
    result = solve_electrode_soh(spec)
    assert result.status == "unique"
    s = result.branch()
    assert [s[k] for k in ("x_0", "x_100", "y_0", "y_100")] == pytest.approx([0.2, 0.8, 0.8, 0.2])
    assert s["cell_capacity_C"] == pytest.approx(2160)
    assert result.at_soc(0.5)["voltage_V"] == pytest.approx(3.65)
    assert result.theoretical_energy_j() == pytest.approx(2160 * 3.65)
    assert result.theoretical_energy_j(0, 1) == -result.theoretical_energy_j()
    for soc in (0, 0.25, 0.5, 0.75, 1):
        recovered = result.soc_at_voltage(result.at_soc(soc)["voltage_V"])
        assert recovered.status == "unique"
        assert recovered.roots[0]["value"][0] == pytest.approx(soc, abs=1e-9)
    local = s["local_parameter_response"]
    assert local["status"] == "local_full_rank"
    assert local["output_derivatives"]["cell_capacity_C"][-2:] == pytest.approx([-4000, 4000])
    json.dumps(result.to_dict(), allow_nan=False)
    assert result.soc_at_voltage(10).status == "no_solution"
    with pytest.raises(ValueError, match="SOC"):
        result.at_soc(1.1)


def test_known_capacity_and_half_cell_use_real_equations():
    spec = linear()
    n = replace(spec.negative_ocv, polynomial_V=(0.3, -0.2, -0.1))
    p = replace(spec.positive_ocv, polynomial_V=(4.4, -0.6, -0.2))
    spec = ElectrodeSOHSpec(n, p, 3600, 5400, 3.7535, 4.2295, 2160, "cell_capacity")
    result = solve_electrode_soh(spec)
    assert result.status == "unique"
    s = result.branch()
    assert [s[k] for k in ("x_0", "x_100", "y_0", "y_100")] == pytest.approx(
        [0.2, 0.8, 0.55, 0.15], abs=1e-8
    )
    assert s["cyclable_lithium_C"] == pytest.approx(3690)
    assert s["local_parameter_response"]["status"] == "local_full_rank"
    half = solve_electrode_soh(
        HalfCellSOHSpec(replace(n, polynomial_V=(0.3, -0.2)), 3600, 0.12, 0.26)
    )
    assert half.status == "unique"
    assert half.branch()["cell_capacity_C"] == pytest.approx(2520)
    assert half.at_soc(1)["y"] == pytest.approx(0.2)
    assert half.theoretical_energy_j() == pytest.approx(2520 * 0.19)


def test_multiple_no_solution_infeasible_and_nonidentifiable_remain_distinct():
    spec = linear()
    poly = np.polynomial.polynomial.polymul([0.0625, -0.5, 1], [0.5625, -1.5, 1]) * 16
    poly[0] += 3
    spec = replace(
        spec,
        negative_ocv=replace(spec.negative_ocv, polynomial_V=(0.0,)),
        positive_ocv=replace(spec.positive_ocv, polynomial_V=tuple(poly)),
        minimum_voltage_V=3.01,
        maximum_voltage_V=3.04,
    )
    result = solve_electrode_soh(spec)
    assert result.status == "multiple"
    assert len(result.solutions) == 2
    with pytest.raises(ValueError, match="explicit branch"):
        result.at_soc(0.5)
    assert result.at_soc(0.5, branch=1)["soc"] == 0.5
    half = solve_electrode_soh(HalfCellSOHSpec(spec.positive_ocv, 3600, 3.01, 3.04))
    assert half.status == "multiple"
    empty = solve_electrode_soh(replace(linear(), minimum_voltage_V=6, maximum_voltage_V=7))
    assert empty.status == "no_solution"
    infeasible = solve_electrode_soh(replace(linear(), known_capacity_C=1e6))
    assert infeasible.status == "infeasible"
    singular = solve_electrode_soh(
        replace(
            linear(),
            positive_capacity_C=5400,
            minimum_voltage_V=3.54,
            maximum_voltage_V=3.92,
            known="cell_capacity",
            known_capacity_C=2160,
        )
    )
    assert singular.status == "unresolved"
    assert not singular.solutions and singular.unresolved


def test_parameter_data_preserves_original_units_uncertainty_and_json():
    spec = linear()
    names = (
        "negative_capacity_C",
        "positive_capacity_C",
        "known_capacity_C",
        "minimum_voltage_V",
        "maximum_voltage_V",
    )
    units = ("Ah", "mAh", "C", "mV", "V")
    values = (1.0, 1000.0, 3600.0, 3380.0, 3.92)
    table = ScientificTable(
        "soh",
        "parameters",
        tuple(
            QuantityColumn(n, u, "parameter", (0, 1e6), "declared SOH input")
            for n, u in zip(names, units, strict=True)
        ),
        (spec.negative_ocv.source,),
        (
            ScientificRow(
                "row",
                "synthetic-ocv",
                "g",
                "s",
                "parameters",
                {
                    n: InputValue(v, u, 0.001, "synthetic standard uncertainty")
                    for n, v, u in zip(names, values, units, strict=True)
                },
            ),
        ),
    )
    restored = ScientificTable.from_json(json.dumps(table.to_dict()))
    bound = ElectrodeSOHSpec.from_parameter_row(
        restored,
        "row",
        bindings=dict(zip(names, names, strict=True)),
        negative_ocv=spec.negative_ocv,
        positive_ocv=spec.positive_ocv,
    )
    assert bound.positive_capacity_C == 3600
    assert dict(bound.parameter_inputs)["positive_capacity_C"].unit == "mAh"
    assert dict(bound.parameter_inputs)["positive_capacity_C"].uncertainty == 0.001
    assert soh_spec_from_dict(json.loads(json.dumps(bound.to_dict()))) == bound
    assert solve_electrode_soh(bound).status == "unique"
    raw = bound.to_dict()
    raw["unrecognized"] = 1
    with pytest.raises(ValueError):
        soh_spec_from_dict(raw)
    half = HalfCellSOHSpec(spec.positive_ocv, 3600, 3.5, 4.0)
    assert soh_spec_from_dict(json.loads(json.dumps(half.to_dict()))) == half


@pytest.mark.parametrize("family", ["SPM", "SPMe", "DFN"])
def test_native_active_inventory_only_readonly_and_new_initial_config(family):
    kwargs = {
        "negative": ParticleSpec(cells=4),
        "positive": ParticleSpec(cells=4),
        "minimum_voltage_V": 3.38,
        "maximum_voltage_V": 3.92,
    }
    if family != "SPM":
        kwargs["electrolyte"] = BatteryElectrolyteSpec(cells=(2, 2, 2))
    spec = (BatteryDFNSpec if family == "DFN" else BatterySPMSpec)(**kwargs)
    initial = spec.initial_state()
    before = dict(initial.amounts_mol)
    result = soh_from_battery(spec, initial)
    assert result.status == "unique"
    assert initial.amounts_mol == before
    assert result.branch()["cyclable_lithium_C"] == pytest.approx(F * 0.03)
    fresh = battery_spec_at_soc(spec, result, 0.5)
    assert fresh.initial_negative_fraction == pytest.approx(0.5)
    assert fresh.initial_positive_fraction == pytest.approx(0.5)
    assert spec.initial_negative_fraction == 0.7
    assert initial.amounts_mol == before
    # An extra non-electrode stock is not cyclable electrode Li.
    with_extra = replace(
        initial, amounts_mol={**initial.amounts_mol, "unrelated-locked-stock": 100.0}
    )
    assert soh_from_battery(spec, with_extra).spec.known_capacity_C == result.spec.known_capacity_C
    with pytest.raises(ValueError, match="capacities"):
        battery_spec_at_soc(
            replace(spec, positive=replace(spec.positive, active_volume_m3=0.5e-6)), result, 0.5
        )


def test_ocv_interval_encloses_values_and_derivatives():
    c = curve()
    for a, b in ((0.01, 0.99), (0.3, 0.5), (0.7, 0.71)):
        for derivative in (False, True):
            interval = c.enclosure(a, b, derivative=derivative)
            assert all(
                interval.lower <= c.derivatives(float(x))[int(derivative)] <= interval.upper
                for x in np.linspace(a, b, 101)
            )
    with pytest.raises(ValueError, match="domain"):
        c.enclosure(0, 1)


@pytest.mark.parametrize("positive_capacity", [3600, 5400])
def test_known_capacity_at_exact_declared_faces_is_not_infeasible(positive_capacity):
    s = linear()
    q = 3600 * (0.99 - 0.01)
    yh, xh = 0.01, 0.99
    y0, x0 = yh + q / positive_capacity, 0.01
    spec = replace(
        s,
        positive_capacity_C=positive_capacity,
        known="cell_capacity",
        known_capacity_C=q,
        minimum_voltage_V=s.positive_ocv.value(y0) - s.negative_ocv.value(x0),
        maximum_voltage_V=s.positive_ocv.value(yh) - s.negative_ocv.value(xh),
    )
    result = solve_electrode_soh(spec)
    assert result.status == "unique"
    assert result.branch()["cell_capacity_C"] == pytest.approx(q)


def test_new_config_rejects_other_ocv_and_invalid_active_inventory():
    spec = BatterySPMSpec(minimum_voltage_V=3.38, maximum_voltage_V=3.92)
    result = soh_from_battery(spec)
    with pytest.raises(ValueError, match="OCV/window"):
        battery_spec_at_soc(replace(spec, positive_ocv_slope_V=-0.7), result, 0.5)
    state = spec.initial_state()
    bad = replace(state, amounts_mol={**state.amounts_mol, spec.negative.keys("negative")[0]: 1.0})
    with pytest.raises(ValueError, match="inventory"):
        soh_from_battery(spec, bad)


@pytest.mark.parametrize(
    "change",
    [
        {"negative_capacity_C": -1},
        {"known_capacity_C": True},
        {"minimum_voltage_V": 5},
        {"known": "fitted"},
        {"positive_ocv": None},
    ],
)
def test_invalid_inputs_reject_without_silent_fallback(change):
    with pytest.raises(ValueError):
        replace(linear(), **change)
