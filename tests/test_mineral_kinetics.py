"""Finite slow kinetics, exact algebraic chemistry and physical stopping."""

from dataclasses import replace

import pytest

from chemworld.physchem.aqueous_mineral import (
    CarbonateEquilibriumSpec,
    carbonate_feed,
    element_amounts,
    solve_carbonate_equilibrium,
)
from chemworld.physchem.mineral_kinetics import (
    MineralGeometry,
    MineralKineticsSpec,
    advance_mineral,
)


def fixture():
    return carbonate_feed(
        water_kg=1,
        sodium_bicarbonate_mol=0.001,
        calcium_chloride_mol=0.001,
        sodium_chloride_mol=0.01,
        calcite_mol=0.01,
    )


def check_balance(feed, report):
    totals = element_amounts(feed)
    for point in report.trajectory:
        actual = element_amounts(point.aqueous.amounts_mol)
        for k, n in totals.items():
            assert abs(actual[k] - n) <= 1e-10 * max(n, 1e-12)
        assert min(point.aqueous.amounts_mol.values()) >= 0
        assert abs(point.aqueous.charge_balance_error_eq) <= 1e-11
        assert point.aqueous.ionic_strength_mol_kg <= 0.1


def test_finite_precipitation_approaches_equilibrium_and_holds_remaining_clock():
    aqueous, model, geometry = CarbonateEquilibriumSpec(), MineralKineticsSpec(), MineralGeometry()
    feed = fixture()
    report = advance_mineral(
        model, aqueous, feed, geometry, 1e6, evaluation_times_s=(0.1, 100, 1e4)
    )
    check_balance(feed, report)
    exact = solve_carbonate_equilibrium(aqueous, feed)
    assert report.actual_duration_s == 1e6
    assert report.chemical_stop_reason == "saturated"
    assert 0 < report.chemical_stop_time_s < 1e6
    assert report.final_aqueous.calcite_mol > feed["Calcite"]
    assert report.final_aqueous.calcite_mol == pytest.approx(exact.calcite_mol, abs=1e-10)
    assert report.final_aqueous.pH == pytest.approx(exact.pH, abs=1e-7)
    assert report.diagnostics["solver_calls"] == 1
    assert report.diagnostics["stationary_hold_s"] > 0


def test_finite_exhaustion_removes_area_and_keeps_actual_dissolved_atoms():
    feed = carbonate_feed(water_kg=1, calcite_mol=1e-6)
    geometry, model = MineralGeometry(0.04, 0.01), MineralKineticsSpec()
    report = advance_mineral(
        model, CarbonateEquilibriumSpec(), feed, geometry, 1e6, evaluation_times_s=(0.1, 1e4, 1e5)
    )
    check_balance(feed, report)
    assert report.chemical_stop_reason == "depleted"
    assert report.final_aqueous.calcite_mol == 0
    assert report.trajectory[-1].area_m2 == 0
    assert report.final_aqueous.amounts_mol["Ca++"] == pytest.approx(1e-6, abs=1e-16)
    assert report.final_aqueous.saturation_ratio < 1
    constant_drive_time = 3 * 1e-6 ** (1 / 3) * 0.01 ** (2 / 3) / (0.04 * 1e-6)
    assert report.chemical_stop_time_s == pytest.approx(constant_drive_time, rel=4e-6)
    assert report.actual_duration_s == 1e6


def test_supersaturated_seedless_solution_does_not_manufacture_surface():
    feed = carbonate_feed(
        water_kg=1,
        sodium_bicarbonate_mol=0.01,
        calcium_chloride_mol=0.001,
        sodium_chloride_mol=0.01,
    )
    report = advance_mineral(
        MineralKineticsSpec(), CarbonateEquilibriumSpec(), feed, MineralGeometry(0, 0), 1e6
    )
    check_balance(feed, report)
    assert report.final_aqueous.saturation_ratio > 1
    assert report.final_aqueous.calcite_mol == 0
    assert report.trajectory[-1].area_m2 == 0
    assert report.chemical_stop_reason == "no_seed"
    assert report.diagnostics["solver_calls"] == 0


@pytest.mark.parametrize("zero", ["rate", "area"])
def test_zero_rate_or_surface_holds_actual_solid_without_equilibrium_allocation(zero):
    model, geometry = MineralKineticsSpec(), MineralGeometry()
    if zero == "rate":
        model = replace(model, rate_constant_mol_m2_s=0)
    else:
        geometry = replace(geometry, reference_area_m2=0)
    feed = fixture()
    report = advance_mineral(model, CarbonateEquilibriumSpec(), feed, geometry, 100)
    check_balance(feed, report)
    assert report.final_aqueous.calcite_mol == feed["Calcite"]
    assert report.final_aqueous.saturation_ratio > 1
    assert report.actual_duration_s == 100


def test_initial_area_tangent_and_constant_drive_radius_are_independent_early_limits():
    model, geometry, feed = MineralKineticsSpec(), MineralGeometry(), fixture()
    report = advance_mineral(model, CarbonateEquilibriumSpec(), feed, geometry, 0.1)
    omega = report.trajectory[0].aqueous.saturation_ratio
    initial_n = feed["Calcite"]
    rate = -geometry.area(initial_n) * model.rate_constant_mol_m2_s * (1 - omega)
    tangent = rate * 0.1
    radius = (
        initial_n ** (1 / 3)
        - geometry.reference_area_m2
        * model.rate_constant_mol_m2_s
        / (3 * geometry.reference_amount_mol ** (2 / 3))
        * (1 - omega)
        * 0.1
    ) ** 3 - initial_n
    extent = report.final_aqueous.calcite_mol - initial_n
    assert abs(extent - tangent) / max(abs(tangent), 1e-12) <= 1e-4
    assert abs(extent - radius) / max(abs(radius), 1e-12) <= 1e-4


def test_unsupported_geometry_temperature_and_times_fail_explicitly():
    with pytest.raises(ValueError):
        MineralGeometry(-1, 0.01)
    with pytest.raises(ValueError):
        MineralGeometry(0.01, 0)
    with pytest.raises(ValueError):
        MineralKineticsSpec(rate_constant_mol_m2_s=-1)
    with pytest.raises(ValueError, match="temperature"):
        advance_mineral(
            MineralKineticsSpec(),
            CarbonateEquilibriumSpec(),
            fixture(),
            MineralGeometry(),
            1,
            temperature_K=310,
        )
    with pytest.raises(ValueError, match="times"):
        advance_mineral(
            MineralKineticsSpec(),
            CarbonateEquilibriumSpec(),
            fixture(),
            MineralGeometry(),
            1,
            evaluation_times_s=(2,),
        )
