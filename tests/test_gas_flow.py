"""Admissible branches and five engineering inverses of one gas-pipe model."""

import json

import pytest

from chemworld.physchem.gas_flow import (
    IsothermalGasState,
    isothermal_critical_pressure_ratio,
    isothermal_gas_pipe,
    solve_isothermal_gas_pipe,
)


def geometry():
    return {"diameter_m": .05, "length_m": 100, "darcy_friction_factor": .02}


@pytest.mark.parametrize("field", ["mass_flow_kg_s", "upstream_pressure_Pa",
    "downstream_pressure_Pa", "diameter_m", "length_m"])
def test_five_unknowns_share_one_equation_and_density_contract(field):
    gas = IsothermalGasState(300, 287.05, .98)
    point = isothermal_gas_pipe(gas, upstream_pressure_Pa=1e6,
        downstream_pressure_Pa=8e5, **geometry())
    inputs = {**geometry(), "upstream_pressure_Pa": 1e6,
        "downstream_pressure_Pa": 8e5, "mass_flow_kg_s": point.mass_flow_kg_s}
    expected = inputs.pop(field)
    inverse = solve_isothermal_gas_pipe(gas, **inputs)
    assert getattr(inverse, field) == pytest.approx(expected, rel=5e-8)
    assert inverse.inlet_density_kg_m3 == pytest.approx(
        inverse.upstream_pressure_Pa / (.98 * 287.05 * 300), rel=1e-12)


def test_choking_is_a_branch_boundary_not_a_clipped_mass_request():
    gas = IsothermalGasState(300, 287.05)
    ratio = isothermal_critical_pressure_ratio(**geometry())
    point = isothermal_gas_pipe(gas, upstream_pressure_Pa=1e6,
        downstream_pressure_Pa=1e6 * ratio, **geometry())
    assert point.choked
    assert point.mass_flow_kg_s == point.maximum_mass_flow_kg_s
    with pytest.raises(ValueError, match="choking"):
        isothermal_gas_pipe(gas, upstream_pressure_Pa=1e6,
            downstream_pressure_Pa=1e6 * ratio * .999, **geometry())
    with pytest.raises(ValueError, match="capacity"):
        solve_isothermal_gas_pipe(gas, upstream_pressure_Pa=1e6,
            mass_flow_kg_s=point.mass_flow_kg_s * 1.001, **geometry())
    inverse = solve_isothermal_gas_pipe(gas, upstream_pressure_Pa=1e6,
        mass_flow_kg_s=point.mass_flow_kg_s, **geometry())
    assert inverse.downstream_pressure_Pa == pytest.approx(ratio * 1e6, rel=2e-12)


def test_equal_pressure_has_no_flow_and_inverse_does_not_invent_a_length():
    gas = IsothermalGasState(300, 287.05)
    point = isothermal_gas_pipe(gas, upstream_pressure_Pa=1e6,
        downstream_pressure_Pa=1e6, **geometry())
    assert point.mass_flow_kg_s == 0
    for name in ("upstream_pressure_Pa", "downstream_pressure_Pa"):
        inverse = solve_isothermal_gas_pipe(gas, **{name: 1e6},
            mass_flow_kg_s=0, **geometry())
        assert inverse.upstream_pressure_Pa == inverse.downstream_pressure_Pa == 1e6
    with pytest.raises(ValueError, match="positive mass"):
        solve_isothermal_gas_pipe(gas, upstream_pressure_Pa=1e6, downstream_pressure_Pa=1e6,
            mass_flow_kg_s=0, diameter_m=.05, darcy_friction_factor=.02)


@pytest.mark.parametrize("resistance", [1e-12, 1e-8, 1, 1e4, 1e6])
def test_critical_solver_preserves_its_dimensionless_equation(resistance):
    from math import log

    ratio = isothermal_critical_pressure_ratio(
        darcy_friction_factor=.02, diameter_m=.1, length_m=resistance * 5)
    # Cancellation near zero is tested through the leading analytical limit.
    if resistance < 1e-7:
        assert -2 * log(ratio) == pytest.approx((2 * resistance)**.5, rel=1e-4)
    else:
        assert (ratio**-2 - 1 + 2 * log(ratio)) / resistance == pytest.approx(1, rel=2e-10)


def test_pressure_units_scale_mass_and_density_together():
    gas = IsothermalGasState(300, 287.05)
    low = isothermal_gas_pipe(gas, upstream_pressure_Pa=1e5,
        downstream_pressure_Pa=8e4, **geometry())
    high = isothermal_gas_pipe(gas, upstream_pressure_Pa=1e6,
        downstream_pressure_Pa=8e5, **geometry())
    assert high.mass_flow_kg_s / low.mass_flow_kg_s == pytest.approx(10)
    assert high.inlet_density_kg_m3 / low.inlet_density_kg_m3 == pytest.approx(10)
    assert IsothermalGasState.from_dict(json.loads(json.dumps(gas.to_dict()))) == gas


def test_unknown_count_and_nonphysical_inputs_rejected():
    gas = IsothermalGasState(300, 287.05)
    with pytest.raises(ValueError, match="exactly one"):
        solve_isothermal_gas_pipe(gas, upstream_pressure_Pa=1e6, **geometry())
    with pytest.raises(ValueError, match="forward"):
        isothermal_gas_pipe(gas, upstream_pressure_Pa=1e5,
            downstream_pressure_Pa=2e5, **geometry())
    with pytest.raises(ValueError, match="positive"):
        IsothermalGasState(300, -287.05)
