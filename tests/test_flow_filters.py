import json
from dataclasses import replace

import pytest
from tests.test_tabulated_functions import function

from chemworld.physchem.flow_filters import (
    FilterCurve,
    filter_loss_coefficient,
    filter_pressure_drop,
)
from chemworld.physchem.transport import FluidState


def test_sourced_quadratic_curve_and_endpoint_contract():
    curve = FilterCurve(function({"open_fraction": (.2, .4, .6, .8)},
                                 lambda x: (1-x)**2, output_unit="dimensionless"), "quadratic")
    restored = FilterCurve.from_dict(json.loads(json.dumps(curve.to_dict())))
    assert restored.value(.5) == pytest.approx(.25)
    assert restored.table_function.table.sources == curve.table_function.table.sources
    with pytest.raises(ValueError, match="outside"):
        curve.value(.1)
    assert replace(curve, endpoint_policy="clip").value(.1) == pytest.approx(.64)


def test_original_fluid_properties_and_approach_velocity_basis():
    beta = FilterCurve(function({"reynolds": (10, 1000)}, lambda re: 1+re/1000,
                                output_unit="dimensionless"))
    fluid = FluidState(1000, .001)
    args = {"approach_velocity_m_s": .1, "open_fraction": .5,
            "characteristic_length_m": .001, "primary_curve": beta}
    first = filter_pressure_drop("round_screen", fluid, **args)
    assert first.reynolds == 100
    assert first.pressure_drop_Pa == pytest.approx(16.5)
    second = filter_pressure_drop("round_screen", replace(fluid, viscosity_Pa_s=.0005), **args)
    assert second.reynolds == 200 and second.pressure_drop_Pa > first.pressure_drop_Pa
    # Gap velocity would be v/alpha and would incorrectly multiply loss by four.
    assert first.pressure_drop_Pa == pytest.approx(.5*1000*.1**2*first.loss_coefficient)
    assert first.source_tables == ("parameter-function",)


def test_explicit_regimes_geometry_and_closed_form_limits():
    assert filter_loss_coefficient("open_mesh", open_fraction=1, reynolds=1000) == 0
    short = filter_loss_coefficient("square_grill", open_fraction=.5)
    long = filter_loss_coefficient("square_grill", open_fraction=.5,
                                   thickness_m=.15, hydraulic_diameter_m=.002,
                                   darcy_friction_factor=.02)
    assert long-short == pytest.approx(.02*.15/(.5**2*.002))
    with pytest.raises(ValueError, match="together"):
        filter_loss_coefficient("square_grill", open_fraction=.5, thickness_m=.1)
    with pytest.raises(ValueError, match="Reynolds >500"):
        filter_loss_coefficient("open_mesh", open_fraction=.9, reynolds=100)
    with pytest.raises(ValueError, match="sourced"):
        filter_loss_coefficient("square_screen", open_fraction=.5)
    with pytest.raises(ValueError, match="inclination"):
        filter_loss_coefficient("square_grill", open_fraction=.5, inclination_degrees=30)
    with pytest.raises(ValueError, match="computes Reynolds"):
        filter_pressure_drop("round_screen", FluidState(1000, .001),
                              approach_velocity_m_s=.1, open_fraction=.5, reynolds=100)
