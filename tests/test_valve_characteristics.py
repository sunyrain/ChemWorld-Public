import json

import pytest
from tests.test_tabulated_functions import function

from chemworld.physchem.fluids_devices import LiquidControlValveSpec, liquid_valve_flow
from chemworld.physchem.transport import FluidState
from chemworld.physchem.valve_characteristics import (
    ValveOpeningCurve,
    convert_valve_coefficient,
    gas_valve_choke_pressure,
    liquid_valve_choke_pressure,
)


def test_coefficients_units_and_choke_inverses():
    av = convert_valve_coefficient(1, from_scale="Kv", to_scale="Av")
    assert 0 < av < 0.001
    assert convert_valve_coefficient(av, from_scale="Av", to_scale="Kv") == pytest.approx(1)
    p2 = gas_valve_choke_pressure(
        pressure_ratio_factor=0.6, heat_capacity_ratio=1.4, upstream_pressure_Pa=1e6
    )
    assert p2 == pytest.approx(4e5)
    assert gas_valve_choke_pressure(
        pressure_ratio_factor=0.6, heat_capacity_ratio=1.4, downstream_pressure_Pa=p2
    ) == pytest.approx(1e6)
    with pytest.raises(ValueError, match="positive outlet"):
        gas_valve_choke_pressure(
            pressure_ratio_factor=1, heat_capacity_ratio=1.4, upstream_pressure_Pa=1e6
        )
    with pytest.raises(ValueError, match="indeterminate"):
        liquid_valve_choke_pressure(
            saturation_pressure_Pa=1000,
            critical_pressure_Pa=2e7,
            effective_recovery_factor=1,
            downstream_pressure_Pa=1000,
        )


def test_sourced_curve_roundtrip_and_actual_liquid_valve_consumption():
    table = function(
        {"opening": (0, 0.25, 0.5, 0.75, 1)},
        lambda opening: opening**2,
        output_unit="dimensionless",
    )
    curve = ValveOpeningCurve("equal_percentage", table)
    restored = ValveOpeningCurve.from_dict(json.loads(json.dumps(curve.to_dict())))
    assert restored.value(0.3) == pytest.approx(0.09)
    full = LiquidControlValveSpec(50, 0.05, 0.05, 0.05, 0.9, 0.5, source_note="test full-open Kv")
    args = {
        "upstream_pressure_Pa": 3e5,
        "downstream_pressure_Pa": 2e5,
        "saturation_pressure_Pa": 1000,
        "critical_pressure_Pa": 2e7,
    }
    fluid = FluidState(1000, 0.001)
    full_flow = liquid_valve_flow(full, fluid, **args).flow_m3_s
    half = restored.apply(full, 0.5)
    assert liquid_valve_flow(half, fluid, **args).flow_m3_s == pytest.approx(full_flow * 0.25)
    assert liquid_valve_flow(restored.apply(full, 0), fluid, **args).flow_m3_s == 0
    with pytest.raises(ValueError, match="sourced curve"):
        ValveOpeningCurve("quick_opening")
    with pytest.raises(ValueError, match="opening"):
        curve.value(1.1)
