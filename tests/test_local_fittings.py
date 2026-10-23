import json
from math import pi

import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.fitting_curves import FittingCurve, FittingCurves
from chemworld.physchem.local_fittings import (
    entrance_loss_coefficient,
    staged_diffuser_loss,
    transition_loss_coefficient,
)
from chemworld.physchem.transport import FluidState, PipeSpec, pipe_pressure_drop


def test_caller_curve_data_roundtrip_and_missing_chart_scope():
    source = InputSource(
        DataSourceProvenance("test", 0, "Synthetic curve", "1", "Test"), "synthetic"
    )
    columns = tuple(
        QuantityColumn(name, "dimensionless", "parameter", (-100, 100), "Synthetic")
        for name in ("x", "value")
    )
    rows = tuple(
        ScientificRow(
            str(i),
            "test",
            "curve",
            str(i),
            "parameters",
            {
                "x": InputValue(x, "dimensionless", None, "Synthetic"),
                "value": InputValue(y, "dimensionless", None, "Synthetic"),
            },
        )
        for i, (x, y) in enumerate(((0, 0.5), (0.1, 0.1)))
    )
    curve = FittingCurve(
        ScientificTable("curve", "parameters", columns, (source,), rows), kind="linear"
    )
    curves = FittingCurves.from_dict(
        json.loads(json.dumps(FittingCurves({"entrance_rounded_Crane": curve}).to_dict()))
    )
    assert entrance_loss_coefficient(
        "rounded", diameter_m=0.1, rounding_m=0.005, method="Crane", curves=curves
    ) == pytest.approx(0.3)
    assert curves.to_dict()["entrance_rounded_Crane"]["table"] == curve.table.to_dict()
    with pytest.raises(ValueError, match="caller fitting curves"):
        entrance_loss_coefficient("rounded", diameter_m=0.1, rounding_m=0.005, method="Crane")


def test_staged_diffuser_uses_one_velocity_basis_and_actual_pipe_pressure():
    diameters = (0.05, 0.075, 0.1)
    lengths = (0.2, 0.3)
    physical = staged_diffuser_loss(diameters, lengths, friction_factor=0.02)
    raw = staged_diffuser_loss(
        diameters, lengths, friction_factor=0.02, common_velocity_basis=False
    )
    assert 0 < physical < raw
    fluid, flow = FluidState(1000, 0.001), 0.001
    expected = 0.0
    for i, length in enumerate(lengths):
        local = transition_loss_coefficient(
            "diffuser conical",
            inlet_diameter_m=diameters[i],
            outlet_diameter_m=diameters[i + 1],
            length_m=length,
            friction_factor=0.02,
        )
        expected += local * 0.5 * fluid.density_kg_m3 * (flow / (pi * diameters[i] ** 2 / 4)) ** 2
    result = pipe_pressure_drop(
        PipeSpec(1, diameters[0], fittings_loss_coefficient=physical),
        fluid,
        volumetric_flow_m3_s=flow,
    )
    assert result.pressure_drop_fittings_Pa == pytest.approx(expected)


def test_formula_geometry_and_zero_entrance_limits():
    assert entrance_loss_coefficient("exit") == 1
    assert (
        transition_loss_coefficient(
            "contraction sharp", inlet_diameter_m=0.1, outlet_diameter_m=0.1
        )
        == 0
    )
    with pytest.raises(ValueError, match="ordering"):
        transition_loss_coefficient(
            "contraction conical",
            inlet_diameter_m=0.1,
            outlet_diameter_m=0.2,
            angle_degrees=30,
            friction_factor=0.02,
        )
    with pytest.raises(ValueError, match="Reynolds"):
        transition_loss_coefficient(
            "diffuser sharp", inlet_diameter_m=0.1, outlet_diameter_m=0.2, method="Hooper"
        )
