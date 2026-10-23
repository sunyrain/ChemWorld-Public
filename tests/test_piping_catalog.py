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
from chemworld.physchem.piping_catalog import (
    PipeScheduleCatalog,
    WireGaugeCatalog,
    piping_erosional_velocity,
)
from chemworld.physchem.transport import FluidState, pipe_pressure_drop


def table(name, definitions, values):
    source = InputSource(
        DataSourceProvenance("synthetic", 0, "Self-authored dimensional fixture", "1", "Synthetic"),
        "synthetic",
    )
    columns = tuple(
        QuantityColumn(k, unit, "parameter", (0, 1000), "Synthetic") for k, unit in definitions
    )
    rows = tuple(
        ScientificRow(
            str(i),
            "synthetic",
            "test",
            str(i),
            "parameters",
            {
                k: InputValue(v, unit, None, "Synthetic")
                for (k, unit), v in zip(definitions, row, strict=True)
            },
        )
        for i, row in enumerate(values)
    )
    return ScientificTable(name, "parameters", columns, (source,), rows)


def test_pipe_selection_consumes_native_bore_and_current_fluid():
    catalog = PipeScheduleCatalog(
        table(
            "pipe",
            (
                ("nominal_size", "dimensionless"),
                ("inner_diameter", "mm"),
                ("outer_diameter", "mm"),
                ("wall_thickness", "mm"),
            ),
            ((1, 10, 12, 1), (2, 20, 24, 2)),
        )
    )
    copy = PipeScheduleCatalog.from_dict(json.loads(json.dumps(catalog.to_dict())))
    selection = copy.select("test", minimum_inner_diameter_m=0.015)
    assert selection.nominal_size == 2 and selection.source_id == "synthetic"
    pipe = selection.pipe_spec(length_m=1)
    assert pipe.diameter_m == 0.02
    fluid = FluidState(1000, 0.001)
    result = pipe_pressure_drop(pipe, fluid, volumetric_flow_m3_s=1e-6)
    # Laminar Hagen-Poiseuille at the actual selected bore.
    expected = 128 * 0.001 * 1 * 1e-6 / (pi * 0.02**4)
    assert result.pressure_drop_total_Pa == pytest.approx(expected)
    changed = pipe_pressure_drop(pipe, FluidState(1000, 0.002), volumetric_flow_m3_s=1e-6)
    assert changed.pressure_drop_total_Pa == pytest.approx(2 * expected)
    with pytest.raises(ValueError, match="exactly one"):
        copy.select("test", nominal_size=1, minimum_inner_diameter_m=0.001)
    with pytest.raises(ValueError, match="outside"):
        copy.select("test", minimum_outer_diameter_m=1)


def test_wire_zero_identifier_nominal_and_si_columns_and_lower_domain():
    catalog = WireGaugeCatalog(
        table(
            "wire",
            (("gauge", "dimensionless"), ("thickness", "mm"), ("nominal_thickness", "inch")),
            ((0, 2.5, 0.1), (1, 1.3, 0.05)),
        )
    )
    copy = WireGaugeCatalog.from_dict(json.loads(json.dumps(catalog.to_dict())))
    zero = copy.from_gauge("test", 0)
    assert zero.thickness_m == 0.0025
    assert zero.nominal_thickness_m == pytest.approx(0.00254)
    assert copy.from_thickness("test", 0.00254).gauge == 0
    low, high = 0.00127, 0.00254
    assert copy.from_thickness("test", low + 0.05 * (high - low)).gauge == 1
    assert copy.from_thickness("test", low + 0.2 * (high - low)).gauge == 0
    assert (
        copy.from_thickness("test", low + 0.05 * (high - low), undersize_gap_fraction=0).gauge == 0
    )
    with pytest.raises(ValueError, match="outside"):
        copy.from_thickness("test", 0.001)


def test_erosion_screen_reads_current_density_and_explicit_factor():
    value = piping_erosional_velocity(FluidState(1000, 0.001), empirical_factor_imperial=100)
    assert value == pytest.approx(3.8576728004)
    assert (
        piping_erosional_velocity(FluidState(250, 0.001), empirical_factor_imperial=100)
        == 2 * value
    )
    with pytest.raises(ValueError, match="positive"):
        piping_erosional_velocity(FluidState(1000, 0.001), empirical_factor_imperial=0)
