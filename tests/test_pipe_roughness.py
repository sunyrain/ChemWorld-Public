import json

import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.pipe_roughness import (
    PipeRoughnessCatalog,
    farshad_pipe_roughness,
    smooth_pipe_observation_pairs,
)
from chemworld.physchem.single_phase_pressure import (
    single_phase_acceleration_gradient,
    single_phase_acceleration_pressure_drop,
    single_phase_friction_pressure_drop,
    single_phase_gravity_pressure_drop,
)
from chemworld.physchem.transport import FluidState


def source():
    return InputSource(
        DataSourceProvenance(
            "synthetic", 0, "Synthetic dimensional fixture", "1", "No measured data implied"
        ),
        "literature",
    )


def catalog():
    rows = tuple(
        ScientificRow(
            str(i),
            "synthetic",
            material,
            material + ":" + statistic,
            "parameters",
            {"roughness": InputValue(value, "mm", None, "Synthetic")},
        )
        for i, (material, statistic, value) in enumerate(
            (
                ("Steel", "average", 1),
                ("Weathered", "min", 0.1),
                ("Weathered", "max", 0.3),
                ("Smooth", "average", 0),
            )
        )
    )
    return PipeRoughnessCatalog(
        ScientificTable(
            "roughness",
            "parameters",
            (QuantityColumn("roughness", "mm", "parameter", (0, 10), "Absolute roughness"),),
            (source(),),
            rows,
        )
    )


def test_source_units_missing_statistics_and_roundtrip():
    original = catalog()
    restored = PipeRoughnessCatalog.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.evaluate("Steel").roughness_m == 0.001
    assert restored.evaluate("Weathered").roughness_m == pytest.approx(0.0003)
    assert restored.evaluate("Weathered", optimism=True).roughness_m == pytest.approx(0.0001)
    assert restored.evaluate("Smooth").roughness_m == 0
    assert restored.evaluate("Steel").source_id == "synthetic"
    with pytest.raises(ValueError, match="diameter"):
        restored.evaluate("Steel", diameter_m=0.0001)


def test_fuzzy_selection_is_explicit_and_farshad_keeps_inch_basis():
    data = catalog()
    with pytest.raises(ValueError, match="suggestions"):
        data.evaluate("steeel")
    selected = data.evaluate("steeel", allow_fuzzy_match=True)
    assert selected.material == "Steel" and selected.fuzzy_match_used
    assert farshad_pipe_roughness(0.1, 0.0014, -1) == pytest.approx(0.0254 * 0.0014)
    with pytest.raises(ValueError, match="microchannel"):
        farshad_pipe_roughness(1e-6, 0.0014, -1)


def test_single_phase_pressure_has_signed_energy_and_momentum_limits():
    common = {
        "mass_flow_kg_s": 0.1,
        "inlet_density_kg_m3": 1000,
        "outlet_density_kg_m3": 1000,
        "inlet_diameter_m": 0.01,
    }
    assert single_phase_acceleration_pressure_drop(**common) == 0
    assert single_phase_acceleration_pressure_drop(**common, outlet_diameter_m=0.02) < 0
    assert single_phase_acceleration_gradient(
        mass_flux_kg_m2_s=100,
        density_kg_m3=1000,
        volume_pressure_derivative_m3_kg_Pa=0,
        pressure_drop_gradient_Pa_m=0,
        area_m2=0.01,
        area_gradient_m=0.001,
    ) == pytest.approx(-1)
    assert single_phase_gravity_pressure_drop(1000, length_m=2, inclination_degrees=-90) == -19613.3
    assert (
        single_phase_friction_pressure_drop(
            FluidState(1000, 0.001), mass_flow_kg_s=0.1, diameter_m=0.01, length_m=0
        )
        == 0
    )


def test_smooth_pipe_observations_are_read_not_refitted():
    rows = tuple(
        ScientificRow(
            str(i),
            "synthetic",
            "pipe",
            str(i),
            "test",
            {
                "reynolds": InputValue(reynolds, "dimensionless", None, "Synthetic"),
                "darcy_factor": InputValue(factor, "dimensionless", None, "Synthetic"),
            },
        )
        for i, (reynolds, factor) in enumerate(((1500, 0.05), (2000, 0.035), (2500, 0.04)))
    )
    table = ScientificTable(
        "measured",
        "measurements",
        (
            QuantityColumn("reynolds", "dimensionless", "condition", (0, 1e5), "Measured Re"),
            QuantityColumn("darcy_factor", "dimensionless", "observation", (0, 1), "Measured f_D"),
        ),
        (source(),),
        rows,
    )
    assert smooth_pipe_observation_pairs(table) == ((1500, 0.05), (2000, 0.035), (2500, 0.04))
    assert table.rows[1].values["darcy_factor"].value == 0.035
