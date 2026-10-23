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
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.two_phase_balances import (
    two_phase_acceleration_gradient,
    two_phase_acceleration_pressure_drop,
    two_phase_gravity_gradient,
    two_phase_gravity_pressure_drop,
)
from chemworld.physchem.two_phase_holdup import TwoPhaseProperties
from chemworld.physchem.two_phase_regimes import (
    LogRegimeBoundary,
    TaitelDuklerBoundaries,
    mandhane_gregory_aziz_regime,
    taitel_dukler_regime,
)

P = TwoPhaseProperties(900, 10, 0.0005, 1.5e-5, 0.05)


def synthetic_boundary(log_value):
    source = InputSource(
        DataSourceProvenance(
            "synthetic-map",
            0,
            "Synthetic constant boundary for branch testing",
            "1",
            "Not a published physical chart",
        ),
        "literature",
    )
    columns = tuple(
        QuantityColumn(name, "dimensionless", "parameter", (-10, 10), "Synthetic log boundary")
        for name in ("X", "Y")
    )
    rows = tuple(
        ScientificRow(
            str(i),
            "synthetic-map",
            "branch-fixture",
            str(i),
            "parameters",
            {
                "X": InputValue(x, "dimensionless", None, "Synthetic"),
                "Y": InputValue(log_value, "dimensionless", None, "Synthetic"),
            },
        )
        for i, x in enumerate((-3, -1, 1, 3))
    )
    return LogRegimeBoundary(
        TabulatedScalar(
            ScientificTable("test-curve", "parameters", columns, (source,), rows),
            (("log10_X", "X", "dimensionless"),),
            "Y",
            "dimensionless",
        )
    )


def test_discrete_momentum_and_consistent_pure_phase_limits():
    common = {
        "mass_flux_kg_m2_s": 100,
        "inlet_quality": 0,
        "outlet_quality": 1,
        "inlet_void_fraction": 0,
        "outlet_void_fraction": 1,
    }
    assert two_phase_acceleration_pressure_drop(P, P, **common) == pytest.approx(
        100**2 * (1 / 10 - 1 / 900)
    )
    reverse = {
        **common,
        "inlet_quality": 1,
        "outlet_quality": 0,
        "inlet_void_fraction": 1,
        "outlet_void_fraction": 0,
    }
    assert two_phase_acceleration_pressure_drop(
        P, P, **reverse
    ) == -two_phase_acceleration_pressure_drop(P, P, **common)
    with pytest.raises(ValueError, match="zero cross section"):
        two_phase_acceleration_pressure_drop(P, P, **{**common, "inlet_quality": 0.1})


def test_expansion_recovers_pressure_and_compression_accelerates():
    common = {
        "mass_flux_kg_m2_s": 100,
        "vapor_quality": 0,
        "pressure_drop_gradient_Pa_m": 0,
        "liquid_volume_pressure_derivative_m3_kg_Pa": 0,
        "vapor_volume_pressure_derivative_m3_kg_Pa": 0,
        "quality_pressure_derivative_per_Pa": 0,
        "area_m2": 0.01,
    }
    expansion = two_phase_acceleration_gradient(P, **common, area_gradient_m=0.001)
    assert expansion == pytest.approx(-(100**2) / 900 * 0.001 / 0.01)
    assert two_phase_acceleration_gradient(P, **common, area_gradient_m=-0.001) == -expansion
    assert two_phase_acceleration_gradient(P, **common) == 0


def test_gravity_sign_zero_and_trapezoidal_density():
    for angle in (-90, 0, 90):
        gradient = two_phase_gravity_gradient(P, void_fraction=0.2, inclination_degrees=angle)
        assert (
            two_phase_gravity_pressure_drop(
                P, inlet_void_fraction=0.2, length_m=5, inclination_degrees=angle
            )
            == 5 * gradient
        )
    assert (
        two_phase_gravity_pressure_drop(
            P, inlet_void_fraction=0.2, length_m=0, inclination_degrees=90
        )
        == 0
    )
    assert two_phase_gravity_gradient(P, void_fraction=0, inclination_degrees=90) == 900 * 9.80665


def test_chart_data_stays_sourced_and_strict():
    curve = synthetic_boundary(-1)
    assert curve.value(1) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="domain"):
        curve.value(1e-5)
    assert curve.value(1e-5, allow_extrapolation=True) == pytest.approx(0.1)
    payload = json.loads(json.dumps(curve.source_curve.to_dict()))
    restored = LogRegimeBoundary(TabulatedScalar.from_dict(payload))
    assert restored.value(10) == pytest.approx(curve.value(10))
    assert restored.source_curve.table.sources[0].provenance.source_id == "synthetic-map"


def test_taitel_bubbly_branch_and_mandhane_si_velocities():
    boundaries = TaitelDuklerBoundaries(
        synthetic_boundary(-4), synthetic_boundary(0), synthetic_boundary(-4)
    )
    result = taitel_dukler_regime(
        P, boundaries, mass_flow_kg_s=1, diameter_m=0.05, vapor_quality=0.01
    )
    assert result.regime == "bubbly"
    assert result.parameters["X"] > 1.7917
    assert result.parameters["T"] > 1e-4
    result = mandhane_gregory_aziz_regime(P, mass_flow_kg_s=1, diameter_m=0.05, vapor_quality=0.1)
    assert result.liquid_superficial_velocity_m_s < 1
    assert result.vapor_superficial_velocity_m_s > 5
    assert json.loads(json.dumps(result.to_dict()))["regime"] == result.regime
