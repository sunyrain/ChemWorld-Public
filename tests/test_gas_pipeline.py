import json

import pytest

from chemworld.physchem.gas_pipeline import (
    GAS_PIPELINE_METHODS,
    EmpiricalGasPipelineSpec,
    solve_empirical_gas_pipeline,
)


def make_spec(method="Weymouth", **kwargs):
    args = {
        "method": method,
        "specific_gravity": 0.693,
        "average_temperature_K": 277.15,
        "applicability_note": "Development example; applicability is not universal",
    }
    if method in {"IGT", "Muller"}:
        args["viscosity_Pa_s"] = 1e-5
    return EmpiricalGasPipelineSpec(**{**args, **kwargs})


@pytest.mark.parametrize("method", GAS_PIPELINE_METHODS)
def test_pipeline_inverse_dimensions_and_reference(method):
    spec = make_spec(method)
    args = {
        "upstream_pressure_Pa": 3e5,
        "downstream_pressure_Pa": 2e5,
        "diameter_m": 0.05,
        "length_m": 100,
    }
    result = solve_empirical_gas_pipeline(spec, **args)
    for unknown in args:
        data = {**args, "reference_flow_m3_s": result.reference_flow_m3_s}
        expected = data.pop(unknown)
        assert getattr(solve_empirical_gas_pipeline(spec, **data), unknown) == pytest.approx(
            expected, rel=1e-8
        )
    assert result.reference_temperature_K == 288.7
    assert result.reference_pressure_Pa == 101325
    assert EmpiricalGasPipelineSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_low_pressure_gauge_datum_and_zero():
    spec = make_spec("Spitzglass_low", pressure_basis="gauge")
    args = {
        "upstream_pressure_Pa": 6700,
        "downstream_pressure_Pa": 0,
        "diameter_m": 0.15,
        "length_m": 50,
    }
    result = solve_empirical_gas_pipeline(spec, **args)
    shifted = solve_empirical_gas_pipeline(
        spec, **{**args, "upstream_pressure_Pa": 10700, "downstream_pressure_Pa": 4000}
    )
    assert shifted.reference_flow_m3_s == result.reference_flow_m3_s
    zero = solve_empirical_gas_pipeline(spec, **{**args, "upstream_pressure_Pa": 0})
    assert zero.reference_flow_m3_s == 0
    with pytest.raises(ValueError, match="indeterminate"):
        solve_empirical_gas_pipeline(
            spec,
            upstream_pressure_Pa=0,
            downstream_pressure_Pa=0,
            length_m=50,
            reference_flow_m3_s=0,
        )


def test_absolute_pressure_model_and_ignored_parameters_rejected():
    with pytest.raises(ValueError, match="absolute pressure"):
        make_spec(pressure_basis="gauge")
    with pytest.raises(ValueError, match="no E or Z"):
        make_spec("Oliphant", compressibility_factor=0.9)
    with pytest.raises(ValueError, match="no viscosity"):
        make_spec(viscosity_Pa_s=1e-5)
    with pytest.raises(ValueError, match="applicability"):
        make_spec(applicability_note="")
    spec = make_spec()
    with pytest.raises(ValueError, match="positive absolute outlet"):
        solve_empirical_gas_pipeline(
            spec, upstream_pressure_Pa=1e5, diameter_m=0.05, length_m=100, reference_flow_m3_s=1e5
        )
    with pytest.raises(ValueError, match="positive"):
        solve_empirical_gas_pipeline(
            spec, upstream_pressure_Pa=1e5, downstream_pressure_Pa=0, diameter_m=0.05, length_m=100
        )


def test_reference_flow_changes_with_reference_state():
    args = {
        "upstream_pressure_Pa": 3e5,
        "downstream_pressure_Pa": 2e5,
        "diameter_m": 0.05,
        "length_m": 100,
    }
    nominal = solve_empirical_gas_pipeline(make_spec(), **args)
    revised = solve_empirical_gas_pipeline(make_spec(reference_pressure_Pa=202650), **args)
    assert revised.reference_flow_m3_s == pytest.approx(nominal.reference_flow_m3_s / 2)
