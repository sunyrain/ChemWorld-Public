"""Native virial declarations, explicit polar choices and unit-bearing reports."""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, cast

from chemworld.physchem.property_equations import evaluate_correlation
from chemworld.physchem.property_reports import PropertyEvaluation, ValidityPolicy
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.virial_kernels import (
    VIRIAL_METHODS,
    meng_polar_parameter,
    tsonopoulos_polar_parameters,
    virial_derivatives,
)


def virial_correlation(
    method: str, *, parameters: Mapping[str, float], temperature_range_K: Sequence[float],
    source_note: str, correlation_id: str | None = None, polar: Mapping[str, Any] | None = None,
) -> PropertyCorrelation:
    if method not in VIRIAL_METHODS:
        raise ValueError("unknown pure virial method")
    if not isinstance(source_note, str) or not source_note.strip():
        raise ValueError("virial correlation requires an explicit source note")
    bounds = tuple(temperature_range_K)
    if (
        len(bounds) != 2
        or any(isinstance(v, bool) or not isfinite(v) for v in bounds)
        or not 0 < bounds[0] < bounds[1]
    ):
        raise ValueError("virial application interval must be positive finite and ordered")
    p = dict(parameters)
    polar = None if polar is None else dict(polar)
    if polar is not None:
        if {"a", "b"} & p.keys():
            raise ValueError("select explicit polar parameters or a polar estimator, not both")
        if method == "Tsonopoulos_extended":
            polar.setdefault("convention", "documented")
            p.update(tsonopoulos_polar_parameters(Tc=p["Tc"], Pc=p["Pc"], **polar))
        elif method == "Meng":
            p["a"] = meng_polar_parameter(Tc=p["Tc"], Pc=p["Pc"], **polar)
        else:
            raise ValueError("this method has no polar estimator")
    for T in bounds:
        virial_derivatives(method, T, p, order=0)
    spec = VIRIAL_METHODS[method]
    return PropertyCorrelation(
        correlation_id=correlation_id or "virial_" + method.lower(),
        equation_id="virial_" + method.lower(),
        coefficients=p,
        input_units={"temperature": "K"},
        output_unit=spec.unit,
        property_id=spec.property_id,
        validity_ranges={"temperature": bounds},
        source_note=source_note,
        metadata={
            "phase": "gas",
            "physical_relation": "density_virial_coefficient",
            "reference_method": method,
            "polar_selection": polar,
            "temperature_derivative_path": "fixed_supplied_parameters",
            "coefficient_basis": "public_call_recovery_in_documented_basis"
            if method in {"Xiang", "Liu_Xiang"}
            else "public_mathematical_documentation",
            "qualification": "development_mathematical_consistency; no material accuracy",
        },
    )


def virial_temperature_report(
    correlation: PropertyCorrelation, *, temperature_K: float, order: int = 3,
    validity_policy: ValidityPolicy = "raise",
) -> tuple[PropertyEvaluation, ...]:
    """Existing PropertyEvaluation records for the value and T derivatives."""
    from chemworld.physchem.correlation_families import CORRELATION_FAMILIES, CorrelationFamily

    family = CORRELATION_FAMILIES.get(correlation.equation_id)
    sourced_crc = correlation.equation_id == "crc_second_virial"
    if not sourced_crc and (family is None or not family.virial_method):
        raise ValueError("requires a pure virial declaration")
    value = evaluate_correlation(
        correlation, temperature_K=temperature_K, validity_policy=validity_policy
    )
    if sourced_crc:
        from chemworld.physchem.caloric_math import derivative_values
        from chemworld.physchem.property_equations import _evaluate_equation

        if correlation.input_units != {"temperature": "K"}:
            raise ValueError("CRC virial derivatives require Kelvin input")
        if set(correlation.coefficients) != {f"a{i}" for i in range(1, 6)}:
            raise ValueError("CRC virial derivatives require all five explicit coefficients")
        values = derivative_values(
            lambda t: _evaluate_equation(correlation, inputs={"temperature": t},
                temperature_K=t, pressure_Pa=None, molecular_weight_g_mol=None),
            temperature_K, order=order,
        )
    else:
        values = virial_derivatives(
            cast(CorrelationFamily, family).virial_method,
            temperature_K, correlation.coefficients, order=order,
        )
    return (
        value,
        *(
            PropertyEvaluation(
                property_id=correlation.property_id + f"_dT{i}",
                correlation_id=correlation.correlation_id,
                equation_id=correlation.equation_id,
                value=number,
                unit=f"({correlation.output_unit})/K" + (f"^{i}" if i > 1 else ""),
                inputs=value.inputs,
                warnings=value.warnings,
            )
            for i, number in enumerate(values[1:], 1)
        ),
    )
