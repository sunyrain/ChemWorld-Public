"""Property-correlation equation evaluator utilities."""

from __future__ import annotations

from math import exp, isfinite, log

from chemworld.foundation.units import convert_value
from chemworld.physchem.property_reports import (
    R_J_PER_MOL_K,
    STANDARD_PRESSURE_PA,
    PropertyEvaluation,
    ValidityPolicy,
)
from chemworld.physchem.specs import PropertyCorrelation


def evaluate_correlation(
    correlation: PropertyCorrelation,
    *,
    temperature_K: float,
    pressure_Pa: float | None = None,
    molecular_weight_g_mol: float | None = None,
    validity_policy: ValidityPolicy = "warn",
) -> PropertyEvaluation:
    """Evaluate a supported property correlation.

    Inputs are supplied in canonical benchmark units. Each correlation declares
    the units its coefficients expect; this function performs the conversion.
    """

    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if pressure_Pa is not None and pressure_Pa <= 0:
        raise ValueError("pressure_Pa must be positive")

    inputs = {"temperature": _convert_input(temperature_K, "K", correlation, "temperature")}
    if pressure_Pa is not None:
        inputs["pressure"] = _convert_input(pressure_Pa, "Pa", correlation, "pressure")
    warnings = _validity_warnings(correlation, inputs)
    if warnings and validity_policy == "raise":
        raise ValueError("; ".join(warnings))

    value = _evaluate_equation(
        correlation,
        inputs=inputs,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        molecular_weight_g_mol=molecular_weight_g_mol,
    )
    _ensure_finite_positive(value, correlation)
    return PropertyEvaluation(
        property_id=correlation.property_id,
        correlation_id=correlation.correlation_id,
        equation_id=correlation.equation_id,
        value=value,
        unit=correlation.output_unit,
        inputs=inputs,
        warnings=() if validity_policy == "ignore" else warnings,
        )


def _evaluate_equation(
    correlation: PropertyCorrelation,
    *,
    inputs: dict[str, float],
    temperature_K: float,
    pressure_Pa: float | None,
    molecular_weight_g_mol: float | None,
) -> float:
    equation = correlation.equation_id
    coeffs = correlation.coefficients
    T = inputs["temperature"]
    if equation == "antoine":
        base = coeffs.get("base", 10.0)
        if T + coeffs["C"] <= 0:
            raise ValueError("Antoine vapor pressure singularity: T + C must be positive")
        return base ** (coeffs["A"] - coeffs["B"] / (T + coeffs["C"]))
    if equation == "wagner":
        Tc = coeffs["Tc"]
        Pc = coeffs["Pc"]
        if temperature_K >= Tc:
            raise ValueError("Wagner vapor pressure requires T < Tc")
        tau = 1.0 - temperature_K / Tc
        exponent = (
            coeffs["a"] * tau
            + coeffs["b"] * tau**1.5
            + coeffs["c"] * tau**3.0
            + coeffs["d"] * tau**6.0
        ) / (1.0 - tau)
        return Pc * exp(exponent)
    if equation == "dippr101_vapor_pressure":
        return exp(
            coeffs["A"]
            + coeffs["B"] / T
            + coeffs["C"] * log(T)
            + coeffs["D"] * T ** coeffs["E"]
        )
    if equation == "cp_polynomial":
        return _cp_polynomial(T, coeffs)
    if equation == "watson_hvap":
        Tc = coeffs["Tc"]
        T_ref = coeffs["T_ref"]
        if temperature_K >= Tc:
            return 0.0
        ratio = (1.0 - temperature_K / Tc) / (1.0 - T_ref / Tc)
        return coeffs["Hvap_ref"] * ratio ** coeffs.get("exponent", 0.38)
    if equation == "constant_phase_change_enthalpy":
        return coeffs["delta_h"]
    if equation == "linear_liquid_density":
        rho = coeffs["rho_ref"] * (1.0 - coeffs.get("alpha", 0.0) * (T - coeffs["T_ref"]))
        return rho
    if equation == "ideal_gas_density":
        if molecular_weight_g_mol is None:
            raise ValueError("ideal_gas_density requires molecular_weight_g_mol")
        pressure = pressure_Pa if pressure_Pa is not None else STANDARD_PRESSURE_PA
        mw_kg_mol = molecular_weight_g_mol / 1000.0
        return pressure * mw_kg_mol / (R_J_PER_MOL_K * temperature_K)
    if equation == "rackett_liquid_molar_volume":
        Tc = coeffs["Tc"]
        Pc = coeffs["Pc"]
        Zc = coeffs["Zc"]
        if Tc <= 0 or Pc <= 0 or Zc <= 0:
            raise ValueError("Rackett requires positive Tc, Pc, and Zc")
        if temperature_K >= Tc:
            raise ValueError("Rackett liquid molar volume requires T < Tc")
        exponent = 1.0 + (1.0 - temperature_K / Tc) ** (2.0 / 7.0)
        return R_J_PER_MOL_K * Tc / Pc * Zc**exponent
    if equation == "crc_second_virial":
        t = 298.15 / T - 1.0
        return 1e-6 * (
            coeffs.get("a1", 0.0)
            + t
            * (
                coeffs.get("a2", 0.0)
                + t
                * (
                    coeffs.get("a3", 0.0)
                    + t * (coeffs.get("a4", 0.0) + t * coeffs.get("a5", 0.0))
                )
            )
        )
    if equation == "andrade_viscosity":
        return coeffs["A"] * exp(coeffs["B"] / T)
    if equation == "sutherland_gas_viscosity":
        T_ref = coeffs["T_ref"]
        if T_ref <= 0:
            raise ValueError("sutherland_gas_viscosity requires positive T_ref")
        S = coeffs["S"]
        if T + S <= 0 or T_ref + S <= 0:
            raise ValueError("sutherland_gas_viscosity has invalid Sutherland constant")
        return coeffs["mu_ref"] * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)
    if equation == "linear_thermal_conductivity":
        conductivity = coeffs["k_ref"] * (
            1.0 + coeffs.get("alpha", 0.0) * (T - coeffs["T_ref"])
        )
        return conductivity
    if equation == "thermal_conductivity_polynomial":
        return (
            coeffs.get("a", 0.0)
            + coeffs.get("b", 0.0) * T
            + coeffs.get("c", 0.0) * T**2
            + coeffs.get("d", 0.0) * T**3
        )
    if equation == "surface_tension_power":
        Tc = coeffs["Tc"]
        T_ref = coeffs["T_ref"]
        if temperature_K >= Tc:
            return 0.0
        ratio = (1.0 - temperature_K / Tc) / (1.0 - T_ref / Tc)
        return coeffs["sigma_ref"] * ratio ** coeffs.get("exponent", 1.26)
    raise ValueError(f"Unsupported property equation_id: {equation}")


def _evaluate_temperature_derivative(
    correlation: PropertyCorrelation,
    *,
    inputs: dict[str, float],
    pressure_value: float,
) -> float:
    equation = correlation.equation_id
    coeffs = correlation.coefficients
    T = inputs["temperature"]
    if equation == "antoine":
        base = coeffs.get("base", 10.0)
        denominator = T + coeffs["C"]
        if denominator <= 0:
            raise ValueError("Antoine vapor pressure singularity: T + C must be positive")
        return pressure_value * log(base) * coeffs["B"] / denominator**2
    if equation == "wagner":
        Tc = coeffs["Tc"]
        if Tc <= T:
            raise ValueError("Wagner vapor pressure derivative requires T < Tc")
        tau = 1.0 - T / Tc
        reduced_temperature = T / Tc
        numerator = (
            coeffs["a"] * tau
            + coeffs["b"] * tau**1.5
            + coeffs["c"] * tau**3.0
            + coeffs["d"] * tau**6.0
        )
        dnumerator_dtau = (
            coeffs["a"]
            + 1.5 * coeffs["b"] * tau**0.5
            + 3.0 * coeffs["c"] * tau**2.0
            + 6.0 * coeffs["d"] * tau**5.0
        )
        dnumerator_dT = -dnumerator_dtau / Tc
        dreduced_temperature_dT = 1.0 / Tc
        dexponent_dT = (
            dnumerator_dT * reduced_temperature
            - numerator * dreduced_temperature_dT
        ) / reduced_temperature**2
        return pressure_value * dexponent_dT
    if equation == "dippr101_vapor_pressure":
        return pressure_value * (
            -coeffs["B"] / T**2
            + coeffs["C"] / T
            + coeffs["D"] * coeffs["E"] * T ** (coeffs["E"] - 1.0)
        )
    raise ValueError(
        f"Unsupported vapor-pressure derivative equation_id: {equation}"
    )


def _cp_polynomial(T: float, coeffs: dict[str, float]) -> float:
    return (
        coeffs.get("a", 0.0)
        + coeffs.get("b", 0.0) * T
        + coeffs.get("c", 0.0) * T**2
        + coeffs.get("d", 0.0) * T**3
        + coeffs.get("e", 0.0) * T**4
    )


def _convert_input(
    value: float,
    source_unit: str,
    correlation: PropertyCorrelation,
    field_name: str,
) -> float:
    target_unit = correlation.input_units.get(field_name, source_unit)
    return convert_value(value, source_unit, target_unit)


def _validity_warnings(
    correlation: PropertyCorrelation,
    inputs: dict[str, float],
) -> tuple[str, ...]:
    warnings = []
    for field_name, value in inputs.items():
        bounds = correlation.validity_ranges.get(field_name)
        if bounds is None:
            continue
        lower, upper = bounds
        if value < lower or value > upper:
            warnings.append(
                f"{field_name}={value:g} outside validity range "
                f"[{lower:g}, {upper:g}] for {correlation.correlation_id}"
            )
    return tuple(warnings)


def _choose_valid_correlation(
    correlations: tuple[PropertyCorrelation, ...],
    *,
    temperature_K: float,
    pressure_Pa: float | None,
) -> PropertyCorrelation:
    for correlation in correlations:
        inputs = {"temperature": _convert_input(temperature_K, "K", correlation, "temperature")}
        if pressure_Pa is not None:
            inputs["pressure"] = _convert_input(pressure_Pa, "Pa", correlation, "pressure")
        if not _validity_warnings(correlation, inputs):
            return correlation
    return correlations[0]


def _ensure_finite_positive(value: float, correlation: PropertyCorrelation) -> None:
    if not isfinite(value):
        raise ValueError(f"Correlation returned non-finite value: {correlation.correlation_id}")
    if correlation.property_id in {
        "vapor_pressure",
        "sublimation_pressure",
        "heat_capacity",
        "ideal_gas_heat_capacity",
        "liquid_heat_capacity",
        "solid_heat_capacity",
        "isochoric_heat_capacity",
        "heat_of_vaporization",
        "heat_of_fusion",
        "liquid_molar_volume",
        "gas_molar_volume",
        "liquid_density",
        "gas_density",
        "liquid_viscosity",
        "gas_viscosity",
        "thermal_conductivity",
        "mixture_viscosity",
        "mixture_thermal_conductivity",
        "binary_gas_diffusivity",
        "effective_gas_diffusivity",
        "thermal_diffusivity",
        "surface_tension",
    } and value < 0:
        raise ValueError(f"Correlation returned negative value: {correlation.correlation_id}")


def _vapor_pressure_method_family(equation_id: str) -> str:
    if equation_id == "dippr101_vapor_pressure":
        return "DIPPR101"
    if equation_id == "antoine":
        return "Antoine"
    if equation_id == "wagner":
        return "Wagner"
    return equation_id


def _molar_volume_method_family(equation_id: str) -> str:
    if equation_id == "rackett_liquid_molar_volume":
        return "Rackett"
    if equation_id == "crc_second_virial":
        return "CRC second virial"
    return equation_id


def _transport_method_family(equation_id: str) -> str:
    if equation_id == "andrade_viscosity":
        return "Andrade/Arrhenius viscosity"
    if equation_id == "sutherland_gas_viscosity":
        return "Sutherland gas viscosity"
    if equation_id == "linear_thermal_conductivity":
        return "linear thermal conductivity"
    if equation_id == "thermal_conductivity_polynomial":
        return "thermal conductivity polynomial"
    return equation_id

