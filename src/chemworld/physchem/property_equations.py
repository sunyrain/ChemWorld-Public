"""Property-correlation equation evaluator utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite

from scipy.integrate import quad

from chemworld.foundation.units import convert_value
from chemworld.physchem.correlation_families import (
    CORRELATION_FAMILIES,
    family_value_and_slope,
    validate_family_interval,
    validate_heat_capacity_interval,
)
from chemworld.physchem.heat_capacity import PiecewiseHeatCapacity
from chemworld.physchem.phase_change_kernels import latent_heat_derivatives
from chemworld.physchem.property_reports import (
    R_J_PER_MOL_K,
    STANDARD_PRESSURE_PA,
    PropertyEvaluation,
    ValidityPolicy,
)
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.vapor_pressure_kernels import pressure_derivatives


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

    if validity_policy not in {"raise", "warn", "ignore"}:
        raise ValueError("unknown correlation validity policy")
    if not isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if pressure_Pa is not None and (not isfinite(pressure_Pa) or pressure_Pa <= 0):
        raise ValueError("pressure_Pa must be positive")
    if "pressure" in correlation.input_units and pressure_Pa is None:
        raise ValueError("pressure_Pa is required by this correlation")
    if isinstance(correlation, PiecewiseHeatCapacity):
        segment = correlation.segment_at(temperature_K)
        result = evaluate_correlation(segment, temperature_K=temperature_K,
                                      pressure_Pa=pressure_Pa, validity_policy="raise")
        return PropertyEvaluation(correlation.property_id, correlation.correlation_id,
            correlation.equation_id, result.to("J/(mol*K)").value, "J/(mol*K)",
            result.inputs, result.warnings)

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
    if equation in {"dimensionless_source_point", "molar_volume_source_point"}:
        from chemworld.physchem.property_points import source_point_value

        return source_point_value(T,coeffs["T_ref"],coeffs["value"])
    if equation in CORRELATION_FAMILIES:
        if CORRELATION_FAMILIES[equation].virial_method:
            from chemworld.physchem.virial_kernels import virial_derivatives

            return virial_derivatives(CORRELATION_FAMILIES[equation].virial_method,
                                      T, coeffs, order=0)[0]
        if CORRELATION_FAMILIES[equation].pressure_method:
            return pressure_derivatives(equation, T, coeffs, order=0)[0]
        if CORRELATION_FAMILIES[equation].phase_change_method:
            return latent_heat_derivatives(equation, T, coeffs, order=0)[0]
        if CORRELATION_FAMILIES[equation].viscosity_method:
            from chemworld.physchem.viscosity_kernels import viscosity_derivatives

            return viscosity_derivatives(
                CORRELATION_FAMILIES[equation].viscosity_method, T, coeffs, order=0,
            )[0]
        if CORRELATION_FAMILIES[equation].conductivity_method:
            from chemworld.physchem.conductivity_kernels import conductivity_derivatives

            return conductivity_derivatives(
                CORRELATION_FAMILIES[equation].conductivity_method, T, coeffs, order=0,
            )[0]
        if CORRELATION_FAMILIES[equation].interface_method:
            from chemworld.physchem.interface_kernels import interface_derivatives

            return interface_derivatives(
                CORRELATION_FAMILIES[equation].interface_method, T, coeffs, order=0,
            )[0]
        if CORRELATION_FAMILIES[equation].volume_method:
            from chemworld.physchem.volume_kernels import volume_derivatives

            family = CORRELATION_FAMILIES[equation]
            parameters = dict(coeffs)
            if ("MW" in parameters and molecular_weight_g_mol is not None
                    and abs(parameters["MW"] - molecular_weight_g_mol)
                    > 1e-9 * molecular_weight_g_mol):
                raise ValueError("volume correlation MW must match its component molecular weight")
            if "pressure" in family.input_units:
                parameters["P"] = pressure_Pa
            return volume_derivatives(family.volume_method, T, parameters,
                                      order=0, output=family.volume_output)[0]
        return family_value_and_slope(equation, T, coeffs)[0]
    if equation in {"antoine", "wagner"}:
        return pressure_derivatives(equation, T if equation == "antoine" else temperature_K,
                                    coeffs, order=0)[0]
    if equation == "dippr101_vapor_pressure":
        return family_value_and_slope("dippr101", T, coeffs)[0]
    if equation == "cp_polynomial":
        return _cp_polynomial(T, coeffs)
    if equation in {"watson_hvap", "constant_phase_change_enthalpy"}:
        return latent_heat_derivatives(equation, temperature_K, coeffs, order=0)[0]
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
        from chemworld.physchem.volume_kernels import volume_value

        if temperature_K >= coeffs["Tc"]:
            raise ValueError("Rackett liquid molar volume requires T < Tc")
        return volume_value("Rackett", temperature_K, **coeffs)
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


def _cp_polynomial(T: float, coeffs: dict[str, float]) -> float:
    return (
        coeffs.get("a", 0.0)
        + coeffs.get("b", 0.0) * T
        + coeffs.get("c", 0.0) * T**2
        + coeffs.get("d", 0.0) * T**3
        + coeffs.get("e", 0.0) * T**4
    )


def correlation_temperature_derivative(
    correlation: PropertyCorrelation, *, temperature_K: float,
    pressure_Pa: float | None = None,
    validity_policy: ValidityPolicy = "raise",
) -> PropertyEvaluation:
    """Analytic first derivative; latent-heat slopes use native J/(mol*K)."""
    if isinstance(correlation, PiecewiseHeatCapacity):
        segment = correlation.segment_at(temperature_K)
        result = correlation_temperature_derivative(segment, temperature_K=temperature_K,
                                                     validity_policy=validity_policy)
        slope = result.value * convert_value(1, segment.output_unit, correlation.output_unit)
        boundary = any(temperature_K == c.validity_ranges["temperature"][0]
                       for c in correlation.segments[1:])
        return PropertyEvaluation(correlation.property_id + "_temperature_derivative",
            correlation.correlation_id, correlation.equation_id + "_dT", slope,
            "(J/(mol*K))/K", result.inputs,
            ("right-sided derivative at piecewise boundary",) if boundary else result.warnings)
    if correlation.equation_id not in {"cp_polynomial", "watson_hvap",
                                       "constant_phase_change_enthalpy", *CORRELATION_FAMILIES}:
        raise ValueError("generic derivative requires a declared correlation family")
    result = evaluate_correlation(correlation, temperature_K=temperature_K, pressure_Pa=pressure_Pa,
                                  validity_policy=validity_policy)
    if correlation.equation_id == "cp_polynomial":
        slope = sum(power * correlation.coefficients.get(key, 0) * result.inputs["temperature"]
                    ** (power - 1) for power, key in enumerate("bcde", 1))
    elif correlation.equation_id in {"watson_hvap", "constant_phase_change_enthalpy"}:
        slope = latent_heat_derivatives(correlation.equation_id, temperature_K,
                                       correlation.coefficients, order=1)[1]
    else:
        slope = family_value_and_slope(correlation.equation_id, temperature_K,
                                       correlation.coefficients, pressure_Pa=pressure_Pa)[1]
    unit = f"({correlation.output_unit})/K"
    if correlation.property_id in {"heat_of_vaporization", "heat_of_fusion", "heat_of_sublimation"}:
        slope *= convert_value(1, correlation.output_unit, "J/mol")
        unit = "J/(mol*K)"
    return PropertyEvaluation(
        correlation.property_id + "_temperature_derivative", correlation.correlation_id,
        correlation.equation_id + "_dT", slope, unit,
        result.inputs, result.warnings)


@dataclass(frozen=True)
class CorrelationIntegral:
    """QUADPACK error is a numerical estimate, not parameter uncertainty."""

    correlation_id: str
    equation_id: str
    value: float
    unit: str
    initial_temperature_K: float
    final_temperature_K: float
    over_temperature: bool
    absolute_error_estimate: float
    evaluations: int
    warnings: tuple[str, ...]
    method: str = "adaptive_gauss_kronrod"
    evaluation_scope: str = "quadrature_integrand_calls_only; excludes domain validation"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def integrate_correlation(
    correlation: PropertyCorrelation, *, initial_temperature_K: float,
    final_temperature_K: float, over_temperature: bool = False,
    validity_policy: ValidityPolicy = "raise", epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    pressure_Pa: float | None = None,
) -> CorrelationIntegral:
    """Definite integral of a declared family; reject poles and failed quadrature.

    Coefficient validity ranges are connected intervals. Their endpoint checks
    cover the whole integration interval; mathematical poles are checked separately.
    Every evaluated property must also satisfy its physical sign constraint.
    """
    if validity_policy not in {"raise", "warn", "ignore"}:
        raise ValueError("unknown correlation validity policy")
    if isinstance(correlation, PiecewiseHeatCapacity):
        return correlation.integral(initial_temperature_K, final_temperature_K,
            over_temperature=over_temperature, epsabs=epsabs, epsrel=epsrel)
    if correlation.equation_id not in CORRELATION_FAMILIES:
        raise ValueError("generic integration requires a declared correlation family")
    if not all(isfinite(v) and v > 0 for v in (epsabs, epsrel)):
        raise ValueError("quadrature tolerances must be finite and positive")
    endpoints = [evaluate_correlation(correlation, temperature_K=T, pressure_Pa=pressure_Pa,
                                      validity_policy=validity_policy)
                 for T in (initial_temperature_K, final_temperature_K)]
    warnings = tuple(dict.fromkeys(w for endpoint in endpoints for w in endpoint.warnings))
    lower, upper = sorted((initial_temperature_K, final_temperature_K))
    validate_family_interval(correlation.equation_id, correlation.coefficients, lower, upper,
                             pressure_Pa=pressure_Pa)
    if "heat_capacity" in correlation.property_id:
        validate_heat_capacity_interval(
            correlation.equation_id, correlation.coefficients, lower, upper,
            allow_zero=correlation.property_id == "isochoric_heat_capacity")

    def integrand(temperature):
        result = evaluate_correlation(correlation, temperature_K=temperature,
                                      pressure_Pa=pressure_Pa,
                                      validity_policy=validity_policy)
        if ("heat_capacity" in correlation.property_id and result.value == 0
                and correlation.property_id != "isochoric_heat_capacity"):
            raise ValueError("Cp must be positive over the integration interval")
        return result.value / temperature if over_temperature else result.value

    # Validate even a zero-width caloric interval, without invoking quadrature.
    integrand(lower)
    if lower == upper:
        value, error, evaluations = 0.0, 0.0, 0
    else:
        result = quad(integrand, lower, upper, epsabs=epsabs, epsrel=epsrel,
                      full_output=True, limit=150)
        value, error, info = result[:3]
        if (len(result) != 3 or not isfinite(value) or not isfinite(error)
                or error > max(epsabs, epsrel * abs(value))):
            diagnostic = result[3] if len(result) > 3 else "requested accuracy not reached"
            raise ValueError(f"correlation quadrature failed: {diagnostic}")
        evaluations = info["neval"]
        if final_temperature_K < initial_temperature_K:
            value = -value
    return CorrelationIntegral(
        correlation.correlation_id, correlation.equation_id, value,
        correlation.output_unit if over_temperature else f"({correlation.output_unit})*K",
        initial_temperature_K, final_temperature_K, over_temperature, error,
        evaluations, warnings)


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


def _ensure_finite_positive(value: float, correlation: PropertyCorrelation) -> None:
    if not isfinite(value):
        raise ValueError(f"Correlation returned non-finite value: {correlation.correlation_id}")
    if (correlation.property_id.endswith(("_molar_volume", "_density")) and value <= 0):
        raise ValueError(f"Volume/density must be positive: {correlation.correlation_id}")
    if correlation.property_id in {"relative_permittivity", "refractive_index"} and value <= 0:
        raise ValueError(
            f"Electro-optical correlation must be positive: {correlation.correlation_id}")
    if (correlation.property_id in {"vapor_pressure", "sublimation_pressure", "henry_pressure"}
            and value == 0):
        raise ValueError(f"Pressure underflowed to zero: {correlation.correlation_id}")
    if correlation.property_id in {
        "relative_permittivity", "refractive_index",
        "henry_pressure",
        "vapor_pressure",
        "sublimation_pressure",
        "heat_capacity",
        "ideal_gas_heat_capacity",
        "liquid_heat_capacity",
        "solid_heat_capacity",
        "isochoric_heat_capacity",
        "heat_of_vaporization",
        "heat_of_fusion",
        "heat_of_sublimation",
        "liquid_molar_volume",
        "gas_molar_volume",
        "solid_molar_volume",
        "liquid_density",
        "solid_density",
        "liquid_molar_density",
        "solid_molar_density",
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
    if equation_id in CORRELATION_FAMILIES and CORRELATION_FAMILIES[equation_id].pressure_method:
        return CORRELATION_FAMILIES[equation_id].pressure_method
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

