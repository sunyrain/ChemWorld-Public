"""Native formula families, independently implemented from published equations.

Reference: chemicals 1.5.2 ``dippr`` and ``heat_capacity`` equation documentation;
no upstream runtime import or parameter database. DIPPR coefficients carry the
caller's output-unit scale; caloric coefficients have an explicit fixed SI output.
This module owns formula metadata as well as evaluation and mathematical domains.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from math import exp, isfinite, tanh
from types import MappingProxyType
from typing import NoReturn, cast

import numpy as np
from scipy.optimize import brentq

from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.caloric_math import oscillator_value_slope as _oscillator
from chemworld.physchem.conductivity_kernels import CONDUCTIVITY_METHODS, conductivity_derivatives
from chemworld.physchem.dippr_kernels import dippr_expression
from chemworld.physchem.heat_capacity_kernels import (
    caloric_domain,
    caloric_value_slope,
    validate_caloric_capacity,
    validate_caloric_parameters,
)
from chemworld.physchem.interface_kernels import INTERFACE_METHODS, interface_derivatives
from chemworld.physchem.lennard_jones import kinetic_viscosity_value_slope
from chemworld.physchem.phase_change_kernels import latent_heat_derivatives
from chemworld.physchem.vapor_pressure_kernels import pressure_derivatives
from chemworld.physchem.virial_kernels import VIRIAL_METHODS, virial_derivatives
from chemworld.physchem.viscosity_kernels import VISCOSITY_METHODS, viscosity_derivatives
from chemworld.physchem.volume_kernels import VOLUME_METHODS, volume_derivatives


@dataclass(frozen=True)
class CorrelationFamily:
    equation_id: str
    required: tuple[str, ...]
    optional: tuple[str, ...]
    property_dimensions: Mapping[str, str]
    mathematical_domain: str
    reference_method: str = ""
    fixed_output_unit: str | None = None
    optional_prefix: str | None = None
    reference_document: str = ""
    pressure_method: str = ""
    phase_change_method: str = ""
    viscosity_method: str = ""
    conductivity_method: str = ""
    interface_method: str = ""
    volume_method: str = ""
    volume_output: str = ""
    virial_method: str = ""

    @property
    def input_units(self) -> dict[str, str]:
        return {"temperature": "K", **({"pressure": "Pa"}
                if self.volume_method and VOLUME_METHODS[self.volume_method].uses_pressure else {})}

    @property
    def input_dimensions(self) -> dict[str, str]:
        return {key: key for key in self.input_units}

    @property
    def reference(self) -> str:
        if self.reference_document:
            return self.reference_document
        if self.pressure_method:
            return "chemicals 1.5.2: vapor_pressure." + self.pressure_method
        if self.phase_change_method:
            return "chemicals 1.5.2: phase_change." + self.phase_change_method
        if self.viscosity_method:
            return "chemicals 1.5.2: viscosity." + self.viscosity_method
        if self.conductivity_method:
            return "chemicals 1.5.2: thermal_conductivity." + self.conductivity_method
        if self.interface_method:
            return "chemicals 1.5.2: interface." + self.interface_method
        if self.volume_method:
            return "chemicals 1.5.2: volume." + self.volume_method
        if self.virial_method:
            return "chemicals 1.5.2: virial public mathematics and documented call characterization"
        return "chemicals 1.5.2: " + (self.reference_method or (
            "dippr." + self.equation_id.replace("dippr", "EQ")))

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "equation_id": self.equation_id,
            "required_coefficients": list(self.required),
            "optional_coefficients": list(self.optional),
            "input_dimensions": self.input_dimensions,
            "input_units": self.input_units,
            "output_dimensions_by_property": dict(self.property_dimensions),
            "mathematical_domain": self.mathematical_domain,
            "temperature_derivative": "analytic_first",
            "integrals": ["adaptive_quadrature", "adaptive_quadrature_over_T"],
            "reference": self.reference,
        }
        if self.fixed_output_unit:
            result["coefficient_output_unit"] = self.fixed_output_unit
        if self.equation_id.startswith("dippr"):
            result["temperature_derivative"] = "analytic_through_third_subcritical"
            result["parameter_derivative"] = "analytic_declared_source_coordinates"
            result["mixed_derivative_order"] = 3
        if self.optional_prefix:
            result["indexed_coefficient_prefix"] = self.optional_prefix
        if (self.pressure_method or self.interface_method
                or self.volume_method or self.virial_method):
            result["temperature_derivative"] = "analytic_through_third_where_differentiable"
        if self.virial_method:
            result["indefinite_temperature_integral_orders"] = (
                [-1, -2] if VIRIAL_METHODS[self.virial_method].integrals else [])
            result["physical_relation"] = "density_virial_coefficient"
        if self.volume_method:
            result["physical_relation"] = VOLUME_METHODS[self.volume_method].relation
            result["temperature_derivative_path"] = (
                "fixed_pressure_and_supplied_parameters"
                if VOLUME_METHODS[self.volume_method].uses_pressure else
                VOLUME_METHODS[self.volume_method].relation)
        if self.equation_id in {"phase_change_point", "interface_source_point",
                                "dimensionless_source_point", "molar_volume_source_point"}:
            result["temperature_derivative"] = "unavailable_single_observation"
            result["integrals"] = []
        return result


_CP = dict.fromkeys(("heat_capacity", "ideal_gas_heat_capacity", "liquid_heat_capacity",
                    "solid_heat_capacity", "isochoric_heat_capacity"), "molar_heat_capacity")
_DENSITY = {"liquid_density": "mass_density", "solid_density": "mass_density",
            "liquid_molar_density": "concentration", "solid_molar_density": "concentration",
            "gas_density": "mass_density"}
_VOLUME = {p + "_molar_volume": "molar_volume" for p in ("liquid", "solid", "gas")}
_PRESSURE_VISCOSITY = {"vapor_pressure": "pressure", "sublimation_pressure": "pressure",
                       "liquid_viscosity": "dynamic_viscosity"}
_POSITIVE_T = "finite T > 0 K; finite parameters and result"
_LATENT = dict.fromkeys(("heat_of_vaporization", "heat_of_fusion", "heat_of_sublimation"),
                        "molar_enthalpy")


def _family(
    number: int | str, required: Iterable[str], optional: Iterable[str],
    properties: Mapping[str, str], domain: str = _POSITIVE_T,
) -> CorrelationFamily:
    return CorrelationFamily("dippr" + str(number), tuple(required), tuple(optional),
                             MappingProxyType(dict(properties)), domain)


def _caloric(
    equation: str, method: str, required: str, optional: str, phase: str,
    domain: str = _POSITIVE_T, *, prefix: str | None = None,
) -> CorrelationFamily:
    phases = ("ideal_gas", "liquid", "solid") if phase == "all" else (phase,)
    return CorrelationFamily(
        equation, tuple(required.split()), tuple(optional.split()),
        MappingProxyType({"heat_capacity": "molar_heat_capacity",
                          **{p + "_heat_capacity": "molar_heat_capacity" for p in phases}}),
        domain, "heat_capacity." + method, "J/(mol*K)", prefix)


def _pressure(
    equation: str, method: str, required: str, optional: str = "", *,
    fixed_unit: str | None = None, sublimation: bool = False, domain: str = _POSITIVE_T,
) -> CorrelationFamily:
    return CorrelationFamily(equation, tuple(required.split()), tuple(optional.split()),
        MappingProxyType({"sublimation_pressure" if sublimation else "vapor_pressure": "pressure",
                          **({"sublimation_pressure": "pressure"} if equation in
                             {"tde_pv_expansion", "yaws_psat", "arrhenius_pressure"} else {})}),
        domain, fixed_output_unit=fixed_unit, pressure_method=method)


def _latent(
    equation: str, method: str, required: str,
    domain: str = "0 < T <= Tc; no supercritical extension",
) -> CorrelationFamily:
    return CorrelationFamily(equation, tuple(required.split()), (),
        MappingProxyType({"heat_of_vaporization": "molar_enthalpy"}), domain,
        fixed_output_unit="J/mol", phase_change_method=method)


_FAMILIES = (
    CorrelationFamily("molar_volume_source_point", ("T_ref", "value"), (),
        MappingProxyType(_VOLUME), "one explicit temperature; no thermal curve or derivative",
        fixed_output_unit="m^3/mol", reference_document="Explicit source observation"),
    CorrelationFamily("permittivity_crc", ("A",), tuple("BCD"),
        MappingProxyType({"relative_permittivity":"dimensionless"}),
        "positive T and permittivity; caller supplies source/application temperature range",
        fixed_output_unit="dimensionless",
        reference_document="CRC Handbook; chemicals 1.5.2 permittivity.permittivity_CRC"),
    CorrelationFamily("dimensionless_source_point", ("T_ref","value"), (),
        MappingProxyType({"relative_permittivity":"dimensionless",
                          "refractive_index":"dimensionless"}),
        "one explicit positive temperature only; no interpolation or derivative",
        fixed_output_unit="dimensionless", reference_document="Explicit source observation"),
    CorrelationFamily("henry_pressure", ("A",), tuple("BCDEF"),
        MappingProxyType({"henry_pressure": "pressure"}),
        "positive T; H=p/x in Pa; explicit solute/solvent and application interval",
        fixed_output_unit="Pa", pressure_method="Henry_pressure",
        reference_document="chemicals 1.5.2 solubility.Henry_pressure; Gmehling 2012"),
    CorrelationFamily("phase_change_point", ("T_ref", "delta_h"), (),
        MappingProxyType(_LATENT), "T = T_ref > 0 only; no derivative or interpolation",
        fixed_output_unit="J/mol", phase_change_method="single_temperature_observation",
        reference_document="Explicit source observation at one declared temperature"),
    _latent("clapeyron_hvap", "Clapeyron", "Tc Pc Psat dZ",
            "0 < T < Tc; 0 < Psat < Pc; dZ > 0; partial T derivative at fixed Psat/dZ"),
    _latent("pitzer_hvap", "Pitzer", "Tc omega"),
    _latent("smk_hvap", "SMK", "Tc omega"),
    _latent("mk_hvap", "MK", "Tc omega"),
    _latent("velasco_hvap", "Velasco", "Tc omega"),
    _latent("ppds12_hvap", "PPDS12", "Tc A B C D E"),
    _latent("alibakhshi_hvap", "Alibakhshi", "Tc C",
            "0 < T <= Tc; empirical fit does not have the critical zero limit; "
            "source/application range must be explicit (reported Tb-50 to Tc-100 K)"),
    _pressure("trc_antoine_extended", "TRC_Antoine_extended", "Tc to A B C n E F",
              domain="T > 0; T+C > 0; n > 0; requested derivatives must exist at the branch"),
    _pressure("wagner25", "Wagner", "Tc Pc a b c d",
              domain="0 < T <= Tc; Pc > 0; requested critical derivatives must exist"),
    _pressure("yaws_psat", "Yaws_Psat", "A B C D E"),
    _pressure("tde_pv_expansion", "TDE_PVExpansion", "a1 a2 a3", "a4 a5 a6 a7 a8"),
    _pressure("boiling_critical_pressure", "boiling_critical_relation", "Tb Tc Pc",
              fixed_unit="Pa"),
    _pressure("lee_kesler_pressure", "Lee_Kesler", "Tc Pc omega", fixed_unit="Pa"),
    _pressure("ambrose_walton_pressure", "Ambrose_Walton", "Tc Pc omega", fixed_unit="Pa"),
    _pressure("sanjari_pressure", "Sanjari", "Tc Pc omega", fixed_unit="Pa"),
    _pressure("edalat_pressure", "Edalat", "Tc Pc omega", fixed_unit="Pa"),
    _pressure("clapeyron_sublimation", "Psub_Clapeyron", "Tt Pt Hsub_t", sublimation=True,
              fixed_unit="Pa"),
    _pressure("arrhenius_pressure", "Arrhenius_extrapolation", "T_ref P_ref slope"),
    _pressure("iapws_saturation", "Psat_IAPWS", "", fixed_unit="Pa",
              domain="273.15 <= T <= 647.096 K; IF97 region 4 only"),
    CorrelationFamily(
        "lj_viscosity_neufeld", ("MW", "sigma", "epsilon_k"), (),
        MappingProxyType({"gas_viscosity": "dynamic_viscosity"}),
        "positive MW (g/mol), sigma (m), epsilon/k (K); 0.3 < T/(epsilon/k) < 100",
        fixed_output_unit="Pa*s", reference_document="Chapman-Enskog; Neufeld et al. 1972",
    ),
    CorrelationFamily(
        "lj_viscosity_kim_monroe", ("MW", "sigma", "epsilon_k"), (),
        MappingProxyType({"gas_viscosity": "dynamic_viscosity"}),
        "positive MW (g/mol), sigma (m), epsilon/k (K); 0.3 < T/(epsilon/k) < 400",
        fixed_output_unit="Pa*s", reference_document="Chapman-Enskog; Kim and Monroe 2014",
    ),
    CorrelationFamily("chemsep16", tuple("ABCDE"), (), MappingProxyType(_CP), _POSITIVE_T,
        reference_document="ChemSep Help, Table 3, equation 16: "
        "https://studylib.es/doc/9235141/chemsep-help"),
    _family(100, "A", "BCDEFG", {**_CP, **_DENSITY, **_LATENT, **_VOLUME,
                                "gas_viscosity": "dynamic_viscosity",
                                "thermal_conductivity": "thermal_conductivity"}),
    _family(101, "AB", "CDE", _PRESSURE_VISCOSITY),
    _family(102, "AB", "CD", {**_CP, "gas_viscosity": "dynamic_viscosity",
                              "thermal_conductivity": "thermal_conductivity"},
            _POSITIVE_T + "; denominator nonzero; integration cannot cross a pole"),
    _family(104, "AB", "CDE", {"second_virial_coefficient": "molar_volume"}),
    _family(105, "ABCD", "", _DENSITY, "A/B/C/D > 0; 0 < T < C (K)"),
    _family(106, ("Tc", "A", "B"), "CDE",
            {"heat_of_vaporization": "molar_enthalpy", "surface_tension": "surface_tension"},
            "Tc > 0; 0 < T < Tc; finite result"),
    _family(107, "A", "BCDE", _CP),
    _family(114, ("Tc", "A", "B", "C", "D"), "", _CP, "Tc > 0; 0 < T < Tc"),
    _family(115, "AB", "CDE", _PRESSURE_VISCOSITY),
    _family(116, ("Tc", "A", "B", "C", "D", "E"), "", _DENSITY,
            "Tc > 0; 0 < T < Tc"),
    _family(127, "ABCDEFG", "", _CP),
    _family("100_reciprocal", "A", "BCDEFG", _VOLUME,
            _POSITIVE_T + "; polynomial nonzero; integration cannot cross a zero"),
    _family("105_reciprocal", "ABCD", "", _VOLUME,
            "A/B/C/D > 0; 0 < T < C (K)"),
    _family("106_reciprocal", ("Tc", "A", "B"), "CDE",
            _VOLUME, "Tc > 0; 0 < T < Tc; base nonzero"),
    _caloric("poling_cp", "Poling", "a b c d e", "", "ideal_gas"),
    _caloric("shomate_cp", "Shomate", "A B C D E", "", "all",
             _POSITIVE_T + "; SI coefficients use T in K, not kilokelvin"),
    _caloric("trc_cp", "TRCCp", "a0 a1 a2 a3 a4 a5 a6 a7", "", "ideal_gas",
             _POSITIVE_T + "; T+a6 nonzero in T>a7 branch; no pole-crossing integral"),
    _caloric("ppds2_cp", "PPDS2", "Ts C_low C_inf a1 a2 a3 a4 a5", "", "ideal_gas",
             _POSITIVE_T + "; Ts > 0"),
    _caloric("ppds15_cp", "PPDS15", "Tc a0 a1 a2 a3 a4 a5", "", "liquid",
             "0 < T < Tc; saturation-liquid heat capacity, not a general pressure surface"),
    _caloric("tde_cs_cp", "TDE_CSExpansion", "Tc b a1", "a2 a3 a4", "liquid",
             "0 < T < Tc; saturation-liquid heat capacity"),
    _caloric("rowlinson_poling_cp", "Rowlinson_Poling", "Tc omega Cpgm", "", "liquid",
             "0 < T < Tc; Cpgm is the explicitly fixed gas Cp parameter; partial T derivative"),
    _caloric("rowlinson_bondi_cp", "Rowlinson_Bondi", "Tc omega Cpgm", "", "liquid",
             "0 < T < Tc; Cpgm is the explicitly fixed gas Cp parameter; partial T derivative"),
    _caloric("zabransky_cubic_cp", "Zabransky_cubic", "a1 a2 a3 a4", "", "liquid"),
    _caloric("zabransky_quasi_cp", "Zabransky_quasi_polynomial", "Tc a1 a2 a3 a4 a5 a6",
             "", "liquid", "0 < T < Tc; isobaric liquid Cp"),
    _caloric("perry151_cp", "Perry_151", "a b c d", "", "solid",
             _POSITIVE_T + "; original calorie coefficients converted with 4.184 J/cal"),
    _caloric("lastovka_shaw_cp", "Lastovka_Shaw", "similarity_variable MW cyclic_aliphatic",
             "term_A", "ideal_gas",
             _POSITIVE_T + "; alpha > 0 mol/g, MW > 0 g/mol; organic estimate"),
    _caloric("dadgostar_shaw_cp", "Dadgostar_Shaw", "similarity_variable MW", "", "liquid",
             _POSITIVE_T + "; alpha > 0 mol/g, MW > 0 g/mol; liquid hydrocarbon estimate"),
    _caloric("lastovka_solid_cp", "Lastovka_solid", "similarity_variable MW", "", "solid",
             _POSITIVE_T + "; organic solid estimate; explicit range below phase transition"),
    _caloric("statistical_cpg", "Cpg_statistical_mechanics", "linear theta_0", "", "ideal_gas",
             _POSITIVE_T + "; explicit linear=0/1 and positive theta_i K; polyatomic rigid rotor",
             prefix="theta_"),
)
def _volume_families() -> Iterator[CorrelationFamily]:
    for method, spec in VOLUME_METHODS.items():
        for output in ("density", "molar_volume"):
            required = list(spec.required.split())
            if spec.uses_pressure:
                required.remove("P")  # Actual state input, never a frozen coefficient.
            if ((output == "molar_volume") != (spec.native_output == "molar_volume")
                    and "MW" not in required):
                required.append("MW")
            yield CorrelationFamily(
                "volume_" + method.lower() + "_" + output,
                tuple(required), tuple(spec.optional.split()),
                MappingProxyType({spec.phase + "_" + output:
                                  "mass_density" if output == "density" else "molar_volume"}),
                "positive T and result; explicit application range and phase; "
                "T <= Tc for saturation, T <= Tt for Goodman, T >= Tm for molten CRC; "
                "no supercritical clipping; derivatives only where finite",
                fixed_output_unit="kg/m^3" if output == "density" else "m^3/mol",
                volume_method=method, volume_output=output,
            )


CORRELATION_FAMILIES: Mapping[str, CorrelationFamily] = MappingProxyType(
    {family.equation_id: family for family in (*_FAMILIES, *(
        CorrelationFamily(
            "viscosity_" + method.lower(), tuple(required.split()), tuple(optional.split()),
            MappingProxyType({p + "_viscosity": "dynamic_viscosity" for p in phase.split()}),
            "positive T; formula-specific poles and branch derivatives rejected; "
            "explicit application range; supplied rho is fixed for T derivatives",
            fixed_output_unit="Pa*s", viscosity_method=method,
        ) for method, (required, optional, phase) in VISCOSITY_METHODS.items()
    ), *(
        CorrelationFamily(
            "conductivity_" + method.lower(), tuple(required.split()), tuple(optional.split()),
            MappingProxyType({"thermal_conductivity": "thermal_conductivity"}),
            "positive T and conductivity; fixed supplied Cvm/mu for T derivatives; "
            "liquid fractional-power models require T <= Tc; explicit application range",
            fixed_output_unit="W/(m*K)", conductivity_method=method,
        ) for method, (required, optional, phase) in CONDUCTIVITY_METHODS.items()
    ), *(
        CorrelationFamily(
            "interface_" + method.lower(), tuple(required.split()), tuple(optional.split()),
            MappingProxyType({"surface_tension": "surface_tension"}),
            "T = T_ref > 0 only; no extrapolation or temperature derivative"
            if method == "source_point" else
            "positive T; nonnegative result; T <= Tc where applicable; "
            "liquid-side critical derivatives only where finite; explicit application range",
            fixed_output_unit="N/m", interface_method=method,
            reference_document="Explicit source observation at one declared temperature"
            if method == "source_point" else "",
        ) for method, (required, optional) in INTERFACE_METHODS.items()
    ), *_volume_families(), *(
        CorrelationFamily(
            "virial_" + method.lower(), tuple(spec.required.split()), (),
            MappingProxyType({spec.property_id: "third_virial_coefficient"
                              if spec.third else "molar_volume"}),
            "positive T/Tc/Pc/Vc; finite explicit omega and polar parameters; "
            "signed B/C; explicit application interval, no stability claim",
            fixed_output_unit=spec.unit, virial_method=method,
        ) for method, spec in VIRIAL_METHODS.items()
    ))})


def validate_family_parameters(equation_id: str, coefficients: Mapping[str, float]) -> None:
    family = CORRELATION_FAMILIES[equation_id]
    missing = set(family.required) - coefficients.keys()
    extra = coefficients.keys() - {*family.required, *family.optional}
    if family.optional_prefix:
        extra = {key for key in extra if not (key.startswith(family.optional_prefix)
                 and key[len(family.optional_prefix):].isascii()
                 and key[len(family.optional_prefix):].isdigit())}
    if missing or extra:
        raise ValueError(f"{equation_id}: missing coefficients {sorted(missing)}, "
                         f"unexpected coefficients {sorted(extra)}")
    if any(isinstance(v, bool) or not isinstance(v, int | float) or not isfinite(v)
           for v in coefficients.values()):
        raise ValueError("correlation coefficients must be finite real numbers")
    if "Tc" in coefficients and coefficients["Tc"] <= 0.0:
        raise ValueError("Tc must be positive")
    if equation_id.startswith("dippr105") and any(coefficients[k] <= 0 for k in "ABCD"):
        raise ValueError("DIPPR105 requires positive A, B, C and D")
    if family.reference_method:
        validate_caloric_parameters(equation_id, coefficients)


def family_value_and_slope(
    equation_id: str, temperature_K: float, coefficients: Mapping[str, float],
    *, pressure_Pa: float | None = None,
) -> tuple[float, float]:
    """Value and analytic first derivative in output units and output units/K."""
    validate_family_parameters(equation_id, coefficients)
    T = temperature_K
    if not isfinite(T) or T <= 0:
        raise ValueError("correlations require finite positive absolute temperature")
    if equation_id in {"dimensionless_source_point", "molar_volume_source_point"}:
        raise ValueError("a single source observation has no temperature derivative")
    if equation_id == "permittivity_crc":
        from chemworld.physchem.electro_optics import relative_permittivity_crc

        return cast(tuple[float, float],
                    derivative_values(lambda t:relative_permittivity_crc(t,**coefficients),T))
    if CORRELATION_FAMILIES[equation_id].reference_method:
        return caloric_value_slope(equation_id, T, coefficients)
    if CORRELATION_FAMILIES[equation_id].pressure_method:
        return pressure_derivatives(equation_id, T, coefficients, order=1)
    if CORRELATION_FAMILIES[equation_id].phase_change_method:
        return latent_heat_derivatives(equation_id, T, coefficients, order=1)
    if CORRELATION_FAMILIES[equation_id].viscosity_method:
        return viscosity_derivatives(
            CORRELATION_FAMILIES[equation_id].viscosity_method, T, coefficients, order=1,
        )
    if equation_id in {"lj_viscosity_neufeld", "lj_viscosity_kim_monroe"}:
        return kinetic_viscosity_value_slope(
            T, coefficients["MW"], coefficients["sigma"], coefficients["epsilon_k"],
            method=("Neufeld_Janzen_Aziz" if equation_id == "lj_viscosity_neufeld"
                    else "Kim_Monroe"),
        )
    if CORRELATION_FAMILIES[equation_id].conductivity_method:
        return conductivity_derivatives(
            CORRELATION_FAMILIES[equation_id].conductivity_method, T, coefficients, order=1,
        )
    if CORRELATION_FAMILIES[equation_id].interface_method:
        return interface_derivatives(
            CORRELATION_FAMILIES[equation_id].interface_method, T, coefficients, order=1,
        )
    if CORRELATION_FAMILIES[equation_id].volume_method:
        family = CORRELATION_FAMILIES[equation_id]
        parameters: dict[str, float | None] = dict(coefficients)
        if VOLUME_METHODS[family.volume_method].uses_pressure:
            parameters["P"] = pressure_Pa
        return volume_derivatives(family.volume_method, T, parameters,
                                  order=1, output=family.volume_output)
    if CORRELATION_FAMILIES[equation_id].virial_method:
        return virial_derivatives(CORRELATION_FAMILIES[equation_id].virial_method,
                                  T, coefficients, order=1)
    if equation_id.startswith("dippr"):
        return cast(tuple[float, float],
                    derivative_values(lambda t: dippr_expression(equation_id, t, coefficients), T))
    A, B, C, D, E = (coefficients[key] for key in "ABCDE")
    try:
        if equation_id == "chemsep16":
            exponential = exp(B / T + C + D * T + E * T**2)
            value = A + exponential
            slope = exponential * (-B / T**2 + D + 2 * E * T)
        else:
            raise ValueError(f"Unsupported family {equation_id}")
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            f"{equation_id} exceeds its finite numerical domain or meets a pole") from exc
    if not isfinite(value) or not isfinite(slope):
        raise ValueError(f"{equation_id} returned a nonfinite value or slope")
    return value, slope


def validate_family_interval(
    equation_id: str, coefficients: Mapping[str, float], lower_K: float, upper_K: float,
    *, pressure_Pa: float | None = None,
) -> None:
    """Reject poles throughout a closed interval, including non-endpoint zeros."""
    family_value_and_slope(equation_id, lower_K, coefficients, pressure_Pa=pressure_Pa)
    family_value_and_slope(equation_id, upper_K, coefficients, pressure_Pa=pressure_Pa)
    if CORRELATION_FAMILIES[equation_id].reference_method:
        caloric_domain(equation_id, coefficients, lower_K, upper_K)
        return
    polynomial = None
    if equation_id == "dippr100_reciprocal":
        polynomial = [coefficients.get(key, 0.0) for key in "ABCDEFG"]
    elif equation_id == "dippr102":
        polynomial = [coefficients.get("D", 0.0), coefficients.get("C", 0.0), 1.0]
    if polynomial is not None:
        roots = cast(Iterable[complex], np.polynomial.Polynomial(polynomial).roots())
        if any(abs(root.imag) <= 1e-8 * max(1, abs(root.real))
               and lower_K <= root.real <= upper_K for root in roots):
            raise ValueError(f"{equation_id} integration interval crosses a pole")


def validate_heat_capacity_interval(
    equation_id: str, coefficients: Mapping[str, float], lower_K: float, upper_K: float,
    *, allow_zero: bool = False, minimum: float = 0.0,
) -> None:
    """Establish the capacity sign on an interval before integrating it.

    Polynomial extrema handle 100 and tau*114. For oscillator sums, monotone
    term ranges (including the cosh maximum) give conservative bounds, refined
    by subdivision for mixed signs. Failure to establish positivity is an error.
    """
    c = coefficients
    if CORRELATION_FAMILIES[equation_id].reference_method:
        validate_caloric_capacity(equation_id, c, lower_K, upper_K,
                                  minimum=minimum, allow_zero=allow_zero)
        return

    def valid(value: float) -> bool:
        return isfinite(value) and (value >= minimum if allow_zero else value > minimum)

    def reject() -> NoReturn:
        raise ValueError("heat capacity is nonpositive or cannot be bounded over the interval")

    for T in (lower_K, upper_K):
        if not valid(family_value_and_slope(equation_id, T, c)[0]):
            reject()
    polynomial: list[float] | None
    if equation_id == "chemsep16":
        # exp is monotone, so every capacity extremum comes from the cubic
        # numerator of d(B/T+C+D*T+E*T²)/dT. This also bounds Cp-R for gas U.
        polynomial = [-c["B"], 0.0, c["D"], 2*c["E"]]
        for root in cast(Iterable[complex], np.polynomial.Polynomial(polynomial).roots()):
            if (abs(root.imag) < 1e-8 and lower_K < root.real < upper_K
                    and not valid(family_value_and_slope(equation_id, root.real, c)[0])):
                reject()
        return
    polynomial = None
    lower, upper = lower_K, upper_K
    if equation_id == "dippr100":
        polynomial = [c.get(key, 0.0) for key in "ABCDEFG"]
        polynomial[0] -= minimum
    elif equation_id == "dippr114":
        A, B, C, D = (c[key] for key in "ABCD")
        polynomial = [A*A, B - minimum, -2*A*C, -A*D, -C*C/3, -C*D/2, -D*D/5]
        lower, upper = 1 - upper_K / c["Tc"], 1 - lower_K / c["Tc"]
    if polynomial is not None:
        poly = np.polynomial.Polynomial(polynomial)
        points = (lower, upper, *(
            root.real for root in cast(Iterable[complex], poly.deriv().roots())
            if abs(root.imag) < 1e-8 and lower < root.real < upper))
        if not all(isfinite(poly(point)) and (poly(point) >= 0 if allow_zero else poly(point) > 0)
                   for point in points):
            reject()
        return
    if equation_id == "dippr102":
        # Positive numerator has fixed sign. For a nonzero minimum, evaluate
        # every stationary point of A*T**(B+2)/(T²+C*T+D).
        if minimum:
            B, C, D = c["B"], c.get("C", 0), c.get("D", 0)
            roots = cast(Iterable[complex],
                         np.polynomial.Polynomial([(B+2)*D, (B+1)*C, B]).roots())
            for root in roots:
                if (abs(root.imag) < 1e-8 and lower_K < root.real < upper_K
                        and not valid(family_value_and_slope(equation_id, root.real, c)[0])):
                    reject()
        return
    if equation_id not in {"dippr107", "dippr127"}:
        raise ValueError("no interval capacity bounds for this equation")
    A, B, C, D, E, F, G = (c.get(key, 0.0) for key in "ABCDEFG")
    terms = ((B, C, False), (D, E, True)) if equation_id == "dippr107" else (
        (B, C / 2, False), (D, E / 2, False), (F, G / 2, False))
    peak_x = brentq(lambda x: x * tanh(x) - 1, 1, 2)
    intervals = [(lower_K, upper_K)]
    for _ in range(2048):
        if not intervals:
            return
        lower, upper = intervals.pop()
        lower_bound = A
        for amplitude, theta, cosh in terms:
            values = [_oscillator(theta / T, T, cosh=cosh)[0] for T in (lower, upper)]
            if cosh and lower < abs(theta) / peak_x < upper:
                values.append(_oscillator(peak_x, 1, cosh=True)[0])
            lower_bound += min(amplitude * value for value in values)
        if valid(lower_bound):
            continue
        middle = lower + (upper - lower) / 2
        if (middle in (lower, upper)
                or not valid(family_value_and_slope(equation_id, middle, c)[0])):
            reject()
        intervals.extend(((lower, middle), (middle, upper)))
    reject()
