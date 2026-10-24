"""Auditable unit and parameter contracts for Arrhenius-family rate laws.

This module is deliberately independent from runtime dispatch.  It describes
the dimensional contract that a future reaction provider must satisfy before
WF-110 integrates it into a new World Law.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any

from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelCard,
    ValidationEvidence,
)
from chemworld.physchem.reaction_network_specs import ReactionSpec
from chemworld.physchem.reaction_rate_laws import prefixed_arrhenius_params, validate_pressure_rate

ARRHENIUS_FAMILY = frozenset(
    {
        "arrhenius",
        "modified_arrhenius",
        "reversible_arrhenius",
        "third_body_arrhenius",
        "lindemann_falloff",
        "troe_falloff",
        "pressure_log",
        "chebyshev",
    }
)
CONCENTRATION_BASIS = "mol/L"
RATE_BASIS = "mol/(L*s)"
ACTIVATION_ENERGY_UNIT = "J/mol"
CANTERA_COMMIT = "ddb114abdbe5170420b475eac3bb2ffa6e19d05b"
RMG_PY_COMMIT = "b858624649205fc8ae08aec601c4c216e9edcee0"


@dataclass(frozen=True)
class ReactionRateContractReport:
    """Dimension and validity report for one rate-law declaration."""

    reaction_id: str
    equation_id: str
    concentration_basis: str
    kinetic_basis: str
    standard_concentration_mol_L: float
    rate_basis: str
    forward_order: float
    effective_forward_order: float
    reverse_order: float | None
    forward_rate_constant_unit: str
    reverse_rate_constant_unit: str | None
    equilibrium_constant_unit: str | None
    low_pressure_rate_constant_unit: str | None
    high_pressure_rate_constant_unit: str | None
    activation_energy_unit: str
    diagnostics: tuple[str, ...]
    violations: tuple[str, ...]
    provenance: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "equation_id": self.equation_id,
            "concentration_basis": self.concentration_basis,
            "kinetic_basis": self.kinetic_basis,
            "standard_concentration_mol_L": self.standard_concentration_mol_L,
            "rate_basis": self.rate_basis,
            "forward_order": self.forward_order,
            "effective_forward_order": self.effective_forward_order,
            "reverse_order": self.reverse_order,
            "forward_rate_constant_unit": self.forward_rate_constant_unit,
            "reverse_rate_constant_unit": self.reverse_rate_constant_unit,
            "equilibrium_constant_unit": self.equilibrium_constant_unit,
            "low_pressure_rate_constant_unit": self.low_pressure_rate_constant_unit,
            "high_pressure_rate_constant_unit": self.high_pressure_rate_constant_unit,
            "activation_energy_unit": self.activation_energy_unit,
            "diagnostics": list(self.diagnostics),
            "violations": list(self.violations),
            "passed": self.passed,
            "provenance": list(self.provenance),
        }


def rate_coefficient_unit(
    overall_order: float,
    *,
    temperature_exponent: float = 0.0,
) -> str:
    """Return the ChemWorld ``mol/L`` pre-exponential-factor unit.

    For ``r = A T**b exp(-Ea/RT) product(C_i**order_i)``, the rate basis is
    ``mol/(L*s)``.  Therefore ``A`` has concentration exponent
    ``1 - overall_order`` and temperature exponent ``-b``.
    """

    if not isfinite(overall_order) or overall_order <= 0.0:
        raise ValueError("overall_order must be finite and positive")
    if not isfinite(temperature_exponent):
        raise ValueError("temperature_exponent must be finite")
    factors = (
        _unit_factor("L", overall_order - 1.0),
        _unit_factor("mol", 1.0 - overall_order),
        "s^-1",
        _unit_factor("K", -temperature_exponent),
    )
    return " ".join(factor for factor in factors if factor)


def concentration_equilibrium_constant_unit(delta_order: float) -> str:
    """Return the unit of ``Kc = product(C_products)/product(C_reactants)``."""

    if not isfinite(delta_order):
        raise ValueError("delta_order must be finite")
    if abs(delta_order) < 1.0e-12:
        return "dimensionless"
    return f"(mol/L)^{_format_exponent(delta_order)}"


def audit_reaction_rate_contract(reaction: ReactionSpec) -> ReactionRateContractReport:
    """Audit one Arrhenius-family declaration without changing its evaluation."""

    if reaction.rate_law.parameters.get("mechanism_schema") == "cantera-gas-v2":
        from chemworld.physchem.mechanism_rates import audit_mechanism_rate_contract

        return audit_mechanism_rate_contract(reaction)
    equation_id = reaction.rate_law.equation_id
    params = reaction.rate_law.parameters
    violations: list[str] = []
    diagnostics: list[str] = []
    forward_order = sum(reaction.kinetic_forward_orders.values())
    reverse_order = sum(reaction.kinetic_reverse_orders.values()) if reaction.reversible else None
    uses_activities = reaction.rate_law.uses_activities
    effective_forward_order = forward_order
    forward_b = 0.0
    reverse_unit: str | None = None
    equilibrium_unit: str | None = None
    low_unit: str | None = None
    high_unit: str | None = None

    if equation_id not in ARRHENIUS_FAMILY:
        violations.append(f"unsupported Arrhenius-family equation_id: {equation_id}")
    elif equation_id in {"pressure_log", "chebyshev"}:
        try:
            validate_pressure_rate(params, equation_id)
        except (ValueError, TypeError, KeyError, OverflowError) as error:
            violations.append(str(error))
        if params.get("pressure_basis") != "ideal_gas":
            violations.append("pressure kinetics require ideal_gas pressure basis")
        if uses_activities:
            violations.append("pressure kinetics use concentrations, not activities")
        diagnostics.append(
            "pressure coefficient uses finite T/P domain and mol/L; no extrapolation"
        )
    elif equation_id in {"lindemann_falloff", "troe_falloff"}:
        low = _falloff_group(params, "low", violations)
        high = _falloff_group(params, "high", violations)
        low_b = _finite_parameter(low, "b", default=0.0, violations=violations)
        high_b = _finite_parameter(high, "b", default=0.0, violations=violations)
        _positive_parameter(low, "A", violations)
        _positive_parameter(high, "A", violations)
        _activation_energy(low, violations)
        _activation_energy(high, violations)
        low_unit = _kinetic_rate_coefficient_unit(
            forward_order + (0.0 if uses_activities else 1.0),
            temperature_exponent=low_b,
            uses_activities=uses_activities,
            extra_concentration_order=1.0,
        )
        high_unit = _kinetic_rate_coefficient_unit(
            forward_order,
            temperature_exponent=high_b,
            uses_activities=uses_activities,
        )
        effective_forward_order = forward_order
        forward_b = high_b
        diagnostics.append("falloff low-pressure order includes one effective third body")
        if equation_id == "troe_falloff":
            _validate_troe(params, violations)
        _validate_third_body(params, violations)
    else:
        forward_b = _finite_parameter(params, "b", default=0.0, violations=violations)
        _positive_parameter(params, "A", violations)
        _activation_energy(params, violations)
        if equation_id == "modified_arrhenius" and "b" not in params:
            violations.append("modified_arrhenius requires explicit b")
        if equation_id == "third_body_arrhenius":
            effective_forward_order += 1.0
            _validate_third_body(params, violations)
            diagnostics.append("effective forward order includes one third body")
        if equation_id == "reversible_arrhenius":
            reverse_unit, equilibrium_unit = _validate_reverse_contract(
                reaction,
                params,
                reverse_order=reverse_order,
                forward_temperature_exponent=forward_b,
                violations=violations,
            )

    if params.get("reverse_basis") == "ideal_gas_nasa7":
        if not reaction.reversible or uses_activities:
            violations.append("NASA7 gas reverse basis requires reversible concentration kinetics")
        _positive_parameter(params, "reference_pressure_Pa", violations)
        for key in set(reaction.kinetic_forward_orders) | set(reaction.kinetic_reverse_orders):
            if (
                abs(
                    reaction.kinetic_reverse_orders.get(key, 0)
                    - reaction.kinetic_forward_orders.get(key, 0)
                    - reaction.stoichiometry.get(key, 0)
                )
                > 1e-12
            ):
                violations.append(
                    "gas reverse orders must obey stoichiometry, including collision partners"
                )
        reverse_unit = rate_coefficient_unit(
            (reverse_order or 1) + (equation_id == "third_body_arrhenius")
        )
        equilibrium_unit = concentration_equilibrium_constant_unit(
            sum(reaction.stoichiometry.values())
        )
        diagnostics.append(
            "ideal-gas NASA7 Kc uses reference_pressure/(R*T*1000), not fixed 1 mol/L"
        )

    forward_unit = _kinetic_rate_coefficient_unit(
        effective_forward_order,
        temperature_exponent=forward_b,
        uses_activities=uses_activities,
        extra_concentration_order=(
            1.0 if uses_activities and equation_id == "third_body_arrhenius" else 0.0
        ),
    )
    diagnostics.extend(
        (
            f"forward concentration order={forward_order:g}",
            f"effective forward order={effective_forward_order:g}",
            (
                "rate powers use dimensionless activities a_i=gamma_i*C_i/C_standard"
                if uses_activities
                else "rate powers use concentrations in mol/L"
            ),
            "activation energy is interpreted in J/mol",
        )
    )
    return ReactionRateContractReport(
        reaction_id=reaction.reaction_id,
        equation_id=equation_id,
        concentration_basis=reaction.rate_law.concentration_basis,
        kinetic_basis="activity" if uses_activities else "concentration",
        standard_concentration_mol_L=(reaction.rate_law.standard_concentration_mol_L),
        rate_basis=RATE_BASIS,
        forward_order=forward_order,
        effective_forward_order=effective_forward_order,
        reverse_order=reverse_order,
        forward_rate_constant_unit=forward_unit,
        reverse_rate_constant_unit=reverse_unit,
        equilibrium_constant_unit=equilibrium_unit,
        low_pressure_rate_constant_unit=low_unit,
        high_pressure_rate_constant_unit=high_unit,
        activation_energy_unit=ACTIVATION_ENERGY_UNIT,
        diagnostics=tuple(diagnostics),
        violations=tuple(dict.fromkeys(violations)),
        provenance=(
            (
                "Cantera commit "
                f"{CANTERA_COMMIT}: include/cantera/kinetics/Arrhenius.h and "
                "doc/sphinx/reference/kinetics/rate-constants.md"
            ),
            (f"RMG-Py commit {RMG_PY_COMMIT}: rmgpy/kinetics/arrhenius.pyx"),
            "ChemWorld concentration basis mol/L; 1 L/mol is numerically 1 m^3/kmol",
        ),
    )


def reaction_rate_contract_model_card() -> ModelCard:
    """Return the model card for the unit-contract validation slice."""

    return ModelCard(
        model_id="chemworld_arrhenius_unit_contract_vnext",
        module_id="reaction_kinetics",
        title="Arrhenius-Family Rate Constant Unit Contract",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        summary=(
            "A diagnostic contract deriving pre-exponential-factor units from "
            "reaction order, temperature exponent, reversibility, and falloff role."
        ),
        equations=(
            "r = A T^b exp(-Ea/RT) product_i C_i^order_i",
            "[A] = L^(n-1) mol^(1-n) s^-1 K^-b",
            "falloff: [A_low] uses n+1 and [A_high] uses n",
        ),
        assumptions=(
            "homogeneous concentration basis is mol/L",
            "reaction rate basis is mol/(L*s)",
            "third-body concentration uses the same mol/L basis",
        ),
        validity_limits=(
            "covers only the declared Arrhenius, reversible, third-body, and falloff laws",
            "does not validate surface, sticking, pressure-log, or Chebyshev kinetics",
            "does not alter or certify the runtime ODE implementation",
        ),
        failure_modes=(
            "missing or nonpositive pre-exponential factors produce violations",
            "nonfinite exponents or activation energies produce violations",
            "incomplete reversible or Troe declarations produce violations",
        ),
        units={
            "concentration": CONCENTRATION_BASIS,
            "reaction_rate": RATE_BASIS,
            "activation_energy": ACTIVATION_ENERGY_UNIT,
            "temperature": "K",
        },
        reference_reading=(
            (
                f"Cantera {CANTERA_COMMIT}: Arrhenius.h documents A units as powers "
                "of m, kmol, and s and Ea in J/kmol."
            ),
            (
                f"RMG-Py {RMG_PY_COMMIT}: arrhenius.pyx represents A, n, Ea, and T0 "
                "with explicit quantity units."
            ),
        ),
        validation_evidence=(
            ValidationEvidence(
                evidence_id="arrhenius-unit-order-tests",
                evidence_type="unit_test",
                description=(
                    "First-, second-, reversible-, third-body-, and falloff-order "
                    "unit identities are checked analytically."
                ),
                status="implemented",
                reference_backend="Cantera/RMG-Py source contracts",
                command_or_path="tests/test_reaction_rate_contracts.py",
                tolerance="exact symbolic unit identity",
            ),
            ValidationEvidence(
                evidence_id="arrhenius-invalid-declaration-domain-tests",
                evidence_type="unit_test",
                description=(
                    "Failure-domain tests reject missing or nonpositive rate "
                    "parameters, invalid third-body efficiencies, and out-of-bound "
                    "Troe declarations with auditable violations."
                ),
                status="implemented",
                reference_backend="ChemWorld declared Arrhenius contract bounds",
                command_or_path="tests/test_reaction_rate_contracts.py",
                tolerance="exact violation category and message match",
            ),
        ),
        model_limit_notes=(
            "This reference-validated label applies only to dimensional and declaration checks.",
            "Runtime kinetics remain at their separately declared maturity.",
        ),
        intended_use=(
            "pre-integration validation for reaction providers",
            "mechanism authoring diagnostics",
            "adapter review for World Law vNext",
        ),
    )


def _validate_reverse_contract(
    reaction: ReactionSpec,
    params: Mapping[str, object],
    *,
    reverse_order: float | None,
    forward_temperature_exponent: float,
    violations: list[str],
) -> tuple[str | None, str | None]:
    if not reaction.reversible:
        violations.append("reversible_arrhenius requires a reversible reaction equation")
    if reverse_order is None or reverse_order <= 0.0:
        violations.append("reversible_arrhenius requires positive product order")
        return None, None
    if "A_reverse" in params:
        _positive_parameter(params, "A_reverse", violations)
        reverse_b = _finite_parameter(
            params,
            "b_reverse",
            default=0.0,
            violations=violations,
        )
        _activation_energy(params, violations, prefix="reverse")
        return (
            _kinetic_rate_coefficient_unit(
                reverse_order,
                temperature_exponent=reverse_b,
                uses_activities=reaction.rate_law.uses_activities,
            ),
            None,
        )
    if "K_eq" in params:
        _positive_parameter(params, "K_eq", violations)
        expected_basis = "activity" if reaction.rate_law.uses_activities else "concentration"
        if str(params.get("K_eq_basis", "")).lower() != expected_basis:
            violations.append(f"explicit K_eq requires K_eq_basis='{expected_basis}'")
        return (
            _kinetic_rate_coefficient_unit(
                reverse_order,
                temperature_exponent=forward_temperature_exponent,
                uses_activities=reaction.rate_law.uses_activities,
            ),
            (
                "dimensionless"
                if reaction.rate_law.uses_activities
                else concentration_equilibrium_constant_unit(
                    reverse_order - reaction_order(reaction)
                )
            ),
        )
    source = str(params.get("K_eq_source", params.get("equilibrium_source", ""))).lower()
    if source not in {"nasa7", "species_thermo", "thermochemistry"}:
        violations.append(
            "reversible_arrhenius requires A_reverse, positive K_eq, or a thermochemistry source"
        )
    elif (
        reaction.kinetic_forward_orders != reaction.reactants
        or reaction.kinetic_reverse_orders != reaction.products
    ):
        violations.append(
            "thermochemical detailed balance requires kinetic orders to match stoichiometry"
        )
    return (
        _kinetic_rate_coefficient_unit(
            reverse_order,
            temperature_exponent=forward_temperature_exponent,
            uses_activities=reaction.rate_law.uses_activities,
        ),
        (
            "dimensionless"
            if reaction.rate_law.uses_activities
            else concentration_equilibrium_constant_unit(reverse_order - reaction_order(reaction))
        ),
    )


def reaction_order(reaction: ReactionSpec) -> float:
    """Return the explicitly declared (or stoichiometric default) forward order."""

    return sum(reaction.kinetic_forward_orders.values())


def _kinetic_rate_coefficient_unit(
    overall_order: float,
    *,
    temperature_exponent: float,
    uses_activities: bool,
    extra_concentration_order: float = 0.0,
) -> str:
    if not uses_activities:
        return rate_coefficient_unit(
            overall_order,
            temperature_exponent=temperature_exponent,
        )
    # Activity powers are dimensionless.  Only an explicit third-body factor
    # contributes a concentration dimension to the rate coefficient.
    if abs(extra_concentration_order) < 1.0e-12:
        factors = (
            "mol L^-1 s^-1",
            _unit_factor("K", -temperature_exponent),
        )
        return " ".join(factor for factor in factors if factor)
    return rate_coefficient_unit(
        extra_concentration_order,
        temperature_exponent=temperature_exponent,
    )


def _falloff_group(
    params: Mapping[str, object],
    prefix: str,
    violations: list[str],
) -> dict[str, object]:
    try:
        return prefixed_arrhenius_params(params, prefix)
    except (TypeError, ValueError) as error:
        violations.append(str(error))
        return {"A": 0.0, "b": 0.0, "Ea_J_per_mol": 0.0}


def _validate_third_body(params: Mapping[str, object], violations: list[str]) -> None:
    default = _finite_parameter(
        params,
        "default_efficiency",
        default=1.0,
        violations=violations,
    )
    if default < 0.0:
        violations.append("default_efficiency must be nonnegative")
    payload = params.get("third_body_efficiencies", params.get("efficiencies", {}))
    if not isinstance(payload, Mapping):
        violations.append("third_body_efficiencies must be a mapping")
        return
    for species_id, raw_value in payload.items():
        value = _as_finite_float(raw_value, f"third-body efficiency {species_id}", violations)
        if value is not None and value < 0.0:
            violations.append(f"third-body efficiency {species_id} must be nonnegative")


def _validate_troe(params: Mapping[str, object], violations: list[str]) -> None:
    alpha = _required_finite_parameter(params, "troe_a", violations)
    if alpha is not None and not 0.0 <= alpha <= 1.0:
        violations.append("troe_a must lie in [0, 1]")
    for key in ("troe_T1", "troe_T3"):
        value = _required_finite_parameter(params, key, violations)
        if value is not None and value <= 0.0:
            violations.append(f"{key} must be positive")
    if "troe_T2" in params:
        value = _required_finite_parameter(params, "troe_T2", violations)
        if value is not None and value <= 0.0:
            violations.append("troe_T2 must be positive")


def _activation_energy(
    params: Mapping[str, object],
    violations: list[str],
    *,
    prefix: str = "",
) -> None:
    if prefix == "reverse":
        key = "Ea_reverse_J_per_mol" if "Ea_reverse_J_per_mol" in params else "Ea_reverse"
    else:
        key = "Ea_J_per_mol" if "Ea_J_per_mol" in params else "Ea"
    if key in params:
        _required_finite_parameter(params, key, violations)


def _positive_parameter(
    params: Mapping[str, object],
    key: str,
    violations: list[str],
) -> None:
    value = _required_finite_parameter(params, key, violations)
    if value is not None and value <= 0.0:
        violations.append(f"{key} must be positive")


def _required_finite_parameter(
    params: Mapping[str, object],
    key: str,
    violations: list[str],
) -> float | None:
    if key not in params:
        violations.append(f"missing numeric rate-law parameter: {key}")
        return None
    return _as_finite_float(params[key], key, violations)


def _finite_parameter(
    params: Mapping[str, object],
    key: str,
    *,
    default: float,
    violations: list[str],
) -> float:
    if key not in params:
        return default
    value = _as_finite_float(params[key], key, violations)
    return default if value is None else value


def _as_finite_float(
    raw_value: object,
    label: str,
    violations: list[str],
) -> float | None:
    if not isinstance(raw_value, int | float | str):
        violations.append(f"{label} must be numeric")
        return None
    try:
        value = float(raw_value)
    except ValueError:
        violations.append(f"{label} must be numeric")
        return None
    if not isfinite(value):
        violations.append(f"{label} must be finite")
        return None
    return value


def _unit_factor(symbol: str, exponent: float) -> str:
    if abs(exponent) < 1.0e-12:
        return ""
    if abs(exponent - 1.0) < 1.0e-12:
        return symbol
    return f"{symbol}^{_format_exponent(exponent)}"


def _format_exponent(exponent: float) -> str:
    rounded = round(exponent)
    if abs(exponent - rounded) < 1.0e-12:
        return str(rounded)
    return f"{exponent:.12g}"


__all__ = [
    "ACTIVATION_ENERGY_UNIT",
    "ARRHENIUS_FAMILY",
    "CANTERA_COMMIT",
    "CONCENTRATION_BASIS",
    "RATE_BASIS",
    "RMG_PY_COMMIT",
    "ReactionRateContractReport",
    "audit_reaction_rate_contract",
    "concentration_equilibrium_constant_unit",
    "rate_coefficient_unit",
    "reaction_order",
    "reaction_rate_contract_model_card",
]
