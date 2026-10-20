"""Heat-capacity, phase-transition, and mixture enthalpy reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from chemworld.physchem.property_cards import _HEAT_ENTHALPY_REFERENCE_READING
from chemworld.physchem.property_equations import (
    _convert_input,
    _cp_polynomial,
    _validity_warnings,
    evaluate_correlation,
)
from chemworld.physchem.property_reports import (
    PropertyEvaluation,
    ValidityPolicy,
    _validate_phase,
)
from chemworld.physchem.specs import PropertyCorrelation


@dataclass(frozen=True)
class PhaseTransitionSpec:
    """A signed, auditable phase-transition enthalpy contract."""

    transition_id: str
    from_phase: str
    to_phase: str
    transition_temperature_K: float
    enthalpy_correlation: PropertyCorrelation
    source_note: str = ""

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise ValueError("transition_id cannot be empty")
        _validate_phase(self.from_phase)
        _validate_phase(self.to_phase)
        if self.from_phase == self.to_phase:
            raise ValueError("phase transition must connect different phases")
        if self.transition_temperature_K <= 0:
            raise ValueError("transition_temperature_K must be positive")
        if self.enthalpy_correlation.property_id not in {
            "heat_of_vaporization",
            "heat_of_fusion",
        }:
            raise ValueError(
                "PhaseTransitionSpec requires heat_of_vaporization or "
                "heat_of_fusion correlation"
            )

    def connects(self, phase_a: str, phase_b: str) -> bool:
        return {phase_a, phase_b} == {self.from_phase, self.to_phase}

    def direction_sign(self, from_phase: str, to_phase: str) -> float:
        if from_phase == self.from_phase and to_phase == self.to_phase:
            return 1.0
        if from_phase == self.to_phase and to_phase == self.from_phase:
            return -1.0
        raise ValueError(
            f"Transition {self.transition_id!r} does not connect "
            f"{from_phase!r} -> {to_phase!r}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "transition_id": self.transition_id,
            "from_phase": self.from_phase,
            "to_phase": self.to_phase,
            "transition_temperature_K": self.transition_temperature_K,
            "enthalpy_correlation": self.enthalpy_correlation.to_dict(),
            "source_note": self.source_note,
        }


@dataclass(frozen=True)
class PhaseEnthalpyReport:
    """Molar enthalpy change across a declared phase path."""

    component_id: str
    initial_phase: str
    final_phase: str
    initial_temperature_K: float
    final_temperature_K: float
    reference_temperature_K: float
    sensible_enthalpy_J_mol: float
    transition_enthalpy_J_mol: float
    total_enthalpy_J_mol: float
    heat_capacity_correlation_ids: tuple[str, ...]
    transition_ids: tuple[str, ...] = ()
    validity_warnings: tuple[str, ...] = ()
    reference_reading: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component_id cannot be empty")
        _validate_phase(self.initial_phase)
        _validate_phase(self.final_phase)
        for name, value in {
            "initial_temperature_K": self.initial_temperature_K,
            "final_temperature_K": self.final_temperature_K,
            "reference_temperature_K": self.reference_temperature_K,
            "sensible_enthalpy_J_mol": self.sensible_enthalpy_J_mol,
            "transition_enthalpy_J_mol": self.transition_enthalpy_J_mol,
            "total_enthalpy_J_mol": self.total_enthalpy_J_mol,
        }.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.initial_temperature_K <= 0 or self.final_temperature_K <= 0:
            raise ValueError("enthalpy report temperatures must be positive")
        if self.reference_temperature_K <= 0:
            raise ValueError("reference_temperature_K must be positive")
        if abs(
            self.sensible_enthalpy_J_mol
            + self.transition_enthalpy_J_mol
            - self.total_enthalpy_J_mol
        ) > 1e-8:
            raise ValueError("total_enthalpy_J_mol must equal sensible + transition")
        object.__setattr__(
            self,
            "heat_capacity_correlation_ids",
            tuple(self.heat_capacity_correlation_ids),
        )
        object.__setattr__(self, "transition_ids", tuple(self.transition_ids))
        object.__setattr__(self, "validity_warnings", tuple(self.validity_warnings))
        object.__setattr__(self, "reference_reading", tuple(self.reference_reading))

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "initial_phase": self.initial_phase,
            "final_phase": self.final_phase,
            "initial_temperature_K": self.initial_temperature_K,
            "final_temperature_K": self.final_temperature_K,
            "reference_temperature_K": self.reference_temperature_K,
            "sensible_enthalpy_J_mol": self.sensible_enthalpy_J_mol,
            "transition_enthalpy_J_mol": self.transition_enthalpy_J_mol,
            "total_enthalpy_J_mol": self.total_enthalpy_J_mol,
            "heat_capacity_correlation_ids": list(self.heat_capacity_correlation_ids),
            "transition_ids": list(self.transition_ids),
            "validity_warnings": list(self.validity_warnings),
            "reference_reading": list(self.reference_reading),
        }


@dataclass(frozen=True)
class MixtureEnthalpyLedger:
    """Mole-weighted enthalpy ledger for reactor and separation duties."""

    ledger_id: str
    contributions: dict[str, dict[str, float]]
    total_enthalpy_change_J: float
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.ledger_id:
            raise ValueError("ledger_id cannot be empty")
        if not self.contributions:
            raise ValueError("MixtureEnthalpyLedger requires contributions")
        if not isfinite(self.total_enthalpy_change_J):
            raise ValueError("total_enthalpy_change_J must be finite")
        recomputed = sum(
            contribution["enthalpy_change_J"]
            for contribution in self.contributions.values()
        )
        if abs(recomputed - self.total_enthalpy_change_J) > 1e-8:
            raise ValueError("total_enthalpy_change_J must match contributions")
        object.__setattr__(
            self,
            "contributions",
            {
                component_id: dict(contribution)
                for component_id, contribution in self.contributions.items()
            },
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "ledger_id": self.ledger_id,
            "contributions": {
                component_id: dict(contribution)
                for component_id, contribution in self.contributions.items()
            },
            "total_enthalpy_change_J": self.total_enthalpy_change_J,
            "warnings": list(self.warnings),
        }


def sensible_enthalpy_change(
    heat_capacity_correlation: PropertyCorrelation,
    *,
    initial_temperature_K: float,
    final_temperature_K: float,
    validity_policy: ValidityPolicy = "warn",
) -> PropertyEvaluation:
    """Integrate a supported molar heat-capacity correlation in J/mol."""

    if heat_capacity_correlation.property_id == "isochoric_heat_capacity":
        raise ValueError("Cv integrates internal energy; use caloric_polynomial_change")
    if heat_capacity_correlation.equation_id != "cp_polynomial":
        raise ValueError("Only cp_polynomial supports analytic enthalpy integration")
    T0 = _convert_input(initial_temperature_K, "K", heat_capacity_correlation, "temperature")
    T1 = _convert_input(final_temperature_K, "K", heat_capacity_correlation, "temperature")
    warnings = _validity_warnings(heat_capacity_correlation, {"temperature": T0})
    warnings += _validity_warnings(heat_capacity_correlation, {"temperature": T1})
    if warnings and validity_policy == "raise":
        raise ValueError("; ".join(warnings))
    coeffs = heat_capacity_correlation.coefficients
    _ensure_positive_heat_capacity_interval(heat_capacity_correlation, T0, T1)
    value = _cp_polynomial_integral(T1, coeffs) - _cp_polynomial_integral(T0, coeffs)
    return PropertyEvaluation(
        property_id="sensible_enthalpy",
        correlation_id=heat_capacity_correlation.correlation_id,
        equation_id="cp_polynomial_integral",
        value=value,
        unit="J/mol",
        inputs={
            "initial_temperature": T0,
            "final_temperature": T1,
        },
        warnings=() if validity_policy == "ignore" else warnings,
    )


def heat_capacity_report(
    heat_capacity_correlation: PropertyCorrelation,
    *,
    temperature_K: float,
    validity_policy: ValidityPolicy = "warn",
) -> PropertyEvaluation:
    """Evaluate a phase-aware heat-capacity correlation with strict positivity."""

    _validate_heat_capacity_correlation(heat_capacity_correlation)
    result = evaluate_correlation(
        heat_capacity_correlation,
        temperature_K=temperature_K,
        validity_policy=validity_policy,
    )
    if result.value <= 0:
        raise ValueError(
            f"Heat capacity must be positive: {heat_capacity_correlation.correlation_id}"
        )
    return result


def phase_sensible_enthalpy_report(
    *,
    component_id: str,
    phase: str,
    heat_capacity_correlation: PropertyCorrelation,
    initial_temperature_K: float,
    final_temperature_K: float,
    reference_temperature_K: float = 298.15,
    validity_policy: ValidityPolicy = "warn",
) -> PhaseEnthalpyReport:
    """Report same-phase sensible enthalpy with an explicit reference state."""

    _validate_phase(phase)
    _validate_heat_capacity_correlation(heat_capacity_correlation)
    _validate_correlation_phase(heat_capacity_correlation, phase)
    if reference_temperature_K <= 0:
        raise ValueError("reference_temperature_K must be positive")
    initial_relative = sensible_enthalpy_change(
        heat_capacity_correlation,
        initial_temperature_K=reference_temperature_K,
        final_temperature_K=initial_temperature_K,
        validity_policy=validity_policy,
    )
    final_relative = sensible_enthalpy_change(
        heat_capacity_correlation,
        initial_temperature_K=reference_temperature_K,
        final_temperature_K=final_temperature_K,
        validity_policy=validity_policy,
    )
    sensible_delta = final_relative.value - initial_relative.value
    warnings = initial_relative.warnings + final_relative.warnings
    return PhaseEnthalpyReport(
        component_id=component_id,
        initial_phase=phase,
        final_phase=phase,
        initial_temperature_K=initial_temperature_K,
        final_temperature_K=final_temperature_K,
        reference_temperature_K=reference_temperature_K,
        sensible_enthalpy_J_mol=sensible_delta,
        transition_enthalpy_J_mol=0.0,
        total_enthalpy_J_mol=sensible_delta,
        heat_capacity_correlation_ids=(heat_capacity_correlation.correlation_id,),
        validity_warnings=() if validity_policy == "ignore" else warnings,
        reference_reading=_HEAT_ENTHALPY_REFERENCE_READING,
    )


def phase_transition_enthalpy(
    transition: PhaseTransitionSpec,
    *,
    from_phase: str,
    to_phase: str,
    temperature_K: float | None = None,
    validity_policy: ValidityPolicy = "warn",
) -> PropertyEvaluation:
    """Return signed molar latent heat for a declared transition direction."""

    _validate_phase(from_phase)
    _validate_phase(to_phase)
    if from_phase == to_phase:
        raise ValueError("phase_transition_enthalpy requires different phases")
    sign = transition.direction_sign(from_phase, to_phase)
    temperature = (
        transition.transition_temperature_K
        if temperature_K is None
        else temperature_K
    )
    latent = evaluate_correlation(
        transition.enthalpy_correlation,
        temperature_K=temperature,
        validity_policy=validity_policy,
    )
    if latent.value < 0:
        raise ValueError(f"Latent heat must be nonnegative: {transition.transition_id}")
    return PropertyEvaluation(
        property_id=f"{transition.enthalpy_correlation.property_id}_signed",
        correlation_id=transition.enthalpy_correlation.correlation_id,
        equation_id=transition.enthalpy_correlation.equation_id,
        value=sign * latent.to("J/mol").value,
        unit="J/mol",
        inputs={
            "temperature": latent.inputs["temperature"],
            "direction_sign": sign,
        },
        warnings=latent.warnings,
    )


def phase_path_enthalpy_report(
    *,
    component_id: str,
    heat_capacity_correlations: Mapping[str, PropertyCorrelation],
    transitions: Sequence[PhaseTransitionSpec],
    initial_phase: str,
    final_phase: str,
    initial_temperature_K: float,
    final_temperature_K: float,
    reference_temperature_K: float = 298.15,
    validity_policy: ValidityPolicy = "warn",
) -> PhaseEnthalpyReport:
    """Integrate heat capacity and latent heat along a phase path."""

    _validate_phase(initial_phase)
    _validate_phase(final_phase)
    if reference_temperature_K <= 0:
        raise ValueError("reference_temperature_K must be positive")
    phase_path = _phase_path(initial_phase, final_phase, transitions)
    current_phase = initial_phase
    current_temperature = initial_temperature_K
    sensible_total = 0.0
    transition_total = 0.0
    warnings: list[str] = []
    cp_ids: list[str] = []
    transition_ids: list[str] = []
    for next_phase, transition in phase_path:
        cp = _heat_capacity_for_phase(heat_capacity_correlations, current_phase)
        cp_ids.append(cp.correlation_id)
        _validate_correlation_phase(cp, current_phase)
        sensible = sensible_enthalpy_change(
            cp,
            initial_temperature_K=current_temperature,
            final_temperature_K=transition.transition_temperature_K,
            validity_policy=validity_policy,
        )
        latent = phase_transition_enthalpy(
            transition,
            from_phase=current_phase,
            to_phase=next_phase,
            validity_policy=validity_policy,
        )
        sensible_total += sensible.value
        transition_total += latent.value
        warnings.extend(sensible.warnings)
        warnings.extend(latent.warnings)
        transition_ids.append(transition.transition_id)
        current_phase = next_phase
        current_temperature = transition.transition_temperature_K
    cp = _heat_capacity_for_phase(heat_capacity_correlations, current_phase)
    cp_ids.append(cp.correlation_id)
    _validate_correlation_phase(cp, current_phase)
    final_sensible = sensible_enthalpy_change(
        cp,
        initial_temperature_K=current_temperature,
        final_temperature_K=final_temperature_K,
        validity_policy=validity_policy,
    )
    sensible_total += final_sensible.value
    warnings.extend(final_sensible.warnings)
    return PhaseEnthalpyReport(
        component_id=component_id,
        initial_phase=initial_phase,
        final_phase=final_phase,
        initial_temperature_K=initial_temperature_K,
        final_temperature_K=final_temperature_K,
        reference_temperature_K=reference_temperature_K,
        sensible_enthalpy_J_mol=sensible_total,
        transition_enthalpy_J_mol=transition_total,
        total_enthalpy_J_mol=sensible_total + transition_total,
        heat_capacity_correlation_ids=tuple(dict.fromkeys(cp_ids)),
        transition_ids=tuple(transition_ids),
        validity_warnings=() if validity_policy == "ignore" else tuple(warnings),
        reference_reading=_HEAT_ENTHALPY_REFERENCE_READING,
    )


def mixture_enthalpy_ledger(
    *,
    component_amounts_mol: Mapping[str, float],
    component_reports: Mapping[str, PhaseEnthalpyReport],
    ledger_id: str = "mixture_enthalpy_ledger",
) -> MixtureEnthalpyLedger:
    """Build a mole-weighted enthalpy ledger from component reports."""

    if not component_amounts_mol:
        raise ValueError("component_amounts_mol cannot be empty")
    contributions: dict[str, dict[str, float]] = {}
    warnings: list[str] = []
    total = 0.0
    for component_id, amount_mol in component_amounts_mol.items():
        if amount_mol < 0:
            raise ValueError(f"amount_mol cannot be negative for {component_id!r}")
        if component_id not in component_reports:
            raise ValueError(f"Missing enthalpy report for component {component_id!r}")
        report = component_reports[component_id]
        if report.component_id != component_id:
            raise ValueError(
                f"Report component_id {report.component_id!r} does not match "
                f"ledger key {component_id!r}"
            )
        contribution = amount_mol * report.total_enthalpy_J_mol
        total += contribution
        contributions[component_id] = {
            "amount_mol": amount_mol,
            "molar_enthalpy_change_J_mol": report.total_enthalpy_J_mol,
            "enthalpy_change_J": contribution,
        }
        warnings.extend(report.validity_warnings)
    return MixtureEnthalpyLedger(
        ledger_id=ledger_id,
        contributions=contributions,
        total_enthalpy_change_J=total,
        warnings=tuple(warnings),
    )


def _cp_polynomial_integral(T: float, coeffs: dict[str, float]) -> float:
    return (
        coeffs.get("a", 0.0) * T
        + 0.5 * coeffs.get("b", 0.0) * T**2
        + (1.0 / 3.0) * coeffs.get("c", 0.0) * T**3
        + 0.25 * coeffs.get("d", 0.0) * T**4
        + 0.2 * coeffs.get("e", 0.0) * T**5
    )


def caloric_polynomial_change(
    correlation: PropertyCorrelation, *, initial_temperature_K: float,
    final_temperature_K: float, entropy: bool = False,
    validity_policy: ValidityPolicy = "raise",
) -> PropertyEvaluation:
    """Integrate a declared Cp or Cv polynomial, in SI, on its valid interval.

    The energy basis stays explicit: Cp integrates H, Cv integrates U. Entropy
    integration uses Cp/T; a caller holding Cv needs its phase/EOS conversion.
    """
    from chemworld.foundation.units import convert_value

    if correlation.input_units.get("temperature", "K") != "K":
        raise ValueError("caloric integration requires an absolute-Kelvin correlation")
    if entropy and correlation.property_id == "isochoric_heat_capacity":
        raise ValueError("entropy needs Cp or a declared phase/EOS conversion from Cv")
    factor = convert_value(1.0, correlation.output_unit, "J/(mol*K)")
    normalized = correlation if factor == 1.0 else PropertyCorrelation(
        correlation_id=correlation.correlation_id, property_id=correlation.property_id,
        equation_id=correlation.equation_id,
        coefficients={key: value * factor for key, value in correlation.coefficients.items()},
        input_units=correlation.input_units, output_unit="J/(mol*K)",
        validity_ranges=correlation.validity_ranges, source_note=correlation.source_note,
        metadata=correlation.metadata,
    )
    _ensure_positive_heat_capacity_interval(normalized, initial_temperature_K, final_temperature_K)
    warnings = tuple(dict.fromkeys(
        (*_validity_warnings(normalized, {"temperature": initial_temperature_K}),
         *_validity_warnings(normalized, {"temperature": final_temperature_K}))))
    if warnings and validity_policy == "raise":
        raise ValueError("; ".join(warnings))
    if entropy:
        from math import log

        def primitive(temperature):
            coefficients = normalized.coefficients
            return coefficients.get("a", 0.0) * log(temperature) + sum(
                coefficients.get(key, 0.0) * temperature ** power / power
                for power, key in enumerate(("b", "c", "d", "e"), 1))
    else:
        def primitive(temperature):
            return _cp_polynomial_integral(temperature, normalized.coefficients)
    if not entropy and not any(normalized.coefficients.get(key, 0) for key in ("b", "c", "d", "e")):
        value = normalized.coefficients.get("a", 0.0) * (
            final_temperature_K - initial_temperature_K)
    else:
        value = primitive(final_temperature_K) - primitive(initial_temperature_K)
    if not isfinite(value):
        raise ValueError("caloric integration exceeds the finite numerical domain")
    basis = ("internal_energy" if correlation.property_id == "isochoric_heat_capacity"
             else "enthalpy")
    return PropertyEvaluation(
        "entropy_change" if entropy else f"{basis}_change", correlation.correlation_id,
        "cp_over_T_integral" if entropy else "caloric_polynomial_integral", value,
        "J/(mol*K)" if entropy else "J/mol",
        {"initial_temperature": initial_temperature_K, "final_temperature": final_temperature_K},
        warnings,
    )


def _validate_heat_capacity_correlation(correlation: PropertyCorrelation) -> None:
    if correlation.equation_id != "cp_polynomial":
        raise ValueError("Only cp_polynomial supports analytic heat-capacity reports")
    if correlation.property_id not in {
        "heat_capacity",
        "ideal_gas_heat_capacity",
        "liquid_heat_capacity",
        "solid_heat_capacity",
        "isochoric_heat_capacity",
    }:
        raise ValueError(
            "heat-capacity report requires heat_capacity, ideal_gas_heat_capacity, "
            "liquid_heat_capacity, or solid_heat_capacity"
        )


def _validate_correlation_phase(correlation: PropertyCorrelation, phase: str) -> None:
    declared_phase = correlation.metadata.get("phase")
    if declared_phase is not None and declared_phase != phase:
        raise ValueError(
            f"Correlation {correlation.correlation_id!r} declares phase "
            f"{declared_phase!r}, not {phase!r}"
        )
    property_phase = {
        "ideal_gas_heat_capacity": "gas",
        "liquid_heat_capacity": "liquid",
        "solid_heat_capacity": "solid",
    }.get(correlation.property_id)
    if property_phase is not None and property_phase != phase:
        raise ValueError(
            f"Correlation {correlation.correlation_id!r} is for "
            f"{property_phase!r}, not {phase!r}"
        )


def _ensure_positive_heat_capacity_interval(
    correlation: PropertyCorrelation,
    initial_temperature: float,
    final_temperature: float,
) -> None:
    _validate_heat_capacity_correlation(correlation)
    if initial_temperature <= 0 or final_temperature <= 0:
        raise ValueError("Heat-capacity integration temperatures must be positive")
    lower = min(initial_temperature, final_temperature)
    upper = max(initial_temperature, final_temperature)
    coefficients = [correlation.coefficients.get(key, 0.0) for key in ("a", "b", "c", "d", "e")]
    critical = (np.polynomial.Polynomial(coefficients).deriv().roots()
                if any(coefficients[1:]) else ())
    sample_temperatures = (lower, upper, *(float(root.real) for root in critical
                                           if abs(root.imag) < 1e-10 and lower < root.real < upper))
    for temperature in sample_temperatures:
        cp = _cp_polynomial(temperature, correlation.coefficients)
        if (cp < 0 or not isfinite(cp)
                or (cp == 0 and correlation.property_id != "isochoric_heat_capacity")):
            raise ValueError(
                f"Heat capacity must stay positive over integration interval: "
                f"{correlation.correlation_id}"
            )


def _heat_capacity_for_phase(
    correlations: Mapping[str, PropertyCorrelation],
    phase: str,
) -> PropertyCorrelation:
    _validate_phase(phase)
    if phase not in correlations:
        raise ValueError(f"Missing heat-capacity correlation for phase {phase!r}")
    correlation = correlations[phase]
    _validate_heat_capacity_correlation(correlation)
    return correlation


def _phase_path(
    initial_phase: str,
    final_phase: str,
    transitions: Sequence[PhaseTransitionSpec],
) -> tuple[tuple[str, PhaseTransitionSpec], ...]:
    if initial_phase == final_phase:
        return ()
    queue: list[tuple[str, tuple[tuple[str, PhaseTransitionSpec], ...]]] = [
        (initial_phase, ())
    ]
    visited = {initial_phase}
    while queue:
        phase, path = queue.pop(0)
        for transition in transitions:
            next_phase: str | None = None
            if transition.from_phase == phase:
                next_phase = transition.to_phase
            elif transition.to_phase == phase:
                next_phase = transition.from_phase
            if next_phase is None or next_phase in visited:
                continue
            next_path = (*path, (next_phase, transition))
            if next_phase == final_phase:
                return next_path
            visited.add(next_phase)
            queue.append((next_phase, next_path))
    raise ValueError(
        f"No phase-transition path from {initial_phase!r} to {final_phase!r}"
    )

