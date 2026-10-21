"""Thermochemistry kernels for species and reactions.

The first professional slice implements the NASA7 polynomial form used by
Cantera YAML species entries and RMG NASA exports.  It intentionally stops at a
small, auditable core: species Cp/H/S/G, reaction Delta H/S/G, and equilibrium
constants from species Gibbs energies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from math import exp, fsum, isfinite, log
from typing import Any

from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.reaction_network import R_J_PER_MOL_K
from chemworld.physchem.thermochemistry_cards import thermochemistry_model_cards

_EXPONENT_LIMIT = 700.0


@dataclass(frozen=True)
class NASA7TemperatureSegment:
    """Single NASA7 coefficient segment.

    Coefficients use the Cantera/RMG order ``a0..a6``:

    - ``Cp/R = a0 + a1 T + a2 T^2 + a3 T^3 + a4 T^4``
    - ``H/RT = a0 + a1 T/2 + a2 T^2/3 + a3 T^3/4 + a4 T^4/5 + a5/T``
    - ``S/R = a0 ln(T) + a1 T + a2 T^2/2 + a3 T^3/3 + a4 T^4/4 + a6``
    """

    min_temperature_K: float
    max_temperature_K: float
    coefficients: tuple[float, float, float, float, float, float, float]
    label: str = ""

    def __post_init__(self) -> None:
        _positive(self.min_temperature_K, "min_temperature_K")
        _positive(self.max_temperature_K, "max_temperature_K")
        if self.min_temperature_K >= self.max_temperature_K:
            raise ValueError("NASA7 segment min_temperature_K must be below max_temperature_K")
        if len(self.coefficients) != 7:
            raise ValueError("NASA7 coefficients must contain exactly seven values")
        for index, value in enumerate(self.coefficients):
            _finite(value, f"coefficients[{index}]")

    def contains_temperature(self, temperature_K: float) -> bool:
        return self.min_temperature_K <= temperature_K <= self.max_temperature_K

    def cp_over_r(self, temperature_K: float) -> float:
        self._validate_temperature(temperature_K)
        a0, a1, a2, a3, a4, _, _ = self.coefficients
        return (
            a0
            + a1 * temperature_K
            + a2 * temperature_K**2
            + a3 * temperature_K**3
            + a4 * temperature_K**4
        )

    def h_over_rt(self, temperature_K: float) -> float:
        self._validate_temperature(temperature_K)
        a0, a1, a2, a3, a4, a5, _ = self.coefficients
        return (
            a0
            + 0.5 * a1 * temperature_K
            + (a2 * temperature_K**2) / 3.0
            + 0.25 * a3 * temperature_K**3
            + 0.2 * a4 * temperature_K**4
            + a5 / temperature_K
        )

    def s_over_r(self, temperature_K: float) -> float:
        self._validate_temperature(temperature_K)
        a0, a1, a2, a3, a4, _, a6 = self.coefficients
        return (
            a0 * log(temperature_K)
            + a1 * temperature_K
            + 0.5 * a2 * temperature_K**2
            + (a3 * temperature_K**3) / 3.0
            + 0.25 * a4 * temperature_K**4
            + a6
        )

    def evaluate(self, temperature_K: float) -> SpeciesThermoState:
        cp_over_r = self.cp_over_r(temperature_K)
        h_over_rt = self.h_over_rt(temperature_K)
        s_over_r = self.s_over_r(temperature_K)
        enthalpy = h_over_rt * R_J_PER_MOL_K * temperature_K
        entropy = s_over_r * R_J_PER_MOL_K
        gibbs = enthalpy - temperature_K * entropy
        return SpeciesThermoState(
            species_id="",
            temperature_K=temperature_K,
            cp_over_R=cp_over_r,
            h_over_RT=h_over_rt,
            s_over_R=s_over_r,
            cp_J_mol_K=cp_over_r * R_J_PER_MOL_K,
            enthalpy_J_mol=enthalpy,
            entropy_J_mol_K=entropy,
            gibbs_J_mol=gibbs,
            segment_label=self.label,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_temperature_K": self.min_temperature_K,
            "max_temperature_K": self.max_temperature_K,
            "coefficients": list(self.coefficients),
            "label": self.label,
        }

    def _validate_temperature(self, temperature_K: float) -> None:
        _positive(temperature_K, "temperature_K")
        if not self.contains_temperature(temperature_K):
            raise ValueError(
                "temperature_K is outside the NASA7 segment validity range "
                f"[{self.min_temperature_K}, {self.max_temperature_K}]"
            )


@dataclass(frozen=True)
class SpeciesThermoState:
    """Species standard-state thermodynamic properties at one temperature."""

    species_id: str
    temperature_K: float
    cp_over_R: float
    h_over_RT: float
    s_over_R: float
    cp_J_mol_K: float
    enthalpy_J_mol: float
    entropy_J_mol_K: float
    gibbs_J_mol: float
    segment_label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "species_id": self.species_id,
            "temperature_K": self.temperature_K,
            "cp_over_R": self.cp_over_R,
            "h_over_RT": self.h_over_RT,
            "s_over_R": self.s_over_R,
            "cp_J_mol_K": self.cp_J_mol_K,
            "enthalpy_J_mol": self.enthalpy_J_mol,
            "entropy_J_mol_K": self.entropy_J_mol_K,
            "gibbs_J_mol": self.gibbs_J_mol,
            "segment_label": self.segment_label,
        }


@dataclass(frozen=True)
class NASA7SpeciesThermo:
    """Piecewise NASA7 species thermochemistry."""

    species_id: str
    segments: tuple[NASA7TemperatureSegment, ...]
    reference_pressure_Pa: float = 101_325.0
    composition: dict[str, float] = field(default_factory=dict)
    source_note: str = ""

    def __post_init__(self) -> None:
        if not self.species_id.strip():
            raise ValueError("species_id cannot be empty")
        if not self.segments:
            raise ValueError("NASA7SpeciesThermo requires at least one segment")
        _positive(self.reference_pressure_Pa, "reference_pressure_Pa")
        ordered = tuple(sorted(self.segments, key=lambda segment: segment.min_temperature_K))
        for left, right in pairwise(ordered):
            if left.max_temperature_K > right.min_temperature_K:
                raise ValueError("NASA7 segments must not overlap")
        object.__setattr__(self, "segments", ordered)
        object.__setattr__(
            self,
            "composition",
            {key: float(value) for key, value in self.composition.items()},
        )

    @classmethod
    def from_cantera_yaml_thermo(
        cls,
        *,
        species_id: str,
        thermo: Mapping[str, Any],
        composition: Mapping[str, float] | None = None,
        reference_pressure_Pa: float = 101_325.0,
    ) -> NASA7SpeciesThermo:
        """Build a species thermo object from a Cantera-style NASA7 YAML node."""

        if thermo.get("model") != "NASA7":
            raise ValueError("only Cantera-style thermo model NASA7 is supported")
        ranges = [float(value) for value in thermo.get("temperature-ranges", ())]
        data = thermo.get("data", ())
        if len(ranges) != len(data) + 1:
            raise ValueError("NASA7 temperature-ranges length must equal len(data) + 1")
        segments = []
        for index, coefficients in enumerate(data):
            if not isinstance(coefficients, Sequence):
                raise ValueError("NASA7 data rows must be coefficient sequences")
            segments.append(
                NASA7TemperatureSegment(
                    min_temperature_K=ranges[index],
                    max_temperature_K=ranges[index + 1],
                    coefficients=_coefficient_tuple(coefficients),
                    label=f"{species_id}:segment-{index}",
                )
            )
        return cls(
            species_id=species_id,
            segments=tuple(segments),
            reference_pressure_Pa=reference_pressure_Pa,
            composition={} if composition is None else dict(composition),
            source_note=str(thermo.get("note", "")),
        )

    def select_segment(self, temperature_K: float) -> NASA7TemperatureSegment:
        _positive(temperature_K, "temperature_K")
        for segment in self.segments:
            if segment.contains_temperature(temperature_K):
                return segment
        raise ValueError(f"no NASA7 segment for {self.species_id!r} at {temperature_K:g} K")

    def evaluate(self, temperature_K: float) -> SpeciesThermoState:
        state = self.select_segment(temperature_K).evaluate(temperature_K)
        return SpeciesThermoState(
            species_id=self.species_id,
            temperature_K=state.temperature_K,
            cp_over_R=state.cp_over_R,
            h_over_RT=state.h_over_RT,
            s_over_R=state.s_over_R,
            cp_J_mol_K=state.cp_J_mol_K,
            enthalpy_J_mol=state.enthalpy_J_mol,
            entropy_J_mol_K=state.entropy_J_mol_K,
            gibbs_J_mol=state.gibbs_J_mol,
            segment_label=state.segment_label,
        )

    def continuity_report(self, *, relative_tolerance: float = 1.0e-2) -> dict[str, Any]:
        """Report Cp/H/S discontinuities at adjacent segment boundaries."""

        _positive(relative_tolerance, "relative_tolerance")
        checks: list[dict[str, Any]] = []
        for left, right in pairwise(self.segments):
            if abs(left.max_temperature_K - right.min_temperature_K) > 1.0e-9:
                checks.append(
                    {
                        "temperature_K": None,
                        "passed": False,
                        "reason": "temperature_gap",
                        "left_max_temperature_K": left.max_temperature_K,
                        "right_min_temperature_K": right.min_temperature_K,
                    }
                )
                continue
            temperature = left.max_temperature_K
            left_state = left.evaluate(temperature)
            right_state = right.evaluate(temperature)
            for key in ("cp_over_R", "h_over_RT", "s_over_R"):
                left_value = float(getattr(left_state, key))
                right_value = float(getattr(right_state, key))
                scale = max(abs(left_value), abs(right_value), 1.0e-12)
                rel_jump = abs(left_value - right_value) / scale
                checks.append(
                    {
                        "temperature_K": temperature,
                        "quantity": key,
                        "left": left_value,
                        "right": right_value,
                        "relative_jump": rel_jump,
                        "passed": rel_jump <= relative_tolerance,
                    }
                )
        return {
            "species_id": self.species_id,
            "relative_tolerance": relative_tolerance,
            "checks": checks,
            "passed": all(bool(check.get("passed", False)) for check in checks),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "species_id": self.species_id,
            "model": "NASA7",
            "reference_pressure_Pa": self.reference_pressure_Pa,
            "composition": dict(self.composition),
            "segments": [segment.to_dict() for segment in self.segments],
            "source_note": self.source_note,
        }


@dataclass(frozen=True)
class ReactionThermoResult:
    """Reaction thermochemistry evaluated from species standard states."""

    reaction_id: str
    temperature_K: float
    stoichiometry: dict[str, float]
    delta_h_J_mol: float
    delta_s_J_mol_K: float
    delta_g_J_mol: float
    equilibrium_constant: float
    species_states: dict[str, SpeciesThermoState]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "temperature_K": self.temperature_K,
            "stoichiometry": dict(self.stoichiometry),
            "delta_h_J_mol": self.delta_h_J_mol,
            "delta_s_J_mol_K": self.delta_s_J_mol_K,
            "delta_g_J_mol": self.delta_g_J_mol,
            "equilibrium_constant": self.equilibrium_constant,
            "species_states": {
                species_id: state.to_dict() for species_id, state in self.species_states.items()
            },
        }


@dataclass(frozen=True)
class CorrelationSpeciesThermo:
    """Existing Cp/H/S correlations and an explicit datum in the reaction API.

    NASA7 and source-derived caloric curves produce the same SpeciesThermoState.
    This adapter owns neither a second thermal inventory nor a reaction solver.
    """

    species_id: str
    heat_capacity: Any
    reference: CaloricReference
    composition: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if (not self.species_id or self.reference.phase is None
                or self.reference.energy_basis != "enthalpy"
                or self.reference.entropy_J_mol_K is None):
            raise ValueError("species thermochemistry requires explicit phase, H datum and S datum")
        self.evaluate(self.reference.temperature_K)

    @property
    def reference_pressure_Pa(self):  # noqa: N802 -- existing species thermochemistry protocol
        return self.reference.pressure_Pa

    def evaluate(self, temperature_K):
        from chemworld.physchem.property_packages import evaluate_caloric_correlation

        values = evaluate_caloric_correlation(self.heat_capacity, reference=self.reference,
            temperature_K=temperature_K,
            quantities=("cp", "enthalpy", "entropy", "chemical_potential"))
        cp, enthalpy, entropy, gibbs = (values[key].value
            for key in ("cp", "enthalpy", "entropy", "chemical_potential"))
        return SpeciesThermoState(self.species_id, temperature_K, cp/R_J_PER_MOL_K,
            enthalpy/(R_J_PER_MOL_K*temperature_K), entropy/R_J_PER_MOL_K,
            cp, enthalpy, entropy, gibbs, self.heat_capacity.correlation_id)

    def to_dict(self):
        return {"model": "correlation-species-thermo-v1", "species_id": self.species_id,
                "heat_capacity": self.heat_capacity.to_dict(),
                "reference": self.reference.to_dict(),
                "composition": dict(self.composition)}

    @classmethod
    def from_dict(cls, payload):
        from chemworld.physchem.heat_capacity import heat_capacity_from_dict

        if payload["model"] != "correlation-species-thermo-v1":
            raise ValueError("unsupported species thermochemistry model")
        return cls(payload["species_id"], heat_capacity_from_dict(payload["heat_capacity"]),
                   CaloricReference(**payload["reference"]), dict(payload["composition"]))


def formation_entropy(*, formation_enthalpy_J_mol, formation_gibbs_J_mol, temperature_K=298.15):
    """(Hf-Gf)/T: formation entropy, distinct from absolute standard entropy."""
    _positive(temperature_K, "temperature_K")
    _finite(formation_enthalpy_J_mol, "formation_enthalpy_J_mol")
    _finite(formation_gibbs_J_mol, "formation_gibbs_J_mol")
    return (formation_enthalpy_J_mol-formation_gibbs_J_mol)/temperature_K


def formation_gibbs_energy(reference, elemental_references, elemental_coefficients):
    """Gibbs energy of one mole formed from explicit elemental reference species.

    All H/S data must share temperature, pressure and enthalpy basis. Nonzero
    elemental reference enthalpies remain explicit; absolute entropy is not Sf.
    Elemental coefficients can come from standard_formation_reaction, divided
    by its compound coefficient when integer stoichiometry was requested.
    """
    references, coefficients = tuple(elemental_references), tuple(elemental_coefficients)
    if not references or len(references) != len(coefficients):
        raise ValueError("elemental references and coefficients must align")
    if any(not isfinite(v) or v <= 0 or isinstance(v, bool) for v in coefficients):
        raise ValueError("elemental coefficients must be finite and positive")
    for item in (reference, *references):
        if (item.energy_basis != "enthalpy" or item.entropy_J_mol_K is None
                or item.phase is None or item.temperature_K != reference.temperature_K
                or item.pressure_Pa != reference.pressure_Pa):
            raise ValueError("formation references require matching T/P and explicit phase/H/S")
    delta_h = reference.energy_J_mol-fsum(c*r.energy_J_mol
                                        for c, r in zip(coefficients, references, strict=True))
    delta_s = reference.entropy_J_mol_K-fsum(c*r.entropy_J_mol_K
                                           for c, r in zip(coefficients, references, strict=True))
    return delta_h-reference.temperature_K*delta_s


def formation_enthalpy_basis(*, vaporization_enthalpy_J_mol,
                             liquid_formation_enthalpy_J_mol=None,
                             gas_formation_enthalpy_J_mol=None, temperature_K=298.15):
    """Convert liquid/gas Hf using Hvap at the SAME explicitly declared temperature."""
    _positive(temperature_K, "temperature_K")
    _finite(vaporization_enthalpy_J_mol, "vaporization_enthalpy_J_mol")
    if vaporization_enthalpy_J_mol < 0:
        raise ValueError("vaporization enthalpy must be nonnegative")
    if (liquid_formation_enthalpy_J_mol is None) == (gas_formation_enthalpy_J_mol is None):
        raise ValueError("supply exactly one liquid or gas formation enthalpy")
    if liquid_formation_enthalpy_J_mol is not None:
        _finite(liquid_formation_enthalpy_J_mol, "liquid_formation_enthalpy_J_mol")
        return liquid_formation_enthalpy_J_mol+vaporization_enthalpy_J_mol
    _finite(gas_formation_enthalpy_J_mol, "gas_formation_enthalpy_J_mol")
    return gas_formation_enthalpy_J_mol-vaporization_enthalpy_J_mol


def formation_reference_phase_change(reference, *, component_id, target_phase,
                                     heat_capacity_correlations, transitions):
    """Convert the phase of Hf at the same T using the existing Cp/latent-heat path.

    Entropy of a new phase is not inferred from enthalpy. Its datum is left absent
    until supplied; returning the same phase preserves the original full datum.
    """
    from chemworld.physchem.enthalpy import phase_path_enthalpy_report

    if reference.energy_basis != "enthalpy" or reference.phase is None:
        raise ValueError("phase conversion requires a declared enthalpy/phase reference")
    if target_phase == reference.phase:
        return reference
    path = phase_path_enthalpy_report(component_id=component_id,
        heat_capacity_correlations=heat_capacity_correlations, transitions=transitions,
        initial_phase=reference.phase, final_phase=target_phase,
        initial_temperature_K=reference.temperature_K, final_temperature_K=reference.temperature_K,
        reference_temperature_K=reference.temperature_K, validity_policy="raise")
    return replace(reference, phase=target_phase,
        energy_J_mol=reference.energy_J_mol+path.total_enthalpy_J_mol, entropy_J_mol_K=None,
        source_note=reference.source_note+f"; same-T phase path {path.transition_ids}")


def reaction_thermochemistry(
    *,
    reaction_id: str,
    stoichiometry: Mapping[str, float],
    species_thermo: Mapping[str, NASA7SpeciesThermo | CorrelationSpeciesThermo],
    temperature_K: float,
) -> ReactionThermoResult:
    """Evaluate reaction Delta H, Delta S, Delta G, and K from species thermo."""

    if not reaction_id.strip():
        raise ValueError("reaction_id cannot be empty")
    _positive(temperature_K, "temperature_K")
    if not stoichiometry:
        raise ValueError("stoichiometry cannot be empty")
    pressures = {species_thermo[key].reference_pressure_Pa
                 for key in stoichiometry if key in species_thermo}
    if len(pressures) > 1:
        raise ValueError("reaction standard states require the same reference pressure")
    species_states: dict[str, SpeciesThermoState] = {}
    delta_h = 0.0
    delta_s = 0.0
    delta_g = 0.0
    for species_id, coefficient in stoichiometry.items():
        _finite(float(coefficient), f"stoichiometry[{species_id!r}]")
        if coefficient == 0.0:
            raise ValueError("zero stoichiometric coefficients are not stored")
        if species_id not in species_thermo:
            raise KeyError(f"missing species thermochemistry for species {species_id!r}")
        state = species_thermo[species_id].evaluate(temperature_K)
        species_states[species_id] = state
        delta_h += coefficient * state.enthalpy_J_mol
        delta_s += coefficient * state.entropy_J_mol_K
        delta_g += coefficient * state.gibbs_J_mol
    return ReactionThermoResult(
        reaction_id=reaction_id,
        temperature_K=temperature_K,
        stoichiometry={key: float(value) for key, value in stoichiometry.items()},
        delta_h_J_mol=delta_h,
        delta_s_J_mol_K=delta_s,
        delta_g_J_mol=delta_g,
        equilibrium_constant=equilibrium_constant_from_delta_g(
            delta_g_J_mol=delta_g,
            temperature_K=temperature_K,
        ),
        species_states=species_states,
    )


def equilibrium_constant_from_delta_g(*, delta_g_J_mol: float, temperature_K: float) -> float:
    """Return ``K = exp(-Delta G / RT)`` with exponent clipping."""

    _finite(delta_g_J_mol, "delta_g_J_mol")
    _positive(temperature_K, "temperature_K")
    exponent = -delta_g_J_mol / (R_J_PER_MOL_K * temperature_K)
    return exp(max(-_EXPONENT_LIMIT, min(_EXPONENT_LIMIT, exponent)))


def _coefficient_tuple(
    values: Sequence[Any],
) -> tuple[float, float, float, float, float, float, float]:
    if len(values) != 7:
        raise ValueError("NASA7 coefficient rows must contain exactly seven values")
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
        float(values[6]),
    )


def _finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _positive(value: float, name: str) -> None:
    _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


__all__ = [
    "CorrelationSpeciesThermo",
    "NASA7SpeciesThermo",
    "NASA7TemperatureSegment",
    "ReactionThermoResult",
    "SpeciesThermoState",
    "equilibrium_constant_from_delta_g",
    "formation_enthalpy_basis",
    "formation_entropy",
    "formation_gibbs_energy",
    "formation_reference_phase_change",
    "reaction_thermochemistry",
    "thermochemistry_model_cards",
]
