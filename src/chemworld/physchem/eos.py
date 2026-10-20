"""Equation-of-state utilities for ChemWorld.

This module implements a compact local EOS core for benchmark use. It supports
ideal-gas states plus Peng-Robinson and Soave-Redlich-Kwong cubic equations of
state with classical one-fluid mixing rules and fugacity coefficients.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import exp, isfinite, log, sqrt
from typing import Literal

import numpy as np

from chemworld.physchem.eos_cards import eos_model_cards

R_J_PER_MOL_K = 8.31446261815324
CubicModel = Literal["peng_robinson", "srk"]
PhaseRoot = Literal["vapor", "liquid", "stable"]


@dataclass(frozen=True)
class EOSComponentSpec:
    component_id: str
    critical_temperature_K: float
    critical_pressure_Pa: float
    acentric_factor: float = 0.0

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component_id cannot be empty")
        if not isfinite(self.critical_temperature_K) or self.critical_temperature_K <= 0:
            raise ValueError("critical_temperature_K must be positive")
        if not isfinite(self.critical_pressure_Pa) or self.critical_pressure_Pa <= 0:
            raise ValueError("critical_pressure_Pa must be positive")
        if not isfinite(self.acentric_factor):
            raise ValueError("acentric_factor must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "critical_temperature_K": self.critical_temperature_K,
            "critical_pressure_Pa": self.critical_pressure_Pa,
            "acentric_factor": self.acentric_factor,
        }


@dataclass(frozen=True)
class BinaryInteractionProvenance:
    component_i: str
    component_j: str
    k_ij: float
    source_id: str
    source_note: str = ""
    temperature_range_K: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.component_i or not self.component_j:
            raise ValueError("binary interaction component ids cannot be empty")
        if self.component_i == self.component_j:
            raise ValueError("binary interaction provenance requires two distinct components")
        if not isfinite(self.k_ij):
            raise ValueError("binary interaction k_ij must be finite")
        if not self.source_id:
            raise ValueError("binary interaction provenance requires source_id")
        if self.temperature_range_K is not None:
            low, high = self.temperature_range_K
            if low <= 0.0 or high <= low:
                raise ValueError("temperature_range_K must be positive and increasing")

    @property
    def pair_key(self) -> str:
        return _canonical_pair_key(self.component_i, self.component_j)

    def to_dict(self) -> dict[str, object]:
        return {
            "component_i": self.component_i,
            "component_j": self.component_j,
            "pair_key": self.pair_key,
            "k_ij": self.k_ij,
            "source_id": self.source_id,
            "source_note": self.source_note,
            "temperature_range_K": list(self.temperature_range_K)
            if self.temperature_range_K is not None
            else None,
        }


@dataclass(frozen=True)
class VolumeTranslationSpec:
    component_id: str
    shift_m3_mol: float
    source_id: str
    source_note: str = ""
    temperature_range_K: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("volume translation component_id cannot be empty")
        if not isfinite(self.shift_m3_mol):
            raise ValueError("volume translation shift_m3_mol must be finite")
        if not self.source_id:
            raise ValueError("volume translation requires source_id")
        if self.temperature_range_K is not None:
            low, high = self.temperature_range_K
            if low <= 0.0 or high <= low:
                raise ValueError("temperature_range_K must be positive and increasing")

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "shift_m3_mol": self.shift_m3_mol,
            "source_id": self.source_id,
            "source_note": self.source_note,
            "temperature_range_K": list(self.temperature_range_K)
            if self.temperature_range_K is not None
            else None,
        }


@dataclass(frozen=True)
class CubicEOSSpec:
    eos_id: str
    model: CubicModel
    components: tuple[EOSComponentSpec, ...]
    binary_interaction: dict[str, float] = field(default_factory=dict)
    binary_interaction_provenance: tuple[BinaryInteractionProvenance, ...] = ()
    volume_translation: tuple[VolumeTranslationSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.model not in {"peng_robinson", "srk"}:
            raise ValueError(f"Unsupported cubic EOS model: {self.model}")
        if not self.components:
            raise ValueError("Cubic EOS requires at least one component")
        ids = [component.component_id for component in self.components]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate EOS component ids are not allowed")
        if any(not isfinite(value) for value in self.binary_interaction.values()):
            raise ValueError("binary interaction values must be finite")
        component_ids = set(ids)
        for key in self.binary_interaction:
            left, right = _parse_pair_key(key)
            if left not in component_ids or right not in component_ids:
                raise ValueError(f"binary interaction pair is not in EOS components: {key}")
            if left == right:
                raise ValueError("binary interaction pair must contain distinct components")
        seen_provenance: set[str] = set()
        for record in self.binary_interaction_provenance:
            if record.component_i not in component_ids or record.component_j not in component_ids:
                raise ValueError("binary interaction provenance references unknown component")
            if record.pair_key in seen_provenance:
                raise ValueError(f"duplicate binary interaction provenance for {record.pair_key}")
            seen_provenance.add(record.pair_key)
        seen_translation: set[str] = set()
        for translation in self.volume_translation:
            if translation.component_id not in component_ids:
                raise ValueError("volume translation references unknown component")
            if translation.component_id in seen_translation:
                raise ValueError(f"duplicate volume translation for {translation.component_id}")
            seen_translation.add(translation.component_id)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(component.component_id for component in self.components)

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        values["components"] = tuple(EOSComponentSpec(**v) for v in values["components"])
        values["binary_interaction_provenance"] = tuple(
            BinaryInteractionProvenance(**{k: v for k, v in record.items() if k != "pair_key"})
            for record in values.get("binary_interaction_provenance", ()))
        values["volume_translation"] = tuple(VolumeTranslationSpec(**v)
                                              for v in values.get("volume_translation", ()))
        return cls(**values)

    def to_dict(self) -> dict[str, object]:
        return {
            "eos_id": self.eos_id,
            "model": self.model,
            "components": [component.to_dict() for component in self.components],
            "binary_interaction": dict(self.binary_interaction),
            "binary_interaction_provenance": [
                record.to_dict() for record in self.binary_interaction_provenance
            ],
            "volume_translation": [
                translation.to_dict() for translation in self.volume_translation
            ],
        }


@dataclass(frozen=True)
class CubicPureParameters:
    component_id: str
    a_c: float
    a_alpha: float
    b: float
    alpha: float
    kappa: float
    da_alpha_dT: float

    def to_dict(self) -> dict[str, object]:
        return {
            "component_id": self.component_id,
            "a_c": self.a_c,
            "a_alpha": self.a_alpha,
            "b": self.b,
            "alpha": self.alpha,
            "kappa": self.kappa,
            "da_alpha_dT": self.da_alpha_dT,
        }


@dataclass(frozen=True)
class EOSMixtureParameters:
    a_mix: float
    da_mix_dT: float
    b_mix: float
    a_matrix: tuple[tuple[float, ...], ...]
    da_matrix_dT: tuple[tuple[float, ...], ...]
    b_values: tuple[float, ...]
    pure_parameters: tuple[CubicPureParameters, ...]
    binary_interaction_provenance: tuple[BinaryInteractionProvenance, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "a_mix": self.a_mix,
            "da_mix_dT": self.da_mix_dT,
            "b_mix": self.b_mix,
            "a_matrix": [list(row) for row in self.a_matrix],
            "da_matrix_dT": [list(row) for row in self.da_matrix_dT],
            "b_values": list(self.b_values),
            "pure_parameters": [params.to_dict() for params in self.pure_parameters],
            "binary_interaction_provenance": [
                record.to_dict() for record in self.binary_interaction_provenance
            ],
        }


@dataclass(frozen=True)
class RootCandidateDiagnostic:
    z_factor: float
    phase_hint: str
    molar_volume_m3_mol: float
    translated_molar_volume_m3_mol: float | None
    translation_shift_m3_mol: float
    molar_residual_gibbs_J_mol: float
    selection_score: float
    rank: int
    selected: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "z_factor": self.z_factor,
            "phase_hint": self.phase_hint,
            "molar_volume_m3_mol": self.molar_volume_m3_mol,
            "translated_molar_volume_m3_mol": self.translated_molar_volume_m3_mol,
            "translation_shift_m3_mol": self.translation_shift_m3_mol,
            "molar_residual_gibbs_J_mol": self.molar_residual_gibbs_J_mol,
            "selection_score": self.selection_score,
            "rank": self.rank,
            "selected": self.selected,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RootGovernanceReport:
    eos_id: str
    model: str
    requested_phase: str
    root_selection_policy: str
    score_basis: str
    selected_z_factor: float
    selected_phase_hint: str
    roots: tuple[float, ...]
    candidates: tuple[RootCandidateDiagnostic, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "eos_id": self.eos_id,
            "model": self.model,
            "requested_phase": self.requested_phase,
            "root_selection_policy": self.root_selection_policy,
            "score_basis": self.score_basis,
            "selected_z_factor": self.selected_z_factor,
            "selected_phase_hint": self.selected_phase_hint,
            "roots": list(self.roots),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class VolumeTranslationReport:
    eos_id: str
    model: str
    phase: str
    temperature_K: float
    pressure_Pa: float
    composition: dict[str, float]
    base_compressibility_factor: float
    translated_compressibility_factor: float
    base_molar_volume_m3_mol: float
    translated_molar_volume_m3_mol: float
    mixture_shift_m3_mol: float
    component_shifts_m3_mol: dict[str, float]
    shift_contributions_m3_mol: dict[str, float]
    translated_roots: tuple[float, ...]
    root_governance: RootGovernanceReport
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "eos_id": self.eos_id,
            "model": self.model,
            "phase": self.phase,
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "composition": dict(self.composition),
            "base_compressibility_factor": self.base_compressibility_factor,
            "translated_compressibility_factor": self.translated_compressibility_factor,
            "base_molar_volume_m3_mol": self.base_molar_volume_m3_mol,
            "translated_molar_volume_m3_mol": self.translated_molar_volume_m3_mol,
            "mixture_shift_m3_mol": self.mixture_shift_m3_mol,
            "component_shifts_m3_mol": dict(self.component_shifts_m3_mol),
            "shift_contributions_m3_mol": dict(self.shift_contributions_m3_mol),
            "translated_roots": list(self.translated_roots),
            "root_governance": self.root_governance.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CubicResidualProperties:
    model: CubicModel
    z_factor: float
    root_selection_policy: str
    molar_residual_enthalpy_J_mol: float
    molar_residual_entropy_J_mol_K: float
    molar_residual_gibbs_J_mol: float
    departure_log_argument: float
    da_mix_dT: float

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "z_factor": self.z_factor,
            "root_selection_policy": self.root_selection_policy,
            "molar_residual_enthalpy_J_mol": self.molar_residual_enthalpy_J_mol,
            "molar_residual_entropy_J_mol_K": self.molar_residual_entropy_J_mol_K,
            "molar_residual_gibbs_J_mol": self.molar_residual_gibbs_J_mol,
            "departure_log_argument": self.departure_log_argument,
            "da_mix_dT": self.da_mix_dT,
        }


@dataclass(frozen=True)
class EOSState:
    eos_id: str
    model: str
    phase: str
    root_selection_policy: str
    temperature_K: float
    pressure_Pa: float
    composition: dict[str, float]
    compressibility_factor: float
    molar_volume_m3_mol: float
    fugacity_coefficients: dict[str, float]
    molar_residual_enthalpy_J_mol: float
    molar_residual_entropy_J_mol_K: float
    molar_residual_gibbs_J_mol: float
    roots: tuple[float, ...]
    mixture_parameters: dict[str, object] = field(default_factory=dict)
    residual_properties: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "eos_id": self.eos_id,
            "model": self.model,
            "phase": self.phase,
            "root_selection_policy": self.root_selection_policy,
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "composition": dict(self.composition),
            "compressibility_factor": self.compressibility_factor,
            "molar_volume_m3_mol": self.molar_volume_m3_mol,
            "fugacity_coefficients": dict(self.fugacity_coefficients),
            "molar_residual_enthalpy_J_mol": self.molar_residual_enthalpy_J_mol,
            "molar_residual_entropy_J_mol_K": self.molar_residual_entropy_J_mol_K,
            "molar_residual_gibbs_J_mol": self.molar_residual_gibbs_J_mol,
            "roots": list(self.roots),
            "mixture_parameters": dict(self.mixture_parameters),
            "residual_properties": dict(self.residual_properties),
        }


def ideal_gas_molar_volume(temperature_K: float, pressure_Pa: float) -> float:
    _validate_tp(temperature_K, pressure_Pa)
    return R_J_PER_MOL_K * temperature_K / pressure_Pa


def ideal_gas_pressure(
    *,
    amount_mol: float,
    volume_m3: float,
    temperature_K: float,
) -> float:
    if amount_mol < 0:
        raise ValueError("amount_mol cannot be negative")
    if volume_m3 <= 0:
        raise ValueError("volume_m3 must be positive")
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    return amount_mol * R_J_PER_MOL_K * temperature_K / volume_m3


def ideal_gas_state(
    composition: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
) -> EOSState:
    normalized = _normalize_composition(composition)
    return EOSState(
        eos_id="ideal_gas",
        model="ideal_gas",
        phase="vapor",
        root_selection_policy="ideal_single_root",
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        composition=normalized,
        compressibility_factor=1.0,
        molar_volume_m3_mol=ideal_gas_molar_volume(temperature_K, pressure_Pa),
        fugacity_coefficients=dict.fromkeys(normalized, 1.0),
        molar_residual_enthalpy_J_mol=0.0,
        molar_residual_entropy_J_mol_K=0.0,
        molar_residual_gibbs_J_mol=0.0,
        roots=(1.0,),
        residual_properties=CubicResidualProperties(
            model="peng_robinson",
            z_factor=1.0,
            root_selection_policy="ideal_single_root",
            molar_residual_enthalpy_J_mol=0.0,
            molar_residual_entropy_J_mol_K=0.0,
            molar_residual_gibbs_J_mol=0.0,
            departure_log_argument=1.0,
            da_mix_dT=0.0,
        ).to_dict(),
    )


def cubic_pure_parameters(
    component: EOSComponentSpec,
    *,
    model: CubicModel,
    temperature_K: float,
) -> CubicPureParameters:
    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    tr_sqrt = sqrt(temperature_K / component.critical_temperature_K)
    omega = component.acentric_factor
    if model == "peng_robinson":
        omega_a = 0.45724
        omega_b = 0.07780
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    elif model == "srk":
        omega_a = 0.42747
        omega_b = 0.08664
        kappa = 0.480 + 1.574 * omega - 0.176 * omega**2
    else:
        raise ValueError(f"Unsupported cubic EOS model: {model}")
    alpha_base = 1.0 + kappa * (1.0 - tr_sqrt)
    alpha = alpha_base**2
    d_alpha_dT = -kappa * alpha_base / (
        component.critical_temperature_K * max(tr_sqrt, 1e-300)
    )
    a = (
        omega_a
        * R_J_PER_MOL_K**2
        * component.critical_temperature_K**2
        / component.critical_pressure_Pa
    )
    b = omega_b * R_J_PER_MOL_K * component.critical_temperature_K
    b /= component.critical_pressure_Pa
    return CubicPureParameters(
        component_id=component.component_id,
        a_c=a,
        a_alpha=a * alpha,
        b=b,
        alpha=alpha,
        kappa=kappa,
        da_alpha_dT=a * d_alpha_dT,
    )


def cubic_mixture_parameters(
    spec: CubicEOSSpec,
    composition: Mapping[str, float],
    *,
    temperature_K: float,
) -> EOSMixtureParameters:
    x = _composition_vector(spec, composition)
    pure = tuple(
        cubic_pure_parameters(component, model=spec.model, temperature_K=temperature_K)
        for component in spec.components
    )
    n = len(spec.components)
    a_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    da_matrix_dT = [[0.0 for _ in range(n)] for _ in range(n)]
    for i, left in enumerate(spec.components):
        for j, right in enumerate(spec.components):
            kij = _binary_interaction(spec, left.component_id, right.component_id)
            a_matrix[i][j] = sqrt(pure[i].a_alpha * pure[j].a_alpha) * (1.0 - kij)
            da_matrix_dT[i][j] = _cross_a_temperature_derivative(
                pure[i],
                pure[j],
                kij=kij,
            )
    a_mix = sum(x[i] * x[j] * a_matrix[i][j] for i in range(n) for j in range(n))
    da_mix_dT = sum(
        x[i] * x[j] * da_matrix_dT[i][j] for i in range(n) for j in range(n)
    )
    b_values = tuple(params.b for params in pure)
    b_mix = sum(x[i] * b_values[i] for i in range(n))
    if a_mix < 0 or b_mix <= 0:
        raise ValueError("Invalid cubic mixture parameters")
    return EOSMixtureParameters(
        a_mix=a_mix,
        da_mix_dT=da_mix_dT,
        b_mix=b_mix,
        a_matrix=tuple(tuple(row) for row in a_matrix),
        da_matrix_dT=tuple(tuple(row) for row in da_matrix_dT),
        b_values=b_values,
        pure_parameters=pure,
        binary_interaction_provenance=spec.binary_interaction_provenance,
    )


def cubic_ab(
    mixture: EOSMixtureParameters,
    *,
    temperature_K: float,
    pressure_Pa: float,
) -> tuple[float, float]:
    _validate_tp(temperature_K, pressure_Pa)
    a_reduced = mixture.a_mix * pressure_Pa / (R_J_PER_MOL_K**2 * temperature_K**2)
    b_reduced = mixture.b_mix * pressure_Pa / (R_J_PER_MOL_K * temperature_K)
    return a_reduced, b_reduced


def cubic_compressibility_roots(
    spec: CubicEOSSpec,
    composition: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
) -> tuple[float, ...]:
    mixture = cubic_mixture_parameters(spec, composition, temperature_K=temperature_K)
    a_reduced, b_reduced = cubic_ab(
        mixture,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
    )
    if spec.model == "peng_robinson":
        coefficients = (
            1.0,
            -(1.0 - b_reduced),
            a_reduced - 3.0 * b_reduced**2 - 2.0 * b_reduced,
            -(a_reduced * b_reduced - b_reduced**2 - b_reduced**3),
        )
    else:
        coefficients = (
            1.0,
            -1.0,
            a_reduced - b_reduced - b_reduced**2,
            -a_reduced * b_reduced,
        )
    roots = np.roots(coefficients)
    real_roots = sorted(
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-8 and root.real > b_reduced + 1e-10
    )
    if not real_roots:
        raise ValueError("Cubic EOS produced no physically admissible Z roots")
    return tuple(real_roots)


def translated_cubic_compressibility_roots(
    spec: CubicEOSSpec,
    composition: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
    volume_translation: tuple[VolumeTranslationSpec, ...] | None = None,
    strict_translation_components: bool = False,
) -> tuple[float, ...]:
    """Return Peneloux-style translated compressibility roots.

    The returned roots correspond to the shifted physical molar volume
    ``V_translated = V_eos - c_mix`` and therefore to
    ``Z_translated = P V_translated / (R T)``. Fugacity and residual-property
    calculations remain the responsibility of the explicit EOS state/reporting
    functions so benchmark traces can see which convention was used.
    """

    _validate_tp(temperature_K, pressure_Pa)
    normalized = _normalize_composition(composition)
    mixture = cubic_mixture_parameters(spec, normalized, temperature_K=temperature_K)
    a_reduced, b_reduced = cubic_ab(
        mixture,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
    )
    mixture_shift, _, _ = mixture_volume_translation(
        spec,
        normalized,
        volume_translation=volume_translation,
        strict_components=strict_translation_components,
    )
    c_reduced = mixture_shift * pressure_Pa / (R_J_PER_MOL_K * temperature_K)
    c1, c2 = _cubic_shape_constants(spec.model)
    coefficients = (
        1.0,
        (c1 + c2 - 1.0) * b_reduced - 1.0 + 3.0 * c_reduced,
        c1 * c2 * b_reduced**2
        - (c1 + c2) * (b_reduced**2 + b_reduced)
        + a_reduced
        + 3.0 * c_reduced**2
        + 2.0 * c_reduced * (-1.0 + b_reduced * (-1.0 + c1 + c2)),
        a_reduced * (-b_reduced + c_reduced)
        + (-1.0 - b_reduced + c_reduced)
        * (c_reduced + c1 * b_reduced)
        * (c_reduced + c2 * b_reduced),
    )
    roots = np.roots(coefficients)
    real_roots = sorted(
        float(root.real)
        for root in roots
        if abs(root.imag) < 1e-8 and root.real > b_reduced - c_reduced + 1e-10
    )
    if not real_roots:
        raise ValueError("Translated cubic EOS produced no physically admissible Z roots")
    return tuple(real_roots)


def evaluate_cubic_eos(
    spec: CubicEOSSpec,
    composition: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
    phase: PhaseRoot = "vapor",
) -> EOSState:
    _validate_tp(temperature_K, pressure_Pa)
    normalized = _normalize_composition(composition)
    mixture = cubic_mixture_parameters(spec, normalized, temperature_K=temperature_K)
    roots = cubic_compressibility_roots(
        spec,
        normalized,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
    )
    root_selection_policy = cubic_root_selection_policy(phase=phase, roots=roots)
    z_factor = select_cubic_root(
        spec,
        normalized,
        roots=roots,
        mixture=mixture,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        phase=phase,
    )
    fugacity = cubic_fugacity_coefficients(
        spec,
        normalized,
        z_factor=z_factor,
        mixture=mixture,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
    )
    residuals = cubic_residual_properties(
        spec,
        normalized,
        z_factor=z_factor,
        mixture=mixture,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        root_selection_policy=root_selection_policy,
        fugacity_coefficients=fugacity,
    )
    return EOSState(
        eos_id=spec.eos_id,
        model=spec.model,
        phase=phase,
        root_selection_policy=root_selection_policy,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        composition=normalized,
        compressibility_factor=z_factor,
        molar_volume_m3_mol=z_factor * ideal_gas_molar_volume(temperature_K, pressure_Pa),
        fugacity_coefficients=fugacity,
        molar_residual_enthalpy_J_mol=residuals.molar_residual_enthalpy_J_mol,
        molar_residual_entropy_J_mol_K=residuals.molar_residual_entropy_J_mol_K,
        molar_residual_gibbs_J_mol=residuals.molar_residual_gibbs_J_mol,
        roots=roots,
        mixture_parameters=mixture.to_dict(),
        residual_properties=residuals.to_dict(),
    )


def select_cubic_root(
    spec: CubicEOSSpec,
    composition: Mapping[str, float],
    *,
    roots: tuple[float, ...],
    mixture: EOSMixtureParameters,
    temperature_K: float,
    pressure_Pa: float,
    phase: PhaseRoot,
) -> float:
    if phase == "liquid":
        return min(roots)
    if phase == "vapor":
        return max(roots)
    if phase != "stable":
        raise ValueError(f"Unsupported phase root selector: {phase}")
    normalized = _normalize_composition(composition)
    x = _composition_vector(spec, normalized)
    candidates = []
    for root in roots:
        fugacity = cubic_fugacity_coefficients(
            spec,
            normalized,
            z_factor=root,
            mixture=mixture,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
        )
        residual_g = sum(
            x[index] * log(max(fugacity[component_id], 1e-300))
            for index, component_id in enumerate(spec.component_ids)
        )
        candidates.append((residual_g, root))
    return min(candidates)[1]


def cubic_root_selection_policy(
    *,
    phase: PhaseRoot,
    roots: tuple[float, ...],
) -> str:
    if not roots:
        raise ValueError("roots cannot be empty")
    if len(roots) == 1:
        return "single_admissible_root"
    if phase == "liquid":
        return "smallest_z_liquid"
    if phase == "vapor":
        return "largest_z_vapor"
    if phase == "stable":
        return "minimum_molar_residual_gibbs"
    raise ValueError(f"Unsupported phase root selector: {phase}")


def cubic_root_governance_report(
    spec: CubicEOSSpec,
    composition: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
    phase: PhaseRoot = "stable",
    volume_translation: tuple[VolumeTranslationSpec, ...] | None = None,
    strict_translation_components: bool = False,
) -> RootGovernanceReport:
    _validate_tp(temperature_K, pressure_Pa)
    normalized = _normalize_composition(composition)
    mixture = cubic_mixture_parameters(spec, normalized, temperature_K=temperature_K)
    roots = cubic_compressibility_roots(
        spec,
        normalized,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
    )
    policy = cubic_root_selection_policy(phase=phase, roots=roots)
    selected_z = select_cubic_root(
        spec,
        normalized,
        roots=roots,
        mixture=mixture,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        phase=phase,
    )
    mixture_shift, _, _ = mixture_volume_translation(
        spec,
        normalized,
        volume_translation=volume_translation,
        strict_components=strict_translation_components,
    )
    score_basis = _root_score_basis(phase, roots)
    scored_candidates: list[tuple[float, RootCandidateDiagnostic]] = []
    for root in roots:
        fugacity = cubic_fugacity_coefficients(
            spec,
            normalized,
            z_factor=root,
            mixture=mixture,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
        )
        residuals = cubic_residual_properties(
            spec,
            normalized,
            z_factor=root,
            mixture=mixture,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            root_selection_policy=policy,
            fugacity_coefficients=fugacity,
        )
        molar_volume = root * ideal_gas_molar_volume(temperature_K, pressure_Pa)
        translated_volume = molar_volume - mixture_shift
        warnings: list[str] = []
        if mixture_shift != 0.0 and translated_volume <= 0.0:
            warnings.append("translated_molar_volume_nonpositive")
        if phase == "vapor" and mixture_shift != 0.0:
            warnings.append("volume_translation_does_not_change_vapor_root_selection")
        selection_score = _root_selection_score(
            phase=phase,
            roots=roots,
            z_factor=root,
            residual_gibbs=residuals.molar_residual_gibbs_J_mol,
        )
        scored_candidates.append(
            (
                selection_score,
                RootCandidateDiagnostic(
                    z_factor=root,
                    phase_hint=_root_phase_hint(root, roots),
                    molar_volume_m3_mol=molar_volume,
                    translated_molar_volume_m3_mol=translated_volume
                    if mixture_shift != 0.0
                    else None,
                    translation_shift_m3_mol=mixture_shift,
                    molar_residual_gibbs_J_mol=residuals.molar_residual_gibbs_J_mol,
                    selection_score=selection_score,
                    rank=0,
                    selected=abs(root - selected_z) <= 1e-10,
                    warnings=tuple(warnings),
                ),
            )
        )
    rank_by_root = {
        candidate.z_factor: rank
        for rank, (_, candidate) in enumerate(
            sorted(scored_candidates, key=lambda item: item[0]),
            1,
        )
    }
    candidates = tuple(
        RootCandidateDiagnostic(
            z_factor=candidate.z_factor,
            phase_hint=candidate.phase_hint,
            molar_volume_m3_mol=candidate.molar_volume_m3_mol,
            translated_molar_volume_m3_mol=candidate.translated_molar_volume_m3_mol,
            translation_shift_m3_mol=candidate.translation_shift_m3_mol,
            molar_residual_gibbs_J_mol=candidate.molar_residual_gibbs_J_mol,
            selection_score=candidate.selection_score,
            rank=rank_by_root[candidate.z_factor],
            selected=candidate.selected,
            warnings=candidate.warnings,
        )
        for _, candidate in scored_candidates
    )
    selected_candidate = next(candidate for candidate in candidates if candidate.selected)
    return RootGovernanceReport(
        eos_id=spec.eos_id,
        model=spec.model,
        requested_phase=phase,
        root_selection_policy=policy,
        score_basis=score_basis,
        selected_z_factor=selected_z,
        selected_phase_hint=selected_candidate.phase_hint,
        roots=roots,
        candidates=candidates,
    )


def evaluate_volume_translated_cubic_eos(
    spec: CubicEOSSpec,
    composition: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
    phase: PhaseRoot = "vapor",
    volume_translation: tuple[VolumeTranslationSpec, ...] | None = None,
    strict_translation_components: bool = False,
    require_binary_interaction_provenance: bool = False,
) -> VolumeTranslationReport:
    if require_binary_interaction_provenance:
        validate_binary_interaction_provenance(spec, require_all_interactions=True)
    base_state = evaluate_cubic_eos(
        spec,
        composition,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        phase=phase,
    )
    mixture_shift, component_shifts, shift_contributions = mixture_volume_translation(
        spec,
        base_state.composition,
        volume_translation=volume_translation,
        strict_components=strict_translation_components,
    )
    translated_volume = base_state.molar_volume_m3_mol - mixture_shift
    if translated_volume <= 0.0:
        raise ValueError("translated molar volume must be positive")
    translated_z = pressure_Pa * translated_volume / (R_J_PER_MOL_K * temperature_K)
    translated_roots = translated_cubic_compressibility_roots(
        spec,
        base_state.composition,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        volume_translation=volume_translation,
        strict_translation_components=strict_translation_components,
    )
    root_governance = cubic_root_governance_report(
        spec,
        base_state.composition,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        phase=phase,
        volume_translation=volume_translation,
        strict_translation_components=strict_translation_components,
    )
    warnings: list[str] = []
    if phase == "vapor" and mixture_shift != 0.0:
        warnings.append(
            "Volume translation changes reported vapor molar volume but not "
            "the fugacity/residual-property convention in this compact report."
        )
    if not any(abs(root - translated_z) <= 1e-7 for root in translated_roots):
        warnings.append("translated selected root was not recovered by translated polynomial")
    return VolumeTranslationReport(
        eos_id=spec.eos_id,
        model=spec.model,
        phase=phase,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        composition=dict(base_state.composition),
        base_compressibility_factor=base_state.compressibility_factor,
        translated_compressibility_factor=translated_z,
        base_molar_volume_m3_mol=base_state.molar_volume_m3_mol,
        translated_molar_volume_m3_mol=translated_volume,
        mixture_shift_m3_mol=mixture_shift,
        component_shifts_m3_mol=component_shifts,
        shift_contributions_m3_mol=shift_contributions,
        translated_roots=translated_roots,
        root_governance=root_governance,
        warnings=tuple(warnings),
    )


def cubic_fugacity_coefficients(
    spec: CubicEOSSpec,
    composition: Mapping[str, float],
    *,
    z_factor: float,
    mixture: EOSMixtureParameters,
    temperature_K: float,
    pressure_Pa: float,
) -> dict[str, float]:
    x = _composition_vector(spec, composition)
    a_reduced, b_reduced = cubic_ab(
        mixture,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
    )
    if b_reduced <= 1e-14 or a_reduced <= 1e-14:
        return dict.fromkeys(spec.component_ids, 1.0)
    if z_factor <= b_reduced:
        raise ValueError("z_factor must be greater than reduced covolume B")
    a_mix = max(mixture.a_mix, 1e-300)
    b_mix = max(mixture.b_mix, 1e-300)
    coefficients = {}
    for i, component_id in enumerate(spec.component_ids):
        bi_over_b = mixture.b_values[i] / b_mix
        attraction_sum = sum(x[j] * mixture.a_matrix[i][j] for j in range(len(x)))
        attraction_term = 2.0 * attraction_sum / a_mix - bi_over_b
        if spec.model == "peng_robinson":
            sqrt2 = sqrt(2.0)
            log_argument = (
                z_factor + (1.0 + sqrt2) * b_reduced
            ) / (z_factor + (1.0 - sqrt2) * b_reduced)
            attraction = a_reduced / (2.0 * sqrt2 * b_reduced)
        else:
            log_argument = (z_factor + b_reduced) / z_factor
            attraction = a_reduced / b_reduced
        ln_phi = (
            bi_over_b * (z_factor - 1.0)
            - log(z_factor - b_reduced)
            - attraction * attraction_term * log(log_argument)
        )
        coefficients[component_id] = exp(ln_phi)
    return coefficients


def cubic_departure_log_argument(
    *,
    model: CubicModel,
    z_factor: float,
    b_reduced: float,
) -> float:
    if b_reduced < 0.0 or not isfinite(b_reduced):
        raise ValueError("b_reduced must be finite and nonnegative")
    if z_factor <= b_reduced or not isfinite(z_factor):
        raise ValueError("z_factor must be finite and greater than B")
    if b_reduced <= 1e-14:
        return 1.0
    if model == "peng_robinson":
        sqrt2 = sqrt(2.0)
        log_argument = (z_factor + (1.0 + sqrt2) * b_reduced) / (
            z_factor + (1.0 - sqrt2) * b_reduced
        )
    elif model == "srk":
        log_argument = (z_factor + b_reduced) / z_factor
    else:
        raise ValueError(f"Unsupported cubic EOS model: {model}")
    if log_argument <= 0.0 or not isfinite(log_argument):
        raise ValueError("cubic departure logarithm argument is not positive")
    return log_argument


def cubic_residual_properties(
    spec: CubicEOSSpec,
    composition: Mapping[str, float],
    *,
    z_factor: float,
    mixture: EOSMixtureParameters,
    temperature_K: float,
    pressure_Pa: float,
    root_selection_policy: str,
    fugacity_coefficients: Mapping[str, float] | None = None,
) -> CubicResidualProperties:
    _validate_tp(temperature_K, pressure_Pa)
    x = _composition_vector(spec, composition)
    a_reduced, b_reduced = cubic_ab(
        mixture,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
    )
    if b_reduced <= 1e-14 or a_reduced <= 1e-14:
        return CubicResidualProperties(
            model=spec.model,
            z_factor=z_factor,
            root_selection_policy=root_selection_policy,
            molar_residual_enthalpy_J_mol=0.0,
            molar_residual_entropy_J_mol_K=0.0,
            molar_residual_gibbs_J_mol=0.0,
            departure_log_argument=1.0,
            da_mix_dT=mixture.da_mix_dT,
        )

    log_argument = cubic_departure_log_argument(
        model=spec.model,
        z_factor=z_factor,
        b_reduced=b_reduced,
    )
    if spec.model == "peng_robinson":
        denominator = 2.0 * sqrt(2.0) * mixture.b_mix
    elif spec.model == "srk":
        denominator = mixture.b_mix
    else:
        raise ValueError(f"Unsupported cubic EOS model: {spec.model}")
    if denominator <= 0.0:
        raise ValueError("cubic residual denominator must be positive")

    departure_log = log(log_argument)
    attractive_departure = (
        (temperature_K * mixture.da_mix_dT - mixture.a_mix)
        / denominator
        * departure_log
    )
    entropy_departure = mixture.da_mix_dT / denominator * departure_log
    residual_enthalpy = (
        R_J_PER_MOL_K * temperature_K * (z_factor - 1.0) + attractive_departure
    )
    z_minus_b = z_factor - b_reduced
    if z_minus_b <= 0.0:
        raise ValueError("z_factor - B must be positive for residual entropy")
    residual_entropy = R_J_PER_MOL_K * log(z_minus_b) + entropy_departure
    if fugacity_coefficients is None:
        fugacity_coefficients = cubic_fugacity_coefficients(
            spec,
            composition,
            z_factor=z_factor,
            mixture=mixture,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
        )
    residual_gibbs = R_J_PER_MOL_K * temperature_K * sum(
        x[index] * log(max(float(fugacity_coefficients[component_id]), 1e-300))
        for index, component_id in enumerate(spec.component_ids)
    )
    return CubicResidualProperties(
        model=spec.model,
        z_factor=z_factor,
        root_selection_policy=root_selection_policy,
        molar_residual_enthalpy_J_mol=residual_enthalpy,
        molar_residual_entropy_J_mol_K=residual_entropy,
        molar_residual_gibbs_J_mol=residual_gibbs,
        departure_log_argument=log_argument,
        da_mix_dT=mixture.da_mix_dT,
    )


def cubic_state_derivatives(
    spec: CubicEOSSpec, composition: Mapping[str, float], *, temperature_K: float,
    pressure_Pa: float, phase: PhaseRoot = "vapor",
) -> dict[str, float | str]:
    """Single-branch caloric/volume derivatives of the existing cubic EOS.

    Mechanical derivatives are analytic. Departure Cp uses a five-point
    constant-P derivative of the same governed H departure, without switching
    to an independent equation or an upstream runtime.
    """
    state = evaluate_cubic_eos(spec, composition, temperature_K=temperature_K,
                               pressure_Pa=pressure_Pa, phase=phase)
    branch = phase
    if phase == "stable":
        branch = "vapor" if state.compressibility_factor == max(state.roots) else "liquid"
    mixture = cubic_mixture_parameters(spec, composition, temperature_K=temperature_K)
    volume, b = state.molar_volume_m3_mol, mixture.b_mix
    s1, s2 = _cubic_shape_constants(spec.model)
    denominator = (volume + s1 * b) * (volume + s2 * b)
    if volume <= b or denominator <= 0:
        raise ValueError("EOS derivative is outside the physical volume domain")
    p_t = R_J_PER_MOL_K / (volume - b) - mixture.da_mix_dT / denominator
    p_v = (-R_J_PER_MOL_K * temperature_K / (volume - b) ** 2
           + mixture.a_mix * (2 * volume + (s1 + s2) * b) / denominator ** 2)
    if not isfinite(p_v) or p_v >= 0:
        raise ValueError("EOS derivative requires a mechanically stable single-phase root")
    step = min(.01 * temperature_K, max(1e-3, 1e-4 * temperature_K))

    def departure(temperature):
        neighboring = evaluate_cubic_eos(spec, composition, temperature_K=temperature,
                                         pressure_Pa=pressure_Pa, phase=branch)
        if len(neighboring.roots) != len(state.roots):
            raise ValueError("EOS derivative stencil crosses a root-topology boundary")
        return neighboring.molar_residual_enthalpy_J_mol

    cp_departure = (-departure(temperature_K + 2 * step)
                    + 8 * departure(temperature_K + step)
                    - 8 * departure(temperature_K - step)
                    + departure(temperature_K - 2 * step)) / (12 * step)
    values = {
        "cp_departure_J_mol_K": cp_departure,
        "cp_minus_cv_J_mol_K": -temperature_K * p_t ** 2 / p_v,
        "d_molar_volume_dT_P": -p_t / p_v,
        "d_molar_volume_dP_T": 1 / p_v,
        "temperature_step_K": step,
    }
    if any(not isfinite(value) for value in values.values()):
        raise ValueError("EOS derivatives must remain finite")
    return {**values, "phase_root": branch,
            "method": "analytic mechanical / five-point constant-P departure"}


def _validate_tp(temperature_K: float, pressure_Pa: float) -> None:
    if not isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if not isfinite(pressure_Pa) or pressure_Pa <= 0:
        raise ValueError("pressure_Pa must be positive")


def _normalize_composition(composition: Mapping[str, float]) -> dict[str, float]:
    if not composition:
        raise ValueError("composition cannot be empty")
    if any(value < 0 or not isfinite(value) for value in composition.values()):
        raise ValueError("composition values must be finite and nonnegative")
    total = sum(composition.values())
    if total <= 0:
        raise ValueError("composition must contain positive material")
    return {component_id: float(value) / total for component_id, value in composition.items()}


def _composition_vector(
    spec: CubicEOSSpec,
    composition: Mapping[str, float],
) -> tuple[float, ...]:
    normalized = _normalize_composition(composition)
    missing = sorted(set(spec.component_ids) - set(normalized))
    extra = sorted(set(normalized) - set(spec.component_ids))
    if missing or extra:
        raise ValueError(f"Composition ids do not match EOS spec: missing={missing}, extra={extra}")
    return tuple(normalized[component_id] for component_id in spec.component_ids)


def _binary_interaction(spec: CubicEOSSpec, left: str, right: str) -> float:
    if left == right:
        return 0.0
    return float(
        spec.binary_interaction.get(
            f"{left}|{right}",
            spec.binary_interaction.get(f"{right}|{left}", 0.0),
        )
    )


def validate_binary_interaction_provenance(
    spec: CubicEOSSpec,
    *,
    require_all_interactions: bool = True,
) -> dict[str, dict[str, object]]:
    component_ids = set(spec.component_ids)
    provenance = binary_interaction_provenance_map(spec)
    provenance_records = {
        record.pair_key: record for record in spec.binary_interaction_provenance
    }
    for record in spec.binary_interaction_provenance:
        if record.component_i not in component_ids or record.component_j not in component_ids:
            raise ValueError("binary interaction provenance references unknown component")
    for key, value in spec.binary_interaction.items():
        left, right = _parse_pair_key(key)
        canonical_key = _canonical_pair_key(left, right)
        if require_all_interactions and canonical_key not in provenance:
            raise ValueError(f"missing binary interaction provenance for {canonical_key}")
        if canonical_key in provenance_records:
            provenance_value = provenance_records[canonical_key].k_ij
            if abs(provenance_value - float(value)) > 1e-12:
                raise ValueError(
                    f"binary interaction provenance k_ij does not match {canonical_key}"
                )
    return provenance


def binary_interaction_provenance_map(spec: CubicEOSSpec) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for record in spec.binary_interaction_provenance:
        if record.pair_key in records:
            raise ValueError(f"duplicate binary interaction provenance for {record.pair_key}")
        records[record.pair_key] = record.to_dict()
    return records


def mixture_volume_translation(
    spec: CubicEOSSpec,
    composition: Mapping[str, float],
    *,
    volume_translation: tuple[VolumeTranslationSpec, ...] | None = None,
    strict_components: bool = False,
) -> tuple[float, dict[str, float], dict[str, float]]:
    normalized = _normalize_composition(composition)
    _composition_vector(spec, normalized)
    translations = spec.volume_translation if volume_translation is None else volume_translation
    component_ids = set(spec.component_ids)
    shift_by_component: dict[str, float] = {}
    for translation in translations:
        if translation.component_id not in component_ids:
            raise ValueError(
                f"volume translation references unknown component: {translation.component_id}"
            )
        if translation.component_id in shift_by_component:
            raise ValueError(f"duplicate volume translation for {translation.component_id}")
        shift_by_component[translation.component_id] = translation.shift_m3_mol
    missing = sorted(component_ids - set(shift_by_component))
    if strict_components and missing:
        raise ValueError(f"missing volume translation for components: {missing}")
    component_shifts = {
        component_id: float(shift_by_component.get(component_id, 0.0))
        for component_id in spec.component_ids
    }
    contributions = {
        component_id: normalized[component_id] * component_shifts[component_id]
        for component_id in spec.component_ids
    }
    mixture_shift = sum(contributions.values())
    if not isfinite(mixture_shift):
        raise ValueError("mixture volume translation must be finite")
    return mixture_shift, component_shifts, contributions


def _parse_pair_key(key: str) -> tuple[str, str]:
    parts = key.split("|")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"binary interaction key must be 'component_i|component_j': {key}")
    return parts[0], parts[1]


def _canonical_pair_key(left: str, right: str) -> str:
    if not left or not right:
        raise ValueError("pair components cannot be empty")
    ordered = sorted((left, right))
    return f"{ordered[0]}|{ordered[1]}"


def _cubic_shape_constants(model: CubicModel) -> tuple[float, float]:
    if model == "peng_robinson":
        sqrt2 = sqrt(2.0)
        return 1.0 - sqrt2, 1.0 + sqrt2
    if model == "srk":
        return 0.0, 1.0
    raise ValueError(f"Unsupported cubic EOS model: {model}")


def _root_phase_hint(root: float, roots: tuple[float, ...]) -> str:
    if len(roots) == 1:
        return "single_root"
    if root == min(roots):
        return "liquid_like"
    if root == max(roots):
        return "vapor_like"
    return "middle_metastable"


def _root_score_basis(phase: PhaseRoot, roots: tuple[float, ...]) -> str:
    if len(roots) == 1:
        return "single_admissible_root"
    if phase == "liquid":
        return "smallest_z"
    if phase == "vapor":
        return "largest_z"
    if phase == "stable":
        return "molar_residual_gibbs_J_mol"
    raise ValueError(f"Unsupported phase root selector: {phase}")


def _root_selection_score(
    *,
    phase: PhaseRoot,
    roots: tuple[float, ...],
    z_factor: float,
    residual_gibbs: float,
) -> float:
    if len(roots) == 1:
        return 0.0
    if phase == "liquid":
        return z_factor
    if phase == "vapor":
        return -z_factor
    if phase == "stable":
        return residual_gibbs
    raise ValueError(f"Unsupported phase root selector: {phase}")


def _cross_a_temperature_derivative(
    left: CubicPureParameters,
    right: CubicPureParameters,
    *,
    kij: float,
) -> float:
    if left.a_alpha <= 0.0 or right.a_alpha <= 0.0:
        return 0.0
    cross_a = sqrt(left.a_alpha * right.a_alpha) * (1.0 - kij)
    return 0.5 * cross_a * (
        left.da_alpha_dT / left.a_alpha + right.da_alpha_dT / right.a_alpha
    )


__all__ = [
    "BinaryInteractionProvenance",
    "CubicEOSSpec",
    "CubicModel",
    "CubicPureParameters",
    "CubicResidualProperties",
    "EOSComponentSpec",
    "EOSMixtureParameters",
    "EOSState",
    "PhaseRoot",
    "RootCandidateDiagnostic",
    "RootGovernanceReport",
    "VolumeTranslationReport",
    "VolumeTranslationSpec",
    "binary_interaction_provenance_map",
    "cubic_ab",
    "cubic_compressibility_roots",
    "cubic_departure_log_argument",
    "cubic_fugacity_coefficients",
    "cubic_mixture_parameters",
    "cubic_pure_parameters",
    "cubic_residual_properties",
    "cubic_root_governance_report",
    "cubic_root_selection_policy",
    "eos_model_cards",
    "evaluate_cubic_eos",
    "evaluate_volume_translated_cubic_eos",
    "ideal_gas_molar_volume",
    "ideal_gas_pressure",
    "ideal_gas_state",
    "mixture_volume_translation",
    "select_cubic_root",
    "translated_cubic_compressibility_roots",
    "validate_binary_interaction_provenance",
]
