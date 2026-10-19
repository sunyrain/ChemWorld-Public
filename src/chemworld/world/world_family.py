"""Typed, executable interventions for vNext benchmark world families."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from math import exp, log
from typing import Any, Literal, cast

import numpy as np

from chemworld.foundation.state import SpeciesLedger
from chemworld.world.scenario import ScenarioInstance

ShiftMode = Literal["interpolation", "extrapolation", "composition", "observation_noise"]
TargetKind = Literal["domain_parameter", "world_parameter", "initial_state"]

WORLD_FAMILY_INTERVENTION_VERSION = "chemworld-world-family-intervention-0.2"


@dataclass(frozen=True)
class WorldAxisSpec:
    axis_id: str
    task_id: str
    label: str
    target_kind: TargetKind
    target_keys: tuple[str, ...]
    rationale: str
    modes: tuple[ShiftMode, ...] = (
        "interpolation",
        "extrapolation",
        "composition",
        "observation_noise",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "task_id": self.task_id,
            "label": self.label,
            "target_kind": self.target_kind,
            "target_keys": list(self.target_keys),
            "rationale": self.rationale,
            "modes": list(self.modes),
        }


@dataclass(frozen=True)
class AxisIntervention:
    axis_id: str
    mode: ShiftMode
    severity: float

    def __post_init__(self) -> None:
        if self.mode not in {
            "interpolation",
            "extrapolation",
            "composition",
            "observation_noise",
        }:
            raise ValueError(f"Unsupported world-family mode: {self.mode}")
        if not np.isfinite(self.severity) or not -1.0 <= self.severity <= 1.0:
            raise ValueError("axis intervention severity must be finite and in [-1, 1]")
        if self.severity == 0.0:
            raise ValueError("axis intervention severity must be non-zero")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AxisIntervention:
        mode = str(payload["mode"])
        return cls(
            axis_id=str(payload["axis_id"]),
            mode=cast(ShiftMode, mode),
            severity=float(payload["severity"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"axis_id": self.axis_id, "mode": self.mode, "severity": self.severity}


def _axis(
    task_id: str,
    axis_id: str,
    label: str,
    target_kind: TargetKind,
    target_keys: tuple[str, ...],
    rationale: str,
    modes: tuple[ShiftMode, ...] = (
        "interpolation",
        "extrapolation",
        "composition",
        "observation_noise",
    ),
) -> WorldAxisSpec:
    return WorldAxisSpec(axis_id, task_id, label, target_kind, target_keys, rationale, modes)


WORLD_AXIS_REGISTRY: dict[str, WorldAxisSpec] = {
    spec.axis_id: spec
    for spec in (
        _axis(
            "partition-discovery",
            "partition.distribution-coefficient",
            "distribution coefficient",
            "domain_parameter",
            ("partition_coefficient_multiplier",),
            "Changes the activity-corrected product and impurity partition regime.",
        ),
        _axis(
            "partition-discovery",
            "partition.phase-volume-ratio",
            "phase-volume ratio",
            "domain_parameter",
            ("partition_phase_volume_multiplier",),
            "Changes the effective organic-to-aqueous contact volume ratio.",
        ),
        _axis(
            "reaction-to-purification",
            "purification.partition-strength",
            "downstream distribution-coefficient scale",
            "domain_parameter",
            ("partition_coefficient_multiplier",),
            "Changes the product/impurity partition scale after the upstream reaction.",
        ),
        _axis(
            "reaction-to-purification",
            "purification.phase-volume-ratio",
            "downstream phase-volume ratio",
            "domain_parameter",
            ("partition_phase_volume_multiplier",),
            "Changes the effective organic-to-aqueous contact-volume ratio.",
        ),
        _axis(
            "reaction-to-crystallization",
            "crystallization.kinetic-profile",
            "primary nucleation-rate scale",
            "domain_parameter",
            ("crystallization_nucleation_multiplier",),
            "Changes primary nucleation while retaining the same population-balance law.",
        ),
        _axis(
            "reaction-to-crystallization",
            "crystallization.solubility-cooling-profile",
            "crystallization solubility regime",
            "domain_parameter",
            ("crystallization_solubility_multiplier",),
            "Changes the bounded van't Hoff solubility scale; the public cooling "
            "trajectory remains an Agent-controlled probe.",
        ),
        _axis(
            "reaction-to-crystallization",
            "crystallization.population-regime",
            "primary-nucleation versus seed-growth regime",
            "domain_parameter",
            (
                "crystallization_nucleation_multiplier",
                "crystallization_growth_multiplier",
            ),
            "Moves primary nucleation and crystal growth in opposite directions while retaining "
            "the same mass-conserving population-balance equations.",
        ),
        _axis(
            "reaction-to-crystallization",
            "crystallization.impurity-occlusion-law",
            "supersaturation transfer versus surface-saturation impurity occlusion",
            "initial_state",
            ("crystallization_impurity_occlusion_law_id",),
            "Switches between two executable impurity-occlusion equations under the same "
            "public cooling and seeding contract.",
            ("extrapolation",),
        ),
        _axis(
            "reaction-to-crystallization",
            "crystallization.impurity-occlusion-capacity",
            "constant impurity-occlusion capacity null",
            "initial_state",
            ("crystallization_impurity_occlusion_capacity_multiplier",),
            "Authoring-only scalar null used to test whether one constant capacity "
            "multiplier can absorb a proposed structural occlusion-law fork.",
            ("extrapolation",),
        ),
        _axis(
            "reaction-to-distillation",
            "distillation.relative-volatility",
            "relative volatility",
            "domain_parameter",
            ("distillation_relative_volatility_multiplier",),
            "Changes light-key versus heavy-key volatility in the bounded column.",
        ),
        _axis(
            "reaction-to-distillation",
            "distillation.reaction-selectivity",
            "reaction selectivity",
            "world_parameter",
            ("catalyst_effects", "solvent_effects"),
            "Changes desired versus undesired reaction-rate modifiers before separation.",
        ),
        _axis(
            "flow-reaction-optimization",
            "flow.reaction-kinetics",
            "reaction kinetics",
            "domain_parameter",
            ("flow_rate_multiplier",),
            "Scales the compiled network rate field inside the PFR only.",
        ),
        _axis(
            "flow-reaction-optimization",
            "flow.residence-thermal-boundary",
            "fixed-reactor thermal boundary",
            "domain_parameter",
            ("flow_boundary_ua_multiplier",),
            "Spans insulated to strongly heat-coupled walls while reactor volume remains "
            "fixed and residence time is derived from flow rate; positive severity is the "
            "weak-boundary direction.",
        ),
        _axis(
            "electrochemical-conversion",
            "electrochem.selectivity-kinetics",
            "overpotential-selectivity response",
            "domain_parameter",
            ("electro_selectivity_decay_multiplier",),
            "Changes the selectivity penalty induced by overpotential while retaining "
            "the same electrode and transport laws.",
        ),
        _axis(
            "electrochemical-conversion",
            "electrochem.mass-transfer-resistance",
            "mass-transfer and resistance regime",
            "domain_parameter",
            ("electro_resistance_multiplier",),
            "Changes electrolyte and contact resistance seen by the cell model.",
        ),
        _axis(
            "equilibrium-characterization",
            "equilibrium.acid-base-constants",
            "acid-base constants",
            "initial_state",
            ("hidden_equilibrium_pka",),
            "Changes the hidden monoprotic-acid pKa within the same solver family.",
        ),
        _axis(
            "equilibrium-characterization",
            "equilibrium.activity-coefficient-regime",
            "solution nonideality regime",
            "initial_state",
            ("equilibrium_activity_coefficient_ratio",),
            "Changes the hidden activity-coefficient ratio while retaining the same "
            "acid-base and precipitation solver laws.",
        ),
        _axis(
            "equilibrium-characterization",
            "equilibrium.mechanism-benchmark",
            "coupled weak-acid precipitation benchmark substrate",
            "initial_state",
            (
                "equilibrium_mechanism_benchmark_version",
                "equilibrium_mechanism_family",
                "hidden_equilibrium_log10_ksp",
                "equilibrium_precipitating_cation_fraction",
            ),
            "Selects the registered EQ-S coupled-equilibrium substrate without exposing "
            "its hidden mechanism or constants.",
            ("extrapolation",),
        ),
        _axis(
            "equilibrium-characterization",
            "equilibrium.aqueous-ion-pair-network",
            "aqueous ion-pair mechanism topology",
            "initial_state",
            (
                "equilibrium_mechanism_family",
                "equilibrium_aqueous_association_beta_L_per_mol",
            ),
            "Adds a distinct aqueous MA intermediate and its mass-action equation to "
            "the coupled weak-acid precipitation network.",
            ("interpolation",),
        ),
    )
}


def axes_for_task(task_id: str) -> tuple[WorldAxisSpec, ...]:
    return tuple(spec for spec in WORLD_AXIS_REGISTRY.values() if spec.task_id == task_id)


def intervention_contract_hash(interventions: tuple[AxisIntervention, ...]) -> str:
    canonical = sorted(interventions, key=lambda item: item.axis_id)
    payload = {
        "version": WORLD_FAMILY_INTERVENTION_VERSION,
        "interventions": [item.to_dict() for item in canonical],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def apply_axis_interventions(
    instance: ScenarioInstance,
    interventions: tuple[AxisIntervention, ...],
) -> ScenarioInstance:
    """Apply task-scoped interventions to an already deterministic base world."""

    if not interventions:
        return instance
    seen: set[str] = set()
    parameters = instance.parameters
    domain = dict(parameters.domain_parameters)
    catalyst_effects = np.array(parameters.catalyst_effects, copy=True)
    solvent_effects = np.array(parameters.solvent_effects, copy=True)
    state = instance.initial_state
    applied: list[dict[str, Any]] = []
    for intervention in sorted(interventions, key=lambda item: item.axis_id):
        if intervention.axis_id in seen:
            raise ValueError(f"Duplicate axis intervention: {intervention.axis_id}")
        seen.add(intervention.axis_id)
        try:
            spec = WORLD_AXIS_REGISTRY[intervention.axis_id]
        except KeyError as exc:
            raise KeyError(f"Unknown world-family axis: {intervention.axis_id}") from exc
        if spec.task_id != instance.spec.scenario_id:
            raise ValueError(
                f"Axis {spec.axis_id} belongs to {spec.task_id}, not {instance.spec.scenario_id}"
            )
        if intervention.mode not in spec.modes:
            raise ValueError(f"Axis {spec.axis_id} does not support {intervention.mode}")
        if intervention.axis_id == "distillation.reaction-selectivity":
            factor = _effect_multiplier(intervention.mode, intervention.severity)
            catalyst_effects[:, 0] *= factor
            catalyst_effects[:, 1:] /= factor**0.25
            solvent_effects[:, 0] *= factor
            solvent_effects[:, 1:] /= factor**0.25
        elif intervention.axis_id == "crystallization.population-regime":
            factor = exp(log(4.0) * intervention.severity)
            domain["crystallization_nucleation_multiplier"] *= factor
            domain["crystallization_growth_multiplier"] /= factor
        elif intervention.axis_id == "crystallization.impurity-occlusion-law":
            if intervention.mode != "extrapolation" or intervention.severity != 1.0:
                raise ValueError(
                    "crystallization impurity-occlusion law is a frozen binary axis; "
                    "mode must be extrapolation and severity must equal 1"
                )
            metadata = dict(state.metadata)
            metadata["crystallization_impurity_occlusion_law_id"] = (
                "surface_saturation_occlusion_v1"
            )
            state = state.replace(metadata=metadata)
        elif intervention.axis_id == "crystallization.impurity-occlusion-capacity":
            if intervention.mode != "extrapolation" or not 0.0 < intervention.severity <= 1.0:
                raise ValueError(
                    "crystallization scalar-null capacity requires extrapolation severity in (0, 1]"
                )
            metadata = dict(state.metadata)
            metadata["crystallization_impurity_occlusion_capacity_multiplier"] = (
                1.0 + 5.0 * intervention.severity
            )
            state = state.replace(metadata=metadata)
        elif intervention.axis_id == "flow.residence-thermal-boundary":
            domain["flow_boundary_ua_multiplier"] *= _flow_boundary_multiplier(
                intervention.mode, intervention.severity
            )
        elif intervention.axis_id == "equilibrium.acid-base-constants":
            metadata = dict(state.metadata)
            metadata["hidden_equilibrium_pka"] = float(
                metadata["hidden_equilibrium_pka"]
            ) + _additive_shift(intervention.mode, intervention.severity, 0.45, 1.2)
            state = state.replace(metadata=metadata)
        elif intervention.axis_id == "equilibrium.activity-coefficient-regime":
            metadata = dict(state.metadata)
            metadata["equilibrium_activity_coefficient_ratio"] = float(
                metadata.get("equilibrium_activity_coefficient_ratio", 1.0)
            ) * _effect_multiplier(intervention.mode, intervention.severity)
            state = state.replace(metadata=metadata)
        elif intervention.axis_id == "equilibrium.mechanism-benchmark":
            if intervention.mode != "extrapolation" or intervention.severity != 1.0:
                raise ValueError(
                    "equilibrium mechanism benchmark is a frozen substrate selector; "
                    "mode must be extrapolation and severity must equal 1"
                )
            metadata = dict(state.metadata)
            metadata.update(
                {
                    "equilibrium_mechanism_benchmark_version": "eq-s-v0.2",
                    "hidden_equilibrium_log10_ksp": -5.2,
                    "hidden_equilibrium_ksp": 10.0**-5.2,
                    "equilibrium_precipitating_cation_fraction": 0.15,
                    "equilibrium_activity_coefficient_ratio": 1.0,
                }
            )
            metadata.setdefault(
                "equilibrium_mechanism_family",
                "direct_free_ion_precipitation",
            )
            state = state.replace(metadata=metadata)
        elif intervention.axis_id == "equilibrium.aqueous-ion-pair-network":
            if intervention.mode != "interpolation":
                raise ValueError("aqueous ion-pair network requires interpolation mode")
            metadata = dict(state.metadata)
            metadata.update(
                {
                    "equilibrium_mechanism_family": "aqueous_ion_pair_intermediate",
                    "equilibrium_aqueous_association_beta_L_per_mol": (
                        7000.0 + 2500.0 * intervention.severity
                    ),
                }
            )
            state = state.replace(metadata=metadata)
        else:
            factor = _effect_multiplier(intervention.mode, intervention.severity)
            for key in spec.target_keys:
                domain[key] *= factor
        if intervention.mode == "observation_noise":
            domain["observation_noise_multiplier"] *= _noise_multiplier(intervention.severity)
        if intervention.mode == "composition":
            state = _shift_initial_composition(state, instance, intervention.severity)
        applied.append({**spec.to_dict(), **intervention.to_dict()})

    contract_hash = intervention_contract_hash(interventions)
    parameters = replace(
        parameters,
        world_id=f"{parameters.world_id}:intervention-{contract_hash[:12]}",
        provider=f"{parameters.provider}+vnext-intervention",
        catalyst_effects=catalyst_effects,
        solvent_effects=solvent_effects,
        domain_parameters=domain,
    )
    metadata = {
        **state.metadata,
        "world_family_intervention_version": WORLD_FAMILY_INTERVENTION_VERSION,
        "world_family_intervention_hash": contract_hash,
        "world_family_interventions": applied,
    }
    state = state.replace(metadata=metadata)
    return replace(instance, parameters=parameters, initial_state=state)


def _effect_multiplier(mode: ShiftMode, severity: float) -> float:
    amplitude = 0.25 if mode in {"interpolation", "observation_noise"} else 0.75
    if mode == "composition":
        amplitude = 0.40
    return exp(log(1.0 + amplitude) * severity)


def _noise_multiplier(severity: float) -> float:
    return exp(log(3.0) * severity)


def _flow_boundary_multiplier(mode: ShiftMode, severity: float) -> float:
    """Return an audited weak-to-strong wall-coupling span for fixed flow hardware."""

    positive_endpoint = {
        "interpolation": 0.25,
        "extrapolation": 0.02,
        "composition": 0.10,
        "observation_noise": 0.25,
    }[mode]
    return exp(log(positive_endpoint) * severity)


def _additive_shift(
    mode: ShiftMode,
    severity: float,
    interpolation_amplitude: float,
    extrapolation_amplitude: float,
) -> float:
    amplitude = extrapolation_amplitude if mode == "extrapolation" else interpolation_amplitude
    return amplitude * severity


def _shift_initial_composition(
    state: Any,
    instance: ScenarioInstance,
    severity: float,
) -> Any:
    limiting = instance.compiled_mechanism.score_spec.initial_limiting_species
    if limiting is None or limiting not in state.species_amounts:
        return state
    factor = exp(log(1.4) * severity)
    amounts = state.species_amounts.copy()
    amounts[limiting] = max(float(amounts[limiting]) * factor, 0.0)
    species = state.species
    if species is not None:
        initial = dict(species.initial_amounts_mol)
        initial[limiting] = max(float(initial.get(limiting, 0.0)) * factor, 0.0)
        species = SpeciesLedger(
            initial_amounts_mol=initial,
            species_roles=species.species_roles,
        )
    return state.replace(species_amounts=amounts, species=species)


__all__ = [
    "WORLD_AXIS_REGISTRY",
    "WORLD_FAMILY_INTERVENTION_VERSION",
    "AxisIntervention",
    "ShiftMode",
    "WorldAxisSpec",
    "apply_axis_interventions",
    "axes_for_task",
    "intervention_contract_hash",
]
