"""Executable mechanism-family interventions for causally reachable tasks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from math import exp, log
from typing import Any, Literal, cast

import numpy as np

from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reaction_network_specs import RateLawSpec, ReactionSpec
from chemworld.runtime.mechanism_manifest import CompiledMechanism
from chemworld.world.scenario import ScenarioInstance

MechanismFamilyMode = Literal[
    "rate_law_family",
    "topology_family",
    "constitutive_law_family",
]
RateLawReactionRole = Literal[
    "primary_competing_pathway",
    "primary_target_pathway",
]
RateLawTransformId = Literal[
    "arrhenius_form_and_scale_stress_v1",
    "catalytic_activity_order_pivot_stress_v1",
]
TopologyReactionRole = Literal[
    "primary_target_pathway",
    "catalyst_deactivation_pathway",
]
TopologyTransformId = Literal[
    "reversible_target_pathway_stress_v1",
    "stable_catalyst_topology_v1",
]
ConstitutiveTransformId = Literal[
    "partition_power_response_stress_v1",
    "partition_composition_response_stress_v1",
    "crystallization_occlusion_response_stress_v1",
    "electrochemical_response_stress_v1",
    "equilibrium_activity_response_stress_v1",
]
ARRHENIUS_FORM_AND_SCALE_STRESS: RateLawTransformId = "arrhenius_form_and_scale_stress_v1"
CATALYTIC_ACTIVITY_ORDER_PIVOT_STRESS: RateLawTransformId = (
    "catalytic_activity_order_pivot_stress_v1"
)
REVERSIBLE_TARGET_PATHWAY_STRESS: TopologyTransformId = "reversible_target_pathway_stress_v1"
STABLE_CATALYST_TOPOLOGY: TopologyTransformId = "stable_catalyst_topology_v1"
PARTITION_POWER_RESPONSE_STRESS: ConstitutiveTransformId = "partition_power_response_stress_v1"
PARTITION_COMPOSITION_RESPONSE_STRESS: ConstitutiveTransformId = (
    "partition_composition_response_stress_v1"
)
CRYSTALLIZATION_OCCLUSION_RESPONSE_STRESS: ConstitutiveTransformId = (
    "crystallization_occlusion_response_stress_v1"
)
ELECTROCHEMICAL_RESPONSE_STRESS: ConstitutiveTransformId = "electrochemical_response_stress_v1"
EQUILIBRIUM_ACTIVITY_RESPONSE_STRESS: ConstitutiveTransformId = (
    "equilibrium_activity_response_stress_v1"
)
CONSTITUTIVE_TRANSFORM_CALIBRATION_FIELDS: dict[str, frozenset[str]] = {
    PARTITION_POWER_RESPONSE_STRESS: frozenset({"partition_coefficient_exponent_at_full_severity"}),
    PARTITION_COMPOSITION_RESPONSE_STRESS: frozenset(
        {"composition_coupling_multiplier_at_full_severity"}
    ),
    CRYSTALLIZATION_OCCLUSION_RESPONSE_STRESS: frozenset(
        {"occlusion_law_selector_at_full_severity"}
    ),
    ELECTROCHEMICAL_RESPONSE_STRESS: frozenset(
        {
            "transfer_asymmetry_multiplier_at_full_severity",
            "selectivity_decay_multiplier_at_full_severity",
            "standard_potential_multiplier_at_full_severity",
        }
    ),
    EQUILIBRIUM_ACTIVITY_RESPONSE_STRESS: frozenset(
        {"activity_coefficient_ratio_at_full_severity"}
    ),
}
CONSTITUTIVE_TASK_TRANSFORMS: dict[str, ConstitutiveTransformId] = {
    "partition-discovery": PARTITION_POWER_RESPONSE_STRESS,
    "reaction-to-purification": PARTITION_COMPOSITION_RESPONSE_STRESS,
    "reaction-to-crystallization": CRYSTALLIZATION_OCCLUSION_RESPONSE_STRESS,
    "electrochemical-conversion": ELECTROCHEMICAL_RESPONSE_STRESS,
    "equilibrium-characterization": EQUILIBRIUM_ACTIVITY_RESPONSE_STRESS,
}
RATE_LAW_TRANSFORM_CALIBRATION_FIELDS: dict[str, frozenset[str]] = {
    ARRHENIUS_FORM_AND_SCALE_STRESS: frozenset(
        {
            "reference_temperature_K",
            "reference_rate_multiplier_at_full_severity",
            "temperature_exponent_at_full_severity",
        }
    ),
    CATALYTIC_ACTIVITY_ORDER_PIVOT_STRESS: frozenset(
        {
            "activity_order_at_full_severity",
            "catalyst_activity_pivot",
        }
    ),
}
MECHANISM_FAMILY_INTERVENTION_VERSION = "chemworld-mechanism-family-intervention-0.8"
REACTION_MECHANISM_TASKS = (
    "reaction-to-crystallization",
    "reaction-to-distillation",
    "flow-reaction-optimization",
)
CATALYST_DEACTIVATION_MECHANISM_TASKS = ("reaction-safety",)
PARTITION_MECHANISM_TASKS = ("partition-discovery", "reaction-to-purification")
CRYSTALLIZATION_CONSTITUTIVE_TASKS = ("reaction-to-crystallization",)
ELECTROCHEMICAL_MECHANISM_TASKS = ("electrochemical-conversion",)
EQUILIBRIUM_MECHANISM_TASKS = ("equilibrium-characterization",)
CONSTITUTIVE_MECHANISM_TASKS = (
    *PARTITION_MECHANISM_TASKS,
    *CRYSTALLIZATION_CONSTITUTIVE_TASKS,
    *ELECTROCHEMICAL_MECHANISM_TASKS,
    *EQUILIBRIUM_MECHANISM_TASKS,
)
MECHANISM_REACHABLE_TASKS = (
    *PARTITION_MECHANISM_TASKS,
    *REACTION_MECHANISM_TASKS,
    *CATALYST_DEACTIVATION_MECHANISM_TASKS,
    *ELECTROCHEMICAL_MECHANISM_TASKS,
    *EQUILIBRIUM_MECHANISM_TASKS,
)
MECHANISM_TASK_MODES: dict[str, tuple[MechanismFamilyMode, ...]] = {
    **dict.fromkeys(CONSTITUTIVE_MECHANISM_TASKS, ("constitutive_law_family",)),
    **dict.fromkeys(REACTION_MECHANISM_TASKS, ("rate_law_family", "topology_family")),
    "reaction-to-purification": ("constitutive_law_family",),
    "reaction-to-crystallization": (
        "rate_law_family",
        "topology_family",
        "constitutive_law_family",
    ),
    **dict.fromkeys(CATALYST_DEACTIVATION_MECHANISM_TASKS, ("topology_family",)),
}
TOPOLOGY_TASK_TRANSFORMS: dict[str, frozenset[TopologyTransformId]] = {
    **dict.fromkeys(REACTION_MECHANISM_TASKS, frozenset({REVERSIBLE_TARGET_PATHWAY_STRESS})),
    **dict.fromkeys(
        CATALYST_DEACTIVATION_MECHANISM_TASKS,
        frozenset({REVERSIBLE_TARGET_PATHWAY_STRESS, STABLE_CATALYST_TOPOLOGY}),
    ),
}


@dataclass(frozen=True)
class RateLawFamilyChange:
    """Declarative target and transform for a reaction rate-law intervention."""

    reaction_role: RateLawReactionRole = "primary_competing_pathway"
    transform_id: RateLawTransformId = ARRHENIUS_FORM_AND_SCALE_STRESS
    reference_temperature_K: float | None = None
    reference_rate_multiplier_at_full_severity: float | None = None
    temperature_exponent_at_full_severity: float | None = None
    activity_order_at_full_severity: float | None = None
    catalyst_activity_pivot: float | None = None

    def __post_init__(self) -> None:
        if self.reaction_role not in {
            "primary_competing_pathway",
            "primary_target_pathway",
        }:
            raise ValueError(f"unsupported rate-law reaction role: {self.reaction_role}")
        if self.transform_id not in RATE_LAW_TRANSFORM_CALIBRATION_FIELDS:
            raise ValueError(f"unsupported rate-law transform: {self.transform_id}")
        if self.transform_id == CATALYTIC_ACTIVITY_ORDER_PIVOT_STRESS:
            if any(
                value is not None
                for value in (
                    self.reference_temperature_K,
                    self.reference_rate_multiplier_at_full_severity,
                    self.temperature_exponent_at_full_severity,
                )
            ):
                raise ValueError(
                    "catalytic activity-order stress does not accept Arrhenius calibration"
                )
            order = (
                0.2
                if self.activity_order_at_full_severity is None
                else self.activity_order_at_full_severity
            )
            if not np.isfinite(order) or order <= 0.0:
                raise ValueError("catalytic activity order must be finite and positive")
            activity_pivot = (
                5.0 if self.catalyst_activity_pivot is None else self.catalyst_activity_pivot
            )
            if not np.isfinite(activity_pivot) or activity_pivot <= 0.0:
                raise ValueError("catalyst activity pivot must be finite and positive")
            object.__setattr__(self, "activity_order_at_full_severity", float(order))
            object.__setattr__(
                self,
                "catalyst_activity_pivot",
                float(activity_pivot),
            )
            return
        if (
            self.activity_order_at_full_severity is not None
            or self.catalyst_activity_pivot is not None
        ):
            raise ValueError(
                "Arrhenius form-and-scale stress does not accept activity-order calibration"
            )
        reference_temperature = (
            350.0 if self.reference_temperature_K is None else self.reference_temperature_K
        )
        reference_multiplier = (
            8.0
            if self.reference_rate_multiplier_at_full_severity is None
            else self.reference_rate_multiplier_at_full_severity
        )
        temperature_exponent = (
            0.75
            if self.temperature_exponent_at_full_severity is None
            else self.temperature_exponent_at_full_severity
        )
        if not np.isfinite(reference_temperature) or reference_temperature <= 0.0:
            raise ValueError("rate-law reference temperature must be finite and positive")
        if not np.isfinite(reference_multiplier) or reference_multiplier <= 1.0:
            raise ValueError("rate-law reference multiplier must be finite and greater than one")
        if not np.isfinite(temperature_exponent) or temperature_exponent <= 0.0:
            raise ValueError("rate-law temperature exponent must be finite and positive")
        object.__setattr__(self, "reference_temperature_K", float(reference_temperature))
        object.__setattr__(
            self,
            "reference_rate_multiplier_at_full_severity",
            float(reference_multiplier),
        )
        object.__setattr__(
            self,
            "temperature_exponent_at_full_severity",
            float(temperature_exponent),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RateLawFamilyChange:
        transform_id = str(payload.get("transform_id", ARRHENIUS_FORM_AND_SCALE_STRESS))
        calibration_fields = RATE_LAW_TRANSFORM_CALIBRATION_FIELDS.get(transform_id)
        if calibration_fields is None:
            raise ValueError(f"unsupported rate-law transform: {transform_id}")
        allowed = {"reaction_role", "transform_id", *calibration_fields}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown rate-law change fields: {unknown}")
        return cls(
            reaction_role=cast(
                RateLawReactionRole,
                str(payload.get("reaction_role", "primary_competing_pathway")),
            ),
            transform_id=cast(
                RateLawTransformId,
                transform_id,
            ),
            reference_temperature_K=(
                float(payload["reference_temperature_K"])
                if "reference_temperature_K" in payload
                else None
            ),
            reference_rate_multiplier_at_full_severity=(
                float(payload["reference_rate_multiplier_at_full_severity"])
                if "reference_rate_multiplier_at_full_severity" in payload
                else None
            ),
            temperature_exponent_at_full_severity=(
                float(payload["temperature_exponent_at_full_severity"])
                if "temperature_exponent_at_full_severity" in payload
                else None
            ),
            activity_order_at_full_severity=(
                float(payload["activity_order_at_full_severity"])
                if "activity_order_at_full_severity" in payload
                else None
            ),
            catalyst_activity_pivot=(
                float(payload["catalyst_activity_pivot"])
                if "catalyst_activity_pivot" in payload
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "reaction_role": self.reaction_role,
            "transform_id": self.transform_id,
        }
        if self.transform_id == CATALYTIC_ACTIVITY_ORDER_PIVOT_STRESS:
            payload["activity_order_at_full_severity"] = self.activity_order_at_full_severity
            payload["catalyst_activity_pivot"] = self.catalyst_activity_pivot
        else:
            payload.update(
                {
                    "reference_temperature_K": self.reference_temperature_K,
                    "reference_rate_multiplier_at_full_severity": (
                        self.reference_rate_multiplier_at_full_severity
                    ),
                    "temperature_exponent_at_full_severity": (
                        self.temperature_exponent_at_full_severity
                    ),
                }
            )
        return payload


@dataclass(frozen=True)
class TopologyFamilyChange:
    """Declarative target and calibration for a reaction-topology intervention."""

    reaction_role: TopologyReactionRole = "primary_target_pathway"
    transform_id: TopologyTransformId = REVERSIBLE_TARGET_PATHWAY_STRESS
    reverse_rate_constant_s_inv_at_full_severity: float | None = 0.000625

    def __post_init__(self) -> None:
        if self.transform_id == REVERSIBLE_TARGET_PATHWAY_STRESS:
            if self.reaction_role != "primary_target_pathway":
                raise ValueError(
                    "reversible target topology requires the primary_target_pathway role"
                )
            rate_constant = self.reverse_rate_constant_s_inv_at_full_severity
            if rate_constant is None or not np.isfinite(rate_constant) or rate_constant <= 0.0:
                raise ValueError("topology reverse rate constant must be finite and positive")
            object.__setattr__(
                self,
                "reverse_rate_constant_s_inv_at_full_severity",
                float(rate_constant),
            )
            return
        if self.transform_id == STABLE_CATALYST_TOPOLOGY:
            if self.reaction_role != "catalyst_deactivation_pathway":
                raise ValueError(
                    "stable-catalyst topology requires the catalyst_deactivation_pathway role"
                )
            if self.reverse_rate_constant_s_inv_at_full_severity is not None:
                raise ValueError("stable-catalyst topology does not accept a reverse rate constant")
            return
        raise ValueError(f"unsupported topology transform: {self.transform_id}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TopologyFamilyChange:
        transform_id = str(payload.get("transform_id", ""))
        allowed = {"reaction_role", "transform_id"}
        if transform_id == REVERSIBLE_TARGET_PATHWAY_STRESS:
            allowed.add("reverse_rate_constant_s_inv_at_full_severity")
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown topology change fields: {unknown}")
        required = {"reaction_role", "transform_id"} - set(payload)
        if transform_id == REVERSIBLE_TARGET_PATHWAY_STRESS:
            required.add("reverse_rate_constant_s_inv_at_full_severity")
            required -= set(payload)
        if required:
            raise ValueError(f"missing topology change fields: {sorted(required)}")
        return cls(
            reaction_role=cast(TopologyReactionRole, str(payload["reaction_role"])),
            transform_id=cast(TopologyTransformId, transform_id),
            reverse_rate_constant_s_inv_at_full_severity=(
                float(payload["reverse_rate_constant_s_inv_at_full_severity"])
                if "reverse_rate_constant_s_inv_at_full_severity" in payload
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "reaction_role": self.reaction_role,
            "transform_id": self.transform_id,
        }
        if self.transform_id == REVERSIBLE_TARGET_PATHWAY_STRESS:
            payload["reverse_rate_constant_s_inv_at_full_severity"] = (
                self.reverse_rate_constant_s_inv_at_full_severity
            )
        return payload


@dataclass(frozen=True)
class ConstitutiveLawFamilyChange:
    """Declarative transform and calibration for a constitutive-law family."""

    transform_id: ConstitutiveTransformId
    partition_coefficient_exponent_at_full_severity: float | None = None
    composition_coupling_multiplier_at_full_severity: float | None = None
    occlusion_law_selector_at_full_severity: float | None = None
    transfer_asymmetry_multiplier_at_full_severity: float | None = None
    selectivity_decay_multiplier_at_full_severity: float | None = None
    standard_potential_multiplier_at_full_severity: float | None = None
    activity_coefficient_ratio_at_full_severity: float | None = None

    def __post_init__(self) -> None:
        calibration_fields = CONSTITUTIVE_TRANSFORM_CALIBRATION_FIELDS.get(self.transform_id)
        if calibration_fields is None:
            raise ValueError(f"unsupported constitutive-law transform: {self.transform_id}")
        defaults = {
            PARTITION_POWER_RESPONSE_STRESS: {
                "partition_coefficient_exponent_at_full_severity": 1.75,
            },
            PARTITION_COMPOSITION_RESPONSE_STRESS: {
                "composition_coupling_multiplier_at_full_severity": 4.0,
            },
            CRYSTALLIZATION_OCCLUSION_RESPONSE_STRESS: {
                "occlusion_law_selector_at_full_severity": 2.0,
            },
            ELECTROCHEMICAL_RESPONSE_STRESS: {
                "transfer_asymmetry_multiplier_at_full_severity": 1.4,
                "selectivity_decay_multiplier_at_full_severity": 3.0,
                "standard_potential_multiplier_at_full_severity": 1.25,
            },
            EQUILIBRIUM_ACTIVITY_RESPONSE_STRESS: {
                "activity_coefficient_ratio_at_full_severity": 6.0,
            },
        }[self.transform_id]
        all_fields = {
            field
            for fields in CONSTITUTIVE_TRANSFORM_CALIBRATION_FIELDS.values()
            for field in fields
        }
        for field_name in all_fields:
            value = getattr(self, field_name)
            if field_name not in calibration_fields and value is not None:
                raise ValueError(f"{self.transform_id} does not accept {field_name}")
        for field_name in calibration_fields:
            value = getattr(self, field_name)
            resolved = defaults[field_name] if value is None else float(value)
            if not np.isfinite(resolved) or resolved <= 1.0:
                raise ValueError(f"{field_name} must be finite and greater than one")
            object.__setattr__(self, field_name, resolved)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConstitutiveLawFamilyChange:
        transform_id = str(payload.get("transform_id", ""))
        calibration_fields = CONSTITUTIVE_TRANSFORM_CALIBRATION_FIELDS.get(transform_id)
        if calibration_fields is None:
            raise ValueError(f"unsupported constitutive-law transform: {transform_id}")
        allowed = {"transform_id", *calibration_fields}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown constitutive-law change fields: {unknown}")
        kwargs = {
            field_name: (float(payload[field_name]) if field_name in payload else None)
            for field_name in calibration_fields
        }
        return cls(
            transform_id=cast(ConstitutiveTransformId, transform_id),
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"transform_id": self.transform_id}
        for field_name in CONSTITUTIVE_TRANSFORM_CALIBRATION_FIELDS[self.transform_id]:
            payload[field_name] = getattr(self, field_name)
        return payload


@dataclass(frozen=True)
class MechanismFamilyIntervention:
    mode: MechanismFamilyMode
    severity: float
    rate_law_change: RateLawFamilyChange | None = None
    topology_change: TopologyFamilyChange | None = None
    constitutive_law_change: ConstitutiveLawFamilyChange | None = None

    def __post_init__(self) -> None:
        if self.mode not in {
            "rate_law_family",
            "topology_family",
            "constitutive_law_family",
        }:
            raise ValueError(f"unsupported mechanism-family mode: {self.mode}")
        if not np.isfinite(self.severity) or not 0.0 < self.severity <= 1.0:
            raise ValueError("mechanism-family severity must be finite and in (0, 1]")
        if self.mode != "rate_law_family" and self.rate_law_change is not None:
            raise ValueError("rate_law_change is only valid for rate_law_family")
        if self.mode != "topology_family" and self.topology_change is not None:
            raise ValueError("topology_change is only valid for topology_family")
        if self.mode != "constitutive_law_family" and self.constitutive_law_change is not None:
            raise ValueError("constitutive_law_change is only valid for constitutive_law_family")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MechanismFamilyIntervention:
        if payload.get("kind") != "mechanism_family":
            raise ValueError("mechanism intervention kind must be 'mechanism_family'")
        allowed = {
            "kind",
            "mode",
            "severity",
            "rate_law_change",
            "topology_change",
            "constitutive_law_change",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown mechanism-family intervention fields: {unknown}")
        mode = cast(MechanismFamilyMode, str(payload["mode"]))
        rate_law_payload = payload.get("rate_law_change")
        if rate_law_payload is not None and not isinstance(rate_law_payload, dict):
            raise ValueError("rate_law_change must be an object")
        topology_payload = payload.get("topology_change")
        if topology_payload is not None and not isinstance(topology_payload, dict):
            raise ValueError("topology_change must be an object")
        constitutive_payload = payload.get("constitutive_law_change")
        if constitutive_payload is not None and not isinstance(
            constitutive_payload,
            dict,
        ):
            raise ValueError("constitutive_law_change must be an object")
        return cls(
            mode=mode,
            severity=float(payload["severity"]),
            rate_law_change=(
                RateLawFamilyChange.from_dict(rate_law_payload)
                if isinstance(rate_law_payload, dict)
                else None
            ),
            topology_change=(
                TopologyFamilyChange.from_dict(topology_payload)
                if isinstance(topology_payload, dict)
                else None
            ),
            constitutive_law_change=(
                ConstitutiveLawFamilyChange.from_dict(constitutive_payload)
                if isinstance(constitutive_payload, dict)
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": "mechanism_family",
            "mode": self.mode,
            "severity": self.severity,
        }
        if self.mode == "rate_law_family":
            payload["rate_law_change"] = self.resolved_rate_law_change.to_dict()
        elif self.mode == "topology_family":
            payload["topology_change"] = self.resolved_topology_change.to_dict()
        elif self.constitutive_law_change is not None:
            payload["constitutive_law_change"] = self.constitutive_law_change.to_dict()
        return payload

    @property
    def resolved_rate_law_change(self) -> RateLawFamilyChange:
        if self.mode != "rate_law_family":
            raise ValueError("only rate_law_family has a rate-law change contract")
        return self.rate_law_change or RateLawFamilyChange()

    @property
    def resolved_topology_change(self) -> TopologyFamilyChange:
        if self.mode != "topology_family":
            raise ValueError("only topology_family has a topology change contract")
        return self.topology_change or TopologyFamilyChange()

    def resolved_constitutive_law_change(
        self,
        task_id: str,
    ) -> ConstitutiveLawFamilyChange:
        if self.mode != "constitutive_law_family":
            raise ValueError("only constitutive_law_family has a constitutive-law contract")
        expected_transform = CONSTITUTIVE_TASK_TRANSFORMS.get(task_id)
        if expected_transform is None:
            raise ValueError(f"task {task_id!r} has no registered constitutive-law transform")
        change = self.constitutive_law_change or ConstitutiveLawFamilyChange(expected_transform)
        if change.transform_id != expected_transform:
            raise ValueError(
                f"task {task_id!r} requires {expected_transform!r}, got {change.transform_id!r}"
            )
        return change


def apply_mechanism_family_intervention(
    instance: ScenarioInstance,
    intervention: MechanismFamilyIntervention,
) -> ScenarioInstance:
    """Bind a derived compiled mechanism and opaque provenance to a world."""

    task_id = instance.spec.scenario_id
    if task_id not in MECHANISM_REACHABLE_TASKS:
        raise ValueError(f"task {task_id!r} does not expose a mechanism-family intervention")
    if intervention.mode not in MECHANISM_TASK_MODES[task_id]:
        raise ValueError(
            f"mode {intervention.mode!r} is not causally reachable for task {task_id!r}"
        )
    if intervention.mode == "topology_family":
        transform_id = intervention.resolved_topology_change.transform_id
        if transform_id not in TOPOLOGY_TASK_TRANSFORMS[task_id]:
            raise ValueError(
                f"task {task_id!r} does not expose topology transform {transform_id!r}"
            )
    base_hash = str(
        instance.compiled_mechanism.network.metadata.get(
            "base_mechanism_hash",
            instance.compiled_mechanism.mechanism_hash,
        )
    )
    constitutive_change = (
        intervention.resolved_constitutive_law_change(task_id)
        if intervention.mode == "constitutive_law_family"
        else None
    )
    resolved_intervention = (
        replace(
            intervention,
            constitutive_law_change=constitutive_change,
        )
        if constitutive_change is not None
        else intervention
    )
    contract_hash = mechanism_intervention_hash(
        base_mechanism_hash=base_hash,
        intervention=resolved_intervention,
    )
    domain_parameters = dict(instance.parameters.domain_parameters)
    if constitutive_change is not None:
        if constitutive_change.transform_id == PARTITION_POWER_RESPONSE_STRESS:
            full_exponent = constitutive_change.partition_coefficient_exponent_at_full_severity
            if full_exponent is None:  # guarded by ConstitutiveLawFamilyChange
                raise RuntimeError("partition constitutive calibration was not resolved")
            domain_parameters["partition_coefficient_exponent"] = (
                1.0 + (full_exponent - 1.0) * intervention.severity
            )
        elif constitutive_change.transform_id == PARTITION_COMPOSITION_RESPONSE_STRESS:
            full_multiplier = (
                constitutive_change.composition_coupling_multiplier_at_full_severity
            )
            if full_multiplier is None:  # guarded by ConstitutiveLawFamilyChange
                raise RuntimeError("partition composition calibration was not resolved")
            # Bound below in private initial-state metadata so adding this new
            # family does not change the identity of legacy parent Worlds.
        elif constitutive_change.transform_id == CRYSTALLIZATION_OCCLUSION_RESPONSE_STRESS:
            full_selector = constitutive_change.occlusion_law_selector_at_full_severity
            if full_selector is None:  # guarded by ConstitutiveLawFamilyChange
                raise RuntimeError("crystallization occlusion calibration was not resolved")
            if intervention.severity != 1.0:
                raise ValueError("crystallization occlusion law fork requires severity 1")
            # Bound below in private initial-state metadata for the same reason.
        elif constitutive_change.transform_id == ELECTROCHEMICAL_RESPONSE_STRESS:
            transfer = constitutive_change.transfer_asymmetry_multiplier_at_full_severity
            selectivity = constitutive_change.selectivity_decay_multiplier_at_full_severity
            potential = constitutive_change.standard_potential_multiplier_at_full_severity
            if (
                transfer is None or selectivity is None or potential is None
            ):  # guarded by ConstitutiveLawFamilyChange
                raise RuntimeError("electrochemical constitutive calibration was not resolved")
            domain_parameters["electro_transfer_asymmetry_multiplier"] = (
                1.0 + (transfer - 1.0) * intervention.severity
            )
            domain_parameters["electro_selectivity_decay_multiplier"] = (
                1.0 + (selectivity - 1.0) * intervention.severity
            )
            domain_parameters["electro_standard_potential_multiplier"] = (
                1.0 + (potential - 1.0) * intervention.severity
            )
    parameters = replace(
        instance.parameters,
        world_id=f"{instance.parameters.world_id}:mechanism-{contract_hash[:12]}",
        provider=f"{instance.parameters.provider}+mechanism-family",
        domain_parameters=domain_parameters,
    )
    metadata = {
        **instance.initial_state.metadata,
        "mechanism_id": instance.compiled_mechanism.mechanism_id,
        "mechanism_hash": instance.compiled_mechanism.mechanism_hash,
        "mechanism_family_intervention_version": MECHANISM_FAMILY_INTERVENTION_VERSION,
        "mechanism_family_intervention_hash": contract_hash,
    }
    if constitutive_change is not None:
        metadata["derived_constitutive_transform_id"] = constitutive_change.transform_id
        metadata["derived_constitutive_calibration"] = constitutive_change.to_dict()
        if constitutive_change.transform_id == PARTITION_COMPOSITION_RESPONSE_STRESS:
            full_multiplier = (
                constitutive_change.composition_coupling_multiplier_at_full_severity
            )
            if full_multiplier is None:
                raise RuntimeError("partition composition calibration was not resolved")
            metadata["partition_composition_coupling_multiplier"] = (
                1.0 + (full_multiplier - 1.0) * intervention.severity
            )
        elif constitutive_change.transform_id == CRYSTALLIZATION_OCCLUSION_RESPONSE_STRESS:
            metadata["crystallization_impurity_occlusion_law_id"] = (
                "surface_saturation_occlusion_v1"
            )
    if (
        constitutive_change is not None
        and constitutive_change.transform_id == EQUILIBRIUM_ACTIVITY_RESPONSE_STRESS
    ):
        activity_ratio = constitutive_change.activity_coefficient_ratio_at_full_severity
        if activity_ratio is None:  # guarded by ConstitutiveLawFamilyChange
            raise RuntimeError("equilibrium constitutive calibration was not resolved")
        metadata["equilibrium_activity_coefficient_ratio"] = exp(
            log(activity_ratio) * intervention.severity
        )
    return replace(
        instance,
        parameters=parameters,
        initial_state=instance.initial_state.replace(metadata=metadata),
    )


def derive_mechanism_family(
    base: CompiledMechanism,
    intervention: MechanismFamilyIntervention,
) -> CompiledMechanism:
    if intervention.mode == "constitutive_law_family":
        raise ValueError(
            "constitutive-law families modify executed world laws, not ReactionNetworkSpec"
        )
    network = (
        _rate_law_variant(
            base,
            intervention.severity,
            intervention.resolved_rate_law_change,
        )
        if intervention.mode == "rate_law_family"
        else _topology_variant(
            base,
            intervention.severity,
            intervention.resolved_topology_change,
        )
    )
    network = replace(
        network,
        metadata={**network.metadata, "base_mechanism_hash": base.mechanism_hash},
    )
    payload = {
        "version": MECHANISM_FAMILY_INTERVENTION_VERSION,
        "base_mechanism_hash": base.mechanism_hash,
        "intervention": intervention.to_dict(),
        "network": network.to_dict(),
    }
    derived_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    opaque_id = f"mechanism-family-{derived_hash[:12]}"
    validation = replace(
        base.manifest.validation_report,
        mechanism_id=opaque_id,
        mechanism_hash=derived_hash,
        source_path=f"{base.manifest.validation_report.source_path}#derived",
        reaction_count=len(network.reactions),
        rate_law_equation_ids=tuple(
            sorted({reaction.rate_law.equation_id for reaction in network.reactions})
        ),
        warnings=(
            *base.manifest.validation_report.warnings,
            "programmatic mechanism-family derivative validated by ReactionNetworkSpec",
        ),
    )
    manifest = replace(
        base.manifest,
        mechanism_id=opaque_id,
        mechanism_hash=derived_hash,
        source_path=validation.source_path,
        species_count=len(network.species),
        reaction_count=len(network.reactions),
        rate_law_equation_ids=validation.rate_law_equation_ids,
        validation_report=validation,
    )
    return replace(
        base,
        mechanism_id=opaque_id,
        mechanism_hash=derived_hash,
        network=network,
        species_index=network.species_index,
        stoichiometric_matrix=network.stoichiometric_matrix(),
        reaction_enthalpies={
            reaction.reaction_id: reaction.delta_h_J_per_mol for reaction in network.reactions
        },
        manifest=manifest,
    )


def mechanism_intervention_hash(
    *,
    base_mechanism_hash: str,
    intervention: MechanismFamilyIntervention,
) -> str:
    payload = {
        "version": MECHANISM_FAMILY_INTERVENTION_VERSION,
        "base_mechanism_hash": base_mechanism_hash,
        "intervention": intervention.to_dict(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _rate_law_variant(
    base: CompiledMechanism,
    severity: float,
    change: RateLawFamilyChange,
) -> ReactionNetworkSpec:
    network = base.network
    reactions = list(network.reactions)
    selected_index = _select_rate_law_reaction(
        base,
        reaction_role=change.reaction_role,
        transform_id=change.transform_id,
    )
    reaction = reactions[selected_index]
    parameters = dict(reaction.rate_law.parameters)
    transform_metadata: dict[str, object]
    if change.transform_id == CATALYTIC_ACTIVITY_ORDER_PIVOT_STRESS:
        baseline_order = float(parameters.get("activity_order", 1.0))
        full_severity_order = change.activity_order_at_full_severity
        activity_pivot = change.catalyst_activity_pivot
        if full_severity_order is None or activity_pivot is None:  # guarded by RateLawFamilyChange
            raise RuntimeError("catalytic activity-order calibration was not resolved")
        shifted_order = baseline_order + severity * (full_severity_order - baseline_order)
        # Pivot-normalize the Arrhenius scale so the old and new laws cross at
        # one declared catalyst activity.  This turns an order change into a
        # relational low-dose/high-dose intervention instead of making every
        # recipe uniformly faster or slower.
        order_delta = shifted_order - baseline_order
        parameters["A"] = float(parameters["A"]) * activity_pivot ** (-order_delta)
        parameters["activity_order"] = shifted_order
        reactions[selected_index] = replace(
            reaction,
            rate_law=replace(
                reaction.rate_law,
                rate_law_id=f"family-{reaction.rate_law.rate_law_id}",
                parameters=parameters,
            ),
        )
        transform_metadata = {
            "derived_family_baseline_activity_order": baseline_order,
            "derived_family_activity_order": shifted_order,
            "derived_family_activity_order_at_full_severity": full_severity_order,
            "derived_family_catalyst_activity_pivot": activity_pivot,
            "derived_family_activity_order_scale_multiplier": (activity_pivot ** (-order_delta)),
        }
    else:
        reference_temperature = change.reference_temperature_K
        reference_multiplier = change.reference_rate_multiplier_at_full_severity
        temperature_exponent = change.temperature_exponent_at_full_severity
        if (
            reference_temperature is None
            or reference_multiplier is None
            or temperature_exponent is None
        ):  # guarded by RateLawFamilyChange
            raise RuntimeError("Arrhenius form-and-scale calibration was not resolved")
        activity_factor = exp(log(reference_multiplier) * severity)
        if reaction.rate_law.equation_id == "arrhenius":
            # A mechanism-family shift must be observable over the task's practical
            # temperature/concentration range, not merely have a different equation id.
            # The transform is an explicitly declared operational stress, not a claim
            # that the synthetic multiplier is a fitted material constant.
            exponent = temperature_exponent * severity
            parameters["b"] = exponent
            parameters["A"] = (
                float(parameters["A"]) * activity_factor / reference_temperature**exponent
            )
            equation_id = "modified_arrhenius"
        else:
            exponent = float(parameters.pop("b", 0.0))
            parameters["A"] = (
                float(parameters["A"]) * reference_temperature**exponent / activity_factor
            )
            equation_id = "arrhenius"
        reactions[selected_index] = replace(
            reaction,
            rate_law=replace(
                reaction.rate_law,
                rate_law_id=f"family-{reaction.rate_law.rate_law_id}",
                equation_id=equation_id,
                parameters=parameters,
            ),
        )
        transform_metadata = {
            "derived_family_reference_temperature_K": reference_temperature,
            "derived_family_reference_rate_multiplier": activity_factor,
        }
    return replace(
        network,
        network_id=f"{network.network_id}-family",
        reactions=tuple(reactions),
        metadata={
            **network.metadata,
            "derived_family": True,
            "derived_family_target_reaction_id": reaction.reaction_id,
            "derived_family_target_reaction_role": change.reaction_role,
            "derived_family_transform_id": change.transform_id,
            **transform_metadata,
        },
    )


def _select_rate_law_reaction(
    base: CompiledMechanism,
    *,
    reaction_role: RateLawReactionRole,
    transform_id: RateLawTransformId,
) -> int:
    """Resolve a semantic reaction role without depending on declaration order."""

    supported_equations = (
        {"catalytic_activity"}
        if transform_id == CATALYTIC_ACTIVITY_ORDER_PIVOT_STRESS
        else {"arrhenius", "modified_arrhenius"}
    )
    targets = set(base.score_spec.target_species)
    limiting = base.score_spec.initial_limiting_species
    feed_species = {limiting}
    feed_species.update(
        species_id
        for species_id, roles in base.species_roles.items()
        if any("reactant" in role for role in roles)
    )
    candidates: list[int] = []
    for index, reaction in enumerate(base.network.reactions):
        if reaction.rate_law.equation_id not in supported_equations:
            continue
        reactants = set(reaction.reactants)
        products = set(reaction.products)
        if reaction_role == "primary_target_pathway":
            matches = bool(reactants & feed_species) and bool(products & targets)
        else:
            matches = (
                bool(reactants & feed_species)
                and not bool(reactants & targets)
                and not bool(products & targets)
            )
        if matches:
            candidates.append(index)
    if len(candidates) != 1:
        reaction_ids = [base.network.reactions[index].reaction_id for index in candidates]
        raise ValueError(
            f"mechanism {base.network.network_id!r} requires exactly one transformable "
            f"{reaction_role!r} rate law for {transform_id!r}; found {reaction_ids}"
        )
    return candidates[0]


def _topology_variant(
    base: CompiledMechanism,
    severity: float,
    change: TopologyFamilyChange,
) -> ReactionNetworkSpec:
    if change.transform_id == STABLE_CATALYST_TOPOLOGY:
        return _stable_catalyst_variant(base, severity, change)
    network = base.network
    limiting = base.score_spec.initial_limiting_species
    targets = set(base.score_spec.target_species)
    candidates = [
        reaction
        for reaction in network.reactions
        if limiting in reaction.reactants and targets.intersection(reaction.products)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"mechanism {network.network_id!r} requires exactly one "
            f"{change.reaction_role!r} topology target; found "
            f"{[reaction.reaction_id for reaction in candidates]}"
        )
    forward = candidates[0]
    full_reverse_rate = change.reverse_rate_constant_s_inv_at_full_severity
    if full_reverse_rate is None:  # guarded by TopologyFamilyChange
        raise RuntimeError("reversible topology calibration was not resolved")
    reverse_rate_constant = full_reverse_rate * severity
    reverse = ReactionSpec(
        reaction_id="family_reverse_channel",
        equation=f"reverse({forward.equation})",
        stoichiometry={
            species_id: -coefficient for species_id, coefficient in forward.stoichiometry.items()
        },
        reversible=False,
        rate_law=RateLawSpec(
            rate_law_id="family_reverse_mass_action",
            equation_id="mass_action",
            parameters={"k": reverse_rate_constant},
        ),
        delta_h_J_per_mol=-forward.delta_h_J_per_mol,
        metadata={
            "derived_family": True,
            "derived_family_transform_id": change.transform_id,
            "derived_family_target_reaction_id": forward.reaction_id,
            "derived_family_target_reaction_role": change.reaction_role,
        },
    )
    return replace(
        network,
        network_id=f"{network.network_id}-family",
        reactions=(*network.reactions, reverse),
        metadata={
            **network.metadata,
            "derived_family": True,
            "derived_family_transform_id": change.transform_id,
            "derived_family_target_reaction_id": forward.reaction_id,
            "derived_family_target_reaction_role": change.reaction_role,
            "derived_family_reverse_rate_constant_s_inv": reverse_rate_constant,
            "derived_family_reverse_rate_constant_s_inv_at_full_severity": (full_reverse_rate),
        },
    )


def _stable_catalyst_variant(
    base: CompiledMechanism,
    severity: float,
    change: TopologyFamilyChange,
) -> ReactionNetworkSpec:
    if severity != 1.0:
        raise ValueError("stable-catalyst topology is discrete and requires severity=1.0")
    candidates = [
        reaction
        for reaction in base.network.reactions
        if reaction.rate_law.equation_id == "catalyst_deactivation"
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"mechanism {base.network.network_id!r} requires exactly one catalyst-deactivation "
            f"topology target; found {[reaction.reaction_id for reaction in candidates]}"
        )
    removed = candidates[0]
    retained = tuple(
        reaction
        for reaction in base.network.reactions
        if reaction.reaction_id != removed.reaction_id
    )
    return replace(
        base.network,
        network_id=f"{base.network.network_id}-family",
        reactions=retained,
        metadata={
            **base.network.metadata,
            "derived_family": True,
            "derived_family_transform_id": change.transform_id,
            "derived_family_target_reaction_id": removed.reaction_id,
            "derived_family_target_reaction_role": change.reaction_role,
            "derived_family_removed_reaction_count": 1,
        },
    )


__all__ = [
    "ARRHENIUS_FORM_AND_SCALE_STRESS",
    "CATALYST_DEACTIVATION_MECHANISM_TASKS",
    "CATALYTIC_ACTIVITY_ORDER_PIVOT_STRESS",
    "CONSTITUTIVE_MECHANISM_TASKS",
    "CONSTITUTIVE_TASK_TRANSFORMS",
    "CONSTITUTIVE_TRANSFORM_CALIBRATION_FIELDS",
    "ELECTROCHEMICAL_MECHANISM_TASKS",
    "ELECTROCHEMICAL_RESPONSE_STRESS",
    "EQUILIBRIUM_ACTIVITY_RESPONSE_STRESS",
    "EQUILIBRIUM_MECHANISM_TASKS",
    "MECHANISM_FAMILY_INTERVENTION_VERSION",
    "MECHANISM_REACHABLE_TASKS",
    "MECHANISM_TASK_MODES",
    "PARTITION_MECHANISM_TASKS",
    "PARTITION_POWER_RESPONSE_STRESS",
    "RATE_LAW_TRANSFORM_CALIBRATION_FIELDS",
    "REACTION_MECHANISM_TASKS",
    "REVERSIBLE_TARGET_PATHWAY_STRESS",
    "STABLE_CATALYST_TOPOLOGY",
    "TOPOLOGY_TASK_TRANSFORMS",
    "ConstitutiveLawFamilyChange",
    "MechanismFamilyIntervention",
    "RateLawFamilyChange",
    "TopologyFamilyChange",
    "apply_mechanism_family_intervention",
    "derive_mechanism_family",
    "mechanism_intervention_hash",
]
