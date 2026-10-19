"""Public material identities without overstating benchmark chemistry realism."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from chemworld.physchem.component_registry import curated_component_registry
from chemworld.world.actions import CATALYSTS, ELECTROLYTE_PROFILES, SOLVENTS
from chemworld.world.crystallization_material_family import (
    REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY,
    crystallization_material_family,
    normalize_crystallization_material_family,
)
from chemworld.world.electrochemical_material_family import (
    NOMINAL_PRIOR_MATERIAL_FAMILY,
    electrochemical_material_family,
    normalize_electrochemical_material_family,
)
from chemworld.world.parameters import (
    REACTION_NOMINAL_CATALYST_ACTIVITY_PROFILES,
    REACTION_NOMINAL_SOLVENT_ACTIVITY_PROFILES,
)
from chemworld.world.phase_kernel import nominal_partition_pair_tables

STATIC_MATERIAL_INFORMATION_VERSION = "chemworld-static-material-information-1.1"
CRYSTALLIZATION_STATIC_MATERIAL_INFORMATION_VERSION = (
    "chemworld-static-crystallization-material-information-1.0"
)
REACTION_STATIC_MATERIAL_INFORMATION_VERSION = "chemworld-static-reaction-material-information-1.0"
PARTITION_STATIC_MATERIAL_INFORMATION_VERSION = (
    "chemworld-static-partition-material-information-1.0"
)
STATIC_MATERIAL_INFORMATION_OPAQUE = "opaque_codes"
STATIC_MATERIAL_INFORMATION_NOMINAL = "anonymous_nominal_properties"
STATIC_MATERIAL_INFORMATION_SHUFFLED = "anonymous_shuffled_properties"
STATIC_MATERIAL_INFORMATION_MISINDEXED = "anonymous_misindexed_properties"
STATIC_MATERIAL_INFORMATION_MODES = frozenset(
    {
        STATIC_MATERIAL_INFORMATION_MISINDEXED,
        STATIC_MATERIAL_INFORMATION_OPAQUE,
        STATIC_MATERIAL_INFORMATION_NOMINAL,
        STATIC_MATERIAL_INFORMATION_SHUFFLED,
    }
)
_ELECTROCHEMICAL_TASK_ID = "electrochemical-conversion"
_CRYSTALLIZATION_TASK_ID = "reaction-to-crystallization"
_DISTILLATION_TASK_ID = "reaction-to-distillation"
_PARTITION_TASK_ID = "partition-discovery"
_FLOW_TASK_ID = "flow-reaction-optimization"
_REACTION_SAFETY_TASK_ID = "reaction-safety-constrained"
_GENERIC_REACTION_PRIOR_TASK_IDS = frozenset(
    {
        _DISTILLATION_TASK_ID,
        _FLOW_TASK_ID,
        _REACTION_SAFETY_TASK_ID,
    }
)
_CONTROLLED_MATERIAL_FIELDS = ("electrolyte_profile", "solvent")
_CONTROLLED_MATERIAL_FIELDS_BY_TASK = {
    _ELECTROCHEMICAL_TASK_ID: _CONTROLLED_MATERIAL_FIELDS,
    _CRYSTALLIZATION_TASK_ID: ("catalyst", "solvent"),
    _DISTILLATION_TASK_ID: ("catalyst", "solvent"),
    _FLOW_TASK_ID: ("catalyst", "solvent"),
    _PARTITION_TASK_ID: ("solvent", "extractant"),
    _REACTION_SAFETY_TASK_ID: ("catalyst", "solvent"),
}
_AUDITED_NOMINAL_TASK_IDS = frozenset(_CONTROLLED_MATERIAL_FIELDS_BY_TASK)


def anonymous_electrochemical_material_catalog() -> dict[str, Any]:
    """Return the identity-free material catalog for electrochemical agents."""

    solvents = [
        {
            "index": index,
            "anonymous_material_id": f"solvent-S{index}",
            "display_name": f"solvent-S{index}",
            "identity_kind": "anonymous_benchmark_solvent_medium",
            "reference_status": "no_real_material_identity_claimed",
        }
        for index in range(len(SOLVENTS))
    ]
    electrolyte_profiles = [
        {
            "index": index,
            "anonymous_material_id": f"electrolyte-E{index}",
            "display_name": f"electrolyte-E{index}",
            "identity_kind": "anonymous_benchmark_electrolyte_formulation",
            "reference_status": "no_real_material_identity_claimed",
        }
        for index in range(len(ELECTROLYTE_PROFILES))
    ]
    return {
        "catalog_version": "chemworld-public-electrochemical-materials-1.0",
        "presentation": "anonymous_material_ids",
        "solvents": solvents,
        "electrolyte_profiles": electrolyte_profiles,
        "reagent": {
            "canonical_id": "limiting_reagent",
            "display_name": "Anonymous limiting reagent",
            "identity_kind": "mechanism_role",
            "reference_status": "mechanism_specific_not_a_real_identity",
        },
        "interpretation_policy": (
            "The solvent and electrolyte IDs are benchmark-only labels. They do not "
            "identify real substances or formulations, and their action indices reveal "
            "no hidden world-specific material residuals."
        ),
    }


def anonymous_partition_material_catalog() -> dict[str, Any]:
    """Expose independent S/X labels without attaching real solvent identities."""
    return {
        "catalog_version": "chemworld-public-partition-materials-1.0",
        "presentation": "anonymous_material_ids",
        **{
            key: [
                {
                    "index": index,
                    "anonymous_material_id": f"{role}-{prefix}{index}",
                    "display_name": f"{role}-{prefix}{index}",
                    "identity_kind": f"anonymous_benchmark_{role}",
                    "reference_status": "no_real_material_identity_claimed",
                }
                for index in range(len(SOLVENTS))
            ]
            for key, role, prefix in (
                ("solvents", "solvent", "S"),
                ("extractants", "extractant", "X"),
            )
        },
        "reagent": {
            "canonical_id": "limiting_reagent",
            "display_name": "Anonymous limiting reagent",
            "identity_kind": "mechanism_role",
            "reference_status": "mechanism_specific_not_a_real_identity",
        },
        "interpretation_policy": (
            "Solvent S and extractant X are independent benchmark labels, not real "
            "substance identities. Equal indices across the two fields imply no shared "
            "identity or property. The catalog provides no material-property estimates."
        ),
    }


def anonymous_crystallization_material_catalog() -> dict[str, Any]:
    """Use the same anonymous C/S identities as the crystallization prior dossier."""
    return {
        "catalog_version": "chemworld-public-crystallization-materials-1.0",
        "presentation": "anonymous_material_ids",
        **{
            key: [
                {
                    "index": index,
                    "anonymous_material_id": f"{role}-{prefix}{index}",
                    "display_name": f"{role}-{prefix}{index}",
                    "identity_kind": f"anonymous_benchmark_{role}",
                    "reference_status": "no_real_material_identity_claimed",
                }
                for index in range(count)
            ]
            for key, role, prefix, count in (
                ("solvents", "solvent", "S", len(SOLVENTS)),
                ("catalysts", "catalyst", "C", len(CATALYSTS)),
            )
        },
        "reagent": {
            "canonical_id": "limiting_reagent",
            "display_name": "Anonymous limiting reagent",
            "identity_kind": "mechanism_role",
            "reference_status": "mechanism_specific_not_a_real_identity",
        },
        "interpretation_policy": (
            "Catalyst C and solvent S are benchmark-only identities, not real substances. "
            "Their action indices reveal no hidden properties or world-specific residuals."
        ),
    }


def public_material_catalog(*, task_id: str | None = None) -> dict[str, Any]:
    """Return names, reference status, and interpretation policy for task materials."""

    if task_id == _ELECTROCHEMICAL_TASK_ID:
        return anonymous_electrochemical_material_catalog()
    if task_id == _PARTITION_TASK_ID:
        return anonymous_partition_material_catalog()
    if task_id == _CRYSTALLIZATION_TASK_ID:
        return anonymous_crystallization_material_catalog()
    registry = curated_component_registry()
    solvents: list[dict[str, Any]] = []
    for index, solvent_id in enumerate(SOLVENTS):
        try:
            component = registry.resolve(solvent_id)
        except KeyError:
            solvents.append(
                {
                    "index": index,
                    "canonical_id": solvent_id,
                    "display_name": solvent_id.replace("_", " ").title(),
                    "identity_kind": "real_named_component",
                    "reference_status": "identity_only_not_in_curated_property_subset",
                    "formula": None,
                    "cas_number": None,
                    "runtime_coupling": "categorical_benchmark_effect",
                }
            )
            continue
        solvents.append(
            {
                "index": index,
                "canonical_id": component.identifier,
                "display_name": str(
                    component.metadata.get("display_name") or component.identifier.title()
                ),
                "identity_kind": "real_reference_component",
                "reference_status": "curated_identity_and_local_property_correlations",
                "formula": component.hill_formula,
                "cas_number": component.cas_number,
                "runtime_coupling": "categorical_benchmark_effect",
            }
        )
    catalysts = [
        {
            "index": index,
            "canonical_id": catalyst_id,
            "display_name": f"Catalyst {chr(ord('A') + index)}",
            "identity_kind": "anonymous_benchmark_formulation",
            "reference_status": "no_real_material_identity_claimed",
            "runtime_coupling": "latent_categorical_activity_profile",
        }
        for index, catalyst_id in enumerate(CATALYSTS)
    ]
    electrolyte_profiles = [
        {
            "index": index,
            "canonical_id": profile_id,
            "display_name": profile_id.replace("_", " ").title(),
            "identity_kind": "anonymous_aqueous_electrolyte_formulation",
            "reference_status": "bounded_benchmark_formulation_not_a_named_product",
            "runtime_coupling": (
                "conductivity_diffusion_layer_acid_base_and_precipitation_profile"
            ),
        }
        for index, profile_id in enumerate(ELECTROLYTE_PROFILES)
    ]
    return {
        "catalog_version": "chemworld-public-materials-0.2",
        "solvents": solvents,
        "catalysts": catalysts,
        "electrolyte_profiles": electrolyte_profiles,
        "reagent": {
            "canonical_id": "limiting_reagent",
            "display_name": "Anonymous limiting reagent",
            "identity_kind": "mechanism_role",
            "reference_status": "mechanism_specific_not_a_real_identity",
        },
        "interpretation_policy": (
            "Real solvent names identify selectable materials, but current reaction-task "
            "kinetic effects are calibrated categorical benchmark effects rather than "
            "predictions from the curated property correlations. Catalysts and reaction "
            "species remain anonymous; do not infer a real catalyst or named synthesis."
        ),
    }


def material_choice_labels(
    field: str,
    *,
    task_id: str | None = None,
) -> dict[str, str]:
    """Map stable numeric action values to honest user-facing labels."""

    catalog = public_material_catalog(task_id=task_id)
    key = (
        "extractants"
        if field == "extractant" and task_id == _PARTITION_TASK_ID
        else "solvents"
        if field in {"solvent", "extractant"}
        else "catalysts"
        if field == "catalyst"
        else "electrolyte_profiles"
        if field == "electrolyte_profile"
        else None
    )
    if key is None:
        return {}
    # Some task-specific catalogs intentionally omit fields that are not
    # disclosed or controllable in that task (for example catalysts in the
    # anonymous electrochemical dossier).  A schema for an otherwise valid
    # operation must remain serializable even when such a catalog key is
    # absent; absence means "no public labels", not a catalog failure.
    if key not in catalog:
        return {}
    labels: dict[str, str] = {}
    for item in catalog[key]:
        if task_id in {_ELECTROCHEMICAL_TASK_ID, _PARTITION_TASK_ID, _CRYSTALLIZATION_TASK_ID}:
            labels[str(item["index"])] = str(item["anonymous_material_id"])
            continue
        reference = str(item["reference_status"])
        suffix = "reference identity" if reference.startswith("curated_") else "benchmark"
        formula = f" · {item['formula']}" if item.get("formula") else ""
        labels[str(item["index"])] = f"{item['display_name']}{formula} · {suffix}"
    return labels


def action_material_display(action: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe action copy annotated with public material names."""

    display = dict(action)
    catalog = public_material_catalog()
    for field, key in (
        ("solvent", "solvents"),
        ("extractant", "solvents"),
        ("catalyst", "catalysts"),
        ("electrolyte_profile", "electrolyte_profiles"),
    ):
        value = action.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        choices = catalog[key]
        if 0 <= value < len(choices):
            display[f"{field}_name"] = choices[value]["display_name"]
    return display


def normalize_static_material_information_config(
    config: Mapping[str, Any] | None,
    *,
    task_ids: Sequence[str],
    material_family_id: object | None = None,
) -> dict[str, Any]:
    """Validate one S0 information condition without reading hidden world state."""

    if config is None:
        return {"mode": STATIC_MATERIAL_INFORMATION_OPAQUE}
    if not isinstance(config, Mapping):
        raise ValueError("material_information must be an object")
    unknown = set(config) - {"mode", "target_field", "descriptor_permutation"}
    if unknown:
        raise ValueError(f"material_information has unsupported fields: {sorted(unknown)}")
    mode = config.get("mode")
    if mode not in STATIC_MATERIAL_INFORMATION_MODES:
        raise ValueError(
            f"material_information.mode must be one of {sorted(STATIC_MATERIAL_INFORMATION_MODES)}"
        )
    task_set = set(task_ids)
    if mode != STATIC_MATERIAL_INFORMATION_OPAQUE and (
        len(task_set) != 1 or next(iter(task_set)) not in _AUDITED_NOMINAL_TASK_IDS
    ):
        raise ValueError("nominal material properties require exactly one audited prior task")
    if mode == STATIC_MATERIAL_INFORMATION_SHUFFLED and task_set != {_ELECTROCHEMICAL_TASK_ID}:
        raise ValueError(
            "shuffled material properties are frozen only for electrochemical legacy studies"
        )
    if mode != STATIC_MATERIAL_INFORMATION_OPAQUE and task_set == {_ELECTROCHEMICAL_TASK_ID}:
        family_id = normalize_electrochemical_material_family(material_family_id)
        if family_id != NOMINAL_PRIOR_MATERIAL_FAMILY:
            raise ValueError(
                "nominal material information requires the nominal-prior material family"
            )
    if mode != STATIC_MATERIAL_INFORMATION_OPAQUE and task_set == {_CRYSTALLIZATION_TASK_ID}:
        family_id = normalize_crystallization_material_family(material_family_id)
        if family_id != REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY:
            raise ValueError(
                "nominal crystallization material information requires the "
                "reaction-crystallization latent material family"
            )
    raw_permutation = config.get("descriptor_permutation")
    raw_target_field = config.get("target_field")
    if mode == STATIC_MATERIAL_INFORMATION_MISINDEXED:
        task_id = next(iter(task_set))
        controlled_fields = _CONTROLLED_MATERIAL_FIELDS_BY_TASK[task_id]
        if not isinstance(raw_target_field, str) or raw_target_field not in controlled_fields:
            raise ValueError(
                "misindexed material information requires target_field to be one of "
                f"{list(controlled_fields)}"
            )
        if not isinstance(raw_permutation, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in raw_permutation
        ):
            raise ValueError(
                "misindexed material information requires descriptor_permutation "
                "to be an integer list"
            )
        expected = list(range(4))
        if sorted(raw_permutation) != expected:
            raise ValueError(
                f"misindexed descriptor_permutation must be a permutation of {expected}"
            )
        moved = [index for index, source in enumerate(raw_permutation) if index != source]
        if (
            len(moved) != 2
            or raw_permutation[moved[0]] != moved[1]
            or raw_permutation[moved[1]] != moved[0]
        ):
            raise ValueError(
                "misindexed descriptor_permutation must be exactly one two-row transposition"
            )
        return {
            "mode": STATIC_MATERIAL_INFORMATION_MISINDEXED,
            "target_field": raw_target_field,
            "descriptor_permutation": list(raw_permutation),
        }
    if mode != STATIC_MATERIAL_INFORMATION_SHUFFLED:
        if raw_permutation is not None or raw_target_field is not None:
            raise ValueError(
                "target_field and descriptor_permutation are allowed only for "
                "misindexed or shuffled material information"
            )
        return {"mode": str(mode)}
    if raw_target_field is not None:
        raise ValueError("target_field is not allowed for anonymous_shuffled_properties")
    if not isinstance(raw_permutation, Mapping) or set(raw_permutation) != set(
        _CONTROLLED_MATERIAL_FIELDS
    ):
        raise ValueError(
            "shuffled material information requires permutations for electrolyte_profile "
            "and solvent"
        )
    normalized_permutation: dict[str, list[int]] = {}
    for field in _CONTROLLED_MATERIAL_FIELDS:
        values = raw_permutation[field]
        if not isinstance(values, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in values
        ):
            raise ValueError(f"descriptor_permutation.{field} must be an integer list")
        expected = list(range(4))
        if sorted(values) != expected:
            raise ValueError(f"descriptor_permutation.{field} must be a permutation of {expected}")
        if any(index == source for index, source in enumerate(values)):
            raise ValueError(f"descriptor_permutation.{field} must be a derangement")
        normalized_permutation[field] = list(values)
    return {
        "mode": STATIC_MATERIAL_INFORMATION_SHUFFLED,
        "descriptor_permutation": normalized_permutation,
    }


def static_material_information_dossier(
    config: Mapping[str, Any] | None,
    *,
    task_id: str,
    material_family_id: object | None = None,
) -> dict[str, Any] | None:
    """Build the model-visible anonymous property dossier for one S0 task."""

    normalized = normalize_static_material_information_config(
        config,
        task_ids=(task_id,),
        material_family_id=material_family_id,
    )
    mode = normalized["mode"]
    if mode == STATIC_MATERIAL_INFORMATION_OPAQUE:
        return None
    permutations = {field: list(range(4)) for field in _CONTROLLED_MATERIAL_FIELDS_BY_TASK[task_id]}
    if mode == STATIC_MATERIAL_INFORMATION_SHUFFLED:
        permutations.update(normalized["descriptor_permutation"])
    elif mode == STATIC_MATERIAL_INFORMATION_MISINDEXED:
        permutations[normalized["target_field"]] = normalized["descriptor_permutation"]
    if task_id == _CRYSTALLIZATION_TASK_ID:
        return _crystallization_material_information_dossier(
            material_family_id,
            permutations=permutations,
        )
    if task_id == _PARTITION_TASK_ID:
        return _partition_material_information_dossier(permutations=permutations)
    if task_id in _GENERIC_REACTION_PRIOR_TASK_IDS:
        return _reaction_material_information_dossier(permutations=permutations)
    electrochemical_permutations = {
        field: permutations[field] for field in _CONTROLLED_MATERIAL_FIELDS
    }
    family = electrochemical_material_family(material_family_id)
    electrolyte_choices = []
    for action_value, source_index in enumerate(
        electrochemical_permutations["electrolyte_profile"]
    ):
        row = family.electrolyte_profiles[source_index]
        electrolyte_choices.append(
            {
                "action_value": action_value,
                "anonymous_material_id": f"electrolyte-E{action_value}",
                "nominal_properties": {
                    "bulk_conductivity_S_m": row["electrolyte_conductivity_S_m"],
                    "diffusivity_m2_s": row["diffusivity_m2_s"],
                    "diffusion_layer_thickness_mm": (1000.0 * row["diffusion_layer_thickness_m"]),
                    "double_layer_capacitance_F_m2": (row["double_layer_capacitance_F_m2"]),
                    "acid_concentration_mol_L": row["acid_concentration_mol_L"],
                    "supporting_electrolyte_concentration_mol_L": (
                        row["supporting_electrolyte_concentration_mol_L"]
                    ),
                    "precipitating_salt_concentration_mol_L": (
                        row["precipitating_salt_concentration_mol_L"]
                    ),
                    "acid_pKa": row["electrolyte_acid_pka"],
                    "precipitation_log10_Ksp": math.log10(row["electrolyte_ksp"]),
                    "standard_potential_shift_V": row["standard_potential_shift_V"],
                },
            }
        )
    solvent_choices = []
    for action_value, source_index in enumerate(electrochemical_permutations["solvent"]):
        row = family.solvent_profiles[source_index]
        solvent_choices.append(
            {
                "action_value": action_value,
                "anonymous_material_id": f"solvent-S{action_value}",
                "nominal_properties": {
                    "relative_conductivity": row["conductivity_multiplier"],
                    "relative_diffusivity": row["diffusivity_multiplier"],
                    "relative_double_layer_capacitance": row["capacitance_multiplier"],
                    "relative_proton_activity": row["proton_activity_multiplier"],
                    "relative_solubility_product": row["ksp_multiplier"],
                    "standard_potential_shift_V": row["standard_potential_shift_V"],
                    "relative_cost_index": row["relative_cost_index"],
                },
            }
        )
    return {
        "contract_version": STATIC_MATERIAL_INFORMATION_VERSION,
        "presentation": "anonymous_material_ids_with_nominal_properties",
        "identity_policy": (
            "The IDs are benchmark-only labels and do not identify real solvents, "
            "electrolytes, products, or commercial formulations."
        ),
        "property_scope": {
            "electrolyte_profile": (
                "Declared nominal electrolyte properties before bounded world residuals "
                "and solvent coupling."
            ),
            "solvent": (
                "Declared nominal solvent factors before bounded world residuals and "
                "electrolyte coupling."
            ),
        },
        "residual_policy": (
            "World-specific transport, solvation, and interface-kinetics latent factors "
            "generate bounded correlated residuals. They are hidden and fixed for the "
            "entire campaign."
        ),
        "cell_fixture_policy": (
            "Electrode area, electrode gap, and base contact resistance are sampled once "
            "per world and remain identical for every material and experiment in that "
            "campaign. They may differ in another world and are not material descriptors."
        ),
        "interpretation_policy": (
            "Treat these foundational properties as mechanistic prior evidence, not as "
            "an objective ranking or a complete response law. Faradaic efficiency and "
            "product selectivity are intentionally not supplied as material descriptors."
        ),
        "choices": {
            "electrolyte_profile": electrolyte_choices,
            "solvent": solvent_choices,
        },
    }


def _positive_geometric_mean(values: Sequence[float]) -> float:
    numeric = tuple(float(value) for value in values)
    if not numeric or any(value <= 0.0 or not math.isfinite(value) for value in numeric):
        raise ValueError("nominal material multipliers must be finite and positive")
    return math.exp(sum(math.log(value) for value in numeric) / len(numeric))


def _log_variability(values: Sequence[float]) -> float:
    numeric = tuple(float(value) for value in values)
    center = sum(math.log(value) for value in numeric) / len(numeric)
    return math.sqrt(sum((math.log(value) - center) ** 2 for value in numeric) / len(numeric))


def _reaction_panel_properties(values: Sequence[float]) -> dict[str, float]:
    numeric = tuple(float(value) for value in values)
    return {
        "reference_panel_activity_geomean": _positive_geometric_mean(numeric),
        "reference_panel_activity_floor": min(numeric),
        "reference_panel_activity_ceiling": max(numeric),
        "reference_panel_log_variability": _log_variability(numeric),
    }


def _crystallization_material_information_dossier(
    material_family_id: object | None,
    *,
    permutations: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Build an anonymous, nominal-only catalyst/solvent dossier.

    The published values are derived only from the frozen family-level nominal
    profiles. They never read the world-specific residual instance, active
    mechanism, realized score, or any experiment observation.
    """

    family = crystallization_material_family(material_family_id)
    catalyst_choices = []
    for action_value, source_index in enumerate(permutations["catalyst"]):
        row = family.catalyst_profiles[source_index]
        catalyst_choices.append(
            {
                "action_value": action_value,
                "anonymous_material_id": f"catalyst-C{action_value}",
                "nominal_properties": _reaction_panel_properties(row["reaction_multipliers"]),
            }
        )
    solvent_choices = []
    for action_value, source_index in enumerate(permutations["solvent"]):
        row = family.solvent_profiles[source_index]
        reaction_properties = _reaction_panel_properties(row["reaction_multipliers"])
        solvent_choices.append(
            {
                "action_value": action_value,
                "anonymous_material_id": f"solvent-S{action_value}",
                "nominal_properties": {
                    **reaction_properties,
                    "relative_solubility": float(row["solubility_multiplier"]),
                    "relative_nucleation_tendency": float(row["nucleation_multiplier"]),
                    "relative_crystal_growth": float(row["growth_multiplier"]),
                    "relative_impurity_occlusion": float(row["impurity_occlusion_multiplier"]),
                },
            }
        )
    if len(catalyst_choices) != 4 or len(solvent_choices) != 4:
        raise ValueError(
            "crystallization nominal material dossier requires four catalyst "
            "and four solvent choices"
        )
    return {
        "contract_version": CRYSTALLIZATION_STATIC_MATERIAL_INFORMATION_VERSION,
        "presentation": "anonymous_material_ids_with_nominal_properties",
        "identity_policy": (
            "Catalyst and solvent IDs are benchmark-only labels and do not "
            "identify real substances, formulations, or synthesis conditions."
        ),
        "property_scope": {
            "catalyst": (
                "Nominal aggregate activity across an anonymous reference-reaction "
                "panel before the fixed-world catalyst residual."
            ),
            "solvent": (
                "Nominal aggregate reference-reaction activity plus relative "
                "solubility, nucleation, growth, and impurity-occlusion tendencies "
                "before fixed-world residuals."
            ),
        },
        "residual_policy": (
            "World-specific catalyst and solvent residual multipliers are hidden, "
            "sampled once per world, and fixed for the entire campaign."
        ),
        "interpretation_policy": (
            "Treat these values as incomplete mechanistic prior evidence. They do "
            "not reveal the active hidden reaction family, realized response law, "
            "objective score, optimal recipe, or world-specific residuals."
        ),
        "choices": {
            "catalyst": catalyst_choices,
            "solvent": solvent_choices,
        },
    }


def _reaction_material_information_dossier(
    *,
    permutations: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Build an incomplete generic reaction catalyst/solvent prior dossier."""

    catalyst_choices = []
    for action_value, source_index in enumerate(permutations["catalyst"]):
        catalyst_choices.append(
            {
                "action_value": action_value,
                "anonymous_material_id": f"catalyst-C{action_value}",
                "nominal_properties": _reaction_panel_properties(
                    REACTION_NOMINAL_CATALYST_ACTIVITY_PROFILES[source_index]
                ),
            }
        )
    solvent_choices = []
    for action_value, source_index in enumerate(permutations["solvent"]):
        solvent_choices.append(
            {
                "action_value": action_value,
                "anonymous_material_id": f"solvent-S{action_value}",
                "nominal_properties": _reaction_panel_properties(
                    REACTION_NOMINAL_SOLVENT_ACTIVITY_PROFILES[source_index]
                ),
            }
        )
    return {
        "contract_version": REACTION_STATIC_MATERIAL_INFORMATION_VERSION,
        "presentation": "anonymous_material_ids_with_nominal_properties",
        "identity_policy": (
            "Catalyst and solvent IDs are benchmark-only labels and do not identify real "
            "substances, formulations, or synthesis conditions."
        ),
        "property_scope": {
            "catalyst": (
                "Nominal aggregate activity across an anonymous reference-reaction panel "
                "before the fixed-world catalyst residual."
            ),
            "solvent": (
                "Nominal aggregate activity across the same anonymous reference-reaction "
                "panel before the fixed-world solvent residual."
            ),
        },
        "residual_policy": (
            "World-specific catalyst and solvent residual multipliers are hidden, sampled "
            "once per world, and fixed for the entire campaign."
        ),
        "interpretation_policy": (
            "Treat these aggregates as incomplete mechanistic prior evidence. They do not "
            "reveal the active reaction network, temperature or duration response, safety "
            "law, downstream separation law, objective score, or optimal recipe."
        ),
        "choices": {
            "catalyst": catalyst_choices,
            "solvent": solvent_choices,
        },
    }


def _partition_profile_properties(
    product_values: Sequence[float],
    impurity_values: Sequence[float],
) -> dict[str, float]:
    product = tuple(float(value) for value in product_values)
    impurity = tuple(float(value) for value in impurity_values)
    selectivity = tuple(
        product_value / impurity_value
        for product_value, impurity_value in zip(product, impurity, strict=True)
    )
    return {
        "partner_panel_product_distribution_geomean": _positive_geometric_mean(product),
        "partner_panel_impurity_distribution_geomean": _positive_geometric_mean(impurity),
        "partner_panel_selectivity_geomean": _positive_geometric_mean(selectivity),
        "partner_panel_selectivity_ceiling": max(selectivity),
        "partner_panel_selectivity_log_variability": _log_variability(selectivity),
    }


def _partition_material_information_dossier(
    *,
    permutations: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Build marginal nominal profiles while withholding the pair interaction table."""

    tables = nominal_partition_pair_tables()
    product_rows = tables["product_distribution_coefficients"]
    impurity_rows = tables["impurity_distribution_coefficients"]
    solvent_choices = []
    for action_value, source_index in enumerate(permutations["solvent"]):
        solvent_choices.append(
            {
                "action_value": action_value,
                "anonymous_material_id": f"solvent-S{action_value}",
                "nominal_properties": _partition_profile_properties(
                    product_rows[source_index],
                    impurity_rows[source_index],
                ),
            }
        )
    product_columns = tuple(zip(*product_rows, strict=True))
    impurity_columns = tuple(zip(*impurity_rows, strict=True))
    extractant_choices = []
    for action_value, source_index in enumerate(permutations["extractant"]):
        extractant_choices.append(
            {
                "action_value": action_value,
                "anonymous_material_id": f"extractant-X{action_value}",
                "nominal_properties": _partition_profile_properties(
                    product_columns[source_index],
                    impurity_columns[source_index],
                ),
            }
        )
    return {
        "contract_version": PARTITION_STATIC_MATERIAL_INFORMATION_VERSION,
        "presentation": "anonymous_material_ids_with_nominal_properties",
        "identity_policy": (
            "Solvent and extractant IDs are independent benchmark-only labels and do not "
            "identify real materials. Equal codes across the two fields have no meaning."
        ),
        "property_scope": {
            "solvent": (
                "Marginal nominal distribution and selectivity tendencies aggregated across "
                "an anonymous extractant reference panel."
            ),
            "extractant": (
                "Marginal nominal distribution and selectivity tendencies aggregated across "
                "an anonymous solvent reference panel."
            ),
        },
        "residual_policy": (
            "The solvent-extractant interaction table, world-level coefficient transform, "
            "phase-volume response, and mixing response are not disclosed. The hidden law is "
            "fixed for the entire campaign."
        ),
        "interpretation_policy": (
            "Treat the marginal profiles as incomplete prior evidence. They cannot identify "
            "the best pair without controlled pairwise experiments and do not reveal the "
            "objective score or optimal process settings."
        ),
        "choices": {
            "solvent": solvent_choices,
            "extractant": extractant_choices,
        },
    }


__all__ = [
    "CRYSTALLIZATION_STATIC_MATERIAL_INFORMATION_VERSION",
    "PARTITION_STATIC_MATERIAL_INFORMATION_VERSION",
    "REACTION_STATIC_MATERIAL_INFORMATION_VERSION",
    "STATIC_MATERIAL_INFORMATION_MISINDEXED",
    "STATIC_MATERIAL_INFORMATION_MODES",
    "STATIC_MATERIAL_INFORMATION_NOMINAL",
    "STATIC_MATERIAL_INFORMATION_OPAQUE",
    "STATIC_MATERIAL_INFORMATION_SHUFFLED",
    "STATIC_MATERIAL_INFORMATION_VERSION",
    "action_material_display",
    "anonymous_electrochemical_material_catalog",
    "material_choice_labels",
    "normalize_static_material_information_config",
    "public_material_catalog",
    "static_material_information_dossier",
]
