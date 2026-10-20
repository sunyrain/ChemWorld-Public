"""Explicit movement contract for the existing liquid and crystal models.

These fields are material properties stored by older services in equipment
settings. Routing extracts only these declarations, never device settings or
execution histories. Active services keep their existing single source of truth.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord

# sum: extensive state; equal: compatibility requirement; copy: equal values
# survive a mixture, unequal derived values are invalidated. Cohorts use their
# count/diameter rule and regenerate all CSD summaries at the runtime boundary.
MATERIAL_FIELDS = {
    "batch_reactor": {
        "solvent": "copy",
        "solvent_volume_L": "sum",
        "catalyst": "catalyst",
        "catalyst_amount_mol": "sum",
        "reaction_stopped": "equal",
    },
    "crystallizer": {
        "seed_target_mol": "sum",
        "dissolved_seed_target_mol": "sum",
        "effective_seed_target_mol": "sum",
        "crystallized_from_solution_mol": "sum",
        "crystal_seeded": "copy",
        "population_cohorts": "cohorts",
        "population_constituents": "cohort_amounts",
        "csd_d50_m": "copy",
        "csd_d10_m": "copy",
        "csd_d90_m": "copy",
        "csd_cv": "copy",
        "csd_fines_number_fraction": "copy",
        "csd_total_particle_count": "sum",
        "csd_number_moment_0": "sum",
        "csd_length_moment_1_m": "sum",
        "csd_area_moment_2_m2": "sum",
        "csd_volume_moment_3_m3": "sum",
        "crystallization_model_id": "equal",
        "solubility_model_id": "equal",
        "kinetics_model_id": "equal",
        "impurity_occlusion_law_id": "equal",
    },
    "crystal_filter": {
        "crystals_filtered": "equal",
        "filtered_product_mol": "sum",
        "retained_seed_mol": "sum",
        "filtered_product_from_solution_mol": "sum",
        "filtered_impurity_mol": "sum",
        "filter_purity": "copy",
    },
}

MATERIAL_METRICS = {
    **dict.fromkeys(
        (
            "pre_separation_product_mol",
            "seed_target_mol",
            "crystallized_from_solution_mol",
            "retained_seed_mol",
            "filtered_product_from_solution_mol",
            "crystal_redissolved_target_mol",
        ),
        "sum",
    ),
    **dict.fromkeys(
        ("reaction_chemistry_stopped", "crystal_yield", "seed_excluded_recovery", "crystal_purity"),
        "copy",
    ),
}


def extract_material_properties(equipment: EquipmentLedger | None) -> dict[str, dict[str, Any]]:
    records = (equipment or EquipmentLedger()).equipment
    return {
        name: deepcopy(
            {key: value for key, value in records[name].settings.items() if key in fields}
        )
        for name, fields in MATERIAL_FIELDS.items()
        if name in records
    }


def install_material_properties(
    equipment: EquipmentLedger | None,
    properties: dict[str, dict[str, Any]],
    vessel_id: str,
) -> EquipmentLedger:
    records = dict((equipment or EquipmentLedger()).equipment)
    for name, fields in MATERIAL_FIELDS.items():
        if name not in records and name not in properties:
            continue
        record = records.get(name) or EquipmentRecord(
            name, "solid_liquid_filter" if name == "crystal_filter" else name, vessel_id
        )
        settings = {k: v for k, v in record.settings.items() if k not in fields}
        settings.update(properties.get(name, {}))
        records[name] = replace(record, settings=settings)
    return EquipmentLedger(
        {
            k: replace(v, status="stale") if k.startswith("instrument:") else v
            for k, v in records.items()
        }
    )


def scale_fields(values: dict[str, Any], rules: dict[str, str], fraction: float) -> dict[str, Any]:
    result = {}
    for key, value in values.items():
        rule = rules[key]
        if rule == "sum":
            result[key] = value * fraction
        elif rule == "cohorts":
            result[key] = [[n * fraction, d] for n, d in value]
        elif rule == "cohort_amounts":
            result[key] = [[n * fraction for n in row] for row in value]
        else:
            result[key] = deepcopy(value)
    return result


def merge_fields(
    left: dict[str, Any], right: dict[str, Any], rules: dict[str, str]
) -> dict[str, Any]:
    result = {}
    for key in left.keys() | right.keys():
        rule = rules[key]
        a, b = left.get(key), right.get(key)
        if rule == "sum":
            result[key] = left.get(key, 0) + right.get(key, 0)
        elif rule in {"cohorts", "cohort_amounts"}:
            if rule == "cohort_amounts" and any(
                values.get("population_cohorts") and key not in values for values in (left, right)
            ):
                raise ValueError("mixing crystal cohorts needs consistent constituent provenance")
            result[key] = deepcopy([*left.get(key, ()), *right.get(key, ())])
        elif rule == "catalyst":
            if (
                left.get("catalyst_amount_mol", 0) > 0
                and right.get("catalyst_amount_mol", 0) > 0
                and a != b
            ):
                raise ValueError("Mixing requires matching catalyst")
            result[key] = b if right.get("catalyst_amount_mol", 0) > 0 else a
        elif rule == "equal":
            if a is not None and b is not None and a != b:
                raise ValueError(f"Mixing requires matching {key}")
            result[key] = deepcopy(a if a is not None else b)
        elif a == b:
            result[key] = deepcopy(a)
    return result
