"""Matched, physically closed EQ worlds for the abductive-inquiry experiment.

This module describes a new experiment using existing runtime interventions. It
does not alter the equilibrium solver or any historical experiment definition.
Private world labels and qualification queries must not enter research prompts.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, cast

from chemworld.physchem.equilibrium_mechanism import (
    EquilibriumMechanismFamily,
    solve_coupled_weak_acid_precipitation,
)

METRICS = ("pH_normalized", "acid_dissociation_fraction", "precipitation_signal")
ASSAY_STD = dict(zip(METRICS, (0.002, 0.006, 0.006), strict=True))
NOTE = "workstreams/flagship_tasks/WORK_II_ABDUCTIVE_COMPLETION_20261016.md"
BASE_PKA = 5.019404789086824


def build_worlds() -> list[dict[str, Any]]:
    """Return ten standard runner worlds with executable intervention contracts."""
    worlds = []
    for index, severity in enumerate((-0.8, -0.4, 0.0, 0.4, 0.8), 1):
        for member in ("A", "B"):
            interventions = []
            if severity:
                interventions.append(
                    {
                        "axis_id": "equilibrium.acid-base-constants",
                        "mode": "interpolation",
                        "severity": severity,
                    }
                )
            interventions.append(
                {
                    "axis_id": "equilibrium.mechanism-benchmark",
                    "mode": "extrapolation",
                    "severity": 1.0,
                }
            )
            if member == "B":
                interventions.append(
                    {
                        "axis_id": "equilibrium.aqueous-ion-pair-network",
                        "mode": "interpolation",
                        "severity": 0.4,
                    }
                )
            worlds.append(
                {
                    "world_id": f"EQ-P{index:02d}-{member}",
                    "world_seed": 0,
                    "world_interventions": interventions,
                    "private_authoring": {
                        "match_id": f"P{index:02d}",
                        "mechanism_id": member,
                        "mechanism_family": (
                            "direct_free_ion_precipitation"
                            if member == "A"
                            else "aqueous_ion_pair_intermediate"
                        ),
                        "effective_pka": BASE_PKA + 0.45 * severity,
                        "log10_ksp": -5.2,
                        "cation_fraction": 0.15,
                        "association_beta_L_per_mol": None if member == "A" else 8000.0,
                        "activity_coefficient_ratio": 1.0,
                    },
                }
            )
    return worlds


def recipe(
    query_id: str,
    concentration: float,
    volume: float,
    *,
    path: str = "bulk",
    coverage: str = "heldout",
) -> dict[str, Any]:
    """Compile a physical amount/volume intervention to ordinary public actions."""
    amount = concentration * volume

    def solvent(v):
        return {"operation": "add_solvent", "solvent": 0, "volume_L": v}

    def reagent(n):
        return {"operation": "add_reagent", "amount_mol": n}

    if path == "staged":
        actions = [
            solvent(volume / 2),
            reagent(amount / 2),
            solvent(volume / 2),
            reagent(amount / 2),
        ]
    elif path == "solvent_first":
        actions = [
            solvent(volume / 2),
            solvent(volume / 2),
            reagent(amount / 2),
            reagent(amount / 2),
        ]
    elif path == "bulk":
        actions = [solvent(volume), reagent(amount)]
    else:
        raise ValueError(f"Unknown addition path: {path}")
    actions += [{"operation": "terminate"}, {"operation": "measure", "instrument": "final_assay"}]
    return {
        "query_id": query_id,
        "coverage": [coverage, path],
        "actions": actions,
        "final_state": {
            "reagent_mol": amount,
            "solvent_L": volume,
            "concentration_mol_L": concentration,
        },
    }


def initial_queries() -> list[dict[str, Any]]:
    return [
        recipe(f"I{i:02d}", 6e-7 / volume, volume, coverage="initial_reference")
        for i, volume in enumerate((0.08, 0.075), 1)
    ]


def diagnostic_queries() -> list[dict[str, Any]]:
    concentrations = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 0.25, 0.5)
    return [
        recipe(f"D{i:02d}", c, 0.02, coverage="qualification_diagnostic")
        for i, c in enumerate(concentrations, 1)
    ]


def build_queries() -> list[dict[str, Any]]:
    queries = [
        recipe(f"Q{i:02d}", c, 0.02) for i, c in enumerate((0.002, 0.006, 0.02, 0.06, 0.2, 0.6), 1)
    ]
    queries += [
        recipe("Q07", 0.06, 0.04, coverage="scale_control"),
        recipe("Q08", 0.06, 0.01, coverage="scale_control"),
        recipe("Q09", 0.2, 0.02, path="staged", coverage="path_control"),
        recipe("Q10", 0.2, 0.02, path="solvent_first", coverage="path_control"),
        recipe("Q11", 0.6, 0.01, coverage="scale_control"),
        recipe("Q12", 0.6, 0.04, coverage="scale_control"),
    ]
    return queries


def build_config(base: Mapping[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(dict(base))
    config.update(
        {
            "schema_version": "chemworld-abductive-matched-worlds-1.1",
            "status": "development_fixed_candidate",
            "protocol": NOTE,
            "worlds": build_worlds(),
            "queries": build_queries(),
            "arms": ["Opaque"],
            "posttest_stages": ["K1", "Q", "K2"],
            "counts": {
                "worlds": 10,
                "matched_pairs": 5,
                "repeats": 3,
                "independent_source_sessions": 30,
                "source_batches_per_session": 12,
                "source_batches": 360,
                "posttests_per_session": 3,
                "posttests": 90,
                "reference_queries_per_world": 2,
                "reference_assays": 20,
                "prediction_queries_per_session": 12,
            },
            "source_resource_card": {
                "operation_attempt_limit": 180,
                "vessel_start_limit": 12,
                "final_assay_limit": 12,
                "nonfinal_instrument_use_limit": 12,
                "reagent_mol": 0.24,
                "solvent_L": 0.72,
                "process_time_limit_s": None,
            },
        }
    )
    return config


def solve_query(world: Mapping[str, Any], query: Mapping[str, Any]) -> dict[str, Any]:
    """Noise-free physical prediction; actual-operation agreement is qualified separately."""
    params = world["private_authoring"]
    final = query["final_state"]
    result = solve_coupled_weak_acid_precipitation(
        acid_total_mol=float(final["reagent_mol"]),
        volume_L=float(final["solvent_L"]),
        pka=float(params["effective_pka"]),
        log10_ksp=float(params["log10_ksp"]),
        mechanism_family=cast(EquilibriumMechanismFamily, params["mechanism_family"]),
        cation_fraction=float(params["cation_fraction"]),
        association_beta_L_per_mol=params["association_beta_L_per_mol"],
        activity_coefficient_ratio=float(params["activity_coefficient_ratio"]),
    )
    return {
        "world_id": world["world_id"],
        "query_id": query["query_id"],
        "metrics": {
            "pH_normalized": result.pH / 14.0,
            "acid_dissociation_fraction": result.acid_dissociation_fraction,
            "precipitation_signal": result.precipitation_signal,
        },
        "physical_state": asdict(result),
    }
