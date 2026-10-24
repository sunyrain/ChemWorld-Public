"""Frozen 40C histories on the already declared synthetic 40B thermochemistry."""

from dataclasses import replace

from chemworld.physchem.aqueous_mineral_kinetics import (
    AqueousMineralKineticsSpec,
    MineralRateLaw,
    prepare_aqueous_mineral_state,
)
from chemworld.physchem.mineral_kinetics import MineralGeometry
from chemworld.physchem.solver_backend import ODESolverPolicy

from .aqueous40_mineral_cases import cases as mineral_cases
from .aqueous40_mineral_cases import inventory, specification


def cases():
    base = mineral_cases()[0]
    updates = (
        {"id": "partial-dissolution", "solid_na": 0.04, "duration_s": 3000},
        {"id": "exhaustion", "na": 0.001, "solid_na": 0.002, "duration_s": 25000},
        {"id": "seeded-precipitation", "na": 0.05, "solid_na": 0.005, "duration_s": 4000},
        {"id": "no-seed", "na": 0.05, "duration_s": 4000},
        {"id": "common-ion", "two": True, "k": 0.03, "solid_na": 0.03, "duration_s": 3000},
        {
            "id": "two-mineral-competition",
            "two": True,
            "k": 0.001,
            "solid_na": 0.04,
            "solid_k": 0.001,
            "duration_s": 25000,
        },
        {
            "id": "fast-complexation",
            "na": 0.05,
            "association": True,
            "solid_na": 0.005,
            "duration_s": 4000,
        },
        {
            "id": "omitted-fast-reaction",
            "na": 0.05,
            "association": True,
            "omit_association": True,
            "complex": 0.003,
            "solid_na": 0.005,
            "duration_s": 4000,
        },
        {
            "id": "hydrated-mineral",
            "hydrate": True,
            "water_kg": 0.25,
            "solid_na": 0.04,
            "duration_s": 3000,
        },
        {"id": "warm-Arrhenius", "temperature_K": 315.0, "solid_na": 0.04, "duration_s": 3000},
        {"id": "pressure-standard-state", "pressure_Pa": 2e5, "solid_na": 0.04, "duration_s": 3000},
        {"id": "Davies", "family": "Davies", "solid_na": 0.04, "duration_s": 3000},
    )
    return [{**base, **values} for values in updates]


def inputs(case, strategy="BDF"):
    equilibrium = specification(case)
    policy = ODESolverPolicy(
        "declared-mineral-qualification-" + strategy,
        method=strategy,
        rtol=1e-8 if strategy == "BDF" else 2e-9,
        atol=1e-11 if strategy == "BDF" else 2e-12,
        dense_output=True,
    )
    spec = AqueousMineralKineticsSpec(
        equilibrium,
        tuple(MineralRateLaw(k, 1e-4, 12000.0) for k in equilibrium.mineral_ids),
        policy,
    )
    amounts = prepare_aqueous_mineral_state(
        spec,
        inventory(case, equilibrium),
        temperature_K=case["temperature_K"],
        pressure_Pa=case["pressure_Pa"],
    )
    geometry = {k: MineralGeometry(0.01, 0.01) for k in equilibrium.mineral_ids}
    return replace(spec), amounts, geometry


def observation_times(case):
    return tuple(case["duration_s"] * fraction / 5 for fraction in range(6))
