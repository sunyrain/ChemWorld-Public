"""Phase-resolved process modes; historical trajectories use their frozen runtime."""

from __future__ import annotations

from dataclasses import replace

from chemworld.foundation import (
    WorldState,
    equipment_settings,
    scale_species_initial_amounts,
    upsert_equipment_record,
)
from chemworld.foundation.state import PhaseLedger, selected_phase_id
from chemworld.foundation.state_helpers import equipment_setting_truth

FULL_PROCESS_CONTRACT = "phase-resolved-process-v1"
FULL_PROCESS_POPULATION_CONTRACT = "phase-resolved-process-v2"
FULL_PROCESS_THERMAL_CONTRACT = "phase-resolved-process-v3"
FULL_PROCESS_SEED_CONTRACT = "phase-resolved-process-v4"
FULL_PROCESS_FREE_RESEARCH_CONTRACT = "phase-resolved-process-v5"
CRYSTAL_RECOVERY_DEFINITION = (
    "crystal_yield is seed-excluded solid target product divided by the tracked initial "
    "reagent charge (both in moles), clipped to [0,1]; it is overall recovery from reagent "
    "charge, not crystallization-stage recovery relative to product before separation."
)
FULL_PROCESS_TASKS = frozenset(
    {
        "reaction-to-purification",
        "reaction-to-crystallization",
        "reaction-to-distillation",
    }
)


def active(state: WorldState) -> bool:
    return state.metadata.get("full_process_contract_id") in {
        FULL_PROCESS_CONTRACT,
        FULL_PROCESS_POPULATION_CONTRACT,
        FULL_PROCESS_THERMAL_CONTRACT,
        FULL_PROCESS_SEED_CONTRACT,
        FULL_PROCESS_FREE_RESEARCH_CONTRACT,
    }


def population_active(state: WorldState) -> bool:
    return state.metadata.get("full_process_contract_id") in {
        FULL_PROCESS_POPULATION_CONTRACT,
        FULL_PROCESS_THERMAL_CONTRACT,
        FULL_PROCESS_SEED_CONTRACT,
        FULL_PROCESS_FREE_RESEARCH_CONTRACT,
    }


def seed_provenance_active(state: WorldState) -> bool:
    return state.metadata.get("full_process_contract_id") in {
        FULL_PROCESS_SEED_CONTRACT,
        FULL_PROCESS_FREE_RESEARCH_CONTRACT,
    }


def unrecoverable_crystallization_result(state: WorldState, *, tolerance: float) -> bool:
    """A completed attempt whose solid inventory is below the operation threshold."""
    if (
        not equipment_setting_truth(state.equipment, "crystallizer", "execution_history")
        or state.phases is None
        or "solid" not in state.phases.phases
    ):
        return False
    return sum(state.phases.phases["solid"].species_amounts_mol.values()) <= tolerance


def population_settings(cohorts: tuple[tuple[float, float], ...]) -> dict:
    """Derive all particle statistics from the same retained count/diameter cohorts."""
    from chemworld.physchem.crystallization_units import _crystal_size_distribution, _CrystalCohort

    csd = _crystal_size_distribution([_CrystalCohort(n, d) for n, d in cohorts], 20.0e-6)
    return {
        "population_cohorts": [list(c) for c in cohorts],
        "csd_d50_m": csd.d50_m,
        "csd_d10_m": csd.d10_m,
        "csd_d90_m": csd.d90_m,
        "csd_cv": csd.coefficient_of_variation,
        "csd_fines_number_fraction": csd.fines_number_fraction,
        "csd_total_particle_count": csd.total_particle_count,
        "csd_number_moment_0": csd.number_moment_0,
        "csd_length_moment_1_m": csd.length_moment_1_m,
        "csd_area_moment_2_m2": csd.area_moment_2_m2,
        "csd_volume_moment_3_m3": csd.volume_moment_3_m3,
    }


def shrink_population(
    cohorts: tuple[tuple[float, float], ...], remaining_fraction: float
) -> tuple[tuple[float, float], ...]:
    """Equal radial recession; smallest particles disappear first, with exact mass closure."""
    if not cohorts or remaining_fraction <= 0:
        return ()
    target = sum(n * d**3 for n, d in cohorts) * remaining_fraction
    low, high = 0.0, max(d for _, d in cohorts)
    for _ in range(80):
        delta = (low + high) / 2
        mass = sum(n * max(d - delta, 0.0) ** 3 for n, d in cohorts)
        if mass > target:
            low = delta
        else:
            high = delta
    delta = (low + high) / 2
    return tuple((n, d - delta) for n, d in cohorts if d > delta)


def sample_domain(state: WorldState) -> tuple[str, ...]:
    """A selected liquid receiver, or the representative crystallizer slurry.

    Crystallizer phases share one analytical slurry preparation. They are not
    independently stored bottles in this bounded task. Liquid separation and
    distillation inventories are sampled independently once selected.
    """
    phases = {} if state.phases is None else state.phases.phases
    if {"solid", "mother_liquor"} <= phases.keys():
        return tuple(key for key in ("solid", "mother_liquor", "cake_liquor") if key in phases)
    selected = selected_phase_id(state.phases)
    if selected is None and "organic" in phases:
        selected = "organic"
    return (selected,) if selected is not None else tuple(phases)


def withdraw_sample(state: WorldState, volume_L: float) -> WorldState:
    """Conserve unselected inventories under the current measurement contract.

    Full-process recovery uses the original charge. The default analytical mode
    uses a sample-normalized reference so a withdrawal alone does not look like
    chemical conversion. These are measurement modes, not historical replay engines.
    """
    if state.phases is None:
        raise ValueError("phase-resolved sampling requires a typed inventory")
    domain = sample_domain(state)
    available = sum(state.phases.phases[k].volume_L for k in domain)
    if volume_L < 0.0 or volume_L > available:
        raise ValueError("insufficient volume in selected sampling domain")
    if volume_L == 0.0:
        return state
    fraction = volume_L / available
    phases = {
        key: replace(
            phase,
            volume_L=phase.volume_L * (1.0 - fraction),
            species_amounts_mol={
                s: a * (1.0 - fraction) for s, a in phase.species_amounts_mol.items()
            },
        )
        if key in domain
        else phase
        for key, phase in state.phases.phases.items()
    }
    phase_ledger = PhaseLedger(phases)
    equipment = state.equipment
    process = state.process
    if "solid" in domain:
        settings = equipment_settings(equipment, "crystallizer")
        # Charged seed mass remains the cumulative stock counter. Active seed
        # provenance and solution-grown inventory decrease with slurry sampling.
        updates: dict[str, float] = {
            key: float(settings[key]) * (1.0 - fraction)
            for key in (
                "seed_target_mol",
                "dissolved_seed_target_mol",
                "effective_seed_target_mol",
                "crystallized_from_solution_mol",
                "csd_total_particle_count",
                "csd_number_moment_0",
                "csd_length_moment_1_m",
                "csd_area_moment_2_m2",
                "csd_volume_moment_3_m3",
            )
            if key in settings
        }
        equipment = upsert_equipment_record(
            equipment,
            equipment_id="crystallizer",
            equipment_type="crystallizer",
            attached_vessel_id=state.vessel_id,
            status="sampled",
            settings={
                **updates,
                **(
                    population_settings(
                        tuple(
                            (n * (1.0 - fraction), d)
                            for n, d in settings.get("population_cohorts", ())
                        )
                    )
                    if population_active(state)
                    else {}
                ),
            },
        )
        if process is not None:
            metrics = dict(process.metrics)
            for key in (
                "seed_target_mol",
                "crystallized_from_solution_mol",
                "retained_seed_mol",
                "filtered_product_from_solution_mol",
            ):
                if key in metrics:
                    metrics[key] *= 1.0 - fraction
            process = replace(process, metrics=metrics)
    return state.replace(
        species_amounts=phase_ledger.total_amounts_mol(),
        phases=phase_ledger,
        volume_L=state.volume_L - volume_L,
        equipment=equipment,
        process=process,
        species=(
            state.species
            if active(state)
            else scale_species_initial_amounts(state.species, 1.0 - fraction)
        ),
        metadata={
            **state.metadata,
            "last_sample_domain": list(domain),
            "last_sample_fraction": fraction,
        },
    )
