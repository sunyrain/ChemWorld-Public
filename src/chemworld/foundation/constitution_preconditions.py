"""Operation precondition checks for the executable physical constitution."""

from __future__ import annotations

from math import pi
from typing import Any

from chemworld.foundation.state import (
    WorldState,
    equipment_settings,
    has_phase_system,
    instrument_completed,
    phases_are_settled,
)

# The public cooling-crystallization runtime fixes these four values in
# CrystallizationServices.cool_crystallize.  Keep the constitution conservative
# by a tiny floating-point margin so every advertised positive seed population
# is accepted by the downstream particle-count check.
_CRYSTALLIZATION_SOLUBILITY_MINIMUM_TEMPERATURE_K = 250.0
_CRYSTALLIZATION_SOLUBILITY_MAXIMUM_TEMPERATURE_K = 430.0
_CRYSTALLIZATION_RUNTIME_SEED_DIAMETER_M = 100.0e-6
_CRYSTALLIZATION_RUNTIME_CRYSTAL_DENSITY_KG_M3 = 1200.0
_CRYSTALLIZATION_MINIMUM_EFFECTIVE_SEED_PARTICLES = 10.0
_CRYSTALLIZATION_MINIMUM_EFFECTIVE_SEED_MASS_G = (
    _CRYSTALLIZATION_MINIMUM_EFFECTIVE_SEED_PARTICLES
    * _CRYSTALLIZATION_RUNTIME_CRYSTAL_DENSITY_KG_M3
    * pi
    / 6.0
    * _CRYSTALLIZATION_RUNTIME_SEED_DIAMETER_M**3
    * 1000.0
    * (1.0 + 1.0e-12)
)


def check_operation_preconditions(
    constitution: Any,
    operation_type: str,
    state: WorldState,
    payload: dict[str, Any],
) -> dict[str, bool]:
    has_volume = state.volume_L > constitution.tolerance
    has_material = sum(state.species_amounts.values()) > constitution.tolerance
    is_terminated = state.terminated
    instrument_id = str(payload.get("instrument", "hplc"))
    instrument = constitution.instruments.get(instrument_id)
    if instrument is None and instrument_id.isdigit():
        instruments = list(constitution.instruments.values())
        instrument = instruments[int(instrument_id) % len(instruments)]
    requires_terminated = bool(instrument.requires_terminated) if instrument is not None else False
    final_assay_done = instrument_completed(state.equipment, "final_assay")
    is_final_assay = (
        operation_type == "measure" and instrument is not None and instrument.id == "final_assay"
    )
    measurement_sample_available = operation_type != "measure" or (
        instrument is not None
        and state.volume_L + constitution.tolerance >= instrument.sample_volume_L
    )
    phase_system = has_phase_system(state.phases)
    phase_settled = phases_are_settled(state.phases)
    crystallized = (
        state.phases is not None
        and "solid" in state.phases.phases
        and sum(state.phases.phases["solid"].species_amounts_mol.values()) > constitution.tolerance
    )
    distillate_ready = (
        state.phases is not None
        and "distillate" in state.phases.phases
        and sum(state.phases.phases["distillate"].species_amounts_mol.values())
        > constitution.tolerance
    )
    flow_settings = equipment_settings(
        state.equipment, "flow_reactor", fields=("flow_rate_mL_min", "residence_time_s")
    )
    potential_settings = equipment_settings(
        state.equipment, "electrochemical_cell", fields=("potential_V", "current_mA")
    )
    reactor_settings = equipment_settings(
        state.equipment, "batch_reactor", fields=("reaction_advance_index",)
    )
    crystallizer_settings = equipment_settings(
        state.equipment, "crystallizer", fields=("seed_target_mol", "crystal_seed_mass_g")
    )
    flow_ready = {"flow_rate_mL_min", "residence_time_s"} <= set(flow_settings)
    potential_ready = {"potential_V", "current_mA"} <= set(potential_settings)
    crystallization_ready = (
        int(reactor_settings.get("reaction_advance_index", 0)) > 0
        or float(crystallizer_settings.get("seed_target_mol", 0.0)) > constitution.tolerance
    )
    seed_mass_g = float(crystallizer_settings.get("crystal_seed_mass_g", 0.0))
    crystallization_reference_temperature_valid = (
        _CRYSTALLIZATION_SOLUBILITY_MINIMUM_TEMPERATURE_K
        <= state.temperature_K
        <= _CRYSTALLIZATION_SOLUBILITY_MAXIMUM_TEMPERATURE_K
    )
    crystallization_seed_population_effective = (
        seed_mass_g <= 0.0 or seed_mass_g >= _CRYSTALLIZATION_MINIMUM_EFFECTIVE_SEED_MASS_G
    )
    phase_operations = {
        "add_extractant",
        "mix",
        "settle",
        "separate_phase",
        "wash",
        "dry",
        "concentrate",
        "transfer",
    }
    needs_volume = operation_type in {
        "heat",
        "wait",
        "sample",
        "quench",
        "measure",
        "add_extractant",
        "mix",
        "settle",
        "separate_phase",
        "wash",
        "dry",
        "concentrate",
        "transfer",
        "seed_crystals",
        "cool_crystallize",
        "filter_crystals",
        "evaporate",
        "distill",
        "collect_fraction",
        "set_potential",
        "run_flow",
        "electrolyze",
    }
    needs_material = operation_type in {
        "resuspend_crystals",
        "heat",
        "wait",
        "terminate",
        "add_extractant",
        "mix",
        "settle",
        "separate_phase",
        "wash",
        "dry",
        "concentrate",
        "transfer",
        "cool_crystallize",
        "filter_crystals",
        "distill",
        "collect_fraction",
        "set_potential",
        "run_flow",
        "electrolyze",
    }
    needs_not_terminated = operation_type in {
        "configure_instrument",
        "select_vessel",
        "route_material",
        "create_container",
        "transfer_material",
        "resuspend_crystals",
        "add_reagent",
        "add_component",
        "add_solvent",
        "add_catalyst",
        "heat",
        "wait",
        "sample",
        "quench",
        "add_phase",
        "add_extractant",
        "mix",
        "settle",
        "separate_phase",
        "wash",
        "dry",
        "concentrate",
        "transfer",
        "seed_crystals",
        "cool_crystallize",
        "filter_crystals",
        "evaporate",
        "distill",
        "collect_fraction",
        "set_flow_rate",
        "run_flow",
        "set_potential",
        "electrolyze",
        "terminate",
    }
    from chemworld.world.control_contract import CONTROL_OPERATIONS
    from chemworld.world.stream_contract import STREAM_OPERATIONS

    if operation_type in CONTROL_OPERATIONS:
        needs_not_terminated = True
        needs_volume = operation_type == "advance_control"
    if operation_type in STREAM_OPERATIONS:
        needs_not_terminated = True
        needs_volume = operation_type == "advance_flow"
    return {
        "instrument_available": operation_type != "measure" or instrument is not None,
        "has_volume": not needs_volume or has_volume,
        "has_material": not needs_material or has_material,
        "not_terminated": not needs_not_terminated or not state.terminated,
        "has_phase_system": operation_type not in phase_operations or phase_system,
        "phase_settled": operation_type != "separate_phase" or phase_settled,
        "measure_final_requires_terminated": operation_type != "measure"
        or not requires_terminated
        or is_terminated,
        "measure_after_termination_requires_final_assay": operation_type != "measure"
        or not is_terminated
        or is_final_assay,
        "measurement_sample_available": measurement_sample_available,
        "measure_final_not_repeated": not is_final_assay or not final_assay_done,
        "terminate_requires_material": operation_type != "terminate" or has_material,
        "filter_requires_crystallization": operation_type != "filter_crystals" or crystallized,
        "resuspend_requires_filtered_crystals": operation_type != "resuspend_crystals"
        or (
            bool(equipment_settings(state.equipment, "crystal_filter").get("crystals_filtered"))
            and state.phases is not None
            and "solid" in state.phases.phases
            and "mother_liquor" in state.phases.phases
            and "cake_liquor" in state.phases.phases
            and state.quenched
        ),
        "cool_crystallize_requires_reaction_or_seed": operation_type != "cool_crystallize"
        or crystallization_ready,
        "cool_crystallize_reference_temperature_in_solubility_domain": operation_type
        != "cool_crystallize"
        or crystallization_reference_temperature_valid,
        "cool_crystallize_seed_population_effective": operation_type != "cool_crystallize"
        or crystallization_seed_population_effective,
        "collect_fraction_requires_distillation": operation_type != "collect_fraction"
        or distillate_ready,
        "run_flow_requires_flow_setup": operation_type != "run_flow" or flow_ready,
        "electrolyze_requires_potential": operation_type != "electrolyze" or potential_ready,
    }


__all__ = ["check_operation_preconditions"]
