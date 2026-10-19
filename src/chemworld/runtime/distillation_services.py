"""Distillation state-update services for the transactional runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np

from chemworld.foundation import (
    WorldState,
    equipment_settings,
    process_with_metrics,
    upsert_equipment_record,
)
from chemworld.foundation.solvents import SolventInventory
from chemworld.foundation.state import PhaseLedger, PhaseRecord
from chemworld.runtime.full_process_contract import population_active
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.runtime.vnext_downstream import run_duty_limited_distillation
from chemworld.world.mixtures import working_solvents
from chemworld.world.parameters import ChemWorldParameters


def _action_float(action: dict[str, Any], key: str, default: float) -> float:
    value = action.get(key, default)
    return float(np.asarray(value).reshape(-1)[0])


def _distillate_phase_amounts(
    state: WorldState,
    *,
    target_species: tuple[str, ...],
    impurity_species: tuple[str, ...],
) -> tuple[float, float]:
    if state.phases is None or "distillate" not in state.phases.phases:
        return 0.0, 0.0
    distillate = state.phases.phases["distillate"]
    return (
        sum(
            float(distillate.species_amounts_mol.get(species_id, 0.0))
            for species_id in target_species
        ),
        sum(
            float(distillate.species_amounts_mol.get(species_id, 0.0))
            for species_id in impurity_species
        ),
    )


def _collect_distillate_fraction(
    state: WorldState,
    *,
    fraction: float,
) -> PhaseLedger:
    """Move a fraction of the current distillate into an explicit inventory."""

    if state.phases is None or "distillate" not in state.phases.phases:
        raise ValueError("distillate phase is required before fraction collection")
    phases = state.phases.phases.copy()
    distillate = phases["distillate"]
    previous_collected = phases.get("collected_fraction")
    component_ids = set(distillate.species_amounts_mol)
    if previous_collected is not None:
        component_ids.update(previous_collected.species_amounts_mol)
    collected_amounts = {
        species_id: (
            0.0
            if previous_collected is None
            else float(previous_collected.species_amounts_mol.get(species_id, 0.0))
        )
        + fraction * float(distillate.species_amounts_mol.get(species_id, 0.0))
        for species_id in component_ids
    }
    remaining_amounts = {
        species_id: (1.0 - fraction) * float(amount)
        for species_id, amount in distillate.species_amounts_mol.items()
    }
    collected_volume = (
        0.0 if previous_collected is None else previous_collected.volume_L
    ) + fraction * distillate.volume_L
    phases["distillate"] = replace(
        distillate,
        volume_L=(1.0 - fraction) * distillate.volume_L,
        species_amounts_mol=remaining_amounts,
        selected=False,
    )
    phases["collected_fraction"] = PhaseRecord(
        phase_id="collected_fraction",
        vessel_id=state.vessel_id,
        phase_type="liquid",
        volume_L=collected_volume,
        species_amounts_mol=collected_amounts,
        settled=True,
        selected=True,
        metadata={"source_phase": "distillate"},
    )
    if "bottoms" in phases:
        phases["bottoms"] = replace(phases["bottoms"], selected=False)
    return PhaseLedger(phases)


def _allocated_amounts(
    source_amounts: dict[str, float],
    species_ids: tuple[str, ...],
    total_mol: float,
) -> dict[str, float]:
    source_total = sum(
        max(float(source_amounts.get(species_id, 0.0)), 0.0) for species_id in species_ids
    )
    allocated_total = min(max(float(total_mol), 0.0), source_total)
    if source_total <= 1.0e-12 or allocated_total <= 0.0:
        return dict.fromkeys(species_ids, 0.0)
    return {
        species_id: allocated_total
        * max(float(source_amounts.get(species_id, 0.0)), 0.0)
        / source_total
        for species_id in species_ids
    }


def _distillation_phases(
    state: WorldState,
    *,
    feed_amounts: dict[str, float],
    target_species: tuple[str, ...],
    impurity_species: tuple[str, ...],
    product_mol: float,
    impurity_mol: float,
    new_distillate_volume_L: float,
    bottoms_volume_L: float,
    distillate_selected: bool = True,
) -> PhaseLedger:
    previous_phases = {} if state.phases is None else state.phases.phases
    previous_distillate = previous_phases.get("distillate")
    previous_collected = previous_phases.get("collected_fraction")
    component_ids = set(state.species_amounts) | set(feed_amounts)
    if previous_distillate is not None:
        component_ids.update(previous_distillate.species_amounts_mol)
    if previous_collected is not None:
        component_ids.update(previous_collected.species_amounts_mol)
    new_cut_amounts = dict.fromkeys(component_ids, 0.0)
    allocated_targets = _allocated_amounts(
        feed_amounts,
        target_species,
        product_mol,
    )
    allocated_impurities = _allocated_amounts(
        feed_amounts,
        impurity_species,
        impurity_mol,
    )
    for species_id, amount_mol in allocated_targets.items():
        new_cut_amounts[species_id] = amount_mol
    for species_id, amount_mol in allocated_impurities.items():
        new_cut_amounts[species_id] = amount_mol
    bottoms_amounts = dict.fromkeys(component_ids, 0.0)
    bottoms_amounts.update(feed_amounts)
    for species_id, amount_mol in new_cut_amounts.items():
        bottoms_amounts[species_id] = max(
            bottoms_amounts.get(species_id, 0.0) - amount_mol,
            0.0,
        )
    distillate_amounts = {
        species_id: (
            0.0
            if previous_distillate is None
            else float(previous_distillate.species_amounts_mol.get(species_id, 0.0))
        )
        + float(new_cut_amounts.get(species_id, 0.0))
        for species_id in component_ids
    }
    phases = {
        "bottoms": PhaseRecord(
            phase_id="bottoms",
            vessel_id=state.vessel_id,
            phase_type="liquid",
            volume_L=max(bottoms_volume_L, 0.0),
            species_amounts_mol=bottoms_amounts,
            settled=True,
            selected=not distillate_selected,
            metadata={},
        ),
        "distillate": PhaseRecord(
            phase_id="distillate",
            vessel_id=state.vessel_id,
            phase_type="liquid",
            volume_L=(0.0 if previous_distillate is None else previous_distillate.volume_L)
            + max(new_distillate_volume_L, 0.0),
            species_amounts_mol=distillate_amounts,
            settled=True,
            selected=distillate_selected,
            metadata={"cut_accumulation": "cumulative_receiver_inventory"},
        ),
    }
    if previous_collected is not None:
        phases["collected_fraction"] = replace(previous_collected, selected=False)
    return PhaseLedger(phases)


class ChemWorldDistillationServices:
    """Apply shortcut distillation and fraction collection updates."""

    def __init__(
        self,
        world: ChemWorldParameters,
        species_view: MechanismSpeciesView,
    ) -> None:
        self.world = world
        self.species_view = species_view

    def distill(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        duration = float(np.clip(_action_float(action, "duration_s", 1200.0), 0.0, 14_400.0))
        target_temperature = float(
            np.clip(_action_float(action, "target_temperature_K", 345.15), 298.15, 430.0)
        )
        reflux = float(np.clip(_action_float(action, "reflux_ratio", 1.5), 0.0, 10.0))
        target_species = self.species_view.target_species_for_state(state)
        impurity_species = self.species_view.impurity_species_for_state(state)
        previous_phases = {} if state.phases is None else state.phases.phases
        bottoms = previous_phases.get("bottoms")
        feed_amounts = (
            state.species_amounts.copy() if bottoms is None else bottoms.species_amounts_mol.copy()
        )
        feed_volume_L = max(
            state.volume_L if bottoms is None else bottoms.volume_L,
            0.0,
        )
        feed_source_phase = "reactor_liquid" if bottoms is None else "bottoms"
        p_mol = sum(float(feed_amounts.get(species_id, 0.0)) for species_id in target_species)
        impurity_mol = sum(
            float(feed_amounts.get(species_id, 0.0)) for species_id in impurity_species
        )
        distillate_cut = float(np.clip(0.25 + duration / 9000.0, 0.05, 0.90))
        if population_active(state) and "cut_fraction" in action:
            distillate_cut = float(action["cut_fraction"])
        theoretical_stages = float(np.clip(2.0 + duration / 900.0, 1.0, 20.0))
        feed_solvents = working_solvents(state) if bottoms is None else bottoms.solvents
        cut_solvents = SolventInventory()
        retained_solvents = feed_solvents
        if feed_solvents.volume_L <= 1.0e-12:
            distillate_product = 0.0
            distillate_impurity = 0.0
            distillation_metadata: dict[str, object] = {"no_distillable_material": True}
            executed_model_id: str | None = None
            heat_duty = (70.0 + 8.0 * reflux) * duration
            distillation_cost = 0.045 + duration / 3600.0 * (0.065 + 0.012 * reflux)
            distillation_risk = 0.035 + 0.06 * ((target_temperature - 298.15) / 132.0)
        else:
            distillation = run_duty_limited_distillation(
                {"product": p_mol, "impurity": impurity_mol},
                pressure_Pa=max(state.pressure_Pa, 1.0),
                initial_temperature_K=min(state.temperature_K, target_temperature),
                operating_temperature_K=target_temperature,
                duration_s=duration,
                reflux_ratio=reflux,
                requested_cut_fraction=distillate_cut,
                relative_volatility_multiplier=self.world.domain_parameter(
                    "distillation_relative_volatility_multiplier"
                ),
                solvents=feed_solvents,
                carrier_heat_capacity_J_L_K=self.world.rho_cp_J_per_L_K,
            )
            distillate = distillation.outlet("distillate")
            distillate_product = distillate.get("product", 0.0)
            distillate_impurity = distillate.get("impurity", 0.0)
            distillate_cut = distillation.actual_distillate_cut_fraction
            cut_solvents = SolventInventory(
                tuple(distillate.get(f"carrier-{i}", 0.0) * 0.018 for i in range(4))
            )
            retained_solvents = SolventInventory(
                tuple(
                    distillation.outlet("bottoms").get(f"carrier-{i}", 0.0) * 0.018
                    for i in range(4)
                )
            )
            distillation_metadata = distillation.to_dict()
            executed_model_id = distillation.model_id
            heat_duty = distillation.total_reboiler_duty_J
            distillation_cost = 0.045 + duration / 3600.0 * (0.065 + 0.012 * reflux)
            distillation_risk = 0.03 + 0.04 * min(
                distillation.average_reboiler_power_W / 420.0, 1.0
            )
        process_metrics = {} if state.process is None else state.process.metrics
        initial_p = max(
            float(process_metrics.get("pre_separation_product_mol", p_mol)),
            p_mol,
            1.0e-12,
        )
        previous_settings = equipment_settings(state.equipment, "distillation_column")
        distillation_history = list(previous_settings.get("distillation_history", ()))
        distillation_history.append(
            {
                "cut_index": len(distillation_history) + 1,
                "feed_source_phase": feed_source_phase,
                "feed_volume_L": feed_volume_L,
                "feed_target_mol": p_mol,
                "feed_impurity_mol": impurity_mol,
                "target_temperature_K": target_temperature,
                "duration_s": duration,
                "reflux_ratio": reflux,
                "new_cut_fraction": distillate_cut,
                "model_id": executed_model_id,
            }
        )
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="distillation_column",
            equipment_type="shortcut_distillation_column",
            attached_vessel_id=state.vessel_id,
            status="distilled",
            settings={
                "distillation_model": executed_model_id,
                "distillation_provider_path": (
                    "chemworld.physchem.distillation_adapter_manifest."
                    "DutyLimitedDistillationProvider"
                    if executed_model_id is not None
                    else None
                ),
                "distillation_adapter_id": (
                    "wf-60-duty-limited-distillation" if executed_model_id is not None else None
                ),
                "distillation_provenance": (
                    list(
                        cast(
                            list[str],
                            distillation_metadata.get("provenance", []),
                        )
                    )
                    if executed_model_id is not None
                    else []
                ),
                "distillation_kernel": distillation_metadata,
                "distillation_history": distillation_history,
                "feed_source_phase": feed_source_phase,
                "distillate_cut_fraction": distillate_cut,
                "theoretical_stages": theoretical_stages,
                "reflux_ratio": reflux,
            },
        )
        solvent_loss = float(process_metrics.get("solvent_loss", 0.0))
        risk = min(1.0, state.ledger.risk + distillation_risk)
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + duration,
            cost=state.ledger.cost + distillation_cost,
            risk=risk,
            energy_jacket_J=state.ledger.energy_jacket_J + heat_duty,
        )
        new_distillate_volume_L = cut_solvents.volume_L
        bottoms_volume_L = retained_solvents.volume_L
        phases = _distillation_phases(
            state,
            feed_amounts=feed_amounts,
            target_species=target_species,
            impurity_species=impurity_species,
            product_mol=distillate_product,
            impurity_mol=distillate_impurity,
            new_distillate_volume_L=new_distillate_volume_L,
            bottoms_volume_L=bottoms_volume_L,
        )
        phase_map = phases.phases.copy()
        old_receiver = previous_phases.get("distillate")
        previous_carriers = SolventInventory() if old_receiver is None else old_receiver.solvents
        phase_map["distillate"] = replace(
            phase_map["distillate"], solvents=previous_carriers + cut_solvents
        )
        phase_map["bottoms"] = replace(phase_map["bottoms"], solvents=retained_solvents)
        phases = PhaseLedger(phase_map)
        receiver = phases.phases["distillate"]
        receiver_product = sum(
            float(receiver.species_amounts_mol.get(species_id, 0.0))
            for species_id in target_species
        )
        receiver_impurity = sum(
            float(receiver.species_amounts_mol.get(species_id, 0.0))
            for species_id in impurity_species
        )
        receiver_purity = receiver_product / max(
            receiver_product + receiver_impurity,
            1.0e-12,
        )
        process = process_with_metrics(
            state.process,
            pre_separation_product_mol=initial_p,
            solvent_loss=solvent_loss,
            distillate_purity=float(np.clip(receiver_purity, 0.0, 1.0)),
            distillate_recovery=float(np.clip(receiver_product / initial_p, 0.0, 1.0)),
        )
        return state.replace(
            volume_L=receiver.volume_L,
            temperature_K=target_temperature,
            ledger=ledger,
            equipment=equipment,
            process=process,
            phases=phases,
        )

    def collect_fraction(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        fraction = float(np.clip(_action_float(action, "transfer_fraction", 0.90), 0.0, 1.0))
        target_species = self.species_view.target_species_for_state(state)
        impurity_species = self.species_view.impurity_species_for_state(state)
        phases = _collect_distillate_fraction(state, fraction=fraction)
        collected = phases.phases["collected_fraction"]
        product = sum(
            float(collected.species_amounts_mol.get(species_id, 0.0))
            for species_id in target_species
        )
        impurity = sum(
            float(collected.species_amounts_mol.get(species_id, 0.0))
            for species_id in impurity_species
        )
        purity = product / max(product + impurity, 1.0e-12)
        process_metrics = {} if state.process is None else state.process.metrics
        target_amount = self.species_view.target_amount(state)
        initial_p = max(
            float(process_metrics.get("pre_separation_product_mol", target_amount)),
            target_amount,
            1.0e-12,
        )
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="distillation_column",
            equipment_type="shortcut_distillation_column",
            attached_vessel_id=state.vessel_id,
            status="fraction_collected",
            settings={
                "fraction_collected": True,
                "transfer_fraction": fraction,
                "collected_product_mol": product,
                "collected_impurity_mol": impurity,
                "collected_purity": purity,
            },
        )
        solvent_loss = float(process_metrics.get("solvent_loss", 0.0))
        process = process_with_metrics(
            state.process,
            pre_separation_product_mol=initial_p,
            solvent_loss=solvent_loss,
            distillate_purity=float(np.clip(purity, 0.0, 1.0)),
            distillate_recovery=float(np.clip(product / initial_p, 0.0, 1.0)),
            purity=float(np.clip(purity, 0.0, 1.0)),
            recovery=float(np.clip(product / initial_p, 0.0, 1.0)),
        )
        ledger = state.ledger.with_updates(cost=state.ledger.cost + 0.018)
        return state.replace(
            volume_L=collected.volume_L,
            ledger=ledger,
            equipment=equipment,
            process=process,
            phases=phases,
        )


__all__ = ["ChemWorldDistillationServices"]
