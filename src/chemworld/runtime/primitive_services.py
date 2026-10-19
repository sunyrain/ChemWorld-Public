"""Primitive operation services for the transactional runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from chemworld.foundation import (
    WorldState,
    equipment_settings,
    process_with_metrics,
    selected_phase_id,
    species_with_added_initial_amounts,
    upsert_equipment_record,
)
from chemworld.foundation.solvents import (
    HEAT_CAPACITY_RATIOS,
    RELATIVE_VOLATILITIES,
    SolventInventory,
)
from chemworld.foundation.state import PhaseLedger
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.world.actions import CATALYSTS, SOLVENTS
from chemworld.world.mixtures import volumetric_heat_capacity, working_solvents
from chemworld.world.parameters import ChemWorldParameters
from chemworld.world.thermal_kernel import account_temperature_transition


def _action_float(action: dict[str, Any], key: str, default: float) -> float:
    value = action.get(key, default)
    return float(np.asarray(value).reshape(-1)[0])


def _action_index(action: dict[str, Any], key: str, default: int, count: int) -> int:
    return int(np.clip(int(_action_float(action, key, float(default))), 0, count - 1))


class ChemWorldPrimitiveOperationServices:
    """Apply primitive material, sampling, quench, evaporation, and penalty updates."""

    def __init__(self, world: ChemWorldParameters, species_view: MechanismSpeciesView) -> None:
        self.world = world
        self.species_view = species_view

    def _declared_phase(self, species_id: str) -> str | None:
        network = self.species_view.mechanism.network
        index = network.species_index.get(species_id)
        if index is None:
            return None
        return str(network.species[index].phase)

    def _material_destination_phase(
        self,
        state: WorldState,
        species_id: str | None = None,
    ) -> str | None:
        if state.phases is None or not state.phases.phases:
            return None
        selected = selected_phase_id(state.phases)
        if selected in state.phases.phases:
            return selected
        if "reactor_liquid" in state.phases.phases:
            return "reactor_liquid"
        if species_id is not None:
            declared = self._declared_phase(species_id)
            if declared in state.phases.phases:
                return declared
        for candidate in ("aqueous", "organic", "mother_liquor", "bottoms"):
            if candidate in state.phases.phases:
                return candidate
        return next(iter(state.phases.phases))

    def _phase_aware_material_state(
        self,
        state: WorldState,
        *,
        additions_mol: dict[str, float] | None = None,
        added_volume_L: float = 0.0,
    ) -> tuple[dict[str, float], PhaseLedger | None]:
        additions = additions_mol or {}
        species = state.species_amounts.copy()
        for species_id, addition in additions.items():
            species[species_id] = species.get(species_id, 0.0) + float(addition)
        if state.phases is None or set(state.phases.phases) == {"reactor_liquid"}:
            return species, state.phases

        phases = state.phases.phases.copy()
        for species_id, addition in additions.items():
            destination = self._material_destination_phase(state, species_id)
            if destination is None:
                continue
            phase = phases[destination]
            amounts = phase.species_amounts_mol.copy()
            amounts[species_id] = amounts.get(species_id, 0.0) + float(addition)
            phases[destination] = replace(phase, species_amounts_mol=amounts)
        if added_volume_L > 0.0:
            destination = self._material_destination_phase(state)
            if destination is not None:
                phase = phases[destination]
                phases[destination] = replace(
                    phase,
                    volume_L=phase.volume_L + added_volume_L,
                )
        ledger = PhaseLedger(phases)
        return ledger.total_amounts_mol(), ledger

    def add_reagent(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        amount = float(np.clip(_action_float(action, "amount_mol", 0.003), 0.0, 0.040))
        additions = self.species_view.reagent_charge_amounts(
            state,
            limiting_amount_mol=amount,
        )
        return self._add_feed(state, additions)

    def add_component(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        component = int(action["component"])
        feeds = self.species_view.feed_species
        if not 0 <= component < len(feeds):
            raise ValueError("Component is outside this mechanism's feed catalog")
        return self._add_feed(state, {feeds[component]: float(action["amount_mol"])})

    def _add_feed(self, state: WorldState, additions: dict[str, float]) -> WorldState:
        species, phases = self._phase_aware_material_state(
            state,
            additions_mol=additions,
        )
        species_ledger = species_with_added_initial_amounts(state.species, additions)
        amount = sum(additions.values())
        ledger = state.ledger.with_updates(cost=state.ledger.cost + 0.03 * amount / 0.01)
        return state.replace(
            species_amounts=species,
            phases=phases,
            ledger=ledger,
            species=species_ledger,
        )

    def add_solvent(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        volume = float(np.clip(_action_float(action, "volume_L", 0.025), 0.0, 0.080))
        solvent = _action_index(action, "solvent", 0, len(SOLVENTS))
        previous_settings = equipment_settings(state.equipment, "batch_reactor")
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="batch_reactor",
            equipment_type="batch_reactor",
            attached_vessel_id=state.vessel_id,
            status="configured",
            settings={
                "solvent": solvent,
                "solvent_volume_L": float(previous_settings.get("solvent_volume_L", 0.0)) + volume,
            },
        )
        ledger = state.ledger.with_updates(
            cost=state.ledger.cost + volume * 8.0 * float(self.world.solvent_costs[solvent])
        )
        species, phases = self._phase_aware_material_state(
            state,
            added_volume_L=volume,
        )
        old_capacity = state.volume_L * volumetric_heat_capacity(state, self.world.rho_cp_J_per_L_K)
        fresh_capacity = volume * self.world.rho_cp_J_per_L_K * HEAT_CAPACITY_RATIOS[solvent]
        mixed_temperature = (
            state.temperature_K
            + fresh_capacity
            / (old_capacity + fresh_capacity)
            * (self.world.environment_temperature_K - state.temperature_K)
            if old_capacity + fresh_capacity > 0
            else state.temperature_K
        )
        return state.replace(
            species_amounts=species,
            phases=phases,
            volume_L=state.volume_L + volume,
            ledger=ledger,
            equipment=equipment,
            temperature_K=mixed_temperature,
        ).replace(phases=self._solvent_addition_phases(state, species, volume, solvent))

    def _solvent_addition_phases(
        self, state: WorldState, species: dict[str, float], volume: float, solvent: int
    ) -> PhaseLedger:
        assert state.phases is not None
        target = self._material_destination_phase(state)
        if target is None:
            raise ValueError("Solvent addition requires a material phase")
        phases = state.phases.phases.copy()
        current = phases[target]
        phases[target] = replace(
            current,
            volume_L=current.volume_L + volume,
            solvents=current.solvents + SolventInventory.pure(solvent, volume),
            species_amounts_mol=species
            if target == "reactor_liquid"
            else current.species_amounts_mol,
        )
        return PhaseLedger(phases)

    def add_catalyst(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        amount = float(np.clip(_action_float(action, "catalyst_amount_mol", 0.00020), 0.0, 0.005))
        catalyst = _action_index(action, "catalyst", 0, len(CATALYSTS))
        previous_settings = equipment_settings(state.equipment, "batch_reactor")
        active_catalyst = self.species_view.active_catalyst_species(state)
        additions = {} if active_catalyst is None else {active_catalyst: amount}
        species, phases = self._phase_aware_material_state(
            state,
            additions_mol=additions,
        )
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="batch_reactor",
            equipment_type="batch_reactor",
            attached_vessel_id=state.vessel_id,
            status="configured",
            settings={
                "catalyst": catalyst,
                "catalyst_amount_mol": float(previous_settings.get("catalyst_amount_mol", 0.0))
                + amount,
            },
        )
        ledger = state.ledger.with_updates(
            cost=state.ledger.cost
            + 4.0 * amount / 0.001 * float(self.world.catalyst_costs[catalyst])
        )
        return state.replace(
            species_amounts=species,
            phases=phases,
            ledger=ledger,
            equipment=equipment,
        )

    def sample(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        from chemworld.runtime.full_process_contract import withdraw_sample

        volume = float(np.clip(_action_float(action, "sample_volume_L", 0.0001), 0.0, 0.002))
        return withdraw_sample(state, volume).replace(
            ledger=state.ledger.with_updates(
                sample_consumed_L=state.ledger.sample_consumed_L + volume,
                cost=state.ledger.cost + 0.01,
            )
        )

    def quench(self, state: WorldState) -> WorldState:
        target = max(298.15, state.temperature_K - 45.0)
        sensible_magnitude = abs(
            volumetric_heat_capacity(state, self.world.rho_cp_J_per_L_K)
            * state.volume_L
            * (target - state.temperature_K)
        )
        duration = max(sensible_magnitude / 250.0, 1.0)
        thermal = account_temperature_transition(
            state=state,
            world=self.world,
            final_temperature_K=target,
            duration_s=duration,
        )
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + duration,
            cost=state.ledger.cost + 0.03 + abs(thermal.jacket_energy_J) / 250_000.0,
            energy_jacket_J=state.ledger.energy_jacket_J + thermal.jacket_energy_J,
            heat_loss_J=state.ledger.heat_loss_J + thermal.heat_loss_J,
        )
        metadata = {
            **state.metadata,
            "last_energy_transition": {"operation": "quench", **thermal.to_dict()},
            "reaction_chemistry_stopped": True,
        }
        reactor_settings = equipment_settings(state.equipment, "batch_reactor")
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="batch_reactor",
            equipment_type="batch_reactor",
            attached_vessel_id=state.vessel_id,
            status="quenched",
            settings={
                "reaction_stopped": True,
                "quench_time_s": state.ledger.time_s + duration,
                "quench_count": int(reactor_settings.get("quench_count", 0)) + 1,
            },
        )
        process = process_with_metrics(
            state.process,
            reaction_chemistry_stopped=1.0,
        )
        return state.replace(
            temperature_K=target,
            quenched=True,
            ledger=ledger,
            equipment=equipment,
            process=process,
            metadata=metadata,
        )

    def evaporate(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        duration = float(np.clip(_action_float(action, "duration_s", 600.0), 0.0, 14_400.0))
        target_temperature = float(
            np.clip(_action_float(action, "target_temperature_K", 328.15), 298.15, 390.0)
        )
        medium = working_solvents(state)
        heat_capacity = (
            volumetric_heat_capacity(state, self.world.rho_cp_J_per_L_K) * state.volume_L
        )
        target_temperature = max(target_temperature, state.temperature_K)
        sensible = min(45.0 * duration, heat_capacity * (target_temperature - state.temperature_K))
        final_temperature = state.temperature_K + sensible / max(heat_capacity, 1e-12)
        requested = state.volume_L * min(
            0.70, duration / 7200.0 * medium.log_property(RELATIVE_VOLATILITIES)
        )
        latent_J_L = 40_700.0 / 0.018
        removed_volume = min(requested, max(45.0 * duration - sensible, 0.0) / latent_J_L)
        latent = removed_volume * latent_J_L
        removal = removed_volume / max(state.volume_L, 1e-12)
        process_metrics = {} if state.process is None else state.process.metrics
        solvent_loss = min(
            1.0,
            float(process_metrics.get("solvent_loss", 0.0)) + removal,
        )
        process = process_with_metrics(state.process, solvent_loss=solvent_loss)
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + duration,
            cost=state.ledger.cost + duration / 3600.0 * 0.040,
            risk=min(1.0, state.ledger.risk + 0.04 * removal),
            energy_jacket_J=state.ledger.energy_jacket_J + sensible + latent,
        )
        assert state.phases is not None
        target_phase = self._material_destination_phase(state)
        phases = state.phases.phases.copy()
        if target_phase is None:
            raise ValueError("Evaporation requires a selected material phase")
        phases[target_phase] = replace(
            phases[target_phase], volume_L=max(phases[target_phase].volume_L - removed_volume, 0.0)
        )
        return state.replace(
            phases=PhaseLedger(phases),
            volume_L=state.volume_L - removed_volume,
            temperature_K=final_temperature,
            ledger=ledger,
            process=process,
            metadata={
                **state.metadata,
                "last_evaporation_energy": {
                    "sensible_J": sensible,
                    "latent_J": latent,
                    "jacket_J": sensible + latent,
                    "removed_carrier_L": removed_volume,
                    "energy_balance_residual_J": 0.0,
                    "model": "finite_carrier_45W_sensible_then_latent_no_environment_loss",
                },
            },
        )

    def penalize_invalid(self, state: WorldState) -> WorldState:
        ledger = state.ledger.with_updates(
            cost=state.ledger.cost + 0.01,
            risk=min(1.0, state.ledger.risk + 0.08),
        )
        return state.replace(ledger=ledger)


__all__ = ["ChemWorldPrimitiveOperationServices"]
