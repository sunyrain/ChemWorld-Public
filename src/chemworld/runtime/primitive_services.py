"""Primitive operation services for the transactional runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any, cast

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
from chemworld.runtime.action_values import action_float
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.world.actions import CATALYSTS, SOLVENTS
from chemworld.world.mixtures import volumetric_heat_capacity, working_solvents
from chemworld.world.parameters import ChemWorldParameters
from chemworld.world.thermal_kernel import account_temperature_transition


def _action_index(action: dict[str, Any], key: str, default: int, count: int) -> int:
    return int(np.clip(int(action_float(action, key, float(default))), 0, count - 1))


class ChemWorldPrimitiveOperationServices:
    """Apply primitive material, sampling, quench, evaporation, and penalty updates."""

    def __init__(
        self, world: ChemWorldParameters, species_view: MechanismSpeciesView, material_router=None
    ) -> None:
        self.world = world
        self.species_view = species_view
        self.material_router = material_router

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

    def feed_additions(self, state: WorldState, action: dict[str, Any]) -> dict[str, float]:
        if self.material_router is not None and self.material_router.uses_native_packets(state):
            return self.material_router.feed_additions(state, action)
        if action.get("operation") == "add_component":
            component = int(action["component"])
            feeds = self.species_view.feed_species
            if not 0 <= component < len(feeds):
                raise ValueError("Component is outside this mechanism's feed catalog")
            return {feeds[component]: float(action["amount_mol"])}
        amount = float(np.clip(action_float(action, "amount_mol", 0.003), 0.0, 0.040))
        return self.species_view.reagent_charge_amounts(
            state,
            limiting_amount_mol=amount,
        )

    def add_reagent(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        return self._add_feed(state, self.feed_additions(state, action))

    def add_component(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        if self.material_router is not None and self.material_router.uses_native_packets(state):
            return self.material_router.add_finite_feed(state, action)
        return self._add_feed(state, self.feed_additions(state, action))

    def _add_feed(self, state: WorldState, additions: dict[str, float]) -> WorldState:
        from chemworld.foundation.discrete_inventory import (
            CONTINUOUS_ADAPTIVE_ALGORITHM,
            CONTINUOUS_SDE_ALGORITHM,
        )

        continuation = state.process.stochastic_continuation
        sde = continuation is not None and continuation.get("algorithm") == CONTINUOUS_SDE_ALGORITHM
        adaptive = (continuation is not None
                    and continuation.get("algorithm") == CONTINUOUS_ADAPTIVE_ALGORITHM)
        discrete = continuation is not None and not (sde or adaptive)
        if discrete:
            from chemworld.foundation.discrete_inventory import counts_to_mol, mol_to_counts

            dose = mol_to_counts(additions)
            existing = mol_to_counts(state.species_amounts)
            final_counts = {key: existing.get(key, 0) + dose.get(key, 0)
                            for key in existing.keys() | dose.keys()}
            exact_amounts = counts_to_mol(final_counts)
        species, phases = self._phase_aware_material_state(
            state,
            additions_mol=additions,
        )
        if discrete:
            species = exact_amounts
            phases = PhaseLedger({key: replace(phase, species_amounts_mol=exact_amounts)
                                  for key, phase in phases.phases.items()})
        species_ledger = species_with_added_initial_amounts(state.species, additions)
        amount = sum(additions.values())
        ledger = state.ledger.with_updates(cost=state.ledger.cost + 0.03 * amount / 0.01)
        process = state.process
        if discrete and amount > 0:
            process = replace(process, stochastic_continuation={
                **process.stochastic_continuation, "pending": None, "signature": None})
        if sde and amount > 0:
            from chemworld.physchem.stochastic_sde import rebind_sde_continuation

            physics = None if self.material_router is None else self.material_router.physics
            model = (None if physics is None else
                     physics.models_of("stochastic_sde").get(state.vessel_id))
            if model is None:
                raise ValueError("SDE dosing requires its actual physical model declaration")
            process = replace(process, stochastic_continuation=rebind_sde_continuation(
                physics.network, model, state.species_amounts, species,
                cast(Mapping[str, object], continuation),
            ))
        if adaptive and amount > 0:
            from chemworld.physchem.stochastic_adaptive_advance import rebind_adaptive_continuation

            physics = None if self.material_router is None else self.material_router.physics
            model = (None if physics is None else
                     physics.models_of("stochastic_hybrid").get(state.vessel_id))
            if model is None:
                raise ValueError("adaptive dosing requires its actual physical model declaration")
            process = replace(process, stochastic_continuation=rebind_adaptive_continuation(
                physics.network, model, state.species_amounts, species, continuation,
            ).continuation)
        return state.replace(
            species_amounts=species,
            phases=phases,
            ledger=ledger,
            species=species_ledger,
            process=process,
        )

    def add_solvent(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        volume = float(np.clip(action_float(action, "volume_L", 0.025), 0.0, 0.080))
        solvent = _action_index(action, "solvent", 0, len(SOLVENTS))
        from chemworld.runtime.condensed_material_routing import add_background, background_model

        if (self.material_router is not None
                and background_model(state,self.material_router.physics) is not None):
            return add_background(state,self.material_router.physics,volume_L=volume,
                solvent_index=solvent,cost_delta=volume*8*float(self.world.solvent_costs[solvent]))
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

    @staticmethod
    def catalyst_charge(action: dict[str, Any]) -> float:
        return float(np.clip(action_float(action, "catalyst_amount_mol", 0.00020), 0.0, 0.005))

    def add_catalyst(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        amount = self.catalyst_charge(action)
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
        from chemworld.runtime.aqueous_mineral import close_aqueous
        from chemworld.runtime.full_process_contract import withdraw_sample

        volume = float(np.clip(action_float(action, "sample_volume_L", 0.0001), 0.0, 0.002))
        return close_aqueous(withdraw_sample(state, volume)).replace(
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
        duration = float(np.clip(action_float(action, "duration_s", 600.0), 0.0, 14_400.0))
        target_temperature = float(
            np.clip(action_float(action, "target_temperature_K", 328.15), 298.15, 390.0)
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

    @staticmethod
    def penalize_invalid(state: WorldState) -> WorldState:
        ledger = state.ledger.with_updates(
            cost=state.ledger.cost + 0.01,
            risk=min(1.0, state.ledger.risk + 0.08),
        )
        return state.replace(ledger=ledger)


__all__ = ["ChemWorldPrimitiveOperationServices"]
