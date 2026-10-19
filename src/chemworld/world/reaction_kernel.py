"""Reaction ODE kernel for the shared ChemWorld law."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from chemworld.foundation import WorldState, equipment_settings
from chemworld.physchem.batch_reactors import DynamicBatchReactorModel
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reactor_shared import (
    HeatTransferSpec,
    JacketTemperatureProgram,
    PressureBoundarySpec,
    ReactorResult,
    ReactorValidityDomain,
)
from chemworld.world.mixtures import volumetric_heat_capacity, working_solvents
from chemworld.world.thermal_control import REACTION_THERMAL_CONTROL_ID

R_GAS = 8.31446261815324


@dataclass(frozen=True)
class ReactionIntegrationResult:
    species_amounts: dict[str, float]
    temperature_K: float
    duration_s: float
    cost_delta: float
    energy_jacket_J: float
    heat_reaction_J: float
    heat_loss_J: float
    stirring_speed_rpm: float
    solver_diagnostic: dict[str, object] = field(default_factory=dict)
    reactor_diagnostic: dict[str, object] = field(default_factory=dict)
    material_balance_error_mol: float = 0.0
    maximum_conservation_drift_mol: float = 0.0
    element_inventory_residuals_mol: dict[str, float] = field(default_factory=dict)
    charge_inventory_residual_mol: float = 0.0
    energy_balance_residual_J: float = 0.0
    termination_reason: str = "duration_reached"
    model_id: str = "dynamic_batch_reaction_runtime_v1"
    provider_id: str = "chemworld_validated_reaction_reactor_runtime_v1"
    provenance: dict[str, object] = field(default_factory=dict)
    trajectory_digest: str = ""


@dataclass(frozen=True)
class RuntimeAdjustedReactionNetwork:
    """Rate-multiplier adapter that preserves the validated network contract."""

    base: ReactionNetworkSpec
    rate_multipliers: dict[str, float]

    @property
    def network_id(self) -> str:
        return self.base.network_id

    @property
    def species(self) -> tuple[Any, ...]:
        return self.base.species

    @property
    def reactions(self) -> tuple[Any, ...]:
        return self.base.reactions

    @property
    def species_ids(self) -> tuple[str, ...]:
        return self.base.species_ids

    @property
    def species_index(self) -> dict[str, int]:
        return self.base.species_index

    def reaction_rates(
        self,
        amounts_mol: Mapping[str, float],
        *,
        volume_L: float,
        temperature_K: float,
        species_thermo: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        rates = self.base.reaction_rates(
            amounts_mol,
            volume_L=volume_L,
            temperature_K=temperature_K,
            species_thermo=species_thermo,
        )
        return {
            reaction_id: rate * self.rate_multipliers[reaction_id]
            for reaction_id, rate in rates.items()
        }

    def amount_derivatives(
        self,
        amounts_mol: Mapping[str, float],
        *,
        volume_L: float,
        temperature_K: float,
        species_thermo: Mapping[str, Any] | None = None,
    ) -> dict[str, float]:
        rates = self.reaction_rates(
            amounts_mol,
            volume_L=volume_L,
            temperature_K=temperature_K,
            species_thermo=species_thermo,
        )
        derivatives = dict.fromkeys(self.species_ids, 0.0)
        for reaction in self.reactions:
            rate_mol_s = rates[reaction.reaction_id] * volume_L
            for species_id, coefficient in reaction.stoichiometry.items():
                derivatives[species_id] += coefficient * rate_mol_s
        return derivatives


def integrate_compiled_reaction_ode(
    *,
    state: WorldState,
    world: Any,
    compiled_mechanism: Any,
    duration_s: float,
    target_temperature_K: float,
    heat: bool,
    stirring_speed_rpm: float,
    jacket_program: JacketTemperatureProgram | None = None,
) -> ReactionIntegrationResult | None:
    """Advance the validated mechanism through the bounded dynamic-batch reactor."""

    if compiled_mechanism is None:
        raise ValueError("compiled_mechanism is required for reaction integration")

    duration = _bounded_finite(duration_s, "duration_s", minimum=1.0e-6, maximum=14_400.0)
    target_temperature = _bounded_finite(
        target_temperature_K,
        "target_temperature_K",
        minimum=250.0,
        maximum=520.0,
    )
    stirring_speed = _bounded_finite(
        stirring_speed_rpm,
        "stirring_speed_rpm",
        minimum=100.0,
        maximum=1200.0,
    )
    if state.volume_L <= 0.0 or not np.isfinite(state.volume_L):
        raise ValueError("reaction advance requires finite positive volume_L")
    if state.temperature_K <= 0.0 or not np.isfinite(state.temperature_K):
        raise ValueError("reaction advance requires finite positive temperature_K")

    network: ReactionNetworkSpec = compiled_mechanism.network
    network.check_conservation(raise_on_error=True)
    species_ids = network.species_ids
    initial_amounts = {
        species_id: float(state.species_amounts.get(species_id, 0.0)) for species_id in species_ids
    }
    if any(value < 0.0 or not np.isfinite(value) for value in initial_amounts.values()):
        raise ValueError("reaction species amounts must be finite and nonnegative")
    reactor_settings = equipment_settings(state.equipment, "batch_reactor")
    catalyst = int(reactor_settings.get("catalyst", 0))
    solvent = int(reactor_settings.get("solvent", 0))
    catalyst_effects, solvent_effects = _reaction_effect_tables(world, state)
    if catalyst_effects.ndim not in {1, 2} or catalyst not in range(catalyst_effects.shape[0]):
        raise ValueError("configured catalyst index is outside the world contract")
    if solvent_effects.ndim not in {1, 2} or solvent not in range(solvent_effects.shape[0]):
        raise ValueError("configured solvent index is outside the world contract")
    stirring_factor = 0.70 + 0.30 * (1.0 - np.exp(-stirring_speed / 420.0))
    multipliers = {
        reaction.reaction_id: 0.0
        if state.quenched
        else _hidden_reaction_modifier(
            world,
            catalyst=catalyst,
            solvent=solvent,
            reaction_index=index,
            stirring_factor=stirring_factor,
            state=state,
        )
        for index, reaction in enumerate(network.reactions)
    }
    adjusted_network = RuntimeAdjustedReactionNetwork(network, multipliers)
    thermal = HeatTransferSpec(
        rho_cp_J_per_L_K=volumetric_heat_capacity(state, world.rho_cp_J_per_L_K),
        ua_W_per_K=float(world.ua_W_per_K),
        environment_temperature_K=float(world.environment_temperature_K),
        jacket_ua_W_per_K=4.0 if heat else 0.0,
        jacket_temperature_K=target_temperature if heat else None,
        maximum_jacket_heating_W=90.0,
        maximum_jacket_cooling_W=70.0,
    )
    maximum_temperature = _maximum_vessel_temperature(state)
    validity_domain = ReactorValidityDomain(
        minimum_temperature_K=250.0,
        maximum_temperature_K=maximum_temperature,
        minimum_pressure_Pa=1.0,
        maximum_pressure_Pa=_maximum_vessel_pressure(state),
        maximum_temperature_rate_K_s=2.0,
        material_balance_tolerance_mol=1.0e-8,
    )
    reactor = DynamicBatchReactorModel(
        cast(ReactionNetworkSpec, adjusted_network),
        reactor_id="chemworld_runtime_dynamic_batch_v1",
    )
    reactor_result = reactor.simulate(
        initial_amounts,
        initial_volume_L=state.volume_L,
        temperature_K=state.temperature_K,
        duration_s=duration,
        heat_transfer=thermal,
        jacket_program=jacket_program,
        pressure_boundary=PressureBoundarySpec(pressure_Pa=state.pressure_Pa),
        validity_domain=validity_domain,
        evaluation_times_s=tuple(np.linspace(0.0, duration, 17)),
    )
    _raise_for_failed_reactor_contract(reactor_result)
    maximum_conservation_drift = _maximum_invariant_drift(network, reactor_result)
    if maximum_conservation_drift > 1.0e-7:
        raise RuntimeError(
            "reaction trajectory conservation drift exceeded tolerance: "
            f"{maximum_conservation_drift:.6g} mol"
        )
    element_residuals = _element_inventory_residuals(network, reactor_result)
    charge_residual = _charge_inventory_residual(network, reactor_result)
    if max((abs(value) for value in element_residuals.values()), default=0.0) > 1.0e-8:
        raise RuntimeError(f"element inventory failed to close: {element_residuals}")
    if abs(charge_residual) > 1.0e-8:
        raise RuntimeError(f"charge inventory failed to close: {charge_residual}")
    energy_residual = _energy_balance_residual(reactor_result, thermal)
    energy_scale = max(
        abs(reactor_result.final_state.energy_jacket_J)
        + abs(reactor_result.final_state.heat_reaction_J)
        + abs(reactor_result.final_state.heat_loss_J),
        1.0,
    )
    if abs(energy_residual) > 1.0e-5 * energy_scale:
        raise RuntimeError(
            f"reaction energy ledger failed to close: residual={energy_residual:.6g} J"
        )
    species_amounts = state.species_amounts.copy()
    species_amounts.update(reactor_result.final_state.amounts_mol)
    solver_payload = reactor_result.metadata.get("solver_diagnostics")
    solver_diagnostics: tuple[Mapping[str, object], ...] = (
        tuple(item for item in solver_payload if isinstance(item, Mapping))
        if isinstance(solver_payload, list | tuple)
        else ()
    )
    solver_diagnostic = (
        dict(solver_diagnostics[-1])
        if solver_diagnostics and isinstance(solver_diagnostics[-1], Mapping)
        else {}
    )
    solver_diagnostic.update(
        {
            "success": True,
            "nonnegative_passed": reactor_result.diagnostics.nonnegative_species,
            "maximum_conservation_drift_mol": maximum_conservation_drift,
            "termination_reason": "duration_reached",
            "evaluation_count": len(reactor_result.times_s),
        }
    )
    provenance = {
        "provider_id": "chemworld_validated_reaction_reactor_runtime_v1",
        "reaction_network_model_id": ("reaction_ode_mass_action_arrhenius_reference_slice"),
        "reactor_model_id": "dynamic_batch_heat_release_jacket_sampling",
        "reactor_class": "chemworld.physchem.batch_reactors.DynamicBatchReactorModel",
        "reaction_network_class": "chemworld.physchem.reaction_network.ReactionNetworkSpec",
        "network_id": network.network_id,
        "mechanism_id": compiled_mechanism.mechanism_id,
        "mechanism_version": compiled_mechanism.mechanism_version,
        "mechanism_hash": compiled_mechanism.mechanism_hash,
        "rate_multipliers": multipliers,
        "operation_semantic": "advance",
        "heat_boundary": "continuous_bounded_linear_jacket" if heat else "environment_only_wait",
        "reaction_thermal_control_id": REACTION_THERMAL_CONTROL_ID,
    }
    digest = _trajectory_digest(reactor_result, provenance)
    return ReactionIntegrationResult(
        species_amounts=species_amounts,
        temperature_K=reactor_result.final_state.temperature_K,
        duration_s=duration,
        cost_delta=duration / 3600.0 * (0.03 if heat else 0.01),
        energy_jacket_J=reactor_result.final_state.energy_jacket_J,
        heat_reaction_J=reactor_result.final_state.heat_reaction_J,
        heat_loss_J=reactor_result.final_state.heat_loss_J,
        stirring_speed_rpm=stirring_speed,
        solver_diagnostic=solver_diagnostic,
        reactor_diagnostic=reactor_result.diagnostics.to_dict(),
        material_balance_error_mol=reactor_result.material_balance_error_mol,
        maximum_conservation_drift_mol=maximum_conservation_drift,
        element_inventory_residuals_mol=element_residuals,
        charge_inventory_residual_mol=charge_residual,
        energy_balance_residual_J=energy_residual,
        provenance=provenance,
        trajectory_digest=digest,
    )


def _bounded_finite(
    value: float,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    resolved = float(value)
    if not np.isfinite(resolved) or not minimum <= resolved <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return resolved


def _maximum_vessel_temperature(state: WorldState) -> float:
    if state.vessels is None or state.vessel_id not in state.vessels.vessels:
        return 470.0
    return min(float(state.vessels.vessels[state.vessel_id].max_temperature_K), 520.0)


def _maximum_vessel_pressure(state: WorldState) -> float:
    if state.vessels is None or state.vessel_id not in state.vessels.vessels:
        return 550_000.0
    return float(state.vessels.vessels[state.vessel_id].max_pressure_Pa)


def _raise_for_failed_reactor_contract(result: ReactorResult) -> None:
    diagnostics = result.diagnostics
    if not diagnostics.nonnegative_species:
        raise RuntimeError("dynamic batch reactor violated species nonnegativity")
    if not diagnostics.material_balance_closed:
        raise RuntimeError(
            "dynamic batch reactor material balance failed: "
            f"{diagnostics.material_balance_error_mol:.6g} mol"
        )
    fatal_warnings = {
        "temperature_outside_validity_domain",
        "pressure_outside_validity_domain",
        "thermal_runaway_rate_exceeds_limit",
    }
    failures = sorted(fatal_warnings.intersection(diagnostics.warnings))
    if failures:
        raise RuntimeError(f"dynamic batch reactor left its validity domain: {failures}")
    solver_diagnostics = result.metadata.get("solver_diagnostics")
    if not isinstance(solver_diagnostics, list) or not solver_diagnostics:
        raise RuntimeError("dynamic batch reactor returned no solver diagnostics")
    if any(not bool(item.get("success")) for item in solver_diagnostics if isinstance(item, dict)):
        raise RuntimeError("dynamic batch reactor solver did not converge")


def _maximum_invariant_drift(
    network: ReactionNetworkSpec,
    result: ReactorResult,
) -> float:
    stoich = np.asarray(network.stoichiometric_matrix(), dtype=float)
    if not stoich.size or len(result.times_s) <= 1:
        return 0.0
    trajectory = np.asarray(
        [result.amounts_mol[species_id] for species_id in network.species_ids],
        dtype=float,
    )
    left_vectors, singular_values, _right = np.linalg.svd(stoich, full_matrices=True)
    tolerance = (
        max(stoich.shape)
        * np.finfo(float).eps
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    rank = int(np.sum(singular_values > tolerance))
    invariants = left_vectors[:, rank:].T
    if not invariants.size:
        return 0.0
    inventory = invariants @ trajectory
    return float(np.max(np.abs(inventory - inventory[:, [0]])))


def _element_inventory_residuals(
    network: ReactionNetworkSpec,
    result: ReactorResult,
) -> dict[str, float]:
    initial: dict[str, float] = {}
    final: dict[str, float] = {}
    for species in network.species:
        for element, count in species.composition.items():
            initial[element] = initial.get(element, 0.0) + count * float(
                result.initial_state.amounts_mol.get(species.species_id, 0.0)
            )
            final[element] = final.get(element, 0.0) + count * float(
                result.final_state.amounts_mol.get(species.species_id, 0.0)
            )
    return {
        element: final.get(element, 0.0) - initial.get(element, 0.0)
        for element in sorted(set(initial) | set(final))
    }


def _charge_inventory_residual(
    network: ReactionNetworkSpec,
    result: ReactorResult,
) -> float:
    return sum(
        float(species.charge)
        * (
            float(result.final_state.amounts_mol.get(species.species_id, 0.0))
            - float(result.initial_state.amounts_mol.get(species.species_id, 0.0))
        )
        for species in network.species
    )


def _energy_balance_residual(
    result: ReactorResult,
    thermal: HeatTransferSpec,
) -> float:
    sensible = (
        thermal.rho_cp_J_per_L_K
        * result.final_state.volume_L
        * (result.final_state.temperature_K - result.initial_state.temperature_K)
    )
    ledger_input = (
        result.final_state.energy_jacket_J
        - result.final_state.heat_reaction_J
        - result.final_state.heat_loss_J
    )
    return sensible - ledger_input


def _trajectory_digest(
    result: ReactorResult,
    provenance: Mapping[str, object],
) -> str:
    payload = {
        "times_s": result.times_s,
        "amounts_mol": result.amounts_mol,
        "temperatures_K": result.temperatures_K,
        "provenance": provenance,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def compiled_reaction_ode_rhs(
    *,
    y: np.ndarray,
    state: WorldState,
    world: Any,
    compiled_mechanism: Any,
    target_temperature_K: float,
    heat: bool,
    stirring_speed_rpm: float,
) -> np.ndarray:
    """Return one instantaneous adjusted-network RHS for diagnostics only."""
    network = compiled_mechanism.network
    species_ids = network.species_ids
    species_count = len(species_ids)
    amounts_array = np.asarray(y[:species_count], dtype=float)
    if np.any(~np.isfinite(amounts_array)) or np.any(amounts_array < -1.0e-8):
        raise RuntimeError("diagnostic reaction RHS received an invalid species state")
    amounts_array = np.maximum(amounts_array, 0.0)
    temperature = float(y[species_count])
    if not np.isfinite(temperature) or not 250.0 <= temperature <= 520.0:
        raise RuntimeError("diagnostic reaction RHS left the temperature domain")
    volume = float(state.volume_L)
    if not np.isfinite(volume) or volume <= 0.0:
        raise RuntimeError("diagnostic reaction RHS requires finite positive volume")
    amounts = {
        species_id: float(amount)
        for species_id, amount in zip(species_ids, amounts_array, strict=True)
    }
    rates_mol_L_s = network.reaction_rates(
        amounts,
        volume_L=volume,
        temperature_K=temperature,
    )
    reactor_settings = equipment_settings(state.equipment, "batch_reactor")
    catalyst = int(reactor_settings.get("catalyst", 0))
    solvent = int(reactor_settings.get("solvent", 0))
    stirring_factor = 0.70 + 0.30 * (1.0 - np.exp(-stirring_speed_rpm / 420.0))

    derivatives = np.zeros(species_count + 4)
    reaction_heat_W = 0.0
    for reaction_index, reaction in enumerate(network.reactions):
        hidden_modifier = _hidden_reaction_modifier(
            world,
            catalyst=catalyst,
            solvent=solvent,
            reaction_index=reaction_index,
            stirring_factor=stirring_factor,
            state=state,
        )
        rate_mol_s = rates_mol_L_s[reaction.reaction_id] * volume * hidden_modifier
        for species_id, coefficient in reaction.stoichiometry.items():
            derivatives[network.species_index[species_id]] += coefficient * rate_mol_s
        reaction_heat_W += reaction.delta_h_J_per_mol * rate_mol_s

    q_jacket = 0.0
    if heat:
        q_jacket = float(np.clip((target_temperature_K - temperature) * 4.0, -70.0, 90.0))
    heat_loss = world.ua_W_per_K * (temperature - world.environment_temperature_K)
    heat_capacity = volumetric_heat_capacity(state, world.rho_cp_J_per_L_K) * volume
    if not np.isfinite(heat_capacity) or heat_capacity <= 0.0:
        raise RuntimeError("diagnostic reaction RHS requires positive heat capacity")
    derivatives[species_count] = (q_jacket - heat_loss - reaction_heat_W) / heat_capacity
    derivatives[species_count + 1] = q_jacket
    derivatives[species_count + 2] = reaction_heat_W
    derivatives[species_count + 3] = heat_loss
    return derivatives


def _hidden_reaction_modifier(
    world: Any,
    *,
    catalyst: int,
    solvent: int,
    reaction_index: int,
    stirring_factor: float,
    state: WorldState,
) -> float:
    catalyst_effects, solvent_effects = _reaction_effect_tables(world, state)
    if catalyst_effects.ndim not in {1, 2} or catalyst not in range(catalyst_effects.shape[0]):
        raise ValueError("catalyst index is outside the world effect table")
    if solvent_effects.ndim not in {1, 2} or solvent not in range(solvent_effects.shape[0]):
        raise ValueError("solvent index is outside the world effect table")
    catalyst_index = catalyst
    catalyst_reaction_index = min(reaction_index, catalyst_effects.shape[-1] - 1)
    solvent_reaction_index = min(reaction_index, solvent_effects.shape[-1] - 1)
    reactor_settings = equipment_settings(state.equipment, "batch_reactor")
    catalyst_charged = float(reactor_settings.get("catalyst_amount_mol", 0.0)) > 0.0
    catalyst_factor = (
        (
            catalyst_effects[catalyst_index, catalyst_reaction_index]
            if catalyst_effects.ndim == 2
            else catalyst_effects[catalyst_index]
        )
        if catalyst_charged
        else 1.0
    )
    solvent_factor = working_solvents(state).log_property(
        solvent_effects[:, solvent_reaction_index] if solvent_effects.ndim == 2 else solvent_effects
    )
    return float(catalyst_factor * solvent_factor * stirring_factor)


def _reaction_effect_tables(
    world: Any,
    state: WorldState,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        state.metadata.get("crystallization_material_family_id")
        == "reaction-crystallization-latent-materials-v1"
    ):
        return (
            np.asarray(world.crystallization_catalyst_effects, dtype=float),
            np.asarray(world.crystallization_solvent_effects, dtype=float),
        )
    return (
        np.asarray(world.catalyst_effects, dtype=float),
        np.asarray(world.solvent_effects, dtype=float),
    )


@dataclass(frozen=True)
class ReactionModuleSpec:
    module_id: str = "reaction"
    version: str = "0.3"
    laws: tuple[str, ...] = (
        "arrhenius_kinetics",
        "catalyst_solvent_effects",
        "byproduct_formation",
        "product_degradation",
        "catalyst_deactivation",
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "module_id": self.module_id,
            "version": self.version,
            "laws": list(self.laws),
            "gas_constant_J_per_mol_K": R_GAS,
            "runtime": "chemworld_validated_reaction_reactor_runtime_v1",
            "reactor_contract": "dynamic_batch_heat_release_jacket_sampling",
            "mechanism_contract": {
                "species": "loaded from CompiledMechanism.network.species_ids",
                "stoichiometry": "loaded from CompiledMechanism.network.reactions",
                "rate_laws": "enumerated rate-law evaluators from mechanism YAML",
            },
        }


__all__ = [
    "R_GAS",
    "ReactionIntegrationResult",
    "ReactionModuleSpec",
    "compiled_reaction_ode_rhs",
    "integrate_compiled_reaction_ode",
]
