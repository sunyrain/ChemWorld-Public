"""Explicit seven-slot reaction reference fixture.

This module preserves the original A/P/B/D/E semi-mechanistic batch reaction as
an auditable reference case. Transactional reaction advancement is mechanism
compiled and lives in :mod:`chemworld.world.reaction_kernel`; new runtime code
must not import this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from chemworld.foundation import Reaction, WorldState, equipment_settings
from chemworld.physchem.solver_backend import REFERENCE_REACTION_ODE_POLICY, solve_ode
from chemworld.world.reaction_kernel import R_GAS, ReactionIntegrationResult

REFERENCE_REACTION_SPECIES = ("A", "P", "B", "D", "E", "Cat_active", "Cat_dead")


def reference_reaction_network() -> tuple[Reaction, ...]:
    return (
        Reaction("r1", "A -> P", {"A": -1.0, "P": 1.0}, -42_000.0),
        Reaction("r2", "A -> B", {"A": -1.0, "B": 1.0}, -25_000.0),
        Reaction("r3", "P -> D", {"P": -1.0, "D": 1.0}, -18_000.0),
        Reaction("r4", "A + P -> E", {"A": -1.0, "P": -1.0, "E": 1.0}, -35_000.0),
        Reaction(
            "r5",
            "Cat_active -> Cat_dead",
            {"Cat_active": -1.0, "Cat_dead": 1.0},
            -5_000.0,
        ),
    )


def integrate_seven_slot_reference_ode(
    *,
    state: WorldState,
    world: Any,
    duration_s: float,
    target_temperature_K: float,
    heat: bool,
    stirring_speed_rpm: float,
    species_map: Mapping[str, str] | None = None,
) -> ReactionIntegrationResult | None:
    """Integrate the fixed A/P/B/D/E reference reaction and thermal ODEs."""

    duration = float(np.clip(duration_s, 0.0, 14_400.0))
    if duration <= 0.0 or state.volume_L <= 0.0:
        return None

    target_temperature = float(np.clip(target_temperature_K, 250.0, 520.0))
    stirring_speed = float(np.clip(stirring_speed_rpm, 100.0, 1200.0))
    resolved_species_map = {
        species_id: (species_map or {}).get(species_id, species_id)
        for species_id in REFERENCE_REACTION_SPECIES
    }
    y0 = np.array(
        [
            state.species_amounts.get(resolved_species_map[key], 0.0)
            for key in REFERENCE_REACTION_SPECIES
        ]
        + [state.temperature_K, 0.0, 0.0, 0.0],
    )
    report = solve_ode(
        lambda _t, y: seven_slot_reference_ode_rhs(
            y=y,
            state=state,
            world=world,
            target_temperature_K=target_temperature,
            heat=heat,
            stirring_speed_rpm=stirring_speed,
        ),
        y0,
        time_span_s=(0.0, duration),
        policy=REFERENCE_REACTION_ODE_POLICY,
    )
    report.raise_for_failure("Seven-slot reference integration")
    result = report.raw_result
    y = np.maximum(result.y[:, -1], 0.0)
    species_amounts = state.species_amounts.copy()
    for index, canonical_species in enumerate(REFERENCE_REACTION_SPECIES):
        mechanism_species = resolved_species_map[canonical_species]
        species_amounts[mechanism_species] = float(y[index])
    return ReactionIntegrationResult(
        species_amounts=species_amounts,
        temperature_K=float(np.clip(y[7], 250.0, 520.0)),
        duration_s=duration,
        cost_delta=duration / 3600.0 * (0.03 if heat else 0.01),
        energy_jacket_J=float(y[8]),
        heat_reaction_J=float(y[9]),
        heat_loss_J=float(y[10]),
        stirring_speed_rpm=stirring_speed,
        solver_diagnostic=report.diagnostic.to_dict(),
        model_id="seven_slot_reference_fixture_v1",
        provider_id="chemworld_reference_fixture_only",
        provenance={
            "execution_role": "reference_fixture",
            "runtime_reachable": False,
            "species_contract": list(REFERENCE_REACTION_SPECIES),
        },
    )


def seven_slot_reference_ode_rhs(
    *,
    y: np.ndarray,
    state: WorldState,
    world: Any,
    target_temperature_K: float,
    heat: bool,
    stirring_speed_rpm: float,
) -> np.ndarray:
    amounts = np.maximum(y[:7], 0.0)
    temperature = float(np.clip(y[7], 250.0, 520.0))
    volume = max(state.volume_L, 1.0e-6)
    reactor_settings = equipment_settings(state.equipment, "batch_reactor")
    catalyst = int(reactor_settings.get("catalyst", 0))
    from chemworld.world.mixtures import volumetric_heat_capacity, working_solvents

    concentrations = amounts / volume
    cat_total = max(amounts[5] + amounts[6], 1.0e-12)
    eta_cat = amounts[5] / cat_total

    k = world.pre_exponential * np.exp(-world.activation_energy / (R_GAS * temperature))
    k *= world.catalyst_effects[catalyst] * np.array(
        [working_solvents(state).log_property(tuple(world.solvent_effects[:, j])) for j in range(5)]
    )
    stir_factor = 0.70 + 0.30 * (1.0 - np.exp(-stirring_speed_rpm / 420.0))
    low_mixing_side_penalty = 1.0 + 0.15 * (
        1.0 / (1.0 + np.exp((stirring_speed_rpm - 360.0) / 90.0))
    )
    rates = np.array(
        [
            k[0] * concentrations[0] * eta_cat * volume * stir_factor,
            k[1] * concentrations[0] * volume * low_mixing_side_penalty,
            k[2] * concentrations[1] * volume,
            k[3] * concentrations[0] * concentrations[1] * volume * low_mixing_side_penalty,
            k[4] * amounts[5],
        ],
    )
    derivatives = np.zeros(11)
    derivatives[0] = -rates[0] - rates[1] - rates[3]
    derivatives[1] = rates[0] - rates[2] - rates[3]
    derivatives[2] = rates[1]
    derivatives[3] = rates[2]
    derivatives[4] = rates[3]
    derivatives[5] = -rates[4]
    derivatives[6] = rates[4]

    q_jacket = 0.0
    if heat:
        q_jacket = float(np.clip((target_temperature_K - temperature) * 4.0, -70.0, 90.0))
    heat_loss = world.ua_W_per_K * (temperature - world.environment_temperature_K)
    heat_reaction = float(np.dot(world.delta_h_J_per_mol, rates))
    heat_capacity = max(volumetric_heat_capacity(state, world.rho_cp_J_per_L_K) * volume, 1.0e-6)
    derivatives[7] = (q_jacket - heat_loss - heat_reaction) / heat_capacity
    derivatives[8] = q_jacket
    derivatives[9] = heat_reaction
    derivatives[10] = heat_loss
    return derivatives


__all__ = [
    "REFERENCE_REACTION_SPECIES",
    "integrate_seven_slot_reference_ode",
    "reference_reaction_network",
    "seven_slot_reference_ode_rhs",
]
