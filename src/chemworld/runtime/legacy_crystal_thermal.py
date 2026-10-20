"""Legacy condensed enthalpy boundary, sharing the finite crystal contribution.

The published liquid Cp model has no general molecular U. Its signed enthalpy
equation is retained and labelled; no Cp is renamed as Cv or given gas data.
"""

import sys
from dataclasses import dataclass, replace

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.coupled_system import HeatFlux
from chemworld.physchem.crystal_thermal_kinetics import (
    CrystalThermalSpec,
    advance_crystal_population,
)
from chemworld.physchem.crystallization_units import (
    CrystallizationKineticsSpec,
    SolubilityCurveSpec,
)
from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.reactor_shared import HeatTransferSpec
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.world.mixtures import (
    crystallization_medium_modifier,
    volumetric_heat_capacity,
    working_solvents,
)


@dataclass(frozen=True)
class LegacyCrystalEnthalpy:
    species: dict[str, float]
    support_heat_capacity_J_K: float
    reference_temperature_K: float = 298.15

    @property
    def inventory_keys(self):
        return tuple(self.species)

    def heat_capacity_j_k(self, state):
        return self.support_heat_capacity_J_K

    def internal_energy_j(self, state):
        return None

    def control_volume_enthalpy_j(self, state):
        return self.support_heat_capacity_J_K * (
            state.temperature_K - self.reference_temperature_K
        ) + sum(self.species[k] * n for k, n in state.amounts_mol.items())

    def species_energy_j_mol(self, key, temperature):
        return self.species[key]

    def chemical_energy_rate_w(self, state, derivatives, contributions):
        if any(c.legacy_reaction_energy_W is not None for c in contributions):
            raise ValueError(
                "legacy crystal contact does not certify simultaneous reactive calorics"
            )
        return sum(self.species[k] * v for k, v in derivatives.items())


class LegacyCrystalThermalRuntime:
    def __init__(self, world, species_view):
        self.world, self.species_view = world, species_view

    def available(self):
        return (
            self.world is not None
            and self.species_view is not None
            and bool(self.species_view.target_species)
            and bool(self.species_view.impurity_species)
        )

    def supports(self, state):
        return (
            self.available()
            and state.phases is not None
            and {"mother_liquor", "solid"} <= state.phases.phases.keys()
        )

    def model(self, state):
        world = self.world
        target, impurity = (
            self.species_view.primary_target_species,
            self.species_view.primary_impurity_species,
        )
        species = self.species_view.mechanism.network.species[
            self.species_view.mechanism.network.species_index[target]
        ]

        def modifier(name):
            return crystallization_medium_modifier(state, world, name)

        capacity = float(
            state.metadata.get("crystallization_impurity_occlusion_capacity_multiplier", 1)
        )
        if not 1 <= capacity <= 6:
            raise ValueError("crystallization occlusion capacity multiplier must be in [1,6]")
        curve = SolubilityCurveSpec(
            "runtime_vanthoff_material_solubility_v2",
            world.crystallization_reference_solubility_mol_L
            * world.domain_parameter("crystallization_solubility_multiplier")
            * modifier("solubility"),
            298.15,
            20000,
            250,
            430,
            "chemworld-world-law-v0.2-solubility-policy",
        )
        kinetics = CrystallizationKineticsSpec(
            "runtime_finite_cohort_kinetics",
            2e7
            * world.domain_parameter("crystallization_nucleation_multiplier")
            * modifier("nucleation"),
            2,
            2e-8 * world.domain_parameter("crystallization_growth_multiplier") * modifier("growth"),
            1,
            1200,
            molecular_weight(species.composition) / 1000,
            8e-6,
            impurity_occlusion_mol_per_mol=0.02 * modifier("occlusion") * capacity,
            supersaturation_occlusion_factor=0.5,
            impurity_occlusion_law_id=str(
                state.metadata.get(
                    "crystallization_impurity_occlusion_law_id",
                    "linear_supersaturation_transfer_v1",
                )
            ),
            impurity_surface_half_saturation_mol_L=0.01,
            provenance_id="chemworld-world-law-v0.2-crystallization-kinetics",
        )
        return CrystalThermalSpec(
            target,
            impurity,
            curve,
            kinetics,
            {target: "crystal-target", impurity: "crystal-impurity"},
        )

    def prepare(self, state):
        if self.supports(state):
            return state
        if state.phases is None or len(state.phases.phases) != 1 or state.phase != "liquid":
            raise ValueError(
                "legacy crystal preparation needs its actual single liquid compartment"
            )
        phase = next(iter(state.phases.phases.values()))
        return state.replace(
            phases=PhaseLedger(
                {
                    "mother_liquor": PhaseRecord(
                        "mother_liquor",
                        state.vessel_id,
                        "liquid",
                        state.volume_L,
                        dict(state.species_amounts),
                        selected=True,
                        settled=True,
                        solvents=working_solvents(state).at_volume(state.volume_L),
                    ),
                    "solid": PhaseRecord(
                        "solid",
                        state.vessel_id,
                        "solid",
                        0,
                        dict.fromkeys(phase.species_amounts_mol, 0.0),
                        settled=True,
                    ),
                }
            )
        )

    def advance(self, state, action, *, heat, jacket_program=None):
        from chemworld.runtime.crystal_thermal import commit_crystal_advance, population_from_state

        if not state.quenched and str(action.get("operation")) != "cool_crystallize":
            return ModelProviderResult(
                outputs={},
                success=False,
                outcome="rejected",
                failure_reason="simultaneous_legacy_reaction_crystallization_unsupported",
                diagnostics={
                    "control_error": "Quench the legacy slurry before thermal contact; "
                    "simultaneous bulk-reaction crystal calorics are not declared."
                },
            )
        try:
            original = state
            capacity = volumetric_heat_capacity(state, self.world.rho_cp_J_per_L_K) * state.volume_L
            state = self.prepare(state)
            model = self.model(state)
            duration = float(action.get("duration_s", 600))
            model = replace(model, birth_interval_s=max(30.0, duration / 120))
            population = population_from_state(state, model)
            calorics = LegacyCrystalEnthalpy(
                {
                    **dict.fromkeys(state.species_amounts, 0.0),
                    "crystal-target": -model.solubility.dissolution_enthalpy_J_mol,
                    "crystal-impurity": 0.0,
                },
                capacity,
            )
            if capacity <= 0:
                raise ValueError(
                    "legacy crystal thermal boundary needs a real positive carrier capacity"
                )
            target = float(action.get("target_temperature_K", state.temperature_K))
            if not 250 <= target <= 430:
                raise ValueError("legacy crystal setpoint lies outside its solubility data")
            thermal = HeatTransferSpec(
                rho_cp_J_per_L_K=capacity / state.volume_L,
                ua_W_per_K=float(self.world.ua_W_per_K),
                environment_temperature_K=float(self.world.environment_temperature_K),
                jacket_ua_W_per_K=4.0 if heat else 0.0,
                jacket_temperature_K=target if heat else None,
                maximum_jacket_heating_W=90,
                maximum_jacket_cooling_W=70,
            )
            start = population.time_s

            def flux(candidate):
                if jacket_program is None:
                    power = thermal.jacket_heat_w(candidate.temperature_K)
                else:
                    setpoint = jacket_program.temperature_at(candidate.time_s - start)
                    power = thermal.limit_jacket_heat_w(4.0 * (setpoint - candidate.temperature_K))
                return HeatFlux(power, thermal.heat_loss_w(candidate.temperature_K))

            result = advance_crystal_population(
                model,
                population,
                calorics,
                duration,
                heat_flux=flux,
                inventory_independent_heat_flux=True,
                progress=lambda row: print(f"Crystal advance: {row}", file=sys.stderr, flush=True),
            )
            committed = commit_crystal_advance(
                original,
                state,
                model,
                result,
                action,
                heat=heat,
                energy_basis="legacy_condensed_enthalpy",
            )
            return committed
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                outputs={},
                success=False,
                outcome="numerical_failure",
                failure_reason="legacy_crystal_solver_failed",
                diagnostics={"solver_error": str(exc), "error_type": type(exc).__name__},
            )
