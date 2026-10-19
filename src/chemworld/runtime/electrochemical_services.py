"""Electrochemical setup and conversion services for the transactional runtime."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemworld.foundation import (
    WorldState,
    equipment_settings,
    process_with_metrics,
    upsert_equipment_record,
)
from chemworld.physchem.electrochem_double_layer import (
    DoubleLayerRCSpec,
    simulate_double_layer_current_step,
)
from chemworld.physchem.electrochem_transport import (
    DiffusionLayerSpec,
    diffusion_layer_current_response,
)
from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_TASK_CONTRACT,
    ElectrochemicalTaskContract,
)
from chemworld.physchem.electrochemistry import (
    FARADAY_C_PER_MOL,
    R_J_PER_MOL_K,
    ElectrodeReactionSpec,
    ElectrolyteResistanceSpec,
    run_electrolysis,
)
from chemworld.physchem.equilibrium_chemistry import (
    SolubilityProductSpec,
    solve_aqueous_electrolyte_equilibrium,
)
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.world.actions import ELECTROLYTE_PROFILES, SOLVENTS
from chemworld.world.electrochemical_material_family import (
    HISTORICAL_ELECTROCHEMICAL_MATERIAL_FAMILY,
    electrochemical_material_family,
    normalize_electrochemical_material_family,
)
from chemworld.world.material_counterfactual import resolve_material_law_index
from chemworld.world.mixtures import working_solvents
from chemworld.world.parameters import ChemWorldParameters

_HISTORICAL_MATERIALS = electrochemical_material_family(HISTORICAL_ELECTROCHEMICAL_MATERIAL_FAMILY)
ELECTROLYTE_MEDIUM_PROFILE_PARAMETERS = _HISTORICAL_MATERIALS.electrolyte_profiles
ELECTROCHEMICAL_SOLVENT_PARAMETERS = _HISTORICAL_MATERIALS.solvent_profiles

# Compatibility alias for downstream characterization code.  The values are
# effective medium profiles; only profile 0 is intended to resemble an aqueous
# electrolyte and reported "pH" is a bounded proton-activity index.
AQUEOUS_ELECTROLYTE_PROFILE_PARAMETERS = ELECTROLYTE_MEDIUM_PROFILE_PARAMETERS


def _action_float(action: dict[str, Any], key: str, default: float) -> float:
    value = action.get(key, default)
    return float(np.asarray(value).reshape(-1)[0])


def _bounded_action_float(
    action: dict[str, Any],
    key: str,
    default: float,
    *,
    low: float,
    high: float,
    inclusive_low: bool = True,
) -> float:
    value = _action_float(action, key, default)
    lower_ok = value >= low if inclusive_low else value > low
    if not np.isfinite(value) or not lower_ok or value > high:
        comparator = "<=" if inclusive_low else "<"
        raise ValueError(f"{key} must satisfy {low} {comparator} value <= {high}")
    return value


def _bounded_action_index(action: dict[str, Any], key: str, count: int) -> int:
    value = action.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < count:
        raise ValueError(f"{key} must be an integer index in [0, {count - 1}]")
    return value


class ChemWorldElectrochemicalServices:
    """Apply electrochemical operating conditions and faradaic conversion."""

    def __init__(self, world: ChemWorldParameters, species_view: MechanismSpeciesView) -> None:
        self.world = world
        self.species_view = species_view
        self.task_contract = (
            ELECTROCHEMICAL_TASK_CONTRACT
            if species_view.mechanism.mechanism_id == ELECTROCHEMICAL_TASK_CONTRACT.mechanism_id
            else None
        )
        if self.task_contract is not None:
            self.task_contract.validate_compiled_mechanism(species_view.mechanism)

    def _require_task_contract(self) -> ElectrochemicalTaskContract:
        if self.task_contract is None:
            raise ValueError(
                "electrochemical operations require the electrochemical-conversion mechanism"
            )
        return self.task_contract

    def set_potential(
        self, state: WorldState, action: dict[str, Any], *, record_setpoint: bool = True
    ) -> WorldState:
        contract = self._require_task_contract()
        material_family_id = normalize_electrochemical_material_family(
            state.metadata.get("electrochemical_material_family_id")
        )
        material_family = electrochemical_material_family(material_family_id)
        electrolyte_profile = _bounded_action_index(
            action,
            "electrolyte_profile",
            len(ELECTROLYTE_PROFILES),
        )
        hidden_profile = resolve_material_law_index(
            state.metadata,
            material_field="electrolyte_profile",
            public_index=electrolyte_profile,
            catalog_size=len(ELECTROLYTE_PROFILES),
        )
        profile = material_family.electrolyte_profiles[hidden_profile]
        medium = working_solvents(state)
        solvent = medium.dominant_index
        hidden_indices = [
            resolve_material_law_index(
                state.metadata, material_field="solvent", public_index=i, catalog_size=len(SOLVENTS)
            )
            for i in range(len(SOLVENTS))
        ]
        solvent_profile = {
            key: (medium.linear_property if key.endswith("_V") else medium.log_property)(
                tuple(material_family.solvent_profiles[i][key] for i in hidden_indices)
            )
            for key in material_family.solvent_profiles[0]
        }
        electrolyte_effects: NDArray[np.float64]
        solvent_effects: NDArray[np.float64]
        if material_family_id == HISTORICAL_ELECTROCHEMICAL_MATERIAL_FAMILY:
            electrolyte_effects = np.ones(7, dtype=float)
            solvent_effects = np.concatenate(
                (
                    np.asarray(
                        [
                            medium.log_property(tuple(self.world.solvent_effects[:, j]))
                            for j in range(5)
                        ],
                        dtype=float,
                    ),
                    np.ones(2, dtype=float),
                )
            )
            electrolyte_potential_residual = 0.0
            solvent_potential_residual = 0.0
        else:
            electrolyte_effects = np.asarray(
                self.world.electrochemical_electrolyte_effects[hidden_profile],
                dtype=float,
            )
            solvent_effects = np.asarray(
                [
                    medium.log_property(
                        tuple(
                            self.world.electrochemical_solvent_effects[i, j] for i in hidden_indices
                        )
                    )
                    for j in range(7)
                ],
                dtype=float,
            )
            electrolyte_potential_residual = float(
                self.world.electrochemical_electrolyte_potential_residual_V[hidden_profile]
            )
            solvent_potential_residual = medium.linear_property(
                tuple(
                    self.world.electrochemical_solvent_potential_residual_V[i]
                    for i in hidden_indices
                )
            )
        if any(
            effects.shape != (7,) or not np.all(np.isfinite(effects)) or np.any(effects <= 0.0)
            for effects in (electrolyte_effects, solvent_effects)
        ):
            raise ValueError("world material residuals are not valid for electrochemistry")
        previous_cell_settings = equipment_settings(state.equipment, "electrochemical_cell")
        potential = _bounded_action_float(
            action,
            "potential_V",
            1.20,
            low=contract.setpoint_input_potential_bounds_V[0],
            high=contract.setpoint_input_potential_bounds_V[1],
        )
        current = _bounded_action_float(
            action,
            "current_mA",
            50.0,
            low=contract.setpoint_current_magnitude_bounds_mA[0],
            high=contract.setpoint_current_magnitude_bounds_mA[1],
        )
        conductivity = float(
            np.clip(
                profile["electrolyte_conductivity_S_m"]
                * solvent_profile["conductivity_multiplier"]
                * electrolyte_effects[0]
                * solvent_effects[0],
                0.01,
                100.0,
            )
        )
        if material_family_id == HISTORICAL_ELECTROCHEMICAL_MATERIAL_FAMILY:
            electrode_gap = float(profile["electrode_gap_m"])
            electrode_area = float(profile["electrode_area_m2"])
            contact_resistance = float(
                np.clip(
                    profile["contact_resistance_ohm"]
                    / max(
                        solvent_profile["conductivity_multiplier"] * solvent_effects[0],
                        0.02,
                    ),
                    0.0,
                    100.0,
                )
            )
        else:
            electrode_gap = self.world.electrochemical_electrode_gap_m
            electrode_area = self.world.electrochemical_electrode_area_m2
            contact_resistance = self.world.electrochemical_base_contact_resistance_ohm
        voltage_window = contract.executable_abs_potential_limit_V
        diffusivity_multiplier = max(
            solvent_profile["diffusivity_multiplier"] * electrolyte_effects[1] * solvent_effects[1],
            0.02,
        )
        diffusivity = float(
            np.clip(profile["diffusivity_m2_s"] * diffusivity_multiplier, 1.0e-12, 1.0e-7)
        )
        diffusion_layer = float(
            np.clip(
                profile["diffusion_layer_thickness_m"] / np.sqrt(diffusivity_multiplier),
                1.0e-7,
                1.0e-2,
            )
        )
        capacitance = float(
            np.clip(
                profile["double_layer_capacitance_F_m2"]
                * solvent_profile["capacitance_multiplier"]
                * electrolyte_effects[2]
                * solvent_effects[2],
                1.0e-5,
                100.0,
            )
        )
        proton_activity_multiplier = max(
            solvent_profile["proton_activity_multiplier"]
            * electrolyte_effects[3]
            * solvent_effects[3],
            1.0e-3,
        )
        acid_total = float(
            np.clip(
                profile["acid_concentration_mol_L"] * state.volume_L,
                0.0,
                max(state.volume_L, 1.0e-6),
            )
        )
        supporting_electrolyte = float(
            profile["supporting_electrolyte_concentration_mol_L"] * state.volume_L
        )
        precipitating_salt = float(
            profile["precipitating_salt_concentration_mol_L"] * state.volume_L
        )
        acid_pka = float(
            np.clip(
                profile["electrolyte_acid_pka"] - np.log10(proton_activity_multiplier),
                -2.0,
                16.0,
            )
        )
        solubility_product = float(
            np.clip(
                profile["electrolyte_ksp"]
                * solvent_profile["ksp_multiplier"]
                * electrolyte_effects[4]
                * solvent_effects[4],
                1.0e-30,
                1.0,
            )
        )
        standard_potential_shift = float(
            profile["standard_potential_shift_V"]
            + solvent_profile["standard_potential_shift_V"]
            + electrolyte_potential_residual
            + solvent_potential_residual
        )
        if material_family_id == HISTORICAL_ELECTROCHEMICAL_MATERIAL_FAMILY:
            # Historical replay retains its old empirical endpoint factors.
            faradaic_efficiency_multiplier = float(
                profile["faradaic_efficiency_multiplier"]
                * solvent_profile["faradaic_efficiency_multiplier"]
                * electrolyte_effects[5]
                * solvent_effects[5]
            )
            product_selectivity_multiplier = float(
                profile["product_selectivity_multiplier"]
                * solvent_profile["product_selectivity_multiplier"]
                * electrolyte_effects[6]
                * solvent_effects[6]
            )
        else:
            # The current material dossier exposes only foundational properties.
            # Endpoint quality is generated by hidden interface residuals and is
            # therefore not a public answer-adjacent descriptor.
            faradaic_efficiency_multiplier = float(electrolyte_effects[5] * solvent_effects[5])
            product_selectivity_multiplier = float(electrolyte_effects[6] * solvent_effects[6])
        activity_iterations_raw = 64.0
        precipitation_passes_raw = 3.0
        activity_tolerance = 1.0e-10
        setpoint_history = list(previous_cell_settings.get("setpoint_history", ()))
        if record_setpoint:
            setpoint_history.append(
                {
                    "setpoint_index": len(setpoint_history) + 1,
                    "configured_time_s": state.ledger.time_s,
                    "potential_V": potential,
                    "current_mA": current,
                    "current_setpoint_semantics": (
                        "nonnegative_magnitude_cap_signed_current_from_butler_volmer"
                    ),
                    "electrolyte_profile": electrolyte_profile,
                }
            )
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="electrochemical_cell",
            equipment_type="electrochemical_cell",
            attached_vessel_id=state.vessel_id,
            status="configured",
            settings={
                "potential_V": potential,
                "current_mA": current,
                "current_setpoint_semantics": (
                    "nonnegative_magnitude_cap_signed_current_from_butler_volmer"
                ),
                "electrolyte_profile": electrolyte_profile,
                "electrolyte_profile_id": ELECTROLYTE_PROFILES[electrolyte_profile],
                "electrolyte_profile_model": "bounded_effective_electrolyte_media_v3",
                "electrochemical_material_family_id": material_family.family_id,
                "electrochemical_material_family_sha256": material_family.family_sha256,
                "medium_thermodynamic_semantics": (
                    "effective_activity_profile_not_literal_nonaqueous_pH"
                ),
                "proton_activity_measurement_semantics": (
                    "bounded_effective_proton_activity_index"
                ),
                "material_balance_scope": (
                    "redox_network_only_electrolyte_medium_is_boundary_condition"
                ),
                "task_contract": contract.provenance(),
                "solvent": solvent,
                "solvent_id": SOLVENTS[solvent],
                "solvent_fractions": list(medium.fractions),
                "electrolyte_conductivity_S_m": conductivity,
                "electrode_gap_m": electrode_gap,
                "electrode_area_m2": electrode_area,
                "contact_resistance_ohm": contact_resistance,
                "voltage_window_V": voltage_window,
                "diffusivity_m2_s": diffusivity,
                "diffusion_layer_thickness_m": diffusion_layer,
                "double_layer_capacitance_F_m2": capacitance,
                "electrolyte_acid_total_mol": acid_total
                if record_setpoint
                else previous_cell_settings["electrolyte_acid_total_mol"],
                "electrolyte_acid_pka": acid_pka,
                "supporting_electrolyte_mol": supporting_electrolyte
                if record_setpoint
                else previous_cell_settings["supporting_electrolyte_mol"],
                "precipitating_salt_mol": precipitating_salt
                if record_setpoint
                else previous_cell_settings["precipitating_salt_mol"],
                "electrolyte_ksp": solubility_product,
                "standard_potential_shift_V": standard_potential_shift,
                "faradaic_efficiency_multiplier": faradaic_efficiency_multiplier,
                "product_selectivity_multiplier": product_selectivity_multiplier,
                "equilibrium_max_activity_iterations": int(activity_iterations_raw),
                "equilibrium_max_precipitation_passes": int(precipitation_passes_raw),
                "equilibrium_activity_tolerance": activity_tolerance,
                "setpoint_history": setpoint_history,
            },
        )
        risk = min(1.0, state.ledger.risk + 0.02 * max(abs(potential) - 1.5, 0.0))
        ledger = state.ledger.with_updates(cost=state.ledger.cost + 0.010, risk=risk)
        return state.replace(
            ledger=ledger if record_setpoint else state.ledger, equipment=equipment
        )

    def electrolyze(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        contract = self._require_task_contract()
        material_family_id = normalize_electrochemical_material_family(
            state.metadata.get("electrochemical_material_family_id")
        )
        duration = _bounded_action_float(action, "duration_s", 900.0, low=1.0, high=14_400.0)
        cell_settings = equipment_settings(state.equipment, "electrochemical_cell")
        if not cell_settings:
            raise ValueError("electrolyze requires a configured electrochemical cell")
        if cell_settings.get("solvent_fractions") != list(working_solvents(state).fractions):
            state = self.set_potential(state, dict(cell_settings), record_setpoint=False)
            cell_settings = equipment_settings(state.equipment, "electrochemical_cell")
        potential = float(cell_settings["potential_V"])
        current_mA = float(cell_settings["current_mA"])
        conductivity = float(cell_settings["electrolyte_conductivity_S_m"])
        if conductivity <= 0.0 or not np.isfinite(conductivity):
            raise ValueError("electrolyze requires a finite positive electrolyte conductivity")
        voltage_window = float(cell_settings["voltage_window_V"])
        if abs(potential) > voltage_window:
            raise ValueError("electrode potential exceeds the configured voltage window")

        try:
            aqueous = solve_aqueous_electrolyte_equilibrium(
                acid_total_mol=float(cell_settings["electrolyte_acid_total_mol"]),
                volume_L=state.volume_L,
                pka=float(cell_settings["electrolyte_acid_pka"]),
                temperature_K=state.temperature_K,
                supporting_electrolyte_mol=float(cell_settings["supporting_electrolyte_mol"]),
                precipitating_salt_mol=float(cell_settings["precipitating_salt_mol"]),
                solubility_product=SolubilityProductSpec(
                    precipitate_id="ElectrolyteSalt(s)",
                    cation_id="Salt+",
                    anion_id="Salt-",
                    ksp=float(cell_settings["electrolyte_ksp"]),
                ),
                max_precipitation_passes=int(cell_settings["equilibrium_max_precipitation_passes"]),
                max_activity_iterations=int(cell_settings["equilibrium_max_activity_iterations"]),
                activity_tolerance=float(cell_settings["equilibrium_activity_tolerance"]),
            )
        except RuntimeError as error:
            raise ValueError(f"electrolyte equilibrium failed closed: {error}") from error
        resistance_multiplier = self.world.domain_parameter("electro_resistance_multiplier")
        resistance = ElectrolyteResistanceSpec(
            electrolyte_conductivity_S_m=float(conductivity / resistance_multiplier),
            electrode_gap_m=float(cell_settings.get("electrode_gap_m", 0.004)),
            electrode_area_m2=float(cell_settings.get("electrode_area_m2", 0.004)),
            contact_resistance_ohm=(
                float(cell_settings.get("contact_resistance_ohm", 0.20)) * resistance_multiplier
            ),
            voltage_window_V=float(cell_settings.get("voltage_window_V", 2.5)),
        )
        species = state.species_amounts.copy()
        reactant = self.species_view.reactant_species(state)
        product = self.species_view.primary_target_species
        impurity = self.species_view.primary_impurity_species
        a_mol = species.get(reactant, 0.0)
        target_mol = species.get(product, 0.0)
        if a_mol <= 0.0 or not np.isfinite(a_mol):
            raise ValueError("electrolyze requires positive finite reactant inventory")
        volume = max(state.volume_L, 1.0e-9)
        reactor_settings = equipment_settings(state.equipment, "batch_reactor")
        catalyst = int(reactor_settings.get("catalyst", 0))
        if material_family_id == HISTORICAL_ELECTROCHEMICAL_MATERIAL_FAMILY:
            exchange_current_density = 28.0 * float(self.world.catalyst_effects[catalyst, 0])
            exchange_current_density *= working_solvents(state).log_property(
                tuple(self.world.solvent_effects[:, 0])
            )
        else:
            # Static S0 has no catalyst control. Use the explicit world-level
            # electrode kinetics parameter rather than an unrelated catalyst table.
            exchange_current_density = float(
                self.world.electrochemical_exchange_current_density_A_m2
            )
        exchange_current_density *= self.world.domain_parameter(
            "electro_exchange_current_multiplier"
        )
        transfer_asymmetry = self.world.domain_parameter("electro_transfer_asymmetry_multiplier")
        alpha_anodic = 0.5 * transfer_asymmetry
        alpha_cathodic = 1.0 - alpha_anodic
        selectivity_decay = self.world.domain_parameter("electro_selectivity_decay_multiplier")
        standard_potential_multiplier = self.world.domain_parameter(
            "electro_standard_potential_multiplier"
        )
        electrochemical_spec = ElectrodeReactionSpec(
            reaction_id=contract.desired_pathway_id,
            electrons_transferred=contract.electrons_transferred,
            standard_potential_V=(
                contract.standard_potential_V * standard_potential_multiplier
                + float(cell_settings["standard_potential_shift_V"])
            ),
            reaction_quotient_exponents={product: 1.0, reactant: -1.0},
            exchange_current_density_A_m2=exchange_current_density,
            electrode_area_m2=float(cell_settings["electrode_area_m2"]),
            alpha_anodic=alpha_anodic,
            alpha_cathodic=alpha_cathodic,
            faradaic_efficiency_ref=float(
                np.clip(
                    0.91 * float(cell_settings["faradaic_efficiency_multiplier"]),
                    0.05,
                    0.99,
                )
            ),
            product_selectivity_ref=float(
                np.clip(
                    0.90 * float(cell_settings["product_selectivity_multiplier"]),
                    0.05,
                    0.99,
                )
            ),
            overpotential_selectivity_sensitivity_V_inv=0.45 * selectivity_decay,
            # The declared A -> P transformation is reduction-like under the
            # Butler-Volmer convention used by the shared kernel.
            forward_current_sign=contract.forward_current_sign,
        )
        redox_activity_coefficient = aqueous.activity_coefficient_ratio**0.5
        activities = {
            reactant: max(a_mol / volume * redox_activity_coefficient, 1.0e-12),
            product: max(species.get(product, 0.0) / volume, 1.0e-12),
        }
        preliminary = run_electrolysis(
            electrochemical_spec,
            electrode_potential_V=potential,
            duration_s=duration,
            activities=activities,
            available_substrate_mol=a_mol,
            available_reverse_substrate_mol=target_mol,
            temperature_K=state.temperature_K,
            applied_current_A=current_mA / 1000.0,
            electrolyte_resistance=resistance,
        )
        if abs(preliminary.actual_current_A) <= 1.0e-12:
            raise ValueError("electrochemical driving force produced no executable current")
        if preliminary.reaction_direction < 0:
            # The declared side-product pool is a forward-reduction branch.  A
            # reverse Ox/Red operation therefore closes only the main couple.
            electrochemical_spec = replace(
                electrochemical_spec,
                product_selectivity_ref=1.0,
                overpotential_selectivity_sensitivity_V_inv=0.0,
            )
        charge_transfer_resistance = (
            R_J_PER_MOL_K
            * state.temperature_K
            / (
                electrochemical_spec.electrons_transferred
                * FARADAY_C_PER_MOL
                * max(electrochemical_spec.exchange_current_a, 1.0e-12)
            )
        )
        double_layer = simulate_double_layer_current_step(
            DoubleLayerRCSpec(
                model_id="runtime_randles_double_layer",
                double_layer_capacitance_F_m2=float(cell_settings["double_layer_capacitance_F_m2"]),
                electrode_area_m2=float(cell_settings["electrode_area_m2"]),
                series_resistance_ohm=resistance.total_resistance_ohm,
                charge_transfer_resistance_ohm=charge_transfer_resistance,
                provenance_id="chemworld_runtime_electrochemical_services",
            ),
            current_step_A=preliminary.actual_current_A,
            duration_s=duration,
            sample_interval_s=max(duration / 20.0, 1.0e-6),
        )
        directional_substrate_mol = a_mol if preliminary.reaction_direction >= 0 else target_mol
        transport = diffusion_layer_current_response(
            DiffusionLayerSpec(
                model_id="runtime_planar_diffusion_layer",
                electrons_transferred=electrochemical_spec.electrons_transferred,
                electrode_area_m2=float(cell_settings["electrode_area_m2"]),
                diffusivity_m2_s=float(cell_settings["diffusivity_m2_s"]),
                diffusion_layer_thickness_m=float(cell_settings["diffusion_layer_thickness_m"]),
                electrolyte_volume_m3=state.volume_L * 1.0e-3,
                provenance_id="chemworld_runtime_electrochemical_services",
            ),
            bulk_concentration_mol_m3=(directional_substrate_mol / (state.volume_L * 1.0e-3)),
            applied_current_A=preliminary.actual_current_A,
            duration_s=duration,
            kinetic_current_A=preliminary.kinetic_current_A,
        )
        useful_charge_limit = min(
            abs(double_layer.faradaic_charge_C),
            transport.useful_charge_C,
        )
        result = run_electrolysis(
            electrochemical_spec,
            electrode_potential_V=potential,
            duration_s=duration,
            activities=activities,
            available_substrate_mol=a_mol,
            available_reverse_substrate_mol=target_mol,
            temperature_K=state.temperature_K,
            applied_current_A=current_mA / 1000.0,
            electrolyte_resistance=resistance,
            useful_charge_limit_C=useful_charge_limit,
            capacitive_charge_C=abs(double_layer.capacitive_charge_C),
        )
        if result.reaction_direction >= 0:
            species[reactant] = a_mol - result.converted_mol
            species[product] = target_mol + result.product_mol
        else:
            species[product] = target_mol - result.converted_mol
            species[reactant] = a_mol + result.product_mol
        species[impurity] = species.get(impurity, 0.0) + result.byproduct_mol
        material_balance_residual = abs(
            result.converted_mol - result.product_mol - result.byproduct_mol
        )
        if material_balance_residual > 1.0e-12:
            raise RuntimeError("electrolysis material ledger did not close")
        if abs(result.charge_balance_residual_C) > 1.0e-9:
            raise RuntimeError("electrolysis charge ledger did not close")
        if abs(result.energy_balance_residual_J) > 1.0e-8:
            raise RuntimeError("electrolysis energy ledger did not close")
        previous_metrics = {} if state.process is None else state.process.metrics
        initial_reactant_mol = float(
            previous_metrics.get("electrochemical_initial_reactant_mol", a_mol)
        )
        initial_target_mol = float(
            previous_metrics.get("electrochemical_initial_target_mol", target_mol)
        )
        initial_impurity_mol = float(
            previous_metrics.get(
                "electrochemical_initial_impurity_mol",
                state.species_amounts.get(impurity, 0.0),
            )
        )
        net_target_mol = max(species.get(product, 0.0) - initial_target_mol, 0.0)
        net_impurity_mol = max(species.get(impurity, 0.0) - initial_impurity_mol, 0.0)
        net_reactant_consumed_mol = max(initial_reactant_mol - species[reactant], 0.0)
        electrochemical_selectivity = (
            0.0
            if net_target_mol + net_impurity_mol <= 1.0e-15
            else net_target_mol / (net_target_mol + net_impurity_mol)
        )
        selective_product_yield = net_target_mol / max(initial_reactant_mol, 1.0e-15)
        electrochemical_conversion = net_reactant_consumed_mol / max(initial_reactant_mol, 1.0e-15)
        cumulative_charge = (
            float(previous_metrics.get("cumulative_charge_C", 0.0)) + result.charge_C
        )
        cumulative_faradaic_charge = (
            float(previous_metrics.get("cumulative_faradaic_charge_C", 0.0))
            + result.faradaic_charge_C
        )
        cumulative_transport_charge = float(
            previous_metrics.get("cumulative_transport_useful_charge_C", 0.0)
        ) + min(result.faradaic_charge_C, transport.useful_charge_C)
        cumulative_electrical_work = (
            float(previous_metrics.get("cumulative_electrical_work_J", 0.0))
            + result.electrical_work_J
        )
        cumulative_ohmic_loss = (
            float(previous_metrics.get("cumulative_ohmic_loss_J", 0.0)) + result.ohmic_loss_J
        )
        cumulative_selective_reversible_work = (
            float(previous_metrics.get("cumulative_selective_reversible_work_J", 0.0))
            + result.reversible_work_J * result.product_selectivity
        )
        process = process_with_metrics(
            state.process,
            electrochemical_initial_reactant_mol=initial_reactant_mol,
            electrochemical_initial_target_mol=initial_target_mol,
            electrochemical_initial_impurity_mol=initial_impurity_mol,
            electrochemical_net_target_mol=net_target_mol,
            electrochemical_net_impurity_mol=net_impurity_mol,
            lumped_side_product_mol=net_impurity_mol,
            electrochemical_net_reactant_consumed_mol=net_reactant_consumed_mol,
            electrochemical_selectivity=float(np.clip(electrochemical_selectivity, 0.0, 1.0)),
            selective_product_yield=float(np.clip(selective_product_yield, 0.0, 1.0)),
            electrochemical_conversion=float(np.clip(electrochemical_conversion, 0.0, 1.0)),
            cumulative_charge_C=cumulative_charge,
            cumulative_faradaic_charge_C=cumulative_faradaic_charge,
            cumulative_transport_useful_charge_C=cumulative_transport_charge,
            cumulative_electrical_work_J=cumulative_electrical_work,
            cumulative_ohmic_loss_J=cumulative_ohmic_loss,
            cumulative_selective_reversible_work_J=(cumulative_selective_reversible_work),
            faradaic_efficiency=float(
                np.clip(cumulative_faradaic_charge / max(cumulative_charge, 1.0e-15), 0.0, 1.0)
            ),
            transport_efficiency=float(
                np.clip(cumulative_transport_charge / max(cumulative_charge, 1.0e-15), 0.0, 1.0)
            ),
            ohmic_efficiency=float(
                np.clip(
                    1.0 - cumulative_ohmic_loss / max(cumulative_electrical_work, 1.0e-15),
                    0.0,
                    1.0,
                )
            ),
            energy_efficiency=float(
                np.clip(
                    cumulative_selective_reversible_work / max(cumulative_electrical_work, 1.0e-15),
                    0.0,
                    1.0,
                )
            ),
            equilibrium_potential_V=result.equilibrium_potential_V,
            measured_potential_V=result.measured_potential_V,
            interfacial_potential_V=result.interfacial_potential_V,
            overpotential_V=result.overpotential_V,
            overpotential_stress_V=max(abs(result.overpotential_V) - 0.15, 0.0),
            kinetic_current_A=result.kinetic_current_A,
            actual_current_A=result.actual_current_A,
            reaction_direction=float(result.reaction_direction),
            charge_C=result.charge_C,
            faradaic_charge_C=result.faradaic_charge_C,
            electrical_work_J=result.electrical_work_J,
            interfacial_work_J=result.interfacial_work_J,
            ohmic_loss_J=result.ohmic_loss_J,
            electrolyte_resistance_ohm=result.electrolyte_resistance_ohm,
            contact_resistance_ohm=result.contact_resistance_ohm,
            total_resistance_ohm=result.total_resistance_ohm,
            uncompensated_voltage_drop_V=result.uncompensated_voltage_drop_V,
            voltage_window_exceeded=float(result.voltage_window_exceeded),
            capacitive_charge_C=result.capacitive_charge_C,
            side_reaction_charge_C=result.side_reaction_charge_C,
            charge_balance_residual_C=result.charge_balance_residual_C,
            material_balance_residual_mol=material_balance_residual,
            energy_balance_residual_J=result.energy_balance_residual_J,
            signed_terminal_work_J=result.signed_terminal_work_J,
            signed_interfacial_work_J=result.signed_interfacial_work_J,
            transport_useful_charge_C=transport.useful_charge_C,
            transport_current_efficiency=transport.current_efficiency,
            limiting_current_A=transport.initial_limiting_current_A,
            mass_transfer_limited=float(transport.mass_transfer_limited_initially),
            double_layer_time_constant_s=double_layer.time_constant_s,
            electrolyte_pH=aqueous.acid_base.pH,
            electrolyte_acid_dissociation_fraction=(aqueous.acid_base.acid_dissociation_fraction),
            electrolyte_ionic_strength_mol_kg=(aqueous.acid_base.ionic_strength_mol_kg),
            electrolyte_charge_balance_error_eq=aqueous.charge_balance_error_eq,
            electrolyte_material_balance_error_mol=aqueous.material_balance_error_mol,
            electrolyte_precipitated_mol=(aqueous.precipitation.total_precipitated_mol),
            redox_activity_coefficient=redox_activity_coefficient,
        )
        overpotential_risk = max(abs(result.overpotential_V) - 0.35, 0.0)
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + duration,
            cost=state.ledger.cost + 0.018 + result.electrical_work_J / 250_000.0,
            risk=min(
                1.0,
                state.ledger.risk
                + 0.015
                + 0.025 * abs(potential)
                + 0.035 * overpotential_risk
                + 0.020 * float(result.voltage_window_exceeded),
            ),
            energy_jacket_J=state.ledger.energy_jacket_J + result.electrical_work_J,
        )
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="electrochemical_cell",
            equipment_type="electrochemical_cell",
            attached_vessel_id=state.vessel_id,
            status="configured",
            settings={
                **cell_settings,
                "electrolysis_history": [
                    *list(cell_settings.get("electrolysis_history", ())),
                    {
                        "execution_index": len(tuple(cell_settings.get("electrolysis_history", ())))
                        + 1,
                        "start_time_s": state.ledger.time_s,
                        "end_time_s": state.ledger.time_s + duration,
                        "potential_V": potential,
                        "current_mA": current_mA,
                        "electrolyte_profile": int(cell_settings["electrolyte_profile"]),
                    },
                ],
                "runtime_model_ids": (
                    contract.runtime_model_ids[0],
                    transport.model_id,
                    double_layer.model_id,
                    contract.runtime_model_ids[3],
                ),
                "task_contract": contract.provenance(),
                "runtime_owned_pathway_ids": contract.pathway_ids,
                "mechanism_id": self.species_view.mechanism.mechanism_id,
                "mechanism_hash": self.species_view.mechanism.mechanism_hash,
                "transport_diagnostic": {
                    "model_id": transport.model_id,
                    "limiting_current_A": transport.initial_limiting_current_A,
                    "current_efficiency": transport.current_efficiency,
                    "charge_balance_residual_C": transport.charge_balance_residual_C,
                    "material_balance_residual_mol": (transport.material_balance_residual_mol),
                    "warnings": transport.warnings,
                },
                "double_layer_diagnostic": {
                    "model_id": double_layer.model_id,
                    "time_constant_s": double_layer.time_constant_s,
                    "faradaic_charge_C": double_layer.faradaic_charge_C,
                    "capacitive_charge_C": double_layer.capacitive_charge_C,
                    "charge_balance_residual_C": (double_layer.charge_balance_residual_C),
                    "warnings": double_layer.warnings,
                },
                "aqueous_equilibrium_diagnostic": {
                    "model_id": "aqueous_acid_base_ph_observation",
                    "task_semantics": "effective_electrolyte_medium_activity_index",
                    "literal_nonaqueous_pH_claimed": False,
                    "converged": aqueous.converged,
                    "iterations": aqueous.iterations,
                    "pH": aqueous.acid_base.pH,
                    "ionic_strength_mol_kg": (aqueous.acid_base.ionic_strength_mol_kg),
                    "charge_balance_error_eq": aqueous.charge_balance_error_eq,
                    "material_balance_error_mol": aqueous.material_balance_error_mol,
                    "precipitated_mol": aqueous.precipitation.total_precipitated_mol,
                    "applicability": aqueous.applicability,
                },
            },
        )
        return state.replace(
            species_amounts=species,
            ledger=ledger,
            process=process,
            equipment=equipment,
        )


__all__ = ["ChemWorldElectrochemicalServices"]
