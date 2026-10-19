"""Continuous-flow state-update services for the transactional runtime."""

from __future__ import annotations

import hashlib
import json
from math import pi
from typing import Any

import numpy as np

from chemworld.foundation import (
    WorldState,
    equipment_settings,
    equipment_status,
    process_with_metrics,
    upsert_equipment_record,
)
from chemworld.foundation.solvents import DENSITY_RATIOS, VISCOSITY_RATIOS
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.physchem.heat_transfer_units import TubularHeatTransferBoundarySpec
from chemworld.physchem.pfr_reactors import PFRGeometrySpec, PFRModel
from chemworld.physchem.reactor_shared import HeatTransferSpec
from chemworld.runtime.reaction_thermal_services import ChemWorldReactionThermalServices
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.world.continuous_flow import (
    FIXED_FLOW_REACTOR_INNER_DIAMETER_M,
    FIXED_FLOW_REACTOR_VOLUME_L,
)
from chemworld.world.mixtures import volumetric_heat_capacity, working_solvents


def _action_float(action: dict[str, Any], key: str, default: float) -> float:
    value = action.get(key, default)
    return float(np.asarray(value).reshape(-1)[0])


def _bounded_action_float(
    action: dict[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = _action_float(action, key, default)
    if not np.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{key} must be finite and in [{minimum}, {maximum}]")
    return value


def _feed_signature(state: WorldState) -> str:
    payload = {
        "species_amounts_mol": {
            species_id: float(amount)
            for species_id, amount in sorted(state.species_amounts.items())
        },
        "volume_L": float(state.volume_L),
        "carrier_volumes_L": working_solvents(state).volumes_L,
        "temperature_K": float(state.temperature_K),
        "initial_charge_ledger_mol": (
            {}
            if state.species is None
            else {
                species_id: float(amount)
                for species_id, amount in sorted(state.species.initial_amounts_mol.items())
            }
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _result_digest(result_payload: dict[str, object]) -> str:
    encoded = json.dumps(result_payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ChemWorldFlowServices:
    """Apply flow setup and a geometry-resolved PFR state update."""

    def __init__(
        self,
        species_view: MechanismSpeciesView,
        reaction_thermal: ChemWorldReactionThermalServices,
    ) -> None:
        self.species_view = species_view
        self.reaction_thermal = reaction_thermal
        self.world = reaction_thermal.world

    def set_flow_rate(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        flow_rate = _bounded_action_float(
            action,
            "flow_rate_mL_min",
            1.0,
            minimum=0.01,
            maximum=20.0,
        )
        requested_residence = _bounded_action_float(
            action,
            "residence_time_s",
            600.0,
            minimum=1.0,
            maximum=7200.0,
        )
        previous_configuration = equipment_settings(
            state.equipment,
            "flow_reactor:configuration",
        )
        revision = int(previous_configuration.get("configuration_revision", 0)) + 1
        volumetric_flow_L_s = flow_rate / 1000.0 / 60.0
        residence = FIXED_FLOW_REACTOR_VOLUME_L / volumetric_flow_L_s
        minimum_run_duration = residence
        inner_diameter_m = FIXED_FLOW_REACTOR_INNER_DIAMETER_M
        cross_section_area_m2 = pi * inner_diameter_m**2 / 4.0
        reactor_length_m = (FIXED_FLOW_REACTOR_VOLUME_L / 1000.0) / cross_section_area_m2
        records = {} if state.equipment is None else state.equipment.equipment.copy()
        records["flow_reactor"] = EquipmentRecord(
            equipment_id="flow_reactor",
            equipment_type="continuous_flow_reactor",
            attached_vessel_id=state.vessel_id,
            status="configured",
            settings={
                "flow_rate_mL_min": flow_rate,
                "residence_time_s": residence,
                "requested_residence_time_s": requested_residence,
                "minimum_run_duration_s": minimum_run_duration,
            },
        )
        equipment = upsert_equipment_record(
            EquipmentLedger(records),
            equipment_id="flow_reactor:configuration",
            equipment_type="continuous_flow_configuration",
            attached_vessel_id=state.vessel_id,
            status="configured",
            settings={
                "configuration_revision": revision,
                "configuration_semantic": "configure_only_no_physical_advance",
                "configured_feed_signature": _feed_signature(state),
                "hardware_contract": "fixed_volume_flow_derived_residence_v1",
                "reactor_volume_L": FIXED_FLOW_REACTOR_VOLUME_L,
                "reactor_length_m": reactor_length_m,
                "inner_diameter_m": inner_diameter_m,
                "roughness_m": 1.0e-6,
                "fluid_density_kg_m3": 950.0,
                "fluid_viscosity_Pa_s": 1.2e-3,
                "runtime_provider_id": "chemworld_geometry_resolved_pfr_v2",
            },
        )
        ledger = state.ledger.with_updates(cost=state.ledger.cost + 0.012)
        return state.replace(ledger=ledger, equipment=equipment)

    def run_flow(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        try:
            return self._run_flow(state, action)
        except (RuntimeError, ValueError, KeyError, TypeError):
            maximum_temperature = 470.0
            if state.vessels is not None and state.vessel_id in state.vessels.vessels:
                maximum_temperature = state.vessels.vessels[state.vessel_id].max_temperature_K
            return state.replace(temperature_K=maximum_temperature + 1.0)

    def _run_flow(self, state: WorldState, action: dict[str, Any]) -> WorldState:
        if equipment_status(state.equipment, "flow_reactor") != "configured":
            raise RuntimeError("run_flow requires a fresh configured flow experiment")
        flow_settings = equipment_settings(state.equipment, "flow_reactor")
        configuration = equipment_settings(state.equipment, "flow_reactor:configuration")
        if equipment_status(state.equipment, "flow_reactor:configuration") != "configured":
            raise RuntimeError("flow configuration record is not active")
        if configuration.get("configured_feed_signature") != _feed_signature(state):
            raise RuntimeError("flow feed changed after configuration; configure a new experiment")
        configured_residence = float(flow_settings["residence_time_s"])
        configured_flow_rate = float(flow_settings["flow_rate_mL_min"])
        if (
            not np.isfinite(configured_residence)
            or not 1.0 <= configured_residence <= 7200.0
            or not np.isfinite(configured_flow_rate)
            or not 0.01 <= configured_flow_rate <= 20.0
        ):
            raise RuntimeError("stored flow configuration is outside its declared domain")
        residence = float(flow_settings.get("minimum_run_duration_s", configured_residence))
        flow_rate = configured_flow_rate
        duration = _bounded_action_float(
            action,
            "duration_s",
            residence,
            minimum=1.0,
            maximum=14_400.0,
        )
        if duration < residence:
            raise ValueError("run_flow duration must reach at least one configured residence time")
        target_temperature = _bounded_action_float(
            action,
            "target_temperature_K",
            348.15,
            minimum=298.15,
            maximum=430.0,
        )
        volumetric_flow_L_s = flow_rate / 1000.0 / 60.0
        reactor_volume_L = float(configuration["reactor_volume_L"])
        inner_diameter_m = float(configuration["inner_diameter_m"])
        thermal_boundary = TubularHeatTransferBoundarySpec(
            inner_diameter_m=inner_diameter_m,
            overall_u_W_m2_K=(400.0 * self.world.domain_parameter("flow_boundary_ua_multiplier")),
            boundary_temperature_K=target_temperature,
            provenance_id="chemworld_tubular_boundary_u_times_wetted_perimeter_v1",
        )
        geometry = PFRGeometrySpec(
            length_m=float(configuration["reactor_length_m"]),
            inner_diameter_m=inner_diameter_m,
            roughness_m=float(configuration["roughness_m"]),
            fluid_density_kg_m3=float(configuration["fluid_density_kg_m3"])
            * working_solvents(state).linear_property(DENSITY_RATIOS),
            fluid_viscosity_Pa_s=float(configuration["fluid_viscosity_Pa_s"])
            * working_solvents(state).log_property(VISCOSITY_RATIOS),
            boundary_ua_W_per_m_K=thermal_boundary.conductance_per_length_W_m_K,
            boundary_temperature_K=target_temperature,
            hydraulic_provenance_id="chemworld_single_phase_darcy_weisbach_v1",
            thermal_boundary_provenance_id=thermal_boundary.provenance_id,
        )
        model = PFRModel(
            network=self.species_view.mechanism.network,
            reactor_volume_L=reactor_volume_L,
            volumetric_flow_L_s=volumetric_flow_L_s,
            geometry=geometry,
            inlet_pressure_Pa=state.pressure_Pa,
            reactor_id="chemworld_runtime_geometry_resolved_pfr_v2",
            rate_multiplier=self.world.domain_parameter("flow_rate_multiplier")
            * working_solvents(state).log_property(tuple(self.world.solvent_effects[:, 0])),
        )
        inlet_concentrations = {
            species_id: float(state.species_amounts.get(species_id, 0.0)) / state.volume_L
            for species_id in model.network.species_ids
        }
        result = model.simulate(
            inlet_concentrations,
            temperature_K=state.temperature_K,
            heat_transfer=HeatTransferSpec(
                rho_cp_J_per_L_K=volumetric_heat_capacity(state, self.world.rho_cp_J_per_L_K),
                ua_W_per_K=0.0,
                environment_temperature_K=self.world.environment_temperature_K,
            ),
            axial_positions_m=tuple(np.linspace(0.0, geometry.length_m, 9)),
        )
        if not result.diagnostics.nonnegative_species:
            raise RuntimeError("PFR result violated species nonnegativity")
        if not result.diagnostics.material_balance_closed:
            raise RuntimeError("PFR result violated material conservation")
        thermal_ledger = result.metadata.get("thermal_ledger")
        hydraulic_ledger = result.metadata.get("hydraulic_ledger")
        axial_profile = result.metadata.get("axial_profile")
        provenance = result.metadata.get("provenance")
        if not isinstance(thermal_ledger, dict):
            raise RuntimeError("PFR result is missing its thermal ledger")
        if not isinstance(hydraulic_ledger, dict):
            raise RuntimeError("PFR result is missing its hydraulic ledger")
        if not isinstance(axial_profile, tuple | list) or len(axial_profile) < 3:
            raise RuntimeError("PFR result is missing its axial diagnostics")
        if not isinstance(provenance, dict):
            raise RuntimeError("PFR result is missing provenance")
        energy_residual = float(thermal_ledger["energy_balance_residual_J"])
        energy_scale = max(
            abs(float(thermal_ledger["energy_jacket_J"]))
            + abs(float(thermal_ledger["heat_reaction_J"]))
            + abs(float(thermal_ledger["heat_loss_J"])),
            1.0,
        )
        if abs(energy_residual) > 1.0e-5 * energy_scale:
            raise RuntimeError("PFR thermal energy ledger failed to close")
        state_scale = state.volume_L / reactor_volume_L
        species_amounts = state.species_amounts.copy()
        for species_id, amount_mol in result.final_state.amounts_mol.items():
            species_amounts[species_id] = amount_mol * state_scale
        reactant_id = self.species_view.reactant_species(state)
        conversion = result.conversion(reactant_id)
        previous_metrics = {} if state.process is None else state.process.metrics
        experiment_count = int(previous_metrics.get("flow_experiment_count", 0.0)) + 1
        throughput_mL = flow_rate * duration / 60.0
        pressure_drop_payload = result.metadata.get("pressure_drop_Pa")
        if not isinstance(pressure_drop_payload, int | float):
            raise RuntimeError("PFR result is missing pressure_drop_Pa")
        hydraulic_energy_J = float(pressure_drop_payload) * throughput_mL / 1.0e6
        process = process_with_metrics(
            state.process,
            flow_conversion=conversion,
            flow_experiment_count=float(experiment_count),
            flow_campaign_time_s=(
                float(previous_metrics.get("flow_campaign_time_s", 0.0)) + duration
            ),
            flow_throughput_mL=(
                float(previous_metrics.get("flow_throughput_mL", 0.0)) + throughput_mL
            ),
            flow_hydraulic_energy_J=(
                float(previous_metrics.get("flow_hydraulic_energy_J", 0.0)) + hydraulic_energy_J
            ),
        )
        ledger = state.ledger.with_updates(
            time_s=state.ledger.time_s + duration,
            cost=state.ledger.cost + duration / 3600.0 * 0.030,
            energy_jacket_J=(
                state.ledger.energy_jacket_J + result.final_state.energy_jacket_J * state_scale
            ),
            heat_reaction_J=(
                state.ledger.heat_reaction_J + result.final_state.heat_reaction_J * state_scale
            ),
            heat_loss_J=(state.ledger.heat_loss_J + result.final_state.heat_loss_J * state_scale),
            risk=min(1.0, state.ledger.risk + 0.015 * (target_temperature > 390.0)),
        )
        pressure_payload = result.pressures_Pa
        reynolds_payload = result.metadata.get("reynolds_number")
        if not isinstance(pressure_payload, tuple | list) or not pressure_payload:
            raise RuntimeError("PFR result is missing its pressure profile")
        if not isinstance(reynolds_payload, int | float):
            raise RuntimeError("PFR result is missing reynolds_number")
        pressures = tuple(float(value) for value in pressure_payload)
        result_payload = result.to_dict()
        result_digest = _result_digest(result_payload)
        equipment = upsert_equipment_record(
            state.equipment,
            equipment_id="flow_reactor",
            equipment_type="continuous_flow_reactor",
            attached_vessel_id=state.vessel_id,
            status="completed",
            settings={
                "flow_model_id": "pfr",
                "runtime_adapter_id": "chemworld_geometry_resolved_pfr_v1",
                "runtime_provider_id": "chemworld_geometry_resolved_pfr_v2",
                "runtime_model_ids": [
                    "reaction_ode_mass_action_arrhenius_reference_slice",
                    "chemworld_geometry_resolved_pfr_v2",
                ],
                "configuration_revision": int(configuration["configuration_revision"]),
                "flow_experiment_index": experiment_count,
                "run_semantic": "advance_one_new_configured_experiment",
                "repeat_semantic": "run requires a fresh set_flow_rate configuration",
                "material_basis": "representative_feed_inventory_no_implicit_addition",
                "configured_feed_signature": configuration["configured_feed_signature"],
                "executed_feed_signature": _feed_signature(state),
                "reactor_volume_L": reactor_volume_L,
                "geometry": geometry.to_dict(),
                "thermal_boundary": thermal_boundary.to_dict(),
                "inlet_temperature_K": state.temperature_K,
                "boundary_temperature_K": target_temperature,
                "outlet_temperature_K": result.final_state.temperature_K,
                "outlet_pressure_Pa": pressures[-1],
                "pressure_drop_Pa": float(pressure_drop_payload),
                "reynolds_number": float(reynolds_payload),
                "solver_diagnostic": result.metadata["solver_diagnostic"],
                "reactor_diagnostic": result.diagnostics.to_dict(),
                "axial_profile": axial_profile,
                "thermal_ledger": thermal_ledger,
                "hydraulic_ledger": hydraulic_ledger,
                "provenance": provenance,
                "result_digest": result_digest,
                "material_balance_error_mol": result.material_balance_error_mol,
                "hydraulic_energy_J": hydraulic_energy_J,
            },
        )
        equipment = upsert_equipment_record(
            equipment,
            equipment_id="flow_reactor:configuration",
            equipment_type="continuous_flow_configuration",
            attached_vessel_id=state.vessel_id,
            status="completed",
            settings={
                "configuration_revision": int(configuration["configuration_revision"]),
                "configured_feed_signature": configuration["configured_feed_signature"],
                "executed_result_digest": result_digest,
            },
        )
        return state.replace(
            species_amounts=species_amounts,
            temperature_K=result.final_state.temperature_K,
            pressure_Pa=pressures[-1],
            ledger=ledger,
            process=process,
            equipment=equipment,
        )


__all__ = ["ChemWorldFlowServices"]
