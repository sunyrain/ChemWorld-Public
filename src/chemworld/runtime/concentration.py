"""Finite vacuum concentration strategy, using the common owned n/H advance."""

from dataclasses import replace
from math import ceil, isclose

import numpy as np

from chemworld.foundation.state_ledgers import EquipmentLedger
from chemworld.physchem.coupled_concentration import (
    COUPLED_CONCENTRATION_MODEL_ID,
    CoupledConcentrationState,
    advance_coupled_concentration,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.phase_inventory import (
    CompartmentLayout,
    install_compartments,
    read_compartments,
    validate_compartments,
)

MODEL_ID = COUPLED_CONCENTRATION_MODEL_ID
STATE_TYPE = "finite-concentrator-enthalpy-v1"
OUTLET_KEY = "concentration_outlet_integral"


def concentration_layout():
    return CompartmentLayout(
        MODEL_ID,
        STATE_TYPE,
        "concentrator",
        "concentration_cell",
        (
            ("concentrator_0_liquid", "concentrator_0_gas"),
            ("concentrator_1_liquid", "concentrator_1_gas"),
        ),
        "concentrator_0_liquid",
    )


def read_concentration(state, spec):
    amounts, enthalpies = read_compartments(state, spec, concentration_layout())
    counter = state.metadata.get(OUTLET_KEY, {})
    return CoupledConcentrationState(
        amounts,
        enthalpies,
        state.ledger.time_s,
        state.ledger.energy_jacket_J,
        state.ledger.heat_loss_J,
        dict(state.species.initial_amounts_mol),
        dict(counter.get("amounts_mol", {})),
        float(counter.get("enthalpy_J", 0)),
    )


def install_concentration(state, spec, inventory):
    candidate = install_compartments(state, spec, inventory, concentration_layout())
    return candidate.replace(
        metadata={
            **candidate.metadata,
            OUTLET_KEY: {
                "amounts_mol": dict(inventory.outlet_amounts_mol),
                "enthalpy_J": inventory.outlet_enthalpy_J,
                "quantity_kind": "cumulative_external_boundary_integral",
            },
        }
    )


def validate_concentration_state(state, spec):
    inventory = read_concentration(state, spec)
    validate_compartments(state, spec, concentration_layout(), inventory)
    pot = spec.at_ph(0, inventory.amounts_mol[0], inventory.enthalpies_J[0])
    if (
        spec.volume_l(pot) > spec.equipment.max_working_volume_L + 1e-12
        or sum(inventory.amounts_mol[1].values()) > spec.collector_capacity_mol + 1e-12
    ):
        raise ValueError("concentrator charge exceeds its finite equipment capacity")


class ConcentrationRuntime:
    def __init__(self, composition):
        self.composition = composition

    def supports(self, state):
        return (
            self.composition is not None
            and self.composition.physics is not None
            and state.vessel_id in self.composition.physics.concentration_models
        )

    def model(self, state):
        return self.composition.physics.concentration_models[state.vessel_id]

    def advance(self, state, action, *, heat, jacket_program=None):
        spec = self.model(state)
        try:
            record, heater = connected_device(state, required=True)
            hardware = next(
                r
                for r in state.equipment.equipment.values()
                if r.equipment_type == "finite_vacuum_concentrator"
                and r.attached_vessel_id == state.vessel_id
            )
            if (
                jacket_program is not None
                or heater.mode != "finite_power"
                or heater.maximum_power_W > spec.equipment.max_heater_power_W
            ):
                raise ValueError("concentration requires its declared finite-power control")
            operation = action.get("operation", "physical_advance")
            transport = bool(hardware.settings["transport_enabled"])
            if operation == "concentrate":
                transport = True
                heater = replace(heater, enabled=True)
            elif heat:
                heater = replace(
                    heater, enabled=True, setpoint_K=float(action["target_temperature_K"])
                )
            duration = float(action["duration_s"])
            if not 0 <= duration <= spec.equipment.maximum_duration_s:
                raise ValueError("duration leaves the concentration operating range")
        except (KeyError, StopIteration, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="invalid_concentration_control",
                diagnostics={"control_error": str(exc)},
            )
        try:
            validate_concentration_state(state, spec)
            initial = read_concentration(state, spec)
            result = advance_coupled_concentration(
                spec,
                initial,
                heater=heater,
                duration_s=duration,
                transport_enabled=transport,
                evaluation_times_s=np.linspace(0, duration, ceil(duration / 2) + 1),
                maximum_volume_L=state.vessels.vessels[state.vessel_id].max_volume_L,
            )
            # Operating stops close transport and heat at the accepted root.
            # Subsequent passive acquisition may cool the finite bottle.
            if result.stopping_event:
                transport, heater = False, replace(heater, enabled=False)
            equipment = dict(state.equipment.equipment)
            equipment[record.equipment_id] = replace(
                record, settings={**record.settings, "thermal_device": heater.to_dict()}
            )
            equipment[hardware.equipment_id] = replace(
                hardware,
                status="concentrated",
                settings={**hardware.settings, "transport_enabled": int(transport)},
            )
            candidate = install_concentration(state, spec, result.final_state).replace(
                equipment=EquipmentLedger(equipment),
                ledger=advance_ledger(
                    state.ledger,
                    result.actual_duration_s,
                    cost_delta=result.actual_duration_s / 3600 * 0.035,
                    energy_deltas={
                        "energy_jacket_J": result.final_state.heater_energy_J
                        - initial.heater_energy_J,
                        "heat_loss_J": result.final_state.condenser_energy_J
                        - initial.condenser_energy_J,
                    },
                ),
            )
            metadata = dict(candidate.metadata)
            if sum(result.vent_amounts_mol.values()) > 0:
                history = list(state.metadata.get("removed_phase_inventory_history", ()))
                history.append(
                    {
                        "operation": operation,
                        "model_id": MODEL_ID,
                        "vessel_id": state.vessel_id,
                        "interval_s": [initial.time_s, result.final_state.time_s],
                        "inventories": {
                            "concentrator_vent": {
                                "species_amounts_mol": result.vent_amounts_mol,
                                "unit": "mol",
                                "enthalpy_J": result.final_state.outlet_enthalpy_J
                                - initial.outlet_enthalpy_J,
                            }
                        },
                    }
                )
                metadata["removed_phase_inventory_history"] = history
            intervals = list(state.metadata.get("concentration_timed_outlets", ()))
            for before, after in zip(result.trajectory[:-1], result.trajectory[1:], strict=True):
                intervals.append(
                    {
                        "start_time_s": before.time_s,
                        "end_time_s": after.time_s,
                        "captured_amounts_mol": {
                            k: after.amounts_mol[1][k] - before.amounts_mol[1][k]
                            for k in spec.properties.component_ids
                        },
                        "vent_amounts_mol": {
                            k: after.outlet_amounts_mol.get(k, 0)
                            - before.outlet_amounts_mol.get(k, 0)
                            for k in spec.properties.component_ids
                        },
                    }
                )
            metadata.update(
                concentration_timed_outlets=intervals, last_concentration_advance=result.to_dict()
            )
            # The result view is a receipt. Persist no second current material DTO.
            metadata["last_concentration_advance"].pop("state")
            for key in ("pot_liquid_amounts_mol", "pot_vapor_amounts_mol",
                        "collector_liquid_amounts_mol", "collector_vapor_amounts_mol"):
                metadata["last_concentration_advance"].pop(key)
            metadata["last_concentration_advance"]["quantity_kind"] = "accepted_advance_receipt"
            candidate = candidate.replace(metadata=metadata)
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="coupled_concentration_solver_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=result.actual_duration_s,
            outcome="stopped" if result.stopping_event else "success",
            diagnostics={
                "model_id": MODEL_ID,
                "stopping_event": result.stopping_event,
                "endpoint_met": result.endpoint_met,
                "solver": result.solver_diagnostic,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
            },
        )


def validate_outlet_execution(before, after, spec, receipt):
    """Authenticate the installed producer using n/H/capture and boundary counters."""
    old, new = read_concentration(before, spec), read_concentration(after, spec)
    expected_interval = [before.ledger.time_s, after.ledger.time_s]
    if (
        receipt.get("model_id") != MODEL_ID
        or receipt.get("vessel_id") != before.vessel_id
        or receipt.get("interval_s") != expected_interval
        or set(receipt.get("inventories", {})) != {"concentrator_vent"}
    ):
        raise ValueError("invalid native concentration outlet producer or interval")
    item = receipt["inventories"]["concentrator_vent"]
    capture = spec.equipment.condenser_recovery_fraction
    for key in spec.properties.component_ids:
        vent = new.outlet_amounts_mol.get(key, 0) - old.outlet_amounts_mol.get(key, 0)
        source = old.amounts_mol[0][key] - new.amounts_mol[0][key]
        retained = new.amounts_mol[1][key] - old.amounts_mol[1][key]
        if (
            vent < -1e-12
            or not isclose(vent, item["species_amounts_mol"][key], abs_tol=1e-12)
            or not isclose(vent, (1 - capture) * source, abs_tol=1e-10)
            or not isclose(retained, capture * source, abs_tol=1e-10)
            or (key in spec.properties.nonvolatile_references and abs(vent) > 1e-14)
        ):
            raise ValueError("concentration outlet disagrees with actual phase transport")
    source_h = (
        new.heater_energy_J - old.heater_energy_J - (new.enthalpies_J[0] - old.enthalpies_J[0])
    )
    retained_h = (
        new.enthalpies_J[1]
        - old.enthalpies_J[1]
        + (new.condenser_energy_J - old.condenser_energy_J)
    )
    vent_h = new.outlet_enthalpy_J - old.outlet_enthalpy_J
    if (
        item.get("unit") != "mol"
        or not isclose(item["enthalpy_J"], vent_h, abs_tol=1e-8)
        or not isclose(vent_h, (1 - capture) * source_h, abs_tol=1e-6)
        or not isclose(retained_h, capture * source_h, abs_tol=1e-6)
    ):
        raise ValueError("concentration outlet enthalpy disagrees with shared advance")
