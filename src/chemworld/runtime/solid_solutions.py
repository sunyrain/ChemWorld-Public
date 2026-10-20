"""Local solid equilibrium on the shared finite-inventory enthalpy strategy."""

from dataclasses import replace
from math import isclose

import numpy as np

from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger, PhaseRecord
from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.solid_solutions import SolidSolutionSpec
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.kernel_contracts import ModelProviderResult

SOLID_TYPE = "binary_solid_equilibrium"


def solid_record(state):
    return attached_device_record(state, SOLID_TYPE)


def solid_model(state):
    record = solid_record(state)
    if record is None:
        raise ValueError("no binary solid model is installed")
    return SolidSolutionSpec.from_dict(record.settings["physical_model"])


def install_solid(state, model, equilibrium):
    total = sum(equilibrium.amounts_mol.values())
    phases = {}
    for i, domain in enumerate(equilibrium.phases):
        n = total * domain.fraction
        if n <= 0:
            continue
        key = f"{domain.phase_name}:{i}"
        phases[key] = PhaseRecord(
            key,
            state.vessel_id,
            "solid",
            n * model.molar_volume_L_mol,
            {"A": n * domain.x_A, "B": n * (1 - domain.x_A)},
            metadata={
                "nominal_solid_phase": domain.phase_name,
                "carrier_representation": "species_inventory",
            },
        )
    return state.replace(
        species_amounts=dict(equilibrium.amounts_mol),
        phases=PhaseLedger(phases),
        phase="solid",
        volume_L=total * model.molar_volume_L_mol,
        temperature_K=equilibrium.temperature_K,
        pressure_Pa=model.pressure_Pa,
        process=replace(state.process, last_observation_record=None),
    )


def solid_enthalpy(state, model):
    by_name = {p.name: p for p in model.phases}
    enthalpy = model.support_heat_capacity_J_K * (
        state.temperature_K - model.reference_temperature_K
    )
    for phase in state.phases.phases.values():
        n = sum(phase.species_amounts_mol.values())
        name = phase.metadata.get("nominal_solid_phase")
        if phase.phase_type != "solid" or name not in by_name or n <= 0:
            raise ValueError("solid phases need positive finite declared inventories")
        x = phase.species_amounts_mol["A"] / n
        enthalpy += n * model.phase_enthalpy(by_name[name], x, state.temperature_K)
    return enthalpy


def validate_solid(state, model):
    if state.state_slots.slots or not isclose(state.pressure_Pa, model.pressure_Pa, abs_tol=1e-7):
        raise ValueError(
            "solid local equilibrium has no extra coupled inventory or variable pressure"
        )
    expected = model.at_tp(state.species_amounts, state.temperature_K)
    installed = install_solid(state, model, expected)
    if set(state.phases.phases) != set(installed.phases.phases):
        raise ValueError("solid phase topology is not at equilibrium")
    for key, phase in installed.phases.phases.items():
        actual = state.phases.phases[key]
        if (
            actual.phase_type != "solid"
            or actual.metadata != phase.metadata
            or set(actual.species_amounts_mol) != {"A", "B"}
            or not isclose(actual.volume_L, phase.volume_L, rel_tol=1e-8, abs_tol=1e-12)
            or any(
                not isclose(actual.species_amounts_mol[k], n, rel_tol=1e-7, abs_tol=1e-12)
                for k, n in phase.species_amounts_mol.items()
            )
        ):
            raise ValueError("solid phase composition is not at equilibrium")
    if not isclose(state.volume_L, installed.volume_L, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("solid geometric volume disagrees with finite inventory")
    return expected


class SolidSolutionRuntime:
    supported_operations = ("heat", "wait", "pause_control", "resume_control")

    def __init__(self, composition=None):
        self.composition = composition

    def supports(self, state):
        return solid_record(state) is not None

    def temperature_bounds_k(self, state):
        model = solid_model(state)
        return model.minimum_temperature_K, model.maximum_temperature_K

    def reconcile_initialization(self, transition):
        # Material replacement does not replace the specimen holder. Its
        # sensible heat equilibrates with the declared fresh solid inventory.
        if transition.receipt.get("scope") not in {"material", "batch"}:
            return transition
        active = transition.state
        locals_ = {active.vessel_id: active.replace(inactive_vessels={}), **active.inactive_vessels}
        changed = {}
        for name, receipt in transition.receipt.get("vessels", {}).items():
            state = locals_[name]
            if not self.supports(state):
                continue
            model = solid_model(state)
            retained_heat = model.support_heat_capacity_J_K * (
                receipt["previous_temperature_K"] - state.temperature_K
            )
            target = solid_enthalpy(state, model) + retained_heat
            equilibrium = model.at_ph(0, state.species_amounts, target)
            locals_[name] = install_solid(state, model, equilibrium)
            changed[name] = {
                "retained_support_heat_relative_to_template_J": retained_heat,
                "total_enthalpy_J": target,
                "temperature_K": equilibrium.temperature_K,
            }
        if not changed:
            return transition
        local = locals_.pop(active.vessel_id)
        return replace(
            transition,
            state=local.replace(inactive_vessels=locals_),
            receipt={**transition.receipt, "solid_equilibration": changed},
        )

    def validate_world(self):
        if self.composition is None or self.composition.physics is None:
            return
        states = self.composition.physics.initial_states
        installed = {v for v, s in states.items() if self.supports(s)}
        declared = {c.vessel for c in self.composition.components if c.kind == "solid_solution"}
        if installed != declared:
            raise ValueError("solid equilibrium declaration must match installed material model")
        if not installed:
            return
        network = self.composition.physics.network
        if (
            network.reactions
            or any(s.phase != "solid" or s.charge != 0 for s in network.species)
            or {s.species_id: s.formula for s in network.species}
            != {
                "A": "Al",
                "B": "Cu",
            }
        ):
            raise ValueError(
                "bounded synthetic binary solid requires declared Al/Cu counters and no reaction"
            )
        for vessel in installed:
            state = states[vessel]
            model = solid_model(state)
            validate_solid(state, model)
            if state.volume_L > state.vessels.vessels[vessel].max_volume_L:
                raise ValueError("solid exceeds finite vessel capacity")

    def advance(self, state, action, *, heat=False, jacket_program=None):
        model = solid_model(state)
        record, device = connected_device(state, required=False)
        try:
            if jacket_program is not None:
                raise ValueError("solid phase slice uses the installed device control")
            if heat:
                if device is None:
                    raise ValueError("heat requires the shared thermal device")
                device = replace(
                    device, setpoint_K=float(action["target_temperature_K"]), enabled=True
                )
            if device and (
                not model.minimum_temperature_K <= device.setpoint_K <= model.maximum_temperature_K
            ):
                raise ValueError("solid control leaves the declared thermodynamic domain")
            if (
                device
                and device.enabled
                and device.mode == "isothermal"
                and not isclose(state.temperature_K, device.setpoint_K, abs_tol=1e-9)
            ):
                raise ValueError(
                    "isothermal control cannot instantaneously change material temperature"
                )
        except ValueError as exc:
            return ModelProviderResult(
                {},
                diagnostics={"control_error": str(exc)},
                success=False,
                outcome="rejected",
                failure_reason="invalid_solid_control",
            )
        try:
            validate_solid(state, model)
            initial = TransportInventory(
                (dict(state.species_amounts),), (solid_enthalpy(state, model),), state.ledger.time_s
            )
            bounds = tuple(
                model.at_tp(state.species_amounts, t).enthalpy_J
                for t in (model.minimum_temperature_K, model.maximum_temperature_K)
            )

            def rates(candidate, equilibria):
                temperature = equilibria[0].temperature_K
                power = 0.0 if device is None else device.delivered_power_w(temperature)
                return PhaseInventoryRates(
                    np.zeros((1, 2)), np.array([power]), max(power, 0.0), max(-power, 0.0), ()
                )

            result = integrate_phase_inventory(
                model,
                initial,
                duration_s=float(action["duration_s"]),
                rates=rates,
                closure_indices=(0,),
                enthalpy_bounds=lambda i, n: bounds,
                # With no reaction or external source, this scalar finite-power
                # device always drives inward toward its validated setpoint.
                # Starting on a domain boundary must not prevent inward motion.
                stop_functions=(),
                solver_id="binary_solid_equilibrium_enthalpy_v1",
            )
            final = result.final_state
            equilibrium = model.at_ph(0, final.amounts_mol[0], final.enthalpies_J[0])
            equipment = dict(state.equipment.equipment)
            if record is not None:
                equipment[record.equipment_id] = replace(
                    record, settings={**record.settings, "thermal_device": device.to_dict()}
                )
            candidate = install_solid(state, model, equilibrium).replace(
                equipment=EquipmentLedger(equipment),
                ledger=advance_ledger(
                    state.ledger,
                    result.actual_duration_s,
                    energy_deltas={
                        "energy_jacket_J": final.heater_energy_J,
                        "heat_loss_J": final.condenser_energy_J,
                    },
                ),
            )
            residual = (
                solid_enthalpy(candidate, model)
                - initial.enthalpies_J[0]
                - final.heater_energy_J
                + final.condenser_energy_J
            )
            if abs(residual) > 1e-7 + 1e-6 * max(
                abs(initial.enthalpies_J[0]), final.heater_energy_J, final.condenser_energy_J
            ):
                raise RuntimeError("solid physical phase inventory failed energy closure")
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            return ModelProviderResult(
                {},
                diagnostics={"solver_error": str(exc)},
                success=False,
                outcome="numerical_failure",
                failure_reason="solid_equilibrium_failed",
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=result.actual_duration_s,
            outcome="stopped" if result.stopping_event else "success",
            diagnostics={
                "stopping_event": result.stopping_event,
                "solver": result.solver_diagnostic,
                "energy_balance_residual_J": residual,
                "support_deficit_J_mol": equilibrium.support_deficit_J_mol,
                "energy_scope": "constant-pressure equilibrium H and shared finite-power device",
            },
        )
