"""Local solid equilibrium on the shared finite-inventory enthalpy strategy."""

from dataclasses import replace
from math import isclose

import numpy as np

from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger, PhaseRecord
from chemworld.physchem.calphad_material import CefMaterialSpec
from chemworld.physchem.phase_inventory import PhaseInventoryRates, integrate_phase_inventory
from chemworld.physchem.solid_solutions import SolidSolutionSpec
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.kernel_contracts import ModelProviderResult

SOLID_TYPE = "binary_solid_equilibrium"
CEF_TYPE = "calphad_material"


def solid_record(state):
    records = [r for r in (attached_device_record(state, SOLID_TYPE),
                           attached_device_record(state, CEF_TYPE)) if r is not None]
    if len(records) > 1:
        raise ValueError("one material model must own the selected phase inventory")
    return records[0] if records else None


def solid_model(state, composition=None):
    record = solid_record(state)
    if record is None:
        raise ValueError("no material model is installed")
    if record.equipment_type == CEF_TYPE:
        if composition is None or composition.physics is None:
            raise ValueError("CEF material requires its declared physical model binding")
        model = composition.physics.models_of("calphad_material").get(state.vessel_id)
        if not isinstance(model, CefMaterialSpec) or record.settings != {
            "model_id": "calphad_material", "vessel": state.vessel_id,
        }:
            raise ValueError("CEF material record must reference its sole declared model")
        phases = tuple(state.phases.phases.values())
        if len(phases) == 1:
            if "site_fractions" not in phases[0].metadata:
                raise ValueError("CEF committed phase requires its internal site fractions")
            return model.with_sites(phases[0].metadata["site_fractions"])
        return model
    return SolidSolutionSpec.from_dict(record.settings["physical_model"])


def install_solid(state, model, equilibrium):
    total = sum(equilibrium.amounts_mol.values())
    if isinstance(model, CefMaterialSpec):
        key = f"{model.phase_model.phase_name}:0"
        phase = PhaseRecord(
            key, state.vessel_id, model.phase_type, total*model.molar_volume_L_mol,
            dict(equilibrium.amounts_mol), metadata={
                "tdb_phase": model.phase_model.phase_name,
                "constitutive_model": "calphad_material_v1", "molar_basis": "nonvacant_atoms",
                "carrier_representation": "species_inventory",
                "site_fractions": [dict(s) for s in equilibrium.constitution.site_fractions],
            },
        )
        return state.replace(
            species_amounts=dict(equilibrium.amounts_mol), phases=PhaseLedger({key: phase}),
            phase=model.phase_type, volume_L=phase.volume_L,
            temperature_K=equilibrium.temperature_K,
            pressure_Pa=model.pressure_Pa,
            process=replace(state.process, last_observation_record=None),
        )
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
    if isinstance(model, CefMaterialSpec):
        if len(state.phases.phases) != 1:
            raise ValueError("CEF material requires its one committed phase")
        phase = next(iter(state.phases.phases.values()))
        properties = model.phase_model.evaluate(
            phase.metadata["site_fractions"], state.temperature_K, state.pressure_Pa,
            pressure_derivative=False,
        )
        return sum(phase.species_amounts_mol.values())*properties.enthalpy_J_mol + (
            model.support_heat_capacity_J_K*(state.temperature_K-model.reference_temperature_K))
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
            actual.phase_type != phase.phase_type
            or set(actual.metadata) != set(phase.metadata)
            or any(actual.metadata[k] != v for k, v in phase.metadata.items()
                   if k != "site_fractions")
            or set(actual.species_amounts_mol) != set(model.component_ids)
            or not isclose(actual.volume_L, phase.volume_L, rel_tol=1e-8, abs_tol=1e-12)
            or any(
                not isclose(actual.species_amounts_mol[k], n, rel_tol=1e-7, abs_tol=1e-12)
                for k, n in phase.species_amounts_mol.items()
            )
        ):
            raise ValueError("solid phase composition is not at equilibrium")
        if isinstance(model, CefMaterialSpec):
            actual_sites = model.phase_model._fractions(actual.metadata["site_fractions"])
            if any(abs(actual_sites[i][s]-y) > 1e-7 for i, subl in enumerate(
                phase.metadata["site_fractions"]) for s, y in subl.items()):
                raise ValueError("CEF committed constitution is not internally stationary")
    if state.phase != installed.phase:
        raise ValueError("material phase type disagrees with its declared model")
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
        model = solid_model(state, self.composition)
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
            model = solid_model(state, self.composition)
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
        declared = {c.vessel for c in self.composition.components
                    if c.kind in {"solid_solution", "calphad_phase"}}
        if installed != declared:
            raise ValueError("solid equilibrium declaration must match installed material model")
        if not installed:
            return
        network = self.composition.physics.network
        for vessel in installed:
            state = states[vessel]
            model = solid_model(state, self.composition)
            components = [c for c in self.composition.components if c.vessel == vessel
                          and c.kind in {"solid_solution", "calphad_phase"}]
            expected_kind = ("calphad_phase" if isinstance(model, CefMaterialSpec)
                             else "solid_solution")
            if len(components) != 1 or components[0].kind != expected_kind:
                raise ValueError("one matching material component must own each finite specimen")
            species = [s for s in network.species if s.species_id in state.species_amounts]
            expected_symbols = (model.element_symbols if isinstance(model, CefMaterialSpec)
                                else {"A": "Al", "B": "Cu"})
            phase_type = model.phase_type if isinstance(model, CefMaterialSpec) else "solid"
            if (
                any(s.phase != phase_type or s.charge != 0 for s in species)
                or {s.species_id: s.formula for s in species} != expected_symbols
                or any(
                    component.vessel == vessel
                    and self.composition.modules.components()[component.kind].processes
                    for component in self.composition.components
                )
            ):
                raise ValueError(
                    "material requires declared elemental counters and no competing reaction"
                )
            validate_solid(state, model)
            if state.volume_L > state.vessels.vessels[vessel].max_volume_L:
                raise ValueError("solid exceeds finite vessel capacity")

    def advance(self, state, action, *, heat=False, jacket_program=None):
        model = solid_model(state, self.composition)
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
                    np.zeros((1, len(model.component_ids))), np.array([power]),
                    max(power, 0.0), max(-power, 0.0), ()
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
                constant_amounts=isinstance(model, CefMaterialSpec),
                solver_id=("calphad_material_enthalpy_v1" if isinstance(model, CefMaterialSpec)
                           else "binary_solid_equilibrium_enthalpy_v1"),
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
