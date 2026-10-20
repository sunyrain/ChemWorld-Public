"""Pack owned inventory for one shared advance and unpack its accepted result.

Process factories contribute physics. This adapter owns neither a solver nor a
second persistent inventory; all commits and resources use the existing runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isclose
from typing import Any

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.coupled_system import (
    CoupledEvent,
    CoupledState,
    CoupledSystem,
    HeatFlux,
)
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.inventory_interactions import with_inventory_interactions
from chemworld.physchem.reaction_network import R_J_PER_MOL_K
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult


@dataclass(frozen=True)
class ProcessFactoryContext:
    state: WorldState
    network: Any
    component: Any
    thermodynamics: Any = None
    physics: Any = None


def connected_device(state: WorldState, *, required: bool = True):
    devices = [
        record
        for record in state.equipment.equipment.values()
        if record.equipment_type == "thermal_device"
        and record.attached_vessel_id == state.vessel_id
    ]
    if not devices and not required:
        return None, None
    if len(devices) != 1:
        raise ValueError("local thermal system requires exactly one connected thermal device")
    record = devices[0]
    return record, ThermalDeviceSpec.from_dict(record.settings["thermal_device"])


def thermal_control(state: WorldState) -> ThermalDeviceSpec | None:
    if any(
        record.equipment_type == "thermal_device" and "thermal_device" in record.settings
        for record in state.equipment.equipment.values()
    ):
        return connected_device(state)[1]
    return None


def set_thermal_enabled(state: WorldState, enabled: bool) -> WorldState:
    record, device = connected_device(state)
    equipment = dict(state.equipment.equipment)
    equipment[record.equipment_id] = replace(
        record,
        settings={
            **record.settings,
            "thermal_device": replace(device, enabled=enabled).to_dict(),
        },
    )
    return state.replace(equipment=EquipmentLedger(equipment))


def reconnect_thermal_device(state: WorldState, device_id: str, destination: str) -> WorldState:
    """Return a zero-time candidate reconnecting one device, preserving identity.

    Like the material-routing helpers this is a state transform; a caller must
    commit it through its transaction boundary. No material or vessel property
    is carried along with the device.
    """
    locals_ = {state.vessel_id: state, **state.inactive_vessels}
    if destination not in locals_:
        raise ValueError("thermal destination must be a declared vessel")
    sources = [name for name, local in locals_.items() if device_id in local.equipment.equipment]
    if len(sources) != 1:
        raise ValueError("thermal device identity must have exactly one owner")
    source = sources[0]
    if source == destination:
        return state
    device = locals_[source].equipment.equipment[device_id]
    if device.equipment_type != "thermal_device":
        raise ValueError("reconnection requires a thermal device")
    if any(
        record.equipment_type == "thermal_device"
        for record in locals_[destination].equipment.equipment.values()
    ):
        raise ValueError("destination already has its thermal controller")
    moved = tuple(
        slot
        for slot in locals_[source].state_slots.slots
        if slot.spec.owner_kind == "equipment" and slot.spec.owner_id == device_id
    )
    source_equipment = dict(locals_[source].equipment.equipment)
    source_equipment.pop(device_id)
    locals_[source] = locals_[source].replace(
        equipment=EquipmentLedger(source_equipment),
        state_slots=StateSlots(
            tuple(slot for slot in locals_[source].state_slots.slots if slot not in moved)
        ),
    )
    locals_[destination] = locals_[destination].replace(
        equipment=EquipmentLedger(
            {
                **locals_[destination].equipment.equipment,
                device_id: replace(device, attached_vessel_id=destination),
            }
        ),
        state_slots=StateSlots(
            (
                *locals_[destination].state_slots.slots,
                *(replace(slot, location_id=destination) for slot in moved),
            )
        ),
    )
    active = locals_.pop(state.vessel_id)
    return active.replace(inactive_vessels=locals_)


def material_inventory(state):
    amounts = dict(state.species_amounts)
    species = {key: key for key in amounts}
    for slot in state.state_slots.slots:
        for field in slot.spec.fields:
            if field.species_id:
                key = f"{slot.spec.slot_id}:{field.name}"
                if key in amounts:
                    raise ValueError("duplicate physical inventory key")
                amounts[key] = slot.values[field.name]
                species[key] = field.species_id
    return amounts, species


def local_thermodynamics(state, model):
    amounts, species_ids = material_inventory(state)
    if set(species_ids.values()) - model.thermodynamics.species.keys():
        raise ValueError("every physical inventory requires internal energy and heat capacity")
    thermo = replace(
        model.thermodynamics,
        species={
            key: model.thermodynamics.species[species] for key, species in species_ids.items()
        },
        support_heat_capacity_J_K=(
            model.thermodynamics.support_heat_capacity_J_K
            + state.state_slots.heat_capacity()
            + (
                0
                if model.condensed_background is None
                else model.condensed_background.volumetric_cv_J_L_K * state.volume_L
            )
        ),
    )
    return with_inventory_interactions(thermo, state.state_slots.slots), CoupledState(
        amounts,
        state.temperature_K,
        (
            state.vessels.vessels[state.vessel_id].max_volume_L
            if state.volume_L == 0 and model.condensed_background is not None
            else state.volume_L
        ),
        state.vessel_elapsed_s.get(state.vessel_id, state.ledger.time_s),
    )


def assemble_local_system(state, physics, modules, components):
    model = physics.thermal_models[state.vessel_id]
    _, device = connected_device(state, required=False)
    if len(state.phases.phases) != 1:
        raise ValueError("local coupled thermal adapter requires one bulk phase")
    if model.condensed_background is not None:
        carrier = model.condensed_background
        phase = next(iter(state.phases.phases.values()))
        if (
            state.phase != "liquid"
            or not isclose(phase.solvents.volume_L, state.volume_L, rel_tol=0, abs_tol=1e-12)
            or any(
                volume > 0
                for i, volume in enumerate(phase.solvents.volumes_L)
                if i != carrier.solvent_index
            )
        ):
            raise ValueError("condensed background requires its declared liquid-volume stock")
    if model.pressure_law == "ideal_gas" and state.phase != "gas":
        raise ValueError("ideal-gas pressure requires a gas bulk phase")
    expected_pressure = pressure_for_state(state, model)
    if not isclose(state.pressure_Pa, expected_pressure, rel_tol=1e-8, abs_tol=1e-6):
        raise ValueError("gas pressure must agree with its inventory, volume and temperature")
    if expected_pressure > model.maximum_pressure_Pa + 1e-6:
        raise ValueError("state lies outside its declared pressure domain")
    amounts, species_ids = material_inventory(state)
    if not model.minimum_temperature_K <= state.temperature_K <= model.maximum_temperature_K:
        raise ValueError("state lies outside its declared thermal model domain")
    if (
        device is not None
        and not model.minimum_temperature_K <= device.setpoint_K <= model.maximum_temperature_K
    ):
        raise ValueError("device setpoint lies outside the thermal model domain")
    thermo, initial = local_thermodynamics(state, model)
    if thermo.heat_capacity_j_k(initial) <= 0:
        raise ValueError("local thermal system needs positive heat capacity")
    factories = modules.process_factories()
    processes = tuple(
        process
        for component in components
        if component.vessel == state.vessel_id
        for process_name in modules.components()[component.kind].processes
        for process in factories[process_name](
            ProcessFactoryContext(state, physics.network, component, thermo, physics)
        )
    )
    network_species = {s.species_id: s for s in physics.network.species}
    laws = {}
    for key, species in species_ids.items():
        for element, coefficient in network_species[species].composition.items():
            laws.setdefault(element, {})[key] = coefficient
    isothermal = device is not None and device.enabled and device.mode == "isothermal"
    if isothermal and not isclose(device.setpoint_K, state.temperature_K, rel_tol=0, abs_tol=1e-9):
        raise ValueError(
            "isothermal reference must start at its setpoint; use finite power for a ramp"
        )
    system = CoupledSystem(
        thermo,
        processes,
        heat_flux=lambda candidate: HeatFlux(
            0.0 if device is None else device.delivered_power_w(candidate.temperature_K),
            model.loss_conductance_W_K
            * (candidate.temperature_K - model.environment_temperature_K),
        ),
        events=(
            CoupledEvent(
                "temperature_limit",
                lambda candidate: model.stop_temperature_K - candidate.temperature_K,
            ),
            CoupledEvent(
                "minimum_temperature",
                lambda candidate: candidate.temperature_K - model.minimum_temperature_K,
            ),
            *(
                ()
                if model.pressure_law != "ideal_gas"
                else (
                    CoupledEvent(
                        "pressure_limit",
                        lambda candidate: (
                            model.maximum_pressure_Pa
                            - sum(candidate.amounts_mol[key] for key in state.species_amounts)
                            * R_J_PER_MOL_K
                            * candidate.temperature_K
                            / (candidate.volume_L / 1000)
                        ),
                    ),
                )
            ),
        ),
        conservation_laws=laws,
        inventory_scales_mol={key: max(value, 1e-8) for key, value in amounts.items()},
        isothermal=isothermal,
    )
    return system, initial


def pressure_for_state(state, model):
    if model.pressure_law == "fixed_pressure":
        return state.pressure_Pa
    return (
        sum(state.species_amounts.values())
        * R_J_PER_MOL_K
        * state.temperature_K
        / (state.volume_L / 1000)
    )


class CoupledThermalRuntime:
    def __init__(self, composition, modules):
        self.composition = composition
        self.modules = modules

    def supports(self, state):
        return (
            self.composition is not None
            and self.composition.physics is not None
            and state.vessel_id in self.composition.physics.thermal_models
            and state.vessel_id not in self.composition.physics.crystal_models
        )

    def advance(self, state, action, *, heat, jacket_program=None):
        if jacket_program is not None:
            raise ValueError(
                "native thermal devices use their own setpoint; "
                "control-program migration is separate"
            )
        record, device = connected_device(state, required=False)
        physics = self.composition.physics
        try:
            if heat:
                if device is None:
                    raise ValueError("heat requires a connected thermal device")
                device = replace(
                    device, setpoint_K=float(action["target_temperature_K"]), enabled=True
                )
            model = physics.thermal_models[state.vessel_id]
            if (
                device is not None
                and not model.minimum_temperature_K
                <= device.setpoint_K
                <= model.maximum_temperature_K
            ):
                raise ValueError("device setpoint lies outside the thermal model domain")
            if (
                device is not None
                and device.enabled
                and device.mode == "isothermal"
                and not isclose(device.setpoint_K, state.temperature_K, rel_tol=0, abs_tol=1e-9)
            ):
                raise ValueError(
                    "isothermal boundary cannot instantaneously change sample temperature"
                )
        except ValueError as exc:
            return ModelProviderResult(
                outputs={},
                diagnostics={"control_error": str(exc)},
                outcome="rejected",
                success=False,
                failure_reason="invalid_thermal_control",
            )
        equipment = dict(state.equipment.equipment)
        if record is not None:
            equipment[record.equipment_id] = replace(
                record,
                settings={
                    **record.settings,
                    "thermal_device": device.to_dict(),
                },
            )
        working = state.replace(equipment=EquipmentLedger(equipment))
        try:
            system, initial = assemble_local_system(
                working,
                physics,
                self.modules,
                self.composition.components,
            )
            result = system.advance(initial, float(action["duration_s"]))
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                outputs={},
                diagnostics={"solver_error": str(exc), "error_type": type(exc).__name__},
                outcome="numerical_failure",
                success=False,
                failure_reason="coupled_solver_failed",
            )
        final = result.final_state
        amounts = {key: final.amounts_mol[key] for key in state.species_amounts}
        slots = StateSlots(
            tuple(
                replace(
                    slot,
                    values={
                        **slot.values,
                        **{
                            field.name: final.amounts_mol[f"{slot.spec.slot_id}:{field.name}"]
                            for field in slot.spec.fields
                            if field.species_id
                        },
                    },
                )
                for slot in state.state_slots.slots
            )
        )
        phases = PhaseLedger(
            {
                key: replace(phase, species_amounts_mol=amounts)
                for key, phase in state.phases.phases.items()
            }
        )
        candidate = working.replace(
            species_amounts=amounts,
            phases=phases,
            state_slots=slots,
            temperature_K=final.temperature_K,
            ledger=advance_ledger(
                state.ledger,
                result.actual_duration_s,
                energy_deltas={
                    "energy_jacket_J": final.boundary_heat_J,
                    "heat_reaction_J": -final.chemical_energy_change_J,
                    "heat_loss_J": final.heat_loss_J,
                },
            ),
        )
        candidate = candidate.replace(
            pressure_Pa=pressure_for_state(candidate, physics.thermal_models[state.vessel_id])
        )
        return ModelProviderResult(
            outputs={},
            state=candidate,
            actual_duration_s=result.actual_duration_s,
            outcome=result.outcome,
            diagnostics={
                "stopping_event": result.stopping_event,
                "solver": dict(result.solver_diagnostic),
                "energy_balance_residual_J": result.energy_balance_residual_J,
                "conservation_residuals": dict(result.conservation_residuals),
            },
        )
