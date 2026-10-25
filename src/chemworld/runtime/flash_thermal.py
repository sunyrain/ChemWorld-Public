"""Constant-P enthalpy strategy selected by the common advance coordinator.

PH is an algebraic closure at each candidate H. This adapter owns no session,
resource ledger, observation loop or persistent duplicate material inventory.
"""

from dataclasses import replace
from math import isclose

from chemworld.foundation import species_with_added_initial_amounts
from chemworld.foundation.balances import MaterialPacket
from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger, PhaseRecord
from chemworld.physchem.property_reports import R_J_PER_MOL_K
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.kernel_contracts import ModelProviderResult


def inventory_enthalpy_j(state, model):
    """Derive H from the primary phase inventory, including pure plateaus."""
    from chemworld.runtime.molecular_flash_thermal import (
        MolecularFlashThermalSpec,
        molecular_flash_inventory_enthalpy_j,
    )
    from chemworld.runtime.molecular_multiphase_thermal import (
        MolecularMultiphaseThermalSpec,
        molecular_multiphase_inventory_enthalpy_j,
    )

    if isinstance(model, MolecularMultiphaseThermalSpec):
        return molecular_multiphase_inventory_enthalpy_j(state, model)
    if isinstance(model, MolecularFlashThermalSpec):
        return molecular_flash_inventory_enthalpy_j(state, model)
    inventories = {
        kind: dict.fromkeys(model.properties.component_ids, 0.0) for kind in ("liquid", "gas")
    }
    for phase in state.phases.phases.values():
        if phase.phase_type not in inventories:
            raise ValueError("VLE inventory contains an unsupported phase")
        for key, amount in phase.species_amounts_mol.items():
            inventories[phase.phase_type][key] += amount
    liquid_h, vapor_h = model.properties.inventory_enthalpies(
        state.temperature_K, model.pressure_Pa, inventories["liquid"], inventories["gas"])
    return liquid_h + vapor_h + model.support_heat_capacity_J_K * (
        state.temperature_K - model.support_reference_temperature_K)


def install_flash(state, model, result):
    if not (model.properties.temperature_bounds_K[0] - 1e-7 <= result.temperature_K
            <= model.stop_temperature_K + 1e-7):
        raise ValueError("flash candidate leaves its declared thermal operating range")
    phases = {}
    for kind, amounts in (("liquid", result.liquid_amounts_mol),
                           ("gas", result.vapor_amounts_mol)):
        total = sum(amounts.values())
        if total == 0 and result.phase_status != "empty":
            continue
        if result.phase_status == "empty" and kind == "gas":
            continue
        volume = (total * model.liquid_molar_holdup_L_mol if kind == "liquid"
                  else total * R_J_PER_MOL_K * result.temperature_K / model.pressure_Pa * 1000)
        key = f"vle_{kind}"
        phases[key] = PhaseRecord(key, state.vessel_id, kind, volume, amounts,
                                  metadata={"model": "state_gamma_phi_flash_v1",
                                            "carrier_representation": "species_inventory"})
    volume = sum(phase.volume_L for phase in phases.values())
    if volume > state.vessels.vessels[state.vessel_id].max_volume_L:
        raise ValueError("flash volume exceeds the finite vessel capacity")
    return state.replace(
        species_amounts=dict(result.feed_amounts_mol), phases=PhaseLedger(phases),
        temperature_K=result.temperature_K, pressure_Pa=model.pressure_Pa, volume_L=volume,
        phase="gas" if result.phase_status == "all_vapor" else "liquid",
        process=replace(state.process, last_observation_record=None))


def validate_flash_state(state, model):
    from chemworld.runtime.molecular_flash_thermal import (
        MolecularFlashThermalSpec,
        validate_molecular_flash_state,
    )
    from chemworld.runtime.molecular_multiphase_thermal import (
        MolecularMultiphaseThermalSpec,
        validate_molecular_multiphase_thermal_state,
    )

    if isinstance(model, MolecularMultiphaseThermalSpec):
        return validate_molecular_multiphase_thermal_state(state, model)
    if isinstance(model, MolecularFlashThermalSpec):
        return validate_molecular_flash_state(state, model)
    if not (model.properties.temperature_bounds_K[0] <= state.temperature_K
            <= model.stop_temperature_K + 1e-7):
        raise ValueError("flash state leaves its declared thermal operating range")
    if state.state_slots.slots:
        raise ValueError("this flash slice does not support additional material/equipment slots")
    if not isclose(state.pressure_Pa, model.pressure_Pa, abs_tol=1e-6, rel_tol=1e-10):
        raise ValueError("flash state pressure differs from its declared constant-P boundary")
    if len(state.phases.phases) > 2:
        raise ValueError("flash supports at most one liquid and one gas phase")
    kinds = [phase.phase_type for phase in state.phases.phases.values()]
    if len(kinds) != len(set(kinds)):
        raise ValueError("two liquid/gas phases are outside this flash slice")
    result = model.at_ph(state.species_amounts, inventory_enthalpy_j(state, model))
    if not isclose(result.temperature_K, state.temperature_K, abs_tol=1e-6, rel_tol=1e-9):
        raise ValueError("initial flash inventory is not at its declared equilibrium")
    actual = {kind: {k: sum(p.species_amounts_mol.get(k, 0) for p in state.phases.phases.values()
                            if p.phase_type == kind) for k in model.properties.component_ids}
              for kind in ("liquid", "gas")}
    for kind, expected in (("liquid", result.liquid_amounts_mol),
                            ("gas", result.vapor_amounts_mol)):
        if any(abs(actual[kind][k] - expected[k]) > 1e-8 * max(1, sum(expected.values()))
               for k in expected):
            raise ValueError("initial phase inventories do not satisfy VLE")


class FlashThermalRuntime:
    supported_operations = ("wait",)

    def reconcile_pressure(self, state):
        if self.multiphase.supports(state):
            return self.multiphase.reconcile_pressure(state)
        model = self.composition.physics.models_of("flash")[state.vessel_id]
        return state.replace(pressure_Pa=model.pressure_Pa)

    def __init__(self, composition):
        from chemworld.runtime.molecular_flash_thermal import MolecularFlashThermalRuntime
        from chemworld.runtime.molecular_multiphase_thermal import MolecularMultiphaseThermalRuntime

        self.composition = composition
        self.molecular = MolecularFlashThermalRuntime(composition)
        self.multiphase = MolecularMultiphaseThermalRuntime(composition)

    def supports(self, state):
        return (self.composition is not None and self.composition.physics is not None
                and state.vessel_id in self.composition.physics.models_of("flash"))

    def advance(self, state, action, *, heat, jacket_program=None):
        if self.multiphase.supports(state):
            return self.multiphase.advance(state, action, heat=heat, jacket_program=jacket_program)
        if self.molecular.supports(state):
            return self.molecular.advance(state, action, heat=heat, jacket_program=jacket_program)
        model = self.composition.physics.models_of("flash")[state.vessel_id]
        record, device = connected_device(state, required=False)
        try:
            if jacket_program is not None:
                raise ValueError("flash devices require their declared native control mode")
            if heat:
                if device is None:
                    raise ValueError("heat requires an attached device")
                device = replace(device, setpoint_K=float(action["target_temperature_K"]),
                                 enabled=True)
            lower, upper = model.properties.temperature_bounds_K
            if device is not None and not lower <= device.setpoint_K <= upper:
                raise ValueError("device setpoint leaves flash domain")
            if device is not None and device.enabled and device.mode == "isothermal" and (
                not isclose(device.setpoint_K, state.temperature_K, abs_tol=1e-9)
            ):
                raise ValueError("isothermal control cannot instantaneously change sample T")
        except ValueError as exc:
            return ModelProviderResult({}, success=False, outcome="rejected",
                                       failure_reason="invalid_thermal_control",
                                       diagnostics={"control_error": str(exc)})
        try:
            validate_flash_state(state, model)
            initial_h = inventory_enthalpy_j(state, model)
            minimum_h = model.at_tp(state.species_amounts, lower).total_enthalpy_j
            maximum_h = model.at_tp(state.species_amounts, upper).total_enthalpy_j
            stopping_h = model.at_tp(state.species_amounts,
                                     model.stop_temperature_K).total_enthalpy_j
            extensions = [0]

            def rhs(time, vector):
                candidate_h = float(vector[0])
                if not minimum_h <= candidate_h <= maximum_h:
                    extensions[0] += 1
                candidate = model.at_ph(state.species_amounts,
                                        min(max(candidate_h, minimum_h), maximum_h))
                temperature = candidate.temperature_K
                loss = model.loss_conductance_W_K * (temperature - model.environment_temperature_K)
                power = 0 if device is None else device.delivered_power_w(temperature)
                if device is not None and device.enabled and device.mode == "isothermal":
                    power = loss
                return [power - loss, power, loss]

            def hot(time, vector):
                return stopping_h - vector[0]

            def cold(time, vector):
                return vector[0] - minimum_h

            hot.terminal, hot.direction = True, -1
            cold.terminal, cold.direction = True, -1
            report = solve_ode(
                rhs, [initial_h, 0, 0], time_span_s=(0, float(action["duration_s"])),
                policy=ODESolverPolicy("constant_p_flash_enthalpy_v1", "DOP853", 1e-8, 1e-10,
                                       event_names=("temperature_limit", "minimum_temperature")),
                events=(hot, cold))
            report.raise_for_failure("constant-P flash advance")
            final_time, vector = float(report.t[-1]), report.y[:, -1]
            terminal = report.terminal_point
            event = None
            if terminal is not None:
                final_time, vector, index = terminal
                event = "temperature_limit" if index == 0 else "minimum_temperature"
            result = model.at_ph(state.species_amounts, float(vector[0]))
            equipment = dict(state.equipment.equipment)
            if record is not None:
                equipment[record.equipment_id] = replace(
                    record, settings={**record.settings, "thermal_device": device.to_dict()})
            candidate = install_flash(state, model, result).replace(
                equipment=EquipmentLedger(equipment),
                ledger=advance_ledger(state.ledger, final_time, energy_deltas={
                    "energy_jacket_J": float(vector[1]), "heat_loss_J": float(vector[2])}))
            residual = inventory_enthalpy_j(candidate, model) - initial_h - vector[1] + vector[2]
            if abs(residual) > 1e-6 * max(1, abs(initial_h), abs(vector[1]), abs(vector[2])):
                raise RuntimeError("constant-P thermal advance failed independent H balance")
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult({}, success=False, outcome="numerical_failure",
                                       failure_reason="flash_solver_failed",
                                       diagnostics={"solver_error": str(exc)})
        return ModelProviderResult(
            {}, state=candidate, actual_duration_s=final_time,
            outcome="stopped" if event else "success",
            diagnostics={"stopping_event": event, "solver": report.diagnostic.to_dict(),
                         "energy_balance_residual_J": float(residual),
                         "energy_balance_basis": "constant_pressure_enthalpy",
                         "trial_domain_extensions": extensions[0],
                         "flash": result.diagnostics})


def exchange_flash_packet(source, target, physics, amounts, *, fraction=0, fixed_source=False):
    from chemworld.runtime.molecular_flash_thermal import MolecularFlashThermalSpec
    from chemworld.runtime.molecular_multiphase_thermal import MolecularMultiphaseThermalSpec

    model = physics.models_of("flash")[target.vessel_id]
    if any(isinstance(physics.models_of("flash").get(vessel),
                      (MolecularFlashThermalSpec, MolecularMultiphaseThermalSpec))
           for vessel in (source.vessel_id, target.vessel_id)):
        raise ValueError("molecular thermal closure does not declare material transfer")
    if source.state_slots.slots or target.state_slots.slots:
        raise ValueError("flash packets do not support extra owned slots in this slice")
    source_model = model if fixed_source else physics.models_of("flash").get(source.vessel_id)
    if (source_model is None or source_model.properties != model.properties
            or source_model.pressure_Pa != model.pressure_Pa):
        raise ValueError("flash endpoints must share constitutive/reference/pressure data")
    available = sum(source.species_amounts.values())
    moved = sum(amounts.values())
    fraction = moved / available
    if not 0 <= fraction <= 1 or any(
        abs(amounts[k] - fraction * source.species_amounts[k]) > 1e-12 for k in amounts
    ):
        raise ValueError("flash packet must represent the actual source phase inventory")
    target_h = inventory_enthalpy_j(target, model)
    source_h = inventory_enthalpy_j(source, source_model)
    apparatus_h = source_model.support_heat_capacity_J_K * (
        source.temperature_K - source_model.support_reference_temperature_K)
    packet_h = fraction * (source_h - apparatus_h)
    packet = MaterialPacket(source.vessel_id, target.vessel_id, tuple(amounts.items()))
    remaining, received = packet.apply(source.species_amounts, target.species_amounts)
    target_candidate = install_flash(target, model, model.at_ph(received, target_h + packet_h))
    if fixed_source:
        phases = {k: replace(p, species_amounts_mol={s: v * (1 - fraction)
                                                   for s, v in p.species_amounts_mol.items()},
                             volume_L=p.volume_L * (1 - fraction))
                  for k, p in source.phases.phases.items()}
        source_candidate = source.replace(species_amounts=remaining, phases=PhaseLedger(phases),
                                           volume_L=source.volume_L * (1 - fraction))
        source_residual = None
    else:
        source_candidate = install_flash(source, source_model, source_model.at_ph(
            remaining, source_h - packet_h))
        source_residual = inventory_enthalpy_j(source_candidate, source_model) - source_h + packet_h
    references = {k: v * fraction for k, v in source.species.initial_amounts_mol.items()}
    target_candidate = target_candidate.replace(
        species=species_with_added_initial_amounts(target.species, references))
    source_candidate = source_candidate.replace(species=replace(source.species,
        initial_amounts_mol={k: v * (1 - fraction)
                            for k, v in source.species.initial_amounts_mol.items()}))
    residual = inventory_enthalpy_j(target_candidate, model) - target_h - packet_h
    if abs(residual) > 1e-6 or (source_residual is not None and abs(source_residual) > 1e-6):
        raise ValueError("flash packet violates its enthalpy balance")
    return source_candidate, target_candidate, {
        **packet.to_dict(),
        "packet_enthalpy_J": packet_h, "destination_energy_residual_J": residual,
        "source_energy_residual_J": source_residual,
        "source_thermal_boundary": "fixed_temperature_stock; utility heat/work not modelled"
        if fixed_source else "adiabatic_constant_pressure_withdrawal"}
