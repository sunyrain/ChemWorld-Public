"""Project existing phase/cohort ownership into the shared crystal advance."""

import sys
from dataclasses import replace
from math import isclose

from scipy.optimize import brentq

from chemworld.foundation import equipment_settings, process_with_metrics, upsert_equipment_record
from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger
from chemworld.physchem.coupled_system import CoupledEvent, HeatFlux
from chemworld.physchem.crystal_thermal_kinetics import (
    CrystalPopulationState,
    ThermalCrystalCohort,
    advance_crystal_population,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.full_process_contract import population_settings
from chemworld.runtime.kernel_contracts import ModelProviderResult


def population_from_state(state, model):
    phases = state.phases.phases
    if set(phases) != {"mother_liquor", "solid"}:
        raise ValueError("crystal advance needs a resuspended liquid/solid compartment")
    liquid, solid = phases["mother_liquor"], phases["solid"]
    if state.volume_L == 0 and any(liquid.species_amounts_mol.values()):
        raise ValueError("zero carrier volume cannot hold a dissolved molecular stock")
    if (
        liquid.phase_type != "liquid"
        or solid.phase_type != "solid"
        or not isclose(liquid.volume_L, state.volume_L, abs_tol=1e-12)
        or state.volume_L < 0
    ):
        raise ValueError("crystal compartments must share the actual liquid volume")
    target, impurity = model.target_component, model.impurity_component
    settings = equipment_settings(state.equipment, "crystallizer")
    pairs = settings.get("population_cohorts", ())
    constituents = settings.get("population_constituents")
    if constituents is not None and len(constituents) != len(pairs):
        raise ValueError("cohort constituent rows must follow the retained size cohorts")
    target_total = solid.species_amounts_mol[target]
    impurity_total = 0 if impurity is None else solid.species_amounts_mol[impurity]
    seed_total = float(settings.get("seed_target_mol", 0.0))
    if not 0 <= seed_total <= target_total + 1e-12:
        raise ValueError("solid seed provenance exceeds its actual molecular phase")
    cohorts = []
    for index, (count, diameter) in enumerate(pairs):
        n = count * model.particle_moles(diameter)
        if constituents is None:
            fraction = n / target_total if target_total else 0
            seed, occluded = seed_total * fraction, impurity_total * fraction
        else:
            seed, occluded, n = constituents[index]
        if count == 0 and n == 0 and seed == 0 and occluded == 0:
            continue
        cohorts.append(ThermalCrystalCohort(count, diameter, occluded, seed, n))
    checks = (
        (sum(c.target_mol for c in cohorts), target_total),
        (sum(c.seed_mol for c in cohorts), seed_total),
        (sum(c.impurity_mol for c in cohorts), impurity_total),
    )
    if any(not isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in checks):
        raise ValueError("retained cohorts do not close actual solid and seed inventories")
    return CrystalPopulationState(
        dict(liquid.species_amounts_mol),
        tuple(cohorts),
        state.temperature_K,
        state.volume_L or state.vessels.vessels[state.vessel_id].max_volume_L,
        float(settings.get("dissolved_seed_target_mol", 0.0)),
        state.vessel_elapsed_s.get(state.vessel_id, state.ledger.time_s),
    )


def population_thermodynamics(state, thermal):
    if thermal.condensed_background is None:
        raise ValueError("crystal contact requires an explicitly declared condensed carrier Cv")
    return replace(
        thermal.thermodynamics,
        support_heat_capacity_J_K=(
            thermal.thermodynamics.support_heat_capacity_J_K
            + state.state_slots.heat_capacity()
            + thermal.condensed_background.volumetric_cv_J_L_K * state.volume_L
        ),
    )


def phase_material_energy(phase, temperature, model, thermal):
    calorics = thermal.thermodynamics
    energy = sum(
        n
        * calorics.species_energy_j_mol(
            model.solid_property_keys.get(k, k) if phase.phase_type == "solid" else k, temperature
        )
        for k, n in phase.species_amounts_mol.items()
    )
    if phase.phase_type == "liquid":
        energy += (
            thermal.condensed_background.volumetric_cv_J_L_K
            * phase.volume_L
            * (temperature - calorics.reference_temperature_K)
        )
    return energy


def crystal_inventory_energy(state, model, thermal):
    reference = thermal.thermodynamics.reference_temperature_K
    return (
        thermal.thermodynamics.support_heat_capacity_J_K + state.state_slots.heat_capacity()
    ) * (state.temperature_K - reference) + sum(
        phase_material_energy(phase, state.temperature_K, model, thermal)
        for phase in state.phases.phases.values()
    )


def validate_crystal_state(state, model, thermal):
    if state.pressure_Pa > thermal.maximum_pressure_Pa:
        raise ValueError("crystal pressure exceeds its declared compartment domain")
    if not thermal.minimum_temperature_K <= state.temperature_K <= thermal.maximum_temperature_K:
        raise ValueError("crystal temperature lies outside its declared thermal domain")
    population = population_from_state(state, model)
    if any(field.species_id for slot in state.state_slots.slots for field in slot.spec.fields):
        raise ValueError(
            "this crystal thermal slice excludes additional bound molecular reservoirs"
        )
    calorics = population_thermodynamics(state, thermal)
    if model.target_component not in state.species_amounts or (
        model.impurity_component is not None
        and model.impurity_component not in state.species_amounts
    ):
        raise ValueError("crystal species must belong to the actual bulk inventory")
    required = set(state.species_amounts) | set(model.solid_property_keys.values())
    if set(calorics.species) != required or thermal.pressure_law != "fixed_pressure":
        raise ValueError("crystal contact needs explicit liquid/solid calorics at fixed pressure")
    for name in state.species_amounts:
        actual = sum(p.species_amounts_mol[name] for p in state.phases.phases.values())
        if not isclose(actual, state.species_amounts[name], abs_tol=1e-12):
            raise ValueError("crystal phase molecular amounts disagree with bulk inventory")
    model.solubility.solubility_mol_per_l(population.temperature_K)
    for temperature in (
        model.solubility.minimum_temperature_K,
        population.temperature_K,
        model.solubility.maximum_temperature_K,
    ):
        difference = calorics.species_energy_j_mol(
            model.target_component, temperature
        ) - calorics.species_energy_j_mol(
            model.solid_property_keys[model.target_component], temperature
        )
        if not isclose(
            difference, model.solubility.dissolution_enthalpy_J_mol, rel_tol=1e-10, abs_tol=1e-6
        ):
            raise ValueError("crystal calorics disagree with solubility enthalpy")
    return population, calorics


def reconcile_crystal_initialization(previous, fresh, transition, physics):
    scope = transition.receipt["scope"]
    if scope not in {"material", "vessel"}:
        return transition

    def locals_of(state):
        return {state.vessel_id: state.replace(inactive_vessels={}), **state.inactive_vessels}

    old, templates, current = locals_of(previous), locals_of(fresh), locals_of(transition.state)
    receipts = {}
    for name in transition.receipt["vessels"]:
        if name not in physics.crystal_models:
            continue
        model, thermal = physics.crystal_models[name], physics.thermal_models[name]
        state = current[name]
        initial_t, retained_t = templates[name].temperature_K, old[name].temperature_K
        capacity = thermal.thermodynamics.support_heat_capacity_J_K if scope == "material" else 0.0
        capacity += sum(
            slot.values[slot.spec.heat_capacity_field]
            for slot in state.state_slots.slots
            if slot.spec.heat_capacity_field is not None
            and (
                slot.spec.owner_kind == "equipment"
                or (scope == "material" and slot.spec.owner_kind == "vessel")
            )
        )
        expected = crystal_inventory_energy(state.replace(temperature_K=initial_t), model, thermal)
        expected += capacity * (retained_t - initial_t)
        temperature = brentq(
            lambda t, state=state, model=model, thermal=thermal, expected=expected: (
                crystal_inventory_energy(state.replace(temperature_K=t), model, thermal) - expected
            ),
            thermal.minimum_temperature_K,
            thermal.maximum_temperature_K,
            xtol=1e-10,
        )
        current[name] = state.replace(temperature_K=temperature)
        receipts[name] = {
            "retained_thermal_capacity_J_K": capacity,
            "expected_internal_energy_J": expected,
            "actual_internal_energy_J": crystal_inventory_energy(current[name], model, thermal),
            "temperature_K": temperature,
        }
    if not receipts:
        return transition
    active = current.pop(transition.state.vessel_id)
    return replace(
        transition,
        state=active.replace(inactive_vessels=current),
        receipt={**transition.receipt, "crystal_thermal_reconciliation": receipts},
    )


def install_population(state, model, population):
    phases = state.phases.phases
    solid = dict.fromkeys(state.species_amounts, 0.0)
    solid[model.target_component] = sum(c.target_mol for c in population.cohorts)
    if model.impurity_component is not None:
        solid[model.impurity_component] = sum(c.impurity_mol for c in population.cohorts)
    phases = PhaseLedger(
        {
            "mother_liquor": replace(
                phases["mother_liquor"], species_amounts_mol=dict(population.liquid_amounts_mol)
            ),
            "solid": replace(phases["solid"], species_amounts_mol=solid),
        }
    )
    equipment = upsert_equipment_record(
        state.equipment,
        equipment_id="crystallizer",
        equipment_type="crystallizer",
        attached_vessel_id=state.vessel_id,
        settings={
            **population_settings(
                tuple((c.particle_count, c.diameter_m) for c in population.cohorts)
            ),
            "population_constituents": [
                [c.seed_mol, c.impurity_mol, c.target_mol] for c in population.cohorts
            ],
            "seed_target_mol": sum(c.seed_mol for c in population.cohorts),
            "dissolved_seed_target_mol": population.dissolved_seed_mol,
        },
    )
    return state.replace(
        phases=phases,
        species_amounts=phases.total_amounts_mol(),
        temperature_K=population.temperature_K,
        equipment=equipment,
    )


class CrystalThermalRuntime:
    def __init__(self, composition, *, legacy_world=None, species_view=None):
        self.composition = composition
        from chemworld.runtime.legacy_crystal_thermal import LegacyCrystalThermalRuntime

        self.legacy = LegacyCrystalThermalRuntime(legacy_world, species_view)

    def is_native(self, state):
        return (
            self.composition is not None
            and self.composition.physics is not None
            and state.vessel_id in self.composition.physics.crystal_models
        )

    def legacy_available(self):
        return (
            self.composition is None or self.composition.physics is None
        ) and self.legacy.available()

    def supports(self, state):
        return self.is_native(state) or (self.legacy_available() and self.legacy.supports(state))

    def model(self, state):
        return (
            self.composition.physics.crystal_models[state.vessel_id]
            if self.is_native(state)
            else self.legacy.model(state)
        )

    def seed_population(self, before, after, amount, diameter=100e-6):
        if not self.is_native(before):
            before = self.legacy.prepare(before)
        model = self.model(before)
        population = population_from_state(before, model)
        if amount:
            population = replace(
                population,
                cohorts=(
                    *population.cohorts,
                    ThermalCrystalCohort(
                        amount / model.particle_moles(diameter), diameter, 0.0, amount, amount
                    ),
                ),
            )
        return install_population(after, model, population)

    def filter(self, state):
        result = self.advance(
            state, {"operation": "filter_crystals", "duration_s": 480}, heat=False
        )
        if result.state is None or result.outcome != "success":
            return result
        candidate = result.state
        liquid = candidate.phases.phases["mother_liquor"]
        phases = PhaseLedger(
            {
                **candidate.phases.phases,
                "mother_liquor": replace(
                    liquid,
                    volume_L=liquid.volume_L * 0.92,
                    solvents=liquid.solvents.scale(0.92),
                    species_amounts_mol={
                        k: n * 0.92 for k, n in liquid.species_amounts_mol.items()
                    },
                ),
                "cake_liquor": replace(
                    liquid,
                    phase_id="cake_liquor",
                    selected=False,
                    volume_L=liquid.volume_L * 0.08,
                    solvents=liquid.solvents.scale(0.08),
                    species_amounts_mol={
                        k: n * 0.08 for k, n in liquid.species_amounts_mol.items()
                    },
                ),
            }
        )
        equipment = upsert_equipment_record(
            candidate.equipment,
            equipment_id="crystal_filter",
            equipment_type="solid_liquid_filter",
            attached_vessel_id=state.vessel_id,
            status="completed",
            settings={
                "crystals_filtered": True,
                "target_solid_retention_fraction": 1.0,
                "impurity_solid_retention_fraction": 1.0,
                "cake_liquid_fraction": 0.08,
                "model_id": "ideal_retention_finite_wet_cake",
            },
        )
        return replace(
            result,
            state=candidate.replace(
                phases=phases,
                equipment=equipment,
                ledger=candidate.ledger.with_updates(cost=candidate.ledger.cost + 0.026),
            ),
        )

    def resuspend(self, before, action, existing_resuspend):
        model = self.model(before)
        thermal = self.composition.physics.thermal_models[before.vessel_id]
        bg = thermal.condensed_background
        if int(action["solvent"]) != bg.solvent_index:
            raise ValueError("resuspension carrier needs its declared caloric stock identity")
        old_energy = crystal_inventory_energy(before, model, thermal)
        filtrate = before.phases.phases["mother_liquor"]
        filtrate_energy = phase_material_energy(filtrate, before.temperature_K, model, thermal)
        inlet_energy = (
            bg.volumetric_cv_J_L_K
            * float(action["volume_L"])
            * (bg.feed_temperature_K - thermal.thermodynamics.reference_temperature_K)
        )
        candidate = existing_resuspend(before, action)
        expected = old_energy - filtrate_energy + inlet_energy
        temperature = brentq(
            lambda t: (
                crystal_inventory_energy(candidate.replace(temperature_K=t), model, thermal)
                - expected
            ),
            thermal.minimum_temperature_K,
            thermal.maximum_temperature_K,
            xtol=1e-10,
        )
        candidate = candidate.replace(temperature_K=temperature)
        receipt = {
            "operation": "resuspend_crystals",
            "filtrate_energy_J": filtrate_energy,
            "inlet_carrier_energy_J": inlet_energy,
            "inlet_temperature_K": bg.feed_temperature_K,
            "energy_balance_residual_J": crystal_inventory_energy(candidate, model, thermal)
            - expected,
        }
        return candidate.replace(
            metadata={**candidate.metadata, "last_crystal_material_packet": receipt}
        )

    def advance(self, state, action, *, heat, jacket_program=None):
        if not self.is_native(state):
            return self.legacy.advance(state, action, heat=heat, jacket_program=jacket_program)
        if jacket_program is not None:
            raise ValueError("crystal device programs use their existing setpoint controls")
        model = self.model(state)
        thermal = self.composition.physics.thermal_models[state.vessel_id]
        record, device = connected_device(state, required=False)
        try:
            if heat:
                if device is None:
                    raise ValueError("crystal thermal control needs a connected finite device")
                device = replace(
                    device, setpoint_K=float(action["target_temperature_K"]), enabled=True
                )
            if device is not None and (
                not thermal.minimum_temperature_K
                <= device.setpoint_K
                <= thermal.maximum_temperature_K
            ):
                raise ValueError("crystal setpoint lies outside declared thermal validity")
            if (
                device is not None
                and device.enabled
                and device.mode == "isothermal"
                and (not isclose(state.temperature_K, device.setpoint_K, abs_tol=1e-9))
            ):
                raise ValueError(
                    "isothermal control cannot instantaneously change crystal temperature"
                )
            population, calorics = validate_crystal_state(state, model, thermal)
        except ValueError as exc:
            return ModelProviderResult(
                outputs={},
                success=False,
                outcome="rejected",
                failure_reason="invalid_crystal_control",
                diagnostics={"control_error": str(exc)},
            )
        working = state
        if record is not None:
            records = dict(state.equipment.equipment)
            records[record.equipment_id] = replace(
                record, settings={**record.settings, "thermal_device": device.to_dict()}
            )
            working = state.replace(equipment=EquipmentLedger(records))
        try:
            active_model = model
            if state.volume_L == 0:
                # Geometric volume only supports the numerical temperature view;
                # zero real solvent disables growth, dissolution and nucleation.
                active_model = replace(
                    model,
                    kinetics=replace(
                        model.kinetics,
                        growth_coefficient_m_s=0.0,
                        primary_nucleation_coefficient_per_L_s=0.0,
                    ),
                )
            result = advance_crystal_population(
                active_model,
                population,
                calorics,
                float(action["duration_s"]),
                heat_flux=lambda s: HeatFlux(
                    0 if device is None else device.delivered_power_w(s.temperature_K),
                    thermal.loss_conductance_W_K
                    * (s.temperature_K - thermal.environment_temperature_K),
                ),
                isothermal=(device is not None and device.enabled and device.mode == "isothermal"),
                events=(
                    CoupledEvent(
                        "thermal_temperature_limit",
                        lambda s: thermal.stop_temperature_K - s.temperature_K,
                    ),
                    CoupledEvent(
                        "thermal_minimum_temperature",
                        lambda s: s.temperature_K - thermal.minimum_temperature_K,
                    ),
                ),
                progress=lambda row: print(f"Crystal advance: {row}", file=sys.stderr, flush=True),
                inventory_independent_heat_flux=True,
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                outputs={},
                success=False,
                outcome="numerical_failure",
                failure_reason="crystal_solver_failed",
                diagnostics={"solver_error": str(exc), "error_type": type(exc).__name__},
            )
        return commit_crystal_advance(state, working, model, result, action, heat=heat)


def commit_crystal_advance(
    state, working, model, result, action, *, heat=False, energy_basis="internal_energy"
):
    candidate = install_population(working, model, result.final_state)
    old_settings = equipment_settings(state.equipment, "crystallizer")
    initial_solid = sum(c.target_mol for c in result.initial_state.cohorts)
    solid_target = sum(c.target_mol for c in result.final_state.cohorts)
    seed = sum(c.seed_mol for c in result.final_state.cohorts)
    impurity = sum(c.impurity_mol for c in result.final_state.cohorts)
    grown = max(solid_target - seed, 0.0)
    metrics = {} if state.process is None else state.process.metrics
    initial_seed = (
        sum(c.seed_mol for c in result.initial_state.cohorts)
        + result.initial_state.dissolved_seed_mol
    )
    reference = max(
        float(
            metrics.get(
                "pre_separation_product_mol",
                state.species_amounts[model.target_component] - initial_seed,
            )
        ),
        1e-12,
    )
    current_settings = equipment_settings(candidate.equipment, "crystallizer")
    recovery = min(max(grown / reference, 0.0), 1.0)
    purity = solid_target / (solid_target + impurity) if solid_target + impurity else 0.0
    component_errors = {
        key: abs(candidate.species_amounts[key] - state.species_amounts[key])
        for key in state.species_amounts
    }
    temperatures = [s.temperature_K for s in result.trajectory]
    saturations = [
        s.liquid_amounts_mol[model.target_component]
        / s.volume_L
        / model.solubility.solubility_mol_per_l(s.temperature_K)
        for s in result.trajectory
    ]
    history = list(old_settings.get("execution_history", ()))
    receipt = {
        "model_id": "finite-crystal-cohorts-temperature-v1",
        "energy_basis": energy_basis,
        "operation": action.get("operation", "heat" if heat else "wait"),
        "actual_duration_s": result.actual_duration_s,
        "initial_temperature_K": state.temperature_K,
        "final_temperature_K": candidate.temperature_K,
        "jacket_energy_J": result.boundary_heat_J,
        "heat_loss_J": result.heat_loss_J,
        "phase_change_heat_J": result.chemical_energy_change_J,
        "energy_balance_residual_J": result.energy_balance_residual_J,
        "stopping_event": result.stopping_event,
        "birth_interval_s": model.birth_interval_s,
        "prior_solid_target_mol": initial_solid,
        "final_solid_target_mol": solid_target,
        "net_redissolved_target_mol": max(initial_solid - solid_target, 0.0),
        "impurity_resolution_fraction": model.impurity_resolution_fraction,
        "maximum_numerical_molar_correction_mol": max(
            (
                value
                for block in result.solver_diagnostics
                for value in block["roundoff_corrections_mol"].values()
            ),
            default=0.0,
        ),
        "sum_numerical_molar_corrections_mol": sum(
            value
            for block in result.solver_diagnostics
            for value in block["roundoff_corrections_mol"].values()
        ),
    }
    history.append(receipt)
    equipment = upsert_equipment_record(
        candidate.equipment,
        equipment_id="crystallizer",
        equipment_type="crystallizer",
        attached_vessel_id=state.vessel_id,
        status="advanced",
        settings={
            "execution_history": history,
            "thermal_transition": receipt,
            "crystallization_model_id": receipt["model_id"],
            "material_balance_error_mol": max(component_errors.values(), default=0),
            "component_balance_errors_mol": component_errors,
            "particle_target_balance_error_mol": abs(
                solid_target
                - sum(
                    c.particle_count * model.particle_moles(c.diameter_m)
                    for c in result.final_state.cohorts
                )
            ),
            "temperature_history_K": temperatures,
            "supersaturation_history": saturations,
            "maximum_supersaturation_ratio": max(saturations),
            "final_supersaturation_ratio": saturations[-1],
        },
    )
    candidate = candidate.replace(
        equipment=equipment,
        process=process_with_metrics(
            candidate.process,
            pre_separation_product_mol=reference,
            crystallized_from_solution_mol=grown,
            crystal_redissolved_target_mol=(
                float(metrics.get("crystal_redissolved_target_mol", 0.0))
                + receipt["net_redissolved_target_mol"]
            ),
            crystal_yield=recovery,
            seed_excluded_recovery=recovery,
            crystal_purity=purity,
            crystal_size=min(current_settings["csd_d50_m"] / 250e-6, 1.0),
            crystal_csd_quality=(
                0.55 * (1 - min(current_settings["csd_cv"], 1.0))
                + 0.45 * (1 - current_settings["csd_fines_number_fraction"])
            ),
            crystal_fines_fraction=current_settings["csd_fines_number_fraction"],
        ),
        ledger=advance_ledger(
            state.ledger,
            result.actual_duration_s,
            energy_deltas={
                "energy_jacket_J": result.boundary_heat_J,
                "heat_reaction_J": -result.chemical_energy_change_J,
                "heat_loss_J": result.heat_loss_J,
            },
        ),
        metadata={**candidate.metadata, "last_energy_transition": receipt},
    )
    return ModelProviderResult(
        outputs={},
        state=candidate,
        actual_duration_s=result.actual_duration_s,
        outcome="stopped" if result.stopping_event else "success",
        diagnostics={
            "stopping_event": result.stopping_event,
            "energy_balance_residual_J": result.energy_balance_residual_J,
            "solver": {
                "blocks": len(result.solver_diagnostics),
                **{
                    key: sum(int(block.get(key, 0)) for block in result.solver_diagnostics)
                    for key in ("nfev", "njev", "nlu")
                },
            },
        },
    )
