"""Gas/solid local energies on original fixed-bed transfers and CoupledSystem."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from math import fsum, isfinite
from typing import Any

from chemworld.foundation.state_ownership import StateSlot
from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledStateView,
    CoupledSystem,
    HeatFlux,
    ProcessContribution,
    SpeciesInternalEnergy,
    assemble_coupled_rates,
)
from chemworld.physchem.fixed_bed import (
    FIXED_BED_POLICY,
    build_fixed_bed_system,
    fixed_bed_gas_key,
)
from chemworld.physchem.fixed_bed_thermal_specs import FixedBedThermalSpec, bed_thermal_key
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.solver_backend import ODESolverPolicy
from chemworld.physchem.sorbent_kinetics import sorbent_key

THERMAL_FIXED_BED_POLICY = replace(
    FIXED_BED_POLICY, solver_id="native-thermal-fixed-bed-v1", rtol=1e-10)


def bed_gas_temperature(spec: FixedBedThermalSpec, state: CoupledStateView, phase: str) -> float:
    return (state.temperature_K if phase == spec.anchor_phase_id
            else state.coordinates[bed_thermal_key("gas_K", phase)])


def bed_solid_temperature(state: CoupledStateView, phase: str) -> float:
    return state.coordinates[bed_thermal_key("solid_K", phase)]


class _BedCalorics:
    """Transient local views; every amount remains an original phase/slot amount."""

    def __init__(self, spec: FixedBedThermalSpec, volumes: Mapping[str, float],
                 slots: Sequence[StateSlot]) -> None:
        self.spec = spec
        self.models: dict[tuple[str, str], ConstantCvThermodynamics] = {}
        self.volumes: dict[tuple[str, str], float] = {}
        bed = spec.bed
        for phase, volume in volumes.items():
            group = ("gas", phase)
            self.models[group] = ConstantCvThermodynamics(
                {fixed_bed_gas_key(phase, k): bed.gas_calorics[k] for k in bed.component_ids},
                reference_temperature_K=bed.sorbent.reference_temperature_K)
            self.volumes[group] = volume
        for phase, slot, volume in zip(bed.cell_phase_ids, slots,
                                       bed.mesh.cell_volumes_m3, strict=True):
            group = ("solid", phase)
            self.models[group] = ConstantCvThermodynamics({
                sorbent_key(slot, k): SpeciesInternalEnergy(
                    bed.gas_calorics[k].u_ref_J_mol + bed.sorbent.adsorption_energy_J_mol[k],
                    bed.gas_calorics[k].cv_J_mol_K)
                for k in bed.sorbent.component_ids}, slot.values["heat_capacity_J_K"],
                bed.sorbent.reference_temperature_K)
            self.volumes[group] = float(volume * (1-bed.void_fraction) * 1000)
        self.anchor = ("gas", spec.anchor_phase_id)
        self.owners = {key: group for group, model in self.models.items()
                       for key in model.inventory_keys}
        if len(self.owners) != sum(len(model.inventory_keys) for model in self.models.values()):
            raise ValueError("thermal gas and sorbent inventory identities must not collide")

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return tuple(key for model in self.models.values() for key in model.inventory_keys)

    def coordinate(self, group: tuple[str, str]) -> str:
        return bed_thermal_key(group[0]+"_K", group[1])

    def view(self, state: CoupledStateView, group: tuple[str, str]) -> CoupledStateView:
        temperature = (bed_gas_temperature(self.spec, state, group[1]) if group[0] == "gas"
                       else bed_solid_temperature(state, group[1]))
        return CoupledStateView(
            {key: state.amounts_mol[key] for key in self.models[group].inventory_keys},
            temperature, self.volumes[group], state.time_s)

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        return self.models[self.anchor].heat_capacity_j_k(self.view(state, self.anchor))

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return fsum(model.internal_energy_j(self.view(state, group))
                    for group, model in self.models.items())

    def chemical_energy_rate_w(self, state: CoupledStateView, derivatives: Mapping[str, float],
                               contributions: Sequence[ProcessContribution]) -> float:
        values = []
        for group, model in self.models.items():
            local = self.view(state, group)
            values.append(model.chemical_energy_rate_w(
                local, {key: derivatives[key] for key in model.inventory_keys}, ()))
            if group != self.anchor:
                values.append(model.heat_capacity_j_k(local) * fsum(
                    c.coordinate_rates.get(self.coordinate(group), 0) for c in contributions))
        return fsum(values)


def build_thermal_fixed_bed_system(
    spec: FixedBedThermalSpec,
    phase_volumes_L: Mapping[str, float],
    slots: Sequence[StateSlot],
    *,
    inlet_phase_id: str | None,
    outlet_phase_id: str | None,
    policy: ODESolverPolicy = THERMAL_FIXED_BED_POLICY,
    maximum_pressure_Pa: float | None = None,
    maximum_temperature_K: float | None = None,
) -> tuple[CoupledSystem, dict[str, Any]]:
    """Compile real finite gas compartments and persistent original sorbent lots."""
    if set(phase_volumes_L) - set(spec.gas_phase_ids):
        raise ValueError("every selected gas phase needs its authored temperature owner")
    base, diagnostic = build_fixed_bed_system(
        spec.bed, phase_volumes_L, slots,
        inlet_phase_id=inlet_phase_id, outlet_phase_id=outlet_phase_id, policy=policy,
        maximum_pressure_Pa=maximum_pressure_Pa,
        _phase_temperature=lambda state, phase: bed_gas_temperature(spec, state, phase),
        _solid_temperature=bed_solid_temperature)
    thermal = _BedCalorics(spec, phase_volumes_L, slots)
    bed, mesh = spec.bed, spec.bed.mesh
    fractions = tuple(float(volume / sum(mesh.cell_volumes_m3))
                      for volume in mesh.cell_volumes_m3)
    upper = float("inf") if maximum_temperature_K is None else maximum_temperature_K
    if maximum_temperature_K is not None and (
        type(upper) not in {int, float} or not isfinite(upper) or upper <= 0
    ):
        raise ValueError("thermal hardware requires a finite positive temperature limit")
    bounds = {group: (spec.gas_temperature_bounds_K if group[0] == "gas"
                      else bed.sorbent.temperature_bounds_K) for group in thermal.models}
    bounds = {group: (lower, min(high, upper)) for group, (lower, high) in bounds.items()}
    if any(lower >= high for lower, high in bounds.values()):
        raise ValueError("thermal hardware and material domains must overlap")

    def jacket_powers(state: CoupledStateView) -> dict[str, float]:
        return {phase: (fraction * spec.jacket.delivered_power_w(
            bed_solid_temperature(state, phase)) if spec.jacket is not None else 0.0)
                for phase, fraction in zip(bed.cell_phase_ids, fractions, strict=True)}

    def evaluate(state: CoupledStateView) -> ProcessContribution:
        contributions = tuple(process.evaluate(state) for process in base.processes)
        if any(c.amount_rates_mol_s or c.coordinate_rates or c.port_rates_mol_s
               or c.energy_port_rates_W for c in contributions):
            raise ValueError("thermal bed needs original conservative internal-transfer processes")
        movement = ProcessContribution(internal_transfers=tuple(
            transfer for c in contributions for transfer in c.internal_transfers))
        # Reuse the common material assembly, without integrating or settling
        # its isothermal bookkeeping. The original transfer objects are kept.
        mass = assemble_coupled_rates(
            state, processes=(CoupledProcess("bed-mass", lambda _: movement),),
            thermodynamics=base.thermodynamics, heat_flux=lambda _: HeatFlux(), isothermal=True)
        powers = dict.fromkeys(thermal.models, 0.0)
        for transfer in movement.internal_transfers:
            source = thermal.owners[transfer.source_key]
            target = thermal.owners[transfer.destination_key]
            donor = thermal.view(state, source)
            if source[0] == "gas":
                free_model, free_key = thermal.models[source], transfer.source_key
            elif target[0] == "gas":
                free_model, free_key = thermal.models[target], transfer.destination_key
            else:
                raise ValueError("fixed carrier cannot advect between solid cells")
            enthalpy = (free_model.species_energy_j_mol(free_key, donor.temperature_K)
                        + R*donor.temperature_K)
            power = transfer.rate_mol_s * enthalpy
            powers[source] -= power
            powers[target] += power
        for phase, volume in zip(bed.cell_phase_ids, mesh.cell_volumes_m3, strict=True):
            power = spec.gas_solid_heat_transfer_W_m3_K * volume * (
                bed_solid_temperature(state, phase) - bed_gas_temperature(spec, state, phase))
            powers["gas", phase] += power
            powers["solid", phase] -= power
        for index, (left, right) in enumerate(zip(bed.cell_phase_ids[:-1],
                                                 bed.cell_phase_ids[1:], strict=True)):
            distance = float(mesh.centers_m[index+1] - mesh.centers_m[index])
            for kind, conductivity, fraction in (
                ("gas", spec.gas_thermal_conductivity_W_m_K, bed.void_fraction),
                ("solid", spec.solid_thermal_conductivity_W_m_K, 1-bed.void_fraction),
            ):
                a, b = (kind, left), (kind, right)
                power = conductivity * fraction * bed.area_m2 / distance * (
                    thermal.view(state, a).temperature_K - thermal.view(state, b).temperature_K)
                powers[a] -= power
                powers[b] += power
        jacket = jacket_powers(state)
        for phase, power in jacket.items():
            powers["solid", phase] += power
        coordinates = {
            bed_thermal_key("heat_import_J"): fsum(max(q, 0) for q in jacket.values()),
            bed_thermal_key("heat_export_J"): fsum(max(-q, 0) for q in jacket.values()),
        }
        for group, model in thermal.models.items():
            if group == thermal.anchor:
                continue
            local = thermal.view(state, group)
            contribution = ProcessContribution(amount_rates_mol_s={
                key: mass.amount_rates_mol_s[key] for key in model.inventory_keys})

            def local_movement(_state: CoupledStateView,
                               value: ProcessContribution = contribution) -> ProcessContribution:
                return value

            def local_heat(_state: CoupledStateView, power: float = powers[group]) -> HeatFlux:
                return HeatFlux(heater_W=power)

            local_rates = assemble_coupled_rates(
                local, processes=(CoupledProcess("bed-local-energy", local_movement),),
                thermodynamics=model, heat_flux=local_heat)
            coordinates[thermal.coordinate(group)] = local_rates.temperature_rate_K_s
        return replace(movement, coordinate_rates=coordinates)

    def validate(state: CoupledStateView) -> None:
        for process in base.processes:
            if process.validate_state is not None:
                process.validate_state(state)
        for group, model in thermal.models.items():
            local = thermal.view(state, group)
            lower, high = bounds[group]
            if not lower-1e-7 <= local.temperature_K <= high+1e-7:
                raise ValueError("a gas or solid temperature exceeds its material/hardware domain")
            if model.heat_capacity_j_k(local) <= 0:
                raise ValueError("every active gas and carrier needs positive actual heat capacity")

    def extend(state: CoupledStateView) -> CoupledStateView:
        coordinates = dict(state.coordinates)
        temperature = state.temperature_K
        for group, (lower, high) in bounds.items():
            if group == thermal.anchor:
                temperature = min(high, max(lower, temperature))
            else:
                key = thermal.coordinate(group)
                coordinates[key] = min(high, max(lower, coordinates[key]))
        return replace(state, temperature_K=temperature, coordinates=coordinates)

    def temperature_event(group: tuple[str, str], bound: float, sign: int) -> CoupledEvent:
        return CoupledEvent(
            f"bed:thermal:{group[0]}:{group[1]}:{sign}",
            lambda state: sign*(thermal.view(state, group).temperature_K-bound),
            feasible_side=True)

    events = tuple(event for process in base.processes for event in process.events) + tuple(
        event for group, (lower, high) in bounds.items()
        for event in (temperature_event(group, lower, 1), temperature_event(group, high, -1)))
    coordinates = tuple(
        CoupledCoordinate(thermal.coordinate(group), "K", 300)
        for group in thermal.models if group != thermal.anchor
    ) + tuple(CoupledCoordinate(bed_thermal_key(kind), "J", 1)
              for kind in ("heat_import_J", "heat_export_J"))
    diagnostic.update(
        thermal_scope="original_finite_gas_and_sorbent_U; separate_gas_and_solid_temperatures",
        gas_phase_ids=tuple(phase_volumes_L), anchor_phase_id=spec.anchor_phase_id,
        gas_solid_heat_transfer_W_m3_K=spec.gas_solid_heat_transfer_W_m3_K,
        thermal_power_distribution=fractions)
    return CoupledSystem(
        thermal, (CoupledProcess(f"{bed.bed_id}:thermal", evaluate, validate_state=validate),),
        heat_flux=lambda state: HeatFlux(heater_W=fsum(jacket_powers(state).values())),
        coordinates=coordinates,
        rate_independent_coordinates=tuple(bed_thermal_key(kind)
                                           for kind in ("heat_import_J", "heat_export_J")),
        conservation_laws=base.conservation_laws, inventory_scales_mol=base.inventory_scales_mol,
        events=events, trial_state_extension=extend, policy=policy), diagnostic
