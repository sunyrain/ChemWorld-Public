"""Finite cohort molecular stocks in the shared inventory/temperature solver.

Geometry coordinates carry metres and molecular ratios, never fictitious molar
stocks. Each candidate reconstructs actual cohort molecules for CoupledSystem's
rate/energy assembler. Its geometry strategy uses the common ODE backend; new
nuclei retain the declared temporal birth quadrature.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isclose, isfinite, pi, ulp
from time import monotonic
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledStateView,
    CoupledSystem,
    HeatFlux,
    InternalTransfer,
    ProcessContribution,
    SpeciesInternalEnergy,
)
from chemworld.physchem.crystallization_units import (
    CrystallizationKineticsSpec,
    SolubilityCurveSpec,
    _occlusion_loading_ratio,
)
from chemworld.physchem.enthalpy import caloric_temperature

if TYPE_CHECKING:
    from chemworld.physchem.legacy_crystal_enthalpy import LegacyCrystalEnthalpy
    from chemworld.physchem.property_packages import (
        PropertyPackageThermodynamics,
        SpeciesCaloricModel,
    )

CrystalNativeThermodynamics: TypeAlias = "PropertyPackageThermodynamics | ConstantCvThermodynamics"
CrystalThermodynamics: TypeAlias = "CrystalNativeThermodynamics | LegacyCrystalEnthalpy"
LiquidKeys: TypeAlias = Mapping[str, tuple[str, ...]]
SolidKeys: TypeAlias = Sequence[Mapping[str, str]]
KineticTerms: TypeAlias = Callable[
    ["CrystalThermalSpec", CoupledStateView, LiquidKeys], tuple[float, float, float]
]


@dataclass(frozen=True)
class CrystalThermalSpec:
    target_component: str
    impurity_component: str | None
    solubility: SolubilityCurveSpec
    kinetics: CrystallizationKineticsSpec
    solid_property_keys: dict[str, str]
    birth_interval_s: float = 30.0
    impurity_resolution_fraction: float = 1e-8

    def __post_init__(self) -> None:
        required = {self.target_component}
        if self.impurity_component is not None:
            required.add(self.impurity_component)
        if (
            not self.target_component
            or len(required) != (1 if self.impurity_component is None else 2)
            or set(self.solid_property_keys) != required
            or any(not isinstance(v, str) or not v for v in self.solid_property_keys.values())
            or not isfinite(self.birth_interval_s)
            or self.birth_interval_s <= 0
        ):
            raise ValueError(
                "crystal model needs distinct components, caloric keys and birth interval"
            )
        object.__setattr__(self, "solid_property_keys", dict(self.solid_property_keys))
        if not 0 < self.impurity_resolution_fraction <= 1e-8:
            raise ValueError("impurity supply resolution must lie within the material tolerance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_component": self.target_component,
            "impurity_component": self.impurity_component,
            "solubility": self.solubility.to_dict(),
            "kinetics": self.kinetics.to_dict(),
            "solid_property_keys": dict(self.solid_property_keys),
            "birth_interval_s": self.birth_interval_s,
            "impurity_resolution_fraction": self.impurity_resolution_fraction,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CrystalThermalSpec:
        return cls(
            **{
                **payload,
                "solubility": SolubilityCurveSpec(**payload["solubility"]),
                "kinetics": CrystallizationKineticsSpec(**payload["kinetics"]),
            }
        )

    def particle_moles(self, diameter_m: float) -> float:
        return (
            self.kinetics.crystal_density_kg_m3
            * pi
            / 6
            * diameter_m**3
            / self.kinetics.target_molecular_weight_kg_mol
        )


@dataclass(frozen=True)
class ThermalCrystalCohort:
    particle_count: float
    diameter_m: float
    impurity_mol: float = 0.0
    seed_mol: float = 0.0
    target_mol: float | None = None

    def __post_init__(self) -> None:
        if (
            any(
                not isfinite(v) or v < 0
                for v in (self.particle_count, self.diameter_m, self.impurity_mol, self.seed_mol)
            )
            or self.particle_count == 0
            or self.diameter_m == 0
        ):
            raise ValueError(
                "retained crystal cohort needs positive count/diameter and finite stocks"
            )
        if self.target_mol is not None and (not isfinite(self.target_mol) or self.target_mol <= 0):
            raise ValueError("retained cohort molecular mass must be positive and finite")


@dataclass(frozen=True)
class CrystalPopulationState:
    liquid_amounts_mol: dict[str, float]
    cohorts: tuple[ThermalCrystalCohort, ...]
    temperature_K: float
    volume_L: float
    dissolved_seed_mol: float = 0.0
    time_s: float = 0.0


@dataclass(frozen=True)
class CrystalThermalAdvance:
    initial_state: CrystalPopulationState
    final_state: CrystalPopulationState
    actual_duration_s: float
    boundary_heat_J: float
    chemical_energy_change_J: float
    heat_loss_J: float
    energy_balance_residual_J: float
    stopping_event: str | None
    solver_diagnostics: tuple[dict[str, Any], ...]
    trajectory: tuple[CrystalPopulationState, ...]


def _layout(
    model: CrystalThermalSpec, population: CrystalPopulationState,
    thermodynamics: CrystalThermodynamics, *, newborn: bool = False,
) -> tuple[CrystalThermodynamics, dict[str, float], dict[str, tuple[str, ...]],
           list[dict[str, str]], list[ThermalCrystalCohort | None]]:
    target, impurity = model.target_component, model.impurity_component
    liquid = population.liquid_amounts_mol
    if (
        target not in liquid
        or (impurity is not None and impurity not in liquid)
        or any(not isfinite(n) or n < 0 for n in liquid.values())
        or not 0 <= population.dissolved_seed_mol <= liquid[target]
    ):
        raise ValueError("finite liquid and dissolved seed must share the actual molecular stock")
    amounts: dict[str, float] = {}
    properties: dict[str, SpeciesCaloricModel | SpeciesInternalEnergy | float] = {}
    liquid_keys: dict[str, tuple[str, ...]] = {}
    # JSON object order is not a physical parameter. Fix the solver coordinate
    # order so a sorted-key trajectory replay takes the same numerical path.
    for key, n in sorted(liquid.items()):
        if key == target:
            for name, value in (
                ("seed", population.dissolved_seed_mol),
                ("grown", n - population.dissolved_seed_mol),
            ):
                packed = f"liquid:{key}:{name}"
                amounts[packed], properties[packed] = value, thermodynamics.species[key]
            liquid_keys[key] = (f"liquid:{key}:seed", f"liquid:{key}:grown")
        else:
            packed = f"liquid:{key}"
            amounts[packed], properties[packed] = n, thermodynamics.species[key]
            liquid_keys[key] = (packed,)
    cohorts = [*population.cohorts, *(() if not newborn else (None,))]
    solid_keys = []
    for index, cohort in enumerate(cohorts):
        total = (
            0.0
            if cohort is None
            else cohort.particle_count * model.particle_moles(cohort.diameter_m)
        )
        if cohort is not None and cohort.target_mol is not None:
            if not isclose(total, cohort.target_mol, rel_tol=1e-10, abs_tol=1e-14):
                raise ValueError("cohort diameter does not close its actual molecular mass")
            total = cohort.target_mol
        seed = 0.0 if cohort is None else cohort.seed_mol
        if seed > total + 1e-12:
            raise ValueError("cohort seed provenance exceeds its actual target molecules")
        fields = {}
        for label, n in (("seed", seed), ("grown", total - seed)):
            key = f"cohort:{index}:{label}"
            amounts[key] = n
            properties[key] = thermodynamics.species[model.solid_property_keys[target]]
            fields[label] = key
        if impurity is not None:
            key = f"cohort:{index}:impurity"
            amounts[key] = 0.0 if cohort is None else cohort.impurity_mol
            properties[key] = thermodynamics.species[model.solid_property_keys[impurity]]
            fields["impurity"] = key
        solid_keys.append(fields)
    from chemworld.physchem.crystal_caloric_layout import CrystalCaloricLayout
    from chemworld.physchem.property_packages import PropertyPackageThermodynamics

    # Do not replace custom caloric subclasses or the legacy enthalpy boundary.
    packed_thermal: CrystalThermodynamics
    if isinstance(thermodynamics, PropertyPackageThermodynamics):
        caloric_species = cast("dict[str, SpeciesCaloricModel]", properties)
        packed_thermal = (
            CrystalCaloricLayout(
                caloric_species,
                thermodynamics.support_heat_capacity_J_K,
                thermodynamics.reference_temperature_K,
            )
            if type(thermodynamics) is PropertyPackageThermodynamics
            else replace(thermodynamics, species=caloric_species)
        )
    elif isinstance(thermodynamics, ConstantCvThermodynamics):
        packed_thermal = replace(
            thermodynamics, species=cast(dict[str, SpeciesInternalEnergy], properties),
        )
    else:
        packed_thermal = replace(thermodynamics, species=cast(dict[str, float], properties))
    return (packed_thermal, amounts, liquid_keys, solid_keys, cohorts)


def _unpack(
    model: CrystalThermalSpec, packed: CoupledStateView, liquid_keys: LiquidKeys,
    solid_keys: SolidKeys, cohorts: Sequence[ThermalCrystalCohort | None],
) -> CrystalPopulationState:
    liquid = {
        key: sum(packed.amounts_mol[k] for k in fields) for key, fields in liquid_keys.items()
    }
    final_cohorts = []
    for keys, initial in zip(solid_keys, cohorts, strict=True):
        target = packed.amounts_mol[keys["seed"]] + packed.amounts_mol[keys["grown"]]
        impurity = packed.amounts_mol[keys["impurity"]] if "impurity" in keys else 0.0
        if target == 0:
            if impurity != 0:
                raise RuntimeError("vanished cohort retains unresolved occluded molecules")
            continue
        count = (
            target / model.particle_moles(model.kinetics.nucleus_diameter_m)
            if initial is None
            else initial.particle_count
        )
        diameter = (target / count / model.particle_moles(1.0)) ** (1 / 3)
        final_cohorts.append(
            ThermalCrystalCohort(
                count, diameter, impurity, packed.amounts_mol[keys["seed"]], target
            )
        )
    return CrystalPopulationState(
        liquid,
        tuple(final_cohorts),
        packed.temperature_K,
        packed.volume_L,
        packed.amounts_mol[liquid_keys[model.target_component][0]],
        packed.time_s,
    )


def crystal_population_energy(
    model: CrystalThermalSpec, state: CrystalPopulationState, thermodynamics: CrystalThermodynamics,
) -> float:
    thermal, amounts, *_ = _layout(model, state, thermodynamics)
    packed = CoupledState(amounts, state.temperature_K, state.volume_L, state.time_s)
    energy = thermal.internal_energy_j(packed)
    if energy is None:
        energy = cast("LegacyCrystalEnthalpy", thermal).control_volume_enthalpy_j(packed)
    return energy


def _release_vanished_hosts(
    model: CrystalThermalSpec, packed: CoupledState, thermal: CrystalThermodynamics,
    liquid_keys: LiquidKeys, solid_keys: SolidKeys, *, isothermal: bool, impurity_stock_mol: float,
) -> tuple[CoupledState, list[dict[str, float]]]:
    """Finish the defined host-disappearance boundary, retaining every molecule.

    Only unresolved residue within the existing 1e-8 stock tolerance qualifies.
    A substantial orphan impurity stock is a solver failure, never discarded.
    """
    if model.impurity_component is None:
        return packed, []
    amounts = dict(packed.amounts_mol)
    releases = []
    for index, fields in enumerate(solid_keys):
        if (
            amounts[fields["seed"]] + amounts[fields["grown"]] == 0
            and amounts[fields["impurity"]] > 0
        ):
            quantity = amounts[fields["impurity"]]
            amounts[fields["impurity"]] = 0.0
            key = liquid_keys[model.impurity_component][0]
            amounts[key] += quantity
            releases.append({"cohort": index, "occluded_mol": quantity, "time_s": packed.time_s})
    if not releases:
        return packed, []
    total = sum(row["occluded_mol"] for row in releases)
    tolerance = 1e-24 + 1e-8 * impurity_stock_mol
    if total > tolerance:
        raise RuntimeError(f"vanished host has unresolved impurity {total} mol > {tolerance} mol")

    def energy(candidate: CoupledStateView) -> float:
        value = thermal.internal_energy_j(candidate)
        return (cast("LegacyCrystalEnthalpy", thermal).control_volume_enthalpy_j(candidate)
                if value is None else value)

    initial_energy = energy(packed)
    candidate = replace(packed, amounts_mol=amounts)
    phase_energy = energy(candidate) - initial_energy
    if not isothermal and phase_energy != 0:
        temperature = caloric_temperature(
            lambda t: energy(replace(candidate, temperature_K=t)),
            initial_energy,
            (model.solubility.minimum_temperature_K, model.solubility.maximum_temperature_K),
            energy_basis=(
                "internal_energy" if thermal.internal_energy_j(packed) is not None else "enthalpy"
            ),
        )
        candidate = replace(candidate, temperature_K=temperature)
    candidate = replace(
        candidate,
        boundary_heat_J=packed.boundary_heat_J + (phase_energy if isothermal else 0),
        chemical_energy_change_J=packed.chemical_energy_change_J + phase_energy,
    )
    return candidate, releases


def crystal_kinetic_terms(
    model: CrystalThermalSpec, state: CoupledStateView, liquid_keys: LiquidKeys,
    *, impurity_resolution_mol: float = 0.0,
) -> tuple[float, float, float]:
    kinetics = model.kinetics
    concentration = (
        sum(state.amounts_mol[k] for k in liquid_keys[model.target_component]) / state.volume_L
    )
    saturation = concentration / model.solubility.solubility_mol_per_l(state.temperature_K)
    # At equilibrium, roundoff in C/C* must not alternate growth/recession
    # and manufacture numerical birth cohorts. This is only a floating-point
    # resolution guard, many orders below the declared diameter tolerance.
    if abs(saturation - 1.0) <= 32 * ulp(1.0):
        saturation = 1.0
    relative = max(saturation - 1, 0.0)
    velocity = (
        2 * kinetics.growth_coefficient_m_s * relative**kinetics.growth_exponent
        - 2 * kinetics.growth_coefficient_m_s * max(1 - saturation, 0.0) ** kinetics.growth_exponent
    )
    nucleation = (
        kinetics.primary_nucleation_coefficient_per_L_s
        * relative**kinetics.primary_nucleation_exponent
        * state.volume_L
        * model.particle_moles(kinetics.nucleus_diameter_m)
    )
    available = (
        0
        if model.impurity_component is None
        else state.amounts_mol[liquid_keys[model.impurity_component][0]]
    )
    capture = (
        0.0
        if available == 0
        else _occlusion_loading_ratio(
            relative_supersaturation=relative,
            dissolved_impurity=abs(available),
            solvent_volume_L=state.volume_L,
            kinetics=kinetics,
        )
    )
    if impurity_resolution_mol > 0:
        capture *= min(1.0, abs(available) / impurity_resolution_mol)
    if available < 0:
        # Odd continuation is only for unaccepted Newton trial states. It
        # restores a negative trial pool instead of introducing a flat clipped
        # RHS at depletion. Accepted stocks retain their nonnegative checks.
        capture = -capture
    return velocity, nucleation, capture


def _process(
    model: CrystalThermalSpec, thermal: CrystalThermodynamics, liquid_keys: LiquidKeys,
    solid_keys: SolidKeys, cohorts: Sequence[ThermalCrystalCohort | None], *,
    kinetic_terms: KineticTerms = crystal_kinetic_terms,
) -> CoupledProcess:
    from chemworld.physchem.crystal_caloric_layout import CrystalCaloricLayout
    from chemworld.physchem.legacy_crystal_enthalpy import LegacyCrystalEnthalpy

    # These exact caloric implementations consume summed molecular rates, not
    # transfer objects. Custom calorics still receive the full transfer records.
    aggregate_transfers = type(thermal) in (CrystalCaloricLayout, LegacyCrystalEnthalpy)
    target, impurity = model.target_component, model.impurity_component
    particle_moles_unit = model.particle_moles(1.0)
    layout = tuple(
        (
            fields,
            cohort,
            tuple(
                (f"crystal:{index}:{label}", fields[label], fluid_key)
                for label, fluid_key in zip(("seed", "grown"), liquid_keys[target], strict=True)
            ),
        )
        for index, (fields, cohort) in enumerate(zip(solid_keys, cohorts, strict=True))
    )
    if aggregate_transfers:
        identities = set()
        for _, _, exchanges in layout:
            for exchange_id, solid_key, fluid_key in exchanges:
                InternalTransfer(exchange_id, fluid_key, solid_key, 0.0)
                if exchange_id in identities:
                    raise ValueError(f"duplicate internal exchange: {exchange_id}")
                identities.add(exchange_id)
                if solid_key not in thermal.species or fluid_key not in thermal.species:
                    raise ValueError(
                        "internal transfer endpoints must belong to the compiled inventory")

    def evaluate(state: CoupledStateView) -> ProcessContribution:
        temperature = state.temperature_K
        liquid_target = sum(state.amounts_mol[k] for k in liquid_keys[target])
        diameter_rate, nucleation_rate, capture_ratio = kinetic_terms(model, state, liquid_keys)
        fluid_energy = thermal.species_energy_j_mol(liquid_keys[target][0], temperature)
        solid_energy = thermal.species_energy_j_mol(solid_keys[0]["seed"], temperature)
        if not isclose(
            fluid_energy - solid_energy,
            model.solubility.dissolution_enthalpy_J_mol,
            rel_tol=1e-10,
            abs_tol=1e-6,
        ):
            raise ValueError("crystal caloric offsets disagree with the solubility energy")
        transfers = []
        amount_rates: dict[str, float] = {}
        for fields, cohort, exchanges in layout:
            n = state.amounts_mol[fields["seed"]] + state.amounts_mol[fields["grown"]]
            if cohort is None:
                rate = nucleation_rate
            else:
                diameter = (n / cohort.particle_count / particle_moles_unit) ** (1 / 3)
                rate = 3 * cohort.particle_count * particle_moles_unit * diameter**2 * diameter_rate
            for exchange_id, solid_key, fluid_key in exchanges:
                if rate >= 0:
                    amount = (
                        0
                        if liquid_target == 0
                        else rate * state.amounts_mol[fluid_key] / liquid_target
                    )
                    source, destination = fluid_key, solid_key
                else:
                    amount = 0 if n == 0 else -rate * state.amounts_mol[solid_key] / n
                    source, destination = solid_key, fluid_key
                if aggregate_transfers:
                    if not isfinite(amount) or amount < 0:
                        raise ValueError("internal transfer rate must be finite and nonnegative")
                    # Same per-component update order as the common assembler.
                    # No reaction/port receipt or seed inventory is removed.
                    amount_rates[source] = amount_rates.get(source, 0.0) - amount
                    amount_rates[destination] = amount_rates.get(destination, 0.0) + amount
                else:
                    transfers.append(InternalTransfer(exchange_id, source, destination, amount))
            if impurity is not None:
                fluid_key = liquid_keys[impurity][0]
                if rate >= 0:
                    amount = rate * capture_ratio
                    source, destination = fluid_key, fields["impurity"]
                else:
                    amount = 0 if n == 0 else -rate * state.amounts_mol[fields["impurity"]] / n
                    source, destination = fields["impurity"], fluid_key
                amount_rates[source] = amount_rates.get(source, 0.0) - amount
                amount_rates[destination] = amount_rates.get(destination, 0.0) + amount
        return ProcessContribution(
            amount_rates_mol_s=amount_rates, internal_transfers=tuple(transfers)
        )

    all_keys = tuple(thermal.inventory_keys)
    dependencies = {key: all_keys for keys in liquid_keys.values() for key in keys}
    fluid_target = liquid_keys[target]
    for fields in solid_keys:
        local = (*fluid_target, fields["seed"], fields["grown"])
        dependencies[fields["seed"]] = local
        dependencies[fields["grown"]] = local
        if impurity is not None:
            dependencies[fields["impurity"]] = (
                *local,
                liquid_keys[impurity][0],
                fields["impurity"],
            )
    return CoupledProcess(
        "crystal-growth-dissolution", evaluate, inventory_dependencies=dependencies
    )


def _inventory_scales(
    amounts: Mapping[str, float], liquid_keys: LiquidKeys, solid_keys: SolidKeys,
    cohorts: Sequence[ThermalCrystalCohort | None],
) -> dict[str, float]:
    """Resolve each existing cohort relative to its own actual molecular stock.

    A common large floor badly perturbs the finite-difference diameter of tiny
    cohorts. New birth reservoirs instead use a finite available-liquid scale.
    """
    scales = {k: max(amounts[k], 1e-8) for fields in liquid_keys.values() for k in fields}
    available = max(sum(amounts[k] for fields in liquid_keys.values() for k in fields), 1e-8)
    for fields, cohort in zip(solid_keys, cohorts, strict=True):
        host = amounts[fields["seed"]] + amounts[fields["grown"]]
        for key in fields.values():
            scales[key] = (
                available
                if cohort is None
                else amounts[key]
                if amounts[key] > 0
                else max(host, 1e-300)
            )
    return scales


def advance_crystal_population(
    model: CrystalThermalSpec,
    state: CrystalPopulationState,
    thermodynamics: CrystalThermodynamics,
    duration_s: float,
    *,
    heat_flux: Callable[[CoupledStateView], HeatFlux] | None = None,
    inventory_independent_heat_flux: bool | None = None,
    isothermal: bool = False,
    evaluation_times_s: Sequence[float] = (),
    events: Sequence[CoupledEvent] = (),
    progress: Callable[[dict[str, float]], None] | None = None,
) -> CrystalThermalAdvance:
    if not isfinite(duration_s) or duration_s < 0:
        raise ValueError("crystal advance needs a finite nonnegative duration")
    if heat_flux is None:

        def heat_flux(_: CoupledStateView) -> HeatFlux:
            return HeatFlux()

        if inventory_independent_heat_flux is None:
            inventory_independent_heat_flux = True
    initial = state
    energy0 = crystal_population_energy(model, state, thermodynamics)
    end = state.time_s + duration_s
    diagnostics, trajectory = [], [state]
    heat, chemical, loss, work = 0.0, 0.0, 0.0, 0.0
    stopping = None
    started, last_report = monotonic(), monotonic()
    rhs_calls = 0
    impurity_stock = (
        0.0
        if model.impurity_component is None
        else state.liquid_amounts_mol[model.impurity_component]
        + sum(c.impurity_mol for c in state.cohorts)
    )
    impurity_resolution = impurity_stock * model.impurity_resolution_fraction

    last_candidate: CoupledStateView | None = None
    last_terms: tuple[float, float, float] | None = None

    def kinetic_terms(
        model: CrystalThermalSpec, candidate: CoupledStateView, liquid_keys: LiquidKeys,
    ) -> tuple[float, float, float]:
        nonlocal last_candidate, last_terms
        # Process assembly and the geometry RHS ask for these same terms on
        # the very same immutable state. No rounding or cross-state cache.
        if candidate is not last_candidate:
            values = crystal_kinetic_terms(
                model, candidate, liquid_keys, impurity_resolution_mol=impurity_resolution
            )
            last_terms = (values[0], values[1], 0.0) if impurity_stock == 0 else values
            last_candidate = candidate
        return cast(tuple[float, float, float], last_terms)

    while state.time_s < end:
        birth = model.kinetics.primary_nucleation_coefficient_per_L_s > 0
        interval = min(end - state.time_s, model.birth_interval_s if birth else duration_s)
        thermal, amounts, liquid_keys, solid_keys, cohorts = _layout(
            model, state, thermodynamics, newborn=birth
        )
        # Empty populations without nucleation still retain the actual liquid
        # and thermal body, with no fabricated positive seed.
        processes = (
            ()
            if not cohorts
            else (
                _process(
                    model, thermal, liquid_keys, solid_keys, cohorts, kinetic_terms=kinetic_terms
                ),
            )
        )
        if progress is not None and processes:
            original_evaluate = processes[0].evaluate
            cohort_count = len(cohorts)

            def counted(
                candidate: CoupledStateView,
                evaluate: Callable[[CoupledStateView], ProcessContribution] = original_evaluate,
                cohort_count: int = cohort_count,
                liquid_keys: LiquidKeys = liquid_keys,
            ) -> ProcessContribution:
                nonlocal rhs_calls, last_report
                rhs_calls += 1
                now = monotonic()
                if now - last_report >= 30:
                    progress(
                        {
                            "rhs_calls": rhs_calls,
                            "cohorts": cohort_count,
                            "physical_elapsed_s": candidate.time_s - initial.time_s,
                            "requested_s": duration_s,
                            "elapsed_wall_s": now - started,
                            "free_impurity_mol": (
                                candidate.amounts_mol[liquid_keys[model.impurity_component][0]]
                                if model.impurity_component is not None
                                else 0.0
                            ),
                            "temperature_K": candidate.temperature_K,
                            "diameter_velocity_m_s": kinetic_terms(model, candidate, liquid_keys)[
                                0
                            ],
                        }
                    )
                    last_report = now
                return evaluate(candidate)

            processes = (replace(processes[0], evaluate=counted),)
        laws = {}
        for component, keys in liquid_keys.items():
            component_keys = list(keys)
            if component == model.target_component:
                component_keys += [k[label] for k in solid_keys for label in ("seed", "grown")]
            elif component == model.impurity_component:
                component_keys += [k["impurity"] for k in solid_keys]
            laws[component] = dict.fromkeys(component_keys, 1.0)
        laws["seed-provenance"] = dict.fromkeys(
            (liquid_keys[model.target_component][0], *(k["seed"] for k in solid_keys)), 1.0
        )
        local_events = [
            *events,
            CoupledEvent(
                "minimum_temperature",
                lambda s: s.temperature_K - model.solubility.minimum_temperature_K,
            ),
            CoupledEvent(
                "maximum_temperature",
                lambda s: model.solubility.maximum_temperature_K - s.temperature_K,
            ),
        ]
        system = CoupledSystem(
            thermal,
            processes,
            heat_flux=heat_flux,
            isothermal=isothermal,
            conservation_laws=laws,
            events=tuple(local_events),
            inventory_scales_mol=_inventory_scales(amounts, liquid_keys, solid_keys, cohorts),
        )
        sampling = tuple(
            t - (state.time_s - initial.time_s)
            for t in evaluation_times_s
            if state.time_s - initial.time_s <= t <= state.time_s - initial.time_s + interval
        )
        from chemworld.physchem.crystal_cohort_solver import solve_crystal_interval

        result = solve_crystal_interval(
            system,
            CoupledState(amounts, state.temperature_K, state.volume_L, state.time_s),
            interval,
            model,
            liquid_keys,
            solid_keys,
            cohorts,
            kinetic_terms=kinetic_terms,
            evaluation_times_s=sampling,
            inventory_independent_heat_flux=inventory_independent_heat_flux,
        )
        impurity_stock = (
            0.0
            if model.impurity_component is None
            else sum(amounts[k] for k in laws[model.impurity_component])
        )
        final, releases = _release_vanished_hosts(
            model,
            result.final_state,
            thermal,
            liquid_keys,
            solid_keys,
            isothermal=isothermal,
            impurity_stock_mol=impurity_stock,
        )
        for packed in result.trajectory[1:]:
            packed, _ = _release_vanished_hosts(
                model,
                packed,
                thermal,
                liquid_keys,
                solid_keys,
                isothermal=isothermal,
                impurity_stock_mol=impurity_stock,
            )
            trajectory.append(_unpack(model, packed, liquid_keys, solid_keys, cohorts))
        state = _unpack(model, final, liquid_keys, solid_keys, cohorts)
        heat += final.boundary_heat_J
        chemical += final.chemical_energy_change_J
        loss += final.heat_loss_J
        work += final.work_J
        diagnostics.append(
            {
                **dict(result.solver_diagnostic),
                "host_disappearance_releases": releases,
                "roundoff_corrections_mol": dict(result.roundoff_corrections_mol),
            }
        )
        if result.stopping_event is not None:
            stopping = result.stopping_event
            break
        if result.actual_duration_s <= 0:
            raise RuntimeError("crystal solver made no physical-time progress")
    residual = (
        crystal_population_energy(model, state, thermodynamics) - energy0 - heat + loss - work
    )
    return CrystalThermalAdvance(
        initial,
        state,
        state.time_s - initial.time_s,
        heat,
        chemical,
        loss,
        residual,
        stopping,
        tuple(diagnostics),
        tuple(trajectory),
    )
