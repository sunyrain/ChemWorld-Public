"""Component-level property-package convenience wrapper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite, log
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from chemworld.physchem.coupled_system import CoupledStateView, ProcessContribution
    from chemworld.physchem.eos import CubicEOSSpec
    from chemworld.physchem.equilibrium import ActivityModelSpec
    from chemworld.physchem.helmholtz_solvers import HelmholtzPhaseModel
    from chemworld.physchem.incompressible_properties import IncompressibleMaterialModel
    from chemworld.physchem.molecular_properties import MolecularEOSSpec
    from chemworld.physchem.phase_evaluation import PhasePropertyModel, PhaseTransportSpec
    from chemworld.physchem.water_if97_properties import IF97WaterModel

from chemworld.foundation.units import convert_value
from chemworld.physchem.enthalpy import (
    _validate_correlation_phase,
    caloric_correlation_change,
    caloric_polynomial_change,
)
from chemworld.physchem.heat_capacity import (
    PiecewiseHeatCapacity,
    heat_capacity_from_dict,
    validate_capacity_curve,
)
from chemworld.physchem.property_equations import (
    _validity_warnings,
    evaluate_correlation,
    integrate_correlation,
)
from chemworld.physchem.property_reports import (
    R_J_PER_MOL_K,
    STANDARD_PRESSURE_PA,
    CaloricReference,
    PhaseLabel,
    PropertyEvaluation,
    StatePropertyReport,
    ValidityPolicy,
)
from chemworld.physchem.property_selection import (
    PropertySelectionPolicy,
    PropertySelectionReport,
    inspect_property_selection,
    select_property_correlation,
)
from chemworld.physchem.saturation import (
    PureSaturationReport,
    normal_boiling_point_report,
    pure_saturation_pressure_report,
    pure_saturation_temperature_report,
)
from chemworld.physchem.specs import (
    ComponentSpec,
    PropertyCorrelation,
    _mw,
    resolve_component_identifier,
)
from chemworld.physchem.vapor_pressure import VaporPressureReport, vapor_pressure_report


def evaluate_caloric_correlation(
    correlation: PropertyCorrelation | PiecewiseHeatCapacity, *, reference: CaloricReference,
    temperature_K: float, quantities: Sequence[str], pressure_Pa: float | None = None,
    validity_policy: ValidityPolicy = "raise",
) -> dict[str, PropertyEvaluation]:
    """Request a caloric pair from existing coefficients and an explicit datum.

    Cp/H and Cv/U are different bases. Only a declared gas phase supplies the
    ideal-gas R and PV conversion; a phase-free inventory retains its own pair.
    This also serves synthetic thermal inventories without inventing identities.
    """
    requested = set(quantities)
    allowed = {"cp", "cv", "enthalpy", "internal_energy", "entropy", "chemical_potential",
               "dH_dT_P", "dU_dT_V"}
    pressure = reference.pressure_Pa if pressure_Pa is None else pressure_Pa
    if (not requested or requested - allowed or validity_policy not in {"raise", "warn", "ignore"}
            or any(not isfinite(v) or v <= 0 for v in (temperature_K, pressure))):
        raise ValueError("invalid requested caloric quantity, policy or physical state")
    basis = ("internal_energy" if correlation.property_id == "isochoric_heat_capacity"
             else "enthalpy")
    if basis != reference.energy_basis:
        raise ValueError("heat-capacity and reference energy bases must agree")
    if basis == "enthalpy" and correlation.property_id not in {
        "heat_capacity", "ideal_gas_heat_capacity", "liquid_heat_capacity", "solid_heat_capacity",
    }:
        raise ValueError("caloric requests require a declared Cp or Cv correlation")
    phase = reference.phase
    _validate_correlation_phase(correlation, phase)
    # Permissive evaluation still retains the validity record in this new API.
    evaluation_policy = "warn" if validity_policy == "ignore" else validity_policy
    capacity = evaluate_correlation(correlation, temperature_K=temperature_K,
                                    pressure_Pa=pressure, validity_policy=evaluation_policy)
    if capacity.unit != "J/(mol*K)":
        capacity = capacity.to("J/(mol*K)")
    cp = capacity.value if basis == "enthalpy" else None
    cv = capacity.value if basis == "internal_energy" else None
    if phase == "gas":
        cp = capacity.value if basis == "enthalpy" else capacity.value + R_J_PER_MOL_K
        cv = capacity.value - R_J_PER_MOL_K if basis == "enthalpy" else capacity.value
    if (not isfinite(capacity.value) or capacity.value < 0
            or (basis == "enthalpy" and capacity.value == 0)
            or (cv is not None and cv < 0)):
        raise ValueError("caloric heat capacity is outside the nonnegative model domain")
    values = {"cp": cp, "cv": cv, "dH_dT_P": cp, "dU_dT_V": cv}
    warning_groups = dict.fromkeys(values, capacity.warnings)
    energy_needed = requested & {"enthalpy", "internal_energy", "chemical_potential"}
    if energy_needed:
        if phase in {"solid", "liquid"} and pressure != reference.pressure_Pa:
            raise ValueError("non-gas caloric pressure dependence needs an additional model")
        change = caloric_correlation_change(
            correlation, initial_temperature_K=reference.temperature_K,
            final_temperature_K=temperature_K, validity_policy=evaluation_policy)
        energy_warnings = tuple(dict.fromkeys((*capacity.warnings, *change.warnings)))
        energy = reference.energy_J_mol + change.value
        enthalpy = energy if basis == "enthalpy" else (
            energy + R_J_PER_MOL_K * temperature_K if phase == "gas" else None)
        internal = energy if basis == "internal_energy" else (
            energy - R_J_PER_MOL_K * temperature_K if phase == "gas" else None)
        values.update(enthalpy=enthalpy, internal_energy=internal)
        warning_groups.update(enthalpy=energy_warnings, internal_energy=energy_warnings)
    if requested & {"entropy", "chemical_potential"}:
        if reference.entropy_J_mol_K is None or cp is None:
            raise ValueError("entropy/chemical potential requires a declared entropy datum and Cp")
        if phase != "gas" and pressure != reference.pressure_Pa:
            raise ValueError("non-gas entropy pressure dependence needs an additional model")
        cp_correlation = correlation
        if basis == "internal_energy" and correlation.equation_id != "cp_polynomial":
            # Integrate Cv/T unchanged, then add the declared ideal-gas R term.
            # Cv=0 remains valid. Shape parameters cannot be scaled or shifted.
            integral = integrate_correlation(
                correlation, initial_temperature_K=reference.temperature_K,
                final_temperature_K=temperature_K, over_temperature=True,
                validity_policy=evaluation_policy)
            entropy_increment = convert_value(
                integral.value, correlation.output_unit, "J/(mol*K)")
            entropy_increment += R_J_PER_MOL_K * log(temperature_K / reference.temperature_K)
            entropy_change = PropertyEvaluation(
                "entropy_change", correlation.correlation_id, "ideal_gas_Cv_over_T_quadrature",
                entropy_increment, "J/(mol*K)", {"temperature": temperature_K}, integral.warnings)
        else:
            if basis == "internal_energy":
                # Preserve the existing polynomial's analytic path and semantics.
                polynomial = cast(PropertyCorrelation, correlation)
                factor = convert_value(1, correlation.output_unit, "J/(mol*K)")
                coefficients = {key: value * factor
                                for key, value in polynomial.coefficients.items()}
                coefficients["a"] = coefficients.get("a", 0.0) + R_J_PER_MOL_K
                cp_correlation = replace(polynomial, property_id="ideal_gas_heat_capacity",
                                         coefficients=coefficients, output_unit="J/(mol*K)")
            entropy_change = caloric_correlation_change(
                cp_correlation, initial_temperature_K=reference.temperature_K,
                final_temperature_K=temperature_K, entropy=True, validity_policy=evaluation_policy)
        warning_groups["entropy"] = tuple(dict.fromkeys(
            (*capacity.warnings, *entropy_change.warnings)))
        entropy = reference.entropy_J_mol_K + entropy_change.value
        if phase == "gas":
            entropy -= R_J_PER_MOL_K * log(pressure / reference.pressure_Pa)
        values["entropy"] = entropy
        if "chemical_potential" in requested:
            # A validated entropy request above supplies Cp and an enthalpy basis.
            values["chemical_potential"] = cast(float, values["enthalpy"]) - temperature_K * entropy
            warning_groups["chemical_potential"] = tuple(dict.fromkeys(
                (*warning_groups["enthalpy"], *warning_groups["entropy"])))
    result = {}
    for quantity in quantities:
        value = values.get(quantity)
        if value is None:
            raise ValueError(f"{quantity} requires a declared phase/EOS closure for {basis}")
        result[quantity] = PropertyEvaluation(
            quantity, correlation.correlation_id, f"caloric_{basis}:{correlation.equation_id}",
            value,
            "J/(mol*K)" if quantity in {"cp", "cv", "dH_dT_P", "dU_dT_V", "entropy"}
            else "J/mol",
            {"temperature": temperature_K, "pressure": pressure}, warning_groups[quantity],
        )
    return result


@dataclass(frozen=True)
class SpeciesCaloricModel:
    """One existing inventory's declared caloric curve and explicit reference.

    Generic inventory labels need no invented chemical formula or registry entry.
    This model is T-dependent/additive; nonideal mixture EOS closure remains a
    separate, explicitly requested state evaluation.
    """

    heat_capacity: PropertyCorrelation | PiecewiseHeatCapacity
    reference: CaloricReference

    def __post_init__(self) -> None:
        if (self.heat_capacity.input_units.get("temperature", "K") != "K"
                or set(self.heat_capacity.validity_ranges) - {"temperature"}
                or set(self.heat_capacity.input_units) - {"temperature"}):
            raise ValueError("native caloric inventory requires a T-only Kelvin capacity")
        self.evaluate(self.reference.temperature_K, ("cv", "internal_energy"))

    def evaluate(
        self, temperature_K: float, quantities: Sequence[str], *, strict: bool = True,
    ) -> dict[str, PropertyEvaluation]:
        return evaluate_caloric_correlation(
            self.heat_capacity, reference=self.reference, temperature_K=temperature_K,
            quantities=quantities, validity_policy="raise" if strict else "warn")

    def validate_interval(self, lower_K: float, upper_K: float) -> None:
        if self.heat_capacity.equation_id != "cp_polynomial":
            for temperature in (lower_K, upper_K):
                self.evaluate(temperature, ("cv", "internal_energy"))
            validate_capacity_curve(self.heat_capacity, lower_K, upper_K, allow_zero=True,
                minimum=R_J_PER_MOL_K if self.reference.energy_basis == "enthalpy" else 0)
            return
        polynomial = cast(PropertyCorrelation, self.heat_capacity)
        factor = convert_value(1, self.heat_capacity.output_unit, "J/(mol*K)")
        coefficients = {key: value * factor
                        for key, value in polynomial.coefficients.items()}
        if self.reference.energy_basis == "enthalpy":
            # Constructor validation already required the gas PV closure for U/Cv.
            coefficients["a"] = coefficients.get("a", 0.0) - R_J_PER_MOL_K
        cv = replace(polynomial, property_id="isochoric_heat_capacity",
                     coefficients=coefficients, output_unit="J/(mol*K)")
        caloric_polynomial_change(cv, initial_temperature_K=lower_K,
                                  final_temperature_K=upper_K)

    def to_dict(self) -> dict[str, object]:
        return {"heat_capacity": self.heat_capacity.to_dict(),
                "reference": self.reference.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SpeciesCaloricModel:
        return cls(heat_capacity_from_dict(payload["heat_capacity"]),
                   CaloricReference(**payload["reference"]))


@dataclass(frozen=True)
class PropertyPackageThermodynamics:
    """Additive U(n,T) and its derivatives from the same caloric evaluations.

    The common integrator owns solver trials and acceptance. Polynomial trial
    extension retains warnings; piecewise gaps never extrapolate and accepted
    states are checked strictly. No state,
    clock, resource or upstream backend is owned here.
    """

    species: Mapping[str, SpeciesCaloricModel]
    support_heat_capacity_J_K: float = 0.0
    reference_temperature_K: float = 350.0

    def __post_init__(self) -> None:
        if (any(not isinstance(key, str) or not key for key in self.species)
                or any(not isinstance(value, SpeciesCaloricModel)
                       for value in self.species.values())
                or not isfinite(self.support_heat_capacity_J_K)
                or self.support_heat_capacity_J_K < 0
                or not isfinite(self.reference_temperature_K) or self.reference_temperature_K <= 0):
            raise ValueError("caloric model needs named species and finite support/reference data")
        object.__setattr__(self, "species", MappingProxyType(dict(self.species)))

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return tuple(self.species)

    def _at(
        self, key: str, temperature: float, quantities: Sequence[str], *, strict: bool,
    ) -> dict[str, PropertyEvaluation]:
        try:
            return self.species[key].evaluate(temperature, quantities, strict=strict)
        except ValueError as exc:
            raise RuntimeError(f"caloric state outside the declared model: {key}: {exc}") from exc

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        from chemworld.physchem.coupled_system import CoupledState

        return self.support_heat_capacity_J_K + sum(
            state.amounts_mol[key] * self._at(
                key, state.temperature_K, ("cv",), strict=isinstance(state, CoupledState)
            )["cv"].value for key in self.species if state.amounts_mol[key])

    def species_energy_j_mol(self, key: str, temperature_K: float) -> float:
        values = self._at(key, temperature_K, ("internal_energy",), strict=True)
        return values["internal_energy"].value

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return (self.support_heat_capacity_J_K
                * (state.temperature_K - self.reference_temperature_K)
                + sum(state.amounts_mol[key] * self.species_energy_j_mol(key, state.temperature_K)
                      for key in self.species if state.amounts_mol[key]))

    def chemical_energy_rate_w(
        self, state: CoupledStateView, derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        if any(item.legacy_reaction_energy_W is not None for item in contributions):
            raise ValueError("native caloric U cannot also consume a separate reaction heat")
        return sum(self._at(key, state.temperature_K, ("internal_energy",), strict=False)[
            "internal_energy"].value * rate for key, rate in derivatives.items() if rate)

    def validate_state(self, state: CoupledStateView) -> None:
        for key in self.species:
            self._at(key, state.temperature_K, ("cv", "internal_energy"), strict=True)

    def validate_temperature_interval(self, lower_K: float, upper_K: float) -> None:
        for value in self.species.values():
            value.validate_interval(lower_K, upper_K)

    def validity_warnings(self, state: CoupledStateView) -> tuple[str, ...]:
        return tuple(dict.fromkeys(warning for value in self.species.values()
                                   for warning in _validity_warnings(
                                       value.heat_capacity, {"temperature": state.temperature_K})))

    def to_dict(self) -> dict[str, object]:
        return {"model": "property-package-caloric-v1",
                "species": {key: value.to_dict() for key, value in self.species.items()},
                "support_heat_capacity_J_K": self.support_heat_capacity_J_K,
                "reference_temperature_K": self.reference_temperature_K}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PropertyPackageThermodynamics:
        values = dict(payload)
        if values.pop("model") != "property-package-caloric-v1":
            raise ValueError("unsupported caloric model version")
        values["species"] = {key: SpeciesCaloricModel.from_dict(value)
                             for key, value in values["species"].items()}
        return cls(**values)


def evaluate_property_state(
    packages: Sequence[ComponentPropertyPackage], component_amounts_mol: Mapping[str, float], *,
    temperature_K: float, pressure_Pa: float, phase: str, quantities: Sequence[str],
    references: Mapping[str, CaloricReference] | None = None,
    eos: CubicEOSSpec | MolecularEOSSpec | None = None,
    activity_model: ActivityModelSpec | None = None,
    validity_policy: ValidityPolicy = "raise", volume_model: PropertyCorrelation | None = None,
    incompressible_model: IncompressibleMaterialModel | None = None,
) -> StatePropertyReport:
    """Evaluate a single-phase state using existing identities/correlations.

    Gas states declare ideal-gas behavior unless an explicit fluid EOS is given.
    EOS liquid states use the same ideal-gas caloric reference plus their
    governed departure. No phase split or missing transport data is invented.
    An explicit mixture volume_model uses the original liquid correlation at
    actual T/P and binds its fixed composition to the component inventory.
    An explicit liquid activity model adds excess H/Cp/S and mixing chemical
    potentials to declared pure-liquid references, without inventing a volume
    or a constant-volume caloric closure.
    """
    if incompressible_model is not None:
        from chemworld.physchem.incompressible_properties import (
            evaluate_incompressible_property_state,
        )

        if (phase != "liquid" or validity_policy != "raise" or references
                or any(model is not None for model in (eos, activity_model, volume_model))):
            raise ValueError("incompressible liquid uses its own strict caloric/volume model")
        return evaluate_incompressible_property_state(
            packages, incompressible_model, component_amounts_mol,
            temperature_K=temperature_K, pressure_Pa=pressure_Pa, quantities=quantities)

    from chemworld.physchem.eos import (
        cubic_state_derivatives,
        evaluate_cubic_eos,
        ideal_gas_state,
    )
    from chemworld.physchem.property_reports import _validate_phase

    _validate_phase(phase)
    if activity_model is not None and (phase != "liquid" or eos is not None):
        raise ValueError("an activity model requires liquid phase without a simultaneous EOS")
    if volume_model is not None and (phase != "liquid" or eos is not None):
        raise ValueError("explicit liquid volume model cannot be combined with a fluid EOS")
    if (validity_policy not in {"raise", "warn", "ignore"}
            or any(not isfinite(v) or v <= 0 for v in (temperature_K, pressure_Pa))):
        raise ValueError("property state requires finite positive T/P and a declared policy")
    by_id = {package.component.identifier: package for package in packages}
    if not by_id or len(by_id) != len(packages):
        raise ValueError("property packages require unique existing component identities")
    components = tuple(package.component for package in packages)
    amounts: dict[str, float] = {}
    for alias, amount in component_amounts_mol.items():
        if not isfinite(amount) or amount < 0:
            raise ValueError("component inventory must be finite and nonnegative")
        canonical = resolve_component_identifier(components, alias)
        amounts[canonical] = amounts.get(canonical, 0.0) + amount
    total = sum(amounts.values())
    if not isfinite(total) or total <= 0:
        raise ValueError("property state requires a positive finite total inventory")
    fractions = {key: value / total for key, value in sorted(amounts.items())}
    active = {key: value for key, value in fractions.items() if value > 0}
    requested = set(quantities)
    caloric = {"cp", "cv", "enthalpy", "internal_energy", "entropy", "chemical_potential",
               "dH_dT_P", "dU_dT_V"}
    volume_quantities = {"molar_volume", "density", "dV_dT_P", "dV_dP_T"}
    transport_quantities = {"viscosity", "thermal_conductivity", "surface_tension"}
    if "surface_tension" in requested and phase != "liquid":
        raise ValueError("surface tension requires the liquid side of a liquid-gas interface")
    if not requested or requested - (caloric | volume_quantities | transport_quantities
                                     | {"fugacity_coefficient", "activity_coefficient"}):
        raise ValueError("unsupported or empty state-property request")
    if "activity_coefficient" in requested and activity_model is None:
        raise ValueError("activity coefficients require an explicit liquid activity model")
    if "chemical_potential" in requested and any(value == 0 for value in fractions.values()):
        raise ValueError("finite chemical potentials are undefined for zero mole fractions")
    if (phase != "gas" and eos is None and activity_model is None and len(active) > 1
            and requested & {
        "entropy", "chemical_potential", "fugacity_coefficient",
    }):
        raise ValueError("non-gas mixture chemical properties require an explicit activity/EOS")
    if eos is not None and phase == "solid":
        raise ValueError("a fluid EOS does not describe a solid phase")
    policy = "warn" if validity_policy == "ignore" else validity_policy
    model_details: dict[str, object] = {}
    excess = None
    if activity_model is not None:
        from chemworld.physchem.activity_thermodynamics import activity_thermodynamic_report

        if set(activity_model.component_ids) != set(by_id):
            raise ValueError("activity model identities must match the component packages")
        excess = activity_thermodynamic_report(
            activity_model, {key: fractions.get(key, 0.0) for key in by_id},
            temperature_K=temperature_K)
        model_details["activity"] = excess.to_dict()
    state = None
    derivatives: Mapping[str, float | str] = {}
    if eos is not None:
        from chemworld.physchem.molecular_properties import MolecularEOSSpec

        if isinstance(eos, MolecularEOSSpec):
            if set(eos.component_ids) != set(by_id):
                raise ValueError("molecular EOS identities must match property packages")
            state, derivatives = eos.evaluate(
                fractions, temperature_K=temperature_K, pressure_Pa=pressure_Pa,
                phase="vapor" if phase == "gas" else "liquid")
        else:
            state = evaluate_cubic_eos(eos, fractions, temperature_K=temperature_K,
                                       pressure_Pa=pressure_Pa,
                                       phase="vapor" if phase == "gas" else "liquid")
            if requested & {"cp", "cv", "dH_dT_P", "dU_dT_V", "dV_dT_P", "dV_dP_T"}:
                derivatives = cubic_state_derivatives(
                    eos, fractions, temperature_K=temperature_K, pressure_Pa=pressure_Pa,
                    phase="vapor" if phase == "gas" else "liquid")
    elif phase == "gas":
        state = ideal_gas_state(fractions, temperature_K=temperature_K, pressure_Pa=pressure_Pa)
        derivatives = {"cp_departure_J_mol_K": 0.0, "cp_minus_cv_J_mol_K": R_J_PER_MOL_K,
                       "d_molar_volume_dT_P": R_J_PER_MOL_K / pressure_Pa,
                       "d_molar_volume_dP_T": -state.molar_volume_m3_mol / pressure_Pa}
    if state is not None:
        model_details.update(eos_id=state.eos_id, model=state.model,
                             root_selection_policy=state.root_selection_policy,
                             roots=list(state.roots), derivatives=derivatives)
    reference_map: dict[str, CaloricReference] = {}
    for alias, reference in (references or {}).items():
        canonical = resolve_component_identifier(components, alias)
        if canonical in reference_map and reference_map[canonical] != reference:
            raise ValueError("component aliases specify inconsistent caloric references")
        reference_map[canonical] = reference
    pure: dict[str, dict[str, PropertyEvaluation]] = {}
    warnings_by_property: dict[str, tuple[str, ...]] = {}
    cp_needed = requested & {"cp", "cv", "dH_dT_P", "dU_dT_V"}
    energy_needed = requested & {"enthalpy", "internal_energy", "chemical_potential"}
    entropy_needed = requested & {"entropy", "chemical_potential"}
    # A condensed Cv/U declaration is already an energy model. It does not
    # need a fictitious Cp/H or density calculation to return its own pair.
    direct_internal = state is None and any(
        reference_map.get(key, CaloricReference()).energy_basis == "internal_energy"
        for key in active
    )
    if direct_internal and requested & caloric:
        if excess is not None:
            raise ValueError("activity excess properties require pure-liquid H references")
        if any(
            key not in reference_map or reference_map[key].energy_basis != "internal_energy"
            for key in active
        ):
            raise ValueError("condensed caloric inventory cannot mix H and U reference bases")
        if requested & (caloric - {"cv", "internal_energy", "dU_dT_V"}):
            raise ValueError(
                "condensed Cv/U needs an additional closure to provide Cp/H or entropy"
            )
        model_details["caloric_basis"] = "internal_energy"
    if requested & caloric:
        base_phase: PhaseLabel = "gas" if eos is not None else cast(PhaseLabel, phase)
        for key in active:
            package = by_id[key]
            reference = reference_map.setdefault(key, CaloricReference(phase=base_phase))
            if reference.phase != base_phase:
                raise ValueError("caloric reference must declare the selected reference phase")
            property_id = ("isochoric_heat_capacity" if reference.energy_basis == "internal_energy"
                           else "ideal_gas_heat_capacity" if base_phase == "gas"
                           else f"{base_phase}_heat_capacity")
            candidates = package.by_property(property_id)
            if not candidates:
                candidates = tuple(c for c in package.by_property("heat_capacity")
                                   if c.metadata.get("phase", package.component.default_phase)
                                   == base_phase and reference.energy_basis == "enthalpy")
            if not candidates:
                raise ValueError(f"No {property_id} correlation for {key}")
            correlation = package.select_correlation(candidates[0].property_id,
                candidates=candidates, temperature_K=temperature_K, pressure_Pa=pressure_Pa)
            needs = ((["cp"] if cp_needed else []) + (["enthalpy"] if energy_needed else [])
                     + (["entropy"] if entropy_needed else []))
            if direct_internal:
                needs = (
                    (["cv"] if cp_needed else []) + (["internal_energy"] if energy_needed else [])
                )
            pure[key] = evaluate_caloric_correlation(
                correlation, reference=reference, temperature_K=temperature_K,
                pressure_Pa=pressure_Pa, quantities=needs, validity_policy=policy)
    values: dict[str, float] = {}
    units = {"cp": "J/(mol*K)", "cv": "J/(mol*K)", "enthalpy": "J/mol",
             "internal_energy": "J/mol", "entropy": "J/(mol*K)",
             "dH_dT_P": "J/(mol*K)", "dU_dT_V": "J/(mol*K)",
             "molar_volume": "m^3/mol", "density": "kg/m^3", "dV_dT_P": "m^3/(mol*K)",
             "dV_dP_T": "m^3/(mol*Pa)", "viscosity": "Pa*s",
             "thermal_conductivity": "W/(m*K)", "surface_tension": "N/m"}
    source_ids = tuple(dict.fromkeys(value.correlation_id
                                    for component in pure.values() for value in component.values()))

    def weighted(quantity: str) -> float:
        return sum(fraction * pure[key][quantity].value for key, fraction in active.items())

    def warnings_for(quantity: str) -> tuple[str, ...]:
        return tuple(dict.fromkeys(warning for values in pure.values()
                                   for warning in values[quantity].warnings))

    molar_volume = None if state is None else state.molar_volume_m3_mol
    if direct_internal:
        if cp_needed:
            values.update(cv=weighted("cv"), dU_dT_V=weighted("cv"))
            warnings_by_property.update(cv=warnings_for("cv"), dU_dT_V=warnings_for("cv"))
        if energy_needed:
            values["internal_energy"] = weighted("internal_energy")
            warnings_by_property["internal_energy"] = warnings_for("internal_energy")
    if cp_needed and not direct_internal:
        cp = weighted("cp") + cast(float, derivatives.get("cp_departure_J_mol_K", 0.0))
        if excess is not None:
            cp += excess.heat_capacity_excess_J_mol_K
            if not isfinite(cp) or cp <= 0:
                raise ValueError("activity mixture Cp is outside the stable caloric domain")
        values.update(cp=cp, dH_dT_P=cp)
        for quantity in ("cp", "cv", "dH_dT_P", "dU_dT_V"):
            warnings_by_property[quantity] = warnings_for("cp")
        if requested & {"cv", "dU_dT_V"}:
            if state is None:
                raise ValueError("Cv requires a declared EOS/volume closure")
            cv = cp - cast(float, derivatives["cp_minus_cv_J_mol_K"])
            if cv < 0 or cp <= 0:
                raise ValueError("state heat capacity is outside the stable caloric domain")
            values.update(cv=cv, dU_dT_V=cv)
    if energy_needed and not direct_internal:
        values["enthalpy"] = weighted("enthalpy") + (
            state.molar_residual_enthalpy_J_mol if state else 0.0)
        if excess is not None:
            values["enthalpy"] += excess.enthalpy_excess_J_mol
        for quantity in ("enthalpy", "internal_energy"):
            warnings_by_property[quantity] = warnings_for("enthalpy")
    if entropy_needed:
        values["entropy"] = weighted("entropy")
        warnings_by_property["entropy"] = warnings_for("entropy")
        if state is not None:
            values["entropy"] += (-R_J_PER_MOL_K * sum(x * log(x) for x in fractions.values() if x)
                                   + state.molar_residual_entropy_J_mol_K)
        elif excess is not None:
            values["entropy"] += (-R_J_PER_MOL_K * sum(x * log(x) for x in fractions.values() if x)
                                   + excess.entropy_excess_J_mol_K)
    properties = {}
    for quantity in requested & transport_quantities:
        if len(active) != 1:
            raise ValueError("mixture transport requires an explicitly selected mixing model")
        package = by_id[next(iter(active))]
        property_id = f"{phase}_viscosity" if quantity == "viscosity" else quantity
        candidates = tuple(c for c in package.by_property(property_id)
                           if c.metadata.get("phase", phase) == phase)
        if not candidates:
            raise ValueError(f"No {property_id} correlation on the requested phase")
        correlation = package.select_correlation(property_id, candidates=candidates,
            temperature_K=temperature_K, pressure_Pa=pressure_Pa)
        properties[quantity] = replace(evaluate_correlation(
            correlation, temperature_K=temperature_K, pressure_Pa=pressure_Pa,
            validity_policy=policy).to(units[quantity]), property_id=quantity)
    requires_volume = requested & volume_quantities or (
        "internal_energy" in requested and not direct_internal
    )
    if state is None and requires_volume:
        if volume_model is not None:
            from chemworld.physchem.specs import MixtureSpec
            from chemworld.physchem.volume_mixtures import mixture_volume_evaluation

            declared = volume_model.metadata.get("mixture")
            if phase != "liquid" or not isinstance(declared, dict):
                raise ValueError("explicit mixture volume model requires its liquid composition")
            mixture = MixtureSpec.from_dict(declared)
            if set(mixture.component_ids) != set(by_id) or any(
                abs(fractions.get(key, 0) - x) > 1e-9 or
                abs(_mw(by_id[key].component) - mw) > 1e-9 * mw
                for key, x, mw in zip(mixture.component_ids, mixture.mole_fractions,
                                     mixture.molecular_weights_g_mol, strict=True)
            ):
                raise ValueError("fixed volume-model composition/MW must match the state")
            current = replace(mixture, temperature_K=temperature_K, pressure_Pa=pressure_Pa)
            expected = mixture_volume_evaluation(
                current, method=volume_model.metadata["mixture_method"],
                component_parameters=volume_model.metadata["component_parameters"],
                **cast(Mapping[str, Any], volume_model.metadata["pressure_parameters"]))
            evaluated = evaluate_correlation(volume_model, temperature_K=temperature_K,
                                            pressure_Pa=pressure_Pa, validity_policy=policy)
            molar_mass = mixture.average_molecular_weight_g_mol / 1000
            molar_volume = (evaluated.to("m^3/mol").value
                            if volume_model.property_id == "liquid_molar_volume"
                            else molar_mass / evaluated.to("kg/m^3").value)
            if abs(molar_volume - expected.value) > 1e-9 * expected.value:
                raise ValueError("mixture volume declaration disagrees with its original inputs")
            source_ids = (*source_ids, evaluated.correlation_id)
            for quantity in ("molar_volume", "density", "internal_energy"):
                warnings_by_property[quantity] = tuple(dict.fromkeys(
                    (*warnings_by_property.get(quantity, ()), *evaluated.warnings)))
            model_details["volume_mixing"] = volume_model.to_dict()
        elif len(active) != 1:
            raise ValueError("non-gas mixture volume requires an explicit volume-mixing model")
        else:
            key = next(iter(active))
            density = by_id[key].evaluate(f"{phase}_density", temperature_K=temperature_K,
                                      pressure_Pa=pressure_Pa, validity_policy=policy).to("kg/m^3")
            molar_volume = _mw(by_id[key].component) / 1000 / density.value
            for quantity in ("molar_volume", "density", "internal_energy"):
                warnings_by_property[quantity] = tuple(dict.fromkeys(
                    (*warnings_by_property.get(quantity, ()), *density.warnings)))
            source_ids = (*source_ids, density.correlation_id)
    if molar_volume is not None:
        values["molar_volume"] = molar_volume
        molar_mass = sum(x * _mw(by_id[key].component) / 1000
                         for key, x in fractions.items())
        values["density"] = molar_mass / molar_volume
        if "internal_energy" in requested and not direct_internal:
            values["internal_energy"] = values["enthalpy"] - pressure_Pa * molar_volume
    for quantity, derivative in (("dV_dT_P", "d_molar_volume_dT_P"),
                                  ("dV_dP_T", "d_molar_volume_dP_T")):
        if quantity in requested:
            if derivative not in derivatives:
                raise ValueError("requested volume derivative requires a supported EOS")
            values[quantity] = cast(float, derivatives[derivative])
    component_properties = {}
    if "activity_coefficient" in requested and excess is not None:
        component_properties["activity_coefficient"] = {
            key: PropertyEvaluation("activity_coefficient", excess.model_id,
                                     "partial_molar_excess_gibbs", value, "dimensionless",
                                     {"temperature": temperature_K, "pressure": pressure_Pa})
            for key, value in excess.activity_coefficients.items()}
    if "fugacity_coefficient" in requested:
        if state is None:
            raise ValueError("fugacity requires a declared EOS")
        component_properties["fugacity_coefficient"] = {
            key: PropertyEvaluation("fugacity_coefficient", state.eos_id,
                                     state.root_selection_policy, value, "dimensionless",
                                     {"temperature": temperature_K, "pressure": pressure_Pa})
            for key, value in state.fugacity_coefficients.items()}
    if "chemical_potential" in requested:
        component_properties["chemical_potential"] = {}
        for key, fraction in fractions.items():
            mu = pure[key]["enthalpy"].value - temperature_K * pure[key]["entropy"].value
            if state is not None:
                mu += R_J_PER_MOL_K * temperature_K * (
                    log(fraction) + log(state.fugacity_coefficients[key]))
            elif excess is not None:
                mu += (R_J_PER_MOL_K * temperature_K * log(fraction)
                       + excess.partial_molar_gibbs_excess_J_mol[key])
            component_properties["chemical_potential"][key] = PropertyEvaluation(
                "chemical_potential", "+".join(source_ids),
                "declared_reference_and_activity" if excess else "declared_reference_and_fugacity",
                mu,
                "J/mol", {"temperature": temperature_K, "pressure": pressure_Pa},
                tuple(dict.fromkeys((*pure[key]["enthalpy"].warnings,
                                     *pure[key]["entropy"].warnings))))
    model_id = state.eos_id if state is not None else "single-phase-caloric-correlations-v1"
    if excess is not None:
        model_id = excess.model_id
    for quantity in requested - properties.keys() - component_properties.keys():
        if quantity not in values:
            raise ValueError(f"{quantity} is unavailable for the selected state model")
        properties[quantity] = PropertyEvaluation(
            quantity, "+".join((*source_ids, model_id)), "requested_single_phase_state",
            values[quantity], units[quantity],
            {"temperature": temperature_K, "pressure": pressure_Pa},
            warnings_by_property.get(quantity, ()))
    properties = {key: properties[key] for key in quantities if key in properties}
    component_properties = {key: component_properties[key] for key in quantities
                            if key in component_properties}
    return StatePropertyReport(model_id, cast(PhaseLabel, phase),
                                temperature_K, pressure_Pa, amounts, properties,
                                reference_map, component_properties, model_details)


@dataclass(frozen=True)
class ComponentPropertyPackage:
    component: ComponentSpec
    correlations: tuple[PropertyCorrelation | PiecewiseHeatCapacity, ...]
    selection_policies: Mapping[str, PropertySelectionPolicy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policies = dict(self.selection_policies)
        for prop, policy in policies.items():
            if not isinstance(policy, PropertySelectionPolicy):
                raise TypeError("package selection needs PropertySelectionPolicy objects")
            if not self.by_property(prop):
                raise ValueError(f"selection names an unavailable property {prop!r}")
        object.__setattr__(self, "selection_policies", MappingProxyType(policies))
        ids = [correlation.correlation_id for correlation in self.correlations]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate correlation_id values are not allowed")
        allowed = set(self.component.allowed_property_correlations)
        if allowed:
            disallowed = [
                correlation.correlation_id
                for correlation in self.correlations
                if correlation.correlation_id not in allowed
                and correlation.property_id not in allowed
                and correlation.equation_id not in allowed
            ]
            if disallowed:
                raise ValueError(
                    "Correlations are not allowed by component policy: "
                    f"{disallowed}"
                )

    def by_property(
        self, property_id: str,
    ) -> tuple[PropertyCorrelation | PiecewiseHeatCapacity, ...]:
        return tuple(
            correlation
            for correlation in self.correlations
            if correlation.property_id == property_id
        )

    def selection_report(
        self, property_id: str, *, temperature_K: float, pressure_Pa: float | None = None,
        policy: PropertySelectionPolicy | None = None,
    ) -> PropertySelectionReport:
        """Inspect all alternatives; this does not evaluate or fit their formulas."""
        return inspect_property_selection(self.by_property(property_id), property_id=property_id,
            temperature_K=temperature_K, pressure_Pa=pressure_Pa,
            policy=policy or self.selection_policies.get(property_id))

    def select_correlation(
        self, property_id: str, *, temperature_K: float, pressure_Pa: float | None = None,
        candidates: Sequence[PropertyCorrelation | PiecewiseHeatCapacity] | None = None,
    ) -> PropertyCorrelation | PiecewiseHeatCapacity:
        return select_property_correlation(
            self.by_property(property_id) if candidates is None else candidates,
            property_id=property_id, temperature_K=temperature_K, pressure_Pa=pressure_Pa,
            policy=self.selection_policies.get(property_id))

    def with_selection(
        self, policies: Mapping[str, PropertySelectionPolicy],
    ) -> ComponentPropertyPackage:
        """Declare choices for future states and preserve them in the original package."""
        return replace(self, selection_policies={**self.selection_policies, **policies})

    def evaluate_state(
        self, *, temperature_K: float, pressure_Pa: float, phase: str,
        quantities: Sequence[str], reference: CaloricReference | None = None,
        validity_policy: ValidityPolicy = "raise", gas_model: str | None = None,
        helmholtz_model: HelmholtzPhaseModel | None = None,
        water_model: IF97WaterModel | None = None,
        incompressible_model: IncompressibleMaterialModel | None = None,
    ) -> StatePropertyReport:
        """Evaluate requested intensive properties on a one-mole pure basis."""
        if incompressible_model is not None:
            if (reference is not None or gas_model is not None or helmholtz_model is not None
                    or water_model is not None):
                raise ValueError("incompressible model uses an exclusive material/caloric binding")
            return evaluate_property_state(
                (self,), {self.component.identifier: 1.0}, temperature_K=temperature_K,
                pressure_Pa=pressure_Pa, phase=phase, quantities=quantities,
                validity_policy=validity_policy, incompressible_model=incompressible_model)
        if water_model is not None:
            from chemworld.physchem.water_if97_properties import evaluate_if97_property_state

            if (reference is not None or gas_model is not None or helmholtz_model is not None
                    or validity_policy != "raise"):
                raise ValueError("IF97 uses its own exclusive caloric reference and strict domain")
            return evaluate_if97_property_state(
                self, water_model, temperature_K=temperature_K, pressure_Pa=pressure_Pa,
                phase=phase, quantities=quantities)
        if helmholtz_model is not None:
            from chemworld.physchem.helmholtz_properties import evaluate_helmholtz_property_state

            if reference is not None or gas_model is not None or validity_policy != "raise":
                raise ValueError(
                    "Helmholtz uses its own caloric reference and strict source domain")
            return evaluate_helmholtz_property_state(
                self, helmholtz_model, temperature_K=temperature_K, pressure_Pa=pressure_Pa,
                phase=phase, quantities=quantities)
        if gas_model is not None:
            from chemworld.physchem.virial_thermodynamics import evaluate_virial_gas_state

            if gas_model != "virial" or phase != "gas" or validity_policy != "raise":
                raise ValueError("virial gas consumption requires gas phase and strict validity")
            ideal = None
            if set(quantities) & {"cp", "cv", "enthalpy", "internal_energy", "entropy",
                                  "chemical_potential", "dH_dT_P", "dU_dT_V"}:
                candidates = (*self.by_property("ideal_gas_heat_capacity"),
                              *self.by_property("isochoric_heat_capacity"))
                if (not candidates or reference is None
                        or len({c.property_id for c in candidates}) != 1
                        or (len(candidates) > 1
                            and candidates[0].property_id not in self.selection_policies)):
                    raise ValueError("virial gas needs one explicit ideal heat capacity/reference")
                ideal = SpeciesCaloricModel(self.select_correlation(candidates[0].property_id,
                    candidates=candidates, temperature_K=temperature_K), reference)
            return evaluate_virial_gas_state(self, temperature_K=temperature_K,
                pressure_Pa=pressure_Pa, quantities=quantities, ideal_caloric=ideal)
        references = None if reference is None else {self.component.identifier: reference}
        return evaluate_property_state(
            (self,), {self.component.identifier: 1.0}, temperature_K=temperature_K,
            pressure_Pa=pressure_Pa, phase=phase, quantities=quantities,
            references=references, validity_policy=validity_policy)

    def phase_model(
        self, *, phase: str, reference: CaloricReference | None = None,
        eos: CubicEOSSpec | MolecularEOSSpec | None = None,
        activity_model: ActivityModelSpec | None = None,
        volume_model: PropertyCorrelation | None = None,
        transport_models: Mapping[str, PropertyCorrelation | PhaseTransportSpec] | None = None,
        helmholtz_model: HelmholtzPhaseModel | None = None,
        water_model: IF97WaterModel | None = None,
        incompressible_model: IncompressibleMaterialModel | None = None,
    ) -> PhasePropertyModel:
        """Freeze a pure-phase declaration over this package's existing consumers."""
        from chemworld.physchem.phase_evaluation import PhasePropertyModel

        return PhasePropertyModel(
            (self,), phase=phase,
            references=None if reference is None else {self.component.identifier: reference},
            eos=eos, activity_model=activity_model, volume_model=volume_model,
            transport_models=transport_models, helmholtz_model=helmholtz_model,
            water_model=water_model, incompressible_model=incompressible_model,
        )

    def evaluate(
        self,
        property_id: str,
        *,
        temperature_K: float,
        pressure_Pa: float | None = None,
        validity_policy: ValidityPolicy = "warn",
    ) -> PropertyEvaluation:
        candidates = self.by_property(property_id)
        if not candidates:
            # Convert an explicitly authored phase volume/density relation.
            # Do not substitute another model when an authored curve fails.
            for phase in ("liquid", "solid", "gas"):
                density_id, volume_id = phase + "_density", phase + "_molar_volume"
                if property_id not in {density_id, volume_id}:
                    continue
                alternate = volume_id if property_id == density_id else density_id
                if self.by_property(alternate):
                    original = self.evaluate(alternate, temperature_K=temperature_K,
                        pressure_Pa=pressure_Pa, validity_policy=validity_policy)
                    value = original.to("m^3/mol" if alternate == volume_id else "kg/m^3").value
                    if value <= 0:
                        raise ValueError("volume/density conversion requires positive source value")
                    return replace(original, property_id=property_id,
                        value=_mw(self.component) / 1000 / value,
                        unit="kg/m^3" if property_id == density_id else "m^3/mol")
            raise ValueError(
                f"No correlation for property {property_id!r} on component "
                f"{self.component.identifier!r}"
            )
        chosen = self.select_correlation(
            property_id, candidates=candidates,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
        )
        return evaluate_correlation(
            chosen,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            molecular_weight_g_mol=self.component.molecular_weight_g_mol,
            validity_policy=validity_policy,
        )

    def vapor_pressure_report(
        self,
        *,
        temperature_K: float,
        property_id: str = "vapor_pressure",
        validity_policy: ValidityPolicy = "warn",
    ) -> VaporPressureReport:
        candidates = self.by_property(property_id)
        if not candidates:
            raise ValueError(
                f"No correlation for property {property_id!r} on component "
                f"{self.component.identifier!r}"
            )
        chosen = self.select_correlation(
            property_id, candidates=candidates,
            temperature_K=temperature_K,
            pressure_Pa=None,
        )
        return vapor_pressure_report(
            # The report's property-ID guard excludes heat-capacity curves.
            cast(PropertyCorrelation, chosen),
            temperature_K=temperature_K,
            validity_policy=validity_policy,
        )

    def pure_saturation_pressure_report(
        self,
        *,
        temperature_K: float,
        property_id: str = "vapor_pressure",
        critical_temperature_K: float | None = None,
        critical_pressure_Pa: float | None = None,
        validity_policy: ValidityPolicy = "warn",
    ) -> PureSaturationReport:
        candidates = self.by_property(property_id)
        if not candidates:
            raise ValueError(
                f"No correlation for property {property_id!r} on component "
                f"{self.component.identifier!r}"
            )
        chosen = self.select_correlation(
            property_id, candidates=candidates,
            temperature_K=temperature_K,
            pressure_Pa=None,
        )
        return pure_saturation_pressure_report(
            cast(PropertyCorrelation, chosen),
            temperature_K=temperature_K,
            critical_temperature_K=critical_temperature_K,
            critical_pressure_Pa=critical_pressure_Pa,
            validity_policy=validity_policy,
        )

    def pure_saturation_temperature_report(
        self,
        *,
        pressure_Pa: float,
        property_id: str = "vapor_pressure",
        temperature_bounds_K: tuple[float, float] | None = None,
        critical_temperature_K: float | None = None,
        critical_pressure_Pa: float | None = None,
        validity_policy: ValidityPolicy = "raise",
    ) -> PureSaturationReport:
        candidates = self.by_property(property_id)
        if not candidates:
            raise ValueError(
                f"No correlation for property {property_id!r} on component "
                f"{self.component.identifier!r}"
            )
        chosen = self.select_correlation(
            property_id, candidates=candidates,
            temperature_K=temperature_bounds_K[0] if temperature_bounds_K else 298.15,
            pressure_Pa=None,
        )
        return pure_saturation_temperature_report(
            cast(PropertyCorrelation, chosen),
            pressure_Pa=pressure_Pa,
            temperature_bounds_K=temperature_bounds_K,
            critical_temperature_K=critical_temperature_K,
            critical_pressure_Pa=critical_pressure_Pa,
            validity_policy=validity_policy,
        )

    def normal_boiling_point_report(
        self,
        *,
        pressure_Pa: float = STANDARD_PRESSURE_PA,
        property_id: str = "vapor_pressure",
        temperature_bounds_K: tuple[float, float] | None = None,
        critical_temperature_K: float | None = None,
        critical_pressure_Pa: float | None = None,
        validity_policy: ValidityPolicy = "raise",
    ) -> PureSaturationReport:
        candidates = self.by_property(property_id)
        if not candidates:
            raise ValueError(
                f"No correlation for property {property_id!r} on component "
                f"{self.component.identifier!r}"
            )
        chosen = self.select_correlation(
            property_id, candidates=candidates,
            temperature_K=temperature_bounds_K[0] if temperature_bounds_K else 298.15,
            pressure_Pa=None,
        )
        return normal_boiling_point_report(
            cast(PropertyCorrelation, chosen),
            pressure_Pa=pressure_Pa,
            temperature_bounds_K=temperature_bounds_K,
            critical_temperature_K=critical_temperature_K,
            critical_pressure_Pa=critical_pressure_Pa,
            validity_policy=validity_policy,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "component": self.component.to_dict(),
            "correlations": [correlation.to_dict() for correlation in self.correlations],
            **({"selection_policies": {k: p.to_dict() for k, p in self.selection_policies.items()}}
               if self.selection_policies else {}),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComponentPropertyPackage:
        if set(payload) - {"component", "correlations", "selection_policies"}:
            raise ValueError("unknown property package fields")
        return cls(ComponentSpec.from_dict(payload["component"]),
            tuple(PiecewiseHeatCapacity.from_dict(c) if "model" in c
                  else PropertyCorrelation.from_dict(c) for c in payload["correlations"]),
            {k: PropertySelectionPolicy.from_dict(p)
             for k, p in payload.get("selection_policies", {}).items()})
