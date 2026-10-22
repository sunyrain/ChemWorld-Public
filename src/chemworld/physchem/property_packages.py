"""Component-level property-package convenience wrapper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite, log
from types import MappingProxyType

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
    _choose_valid_correlation,
    _validity_warnings,
    evaluate_correlation,
    integrate_correlation,
)
from chemworld.physchem.property_reports import (
    R_J_PER_MOL_K,
    STANDARD_PRESSURE_PA,
    CaloricReference,
    PropertyEvaluation,
    StatePropertyReport,
    ValidityPolicy,
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
    resolve_component_identifier,
)
from chemworld.physchem.vapor_pressure import VaporPressureReport, vapor_pressure_report


def evaluate_caloric_correlation(
    correlation: PropertyCorrelation, *, reference: CaloricReference,
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
            change = convert_value(integral.value, correlation.output_unit, "J/(mol*K)")
            change += R_J_PER_MOL_K * log(temperature_K / reference.temperature_K)
            entropy_change = PropertyEvaluation(
                "entropy_change", correlation.correlation_id, "ideal_gas_Cv_over_T_quadrature",
                change, "J/(mol*K)", {"temperature": temperature_K}, integral.warnings)
        else:
            if basis == "internal_energy":
                # Preserve the existing polynomial's analytic path and semantics.
                factor = convert_value(1, correlation.output_unit, "J/(mol*K)")
                coefficients = {key: value * factor
                                for key, value in correlation.coefficients.items()}
                coefficients["a"] = coefficients.get("a", 0.0) + R_J_PER_MOL_K
                cp_correlation = replace(correlation, property_id="ideal_gas_heat_capacity",
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
            values["chemical_potential"] = values["enthalpy"] - temperature_K * entropy
            warning_groups["chemical_potential"] = tuple(dict.fromkeys(
                (*warning_groups["enthalpy"], *warning_groups["entropy"])))
    result = {}
    for quantity in quantities:
        if values.get(quantity) is None:
            raise ValueError(f"{quantity} requires a declared phase/EOS closure for {basis}")
        result[quantity] = PropertyEvaluation(
            quantity, correlation.correlation_id, f"caloric_{basis}:{correlation.equation_id}",
            values[quantity],
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

    def __post_init__(self):
        if (self.heat_capacity.input_units.get("temperature", "K") != "K"
                or set(self.heat_capacity.validity_ranges) - {"temperature"}
                or set(self.heat_capacity.input_units) - {"temperature"}):
            raise ValueError("native caloric inventory requires a T-only Kelvin capacity")
        self.evaluate(self.reference.temperature_K, ("cv", "internal_energy"))

    def evaluate(self, temperature_K, quantities, *, strict=True):
        return evaluate_caloric_correlation(
            self.heat_capacity, reference=self.reference, temperature_K=temperature_K,
            quantities=quantities, validity_policy="raise" if strict else "warn")

    def validate_interval(self, lower_K, upper_K):
        if self.heat_capacity.equation_id != "cp_polynomial":
            for temperature in (lower_K, upper_K):
                self.evaluate(temperature, ("cv", "internal_energy"))
            validate_capacity_curve(self.heat_capacity, lower_K, upper_K, allow_zero=True,
                minimum=R_J_PER_MOL_K if self.reference.energy_basis == "enthalpy" else 0)
            return
        factor = convert_value(1, self.heat_capacity.output_unit, "J/(mol*K)")
        coefficients = {key: value * factor
                        for key, value in self.heat_capacity.coefficients.items()}
        if self.reference.energy_basis == "enthalpy":
            # Constructor validation already required the gas PV closure for U/Cv.
            coefficients["a"] = coefficients.get("a", 0.0) - R_J_PER_MOL_K
        cv = replace(self.heat_capacity, property_id="isochoric_heat_capacity",
                     coefficients=coefficients, output_unit="J/(mol*K)")
        caloric_polynomial_change(cv, initial_temperature_K=lower_K,
                                  final_temperature_K=upper_K)

    def to_dict(self):
        return {"heat_capacity": self.heat_capacity.to_dict(),
                "reference": self.reference.to_dict()}

    @classmethod
    def from_dict(cls, payload):
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

    def __post_init__(self):
        if (any(not isinstance(key, str) or not key for key in self.species)
                or any(not isinstance(value, SpeciesCaloricModel)
                       for value in self.species.values())
                or not isfinite(self.support_heat_capacity_J_K)
                or self.support_heat_capacity_J_K < 0
                or not isfinite(self.reference_temperature_K) or self.reference_temperature_K <= 0):
            raise ValueError("caloric model needs named species and finite support/reference data")
        object.__setattr__(self, "species", MappingProxyType(dict(self.species)))

    @property
    def inventory_keys(self):
        return tuple(self.species)

    def _at(self, key, temperature, quantities, *, strict):
        try:
            return self.species[key].evaluate(temperature, quantities, strict=strict)
        except ValueError as exc:
            raise RuntimeError(f"caloric state outside the declared model: {key}: {exc}") from exc

    def heat_capacity_j_k(self, state):
        from chemworld.physchem.coupled_system import CoupledState

        return self.support_heat_capacity_J_K + sum(
            state.amounts_mol[key] * self._at(
                key, state.temperature_K, ("cv",), strict=isinstance(state, CoupledState)
            )["cv"].value for key in self.species if state.amounts_mol[key])

    def species_energy_j_mol(self, key, temperature_K):
        values = self._at(key, temperature_K, ("internal_energy",), strict=True)
        return values["internal_energy"].value

    def internal_energy_j(self, state):
        return (self.support_heat_capacity_J_K
                * (state.temperature_K - self.reference_temperature_K)
                + sum(state.amounts_mol[key] * self.species_energy_j_mol(key, state.temperature_K)
                      for key in self.species if state.amounts_mol[key]))

    def chemical_energy_rate_w(self, state, derivatives, contributions):
        if any(item.legacy_reaction_energy_W is not None for item in contributions):
            raise ValueError("native caloric U cannot also consume a separate reaction heat")
        return sum(self._at(key, state.temperature_K, ("internal_energy",), strict=False)[
            "internal_energy"].value * rate for key, rate in derivatives.items() if rate)

    def validate_state(self, state):
        for key in self.species:
            self._at(key, state.temperature_K, ("cv", "internal_energy"), strict=True)

    def validate_temperature_interval(self, lower_K, upper_K):
        for value in self.species.values():
            value.validate_interval(lower_K, upper_K)

    def validity_warnings(self, state):
        return tuple(dict.fromkeys(warning for value in self.species.values()
                                   for warning in _validity_warnings(
                                       value.heat_capacity, {"temperature": state.temperature_K})))

    def to_dict(self):
        return {"model": "property-package-caloric-v1",
                "species": {key: value.to_dict() for key, value in self.species.items()},
                "support_heat_capacity_J_K": self.support_heat_capacity_J_K,
                "reference_temperature_K": self.reference_temperature_K}

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        if values.pop("model") != "property-package-caloric-v1":
            raise ValueError("unsupported caloric model version")
        values["species"] = {key: SpeciesCaloricModel.from_dict(value)
                             for key, value in values["species"].items()}
        return cls(**values)


def evaluate_property_state(
    packages: Sequence[ComponentPropertyPackage], component_amounts_mol: Mapping[str, float], *,
    temperature_K: float, pressure_Pa: float, phase: str, quantities: Sequence[str],
    references: Mapping[str, CaloricReference] | None = None, eos=None,
    validity_policy: ValidityPolicy = "raise", volume_model: PropertyCorrelation | None = None,
) -> StatePropertyReport:
    """Evaluate a single-phase state using existing identities/correlations.

    Gas states declare ideal-gas behavior unless an existing cubic EOS is given.
    Cubic liquid states use the same ideal-gas caloric reference plus their
    governed departure. No phase split or missing transport data is invented.
    An explicit mixture volume_model uses the original liquid correlation at
    actual T/P and binds its fixed composition to the component inventory.
    """
    from chemworld.physchem.eos import (
        cubic_state_derivatives,
        evaluate_cubic_eos,
        ideal_gas_state,
    )
    from chemworld.physchem.property_reports import _validate_phase

    _validate_phase(phase)
    if volume_model is not None and (phase != "liquid" or eos is not None):
        raise ValueError("explicit liquid volume model cannot be combined with a fluid EOS")
    if (validity_policy not in {"raise", "warn", "ignore"}
            or any(not isfinite(v) or v <= 0 for v in (temperature_K, pressure_Pa))):
        raise ValueError("property state requires finite positive T/P and a declared policy")
    by_id = {package.component.identifier: package for package in packages}
    if not by_id or len(by_id) != len(packages):
        raise ValueError("property packages require unique existing component identities")
    components = tuple(package.component for package in packages)
    amounts = {}
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
                                     | {"fugacity_coefficient"}):
        raise ValueError("unsupported or empty state-property request")
    if "chemical_potential" in requested and any(value == 0 for value in fractions.values()):
        raise ValueError("finite chemical potentials are undefined for zero mole fractions")
    if phase != "gas" and eos is None and len(active) > 1 and requested & {
        "entropy", "chemical_potential", "fugacity_coefficient",
    }:
        raise ValueError("non-gas mixture chemical properties require an explicit activity/EOS")
    if eos is not None and phase == "solid":
        raise ValueError("a cubic fluid EOS does not describe a solid phase")
    policy = "warn" if validity_policy == "ignore" else validity_policy
    model_details = {}
    state = None
    derivatives = {}
    if eos is not None:
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
    reference_map = {}
    for alias, reference in (references or {}).items():
        canonical = resolve_component_identifier(components, alias)
        if canonical in reference_map and reference_map[canonical] != reference:
            raise ValueError("component aliases specify inconsistent caloric references")
        reference_map[canonical] = reference
    pure = {}
    warnings_by_property = {}
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
        base_phase = "gas" if eos is not None else phase
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
            correlation = _choose_valid_correlation(candidates, temperature_K=temperature_K,
                                                     pressure_Pa=pressure_Pa)
            needs = ((["cp"] if cp_needed else []) + (["enthalpy"] if energy_needed else [])
                     + (["entropy"] if entropy_needed else []))
            if direct_internal:
                needs = (
                    (["cv"] if cp_needed else []) + (["internal_energy"] if energy_needed else [])
                )
            pure[key] = evaluate_caloric_correlation(
                correlation, reference=reference, temperature_K=temperature_K,
                pressure_Pa=pressure_Pa, quantities=needs, validity_policy=policy)
    values = {}
    units = {"cp": "J/(mol*K)", "cv": "J/(mol*K)", "enthalpy": "J/mol",
             "internal_energy": "J/mol", "entropy": "J/(mol*K)",
             "dH_dT_P": "J/(mol*K)", "dU_dT_V": "J/(mol*K)",
             "molar_volume": "m^3/mol", "density": "kg/m^3", "dV_dT_P": "m^3/(mol*K)",
             "dV_dP_T": "m^3/(mol*Pa)", "viscosity": "Pa*s",
             "thermal_conductivity": "W/(m*K)", "surface_tension": "N/m"}
    source_ids = tuple(dict.fromkeys(value.correlation_id
                                    for component in pure.values() for value in component.values()))

    def weighted(quantity):
        return sum(fraction * pure[key][quantity].value for key, fraction in active.items())

    def warnings_for(quantity):
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
        cp = weighted("cp") + derivatives.get("cp_departure_J_mol_K", 0.0)
        values.update(cp=cp, dH_dT_P=cp)
        for quantity in ("cp", "cv", "dH_dT_P", "dU_dT_V"):
            warnings_by_property[quantity] = warnings_for("cp")
        if requested & {"cv", "dU_dT_V"}:
            if state is None:
                raise ValueError("Cv requires a declared EOS/volume closure")
            cv = cp - derivatives["cp_minus_cv_J_mol_K"]
            if cv < 0 or cp <= 0:
                raise ValueError("state heat capacity is outside the stable caloric domain")
            values.update(cv=cv, dU_dT_V=cv)
    if energy_needed and not direct_internal:
        values["enthalpy"] = weighted("enthalpy") + (
            state.molar_residual_enthalpy_J_mol if state else 0.0)
        for quantity in ("enthalpy", "internal_energy"):
            warnings_by_property[quantity] = warnings_for("enthalpy")
    if entropy_needed:
        values["entropy"] = weighted("entropy")
        warnings_by_property["entropy"] = warnings_for("entropy")
        if state is not None:
            values["entropy"] += (-R_J_PER_MOL_K * sum(x * log(x) for x in fractions.values() if x)
                                   + state.molar_residual_entropy_J_mol_K)
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
        correlation = _choose_valid_correlation(candidates, temperature_K=temperature_K,
                                                pressure_Pa=pressure_Pa)
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
                abs(by_id[key].component.molecular_weight_g_mol - mw) > 1e-9 * mw
                for key, x, mw in zip(mixture.component_ids, mixture.mole_fractions,
                                     mixture.molecular_weights_g_mol, strict=True)
            ):
                raise ValueError("fixed volume-model composition/MW must match the state")
            current = replace(mixture, temperature_K=temperature_K, pressure_Pa=pressure_Pa)
            expected = mixture_volume_evaluation(
                current, method=volume_model.metadata["mixture_method"],
                component_parameters=volume_model.metadata["component_parameters"],
                **volume_model.metadata["pressure_parameters"])
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
            molar_volume = by_id[key].component.molecular_weight_g_mol / 1000 / density.value
            for quantity in ("molar_volume", "density", "internal_energy"):
                warnings_by_property[quantity] = tuple(dict.fromkeys(
                    (*warnings_by_property.get(quantity, ()), *density.warnings)))
            source_ids = (*source_ids, density.correlation_id)
    if molar_volume is not None:
        values["molar_volume"] = molar_volume
        molar_mass = sum(x * by_id[key].component.molecular_weight_g_mol / 1000
                         for key, x in fractions.items())
        values["density"] = molar_mass / molar_volume
        if "internal_energy" in requested and not direct_internal:
            values["internal_energy"] = values["enthalpy"] - pressure_Pa * molar_volume
    for quantity, derivative in (("dV_dT_P", "d_molar_volume_dT_P"),
                                  ("dV_dP_T", "d_molar_volume_dP_T")):
        if quantity in requested:
            if derivative not in derivatives:
                raise ValueError("requested volume derivative requires a supported EOS")
            values[quantity] = derivatives[derivative]
    component_properties = {}
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
            component_properties["chemical_potential"][key] = PropertyEvaluation(
                "chemical_potential", "+".join(source_ids), "declared_reference_and_fugacity", mu,
                "J/mol", {"temperature": temperature_K, "pressure": pressure_Pa},
                tuple(dict.fromkeys((*pure[key]["enthalpy"].warnings,
                                     *pure[key]["entropy"].warnings))))
    model_id = state.eos_id if state is not None else "single-phase-caloric-correlations-v1"
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
    return StatePropertyReport(model_id, phase, temperature_K, pressure_Pa, amounts, properties,
                                reference_map, component_properties, model_details)


@dataclass(frozen=True)
class ComponentPropertyPackage:
    component: ComponentSpec
    correlations: tuple[PropertyCorrelation, ...]

    def __post_init__(self) -> None:
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

    def by_property(self, property_id: str) -> tuple[PropertyCorrelation, ...]:
        return tuple(
            correlation
            for correlation in self.correlations
            if correlation.property_id == property_id
        )

    def evaluate_state(
        self, *, temperature_K: float, pressure_Pa: float, phase: str,
        quantities: Sequence[str], reference: CaloricReference | None = None,
        validity_policy: ValidityPolicy = "raise", gas_model: str | None = None,
    ) -> StatePropertyReport:
        """Evaluate requested intensive properties on a one-mole pure basis."""
        if gas_model is not None:
            from chemworld.physchem.virial_thermodynamics import evaluate_virial_gas_state

            if gas_model != "virial" or phase != "gas" or validity_policy != "raise":
                raise ValueError("virial gas consumption requires gas phase and strict validity")
            ideal = None
            if set(quantities) & {"cp", "cv", "enthalpy", "internal_energy", "entropy",
                                  "chemical_potential", "dH_dT_P", "dU_dT_V"}:
                candidates = (*self.by_property("ideal_gas_heat_capacity"),
                              *self.by_property("isochoric_heat_capacity"))
                if len(candidates) != 1 or reference is None:
                    raise ValueError("virial gas needs one explicit ideal heat capacity/reference")
                ideal = SpeciesCaloricModel(candidates[0], reference)
            return evaluate_virial_gas_state(self, temperature_K=temperature_K,
                pressure_Pa=pressure_Pa, quantities=quantities, ideal_caloric=ideal)
        references = None if reference is None else {self.component.identifier: reference}
        return evaluate_property_state(
            (self,), {self.component.identifier: 1.0}, temperature_K=temperature_K,
            pressure_Pa=pressure_Pa, phase=phase, quantities=quantities,
            references=references, validity_policy=validity_policy)

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
                        value=self.component.molecular_weight_g_mol / 1000 / value,
                        unit="kg/m^3" if property_id == density_id else "m^3/mol")
            raise ValueError(
                f"No correlation for property {property_id!r} on component "
                f"{self.component.identifier!r}"
            )
        chosen = _choose_valid_correlation(
            candidates,
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
        chosen = _choose_valid_correlation(
            candidates,
            temperature_K=temperature_K,
            pressure_Pa=None,
        )
        return vapor_pressure_report(
            chosen,
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
        chosen = _choose_valid_correlation(
            candidates,
            temperature_K=temperature_K,
            pressure_Pa=None,
        )
        return pure_saturation_pressure_report(
            chosen,
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
        chosen = _choose_valid_correlation(
            candidates,
            temperature_K=temperature_bounds_K[0] if temperature_bounds_K else 298.15,
            pressure_Pa=None,
        )
        return pure_saturation_temperature_report(
            chosen,
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
        chosen = _choose_valid_correlation(
            candidates,
            temperature_K=temperature_bounds_K[0] if temperature_bounds_K else 298.15,
            pressure_Pa=None,
        )
        return normal_boiling_point_report(
            chosen,
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
        }
