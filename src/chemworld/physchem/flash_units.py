"""Auditable TP flash unit with material, phase, and enthalpy ledgers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
from math import isfinite, log, ulp
from types import MappingProxyType
from typing import Any

from scipy.optimize import brentq

from chemworld.physchem.composition_math import _positive_feed_amounts
from chemworld.physchem.equilibrium import (
    ActivityModelSpec,
    gamma_phi_k_value_report,
    rachford_rice_diagnostic_report,
)
from chemworld.physchem.property_packages import ComponentPropertyPackage, evaluate_property_state
from chemworld.physchem.property_reports import R_J_PER_MOL_K, CaloricReference
from chemworld.physchem.specs import resolve_component_identifier


def phase_advection_rate(amounts_mol, enthalpy_J, rate_mol_s):
    """One phase-stream contribution, with mol/s and J/s from the same phase.

    The caller supplies its constitutive flow law and debits/credits endpoints.
    This function owns no material, utility account or elapsed-time loop.
    """
    if (not isfinite(enthalpy_J) or not isfinite(rate_mol_s) or rate_mol_s < 0
            or any(not isfinite(v) or v < 0 for v in amounts_mol.values())):
        raise ValueError("phase advection requires finite H and nonnegative amounts/rate")
    total = sum(amounts_mol.values())
    if not total:
        if rate_mol_s:
            raise ValueError("an empty phase cannot supply a material stream")
        return dict.fromkeys(amounts_mol,0.0),0.0
    return ({key:rate_mol_s*amount/total for key,amount in amounts_mol.items()},
            rate_mol_s*enthalpy_J/total)


@dataclass(frozen=True)
class TPFlashUnitResult:
    model_id: str
    temperature_K: float
    pressure_Pa: float
    feed_amounts_mol: dict[str, float]
    vapor_fraction: float
    liquid_amounts_mol: dict[str, float]
    vapor_amounts_mol: dict[str, float]
    liquid_composition: dict[str, float]
    vapor_composition: dict[str, float]
    k_values: dict[str, float]
    activity_coefficients: dict[str, float]
    vapor_fugacity_coefficients: dict[str, float]
    feed_enthalpy_J: float | None
    liquid_enthalpy_J: float
    vapor_enthalpy_J: float
    heat_duty_J: float | None
    material_balance_error_mol: float
    energy_balance_error_J: float | None
    iterations: int
    converged: bool
    phase_status: str
    warnings: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)
    energy_specification: str = "TP"
    support_enthalpy_J: float = 0.0

    @property
    def total_enthalpy_j(self):
        return self.liquid_enthalpy_J + self.vapor_enthalpy_J + self.support_enthalpy_J

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "feed_amounts_mol": dict(self.feed_amounts_mol),
            "vapor_fraction": self.vapor_fraction,
            "liquid_amounts_mol": dict(self.liquid_amounts_mol),
            "vapor_amounts_mol": dict(self.vapor_amounts_mol),
            "liquid_composition": dict(self.liquid_composition),
            "vapor_composition": dict(self.vapor_composition),
            "k_values": dict(self.k_values),
            "activity_coefficients": dict(self.activity_coefficients),
            "vapor_fugacity_coefficients": dict(self.vapor_fugacity_coefficients),
            "feed_enthalpy_J": self.feed_enthalpy_J,
            "liquid_enthalpy_J": self.liquid_enthalpy_J,
            "vapor_enthalpy_J": self.vapor_enthalpy_J,
            "heat_duty_J": self.heat_duty_J,
            "material_balance_error_mol": self.material_balance_error_mol,
            "energy_balance_error_J": self.energy_balance_error_J,
            "iterations": self.iterations,
            "converged": self.converged,
            "phase_status": self.phase_status,
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
            "energy_specification": self.energy_specification,
            "total_enthalpy_j": self.total_enthalpy_j,
            "support_enthalpy_J": self.support_enthalpy_J,
        }


def _iterate_vle(
    overall, evaluate_k: Callable, *, tolerance, max_iterations, nonvolatile_component_ids=(),
):
    """One iteration core for state-aware and fixed-property entry points."""
    if not isfinite(tolerance) or tolerance <= 0 or max_iterations <= 0:
        raise ValueError("flash requires positive tolerance and iteration limit")
    liquid, vapor = dict(overall), dict(overall)
    for iteration in range(1, max_iterations + 1):
        report = evaluate_k(liquid, vapor)
        phase = rachford_rice_diagnostic_report(overall, report.k_values,
            nonvolatile_component_ids=nonvolatile_component_ids)
        refreshed = evaluate_k(phase.liquid_composition, phase.vapor_composition)
        error = max(abs(log(refreshed.k_values[k] / report.k_values[k]))
                    for k in overall if k not in nonvolatile_component_ids)
        if error <= tolerance:
            return refreshed, phase, iteration, True, error
        liquid = {k: .5 * liquid[k] + .5 * phase.liquid_composition[k] for k in overall}
        vapor = {k: .5 * vapor[k] + .5 * phase.vapor_composition[k] for k in overall}
    return report, phase, iteration, False, error


@dataclass(frozen=True)
class VLEPropertyPackage:
    """One gas reference and a Raoult liquid free-energy constitutive model.

    Liquid volume/Poynting work is neglected. Liquid H follows the temperature
    derivative of the same Psat/gamma used for equilibrium, not a second zero.
    An optional existing gas EOS supplies current-y phi and departure H.
    """

    packages: tuple[ComponentPropertyPackage, ...]
    gas_references: Mapping[str, CaloricReference]
    temperature_bounds_K: tuple[float, float]
    activity_model: ActivityModelSpec | None = None
    vapor_eos: object = None
    nonvolatile_references: Mapping[str, CaloricReference] = field(default_factory=dict)

    def __post_init__(self):
        # Endpoint caches must not outlive a mutation of correlation data.
        # Own immutable copies rather than relying on shallow frozen dataclasses.
        packages = []
        for package in self.packages:
            correlations = []
            for correlation in package.correlations:
                copied = replace(correlation)
                for name in ("coefficients", "input_units", "validity_ranges", "metadata"):
                    values = dict(getattr(copied,name))
                    if name == "validity_ranges":
                        values = {key:tuple(value) for key,value in values.items()}
                    object.__setattr__(copied,name,MappingProxyType(values))
                correlations.append(copied)
            packages.append(replace(package, correlations=tuple(correlations)))
        object.__setattr__(self, "packages", tuple(packages))
        object.__setattr__(self, "gas_references", MappingProxyType(dict(self.gas_references)))
        object.__setattr__(self, "temperature_bounds_K", tuple(self.temperature_bounds_K))
        object.__setattr__(self,"nonvolatile_references",
                           MappingProxyType(dict(self.nonvolatile_references)))
        ids = self.component_ids
        if len(ids) < 2 or len(ids) > 3 or len(set(ids)) != len(ids):
            raise ValueError("VLE supports two/three components and at most one liquid and one gas")
        if (set(self.gas_references)|set(self.nonvolatile_references) != set(ids)
                or set(self.gas_references)&set(self.nonvolatile_references)
                or not self.gas_references
                or any(ref.phase != "gas" or ref.energy_basis != "enthalpy"
                       for ref in self.gas_references.values())
                or any(ref.phase != "liquid" or ref.energy_basis != "enthalpy"
                       for ref in self.nonvolatile_references.values())):
            raise ValueError(
                "VLE needs gas H for volatile species and liquid H for nonvolatile species")
        lower, upper = self.temperature_bounds_K
        if not all(isfinite(v) for v in (lower, upper)) or not 0 < lower < upper:
            raise ValueError("flash temperature bounds must be finite, positive and ordered")
        if self.activity_model is None:
            object.__setattr__(self, "activity_model", ActivityModelSpec("ideal-vle", ids))
        if set(self.activity_model.component_ids) != set(ids):
            raise ValueError("VLE activity identities must match property packages")
        if self.nonvolatile_references and (
            self.vapor_eos is not None or self.activity_model.model != "ideal"
        ):
            raise ValueError("the nonvolatile-liquid slice requires ideal liquid and vapor")
        if self.vapor_eos is not None and set(self.vapor_eos.component_ids) != set(ids):
            raise ValueError("VLE gas EOS identities must match property packages")

    @property
    def component_ids(self):
        return tuple(package.component.identifier for package in self.packages)

    def to_dict(self):
        return {"packages": [p.to_dict() for p in self.packages],
                "gas_references": {k: v.to_dict() for k, v in self.gas_references.items()},
                "temperature_bounds_K": list(self.temperature_bounds_K),
                "activity_model": self.activity_model.to_dict(),
                "vapor_eos": None if self.vapor_eos is None else self.vapor_eos.to_dict(),
                **({"nonvolatile_references":{
                    k:v.to_dict() for k,v in self.nonvolatile_references.items()}}
                   if self.nonvolatile_references else {})}

    @classmethod
    def from_dict(cls, payload):
        from chemworld.physchem.eos import fluid_eos_from_dict

        values = dict(payload)
        values["packages"] = tuple(ComponentPropertyPackage.from_dict(p)
            for p in values["packages"])
        values["gas_references"] = {k: CaloricReference(**v)
                                    for k, v in values["gas_references"].items()}
        values["nonvolatile_references"] = {k:CaloricReference(**v)
            for k,v in values.get("nonvolatile_references",{}).items()}
        values["temperature_bounds_K"] = tuple(values["temperature_bounds_K"])
        model = dict(values["activity_model"])
        model["component_ids"] = tuple(model["component_ids"])
        values["activity_model"] = ActivityModelSpec(**model)
        if values["vapor_eos"] is not None:
            values["vapor_eos"] = fluid_eos_from_dict(values["vapor_eos"])
        return cls(**values)

    def canonical_amounts(self, amounts):
        components = tuple(package.component for package in self.packages)
        canonical = dict.fromkeys(self.component_ids, 0.0)
        for alias, amount in _positive_feed_amounts(amounts).items():
            canonical[resolve_component_identifier(components, alias)] += amount
        return canonical

    def _check_temperature(self, temperature):
        if not isfinite(temperature) or not self.temperature_bounds_K[0] <= temperature <= (
            self.temperature_bounds_K[1]
        ):
            raise ValueError("flash temperature is outside its declared domain")

    def saturation(self, temperature):
        self._check_temperature(temperature)
        return {p.component.identifier: p.vapor_pressure_report(
            temperature_K=temperature, validity_policy="raise") for p in self.packages
            if p.component.identifier not in self.nonvolatile_references}

    def k_report(self, temperature, pressure, liquid, vapor):
        phi = None
        if self.vapor_eos is not None:
            phi_report = evaluate_property_state(
                self.packages, vapor, temperature_K=temperature, pressure_Pa=pressure,
                phase="gas", quantities=("fugacity_coefficient",), eos=self.vapor_eos)
            phi = {k: v.value for k, v in phi_report.component_properties[
                "fugacity_coefficient"].items()}
        return gamma_phi_k_value_report(
            self.activity_model, liquid, temperature_K=temperature, pressure_Pa=pressure,
            vapor_pressures_Pa={**{k:v.pressure_pa for k,v in self.saturation(temperature).items()},
                                **dict.fromkeys(self.nonvolatile_references,0.0)},
            vapor_fugacity_coefficients=phi,
            nonvolatile_component_ids=tuple(self.nonvolatile_references))

    def phase_enthalpies(self, temperature, pressure, liquid, vapor):
        saturation = self.saturation(temperature)
        # Gas H uses task 17's datum, integration and EOS departure evaluation.
        pure_h = {p.component.identifier: p.evaluate_state(
            temperature_K=temperature, pressure_Pa=pressure, phase="gas",
            quantities=("enthalpy",), reference=self.gas_references[p.component.identifier]
        ).properties["enthalpy"].value for p in self.packages
            if p.component.identifier in self.gas_references}
        liquid_pure_h = {key:value-R_J_PER_MOL_K*temperature**2
                         *saturation[key].dln_pressure_dT_1_K for key,value in pure_h.items()}
        for package in self.packages:
            key = package.component.identifier
            if key in self.nonvolatile_references:
                liquid_pure_h[key] = package.evaluate_state(
                    temperature_K=temperature,pressure_Pa=pressure,phase="liquid",
                    quantities=("enthalpy",),reference=self.nonvolatile_references[key]
                ).properties["enthalpy"].value
        liquid_h = sum(liquid[k]*liquid_pure_h[k] for k in liquid)
        if self.activity_model.model != "ideal":
            from chemworld.physchem.equilibrium import activity_coefficients

            # Temperature derivative of the same excess free energy at fixed x.
            step = min(temperature * 1e-4, (self.temperature_bounds_K[1]
                       - self.temperature_bounds_K[0]) * 1e-4)
            lo = max(self.temperature_bounds_K[0], temperature - step)
            hi = min(self.temperature_bounds_K[1], temperature + step)
            low = activity_coefficients(self.activity_model, liquid, temperature_K=lo)
            high = activity_coefficients(self.activity_model, liquid, temperature_K=hi)
            liquid_h -= R_J_PER_MOL_K * temperature ** 2 * sum(
                liquid[k] * log(high[k] / low[k]) / (hi - lo) for k in liquid)
        if self.vapor_eos is None:
            vapor_h = (sum(vapor[k]*pure_h[k] for k in pure_h)/sum(vapor.values())
                if sum(vapor.values()) and not any(
                    vapor.get(k,0) for k in self.nonvolatile_references)
                else None)
            return liquid_h, vapor_h
        vapor_h = evaluate_property_state(
            self.packages, vapor, temperature_K=temperature, pressure_Pa=pressure, phase="gas",
            quantities=("enthalpy",), references=self.gas_references,
            eos=self.vapor_eos).properties["enthalpy"].value
        return liquid_h, vapor_h

    def inventory_enthalpies(self, temperature, pressure, liquid_amounts, vapor_amounts):
        """H of the actual phase inventories, including pure saturation plateaus.

        This does not reflash n/T or change the supplied phase split. Both
        standalone closures and runtime inventory readers use the same datum.
        """
        self._check_temperature(temperature)
        if not isfinite(pressure) or pressure <= 0:
            raise ValueError("phase inventory enthalpy requires finite positive pressure")
        ids = set(self.component_ids)
        for amounts in (liquid_amounts, vapor_amounts):
            if set(amounts) != ids or any(not isfinite(n) or n < 0 for n in amounts.values()):
                raise ValueError("phase inventory must match the declared finite component layout")
        nl, nv = sum(liquid_amounts.values()), sum(vapor_amounts.values())
        if not nl and not nv:
            return 0.0, 0.0
        liquid = {k: n / nl for k, n in liquid_amounts.items()} if nl else None
        vapor = {k: n / nv for k, n in vapor_amounts.items()} if nv else None
        hl, hv = self.phase_enthalpies(temperature, pressure, liquid or vapor, vapor or liquid)
        if nv and hv is None:
            raise ValueError("nonvolatile liquid material cannot own gas-phase enthalpy")
        return nl * hl, nv * hv if nv else 0.0

    @cached_property
    def _ideal_endpoint_data(self):
        if (self.activity_model.model != "ideal" or self.vapor_eos is not None
                or self.nonvolatile_references):
            return None
        data = []
        for temperature in self.temperature_bounds_K:
            psat = tuple(r.pressure_pa for r in self.saturation(temperature).values())
            energies = {}
            for key in self.component_ids:
                pure = {k: float(k == key) for k in self.component_ids}
                energies[key] = self.phase_enthalpies(temperature, 1e5, pure, pure)
            data.append((psat, energies))
        return tuple(data)

    def ideal_enthalpy_bounds(self, amounts, pressure, capacity, reference_temperature):
        """Exact linear H bounds only when both endpoints are single phase.

        Cached data depend solely on immutable constitutive inputs. Nonideal
        models or endpoints inside a phase envelope use the general TP closure.
        """
        data = self._ideal_endpoint_data
        if data is None or not (max(data[0][0]) < pressure < min(data[1][0])):
            return None
        return tuple(
            sum(amounts[k] * data[i][1][k][i] for k in self.component_ids)
            + capacity * (temperature - reference_temperature)
            for i, temperature in enumerate(self.temperature_bounds_K)
        )

    def liquid_stability(self, temperature):
        """Reject known nonconvex liquid branches; full LLE belongs to task 19."""
        if self.activity_model.model == "ideal":
            return "convex_ideal_liquid"
        # This deliberately finite first VLE slice does not claim a general TPD
        # solver. Only a globally convex binary Margules liquid is admitted.
        if self.activity_model.model != "margules" or len(self.component_ids) != 2:
            raise ValueError("this VLE slice requires a certified convex liquid; LLE unsupported")
        from chemworld.physchem.equilibrium import activity_coefficients

        left, right = self.component_ids
        def g(x):
            composition = {left: x, right: 1 - x}
            gamma = activity_coefficients(
                self.activity_model, composition, temperature_K=temperature)
            return sum(v * log(v * gamma[k]) for k, v in composition.items())
        # Three-suffix Margules has linear excess curvature. Since ideal
        # curvature >=4, max(A_ij)<=2 and min(A_ij)>=0 is a sufficient bound.
        values = tuple(self.activity_model.parameters.values())
        if any(not 0 <= value < 2 for value in values):
            raise ValueError("nonconvex/uncertified liquid may require two liquid phases")
        # Asymmetric models can violate the simple symmetric bound: inspect
        # the exact cubic excess curvature at both ends, bounded by ideal >=4.
        # The GE contract treats a single declared direction as symmetric.
        # Stability must use that same pair, not invent a zero reverse term.
        parameters = self.activity_model.parameters
        ij, ji = f"A:{left}|{right}", f"A:{right}|{left}"
        a12 = parameters[ij] if ij in parameters else parameters[ji]
        a21 = parameters[ji] if ji in parameters else parameters[ij]
        if min(2 * a12 - 4 * a21, 2 * a21 - 4 * a12) < -4:
            raise ValueError("uncertified asymmetric liquid stability; LLE unsupported")
        if not isfinite(g(.5)):
            raise ValueError("invalid liquid free energy")
        return "convex_binary_margules_sufficient_bound"


def tp_flash(
    properties: VLEPropertyPackage, feed_amounts_mol: Mapping[str, float], *,
    temperature_K: float, pressure_Pa: float, feed_enthalpy_J: float | None = None,
    tolerance: float = 1e-10, max_iterations: int = 200,
    support_heat_capacity_J_K: float = 0.0, support_reference_temperature_K: float = 298.15,
) -> TPFlashUnitResult:
    if (not isfinite(support_heat_capacity_J_K) or support_heat_capacity_J_K < 0
            or not isfinite(support_reference_temperature_K)
            or support_reference_temperature_K <= 0):
        raise ValueError("flash support requires nonnegative capacity and positive reference")
    feed = properties.canonical_amounts(feed_amounts_mol)
    total = sum(feed.values())
    overall = {k: v / total for k, v in feed.items()}
    stability = properties.liquid_stability(temperature_K)
    report, phase, iterations, converged, error = _iterate_vle(
        overall, lambda x, y: properties.k_report(temperature_K, pressure_Pa, x, y),
        tolerance=tolerance, max_iterations=max_iterations,
        nonvolatile_component_ids=tuple(properties.nonvolatile_references))
    if not converged:
        raise RuntimeError(f"state-dependent VLE did not converge: residual={error:g}")
    liquid = {k: total * (1 - phase.vapor_fraction) * phase.liquid_composition[k] for k in feed}
    vapor = {k: total * phase.vapor_fraction * phase.vapor_composition[k] for k in feed}
    liquid_total, vapor_total = total*(1-phase.vapor_fraction),total*phase.vapor_fraction
    # Single-phase closure allocates the supplied inventory directly. Rebuilding
    # n from total*x would change stationary inventories by roundoff on each PH
    # query, even when the physical process transfers no material.
    if phase.phase_status == "all_liquid":
        liquid, vapor = dict(feed), dict.fromkeys(feed, 0.0)
    elif phase.phase_status == "all_vapor":
        liquid, vapor = dict.fromkeys(feed, 0.0), dict(feed)
    if properties.nonvolatile_references:
        # A declared nonvolatile inventory has the exact constraint n_V=0.
        # Allocate complementary inventories directly; do not independently
        # renormalize its retained amount or invent a gas caloric reference.
        if phase.phase_status not in {"all_liquid", "all_vapor"}:
            fraction = phase.liquid_fraction
            liquid = {key:(feed[key] if key in properties.nonvolatile_references else
                feed[key]*fraction/(fraction+phase.vapor_fraction*report.k_values[key]))
                for key in feed}
            vapor = {key:feed[key]-liquid[key] for key in feed}
        liquid_total,vapor_total = sum(liquid.values()),sum(vapor.values())
        phase = replace(phase,vapor_fraction=vapor_total/total,
            liquid_composition=({k:v/liquid_total for k,v in liquid.items()}
                                if liquid_total else phase.liquid_composition),
            vapor_composition=({k:v/vapor_total for k,v in vapor.items()}
                               if vapor_total else phase.vapor_composition))
    liquid_h, vapor_h = properties.inventory_enthalpies(
        temperature_K, pressure_Pa, liquid, vapor)
    if feed_enthalpy_J is not None and not isfinite(feed_enthalpy_J):
        raise ValueError("feed enthalpy must be finite")
    support_h = support_heat_capacity_J_K * (temperature_K - support_reference_temperature_K)
    duty = None if feed_enthalpy_J is None else liquid_h + vapor_h + support_h - feed_enthalpy_J
    return TPFlashUnitResult(
        "state_gamma_phi_flash_v1", temperature_K, pressure_Pa, feed, phase.vapor_fraction,
        liquid, vapor, phase.liquid_composition, phase.vapor_composition, report.k_values,
        report.activity_coefficients, report.vapor_fugacity_coefficients, feed_enthalpy_J,
        liquid_h, vapor_h, duty,
        sum(abs(feed[k] - liquid[k] - vapor[k]) for k in feed),
        None if duty is None else abs(feed_enthalpy_J + duty - liquid_h - vapor_h - support_h),
        iterations, True, phase.phase_status, phase.warnings,
        {"iteration_residual": error, "liquid_stability": stability,
         "phase_capacity": "one_liquid_one_gas", "liquid_volume_model": "neglected",
         "vapor_composition_updated": True}, support_enthalpy_J=support_h)


def ph_flash(
    properties: VLEPropertyPackage,
    feed_amounts_mol: Mapping[str, float],
    *,
    pressure_Pa: float,
    enthalpy_J: float,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
    support_heat_capacity_J_K: float = 0.0,
    support_reference_temperature_K: float = 298.15,
) -> TPFlashUnitResult:
    if (
        not isfinite(pressure_Pa)
        or pressure_Pa <= 0
        or not isfinite(support_heat_capacity_J_K)
        or support_heat_capacity_J_K < 0
        or not isfinite(support_reference_temperature_K)
        or support_reference_temperature_K <= 0
    ):
        raise ValueError("PH requires positive pressure/reference and nonnegative support capacity")
    if not isfinite(enthalpy_J):
        raise ValueError("PH flash requires finite prescribed total enthalpy")

    def at(temperature):
        return tp_flash(properties, feed_amounts_mol, temperature_K=temperature,
                        pressure_Pa=pressure_Pa, tolerance=tolerance, max_iterations=max_iterations,
                        support_heat_capacity_J_K=support_heat_capacity_J_K,
                        support_reference_temperature_K=support_reference_temperature_K)

    lower, upper = properties.temperature_bounds_K
    feed = properties.canonical_amounts(feed_amounts_mol)
    bounds = properties.ideal_enthalpy_bounds(
        feed, pressure_Pa, support_heat_capacity_J_K, support_reference_temperature_K
    )
    if bounds is None:
        bounds = (at(lower).total_enthalpy_j, at(upper).total_enthalpy_j)
    if not bounds[0] <= enthalpy_J <= bounds[1]:
        raise ValueError("prescribed enthalpy is unreachable inside the flash temperature domain")
    # A vanished component leaves a pure saturation plateau. PH then solves
    # the lever rule at Tsat; a scalar T root cannot traverse a discontinuous H.
    active = [key for key, value in feed.items() if value]
    if len(active) == 1 and active[0] not in properties.nonvolatile_references:
        total = sum(feed.values())
        composition = {key: value / total for key, value in feed.items()}
        key = active[0]
        def ln_k(temperature):
            return log(properties.k_report(temperature, pressure_Pa,
                                            composition, composition).k_values[key])
        if ln_k(lower) < 0 < ln_k(upper):
            saturation = brentq(ln_k, lower, upper, xtol=1e-10)
            hl, hv = properties.phase_enthalpies(saturation, pressure_Pa,
                                                 composition, composition)
            support_h = support_heat_capacity_J_K * (saturation - support_reference_temperature_K)
            fluid_h = enthalpy_J - support_h
            if hl * total <= fluid_h <= hv * total:
                beta = (fluid_h / total - hl) / (hv - hl)
                result = at(saturation)
                return replace(
                    result, vapor_fraction=beta,
                    liquid_amounts_mol={k: v * (1 - beta) for k, v in feed.items()},
                    vapor_amounts_mol={k: v * beta for k, v in feed.items()},
                    liquid_composition=composition, vapor_composition=composition,
                    liquid_enthalpy_J=total * (1 - beta) * hl, vapor_enthalpy_J=total * beta * hv,
                    energy_specification="PH", feed_enthalpy_J=enthalpy_J, heat_duty_J=0.0,
                    energy_balance_error_J=abs(total * ((1 - beta) * hl + beta * hv) - fluid_h),
                    phase_status="two_phase" if 0 < beta < 1 else (
                        "all_liquid" if beta == 0 else "all_vapor"),
                    diagnostics={**result.diagnostics, "pure_saturation_lever_rule": True})
    # The exact single-phase bound and the fully reported TP sum can differ
    # by a few ulps in summation order. Preserve endpoint solutions rather than
    # asking a sign-bracket solver to cross that floating-point discrepancy.
    endpoint = next(
        (
            temperature
            for temperature, value in zip((lower, upper), bounds, strict=True)
            if abs(enthalpy_J - value) <= 16 * ulp(max(1.0, abs(value)))
        ),
        None,
    )
    if endpoint is not None:
        root, temperature_iterations = endpoint, 0
    else:
        root, diagnostic = brentq(
            lambda temperature: at(temperature).total_enthalpy_j - enthalpy_J,
            lower,
            upper,
            xtol=1e-9,
            rtol=1e-12,
            maxiter=max_iterations,
            full_output=True,
        )
        temperature_iterations = diagnostic.iterations
    result = at(float(root))
    residual = abs(result.total_enthalpy_j - enthalpy_J)
    if residual > 1e-6 * max(1, abs(enthalpy_J)):
        raise RuntimeError("PH enthalpy closure failed")
    return replace(result, energy_specification="PH", feed_enthalpy_J=enthalpy_J,
                   heat_duty_J=0.0, energy_balance_error_J=residual,
                   diagnostics={**result.diagnostics,
                                "temperature_iterations": temperature_iterations,
                                "prescribed_enthalpy_J": enthalpy_J})


@dataclass(frozen=True)
class FlashSystemSpec:
    """Finite, closed-material constant-P caloric system; owns no state or clock.

    H is the conserved thermal coordinate at fixed P. The nominal liquid molar
    volume is a geometric filling model, omitted from the Raoult caloric EOS.
    This model does not provide an exact liquid U/PV-work balance.
    """

    properties: VLEPropertyPackage
    pressure_Pa: float = 1e5
    support_heat_capacity_J_K: float = 1.0
    support_reference_temperature_K: float = 298.15
    environment_temperature_K: float = 300.0
    loss_conductance_W_K: float = .01
    stop_temperature_K: float = 490.0
    liquid_molar_holdup_L_mol: float = .05

    def __post_init__(self):
        if (any(not isfinite(v) or v <= 0 for v in (
                self.pressure_Pa, self.support_heat_capacity_J_K,
                self.support_reference_temperature_K, self.environment_temperature_K,
                self.liquid_molar_holdup_L_mol))
                or not isfinite(self.loss_conductance_W_K) or self.loss_conductance_W_K < 0
                or not self.properties.temperature_bounds_K[0] < self.stop_temperature_K < (
                    self.properties.temperature_bounds_K[1])):
            raise ValueError("invalid finite flash thermal/pressure/holdup parameters")

    def _kwargs(self):
        return {"pressure_Pa": self.pressure_Pa,
                "support_heat_capacity_J_K": self.support_heat_capacity_J_K,
                "support_reference_temperature_K": self.support_reference_temperature_K}

    def at_tp(self, amounts, temperature):
        if any(not isfinite(v) or v < 0 for v in amounts.values()):
            raise ValueError("flash amounts must be finite and nonnegative")
        if sum(amounts.values()) == 0:
            if set(amounts) != set(self.properties.component_ids):
                raise ValueError("empty flash inventory must retain its component identities")
            # Empty material retains the apparatus capacity, not a ghost fluid.
            report = tp_flash(self.properties, dict.fromkeys(self.properties.component_ids, 1.0),
                              temperature_K=temperature, **self._kwargs())
            zeros = dict.fromkeys(self.properties.component_ids, 0.0)
            return replace(report, feed_amounts_mol=zeros, liquid_amounts_mol=zeros,
                           vapor_amounts_mol=zeros, liquid_enthalpy_J=0, vapor_enthalpy_J=0,
                           vapor_fraction=0, phase_status="empty", material_balance_error_mol=0)
        return tp_flash(self.properties, amounts, temperature_K=temperature, **self._kwargs())

    def at_ph(self, amounts, enthalpy):
        if sum(amounts.values()) == 0:
            temperature = (self.support_reference_temperature_K
                           + enthalpy / self.support_heat_capacity_J_K)
            return replace(self.at_tp(amounts, temperature), energy_specification="PH",
                           feed_enthalpy_J=enthalpy, heat_duty_J=0, energy_balance_error_J=0)
        return ph_flash(self.properties, amounts, enthalpy_J=enthalpy, **self._kwargs())

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import fields

        return {f.name: (self.properties.to_dict() if f.name == "properties"
                         else getattr(self, f.name))
                for f in fields(self)}

    @classmethod
    def from_dict(cls, payload):
        values = dict(payload)
        values["properties"] = VLEPropertyPackage.from_dict(values["properties"])
        return cls(**values)


def tp_flash_with_energy_balance(
    feed_amounts_mol: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
    vapor_pressures_Pa: Mapping[str, float],
    feed_molar_enthalpies_J_mol: Mapping[str, float],
    liquid_molar_enthalpies_J_mol: Mapping[str, float],
    vapor_molar_enthalpies_J_mol: Mapping[str, float],
    activity_model: ActivityModelSpec | None = None,
    vapor_fugacity_coefficients: Mapping[str, float] | None = None,
    liquid_reference_fugacity_coefficients: Mapping[str, float] | None = None,
    poynting_factors: Mapping[str, float] | None = None,
    tolerance: float = 1.0e-10,
    max_iterations: int = 100,
) -> TPFlashUnitResult:
    """Solve a fixed-TP gamma-phi flash and close material/enthalpy ledgers."""

    if temperature_K <= 0.0 or not isfinite(temperature_K):
        raise ValueError("temperature_K must be finite and positive")
    if pressure_Pa <= 0.0 or not isfinite(pressure_Pa):
        raise ValueError("pressure_Pa must be finite and positive")
    if tolerance <= 0.0 or not isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    feed = _positive_feed_amounts(feed_amounts_mol)
    component_ids = tuple(feed)
    total_feed = sum(feed.values())
    overall = {component_id: amount / total_feed for component_id, amount in feed.items()}
    _require_component_mapping(component_ids, vapor_pressures_Pa, "vapor_pressures_Pa")
    _require_component_mapping(
        component_ids,
        feed_molar_enthalpies_J_mol,
        "feed_molar_enthalpies_J_mol",
        positive=False,
    )
    _require_component_mapping(
        component_ids,
        liquid_molar_enthalpies_J_mol,
        "liquid_molar_enthalpies_J_mol",
        positive=False,
    )
    _require_component_mapping(
        component_ids,
        vapor_molar_enthalpies_J_mol,
        "vapor_molar_enthalpies_J_mol",
        positive=False,
    )
    resolved_activity = activity_model or ActivityModelSpec(
        "ideal_tp_flash",
        component_ids,
        "ideal",
    )
    if set(resolved_activity.component_ids) != set(component_ids):
        raise ValueError("activity_model components must match feed components")

    report, phase, iterations, converged, _error = _iterate_vle(
        overall,
        lambda liquid, vapor: gamma_phi_k_value_report(
            resolved_activity, liquid, vapor_pressures_Pa=vapor_pressures_Pa,
            pressure_Pa=pressure_Pa, temperature_K=temperature_K,
            vapor_fugacity_coefficients=vapor_fugacity_coefficients,
            liquid_reference_fugacity_coefficients=liquid_reference_fugacity_coefficients,
            poynting_factors=poynting_factors),
        tolerance=tolerance, max_iterations=max_iterations,
    )

    liquid_total = total_feed * (1.0 - phase.vapor_fraction)
    vapor_total = total_feed * phase.vapor_fraction
    liquid_amounts = {key: liquid_total * phase.liquid_composition[key] for key in component_ids}
    vapor_amounts = {key: vapor_total * phase.vapor_composition[key] for key in component_ids}
    material_error = sum(
        abs(feed[key] - liquid_amounts[key] - vapor_amounts[key]) for key in component_ids
    )
    feed_enthalpy = sum(
        feed[key] * float(feed_molar_enthalpies_J_mol[key]) for key in component_ids
    )
    liquid_enthalpy = sum(
        liquid_amounts[key] * float(liquid_molar_enthalpies_J_mol[key]) for key in component_ids
    )
    vapor_enthalpy = sum(
        vapor_amounts[key] * float(vapor_molar_enthalpies_J_mol[key]) for key in component_ids
    )
    heat_duty = liquid_enthalpy + vapor_enthalpy - feed_enthalpy
    energy_error = abs(feed_enthalpy + heat_duty - liquid_enthalpy - vapor_enthalpy)
    warnings = list(phase.warnings)
    if not converged:
        warnings.append("gamma_phi_iteration_not_converged")
    return TPFlashUnitResult(
        model_id="tp_gamma_phi_flash_energy_balance_v1",
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        feed_amounts_mol=feed,
        vapor_fraction=phase.vapor_fraction,
        liquid_amounts_mol=liquid_amounts,
        vapor_amounts_mol=vapor_amounts,
        liquid_composition=phase.liquid_composition,
        vapor_composition=phase.vapor_composition,
        k_values=report.k_values,
        activity_coefficients=report.activity_coefficients,
        vapor_fugacity_coefficients=report.vapor_fugacity_coefficients,
        feed_enthalpy_J=feed_enthalpy,
        liquid_enthalpy_J=liquid_enthalpy,
        vapor_enthalpy_J=vapor_enthalpy,
        heat_duty_J=heat_duty,
        material_balance_error_mol=material_error,
        energy_balance_error_J=energy_error,
        iterations=iterations,
        converged=converged,
        phase_status=phase.phase_status,
        warnings=tuple(warnings),
    )


def _require_component_mapping(
    component_ids: tuple[str, ...],
    values: Mapping[str, float],
    field_name: str,
    *,
    positive: bool = True,
) -> None:
    if set(values) != set(component_ids):
        raise ValueError(f"{field_name} keys must exactly match feed components")
    converted = [float(values[key]) for key in component_ids]
    if any(not isfinite(value) for value in converted):
        raise ValueError(f"{field_name} must contain finite values")
    if positive and any(value <= 0.0 for value in converted):
        raise ValueError(f"{field_name} must contain positive values")


__all__ = ["FlashSystemSpec", "TPFlashUnitResult", "VLEPropertyPackage", "ph_flash", "tp_flash",
           "tp_flash_with_energy_balance"]
