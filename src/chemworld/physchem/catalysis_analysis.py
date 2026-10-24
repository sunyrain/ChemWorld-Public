"""Steady surface/control queries over the original declared catalytic mechanism.

Gas pressures are an explicit analysis boundary. No gas reservoir, world state,
or alternative rate evaluator is persisted or simulated by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import exp, isfinite, log
from types import MappingProxyType
from typing import Any

import numpy as np

from chemworld.physchem.catalysis_network import (
    CatalyticMechanismSpec,
    CatalyticReactionSpec,
    catalytic_inventory_key,
    evaluate_catalytic_rates,
    make_catalytic_slot,
    pack_catalytic_state,
)
from chemworld.physchem.catalysis_steady_chart import build_catalytic_steady_chart
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualDomainError,
    ResidualEquation,
    ResidualEvaluator,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)


@dataclass(frozen=True)
class CatalyticAnalysisConditions:
    temperature_K: float
    gas_partial_pressures_Pa: Mapping[str, float]
    capacities_mol: Mapping[str, float]
    volume_L: float = 1.0

    def __post_init__(self) -> None:
        for name in ("gas_partial_pressures_Pa", "capacities_mol"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (self.temperature_K, self.volume_L, *self.capacities_mol.values())
        ):
            raise ValueError("surface analysis requires positive finite T, volume and capacities")
        if (
            not self.capacities_mol
            or not self.gas_partial_pressures_Pa
            or any(
                isinstance(v, bool) or not isfinite(v) or v < 0
                for v in self.gas_partial_pressures_Pa.values()
            )
        ):
            raise ValueError("surface analysis requires declared finite nonnegative gas pressures")

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_K": self.temperature_K,
            "volume_L": self.volume_L,
            "gas_partial_pressures_Pa": dict(self.gas_partial_pressures_Pa),
            "capacities_mol": dict(self.capacities_mol),
            "boundary": "held-gas-partial-pressures; surface-only steady analysis",
        }


def with_catalytic_descriptors(
    spec: CatalyticMechanismSpec, descriptors: Mapping[str, float]
) -> CatalyticMechanismSpec:
    """Re-evaluate declared adsorbate scaling without modifying the input model."""
    axes = {
        a.descriptor_id
        for s in spec.species
        if s.adsorbate is not None
        for a in s.adsorbate.electronic_energy.axes
    }
    if set(descriptors) != axes:
        raise ValueError("descriptor point must name exactly the original declared axes")
    species = []
    for item in spec.species:
        adsorbate = item.adsorbate
        if adsorbate is not None:
            values = {
                a.descriptor_id: descriptors[a.descriptor_id]
                for a in adsorbate.electronic_energy.axes
            }
            item = replace(item, adsorbate=replace(adsorbate, descriptors=values))
        species.append(item)
    return replace(spec, species=tuple(species))


def solve_catalytic_steady_state(
    spec: CatalyticMechanismSpec,
    conditions: CatalyticAnalysisConditions,
    initial_coverages: Mapping[str, float],
    *,
    target_species: str,
    policy: RootPolicy | None = None,
    derivative_step: float = 6e-6,
    coordinate_order: Sequence[str] | None = None,
    product_species: Sequence[str] | None = None,
    element_basis: str | None = None,
) -> dict[str, Any]:
    """Original residual solver on coverage rates, with explicit numerical Jacobian.

    A root with a singular/marginal mode is reported as such. No conservation
    mode, failed root, invalid BEP barrier or overfilled site is silently repaired.
    """
    policy = RootPolicy() if policy is None else policy
    if (product_species is None) != (element_basis is None):
        raise ValueError("selectivity requires both explicit products and an elemental basis")
    if coordinate_order is not None and not spec.site_species_complete:
        raise ValueError("coordinate_order is only defined for complete surface charts")
    if (
        set(conditions.gas_partial_pressures_Pa) != set(spec.gas_species)
        or set(conditions.capacities_mol) != set(spec.site_types)
        or set(initial_coverages) != set(spec.adsorbates)
        or target_species not in spec.gas_species
    ):
        raise ValueError(
            "analysis conditions must match original gas, sites, adsorbates and target"
        )
    if not spec.minimum_temperature_K <= conditions.temperature_K <= spec.maximum_temperature_K:
        raise ValueError("surface analysis temperature outside declared mechanism domain")
    names = spec.adsorbates
    scales = spec.coverage_amount_scales_mol(conditions.capacities_mol)
    caps = np.array([scales[k] for k in names])
    initial = np.array([initial_coverages[k] for k in names], dtype=float)
    chart = (
        build_catalytic_steady_chart(
            spec, conditions.capacities_mol, initial_coverages, coordinate_order=coordinate_order
        )
        if spec.site_species_complete
        else None
    )
    dimension = len(names) if chart is None else chart.rank
    coordinate_names = names if chart is None else tuple(names[i] for i in chart.coordinate_indices)
    root_initial = initial if chart is None else np.zeros(dimension)
    slot = make_catalytic_slot(
        "steady-analysis",
        "analysis",
        spec,
        capacities_mol=conditions.capacities_mol,
        initial_amounts_mol=dict(zip(names, initial * caps, strict=True)),
    )
    gas = {
        k: p * conditions.volume_L * 1e-3 / (R * conditions.temperature_K)
        for k, p in conditions.gas_partial_pressures_Pa.items()
    }

    def evaluate(coverage: Any) -> Any:
        amounts = dict(zip(names, coverage * caps, strict=True))
        current = replace(slot, values={**slot.values, **amounts})
        state = pack_catalytic_state(
            spec,
            (current,),
            gas,
            temperature_K=conditions.temperature_K,
            volume_L=conditions.volume_L,
        )
        return evaluate_catalytic_rates(state, current, spec)

    def coverage(values: Any) -> Any:
        return values if chart is None else chart.reconstruct(values)

    def full_residual(values: Any) -> Any:
        try:
            rates = evaluate(coverage(values))
            return (
                np.array(
                    [rates.amount_rates_mol_s[catalytic_inventory_key(slot, k)] for k in names]
                )
                / caps
            )
        except (ArithmeticError, ValueError) as exc:
            raise ResidualDomainError(str(exc)) from exc

    def residual(time: float, values: Any, rate: Any, parameters: Any) -> Any:
        full = full_residual(values)
        return full if chart is None else chart.reduce_rates(full)

    # A rank-zero declaration has no reachable dynamics and needs no root solve.
    result: dict[str, Any]
    if dimension == 0:
        result = {
            "model": "declared-catalysis-steady-v1",
            "mechanism_id": spec.mechanism_id,
            "status": "converged",
            "failure_reason": "",
            "conditions": conditions.to_dict(),
            "initial_coverages": dict(initial_coverages),
            "target_species": target_species,
            "root_diagnostic": {"algorithm": "no_reachable_dynamics", "iterations": 0},
            "derivative_diagnostic": {},
        }
        values = root_initial
        jacobian = np.empty((0, 0))
        eigenvalues = np.array([], dtype=complex)
    else:
        system = ResidualSystem(
            "declared-catalytic-steady-surface",
            tuple(
                ResidualVariable(k, "1", 1.0, "algebraic", 0.0, 1.0)
                if chart is None
                else ResidualVariable(k, "1", 1.0, "algebraic")
                for k in coordinate_names
            ),
            tuple(ResidualEquation("surface-rate:" + k, "1/s", 1.0) for k in coordinate_names),
            residual,
            ResidualDerivatives("finite_difference", relative_step=derivative_step),
        )
        solved = solve_steady_residual(system, root_initial, policy=policy)
        result = {
            "model": "declared-catalysis-steady-v1",
            "mechanism_id": spec.mechanism_id,
            "status": "converged" if solved.root.success else "not_converged",
            "failure_reason": solved.root.failure_reason,
            "conditions": conditions.to_dict(),
            "initial_coverages": dict(initial_coverages),
            "target_species": target_species,
            "root_diagnostic": solved.root.diagnostics,
            "derivative_diagnostic": solved.diagnostics,
            "coverages": None,
            "turnover_rate_s_inv": None,
        }
        if chart is not None:
            result["steady_chart"] = chart.to_dict()
        if not solved.root.success:
            return result
        values = solved.root.value
        evaluator = ResidualEvaluator(system)
        try:
            jacobian = evaluator.derivative("state", 0.0, values, np.zeros(dimension))
            eigenvalues = np.linalg.eigvals(jacobian)
        except (ArithmeticError, ValueError) as exc:
            # A valid physical root can lie on a corner without an admissible
            # numerical stencil. Preserve the root; do not invent its spectrum.
            jacobian, eigenvalues = None, None
            result["spectrum_failure_reason"] = str(exc)
    physical = coverage(values)
    rates = evaluate(physical)
    leading = float(np.max(np.real(eigenvalues))) if eigenvalues is not None and dimension else 0.0
    stability = (
        "no_reachable_dynamics"
        if not dimension
        else "stable"
        if leading < -1e-8
        else "unstable"
        if leading > 1e-8
        else "marginal_or_unresolved"
    )
    gross_target = sum(
        abs(reaction.stoichiometry.get(target_species, 0))
        * (
            rates.forward_rates_mol_s[reaction.reaction_id]
            + rates.reverse_rates_mol_s[reaction.reaction_id]
        )
        for reaction in spec.reactions
    ) / sum(conditions.capacities_mol.values())
    result.update(
        {
            "coverages": dict(zip(names, map(float, physical), strict=True)),
            "adsorbate_amounts_mol": dict(zip(names, map(float, physical * caps), strict=True)),
            "gas_amounts_mol": gas,
            "gas_rates_mol_s": {k: rates.amount_rates_mol_s[k] for k in spec.gas_species},
            "forward_rates_mol_s": rates.forward_rates_mol_s,
            "reverse_rates_mol_s": rates.reverse_rates_mol_s,
            "turnover_rate_s_inv": rates.amount_rates_mol_s[target_species]
            / sum(conditions.capacities_mol.values()),
            "turnover_resolution_s_inv": max(policy.tolerance, 1e-10 * gross_target),
            "turnover_normalization": "net target gas production / total declared mol of sites",
            "surface_residual_s_inv": full_residual(values).tolist(),
            "jacobian_s_inv": None if jacobian is None else jacobian.tolist(),
            "stability": stability,
            "eigenvalues_s_inv": [
                {"real": float(z.real), "imag": float(z.imag)} for z in eigenvalues
            ]
            if eigenvalues is not None
            else None,
        }
    )
    if chart is not None:
        result["steady_chart"] = chart.to_dict()
        result["conserved_totals_at_root_mol"] = chart.totals(result["coverages"]).tolist()
    if product_species is not None and element_basis is not None:
        result["selectivity"] = catalytic_product_selectivity(
            spec,
            result["gas_rates_mol_s"],
            product_species=product_species,
            element_basis=element_basis,
        )
    return result


def catalytic_steady_states(
    spec: CatalyticMechanismSpec,
    conditions: CatalyticAnalysisConditions,
    seeds: Sequence[Mapping[str, float]],
    *,
    target_species: str,
    policy: RootPolicy | None = None,
    product_species: Sequence[str] | None = None,
    element_basis: str | None = None,
) -> dict[str, Any]:
    """Retain every declared seed attempt and deduplicate converged roots only."""
    if not seeds:
        raise ValueError("root enumeration requires explicit finite seed coverage")
    if spec.site_species_complete:
        chart = build_catalytic_steady_chart(spec, conditions.capacities_mol, seeds[0])
        if any(not chart.same_class(seed) for seed in seeds[1:]):
            raise ValueError("multiseed analysis requires one full conserved compatibility class")
    attempts: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    for seed in seeds:
        try:
            result = solve_catalytic_steady_state(
                spec,
                conditions,
                seed,
                target_species=target_species,
                policy=policy,
                product_species=product_species,
                element_basis=element_basis,
            )
        except (ValueError, ArithmeticError) as exc:
            result = {
                "status": "rejected",
                "initial_coverages": dict(seed),
                "failure_reason": str(exc),
                "coverages": None,
            }
        attempts.append(result)
        if result["status"] == "converged" and not any(
            max(abs(result["coverages"][k] - r["coverages"][k]) for k in spec.adsorbates) < 1e-7
            for r in roots
        ):
            roots.append(result)
    roots.sort(key=lambda r: tuple(r["coverages"][k] for k in spec.adsorbates))
    return {
        "attempts": attempts,
        "roots": roots,
        "seed_count": len(seeds),
        "converged_attempts": sum(a["status"] == "converged" for a in attempts),
        "distinct_roots": len(roots),
        "scope": "declared seeds; not proof of all roots",
    }


def catalytic_product_selectivity(
    spec: CatalyticMechanismSpec,
    gas_rates_mol_s: Mapping[str, float],
    *,
    product_species: Sequence[str],
    element_basis: str,
    rate_tolerance_mol_s: float = 1e-14,
) -> dict[str, Any]:
    """Signed local net-product selectivity on an explicitly declared atom basis.

    This is neither finite-reactor conversion nor yield. Missing significant
    products, reversed declared products and unresolved denominators are visible.
    """
    products = tuple(product_species)
    if (
        not products
        or len(products) != len(set(products))
        or set(products) - set(spec.gas_species)
        or set(gas_rates_mol_s) != set(spec.gas_species)
        or not isfinite(rate_tolerance_mol_s)
        or rate_tolerance_mol_s <= 0
        or any(not isfinite(v) for v in gas_rates_mol_s.values())
    ):
        raise ValueError(
            "selectivity requires explicit unique gas products and finite signed rates"
        )
    weights = {
        k: spec.species_by_id[k].species.composition.get(element_basis, 0) for k in spec.gas_species
    }
    if any(weights[k] <= 0 for k in products):
        raise ValueError("every declared product must contain the explicit elemental basis")
    weighted = {k: weights[k] * gas_rates_mol_s[k] for k in spec.gas_species}
    denominator = sum(weighted[k] for k in products)
    reversed_products = [k for k in products if gas_rates_mol_s[k] < -rate_tolerance_mol_s]
    omitted = [
        k
        for k in spec.gas_species
        if k not in products and weights[k] > 0 and gas_rates_mol_s[k] > rate_tolerance_mol_s
    ]
    unresolved = denominator <= rate_tolerance_mol_s * sum(weights[k] for k in products)
    status = (
        "unavailable"
        if reversed_products or unresolved
        else "incomplete"
        if omitted
        else "available"
    )
    return {
        "status": status,
        "product_species": list(products),
        "element_basis": element_basis,
        "element_counts": {k: weights[k] for k in products},
        "signed_gas_rates_mol_s": dict(gas_rates_mol_s),
        "signed_element_rates_mol_s": weighted,
        "denominator_element_mol_s": denominator,
        "fractions": {k: weighted[k] / denominator for k in products}
        if status == "available"
        else None,
        "reversed_products": reversed_products,
        "omitted_positive_products": omitted,
        "denominator_unresolved": unresolved,
        "rate_tolerance_mol_s": rate_tolerance_mol_s,
        "scope": "held-pressure local net production selectivity; not reactor conversion or yield",
    }


def map_catalytic_conditions(
    spec: CatalyticMechanismSpec,
    conditions: Sequence[CatalyticAnalysisConditions],
    seeds: Sequence[Mapping[str, float]],
    *,
    target_species: str,
    product_species: Sequence[str] | None = None,
    element_basis: str | None = None,
    policy: RootPolicy | None = None,
) -> dict[str, Any]:
    """Explicit T/pressure points with retained failures and successful warm starts."""
    if not conditions or not seeds:
        raise ValueError("condition map requires explicit boundaries and finite seeds")
    rows: list[dict[str, Any]] = []
    warm = None
    for index, boundary in enumerate(conditions):
        try:
            initials = [*seeds]
            if warm is not None:
                # Across different capacities a previous coverage may represent
                # a different molar class; only reuse compatible warm starts.
                chart = (
                    build_catalytic_steady_chart(spec, boundary.capacities_mol, seeds[0])
                    if spec.site_species_complete
                    else None
                )
                if chart is None or chart.same_class(warm):
                    initials.append(warm)
            solved = catalytic_steady_states(
                spec,
                boundary,
                initials,
                target_species=target_species,
                policy=policy,
                product_species=product_species,
                element_basis=element_basis,
            )
            row = {
                "index": index,
                "conditions": boundary.to_dict(),
                **solved,
                "status": "converged" if solved["roots"] else "not_converged",
            }
            if solved["roots"]:
                warm = solved["roots"][0]["coverages"]
        except (ValueError, ArithmeticError) as exc:
            row = {
                "index": index,
                "conditions": boundary.to_dict(),
                "status": "rejected",
                "failure_reason": str(exc),
                "attempts": [],
                "roots": [],
            }
        rows.append(row)
    return {
        "model": "declared-catalysis-condition-map-v1",
        "points": rows,
        "points_total": len(rows),
        "points_converged": sum(r["status"] == "converged" for r in rows),
        "continuation": "successive-condition warm starts; no pseudo-arclength or all-root proof",
    }


def map_catalytic_descriptors(
    spec: CatalyticMechanismSpec,
    conditions: CatalyticAnalysisConditions,
    points: Sequence[Mapping[str, float]],
    seeds: Sequence[Mapping[str, float]],
    *,
    target_species: str,
    policy: RootPolicy | None = None,
) -> dict[str, Any]:
    """Solve every requested descriptor point; warm starts never fill missing roots."""
    if not points or not seeds:
        raise ValueError("descriptor map requires explicit points and initial seeds")
    rows, warm = [], None
    for index, point in enumerate(points):
        try:
            mechanism = with_catalytic_descriptors(spec, point)
            initials = [*seeds, *([warm] if warm is not None else [])]
            solved = catalytic_steady_states(
                mechanism, conditions, initials, target_species=target_species, policy=policy
            )
            row = {
                "index": index,
                "descriptors_J_mol": dict(point),
                **solved,
                "status": "converged" if solved["roots"] else "not_converged",
            }
            if solved["roots"]:
                warm = solved["roots"][0]["coverages"]
        except (ValueError, ArithmeticError) as exc:
            row = {
                "index": index,
                "descriptors_J_mol": dict(point),
                "status": "rejected",
                "failure_reason": str(exc),
                "attempts": [],
                "roots": [],
            }
        rows.append(row)
    return {
        "model": "declared-catalysis-descriptor-map-v1",
        "points": rows,
        "points_total": len(rows),
        "points_converged": sum(r["status"] == "converged" for r in rows),
        "continuation": "successive-parameter warm starts; no pseudo-arclength or hole filling",
    }


def catalytic_rate_control(
    spec: CatalyticMechanismSpec,
    conditions: CatalyticAnalysisConditions,
    initial_coverages: Mapping[str, float],
    *,
    target_species: str,
    barrier_step_J_mol: float = 1.0,
    log_pressure_step: float = 1e-4,
    temperature_step_K: float = 0.02,
    policy: RootPolicy | None = None,
) -> dict[str, Any]:
    """DRC, gas order and apparent Ea with explicit symmetric numerical stencils."""
    if any(
        not isfinite(v) or v <= 0
        for v in (barrier_step_J_mol, log_pressure_step, temperature_step_K)
    ):
        raise ValueError("rate control requires finite positive declared stencil sizes")
    baseline = solve_catalytic_steady_state(
        spec, conditions, initial_coverages, target_species=target_species, policy=policy
    )
    result: dict[str, Any] = {
        "baseline": baseline,
        "status": "unavailable",
        "derivative_method": "centered re-solved stencils; equilibrium constants unchanged",
        "stencils": {
            "barrier_step_J_mol": barrier_step_J_mol,
            "log_pressure_step": log_pressure_step,
            "temperature_step_K": temperature_step_K,
        },
        "rate_control": {},
        "reaction_orders": {},
        "apparent_activation_energy": {},
    }
    if (
        baseline["status"] != "converged"
        or baseline["turnover_rate_s_inv"] <= baseline["turnover_resolution_s_inv"]
    ):
        result["failure_reason"] = "target requires a resolved positive net production rate"
        return result
    seed = baseline["coverages"]

    def stencil(
        cases: Sequence[tuple[CatalyticMechanismSpec, CatalyticAnalysisConditions]], scale: float
    ) -> dict[str, Any]:
        solutions = []
        for mechanism, boundary in cases:
            try:
                value = solve_catalytic_steady_state(
                    mechanism, boundary, seed, target_species=target_species, policy=policy
                )
            except (ValueError, ArithmeticError) as exc:
                value = {"status": "rejected", "failure_reason": str(exc)}
            solutions.append(value)
        if any(
            s["status"] != "converged" or s["turnover_rate_s_inv"] <= s["turnover_resolution_s_inv"]
            for s in solutions
        ):
            return {
                "status": "unavailable",
                "value": None,
                "solutions": solutions,
                "failure_reason": "both centered physical roots must have positive target rate",
            }
        return {
            "status": "available",
            "value": scale
            * (log(solutions[1]["turnover_rate_s_inv"]) - log(solutions[0]["turnover_rate_s_inv"])),
            "solutions": solutions,
        }

    for index, reaction in enumerate(spec.reactions):
        if not isinstance(reaction, CatalyticReactionSpec):
            raise ValueError(
                "barrier control currently requires declared free-energy BEP reactions"
            )
        try:
            models = []
            for sign in (-1, 1):
                reactions = list(spec.reactions)
                reactions[index] = replace(
                    reaction,
                    bep=replace(
                        reaction.bep,
                        intercept_J_mol=reaction.bep.intercept_J_mol + sign * barrier_step_J_mol,
                    ),
                )
                models.append((replace(spec, reactions=tuple(reactions)), conditions))
            result["rate_control"][reaction.reaction_id] = stencil(
                models, -R * conditions.temperature_K / (2 * barrier_step_J_mol)
            )
        except ValueError as exc:
            result["rate_control"][reaction.reaction_id] = {
                "status": "unavailable",
                "value": None,
                "failure_reason": str(exc),
                "solutions": [],
            }
    for gas, pressure in conditions.gas_partial_pressures_Pa.items():
        if pressure == 0:
            result["reaction_orders"][gas] = {
                "status": "unavailable",
                "value": None,
                "failure_reason": "log pressure undefined at zero",
            }
        else:
            cases = [
                (
                    spec,
                    replace(
                        conditions,
                        gas_partial_pressures_Pa={
                            **conditions.gas_partial_pressures_Pa,
                            gas: pressure * exp(sign * log_pressure_step),
                        },
                    ),
                )
                for sign in (-1, 1)
            ]
            result["reaction_orders"][gas] = stencil(cases, 1 / (2 * log_pressure_step))
    result["apparent_activation_energy"] = {
        **stencil(
            [
                (
                    spec,
                    replace(
                        conditions,
                        temperature_K=conditions.temperature_K + sign * temperature_step_K,
                    ),
                )
                for sign in (-1, 1)
            ],
            R * conditions.temperature_K**2 / (2 * temperature_step_K),
        ),
        "unit": "J/mol",
        "boundary": "fixed gas partial pressures",
    }
    parts = [
        *result["rate_control"].values(),
        *result["reaction_orders"].values(),
        result["apparent_activation_energy"],
    ]
    result["status"] = "available" if all(p["status"] == "available" for p in parts) else "partial"
    return result
