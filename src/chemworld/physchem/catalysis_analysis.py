"""Steady surface/control queries over the original declared catalytic mechanism.

Gas pressures are an explicit analysis boundary. No gas reservoir, world state,
or alternative rate evaluator is persisted or simulated by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import exp, isclose, isfinite, log, log10
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.special import softmax

from chemworld.physchem.catalysis_network import (
    CatalyticMechanismSpec,
    CatalyticReactionSpec,
    catalytic_inventory_key,
    evaluate_catalytic_rates,
    make_catalytic_slot,
    pack_catalytic_state,
)
from chemworld.physchem.catalysis_reservoirs import CatalyticReservoirBoundary
from chemworld.physchem.catalysis_steady_chart import build_catalytic_steady_chart
from chemworld.physchem.chemostatted_catalysis import (
    chemostatted_readout,
    faradaic_current,
    open_catalytic_rates,
    pack_chemostatted_state,
    validate_boundary,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualDomainError,
    ResidualEquation,
    ResidualEvaluator,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_steady_residual,
)
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode


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


@dataclass(frozen=True)
class ChemostattedAnalysisConditions:
    """A held-gas, held-aqueous, fixed-SHE boundary for open catalysis.

    The boundary is deliberately explicit: the liquid activities and pH come
    from ``CatalyticReservoirBoundary`` and are never inferred from a gas or a
    hidden charge reservoir.  This is an analysis boundary, not a new runtime
    state representation.
    """

    boundary: CatalyticReservoirBoundary
    potential_V: float
    gas_partial_pressures_Pa: Mapping[str, float]
    capacities_mol: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "gas_partial_pressures_Pa", MappingProxyType(dict(self.gas_partial_pressures_Pa))
        )
        object.__setattr__(self, "capacities_mol", MappingProxyType(dict(self.capacities_mol)))
        if isinstance(self.potential_V, bool) or not isfinite(self.potential_V):
            raise ValueError("electrochemical analysis requires a finite potential")
        if not self.gas_partial_pressures_Pa or any(
            isinstance(v, bool) or not isfinite(v) or v < 0
            for v in self.gas_partial_pressures_Pa.values()
        ):
            raise ValueError("electrochemical analysis requires finite nonnegative gas pressures")
        if not self.capacities_mol or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0 for v in self.capacities_mol.values()
        ):
            raise ValueError("electrochemical analysis requires positive finite capacities")

    @property
    def temperature_K(self) -> float:  # noqa: N802 - public unit-explicit contract
        return self.boundary.bath.setpoint_K

    @property
    def pH(self) -> float | None:  # noqa: N802 - public chemistry notation
        species_id = self.boundary.proton_species_id
        if species_id is None:
            return None
        return -log10(self.boundary.activities[species_id])

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature_K": self.temperature_K,
            "potential_V": self.potential_V,
            "potential_reference": self.boundary.potential_reference,
            "pH": self.pH,
            "reservoir_activities": self.boundary.activities,
            "gas_partial_pressures_Pa": dict(self.gas_partial_pressures_Pa),
            "capacities_mol": dict(self.capacities_mol),
            "boundary": (
                "held-gas-and-real-aqueous-chemostats; fixed-SHE potential; "
                "surface-only steady analysis"
            ),
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
    relaxation_time_s: float = 1e4,
    relaxation_maximum_evaluations: int = 100000,
) -> dict[str, Any]:
    """Original rates and residual solver in the declared compatibility class.

    A root with a singular/marginal mode is reported as such. No conservation
    mode, failed root, invalid BEP barrier or overfilled site is silently repaired.
    Positive connected single-site pools use log-ratios and bounded numerical
    BDF relaxation before root polishing. Set ``relaxation_time_s=0`` for a
    direct local root. Relaxation owns no world time or experimental resources;
    its rate-evaluation budget is additional to the supplied root policy.
    Reachable eigenvalues always use physical coverage coordinates.
    """
    policy = RootPolicy() if policy is None else policy
    if (
        not isfinite(relaxation_time_s)
        or relaxation_time_s < 0
        or type(relaxation_maximum_evaluations) is not int
        or relaxation_maximum_evaluations < 1
    ):
        raise ValueError(
            "steady relaxation requires a finite horizon and positive evaluation budget"
        )
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
    positive_coordinates = (
        chart is not None
        and chart.rank == len(names) - 1
        and len(spec.site_types) == 1
        and np.all(initial > 0)
    )
    if positive_coordinates and coordinate_order is None:
        reference = names[int(np.argmax(initial))]
        chart = build_catalytic_steady_chart(
            spec,
            conditions.capacities_mol,
            initial_coverages,
            coordinate_order=(*(k for k in names if k != reference), reference),
        )
    dimension = len(names) if chart is None else chart.rank
    coordinate_names = names if chart is None else tuple(names[i] for i in chart.coordinate_indices)
    root_initial = initial if chart is None else np.zeros(dimension)
    if positive_coordinates and chart is not None:
        dependent = next(i for i in range(len(names)) if i not in chart.coordinate_indices)
        root_initial = np.log(initial[list(chart.coordinate_indices)] / initial[dependent])
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
        if positive_coordinates and chart is not None:
            logits = np.zeros(len(names))
            logits[list(chart.coordinate_indices)] = values
            return initial.sum() * softmax(logits)
        return values if chart is None else chart.reconstruct(values)

    def physical_residual(physical: Any) -> Any:
        try:
            rates = evaluate(physical)
            return (
                np.array(
                    [rates.amount_rates_mol_s[catalytic_inventory_key(slot, k)] for k in names]
                )
                / caps
            )
        except (ArithmeticError, ValueError) as exc:
            raise ResidualDomainError(str(exc)) from exc

    def full_residual(values: Any) -> Any:
        return physical_residual(coverage(values))

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
        initialization: dict[str, Any] = {"method": "direct_root", "rate_evaluations": 0}
        candidate = root_initial
        if positive_coordinates and chart is not None and relaxation_time_s > 0:
            evaluations = 0
            indices = list(chart.coordinate_indices)

            def relaxation_rates(q: Any) -> Any:
                nonlocal evaluations
                if evaluations >= relaxation_maximum_evaluations:
                    raise ResidualDomainError("steady_relaxation_evaluation_budget_exhausted")
                evaluations += 1
                return full_residual(q)

            def relaxation_rhs(time: float, q: Any) -> Any:
                theta = coverage(q)
                if np.any(theta <= 0):
                    raise ResidualDomainError("positive coverage chart underflow")
                logarithmic_rates = relaxation_rates(q) / theta
                return logarithmic_rates[indices] - logarithmic_rates[dependent]

            threshold = max(1e-6, 10 * policy.tolerance)

            def ready(time: float, q: Any) -> float:
                return float(np.max(np.abs(relaxation_rates(q)))) - threshold

            ready.terminal = True  # type: ignore[attr-defined]
            ready.direction = -1  # type: ignore[attr-defined]
            initialization = {
                "method": "shared_BDF_positive_log_coordinates",
                "numerical_horizon_s": relaxation_time_s,
                "maximum_rate_evaluations": relaxation_maximum_evaluations,
                "physical_time_advanced_s": 0.0,
                "experiment_resource_debit": None,
            }
            try:
                if ready(0.0, candidate) > 0:
                    first_step = min(
                        relaxation_time_s,
                        0.01 / max(1.0, float(np.max(np.abs(relaxation_rhs(0.0, candidate))))),
                    )
                    relaxed = solve_ode(
                        relaxation_rhs,
                        candidate,
                        time_span_s=(0.0, relaxation_time_s),
                        events=(ready,),
                        policy=ODESolverPolicy(
                            "catalytic-steady-initialization-v1",
                            method="BDF",
                            rtol=1e-9,
                            atol=1e-10,
                            first_step_s=first_step,
                            event_names=("near_steady_initialization",),
                        ),
                    )
                    initialization["solver"] = relaxed.diagnostic.to_dict()
                    relaxed.raise_for_failure("catalytic steady initialization")
                    candidate = relaxed.raw_result.y[:, -1]
            except (ValueError, ArithmeticError, RuntimeError) as exc:
                initialization.update(
                    success=False, failure_reason=str(exc), rate_evaluations=evaluations
                )
                return {
                    "model": "declared-catalysis-steady-v1",
                    "mechanism_id": spec.mechanism_id,
                    "status": "not_converged",
                    "failure_reason": str(exc),
                    "conditions": conditions.to_dict(),
                    "initial_coverages": dict(initial_coverages),
                    "target_species": target_species,
                    "coverages": None,
                    "turnover_rate_s_inv": None,
                    "steady_chart": chart.to_dict(),
                    "steady_initialization": initialization,
                    "root_coordinates": "positive_log_ratios",
                    "root_iterations_total": 0,
                }
            initialization.update(success=True, rate_evaluations=evaluations)
        solved = solve_steady_residual(system, candidate, policy=policy)
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
            "root_coordinates": "positive_log_ratios"
            if positive_coordinates
            else "affine_coverages",
            "steady_initialization": initialization,
            "root_iterations_total": solved.root.diagnostics["iterations"],
        }
        if chart is not None:
            result["steady_chart"] = chart.to_dict()
        if not solved.root.success:
            return result
        values = solved.root.value
        evaluator = ResidualEvaluator(system)
        try:
            if positive_coordinates:
                physical = coverage(values)
                dependent_species = names[int(np.argmax(physical))]
                spectrum_chart = build_catalytic_steady_chart(
                    spec,
                    conditions.capacities_mol,
                    dict(zip(names, physical, strict=True)),
                    coordinate_order=(
                        *(k for k in names if k != dependent_species),
                        dependent_species,
                    ),
                )
                jacobian = finite_jacobian(
                    lambda q: spectrum_chart.reduce_rates(
                        physical_residual(spectrum_chart.reconstruct(q))
                    ),
                    np.zeros(dimension),
                    np.ones(dimension),
                    relative_step=derivative_step,
                )
                result["spectrum_chart"] = spectrum_chart.to_dict()
            else:
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


def _validate_chemostatted_conditions(
    spec: CatalyticMechanismSpec, conditions: ChemostattedAnalysisConditions
) -> None:
    validate_boundary(spec, conditions.boundary, conditions.potential_V)
    if set(conditions.gas_partial_pressures_Pa) != set(spec.gas_species) or set(
        conditions.capacities_mol
    ) != set(spec.site_types):
        raise ValueError("electrochemical conditions must match original gas and site identities")


def solve_chemostatted_catalytic_steady_state(
    spec: CatalyticMechanismSpec,
    conditions: ChemostattedAnalysisConditions,
    initial_coverages: Mapping[str, float],
    *,
    target_species: str,
    policy: RootPolicy | None = None,
    derivative_step: float = 6e-6,
    coordinate_order: Sequence[str] | None = None,
    product_species: Sequence[str] | None = None,
    element_basis: str | None = None,
) -> dict[str, Any]:
    """Solve an open catalytic steady state using the original rate evaluator.

    Gas pressures, aqueous activities/pH and SHE potential are held boundary
    values.  Only catalyst inventories are solved, so the result can be
    compared directly with the closed ``solve_catalytic_steady_state`` output.
    """
    policy = RootPolicy() if policy is None else policy
    if (product_species is None) != (element_basis is None):
        raise ValueError("selectivity requires both explicit products and an elemental basis")
    if coordinate_order is not None and not spec.site_species_complete:
        raise ValueError("coordinate_order is only defined for complete surface charts")
    if set(initial_coverages) != set(spec.adsorbates) or target_species not in spec.gas_species:
        raise ValueError(
            "analysis conditions must match original gas, sites, adsorbates and target"
        )
    _validate_chemostatted_conditions(spec, conditions)
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
        k: p * conditions.boundary.gas_volume_L * 1e-3 / (R * conditions.temperature_K)
        for k, p in conditions.gas_partial_pressures_Pa.items()
    }

    def evaluate(coverage: Any) -> Any:
        amounts = dict(zip(names, coverage * caps, strict=True))
        current = replace(slot, values={**slot.values, **amounts})
        state = pack_chemostatted_state(spec, current, conditions.boundary, gas)
        return open_catalytic_rates(
            spec, current, conditions.boundary, conditions.potential_V, state, solver_trial=True
        )

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

    result: dict[str, Any]
    if dimension == 0:
        result = {
            "model": "declared-chemostatted-catalysis-steady-v1",
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
            "declared-chemostatted-catalytic-steady-surface",
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
            "model": "declared-chemostatted-catalysis-steady-v1",
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
            jacobian, eigenvalues = None, None
            result["spectrum_failure_reason"] = str(exc)

    physical = coverage(values)
    amounts = dict(zip(names, physical * caps, strict=True))
    current_slot = replace(slot, values={**slot.values, **amounts})
    state = pack_chemostatted_state(spec, current_slot, conditions.boundary, gas)
    rates = open_catalytic_rates(
        spec,
        current_slot,
        conditions.boundary,
        conditions.potential_V,
        state,
        solver_trial=True,
    )
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
    electrical_readout: dict[str, Any] = {}
    try:
        readout = chemostatted_readout(
            spec, current_slot, conditions.boundary, conditions.potential_V, state
        )
        electrical_readout = {
            "terminal_potential_V": readout.terminal_potential_V,
            "applied_current_A": readout.applied_current_A,
            "standard_delta_g_J_mol": readout.standard_delta_g_J_mol,
            "affinities_J_mol": readout.affinities_J_mol,
        }
    except (ArithmeticError, ValueError) as exc:
        electrical_readout = {"failure_reason": str(exc)}
    result.update(
        {
            "coverages": dict(zip(names, map(float, physical), strict=True)),
            "adsorbate_amounts_mol": dict(zip(names, map(float, physical * caps), strict=True)),
            "gas_amounts_mol": gas,
            "gas_rates_mol_s": {k: rates.amount_rates_mol_s[k] for k in spec.gas_species},
            "liquid_rates_mol_s": {
                k: rates.amount_rates_mol_s[k] for k in conditions.boundary.solutes
            },
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
            "faradaic_current_A": faradaic_current(spec, rates),
            "electrical_work_rate_W": conditions.potential_V * faradaic_current(spec, rates),
            "electrical_readout": electrical_readout,
            "scope": (
                "held gas/aquatic chemostats and fixed SHE potential; declared mechanism; "
                "no pseudo-arclength, all-root proof or CatMAP qualification"
            ),
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


def map_chemostatted_catalytic_conditions(
    spec: CatalyticMechanismSpec,
    conditions: Sequence[ChemostattedAnalysisConditions],
    seeds: Sequence[Mapping[str, float]],
    *,
    target_species: str,
    product_species: Sequence[str] | None = None,
    element_basis: str | None = None,
    policy: RootPolicy | None = None,
) -> dict[str, Any]:
    """Map T/pressure/potential/pH points while retaining every failure."""
    if not conditions or not seeds:
        raise ValueError(
            "electrochemical condition map requires explicit boundaries and finite seeds"
        )
    rows: list[dict[str, Any]] = []
    warm = None
    for index, boundary in enumerate(conditions):
        try:
            initials = [*seeds]
            if warm is not None:
                chart = (
                    build_catalytic_steady_chart(spec, boundary.capacities_mol, seeds[0])
                    if spec.site_species_complete
                    else None
                )
                if chart is None or chart.same_class(warm):
                    initials.append(warm)
            attempts = []
            for seed in initials:
                try:
                    attempts.append(
                        solve_chemostatted_catalytic_steady_state(
                            spec,
                            boundary,
                            seed,
                            target_species=target_species,
                            policy=policy,
                            product_species=product_species,
                            element_basis=element_basis,
                        )
                    )
                except (ValueError, ArithmeticError) as exc:
                    attempts.append(
                        {
                            "status": "rejected",
                            "initial_coverages": dict(seed),
                            "failure_reason": str(exc),
                            "coverages": None,
                        }
                    )
            roots: list[dict[str, Any]] = []
            for result in attempts:
                if result["status"] == "converged" and not any(
                    max(abs(result["coverages"][k] - root["coverages"][k]) for k in spec.adsorbates)
                    < 1e-7
                    for root in roots
                ):
                    roots.append(result)
            row = {
                "index": index,
                "conditions": boundary.to_dict(),
                "attempts": attempts,
                "roots": roots,
                "seed_count": len(initials),
                "converged_attempts": sum(a["status"] == "converged" for a in attempts),
                "distinct_roots": len(roots),
                "status": (
                    "converged"
                    if roots
                    else "rejected"
                    if attempts and all(a["status"] == "rejected" for a in attempts)
                    else "not_converged"
                ),
                "scope": "declared seeds; not proof of all roots",
            }
            if roots:
                warm = roots[0]["coverages"]
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
        "model": "declared-chemostatted-catalytic-condition-map-v1",
        "points": rows,
        "points_total": len(rows),
        "points_converged": sum(r["status"] == "converged" for r in rows),
        "continuation": (
            "successive T/pressure/potential/pH warm starts; explicit failures retained; "
            "no pseudo-arclength or hole filling"
        ),
    }


def chemostatted_rate_control(
    spec: CatalyticMechanismSpec,
    conditions: ChemostattedAnalysisConditions,
    initial_coverages: Mapping[str, float],
    *,
    target_species: str,
    barrier_step_J_mol: float = 1.0,
    log_pressure_step: float = 1e-4,
    potential_step_V: float = 1e-4,
    temperature_step_K: float = 0.02,
    pH_pair: tuple[ChemostattedAnalysisConditions, ChemostattedAnalysisConditions] | None = None,
    product_species: Sequence[str] | None = None,
    element_basis: str | None = None,
    policy: RootPolicy | None = None,
) -> dict[str, Any]:
    """Electrochemical rate/selectivity control with explicit centered stencils."""
    if any(
        not isfinite(v) or v <= 0
        for v in (barrier_step_J_mol, log_pressure_step, potential_step_V, temperature_step_K)
    ):
        raise ValueError("electrochemical rate control requires finite positive stencil sizes")
    baseline = solve_chemostatted_catalytic_steady_state(
        spec,
        conditions,
        initial_coverages,
        target_species=target_species,
        policy=policy,
        product_species=product_species,
        element_basis=element_basis,
    )
    result: dict[str, Any] = {
        "baseline": baseline,
        "status": "unavailable",
        "derivative_method": "centered re-solved stencils; equilibrium constants unchanged",
        "stencils": {
            "barrier_step_J_mol": barrier_step_J_mol,
            "log_pressure_step": log_pressure_step,
            "potential_step_V": potential_step_V,
            "temperature_step_K": temperature_step_K,
        },
        "rate_control": {},
        "reaction_orders": {},
        "apparent_activation_energy": {},
        "potential_sensitivity": {},
        "pH_sensitivity": {},
        "selectivity_control": {},
    }
    if (
        baseline["status"] != "converged"
        or baseline["turnover_rate_s_inv"] <= baseline["turnover_resolution_s_inv"]
    ):
        result["failure_reason"] = "target requires a resolved positive net production rate"
        return result
    seed = baseline["coverages"]

    def solve_case(
        mechanism: CatalyticMechanismSpec, boundary: ChemostattedAnalysisConditions
    ) -> dict[str, Any]:
        try:
            return solve_chemostatted_catalytic_steady_state(
                mechanism,
                boundary,
                seed,
                target_species=target_species,
                policy=policy,
                product_species=product_species,
                element_basis=element_basis,
            )
        except (ValueError, ArithmeticError) as exc:
            return {"status": "rejected", "failure_reason": str(exc)}

    def stencil(
        cases: Sequence[tuple[CatalyticMechanismSpec, ChemostattedAnalysisConditions]], scale: float
    ) -> dict[str, Any]:
        solutions = [solve_case(mechanism, boundary) for mechanism, boundary in cases]
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

    def selectivity_stencil(
        cases: Sequence[tuple[CatalyticMechanismSpec, ChemostattedAnalysisConditions]],
        denominator: float,
    ) -> dict[str, Any]:
        if product_species is None or element_basis is None:
            return {
                "status": "unavailable",
                "value": None,
                "failure_reason": (
                    "selectivity control requires explicit products and an elemental basis"
                ),
            }
        solutions = [solve_case(mechanism, boundary) for mechanism, boundary in cases]
        if any(
            s["status"] != "converged" or s.get("selectivity", {}).get("status") != "available"
            for s in solutions
        ):
            return {"status": "unavailable", "value": None, "solutions": solutions}
        values = {
            product: (
                solutions[1]["selectivity"]["fractions"][product]
                - solutions[0]["selectivity"]["fractions"][product]
            )
            / denominator
            for product in product_species
        }
        return {"status": "available", "value": values, "solutions": solutions}

    for index, reaction in enumerate(spec.reactions):
        if not isinstance(reaction, CatalyticReactionSpec):
            result["rate_control"][reaction.reaction_id] = {
                "status": "unavailable",
                "value": None,
                "failure_reason": "barrier control requires declared free-energy BEP reactions",
                "solutions": [],
            }
            continue
        reactions = []
        for sign in (-1, 1):
            changed = list(spec.reactions)
            changed[index] = replace(
                reaction,
                bep=replace(
                    reaction.bep,
                    intercept_J_mol=reaction.bep.intercept_J_mol + sign * barrier_step_J_mol,
                ),
            )
            reactions.append((replace(spec, reactions=tuple(changed)), conditions))
        result["rate_control"][reaction.reaction_id] = stencil(
            reactions, -R * conditions.temperature_K / (2 * barrier_step_J_mol)
        )

    for gas, pressure in conditions.gas_partial_pressures_Pa.items():
        if pressure == 0:
            result["reaction_orders"][gas] = {
                "status": "unavailable",
                "value": None,
                "failure_reason": "log pressure undefined at zero",
            }
            continue
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
                        boundary=replace(
                            conditions.boundary,
                            bath=replace(
                                conditions.boundary.bath,
                                setpoint_K=conditions.temperature_K + sign * temperature_step_K,
                            ),
                        ),
                    ),
                )
                for sign in (-1, 1)
            ],
            R * conditions.temperature_K**2 / (2 * temperature_step_K),
        ),
        "unit": "J/mol",
        "boundary": "fixed gas partial pressures, aqueous activities and SHE potential",
    }
    potential_cases = [
        (spec, replace(conditions, potential_V=conditions.potential_V + sign * potential_step_V))
        for sign in (-1, 1)
    ]
    result["potential_sensitivity"] = {
        **stencil(potential_cases, 1 / (2 * potential_step_V)),
        "unit": "V^-1",
        "boundary": "fixed gas partial pressures and explicit aqueous activities",
    }
    pH_low: float | None = None
    pH_high: float | None = None
    if pH_pair is None:
        result["pH_sensitivity"] = {
            "status": "unavailable",
            "value": None,
            "failure_reason": "pH sensitivity requires an explicit electroneutral boundary pair",
            "unit": "pH^-1",
        }
    else:
        low, high = pH_pair
        pH_low, pH_high = low.pH, high.pH
        same_non_pH_boundary = (
            low.gas_partial_pressures_Pa == high.gas_partial_pressures_Pa
            and low.capacities_mol == high.capacities_mol
            and low.potential_V == high.potential_V
            and low.temperature_K == high.temperature_K
        )
        if not same_non_pH_boundary:
            result["pH_sensitivity"] = {
                "status": "unavailable",
                "value": None,
                "failure_reason": (
                    "pH pair must hold gas, capacity, temperature and potential fixed"
                ),
                "unit": "pH^-1",
            }
        elif (
            pH_low is None or pH_high is None or isclose(pH_low, pH_high, rel_tol=0, abs_tol=1e-15)
        ):
            result["pH_sensitivity"] = {
                "status": "unavailable",
                "value": None,
                "failure_reason": "pH sensitivity requires two distinct proton activities",
                "unit": "pH^-1",
            }
        else:
            if pH_low > pH_high:
                low, high = high, low
                pH_low, pH_high = pH_high, pH_low
            result["pH_sensitivity"] = {
                **stencil([(spec, low), (spec, high)], 1 / (pH_high - pH_low)),
                "unit": "pH^-1",
                "boundary": "explicit electroneutral boundary pair",
            }

    if product_species is not None and element_basis is not None:
        pH_selectivity: dict[str, Any] = {"status": "unavailable", "value": None}
        if result["pH_sensitivity"]["status"] == "available":
            assert pH_low is not None and pH_high is not None
            pH_selectivity = selectivity_stencil([(spec, low), (spec, high)], pH_high - pH_low)
        result["selectivity_control"] = {
            "potential_V": selectivity_stencil(potential_cases, 2 * potential_step_V),
            "pH": pH_selectivity,
        }
    parts = [
        *result["rate_control"].values(),
        *result["reaction_orders"].values(),
        result["apparent_activation_energy"],
        result["potential_sensitivity"],
        result["pH_sensitivity"],
    ]
    if product_species is not None and element_basis is not None:
        parts.extend(result["selectivity_control"].values())
    result["status"] = "available" if all(p["status"] == "available" for p in parts) else "partial"
    return result
