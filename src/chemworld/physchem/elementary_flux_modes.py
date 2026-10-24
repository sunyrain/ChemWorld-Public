"""Exact, explicitly bounded support-minimal modes of original reaction fluxes.

Modes are not an arbitrary kernel basis or a list of cone extreme rays. Original
reversible coordinates stay signed, so cancelling copies of one reaction never
become invented pathways. This analysis owns no live physical state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from itertools import combinations
from math import comb, fsum, gcd, isfinite, lcm
from numbers import Integral, Real
from typing import Any, cast

from chemworld.physchem.biochemical_analysis import _check_scope
from chemworld.physchem.sbml_kinetics import SBMLKinetics
from chemworld.physchem.stoichiometry import (
    RationalEliminationLimitError,
    _check_rational_bits,
    _rref,
)


@dataclass(frozen=True)
class FluxModePolicy:
    max_support_evaluations: int = 10000
    max_integer_bits: int = 8192

    def __post_init__(self) -> None:
        if type(self.max_support_evaluations) is not int or self.max_support_evaluations < 0:
            raise ValueError("support budget must be a nonnegative integer")
        if type(self.max_integer_bits) is not int or self.max_integer_bits < 1:
            raise ValueError("exact arithmetic bit budget must be a positive integer")


def _exact(value: Any) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("stoichiometric coefficients cannot be Boolean")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if not isinstance(value, Real) or not isfinite(value):
        raise ValueError("stoichiometric coefficients must be finite real numbers")
    return Fraction(float(value))


def _names(names: Sequence[str], label: str) -> tuple[str, ...]:
    names = tuple(names)
    if any(not isinstance(n, str) or not n or n != n.strip() for n in names) or (
        len(set(names)) != len(names)
    ):
        raise ValueError(label + " identities must be unique nonempty strings")
    return names


def _matrix(
    rows: Sequence[Sequence[Any]], width: int, height: int
) -> tuple[tuple[Fraction, ...], ...]:
    matrix = tuple(tuple(_exact(v) for v in row) for row in rows)
    if len(matrix) != height or any(len(row) != width for row in matrix):
        raise ValueError("stoichiometric matrix must match the declared species and reactions")
    return matrix


def _primitive(direction: Sequence[Fraction], max_bits: int) -> tuple[int, ...]:
    multiple = 1
    for value in direction:
        multiple = lcm(multiple, value.denominator)
        if multiple.bit_length() > max_bits:
            raise RationalEliminationLimitError("primitive mode integer size budget exhausted")
    integers = tuple(int(v * multiple) for v in direction)
    divisor = reduce(gcd, (abs(v) for v in integers))
    result = tuple(v // divisor for v in integers)
    if any(abs(v).bit_length() > max_bits for v in result):
        raise RationalEliminationLimitError("primitive mode integer size budget exhausted")
    return result


def enumerate_elementary_flux_modes(
    matrix: Sequence[Sequence[Any]],
    *,
    reaction_ids: Sequence[str],
    irreversible: Sequence[bool],
    internal_species_ids: Sequence[str],
    external_matrix: Sequence[Sequence[Any]] = (),
    external_species_ids: Sequence[str] = (),
    policy: FluxModePolicy | None = None,
) -> dict[str, Any]:
    """Enumerate exact circuits compatible with declared irreversible directions.

    A feasible support-minimal kernel vector has a one-dimensional restricted
    kernel, all coordinates nonzero, and support size at most rank(S)+1. Exhaustive
    enumeration of those supports therefore supplies a finite completeness proof.
    An all-reversible support has two feasible orientations represented once with
    bidirectional=True. Arithmetic/support limits return partial results, never
    a false empty-cone conclusion. Floats mean their exact binary rational value.
    """
    policy = policy or FluxModePolicy()
    if not isinstance(policy, FluxModePolicy):
        raise ValueError("flux-mode policy must be explicitly declared")
    names = _names(reaction_ids, "reaction")
    internal = _names(internal_species_ids, "internal species")
    external = _names(external_species_ids, "external species")
    if set(internal) & set(external):
        raise ValueError("internal and external species must be disjoint")
    flags = tuple(irreversible)
    n = len(names)
    if len(flags) != n or any(type(value) is not bool for value in flags):
        raise ValueError("each original reaction needs one explicit irreversible flag")
    source = _matrix(matrix, n, len(internal))
    boundary = _matrix(external_matrix, n, len(external))
    modes: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema_version": "chemworld-elementary-flux-modes-1",
        "object": "support_minimal_feasible_flux_modes",
        "extreme_rays_computed": False,
        "coefficient_interpretation": "exact_integers_fractions_and_binary_float_values",
        "reaction_ids": list(names),
        "irreversible": list(flags),
        "internal_species_ids": list(internal),
        "external_species_ids": list(external),
        "internal_matrix": [[str(v) for v in row] for row in source],
        "external_matrix": [[str(v) for v in row] for row in boundary],
        "normalization": "primitive_integer_ratio_in_original_reaction_extent_coordinates",
        "status": "incomplete",
        "complete": False,
        "no_modes_proven": False,
        "stop_reason": None,
        "rank": None,
        "kernel_dimension": None,
        "lineality_dimension": None,
        "total_supports": None,
        "attempted_supports": 0,
        "evaluated_supports": 0,
        "rank_evaluations": 0,
        "policy": {
            "max_support_evaluations": policy.max_support_evaluations,
            "max_integer_bits": policy.max_integer_bits,
        },
        "modes": modes,
        "boundary": "structural reaction balance; no kinetic or thermodynamic attainability claim",
    }

    def eliminate(rows: Any, width: int) -> tuple[Any, Any]:
        report["rank_evaluations"] += 1
        return _rref(rows, width=width, max_integer_bits=policy.max_integer_bits)

    try:
        for row in boundary:
            _check_rational_bits(row, policy.max_integer_bits)
        _, pivots = eliminate(source, n)
        rank = len(pivots)
        reversible = [i for i, flag in enumerate(flags) if not flag]
        _, reverse_pivots = eliminate(
            [[row[i] for i in reversible] for row in source], len(reversible)
        )
        report.update(
            rank=rank,
            kernel_dimension=n - rank,
            lineality_dimension=len(reversible) - len(reverse_pivots),
        )
        maximum_size = min(n, rank + 1)
        total = 0 if rank == n else sum(comb(n, k) for k in range(1, maximum_size + 1))
        report["total_supports"] = total
        if rank < n:
            for size in range(1, maximum_size + 1):
                for support in combinations(range(n), size):
                    if report["evaluated_supports"] >= policy.max_support_evaluations:
                        report["stop_reason"] = "support_evaluation_budget_exhausted"
                        return report
                    report["attempted_supports"] += 1
                    reduced, columns = eliminate(
                        [[row[i] for i in support] for row in source], size
                    )
                    direction = [Fraction(0)] * size
                    if len(columns) == size - 1:
                        free = next(i for i in range(size) if i not in columns)
                        direction[free] = Fraction(1)
                        for row_index, column in enumerate(columns):
                            direction[column] = -reduced[row_index][free]
                    if all(direction):
                        constrained = [direction[i] for i, j in enumerate(support) if flags[j]]
                        feasible = (
                            not constrained
                            or all(v > 0 for v in constrained)
                            or (all(v < 0 for v in constrained))
                        )
                        if feasible:
                            sign = (
                                1 if (constrained[0] if constrained else direction[0]) > 0 else -1
                            )
                            primitive = _primitive(
                                [sign * v for v in direction], policy.max_integer_bits
                            )
                            vector = [0] * n
                            for column, value in zip(support, primitive, strict=True):
                                vector[column] = value
                            exchanges = [
                                sum(
                                    (
                                        coefficient * v
                                        for coefficient, v in zip(row, vector, strict=True)
                                    ),
                                    Fraction(0),
                                )
                                for row in boundary
                            ]
                            _check_rational_bits(exchanges, policy.max_integer_bits)
                            modes.append(
                                {
                                    "vector": [str(v) for v in vector],
                                    "support": [names[i] for i in support],
                                    "bidirectional": not constrained,
                                    "external_exchange": dict(
                                        zip(external, map(str, exchanges), strict=True)
                                    ),
                                    "internal_cycle": not any(exchanges),
                                }
                            )
                    report["evaluated_supports"] += 1
        report.update(
            status="complete",
            complete=True,
            no_modes_proven=not modes,
            stop_reason="full_column_rank" if rank == n else "all_candidate_supports_evaluated",
        )
    except RationalEliminationLimitError as error:
        report["stop_reason"] = str(error)
    return report


def sbml_elementary_flux_modes(
    frame: SBMLKinetics,
    values: Mapping[str, float],
    *,
    time_s: float = 0.0,
    policy: FluxModePolicy | None = None,
) -> dict[str, Any]:
    """Original-source reaction modes at the actual supplied conversion controls.

    Nonzero species conversion factors scale whole rows, preserving the kernel;
    a zero factor removes that row's balance constraint. Current kinetic rates
    are diagnostics only and never silently disable structural reaction columns.
    """
    if not isinstance(frame, SBMLKinetics) or isinstance(time_s, bool) or not isfinite(time_s):
        raise ValueError("SBML flux modes need the original frame and finite current time")
    _check_scope(frame)
    current = dict(values)
    frame.validate_values(current, time_s)
    context = frame.context(current, time_s)
    declarations = cast(Mapping[str, Any], frame._species)
    amount_scales = cast(Mapping[str, float], frame._amount_scale)
    species = tuple(declarations)
    internal = tuple(
        s
        for s in species
        if not declarations[s].get("boundaryCondition") and not declarations[s].get("constant")
    )
    external = tuple(s for s in species if s not in internal)
    coefficients = dict(frame._stoichiometries)
    default_factor = frame.model.declaration.get("conversionFactor")
    factors, native_factors, rows = {}, {}, {}
    for name in species:
        factor_id = declarations[name].get("conversionFactor", default_factor)
        factor = float(context[factor_id]) if factor_id else 1.0
        scale = _exact(amount_scales[name]) / _exact(frame._extent_scale or 1.0)
        factors[name] = _exact(factor) * scale
        native_factors[name] = factor * amount_scales[name] / (frame._extent_scale or 1.0)
        rows[name] = tuple(
            _exact(coefficients[r].get(name, 0.0)) * factors[name] for r in frame._reaction_ids
        )
    reactions = frame.model.declaration.items("listOfReactions")
    report = enumerate_elementary_flux_modes(
        tuple(rows[s] for s in internal),
        reaction_ids=frame._reaction_ids,
        irreversible=tuple(not r.get("reversible") for r in reactions),
        internal_species_ids=internal,
        external_matrix=tuple(rows[s] for s in external),
        external_species_ids=external,
        policy=policy,
    )
    current_rates = frame.rates(current, time_s)
    residuals = {}
    for index, mode in enumerate(report["modes"]):
        vector = [Fraction(v) for v in mode["vector"]]
        scale = max(abs(v) for v in vector)
        normalized = [float(v / scale) for v in vector]
        balances = {
            s: fsum(
                float(coefficients[r].get(s, 0.0)) * native_factors[s] * v
                for r, v in zip(frame._reaction_ids, normalized, strict=True)
            )
            for s in internal
        }
        if any(not isfinite(v) for v in balances.values()):
            raise ValueError("native float stoichiometric balance exceeds its finite domain")
        residuals[str(index)] = balances
    report.update(
        source="original_SBMLKinetics_current_coordinates",
        time_s=float(time_s),
        current_controls={p: float(context[p]) for p in frame.controlled_parameters},
        current_row_factors={s: str(v) for s, v in factors.items()},
        coefficient_interpretation="exact_compiled_binary_stoichiometry_times_exact_current_row_scales",
        native_normalized_mode_balance=residuals,
        current_reaction_rates_mol_s=dict(current_rates.reaction_extents_mol_s),
        current_internal_rates_mol_s={
            s: current_rates.coordinate_rates["N:" + s] for s in internal
        },
        chemical_composition_scope="unknown_formal_entities",
        energy_scope="not_declared_formal_kinetics",
    )
    return report


__all__ = ["FluxModePolicy", "enumerate_elementary_flux_modes", "sbml_elementary_flux_modes"]
