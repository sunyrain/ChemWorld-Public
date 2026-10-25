"""Explicit molecular-to-element reference bridge and CEF appearance boundaries.

This is an infinitesimal solid stability calculation at a declared homogeneous
fluid state. It does not consume fluid, create solid stock or solve a coupled
reacting fluid/solid assemblage. Original CALPHAD searches and atom inventories
remain the only solid equilibrium implementation and ownership contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import fsum, isfinite, log
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.calphad_equilibrium import CalphadEquilibriumPhase, CalphadEquilibriumSpec
from chemworld.physchem.calphad_equilibrium_search import PhaseSurface, search_phase
from chemworld.physchem.molecular_eos import molecular_residual_chemical_potentials
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_mixture import MolecularCaloricModel
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)
from chemworld.physchem.tdb_expressions import TDB_GAS_CONSTANT_J_MOL_K

SOLID_BOUNDARY_POLICY = RootPolicy(tolerance=1e-10, maximum_iterations=80)


def _strict(data: Mapping[str, Any], fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or set(data) != fields:
        raise ValueError(f"{label} requires exactly {sorted(fields)}")
    return dict(data)


@dataclass(frozen=True)
class MolecularElementReferenceShift:
    """Solid-reference potential MINUS molecular-reference potential per element.

    lambda_solid = lambda_fluid + h - T*s + v*(P-P_ref), in J/mol atoms.
    Constants describe an explicitly sourced reference conversion, not a fit
    to force coexistence. Being element-linear preserves every reaction
    affinity; arbitrary per-molecule offsets are intentionally unsupported.
    """

    element: str
    enthalpy_J_mol_atoms: float
    entropy_J_mol_atoms_K: float
    volume_m3_mol_atoms: float
    reference_pressure_Pa: float
    source: str

    def __post_init__(self) -> None:
        if (
            not self.element
            or not self.source.strip()
            or any(
                isinstance(v, bool) or not isfinite(v)
                for v in (
                    self.enthalpy_J_mol_atoms,
                    self.entropy_J_mol_atoms_K,
                    self.volume_m3_mol_atoms,
                    self.reference_pressure_Pa,
                )
            )
            or self.reference_pressure_Pa <= 0
        ):
            raise ValueError(
                "element reference needs explicit identity, finite offsets and provenance"
            )

    def gibbs_shift(self, temperature_K: float, pressure_Pa: float) -> float:
        return (
            self.enthalpy_J_mol_atoms
            - temperature_K * self.entropy_J_mol_atoms_K
            + self.volume_m3_mol_atoms * (pressure_Pa - self.reference_pressure_Pa)
        )


@dataclass(frozen=True)
class MolecularSolidBoundarySpec:
    fluid: MolecularCaloricModel
    solids: CalphadEquilibriumSpec
    component_stoichiometry: Mapping[str, Mapping[str, float]]
    element_reference_shifts: tuple[MolecularElementReferenceShift, ...]
    source: str
    reaction_tolerance_J_mol: float = 1e-5
    rank_relative_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(self.fluid, MolecularCaloricModel) or not isinstance(
            self.solids, CalphadEquilibriumSpec
        ):
            raise ValueError("solid bridge requires original molecular caloric and CALPHAD models")
        if set(self.component_stoichiometry) != set(self.fluid.component_ids):
            raise ValueError("every molecular component needs an explicit stoichiometric mapping")
        elements = set(cast(Mapping[str, Any], self.solids.models[0].database.elements)) - {
            "VA",
            "/-",
        }
        mapping = {}
        for component, row in self.component_stoichiometry.items():
            if (
                not row
                or set(row) - elements
                or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in row.values())
            ):
                raise ValueError(
                    "stoichiometry needs positive atom counts of declared real elements"
                )
            mapping[component] = MappingProxyType(dict(row))
        shifts = tuple(self.element_reference_shifts)
        mapped_elements = {e for row in mapping.values() for e in row}
        if (
            len({s.element for s in shifts}) != len(shifts)
            or {s.element for s in shifts} != mapped_elements
        ):
            raise ValueError(
                "declare exactly one reference conversion per mapped element, including zeros"
            )
        if (
            not self.source.strip()
            or not 0 < self.rank_relative_tolerance < 1
            or not isfinite(self.rank_relative_tolerance)
            or not isfinite(self.reaction_tolerance_J_mol)
            or self.reaction_tolerance_J_mol <= 0
        ):
            raise ValueError(
                "bridge provenance and explicit positive mapping tolerances are required"
            )
        object.__setattr__(self, "component_stoichiometry", MappingProxyType(mapping))
        object.__setattr__(self, "element_reference_shifts", shifts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "chemworld_molecular_solid_boundary_v1",
            "fluid": self.fluid.to_dict(),
            "solids": self.solids.to_dict(),
            "component_stoichiometry": {
                k: dict(v) for k, v in self.component_stoichiometry.items()
            },
            "element_reference_shifts": [asdict(s) for s in self.element_reference_shifts],
            "source": self.source,
            "reaction_tolerance_J_mol": self.reaction_tolerance_J_mol,
            "rank_relative_tolerance": self.rank_relative_tolerance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MolecularSolidBoundarySpec:
        data = _strict(payload, set(cls.__dataclass_fields__) | {"schema"}, "solid boundary spec")
        if data.pop("schema") != "chemworld_molecular_solid_boundary_v1":
            raise ValueError("unknown solid boundary encoding")
        _strict(data["fluid"], {"model", "residual_model", "ideal_references"}, "caloric model")
        data["fluid"] = MolecularCaloricModel.from_dict(data["fluid"])
        data["solids"] = CalphadEquilibriumSpec.from_dict(data["solids"])
        data["element_reference_shifts"] = tuple(
            MolecularElementReferenceShift(
                **_strict(
                    s, set(MolecularElementReferenceShift.__dataclass_fields__), "element reference"
                )
            )
            for s in data["element_reference_shifts"]
        )
        return cls(**data)


@dataclass(frozen=True)
class MolecularSolidCandidate:
    phase_name: str
    site_fractions: tuple[dict[str, float], ...]
    element_fractions: dict[str, float]
    gibbs_J_mol_atoms: float
    enthalpy_J_mol_atoms: float
    tangent_distance_J_mol_atoms: float

    def calphad_seed(self) -> CalphadEquilibriumPhase:
        """Original42 constitution seed with zero allocated matter, never new stock."""
        return CalphadEquilibriumPhase(
            self.phase_name,
            self.site_fractions,
            self.element_fractions,
            0.0,
            0.0,
            self.gibbs_J_mol_atoms,
            self.enthalpy_J_mol_atoms,
        )


@dataclass(frozen=True)
class MolecularSolidBoundaryResult:
    success: bool
    temperature_K: float
    pressure_Pa: float
    element_amounts_mol: dict[str, float]
    molecular_potentials_J_mol: dict[str, float]
    element_potentials_J_mol_atoms: dict[str, float] | None = None
    candidates: tuple[MolecularSolidCandidate, ...] = ()
    minimum_distance_J_mol_atoms: float | None = None
    solid_favorable: bool | None = None
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": "chemworld_molecular_solid_boundary_result_v1", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MolecularSolidBoundaryResult:
        data = _strict(payload, set(cls.__dataclass_fields__) | {"schema"}, "solid boundary result")
        if data.pop("schema") != "chemworld_molecular_solid_boundary_result_v1":
            raise ValueError("unknown solid boundary result encoding")
        candidates = []
        for raw in data["candidates"]:
            row = _strict(raw, set(MolecularSolidCandidate.__dataclass_fields__), "solid candidate")
            row["site_fractions"] = tuple(row["site_fractions"])
            candidates.append(MolecularSolidCandidate(**row))
        data["candidates"] = tuple(candidates)
        return cls(**data)


def evaluate_molecular_solid_boundary(
    specification: MolecularSolidBoundarySpec,
    temperature_K: float,
    molar_density_mol_m3: float,
    component_amounts_mol: Mapping[str, float],
) -> MolecularSolidBoundaryResult:
    """Convert a reaction-compatible fluid tangent and search every allowed solid.

    Exact absent molecular components carry no finite chemical potential.
    Exact absent elements are excluded from solid constitutions as in42.
    Full active-element rank and every active-species affinity are checked;
    a least-squares compromise is never silently used as an element reference.
    """
    spec, ids = specification, specification.fluid.component_ids
    if (
        set(component_amounts_mol) != set(ids)
        or any(
            isinstance(v, bool) or not isfinite(v) or v < 0 for v in component_amounts_mol.values()
        )
        or fsum(component_amounts_mol.values()) <= 0
    ):
        raise ValueError(
            "solid boundary requires matching finite nonnegative actual molecule amounts"
        )
    total = fsum(component_amounts_mol.values())
    xs = tuple(component_amounts_mol[k] / total for k in ids)
    state = spec.fluid.state(temperature_K, molar_density_mol_m3, xs)
    p, t, rt = state.pressure_Pa, temperature_K, spec.fluid.gas_constant_J_mol_K * temperature_K
    residual = molecular_residual_chemical_potentials(
        spec.fluid.residual_model, t, molar_density_mol_m3, xs
    )
    molecular = {}
    for k, ref, x, mu in zip(ids, spec.fluid.ideal_references, xs, residual, strict=True):
        if x > 0:
            ideal = ref.state(temperature_K=t, molar_density_mol_m3=molar_density_mol_m3)
            molecular[k] = (
                ideal.internal_energy_J_mol + rt - t * ideal.entropy_J_mol_K + rt * log(x) + mu
            )
    stock = {
        e: fsum(
            n * spec.component_stoichiometry[k].get(e, 0.0)
            for k, n in component_amounts_mol.items()
        )
        for e in sorted({s.element for s in spec.element_reference_shifts})
    }
    elements = tuple(e for e, n in stock.items() if n > 0)
    active = tuple(k for k in ids if component_amounts_mol[k] > 0)
    matrix = np.array(
        [[spec.component_stoichiometry[k].get(e, 0.0) for e in elements] for k in active]
    )
    rhs = np.array([molecular[k] for k in active])
    potentials, _, rank, singular = np.linalg.lstsq(matrix, rhs, rcond=spec.rank_relative_tolerance)
    compatibility = float(np.max(np.abs(matrix @ potentials - rhs)))
    diagnostics: dict[str, Any] = {
        "scope": "infinitesimal_solid_tangent; no_fluid_solid_inventory_allocation",
        "search_scope": "finite_grid_and_multistart_local_tangent_search; no_global_certificate",
        "reference_conversion": "lambda_solid=lambda_fluid+h-T*s+v*(P-P_ref)",
        "reference_contract": [asdict(s) for s in spec.element_reference_shifts],
        "source": spec.source,
        "active_components": list(active),
        "active_elements": list(elements),
        "stoichiometric_rank": int(rank),
        "required_rank": len(elements),
        "singular_values": singular.tolist(),
        "reaction_residual_J_mol": compatibility,
        "reaction_tolerance_J_mol": spec.reaction_tolerance_J_mol,
        "molecular_euler_error_J_mol": abs(
            fsum(xs[i] * molecular[k] for i, k in enumerate(ids) if xs[i] > 0)
            - state.gibbs_energy_J_mol
        ),
        "search": [],
        "inactive_phases": [],
    }

    def failure(reason: str) -> MolecularSolidBoundaryResult:
        return MolecularSolidBoundaryResult(
            False, t, p, stock, molecular, failure_reason=reason, diagnostics=diagnostics
        )

    if rank < len(elements):
        return failure("underdetermined_element_potentials")
    if compatibility > spec.reaction_tolerance_J_mol:
        return failure("molecular_state_not_reaction_compatible")
    shifts = {s.element: s.gibbs_shift(t, p) for s in spec.element_reference_shifts}
    target = potentials + np.array([shifts[e] for e in elements])
    diagnostics["fluid_reference_element_potentials_J_mol_atoms"] = dict(
        zip(elements, potentials.tolist(), strict=True)
    )
    candidates = []
    database, policy = spec.solids.models[0].database, spec.solids.policy
    species = cast(Mapping[str, Any], database.species)
    for model in spec.solids.models:
        support = {
            e
            for subl in model.constituents
            for s in subl
            for e, n in species[s].constituents
            if n > 0
        }
        if support - set(elements):
            try:
                model = CefPhaseModel(
                    database,
                    model.phase_name,
                    elements,
                    model.extrapolation,
                    model.unspecified_endmembers,
                    model.activate_suppressed,
                )
            except ValueError as exc:
                diagnostics["inactive_phases"].append(
                    {"phase": model.phase_name, "reason": str(exc)}
                )
                continue
        try:
            surface = PhaseSurface(model, elements, t, p, policy.site_floor)
            minima, attempts = search_phase(
                surface,
                surface.samples(policy.grid_intervals),
                target,
                TDB_GAS_CONSTANT_J_MOL_K * t,
                policy,
            )
            diagnostics["search"].append(
                {
                    "phase": model.phase_name,
                    "attempts": attempts,
                    "successful_minima": len(minima),
                    "minimum_distance_J_mol_atoms": min(
                        (distance for _, distance in minima), default=None
                    ),
                }
            )
            if not minima or any(not a["success"] for a in attempts):
                return failure("solid_tangent_search_incomplete")
            q, distance = min(minima, key=lambda item: item[1])
            sites = surface.sites(q)
            props = model.evaluate(sites, t, p)
            candidates.append(
                MolecularSolidCandidate(
                    model.phase_name,
                    tuple({k: float(v) for k, v in row.items()} for row in sites),
                    {k: float(v) for k, v in props.element_fractions.items()},
                    float(props.gibbs_J_mol),
                    float(props.enthalpy_J_mol),
                    float(distance),
                )
            )
        except (ValueError, ArithmeticError) as exc:
            diagnostics["search"].append({"phase": model.phase_name, "error": str(exc)})
            return failure("solid_tangent_evaluation_failed")
    if not candidates:
        return failure("no_solid_supports_actual_element_stock")
    minimum = min(c.tangent_distance_J_mol_atoms for c in candidates)
    return MolecularSolidBoundaryResult(
        True,
        t,
        p,
        stock,
        molecular,
        dict(zip(elements, target.tolist(), strict=True)),
        tuple(candidates),
        minimum,
        minimum < -policy.tangent_tolerance_J_mol,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class MolecularSolidBoundaryPath:
    """Fixed composition, affine T/P path and one declared homogeneous density basin."""

    start_temperature_K: float
    end_temperature_K: float
    start_pressure_Pa: float
    end_pressure_Pa: float
    density_basin: MolecularDensityBasin

    def __post_init__(self) -> None:
        if any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (
                self.start_temperature_K,
                self.end_temperature_K,
                self.start_pressure_Pa,
                self.end_pressure_Pa,
            )
        ) or (
            self.start_temperature_K == self.end_temperature_K
            and self.start_pressure_Pa == self.end_pressure_Pa
        ):
            raise ValueError("solid boundary path needs distinct finite positive T/P endpoints")

    def conditions(self, coordinate: float) -> tuple[float, float]:
        return (
            (1 - coordinate) * self.start_temperature_K + coordinate * self.end_temperature_K,
            (1 - coordinate) * self.start_pressure_Pa + coordinate * self.end_pressure_Pa,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MolecularSolidBoundaryPath:
        data = _strict(payload, set(cls.__dataclass_fields__), "solid boundary path")
        data["density_basin"] = MolecularDensityBasin(
            **_strict(
                data["density_basin"],
                set(MolecularDensityBasin.__dataclass_fields__),
                "path density basin",
            )
        )
        return cls(**data)


def solve_molecular_solid_boundary_path(
    specification: MolecularSolidBoundarySpec,
    component_amounts_mol: Mapping[str, float],
    path: MolecularSolidBoundaryPath,
    *,
    phase_name: str,
    policy: RootPolicy = SOLID_BOUNDARY_POLICY,
) -> dict[str, Any]:
    """Resolve a bracketed named-solid onset; reject preemption by another solid.

    A sign-changing declared interval is required. This finds one local path
    crossing, not every boundary on that interval. Every sampled fluid state
    must remain reaction compatible and every solid search must complete.
    """
    if phase_name not in {m.phase_name for m in specification.solids.models}:
        raise ValueError("path target must name an explicitly allowed solid phase")
    if set(component_amounts_mol) != set(specification.fluid.component_ids):
        raise ValueError("path feed must exactly match molecular components")
    total = fsum(component_amounts_mol.values())
    if (
        not isfinite(total)
        or total <= 0
        or any(
            isinstance(v, bool) or not isfinite(v) or v < 0 for v in component_amounts_mol.values()
        )
    ):
        raise ValueError("path feed must be finite nonnegative positive stock")
    xs = tuple(component_amounts_mol[k] / total for k in specification.fluid.component_ids)
    attempts: list[dict[str, Any]] = []
    cache: dict[float, MolecularSolidBoundaryResult] = {}

    def evaluate(coordinate: float) -> tuple[MolecularSolidBoundaryResult, float]:
        if coordinate not in cache:
            t, p = path.conditions(coordinate)
            try:
                state = specification.fluid.state_tp(
                    t,
                    p,
                    xs,
                    initial_density_mol_m3=path.density_basin.initial_density_mol_m3,
                    density_bounds_mol_m3=path.density_basin.density_bounds_mol_m3,
                )
                result = evaluate_molecular_solid_boundary(
                    specification, t, state.molar_density_mol_m3, component_amounts_mol
                )
                attempts.append(
                    {
                        "coordinate": coordinate,
                        "success": result.success,
                        "failure_reason": result.failure_reason,
                    }
                )
                cache[coordinate] = result
            except (ValueError, ArithmeticError) as exc:
                attempts.append(
                    {"coordinate": coordinate, "success": False, "failure_reason": str(exc)}
                )
                raise ResidualDomainError(str(exc)) from exc
        result = cache[coordinate]
        if not result.success:
            raise ResidualDomainError(result.failure_reason)
        candidate = next((c for c in result.candidates if c.phase_name == phase_name), None)
        if candidate is None:
            raise ResidualDomainError("requested_solid_inactive_on_actual_element_support")
        return result, candidate.tangent_distance_J_mol_atoms

    output: dict[str, Any] = {
        "success": False,
        "scope": "local_bracketed_incipient_solid; no_stock_allocation",
        "path": path.to_dict(),
        "phase_name": phase_name,
        "attempts": attempts,
        "coordinate": None,
        "boundary": None,
        "failure_reason": "",
    }
    try:
        _, left = evaluate(0.0)
        _, right = evaluate(1.0)
        tolerance = specification.solids.policy.tangent_tolerance_J_mol
        if min(abs(left), abs(right)) > tolerance and left * right >= 0:
            output["failure_reason"] = "solid_onset_not_bracketed"
            return output
        seed = 0.0 if abs(left) <= tolerance else 1.0 if abs(right) <= tolerance else 0.5
        scale = TDB_GAS_CONSTANT_J_MOL_K * path.conditions(0.5)[0]

        def residual(values: Any) -> list[float]:
            return [evaluate(float(values[0]))[1] / scale]

        root = solve_root(
            residual,
            [seed],
            jacobian=lambda v: finite_jacobian(residual, v, [1.0], lower=[0.0], upper=[1.0]),
            variables=(ResidualVariable("path_fraction", "1", lower=0.0, upper=1.0),),
            residual_scales=[1.0],
            policy=policy,
        )
        output["root"] = root.diagnostics
        if not root.success:
            output["failure_reason"] = root.failure_reason
            return output
        result, distance = evaluate(float(root.value[0]))
        output.update(
            coordinate=float(root.value[0]),
            boundary=result.to_dict(),
            tangent_distance_J_mol_atoms=distance,
        )
        if abs(distance) > tolerance:
            output["failure_reason"] = "solid_onset_tangent_not_resolved"
        elif (
            result.minimum_distance_J_mol_atoms is not None
            and result.minimum_distance_J_mol_atoms < -tolerance
        ):
            output["failure_reason"] = "another_allowed_solid_preempts_onset"
        else:
            output["success"] = True
    except (ValueError, ArithmeticError, ResidualDomainError) as exc:
        output["failure_reason"] = str(exc)
    return output
