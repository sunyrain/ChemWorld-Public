"""Finite seeded mineral rates with fast declared liquid equilibrium.

Only caller-owned mineral amounts evolve. Relative radii regularize exhaustion;
the common ODE backend locates its zero and continues the remaining minerals.
An algebraic liquid closure preserves the full selected reaction invariants.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from math import exp, expm1, fsum, isfinite
from typing import Any, cast

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from chemworld.physchem import solver_backend
from chemworld.physchem.aqueous_mineral_equilibrium import (
    AqueousMineralEquilibriumSpec,
    aqueous_mineral_properties,
)
from chemworld.physchem.aqueous_reaction_equilibrium import (
    AqueousEquilibriumSpec,
    solve_aqueous_reaction_equilibrium,
)
from chemworld.physchem.elements import element_matrix
from chemworld.physchem.mineral_kinetics import MineralGeometry
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.solver_backend import ODESolverPolicy


@dataclass(frozen=True)
class MineralRateLaw:
    mineral_id: str
    rate_constant_mol_m2_s: float
    activation_energy_J_mol: float = 0.0
    reference_temperature_K: float = 298.15

    def __post_init__(self) -> None:
        if (
            not self.mineral_id
            or any(
                isinstance(x, bool) or not isfinite(x) or x < 0
                for x in (
                    self.rate_constant_mol_m2_s,
                    self.activation_energy_J_mol,
                )
            )
            or not isfinite(self.reference_temperature_K)
            or self.reference_temperature_K <= 0
        ):
            raise ValueError("mineral rate requires explicit finite nonnegative kinetic data")

    def coefficient(self, temperature_K: float) -> float:
        return self.rate_constant_mol_m2_s * exp(
            -self.activation_energy_J_mol
            / R
            * (1 / temperature_K - 1 / self.reference_temperature_K)
        )


@dataclass(frozen=True)
class AqueousMineralKineticsSpec:
    equilibrium: AqueousMineralEquilibriumSpec
    rates: tuple[MineralRateLaw, ...]
    ode_policy: ODESolverPolicy = field(
        default_factory=lambda: ODESolverPolicy(
            "declared-mineral-radius-bdf-v1",
            method="BDF",
            rtol=1e-8,
            atol=1e-11,
            dense_output=True,
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "rates", tuple(self.rates))
        ids = {r.mineral_id for r in self.rates}
        if (
            len(ids) != len(self.rates)
            or ids != set(self.equilibrium.mineral_ids)
            or ids != set(self.equilibrium.dissolution_reactions)
        ):
            raise ValueError("each declared mineral needs one selected dissolution rate")
        if self.ode_policy.method not in {"BDF", "Radau", "DOP853", "RK45"}:
            raise ValueError("declare a supported deterministic mineral ODE strategy")

    @property
    def fast_spec(self) -> AqueousEquilibriumSpec:
        slow = set(self.equilibrium.dissolution_reactions.values())
        return AqueousEquilibriumSpec(
            self.equilibrium.database,
            tuple(r for r in self.equilibrium.reaction_ids if r not in slow),
            self.equilibrium.root_policy,
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                {
                    "model": "declared-aqueous-mineral-kinetics-v1",
                    "equilibrium": self.equilibrium.to_dict(),
                    "rates": [asdict(r) for r in self.rates],
                    "ode_policy": asdict(self.ode_policy),
                },
                allow_nan=False,
            )
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AqueousMineralKineticsSpec:
        if payload.get("model") != "declared-aqueous-mineral-kinetics-v1":
            raise ValueError("unknown declared mineral kinetics encoding")
        policy = dict(payload["ode_policy"])
        policy["event_names"] = tuple(policy.get("event_names", ()))
        return cls(
            AqueousMineralEquilibriumSpec.from_dict(payload["equilibrium"]),
            tuple(MineralRateLaw(**r) for r in payload["rates"]),
            ODESolverPolicy(**policy),
        )


def prepare_aqueous_mineral_state(
    spec: AqueousMineralKineticsSpec,
    amounts_mol: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
) -> dict[str, float]:
    """Prepare only fast liquid reactions; never equilibrate or create a mineral."""
    db = spec.equilibrium.database
    ids = {s.species_id for s in db.species}
    if set(amounts_mol) - ids or any(
        isinstance(n, bool) or not isfinite(n) or n < 0 for n in amounts_mol.values()
    ):
        raise ValueError("mineral kinetics needs declared finite nonnegative original stocks")
    feed = {key: float(amounts_mol.get(key, 0)) for key in ids}
    liquid = {s.species_id: feed[s.species_id] for s in db.species if s.phase != "mineral"}
    charge = fsum(s.charge * feed[s.species_id] for s in db.species)
    charge_scale = max(fsum(abs(s.charge) * feed[s.species_id] for s in db.species), 1e-12)
    if abs(charge) > 1e-10 * charge_scale:
        raise ValueError("initial liquid must be electroneutral without counterion repair")
    if spec.fast_spec.reaction_ids:
        result = solve_aqueous_reaction_equilibrium(
            spec.fast_spec,
            liquid,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
        )
        if not result.success or result.amounts_mol is None:
            raise ValueError("fast aqueous closure failed: " + result.failure_reason)
        liquid = result.amounts_mol
    else:
        db.chemical_potentials(liquid, temperature_K=temperature_K, pressure_Pa=pressure_Pa)
    return {**feed, **liquid}


@dataclass(frozen=True)
class AqueousMineralKineticPoint:
    time_s: float
    amounts_mol: dict[str, float]
    surface_areas_m2: dict[str, float]
    dissolution_rates_mol_s: dict[str, float]
    saturation_indices: dict[str, float | None]
    heat_duty_J: float
    boundary_work_J: float
    internal_energy_change_J: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AqueousMineralKineticResult:
    trajectory: tuple[AqueousMineralKineticPoint, ...]
    actual_duration_s: float
    events: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]

    @property
    def final(self) -> AqueousMineralKineticPoint:
        return self.trajectory[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory": [p.to_dict() for p in self.trajectory],
            "actual_duration_s": self.actual_duration_s,
            "events": self.events,
            "diagnostics": self.diagnostics,
        }


def advance_aqueous_minerals(
    spec: AqueousMineralKineticsSpec,
    amounts_mol: Mapping[str, float],
    geometries: Mapping[str, MineralGeometry],
    duration_s: float,
    *,
    temperature_K: float,
    pressure_Pa: float,
    evaluation_times_s: Sequence[float] = (),
    maximum_volume_L: float | None = None,
) -> AqueousMineralKineticResult:
    """Advance one finite TP inventory; failures return no installable candidate.

    A negative trial radius is extended at its zero face only during root search.
    The event supplies exact exhaustion; every accepted/output radius is checked,
    never clipped. Exhausted/no-seed minerals cannot spontaneously nucleate.
    """
    if isinstance(duration_s, bool) or not isfinite(duration_s) or duration_s < 0:
        raise ValueError("mineral duration must be finite and nonnegative")
    times = tuple(sorted({0.0, float(duration_s), *map(float, evaluation_times_s)}))
    if any(not isfinite(t) or t < 0 or t > duration_s for t in times):
        raise ValueError("observation times must lie inside the requested interval")
    if maximum_volume_L is not None and (not isfinite(maximum_volume_L) or maximum_volume_L <= 0):
        raise ValueError("finite vessel capacity must be positive")
    eq, db = spec.equilibrium, spec.equilibrium.database
    minerals = eq.mineral_ids
    if set(geometries) != set(minerals):
        raise ValueError("every mineral needs its original declared reference geometry")
    initial = prepare_aqueous_mineral_state(
        spec,
        amounts_mol,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
    )
    if any(abs(initial[k] - amounts_mol.get(k, 0)) > 1e-11 + 1e-8 * initial[k] for k in initial):
        raise ValueError("initial liquid must already satisfy its fast equilibrium")
    before = aqueous_mineral_properties(
        eq, initial, temperature_K=temperature_K, pressure_Pa=pressure_Pa
    )
    if maximum_volume_L is not None and before.volume_m3 * 1000 > maximum_volume_L:
        raise ValueError("initial mineral inventory exceeds vessel capacity")
    for key in minerals:
        if initial[key] > 0 and geometries[key].reference_amount_mol <= 0:
            raise ValueError("finite mineral stock requires its material-owned reference geometry")
    liquid_ids = tuple(s.species_id for s in db.species if s.phase != "mineral")
    reactions = {r.reaction_id: r for r in db.reactions}
    slow = np.array(
        [
            [reactions[eq.dissolution_reactions[m]].stoichiometry.get(k, 0) for m in minerals]
            for k in liquid_ids
        ],
        dtype=float,
    )
    fast_spec = spec.fast_spec
    fast = np.array(
        [
            [reactions[r].stoichiometry.get(k, 0) for r in fast_spec.reaction_ids]
            for k in liquid_ids
        ],
        dtype=float,
    ).reshape((len(liquid_ids), len(fast_spec.reaction_ids)))
    fast_invariants = null_space(fast.T, rcond=1e-12).T
    initial_liquid = np.array([initial[k] for k in liquid_ids])
    initial_solids = np.array([initial[k] for k in minerals])
    laws = {r.mineral_id: r for r in spec.rates}
    coefficients = {k: laws[k].coefficient(temperature_K) for k in minerals}
    standards = db.standard_states(temperature_K, pressure_Pa)
    counters = {
        "liquid_closures": 0,
        "feasible_liquid_initializations": 0,
        "negative_radius_trials": 0,
    }
    cache: dict[
        tuple[float, ...], tuple[dict[str, float], dict[str, float | None], dict[str, float]]
    ] = {}

    def close(
        stocks: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float | None], dict[str, float]]:
        signature = tuple(map(float, stocks))
        if signature in cache:
            return cache[signature]
        counters["liquid_closures"] += 1
        affine = initial_liquid + slow @ (initial_solids - stocks)
        if np.any(affine < 0):
            # Find only a numerical feed on the fast-reaction affine subspace.
            # The resulting equilibrium still has exactly the original targets.
            if not fast_spec.reaction_ids:
                raise ValueError("kinetic trial leaves nonnegative liquid inventory")
            row_scale = np.maximum(abs(fast_invariants) @ np.maximum(abs(affine), 1e-8), 1e-15)
            feasible = linprog(
                np.zeros(len(liquid_ids)),
                A_eq=fast_invariants / row_scale[:, None],
                b_eq=fast_invariants @ affine / row_scale,
                bounds=(0, None),
                method="highs",
            )
            counters["feasible_liquid_initializations"] += 1
            if not feasible.success:
                raise ValueError("no nonnegative liquid in the current kinetic reaction subspace")
            affine = feasible.x
        liquid = dict(zip(liquid_ids, map(float, affine), strict=True))
        if fast_spec.reaction_ids:
            result = solve_aqueous_reaction_equilibrium(
                fast_spec,
                liquid,
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
            )
            if not result.success or result.amounts_mol is None:
                raise ValueError("kinetic fast liquid closure: " + result.failure_reason)
            liquid = result.amounts_mol
        chemical = db.chemical_potentials(
            liquid, temperature_K=temperature_K, pressure_Pa=pressure_Pa
        )
        amounts = {**liquid, **dict(zip(minerals, map(float, stocks), strict=True))}
        indices, drives = {}, {}
        for mineral in minerals:
            reaction = reactions[eq.dissolution_reactions[mineral]]
            value = -standards[mineral].gibbs_J_mol
            absent_product = False
            for key, nu in reaction.stoichiometry.items():
                if key == mineral:
                    continue
                mu = chemical.chemical_potentials_J_mol[key]
                if mu is None:
                    if nu < 0:
                        raise ValueError("missing aqueous reactant has no resolved mineral rate")
                    absent_product = True
                else:
                    value += nu * mu
            si = None if absent_product else value / (R * temperature_K)
            indices[mineral] = si
            drives[mineral] = 1.0 if si is None else -expm1(si)
        cache[signature] = (amounts, indices, drives)
        return cache[signature]

    active = [
        k
        for k in minerals
        if initial[k] > 0 and geometries[k].reference_area_m2 > 0 and coefficients[k] > 0
    ]
    stocks = initial_solids.copy()
    segments: list[tuple[float, float, tuple[str, ...], Any, np.ndarray]] = []
    events: list[dict[str, Any]] = []
    diagnostics = []
    start = 0.0
    while active and start < duration_s:
        keys = tuple(active)
        baseline = stocks.copy()
        scales = {k: stocks[minerals.index(k)] for k in keys}
        factors = {
            k: coefficients[k]
            * geometries[k].reference_area_m2
            / (3 * geometries[k].reference_amount_mol ** (2 / 3) * scales[k] ** (1 / 3))
            for k in keys
        }

        def decode(
            radius: np.ndarray,
            *,
            accepted: bool = False,
            baseline: np.ndarray = baseline,
            keys: tuple[str, ...] = keys,
            scales: dict[str, float] = scales,
        ) -> np.ndarray:
            value = baseline.copy()
            for j, key in enumerate(keys):
                u = float(radius[j])
                if u < 0:
                    if accepted:
                        raise RuntimeError("accepted mineral radius is negative")
                    counters["negative_radius_trials"] += 1
                    u = 0.0  # Root-search extension; never an accepted stock correction.
                value[minerals.index(key)] = scales[key] * u**3
            return value

        def rhs(
            t: float,
            radius: np.ndarray,
            factors: dict[str, float] = factors,
            keys: tuple[str, ...] = keys,
            decode: Any = decode,
        ) -> np.ndarray:
            _, _, drives = close(decode(radius))
            return np.array([-factors[k] * drives[k] for k in keys])

        roots: list[Any] = []
        for j in range(len(keys)):

            def exhausted(t: float, radius: np.ndarray, j: int = j) -> float:
                return float(radius[j])

            cast(Any, exhausted).terminal = True
            cast(Any, exhausted).direction = -1
            roots.append(exhausted)
        if maximum_volume_L is not None:

            def capacity(t: float, radius: np.ndarray) -> float:
                amounts, _, _ = close(decode(radius))
                props = aqueous_mineral_properties(
                    eq, amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa
                )
                return maximum_volume_L - props.volume_m3 * 1000

            cast(Any, capacity).terminal = True
            cast(Any, capacity).direction = -1
            roots.append(capacity)
        policy = replace(
            spec.ode_policy,
            dense_output=True,
            event_names=tuple("exhausted:" + k for k in keys)
            + (("vessel_capacity",) if maximum_volume_L is not None else ()),
        )
        report = solver_backend.solve_ode(
            rhs,
            np.ones(len(keys)),
            time_span_s=(start, duration_s),
            policy=policy,
            events=tuple(roots),
        )
        report.raise_for_failure("declared mineral kinetics")
        diagnostics.append(report.diagnostic.to_dict())
        terminal = report.terminal_point
        end = float(report.t[-1]) if terminal is None else terminal[0]
        radius = report.y[:, -1] if terminal is None else terminal[1].copy()
        if terminal is not None:
            j = terminal[2]
            if j == len(keys):
                raise ValueError("finite vessel capacity reached during mineral kinetics")
            radius[j] = 0.0  # The root defines the exact exhausted face.
            events.append({"event": "mineral_exhausted", "mineral_id": keys[j], "time_s": end})
            active.remove(keys[j])
        stocks = decode(radius, accepted=True)
        segments.append((start, end, keys, report.raw_result.sol, baseline))
        if end <= start:
            raise RuntimeError("mineral event failed to advance the common interval")
        start = end

    def point(time: float) -> AqueousMineralKineticPoint:
        values = stocks.copy()
        for lo, hi, keys, solution, baseline in segments:
            if lo <= time < hi:
                values = baseline.copy()
                radius = solution(time)
                for j, k in enumerate(keys):
                    if radius[j] < 0:
                        raise RuntimeError("accepted dense mineral radius is negative")
                    values[minerals.index(k)] = baseline[minerals.index(k)] * float(radius[j]) ** 3
                break
        amounts, si, drives = close(values)
        properties = aqueous_mineral_properties(
            eq, amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa
        )
        if maximum_volume_L is not None and properties.volume_m3 * 1000 > maximum_volume_L * (
            1 + 1e-12
        ):
            raise ValueError("accepted mineral inventory exceeds capacity")
        areas = {k: geometries[k].area(amounts[k]) for k in minerals}
        return AqueousMineralKineticPoint(
            float(time),
            amounts,
            areas,
            {k: coefficients[k] * areas[k] * drives[k] for k in minerals},
            si,
            properties.enthalpy_J - before.enthalpy_J,
            pressure_Pa * (properties.volume_m3 - before.volume_m3),
            properties.internal_energy_J - before.internal_energy_J,
        )

    trajectory = tuple(point(t) for t in times)
    ids = tuple(s.species_id for s in db.species)
    selected = np.array(
        [[reactions[r].stoichiometry.get(k, 0) for r in eq.reaction_ids] for k in ids]
    )
    invariants = null_space(selected.T, rcond=1e-12).T
    atom_rows, elements = element_matrix([dict(s.composition) for s in db.species])
    atoms = np.array(atom_rows).T
    changes = np.array([[p.amounts_mol[k] - initial[k] for k in ids] for p in trajectory])
    charges = np.array([s.charge for s in db.species])
    max_invariant = float(np.max(abs(changes @ invariants.T), initial=0))
    max_charge = float(np.max(abs(changes @ charges), initial=0))
    max_element = {
        e: float(np.max(abs(changes @ row), initial=0))
        for e, row in zip(elements, atoms, strict=True)
    }
    if max(max_invariant, max_charge, *max_element.values()) > 2e-8:
        raise RuntimeError("accepted mineral trajectory violates original reaction invariants")
    return AqueousMineralKineticResult(
        trajectory,
        float(duration_s),
        tuple(events),
        {
            **counters,
            "solver_calls": len(diagnostics),
            "solvers": diagnostics,
            "maximum_invariant_residual_mol": max_invariant,
            "maximum_charge_residual_mol": max_charge,
            "maximum_element_residuals_mol": max_element,
            "scope": "closed_isothermal_isobaric_seeded_kinetics; no_nucleation_or_transport",
            "no_seed_minerals": [k for k in minerals if initial[k] == 0],
            "inactive_surfaces": [k for k in minerals if geometries[k].reference_area_m2 == 0],
        },
    )
