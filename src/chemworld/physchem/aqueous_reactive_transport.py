"""Finite spatial fast/slow aqueous chemistry on caller-owned original species."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import exp, fsum, isclose, isfinite
from typing import Any

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog
from scipy.sparse import diags

from chemworld.physchem import solver_backend
from chemworld.physchem.aqueous_electrodiffusion import (
    AqueousDiffusionCoefficient,
    zero_current_aqueous_face,
)
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.aqueous_reaction_equilibrium import (
    AqueousEquilibriumSpec,
    solve_aqueous_reaction_equilibrium,
)
from chemworld.physchem.aqueous_transport_flow import (
    AqueousColumnFlowSpec,
    add_flow_faces,
    flow_horizon,
    inventory_properties,
    water_at,
    water_molar_volume,
)
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.solver_backend import ODESolverPolicy


@dataclass(frozen=True)
class AqueousSlowConversion:
    reaction_id: str
    reactant_id: str
    product_id: str
    forward_rate_s: float
    source_id: str

    def __post_init__(self) -> None:
        if (
            not all((self.reaction_id, self.reactant_id, self.product_id, self.source_id))
            or self.reactant_id == self.product_id
            or not isfinite(self.forward_rate_s)
            or self.forward_rate_s <= 0
        ):
            raise ValueError("slow conversion requires identities, source and positive s^-1")


@dataclass(frozen=True)
class AqueousReactiveTransportSpec:
    equilibrium: AqueousEquilibriumSpec
    mesh: DiffusionMesh
    cell_vessel_ids: tuple[str, ...]
    coefficients: tuple[AqueousDiffusionCoefficient, ...]
    slow_conversions: tuple[AqueousSlowConversion, ...] = ()
    temperature_K: float = 298.15
    pressure_Pa: float = 1e5
    ode_policy: ODESolverPolicy = field(
        default_factory=lambda: ODESolverPolicy(
            "aqueous-reactive-transport-bdf-v1",
            method="BDF",
            rtol=2e-8,
            atol=1e-12,
        )
    )
    flow: AqueousColumnFlowSpec | None = None

    def __post_init__(self) -> None:
        for key in ("cell_vessel_ids", "coefficients", "slow_conversions"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        db = self.equilibrium.database
        names = self.cell_vessel_ids
        if (
            self.mesh.geometry != "planar"
            or not 2 <= self.mesh.cells <= 64
            or len(names) != self.mesh.cells
            or len(set(names)) != len(names)
            or not all(names)
            or db.activity_model.family != "ideal"
            or any(s.phase not in {"solvent", "aqueous"} for s in db.species)
        ):
            raise ValueError("reactive transport needs 2..64 finite planar ideal-liquid cells")
        if any(not isfinite(v) or v <= 0 for v in (self.temperature_K, self.pressure_Pa)):
            raise ValueError("fixed T/P must be finite and positive")
        standard = db.standard_states(self.temperature_K, self.pressure_Pa)
        water = db.solvent_species.species_id
        water_volume = standard[water].molar_volume_m3_mol
        if (
            any(s.standard_state.pressure_model != "constant-volume" for s in db.species)
            or water_volume is None
            or water_volume <= 0
            or any(standard[k].molar_volume_m3_mol != 0 for k in self.solute_ids)
        ):
            raise ValueError("stationary solvent requires explicit zero solute molar volume")
        if not any(s.charge for s in db.species):
            raise ValueError("electrodiffusion requires declared mobile ions")
        sources = {s.source_id for s in db.sources}
        if self.flow is not None and (
            self.flow.source_id not in sources
            or {self.flow.feed_vessel_id, self.flow.collector_vessel_id} & set(names)
        ):
            raise ValueError(
                "flow reservoirs must be distinct from cells and use a database source"
            )
        if (
            len(self.coefficients) != len(self.solute_ids)
            or {c.species_id for c in self.coefficients} != set(self.solute_ids)
            or any(c.source_id not in sources for c in self.coefficients)
            or any(
                not c.temperature_bounds_K[0] <= self.temperature_K <= c.temperature_bounds_K[1]
                for c in self.coefficients
            )
        ):
            raise ValueError(
                "all solutes require sourced diffusivities inside their temperature domain"
            )
        reactions = {r.reaction_id: r for r in db.reactions}
        if len({r.reaction_id for r in self.slow_conversions}) != len(self.slow_conversions):
            raise ValueError("slow reaction identities must be unique")
        selected = {*self.equilibrium.reaction_ids, *(r.reaction_id for r in self.slow_conversions)}
        if selected - reactions.keys():
            raise ValueError("transport references an undeclared slow reaction")
        if any(water in reactions[k].stoichiometry for k in selected):
            raise ValueError("stationary water cannot participate in selected reactions")
        species = {s.species_id: s for s in db.species}
        for slow in self.slow_conversions:
            if (
                slow.reaction_id in self.equilibrium.reaction_ids
                or slow.source_id not in sources
                or slow.reaction_id not in reactions
                or slow.reactant_id not in self.solute_ids
                or slow.product_id not in self.solute_ids
                or species[slow.reactant_id].charge
                or species[slow.product_id].charge
                or dict(reactions[slow.reaction_id].stoichiometry)
                != {slow.reactant_id: -1, slow.product_id: 1}
            ):
                raise ValueError("slow conversion must be a distinct declared neutral A <-> B")
        if self.ode_policy.method not in {"BDF", "Radau", "DOP853", "RK45"}:
            raise ValueError("unsupported aqueous transport ODE policy")

    @property
    def solute_ids(self) -> tuple[str, ...]:
        return tuple(
            s.species_id for s in self.equilibrium.database.species if s.phase == "aqueous"
        )

    @property
    def anchor_vessel_id(self) -> str:
        return self.cell_vessel_ids[0]

    @property
    def topology(self) -> tuple[tuple[str, str, str], ...]:
        internal = self.cell_topology
        if self.flow is None:
            return internal
        return (
            ("aqueous-inlet", self.flow.feed_vessel_id, self.cell_vessel_ids[0]),
            *internal,
            ("aqueous-outlet", self.cell_vessel_ids[-1], self.flow.collector_vessel_id),
        )

    @property
    def cell_topology(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (f"aqueous-face:{i}", self.cell_vessel_ids[i - 1], self.cell_vessel_ids[i])
            for i in range(1, self.mesh.cells)
        )

    @property
    def member_vessel_ids(self) -> tuple[str, ...]:
        if self.flow is None:
            return self.cell_vessel_ids
        return (self.flow.feed_vessel_id, *self.cell_vessel_ids, self.flow.collector_vessel_id)

    def vessel_capacity_m3(self, name: str) -> float:
        if name in self.cell_vessel_ids:
            return float(self.mesh.cell_volumes_m3[self.cell_vessel_ids.index(name)])
        if self.flow is not None:
            if name == self.flow.feed_vessel_id:
                return self.flow.feed_capacity_m3
            if name == self.flow.collector_vessel_id:
                return self.flow.collector_capacity_m3
        raise ValueError("unknown aqueous member")

    def reverse_rates(self) -> dict[str, float]:
        standards = self.equilibrium.database.standard_states(self.temperature_K, self.pressure_Pa)
        return {
            r.reaction_id: r.forward_rate_s
            / exp(
                -(standards[r.product_id].gibbs_J_mol - standards[r.reactant_id].gibbs_J_mol)
                / (R * self.temperature_K)
            )
            for r in self.slow_conversions
        }

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                {
                    "model": "finite-aqueous-reactive-transport-v1"
                    if self.flow is None
                    else "finite-aqueous-reactive-transport-v2",
                    "equilibrium": self.equilibrium.to_dict(),
                    "mesh": {
                        "geometry": self.mesh.geometry,
                        "faces_m": self.mesh.faces_m,
                        "area_m2": self.mesh.area_m2,
                    },
                    "cell_vessel_ids": self.cell_vessel_ids,
                    "coefficients": [c.to_dict() for c in self.coefficients],
                    "slow_conversions": [asdict(r) for r in self.slow_conversions],
                    "temperature_K": self.temperature_K,
                    "pressure_Pa": self.pressure_Pa,
                    "ode_policy": asdict(self.ode_policy),
                    **({"flow": self.flow.to_dict()} if self.flow is not None else {}),
                },
                allow_nan=False,
            )
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AqueousReactiveTransportSpec:
        data = dict(payload)
        model = data.pop("model")
        if model not in {
            "finite-aqueous-reactive-transport-v1",
            "finite-aqueous-reactive-transport-v2",
        }:
            raise ValueError("unsupported aqueous reactive transport schema")
        if (model.endswith("v1") and "flow" in data) or (
            model.endswith("v2") and "flow" not in data
        ):
            raise ValueError("aqueous flow schema and declaration disagree")
        if "flow" in data:
            data["flow"] = AqueousColumnFlowSpec(**data["flow"])
        data["equilibrium"] = AqueousEquilibriumSpec.from_dict(data["equilibrium"])
        data["mesh"] = DiffusionMesh(**data["mesh"])
        data["coefficients"] = tuple(AqueousDiffusionCoefficient(**c) for c in data["coefficients"])
        data["slow_conversions"] = tuple(
            AqueousSlowConversion(**r) for r in data["slow_conversions"]
        )
        data["ode_policy"]["event_names"] = tuple(data["ode_policy"]["event_names"])
        data["ode_policy"] = ODESolverPolicy(**data["ode_policy"])
        return cls(**data)


def _validate_inventory(
    spec: AqueousReactiveTransportSpec, name: str, amounts: Mapping[str, float]
) -> None:
    db = spec.equilibrium.database
    if set(amounts) != {s.species_id for s in db.species} or any(
        not isfinite(n) or n < 0 for n in amounts.values()
    ):
        raise ValueError("cell requires every original finite nonnegative species inventory")
    if (
        spec.flow is not None
        and name == spec.flow.collector_vessel_id
        and not any(amounts.values())
    ):
        return
    if any(amounts[s.species_id] <= 0 for s in db.species if s.charge):
        raise ValueError("all declared ions must be strictly positive in every cell")
    charge = fsum(s.charge * amounts[s.species_id] for s in db.species)
    scale = fsum(abs(s.charge) * amounts[s.species_id] for s in db.species)
    if abs(charge) > 1e-10 * scale:
        raise ValueError("cell must be electroneutral without counterion repair")
    props = aqueous_phase_properties(
        db, amounts, temperature_K=spec.temperature_K, pressure_Pa=spec.pressure_Pa
    )
    if name in spec.cell_vessel_ids:
        if not isclose(
            props.volume_m3, spec.vessel_capacity_m3(name), rel_tol=1e-10, abs_tol=1e-16
        ):
            raise ValueError("actual finite water volume differs from the declared mesh cell")
    elif spec.flow is None or name not in spec.member_vessel_ids:
        raise ValueError("unknown aqueous reservoir")
    elif props.volume_m3 > spec.vessel_capacity_m3(name) + 1e-15:
        raise ValueError("finite aqueous reservoir exceeds capacity")
    elif (
        name == spec.flow.feed_vessel_id
        and props.volume_m3 < spec.flow.minimum_feed_volume_m3 - 1e-15
    ):
        raise ValueError("finite aqueous feed below declared reserve")


def prepare_aqueous_transport_state(
    spec: AqueousReactiveTransportSpec,
    amounts_by_vessel: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if set(amounts_by_vessel) != set(spec.member_vessel_ids):
        raise ValueError("all declared cells need their actual inventory")
    result = {}
    for name in spec.member_vessel_ids:
        n = dict(amounts_by_vessel[name])
        _validate_inventory(spec, name, n)
        if spec.equilibrium.reaction_ids and any(n.values()):
            solved = solve_aqueous_reaction_equilibrium(
                spec.equilibrium, n, temperature_K=spec.temperature_K, pressure_Pa=spec.pressure_Pa
            )
            if not solved.success or solved.amounts_mol is None:
                raise ValueError("fast aqueous closure failed: " + solved.failure_reason)
            n = solved.amounts_mol
        _validate_inventory(spec, name, n)
        result[name] = n
    return result


def aqueous_transport_faces(
    spec: AqueousReactiveTransportSpec,
    amounts_by_vessel: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    db = spec.equilibrium.database
    charges = {s.species_id: s.charge for s in db.species if s.phase == "aqueous"}
    diffusion = {c.species_id: c.value_m2_s for c in spec.coefficients}
    concentrations = {}
    for i, name in enumerate(spec.cell_vessel_ids):
        _validate_inventory(spec, name, amounts_by_vessel[name])
        concentrations[name] = {
            k: amounts_by_vessel[name][k] / spec.mesh.cell_volumes_m3[i] for k in spec.solute_ids
        }
    standards = db.standard_states(spec.temperature_K, spec.pressure_Pa)
    faces = {}
    for i, (identity, left, right) in enumerate(spec.cell_topology, 1):
        row = zero_current_aqueous_face(
            concentrations[left],
            concentrations[right],
            charges,
            diffusion,
            distance_m=float(spec.mesh.centers_m[i] - spec.mesh.centers_m[i - 1]),
            area_m2=float(spec.mesh.face_areas_m2[i]),
            temperature_K=spec.temperature_K,
        )
        row.update(
            source=left,
            target=right,
            enthalpy_W=fsum(
                standards[k].enthalpy_J_mol * v for k, v in row["amount_rates_mol_s"].items()
            ),
        )
        faces[identity] = row
    if spec.flow is not None:
        return add_flow_faces(spec, amounts_by_vessel, faces)
    return {
        "faces": faces,
        "boundary_amount_rates_mol_s": {
            side: {s.species_id: 0.0 for s in db.species} for side in ("left", "right")
        },
    }


@dataclass(frozen=True)
class AqueousReactiveTransportResult:
    trajectory: tuple[dict[str, Any], ...]
    actual_duration_s: float
    diagnostics: dict[str, Any]

    @property
    def final(self) -> dict[str, Any]:
        return self.trajectory[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory": list(self.trajectory),
            "actual_duration_s": self.actual_duration_s,
            "diagnostics": self.diagnostics,
        }


def advance_aqueous_reactive_transport(
    spec: AqueousReactiveTransportSpec,
    amounts_by_vessel: Mapping[str, Mapping[str, float]],
    duration_s: float,
    *,
    evaluation_times_s: Sequence[float] = (),
) -> AqueousReactiveTransportResult:
    """Integrate temporary invariant coordinates; accepted stocks remain species n."""
    if not isfinite(duration_s) or duration_s < 0:
        raise ValueError("transport duration must be finite and nonnegative")
    times = tuple(sorted({0.0, float(duration_s), *map(float, evaluation_times_s)}))
    if any(not isfinite(t) or t < 0 or t > duration_s for t in times):
        raise ValueError("observation outside requested interval")
    initial = prepare_aqueous_transport_state(spec, amounts_by_vessel)
    if any(
        not isclose(initial[v][s], n, rel_tol=1e-8, abs_tol=1e-14)
        for v, ns in amounts_by_vessel.items()
        for s, n in ns.items()
    ):
        raise ValueError("initial cells must already satisfy their fast equilibrium")
    requested_duration = duration_s
    duration_s, stopping_events = flow_horizon(spec, initial, duration_s)
    times = tuple(sorted({0.0, duration_s, *(t for t in times if t <= duration_s)}))
    db, ids, vessels = spec.equilibrium.database, spec.solute_ids, spec.member_vessel_ids
    node_count = len(vessels)
    vessel_index = {v: i for i, v in enumerate(vessels)}
    water = db.solvent_species.species_id
    index = {k: i for i, k in enumerate(ids)}
    reactions = {r.reaction_id: r for r in db.reactions}
    fast = np.array(
        [
            [reactions[r].stoichiometry.get(k, 0) for k in ids]
            for r in spec.equilibrium.reaction_ids
        ],
        dtype=float,
    ).reshape((-1, len(ids)))
    charge = np.array([next(s.charge for s in db.species if s.species_id == k) for k in ids])
    basis = null_space(np.vstack((fast, charge)), rcond=1e-12).T
    n0 = np.array([[initial[v][k] for k in ids] for v in vessels])
    b0 = n0 @ basis.T
    # Concentration scaling avoids a grid-dependent absolute mole tolerance.
    volumes = (
        np.asarray(spec.mesh.cell_volumes_m3)
        if spec.flow is None
        else np.array(
            [
                max(
                    initial[v][water] * water_molar_volume(spec),
                    float(spec.mesh.cell_volumes_m3[0]),
                )
                if v not in spec.cell_vessel_ids
                else spec.vessel_capacity_m3(v)
                for v in vessels
            ]
        )
    )
    width = basis.shape[0]
    reverse = spec.reverse_rates()
    counters = {"rhs_evaluations": 0, "fast_closures": 0, "feasible_initializations": 0}
    last_progress = time.monotonic()
    cache: dict[tuple[int, tuple[float, ...]], dict[str, float]] = {}

    def close(t: float, y: np.ndarray, *, observation: bool = True) -> dict[str, dict[str, float]]:
        coordinates = y[: node_count * width].reshape((node_count, width)) * volumes[:, None]
        waters = water_at(spec, initial, t)
        result = {}
        for i, name in enumerate(vessels):
            if spec.flow is not None and name == spec.flow.collector_vessel_id and not observation:
                # No return flux and no slow reaction in the receiving tank:
                # collector chemistry cannot affect this RHS. Its actual fast
                # equilibrium is restored from accumulated invariants at output.
                result[name] = dict(initial[name])
                continue
            if waters[name] == 0:
                if np.any(coordinates[i] != 0):
                    raise ValueError("zero aqueous water cannot contain transported solutes")
                result[name] = dict.fromkeys(initial[name], 0.0)
                continue
            signature = (i, (float(waters[name]), *tuple(map(float, coordinates[i]))))
            if signature in cache:
                result[name] = cache[signature]
                continue
            affine = n0[i] + basis.T @ (coordinates[i] - b0[i])
            if np.any(affine < 0):
                constraints = np.vstack((basis, charge))
                target = np.r_[coordinates[i], 0.0]
                scale = np.maximum(abs(constraints) @ abs(n0[i]), 1e-16)
                feasible = linprog(
                    np.zeros(len(ids)),
                    A_eq=constraints / scale[:, None],
                    b_eq=target / scale,
                    bounds=(0, None),
                    method="highs",
                )
                counters["feasible_initializations"] += 1
                if not feasible.success:
                    raise ValueError("no nonnegative liquid on transported reaction invariants")
                affine = feasible.x
            feed = {water: waters[name], **dict(zip(ids, map(float, affine), strict=True))}
            if spec.equilibrium.reaction_ids:
                counters["fast_closures"] += 1
                solved = solve_aqueous_reaction_equilibrium(
                    spec.equilibrium,
                    feed,
                    temperature_K=spec.temperature_K,
                    pressure_Pa=spec.pressure_Pa,
                )
                if not solved.success or solved.amounts_mol is None:
                    raise ValueError("transport fast closure failed: " + solved.failure_reason)
                feed = solved.amounts_mol
            _validate_inventory(spec, name, feed)
            cache[signature] = feed
            result[name] = feed
        if len(cache) > 2048:
            cache.clear()
        return result

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal last_progress
        counters["rhs_evaluations"] += 1
        cells = close(t, y, observation=False)
        rows = aqueous_transport_faces(spec, cells)["faces"]
        rates = np.zeros_like(n0)
        h_rates = np.zeros(node_count)
        for row in rows.values():
            values = np.array([row["amount_rates_mol_s"][k] for k in ids])
            left, right = vessel_index[row["source"]], vessel_index[row["target"]]
            rates[left] -= values
            rates[right] += values
            h_rates[left] -= row["enthalpy_W"]
            h_rates[right] += row["enthalpy_W"]
        for i, name in enumerate(vessels):
            if name not in spec.cell_vessel_ids:
                continue
            for slow in spec.slow_conversions:
                rate = (
                    slow.forward_rate_s * cells[name][slow.reactant_id]
                    - reverse[slow.reaction_id] * cells[name][slow.product_id]
                )
                rates[i, index[slow.reactant_id]] -= rate
                rates[i, index[slow.product_id]] += rate
        if time.monotonic() - last_progress >= 30:
            print(
                f"aqueous transport t={t:.6g}/{duration_s:g}s "
                f"rhs={counters['rhs_evaluations']} fast_closures={counters['fast_closures']}",
                flush=True,
            )
            last_progress = time.monotonic()
        return np.r_[(rates @ basis.T / volumes[:, None]).ravel(), h_rates]

    y0 = np.r_[(b0 / volumes[:, None]).ravel(), np.zeros(node_count)]
    diagnostic: dict[str, Any] = {"success": True, "nfev": 0}
    if duration_s:
        # Every invariant depends only on its cell and adjacent cells; enthalpy
        # integrals depend on those same stocks, never on the integrals themselves.
        adjacent = diags(
            [np.ones(node_count - 1), np.ones(node_count), np.ones(node_count - 1)],
            [-1, 0, 1],
        ).toarray()
        sparsity = np.zeros((y0.size, y0.size), dtype=bool)
        sparsity[: b0.size, : b0.size] = np.kron(adjacent, np.ones((width, width)))
        sparsity[b0.size :, : b0.size] = np.kron(adjacent, np.ones((1, width)))
        solved = solver_backend.solve_ode(
            rhs,
            y0,
            time_span_s=(0, duration_s),
            evaluation_times_s=times,
            policy=spec.ode_policy,
            jac_sparsity=sparsity if spec.ode_policy.method in {"BDF", "Radau"} else None,
        )
        diagnostic = solved.diagnostic.to_dict()
        if not solved.raw_result.success or len(solved.raw_result.t) != len(times):
            raise ValueError("aqueous transport ODE failed: " + solved.raw_result.message)
        states = solved.raw_result.y.T
    else:
        states = np.array([y0])
    before = {v: inventory_properties(spec, n) for v, n in initial.items()}
    trajectory = []
    for t, y in zip(times, states, strict=True):
        cells = close(float(t), y)
        face_data = aqueous_transport_faces(spec, cells)
        details = {}
        for i, name in enumerate(vessels):
            current = inventory_properties(spec, cells[name])
            delta_h = current.enthalpy_J - before[name].enthalpy_J
            transferred_h = float(y[b0.size + i])
            details[name] = {
                "enthalpy_change_J": delta_h,
                "internal_energy_change_J": current.internal_energy_J
                - before[name].internal_energy_J,
                "transported_enthalpy_J": transferred_h,
                "bath_heat_J": delta_h - transferred_h,
                **(
                    {
                        "volume_m3": current.volume_m3,
                        "boundary_work_J": spec.pressure_Pa
                        * (current.volume_m3 - before[name].volume_m3),
                    }
                    if spec.flow is not None
                    else {}
                ),
            }
        trajectory.append(
            {
                "time_s": float(t),
                "amounts_by_vessel": cells,
                **face_data,
                "cells": details,
                "bath_heat_J": fsum(r["bath_heat_J"] for r in details.values()),
            }
        )
    return AqueousReactiveTransportResult(
        tuple(trajectory),
        duration_s,
        {
            **counters,
            "solver": diagnostic,
            "coordinate_scope": (
                "temporary fast-reaction invariants; persistent stocks are original species moles"
            ),
            **(
                {
                    "requested_duration_s": requested_duration,
                    "stopping_events": stopping_events,
                    "water_model": (
                        "actual finite water; equal column face inflow/outflow; "
                        "exact constant-q reservoir balance"
                    ),
                    "mechanical_model": (
                        "same-pressure kinematic flow; no pump work or pressure drop"
                    ),
                    "slow_reaction_scope": "column cells only; reservoirs use fast equilibrium",
                }
                if spec.flow is not None
                else {}
            ),
        },
    )
