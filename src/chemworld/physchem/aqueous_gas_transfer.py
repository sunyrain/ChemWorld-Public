"""Finite ideal headspace coupled to the original fast aqueous equilibrium.

Solvent is explicitly nonvolatile. The sealed rigid vessel has fixed liquid and
gas volumes, a common temperature, and pressure from its actual gas moles only.
No external fugacity, Henry fit, hidden liquid stock or separate time loop exists.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import exp, fsum, isclose, isfinite
from typing import Any, cast

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from chemworld.physchem import solver_backend
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.aqueous_reaction_equilibrium import (
    AqueousEquilibriumSpec,
    solve_aqueous_reaction_equilibrium,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.solver_backend import ODESolverPolicy


@dataclass(frozen=True)
class AqueousGasTransfer:
    reaction_id: str
    gas_species_id: str
    aqueous_species_id: str
    coefficient_mol_s: float
    source_id: str

    def __post_init__(self) -> None:
        if (
            not all(
                (self.reaction_id, self.gas_species_id, self.aqueous_species_id, self.source_id)
            )
            or self.gas_species_id == self.aqueous_species_id
            or isinstance(self.coefficient_mol_s, bool)
            or not isfinite(self.coefficient_mol_s)
            or self.coefficient_mol_s < 0
        ):
            raise ValueError("gas transfer needs sourced identities and finite nonnegative mol/s")


@dataclass(frozen=True)
class AqueousHeadspaceSpec:
    equilibrium: AqueousEquilibriumSpec
    transfers: tuple[AqueousGasTransfer, ...]
    liquid_volume_m3: float
    gas_volume_m3: float
    temperature_K: float = 298.15
    pressure_bounds_Pa: tuple[float, float] = (1e3, 2e6)
    ode_policy: ODESolverPolicy = field(
        default_factory=lambda: ODESolverPolicy(
            "finite-aqueous-headspace-dop853-v1",
            method="DOP853",
            rtol=2e-9,
            atol=1e-12,
        )
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "transfers", tuple(self.transfers))
        object.__setattr__(self, "pressure_bounds_Pa", tuple(self.pressure_bounds_Pa))
        db = self.equilibrium.database
        if (
            not self.transfers
            or len(self.transfers) > 8
            or any(
                not isfinite(v) or v <= 0
                for v in (
                    self.liquid_volume_m3,
                    self.gas_volume_m3,
                    self.temperature_K,
                    *self.pressure_bounds_Pa,
                )
            )
            or not self.pressure_bounds_Pa[0] < self.pressure_bounds_Pa[1]
            or db.activity_model.family != "ideal"
            or any(s.phase not in {"solvent", "aqueous", "gas"} for s in db.species)
        ):
            raise ValueError("headspace requires ideal liquid/gas, fixed positive volumes/T/domain")
        water = db.solvent_species
        for p in self.pressure_bounds_Pa:
            states = db.standard_states(self.temperature_K, p)
            for species in db.species:
                state = states[species.species_id]
                if species.phase == "gas":
                    if species.charge or state.pressure_model != "gas-fugacity-reference":
                        raise ValueError("headspace gases must be neutral ideal fugacity species")
                elif state.pressure_model != "constant-volume" or (
                    state.molar_volume_m3_mol is None
                    or (species.phase == "solvent" and state.molar_volume_m3_mol <= 0)
                    or (species.phase == "aqueous" and state.molar_volume_m3_mol != 0)
                ):
                    raise ValueError("fixed water needs constant volume and zero solute volumes")
        declared_species = {s.species_id: s for s in db.species}
        reactions = {r.reaction_id: r for r in db.reactions}
        sources = {s.source_id for s in db.sources}
        if any(
            water.species_id in reactions[r].stoichiometry for r in self.equilibrium.reaction_ids
        ):
            raise ValueError("fixed nonvolatile water cannot participate in selected reactions")
        if any(
            len({getattr(t, field) for t in self.transfers}) != len(self.transfers)
            for field in ("reaction_id", "gas_species_id", "aqueous_species_id")
        ):
            raise ValueError("each transfer requires distinct gas, liquid and reaction identities")
        for transfer in self.transfers:
            gas, liquid = (
                declared_species.get(transfer.gas_species_id),
                declared_species.get(transfer.aqueous_species_id),
            )
            reaction = reactions.get(transfer.reaction_id)
            if (
                gas is None
                or gas.phase != "gas"
                or liquid is None
                or liquid.phase != "aqueous"
                or liquid.charge
                or transfer.source_id not in sources
                or reaction is None
                or dict(reaction.stoichiometry) != {gas.species_id: -1, liquid.species_id: 1}
            ):
                raise ValueError("transfer must match a declared neutral gas -> liquid reaction")
        if self.ode_policy.method not in {"DOP853", "RK45"}:
            raise ValueError("finite headspace uses an explicit original ODE policy")

    @property
    def liquid_species_ids(self) -> tuple[str, ...]:
        return tuple(s.species_id for s in self.equilibrium.database.species if s.phase != "gas")

    @property
    def gas_species_ids(self) -> tuple[str, ...]:
        return tuple(s.species_id for s in self.equilibrium.database.species if s.phase == "gas")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                {
                    "model": "finite-aqueous-headspace-v1",
                    "equilibrium": self.equilibrium.to_dict(),
                    "transfers": [asdict(t) for t in self.transfers],
                    "liquid_volume_m3": self.liquid_volume_m3,
                    "gas_volume_m3": self.gas_volume_m3,
                    "temperature_K": self.temperature_K,
                    "pressure_bounds_Pa": self.pressure_bounds_Pa,
                    "ode_policy": asdict(self.ode_policy),
                },
                allow_nan=False,
            )
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AqueousHeadspaceSpec:
        data = dict(raw)
        if data.pop("model") != "finite-aqueous-headspace-v1":
            raise ValueError("unsupported aqueous headspace schema")
        data["equilibrium"] = AqueousEquilibriumSpec.from_dict(data["equilibrium"])
        data["transfers"] = tuple(AqueousGasTransfer(**t) for t in data["transfers"])
        data["ode_policy"]["event_names"] = tuple(data["ode_policy"]["event_names"])
        data["ode_policy"] = ODESolverPolicy(**data["ode_policy"])
        return cls(**data)


def headspace_pressure(spec: AqueousHeadspaceSpec, amounts_mol: Mapping[str, float]) -> float:
    gas = [amounts_mol[k] for k in spec.gas_species_ids]
    if any(not isfinite(n) or n < 0 for n in gas):
        raise ValueError("headspace gas inventory must be finite and nonnegative")
    pressure = fsum(gas) * R * spec.temperature_K / spec.gas_volume_m3
    if not spec.pressure_bounds_Pa[0] <= pressure <= spec.pressure_bounds_Pa[1]:
        raise ValueError("actual gas pressure outside declared positive domain; no gas seeding")
    return pressure


def _inventory(spec: AqueousHeadspaceSpec, amounts: Mapping[str, float]) -> dict[str, float]:
    db = spec.equilibrium.database
    if set(amounts) != {s.species_id for s in db.species} or any(
        isinstance(n, bool) or not isfinite(n) or n < 0 for n in amounts.values()
    ):
        raise ValueError("headspace requires all original finite nonnegative species")
    p = headspace_pressure(spec, amounts)
    liquid = {k: amounts[k] for k in spec.liquid_species_ids}
    props = aqueous_phase_properties(db, liquid, temperature_K=spec.temperature_K, pressure_Pa=p)
    if not isclose(props.volume_m3, spec.liquid_volume_m3, rel_tol=1e-10, abs_tol=1e-15):
        raise ValueError("actual water volume differs from declared finite liquid geometry")
    charge = fsum(s.charge * amounts[s.species_id] for s in db.species)
    scale = fsum(abs(s.charge) * amounts[s.species_id] for s in db.species)
    if abs(charge) > 1e-10 * max(scale, amounts[db.solvent_species.species_id] * 1e-14):
        raise ValueError("headspace liquid must be electroneutral without counterion repair")
    return dict(amounts)


def prepare_aqueous_headspace_state(
    spec: AqueousHeadspaceSpec,
    amounts_mol: Mapping[str, float],
) -> dict[str, float]:
    original = _inventory(spec, amounts_mol)
    if not spec.equilibrium.reaction_ids:
        return original
    solved = solve_aqueous_reaction_equilibrium(
        spec.equilibrium,
        {k: original[k] for k in spec.liquid_species_ids},
        temperature_K=spec.temperature_K,
        pressure_Pa=headspace_pressure(spec, original),
    )
    if not solved.success or solved.amounts_mol is None:
        raise ValueError("headspace fast aqueous closure failed: " + solved.failure_reason)
    # The selected reactions have exactly zero water stoichiometry. Water is a
    # caller-owned fixed coordinate, not a logarithmic root reconstruction.
    water = spec.equilibrium.database.solvent_species.species_id
    return _inventory(spec, {**original, **solved.amounts_mol, water: original[water]})


def headspace_properties(
    spec: AqueousHeadspaceSpec,
    amounts_mol: Mapping[str, float],
) -> dict[str, Any]:
    amounts = _inventory(spec, amounts_mol)
    db, temperature = spec.equilibrium.database, spec.temperature_K
    pressure = headspace_pressure(spec, amounts)
    fugacity = {k: amounts[k] * R * temperature / spec.gas_volume_m3 for k in spec.gas_species_ids}
    chemical = db.chemical_potentials(
        amounts,
        temperature_K=temperature,
        pressure_Pa=pressure,
        gas_fugacities_Pa=fugacity,
    )
    liquid = aqueous_phase_properties(
        db,
        {k: amounts[k] for k in spec.liquid_species_ids},
        temperature_K=temperature,
        pressure_Pa=pressure,
    )
    hgas = fsum(
        amounts[k] * chemical.standard_states[k].enthalpy_J_mol for k in spec.gas_species_ids
    )
    ugas = hgas - pressure * spec.gas_volume_m3
    ggas = fsum(
        amounts[k] * cast(float, chemical.chemical_potentials_J_mol[k])
        for k in spec.gas_species_ids
        if amounts[k] > 0
    )
    return {
        "pressure_Pa": pressure,
        "partial_pressures_Pa": fugacity,
        "liquid": liquid.to_dict(),
        "gas": {
            "volume_m3": spec.gas_volume_m3,
            "enthalpy_J": hgas,
            "internal_energy_J": ugas,
            "gibbs_energy_J": ggas,
            "entropy_J_K": (hgas - ggas) / temperature,
        },
        "internal_energy_J": liquid.internal_energy_J + ugas,
        "enthalpy_J": liquid.enthalpy_J + hgas,
        "volume_m3": spec.liquid_volume_m3 + spec.gas_volume_m3,
    }


def headspace_transfer_rates(
    spec: AqueousHeadspaceSpec,
    amounts_mol: Mapping[str, float],
) -> dict[str, dict[str, float | None]]:
    db, temperature = spec.equilibrium.database, spec.temperature_K
    pressure = headspace_pressure(spec, amounts_mol)
    fugacity = {
        k: amounts_mol[k] * R * temperature / spec.gas_volume_m3 for k in spec.gas_species_ids
    }
    chemical = db.chemical_potentials(
        amounts_mol,
        temperature_K=temperature,
        pressure_Pa=pressure,
        gas_fugacities_Pa=fugacity,
    )
    result = {}
    for transfer in spec.transfers:
        gas, liquid = transfer.gas_species_id, transfer.aqueous_species_id
        standard = chemical.standard_states
        log_k = (standard[gas].gibbs_J_mol - standard[liquid].gibbs_J_mol) / (R * temperature)
        equilibrium_constant = exp(log_k)
        ag = fugacity[gas] / standard[gas].reference_pressure_Pa
        log_aq = chemical.log_activities[liquid]
        aq = 0.0 if log_aq is None else exp(log_aq)
        forward = transfer.coefficient_mol_s * ag
        reverse = transfer.coefficient_mol_s * aq / equilibrium_constant
        mug, mul = (
            chemical.chemical_potentials_J_mol[gas],
            chemical.chemical_potentials_J_mol[liquid],
        )
        result[transfer.reaction_id] = {
            "forward_mol_s": forward,
            "reverse_mol_s": reverse,
            "net_mol_s": forward - reverse,
            "gas_activity": ag,
            "aqueous_activity": aq,
            "equilibrium_constant": equilibrium_constant,
            "reaction_gibbs_J_mol": None if mug is None or mul is None else mul - mug,
        }
    return result


@dataclass(frozen=True)
class AqueousHeadspaceResult:
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


def advance_aqueous_headspace(
    spec: AqueousHeadspaceSpec,
    amounts_mol: Mapping[str, float],
    duration_s: float,
    *,
    evaluation_times_s: Sequence[float] = (),
) -> AqueousHeadspaceResult:
    if not isfinite(duration_s) or duration_s < 0:
        raise ValueError("headspace duration must be finite and nonnegative")
    times = tuple(sorted({0.0, duration_s, *map(float, evaluation_times_s)}))
    if any(not isfinite(t) or t < 0 or t > duration_s for t in times):
        raise ValueError("headspace observation outside requested duration")
    initial = prepare_aqueous_headspace_state(spec, amounts_mol)
    if any(not isclose(initial[k], n, rel_tol=1e-8, abs_tol=1e-13) for k, n in amounts_mol.items()):
        raise ValueError("headspace initial liquid must already satisfy fast equilibrium")
    db = spec.equilibrium.database
    water = db.solvent_species.species_id
    solutes = tuple(k for k in spec.liquid_species_ids if k != water)
    n0 = np.array([initial[k] for k in solutes])
    reactions = {r.reaction_id: r for r in db.reactions}
    fast = np.array(
        [
            [reactions[r].stoichiometry.get(k, 0) for k in solutes]
            for r in spec.equilibrium.reaction_ids
        ],
        dtype=float,
    ).reshape((-1, len(solutes)))
    invariants = null_space(fast, rcond=1e-12).T
    projection = np.array(
        [[float(k == t.aqueous_species_id) for t in spec.transfers] for k in solutes]
    )
    scale = max(
        fsum(initial[k] for k in solutes), fsum(initial[t.gas_species_id] for t in spec.transfers)
    )
    # Purely numerical extent scaling. No amount is added to either phase.
    if scale == 0:
        scale = initial[water]
    counters = {"rhs_evaluations": 0, "fast_closures": 0, "feasible_initializations": 0}
    last_progress = time.monotonic()

    def close(y: np.ndarray) -> dict[str, float]:
        extent = y * scale
        current = dict(initial)
        for transfer, x in zip(spec.transfers, extent, strict=True):
            current[transfer.gas_species_id] -= float(x)
        headspace_pressure(spec, current)
        feed = n0 + projection @ extent
        if np.any(feed < 0):
            target = invariants @ feed
            row_scale = np.maximum(abs(invariants) @ abs(n0), scale * 1e-12)
            feasible = linprog(
                np.zeros(len(solutes)),
                A_eq=invariants / row_scale[:, None],
                b_eq=target / row_scale,
                bounds=(0, None),
                method="highs",
            )
            counters["feasible_initializations"] += 1
            if not feasible.success:
                raise ValueError("no nonnegative liquid on transported reaction invariants")
            feed = feasible.x
        current.update(dict(zip(solutes, map(float, feed), strict=True)))
        counters["fast_closures"] += bool(spec.equilibrium.reaction_ids)
        return prepare_aqueous_headspace_state(spec, current)

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        nonlocal last_progress
        counters["rhs_evaluations"] += 1
        rates = headspace_transfer_rates(spec, close(y))
        if time.monotonic() - last_progress >= 30:
            print(
                f"aqueous headspace t={t:.6g}/{duration_s:g}s "
                f"rhs={counters['rhs_evaluations']} fast={counters['fast_closures']}",
                flush=True,
            )
            last_progress = time.monotonic()
        return np.array([rates[t.reaction_id]["net_mol_s"] for t in spec.transfers]) / scale

    y0 = np.zeros(len(spec.transfers))
    if duration_s:
        solved = solver_backend.solve_ode(
            rhs, y0, time_span_s=(0, duration_s), evaluation_times_s=times, policy=spec.ode_policy
        )
        if not solved.raw_result.success or len(solved.raw_result.t) != len(times):
            raise ValueError("headspace ODE failed: " + solved.raw_result.message)
        coordinates, diagnostic = solved.raw_result.y.T, solved.diagnostic.to_dict()
    else:
        coordinates, diagnostic = np.array([y0]), {"success": True, "nfev": 0}
    before = headspace_properties(spec, initial)
    trajectory = []
    for t, y in zip(times, coordinates, strict=True):
        amounts = close(y)
        props = headspace_properties(spec, amounts)
        trajectory.append(
            {
                "time_s": float(t),
                "amounts_mol": amounts,
                "properties": props,
                "transfers": headspace_transfer_rates(spec, amounts),
                "bath_heat_J": props["internal_energy_J"] - before["internal_energy_J"],
                "enthalpy_change_J": props["enthalpy_J"] - before["enthalpy_J"],
                "boundary_work_J": 0.0,
            }
        )
    return AqueousHeadspaceResult(
        tuple(trajectory),
        duration_s,
        {
            **counters,
            "solver": diagnostic,
            "energy_scope": "rigid isothermal closed vessel: Q=delta U, W=0; variable actual gas P",
            "solvent_scope": "declared nonvolatile water, fixed liquid volume; no gas seeding",
            "coordinate_scope": (
                "temporary transfer extents; original phase moles own persistent stock"
            ),
        },
    )
