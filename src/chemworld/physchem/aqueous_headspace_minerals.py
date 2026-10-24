"""One original ODE for finite gas transfer and seeded mineral surface rates.

Fast liquid reactions close their shared invariants. Actual solid volume removes
headspace; the rigid isothermal bath supplies delta U, not delta H. Material moles
and surface references remain caller-owned. No separate mineral/gas time loops.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from math import exp, expm1, fsum, isclose, isfinite
from typing import Any, cast

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from chemworld.physchem import solver_backend
from chemworld.physchem.aqueous_gas_transfer import AqueousGasTransfer, AqueousHeadspaceResult
from chemworld.physchem.aqueous_mineral_kinetics import MineralRateLaw
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.aqueous_reaction_equilibrium import (
    AqueousEquilibriumSpec,
    solve_aqueous_reaction_equilibrium,
)
from chemworld.physchem.mineral_kinetics import MineralGeometry
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.solver_backend import ODESolverPolicy


@dataclass(frozen=True)
class AqueousHeadspaceMineralSpec:
    equilibrium: AqueousEquilibriumSpec
    transfers: tuple[AqueousGasTransfer, ...]
    mineral_reaction_ids: tuple[str, ...]
    mineral_rates: tuple[MineralRateLaw, ...]
    liquid_volume_m3: float
    container_volume_m3: float
    temperature_K: float = 298.15
    pressure_bounds_Pa: tuple[float, float] = (1e3, 2e6)
    ode_policy: ODESolverPolicy = field(
        default_factory=lambda: ODESolverPolicy(
            "finite-headspace-mineral-radius-dop853-v1",
            method="DOP853",
            rtol=2e-9,
            atol=1e-12,
            dense_output=True,
        )
    )

    def __post_init__(self) -> None:
        for name in ("transfers", "mineral_reaction_ids", "mineral_rates", "pressure_bounds_Pa"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        db = self.equilibrium.database
        if (
            len(self.pressure_bounds_Pa) != 2
            or any(
                isinstance(v, bool) or not isfinite(v) or v <= 0
                for v in (
                    self.liquid_volume_m3,
                    self.container_volume_m3,
                    self.temperature_K,
                    *self.pressure_bounds_Pa,
                )
            )
            or self.container_volume_m3 <= self.liquid_volume_m3
            or not self.pressure_bounds_Pa[0] < self.pressure_bounds_Pa[1]
            or db.activity_model.family != "ideal"
            or not 1 <= len(self.transfers) <= 2
            or not 1 <= len(self.mineral_ids) <= 2
        ):
            raise ValueError(
                "headspace minerals require ideal phases, 1-2 transfers/minerals, V/T/P"
            )
        species = {s.species_id: s for s in db.species}
        reactions = {r.reaction_id: r for r in db.reactions}
        sources = {s.source_id for s in db.sources}
        water = db.solvent_species.species_id
        for pressure in self.pressure_bounds_Pa:
            states = db.standard_states(self.temperature_K, pressure)
            for key, s in species.items():
                standard = states[key]
                if s.phase == "gas":
                    if s.charge or standard.pressure_model != "gas-fugacity-reference":
                        raise ValueError("headspace gas requires neutral ideal fugacity standard")
                elif (
                    standard.pressure_model != "constant-volume"
                    or standard.molar_volume_m3_mol is None
                    or (s.phase in {"solvent", "mineral"} and standard.molar_volume_m3_mol <= 0)
                    or (s.phase == "aqueous" and standard.molar_volume_m3_mol != 0)
                ):
                    raise ValueError(
                        "positive fixed water/mineral volumes and zero solute volumes required"
                    )
        if any(water in reactions[k].stoichiometry for k in self.equilibrium.reaction_ids):
            raise ValueError("fixed nonvolatile water cannot participate in selected reactions")
        for name in ("reaction_id", "gas_species_id", "aqueous_species_id"):
            if len({getattr(t, name) for t in self.transfers}) != len(self.transfers):
                raise ValueError("transfer identities must be distinct")
        for t in self.transfers:
            gas, aq, reaction = (
                species.get(t.gas_species_id),
                species.get(t.aqueous_species_id),
                reactions.get(t.reaction_id),
            )
            if (
                gas is None
                or aq is None
                or gas.phase != "gas"
                or aq.phase != "aqueous"
                or aq.charge
                or t.source_id not in sources
                or reaction is None
                or dict(reaction.stoichiometry) != {gas.species_id: -1, aq.species_id: 1}
            ):
                raise ValueError("transfer must declare neutral gas -> liquid stoichiometry")
        laws = [r.mineral_id for r in self.mineral_rates]
        if (
            len(set(laws)) != len(laws)
            or set(laws) != set(self.mineral_ids)
            or len(set(self.mineral_reaction_ids)) != len(self.mineral_ids)
            or set(self.mineral_reaction_ids) - reactions.keys()
        ):
            raise ValueError("one rate and dissolution reaction required per original mineral")
        selected = []
        for rid in self.mineral_reaction_ids:
            stoich = reactions[rid].stoichiometry
            minerals = set(stoich) & set(self.mineral_ids)
            if (
                len(minerals) != 1
                or water in stoich
                or set(stoich) - set(self.liquid_species_ids) - minerals
            ):
                raise ValueError(
                    "dissolution selects one mineral and aqueous solutes, no water/gas"
                )
            mineral = next(iter(minerals))
            if stoich[mineral] != -1:
                raise ValueError("mineral dissolution coefficient must equal -1")
            selected.append(mineral)
        if set(selected) != set(self.mineral_ids):
            raise ValueError("each original mineral needs its own dissolution reaction")
        if self.ode_policy.method not in {"DOP853", "RK45"}:
            raise ValueError("joint headspace minerals require an explicit original ODE policy")

    @property
    def liquid_species_ids(self) -> tuple[str, ...]:
        return tuple(
            s.species_id
            for s in self.equilibrium.database.species
            if s.phase in {"solvent", "aqueous"}
        )

    @property
    def gas_species_ids(self) -> tuple[str, ...]:
        return tuple(s.species_id for s in self.equilibrium.database.species if s.phase == "gas")

    @property
    def mineral_ids(self) -> tuple[str, ...]:
        return tuple(
            s.species_id for s in self.equilibrium.database.species if s.phase == "mineral"
        )

    @property
    def dissolution_reactions(self) -> dict[str, str]:
        selected = set(self.mineral_reaction_ids)
        return {
            m: r.reaction_id
            for r in self.equilibrium.database.reactions
            if r.reaction_id in selected
            for m in self.mineral_ids
            if m in r.stoichiometry
        }

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                {
                    "model": "finite-aqueous-headspace-minerals-v2",
                    "equilibrium": self.equilibrium.to_dict(),
                    "transfers": [asdict(t) for t in self.transfers],
                    "mineral_reaction_ids": self.mineral_reaction_ids,
                    "mineral_rates": [asdict(r) for r in self.mineral_rates],
                    "liquid_volume_m3": self.liquid_volume_m3,
                    "container_volume_m3": self.container_volume_m3,
                    "temperature_K": self.temperature_K,
                    "pressure_bounds_Pa": self.pressure_bounds_Pa,
                    "ode_policy": asdict(self.ode_policy),
                },
                allow_nan=False,
            )
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AqueousHeadspaceMineralSpec:
        data = dict(raw)
        if data.pop("model") != "finite-aqueous-headspace-minerals-v2":
            raise ValueError("unsupported headspace mineral schema")
        data["equilibrium"] = AqueousEquilibriumSpec.from_dict(data["equilibrium"])
        data["transfers"] = tuple(AqueousGasTransfer(**t) for t in data["transfers"])
        data["mineral_rates"] = tuple(MineralRateLaw(**r) for r in data["mineral_rates"])
        policy = dict(data["ode_policy"])
        policy["event_names"] = tuple(policy["event_names"])
        data["ode_policy"] = ODESolverPolicy(**policy)
        return cls(**data)


def headspace_mineral_volumes(
    spec: AqueousHeadspaceMineralSpec, amounts: Mapping[str, float]
) -> dict[str, float]:
    standards = spec.equilibrium.database.standard_states(
        spec.temperature_K, spec.pressure_bounds_Pa[0]
    )
    volumes = {
        m: amounts[m] * cast(float, standards[m].molar_volume_m3_mol) for m in spec.mineral_ids
    }
    gas = spec.container_volume_m3 - spec.liquid_volume_m3 - fsum(volumes.values())
    if any(not isfinite(v) or v < 0 for v in volumes.values()) or not isfinite(gas) or gas <= 0:
        raise ValueError("actual solids leave no positive finite headspace")
    return {
        "aqueous": spec.liquid_volume_m3,
        "headspace": gas,
        **{"mineral:" + m: v for m, v in volumes.items()},
    }


def headspace_mineral_pressure(
    spec: AqueousHeadspaceMineralSpec, amounts: Mapping[str, float]
) -> float:
    gas = [amounts[k] for k in spec.gas_species_ids]
    if any(not isfinite(n) or n < 0 for n in gas):
        raise ValueError("headspace gas inventory must be finite and nonnegative")
    pressure = (
        fsum(gas) * R * spec.temperature_K / headspace_mineral_volumes(spec, amounts)["headspace"]
    )
    if not spec.pressure_bounds_Pa[0] <= pressure <= spec.pressure_bounds_Pa[1]:
        raise ValueError("actual gas pressure outside declared positive domain; no gas seeding")
    return pressure


def _inventory(spec: AqueousHeadspaceMineralSpec, amounts: Mapping[str, float]) -> dict[str, float]:
    db = spec.equilibrium.database
    if set(amounts) != {s.species_id for s in db.species} or any(
        isinstance(n, bool) or not isfinite(n) or n < 0 for n in amounts.values()
    ):
        raise ValueError("headspace minerals require all original finite nonnegative species")
    p = headspace_mineral_pressure(spec, amounts)
    liquid = aqueous_phase_properties(
        db,
        {k: amounts[k] for k in spec.liquid_species_ids},
        temperature_K=spec.temperature_K,
        pressure_Pa=p,
    )
    if not isclose(liquid.volume_m3, spec.liquid_volume_m3, rel_tol=1e-10, abs_tol=1e-15):
        raise ValueError("actual water volume differs from declared finite liquid geometry")
    charge = fsum(s.charge * amounts[s.species_id] for s in db.species)
    scale = fsum(abs(s.charge) * amounts[s.species_id] for s in db.species)
    if abs(charge) > 1e-10 * max(scale, amounts[db.solvent_species.species_id] * 1e-14):
        raise ValueError("headspace liquid must be electroneutral without counterion repair")
    return dict(amounts)


def prepare_aqueous_headspace_mineral_state(
    spec: AqueousHeadspaceMineralSpec, amounts_mol: Mapping[str, float]
) -> dict[str, float]:
    original = _inventory(spec, amounts_mol)
    if not spec.equilibrium.reaction_ids:
        return original
    solved = solve_aqueous_reaction_equilibrium(
        spec.equilibrium,
        {k: original[k] for k in spec.liquid_species_ids},
        temperature_K=spec.temperature_K,
        pressure_Pa=headspace_mineral_pressure(spec, original),
    )
    if not solved.success or solved.amounts_mol is None:
        raise ValueError("headspace mineral fast closure failed: " + solved.failure_reason)
    water = spec.equilibrium.database.solvent_species.species_id
    return _inventory(spec, {**original, **solved.amounts_mol, water: original[water]})


def headspace_mineral_properties(
    spec: AqueousHeadspaceMineralSpec, amounts_mol: Mapping[str, float]
) -> dict[str, Any]:
    amounts = _inventory(spec, amounts_mol)
    db, temperature = spec.equilibrium.database, spec.temperature_K
    pressure = headspace_mineral_pressure(spec, amounts)
    volumes = headspace_mineral_volumes(spec, amounts)
    fugacity = {
        k: amounts[k] * R * temperature / volumes["headspace"] for k in spec.gas_species_ids
    }
    chemical = db.chemical_potentials(
        amounts, temperature_K=temperature, pressure_Pa=pressure, gas_fugacities_Pa=fugacity
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
    ugas = hgas - pressure * volumes["headspace"]
    ggas = fsum(
        amounts[k] * cast(float, chemical.chemical_potentials_J_mol[k])
        for k in spec.gas_species_ids
        if amounts[k] > 0
    )
    minerals = {}
    for m in spec.mineral_ids:
        h = amounts[m] * chemical.standard_states[m].enthalpy_J_mol
        g = amounts[m] * chemical.standard_states[m].gibbs_J_mol
        minerals[m] = {
            "volume_m3": volumes["mineral:" + m],
            "enthalpy_J": h,
            "internal_energy_J": h - pressure * volumes["mineral:" + m],
            "gibbs_energy_J": g,
            "entropy_J_K": (h - g) / temperature,
        }
    return {
        "pressure_Pa": pressure,
        "partial_pressures_Pa": fugacity,
        "liquid": liquid.to_dict(),
        "gas": {
            "volume_m3": volumes["headspace"],
            "enthalpy_J": hgas,
            "internal_energy_J": ugas,
            "gibbs_energy_J": ggas,
            "entropy_J_K": (hgas - ggas) / temperature,
        },
        "minerals": minerals,
        "internal_energy_J": liquid.internal_energy_J
        + ugas
        + fsum(p["internal_energy_J"] for p in minerals.values()),
        "enthalpy_J": liquid.enthalpy_J + hgas + fsum(p["enthalpy_J"] for p in minerals.values()),
        "volume_m3": spec.container_volume_m3,
    }


def headspace_mineral_rates(
    spec: AqueousHeadspaceMineralSpec,
    amounts: Mapping[str, float],
    geometries: Mapping[str, MineralGeometry],
) -> dict[str, Any]:
    db, temperature = spec.equilibrium.database, spec.temperature_K
    pressure = headspace_mineral_pressure(spec, amounts)
    volume = headspace_mineral_volumes(spec, amounts)["headspace"]
    fugacity = {k: amounts[k] * R * temperature / volume for k in spec.gas_species_ids}
    chemical = db.chemical_potentials(
        amounts, temperature_K=temperature, pressure_Pa=pressure, gas_fugacities_Pa=fugacity
    )
    standards, transfers, minerals = chemical.standard_states, {}, {}
    for t in spec.transfers:
        gas, aq = t.gas_species_id, t.aqueous_species_id
        k = exp((standards[gas].gibbs_J_mol - standards[aq].gibbs_J_mol) / (R * temperature))
        ag = fugacity[gas] / standards[gas].reference_pressure_Pa
        la = chemical.log_activities[aq]
        aa = 0.0 if la is None else exp(la)
        forward, reverse = t.coefficient_mol_s * ag, t.coefficient_mol_s * aa / k
        mug, mua = chemical.chemical_potentials_J_mol[gas], chemical.chemical_potentials_J_mol[aq]
        transfers[t.reaction_id] = {
            "forward_mol_s": forward,
            "reverse_mol_s": reverse,
            "net_mol_s": forward - reverse,
            "gas_activity": ag,
            "aqueous_activity": aa,
            "equilibrium_constant": k,
            "reaction_gibbs_J_mol": None if mug is None or mua is None else mua - mug,
        }
    reactions = {r.reaction_id: r for r in db.reactions}
    laws = {r.mineral_id: r for r in spec.mineral_rates}
    for m, rid in spec.dissolution_reactions.items():
        value, absent = -standards[m].gibbs_J_mol, False
        for key, nu in reactions[rid].stoichiometry.items():
            if key == m:
                continue
            mu = chemical.chemical_potentials_J_mol[key]
            if mu is None:
                if nu < 0:
                    raise ValueError("missing aqueous reactant has no resolved mineral rate")
                absent = True
            else:
                value += nu * mu
        si = None if absent else value / (R * temperature)
        drive = 1.0 if si is None else -expm1(si)
        area = geometries[m].area(amounts[m])
        minerals[m] = {
            "surface_area_m2": area,
            "saturation_index": si,
            "drive": drive,
            "dissolution_mol_s": laws[m].coefficient(temperature) * area * drive,
        }
    return {"transfers": transfers, "minerals": minerals}


def advance_aqueous_headspace_minerals(
    spec: AqueousHeadspaceMineralSpec,
    amounts_mol: Mapping[str, float],
    geometries: Mapping[str, MineralGeometry],
    duration_s: float,
    *,
    evaluation_times_s: Sequence[float] = (),
    maximum_pressure_Pa: float | None = None,
) -> AqueousHeadspaceResult:
    if isinstance(duration_s, bool) or not isfinite(duration_s) or duration_s < 0:
        raise ValueError("headspace mineral duration must be finite and nonnegative")
    times = tuple(sorted({0.0, float(duration_s), *map(float, evaluation_times_s)}))
    if any(not isfinite(t) or t < 0 or t > duration_s for t in times):
        raise ValueError("observation outside requested duration")
    if maximum_pressure_Pa is not None and (
        isinstance(maximum_pressure_Pa, bool)
        or not isfinite(maximum_pressure_Pa)
        or maximum_pressure_Pa <= 0
    ):
        raise ValueError("vessel pressure limit must be finite and positive")
    minerals, db = spec.mineral_ids, spec.equilibrium.database
    if set(geometries) != set(minerals):
        raise ValueError("every mineral requires its original material geometry")
    initial = prepare_aqueous_headspace_mineral_state(spec, amounts_mol)
    if any(not isclose(initial[k], n, rel_tol=1e-8, abs_tol=1e-13) for k, n in amounts_mol.items()):
        raise ValueError("initial liquid must satisfy fast equilibrium")
    for m in minerals:
        if initial[m] > 0 and geometries[m].reference_amount_mol <= 0:
            raise ValueError("positive solid requires a positive material reference amount")
    water = db.solvent_species.species_id
    solutes = tuple(k for k in spec.liquid_species_ids if k != water)
    reactions = {r.reaction_id: r for r in db.reactions}
    fast = np.array(
        [
            [reactions[r].stoichiometry.get(k, 0) for k in solutes]
            for r in spec.equilibrium.reaction_ids
        ],
        dtype=float,
    ).reshape((-1, len(solutes)))
    invariants = null_space(fast, rcond=1e-12).T
    transfer_projection = np.array(
        [[float(k == t.aqueous_species_id) for t in spec.transfers] for k in solutes]
    )
    mineral_projection = np.array(
        [
            [reactions[spec.dissolution_reactions[m]].stoichiometry.get(k, 0) for m in minerals]
            for k in solutes
        ]
    )
    n0 = np.array([initial[k] for k in solutes])
    solids0 = np.array([initial[m] for m in minerals])
    scale = (
        max(fsum(n0), fsum(initial[t.gas_species_id] for t in spec.transfers), fsum(solids0))
        or initial[water]
    )
    counters = {
        "rhs_evaluations": 0,
        "fast_closures": 0,
        "feasible_initializations": 0,
        "negative_radius_trials": 0,
    }
    last_progress = time.monotonic()
    coefficients = {r.mineral_id: r.coefficient(spec.temperature_K) for r in spec.mineral_rates}
    cache: dict[tuple[float, ...], tuple[dict[str, float], dict[str, Any]]] = {}

    def close(extents: np.ndarray, stocks: np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
        signature = (*map(float, extents), *map(float, stocks))
        if signature in cache:
            return cache[signature]
        current = dict(initial)
        for t, x in zip(spec.transfers, extents, strict=True):
            current[t.gas_species_id] -= float(x)
        current.update(dict(zip(minerals, map(float, stocks), strict=True)))
        feed = n0 + transfer_projection @ extents + mineral_projection @ (solids0 - stocks)
        if np.any(feed < 0):
            row_scale = np.maximum(abs(invariants) @ abs(feed), scale * 1e-12)
            feasible = linprog(
                np.zeros(len(solutes)),
                A_eq=invariants / row_scale[:, None],
                b_eq=(invariants @ feed) / row_scale,
                bounds=(0, None),
                method="highs",
            )
            counters["feasible_initializations"] += 1
            if not feasible.success:
                raise ValueError("no nonnegative liquid on shared transfer/mineral invariants")
            feed = feasible.x
        current.update(dict(zip(solutes, map(float, feed), strict=True)))
        if maximum_pressure_Pa is not None and (
            headspace_mineral_pressure(spec, current) > maximum_pressure_Pa
        ):
            raise ValueError("actual joint headspace pressure exceeds vessel operating limit")
        counters["fast_closures"] += bool(spec.equilibrium.reaction_ids)
        current = prepare_aqueous_headspace_mineral_state(spec, current)
        rates = headspace_mineral_rates(spec, current, geometries)
        cache[signature] = current, rates
        return current, rates

    active = [
        m
        for m in minerals
        if initial[m] > 0 and geometries[m].reference_area_m2 > 0 and coefficients[m] > 0
    ]
    extents = np.zeros(len(spec.transfers))
    stocks = solids0.copy()
    start, segments, events, diagnostics = 0.0, [], [], []
    ng = len(spec.transfers)
    while start < duration_s:
        keys, baseline = tuple(active), stocks.copy()
        factors = {
            m: coefficients[m]
            * geometries[m].reference_area_m2
            / (
                3
                * geometries[m].reference_amount_mol ** (2 / 3)
                * baseline[minerals.index(m)] ** (1 / 3)
            )
            for m in keys
        }

        def decode(
            y: np.ndarray,
            *,
            accepted: bool = False,
            baseline: np.ndarray = baseline,
            keys: tuple[str, ...] = keys,
        ) -> tuple[np.ndarray, np.ndarray]:
            solid = baseline.copy()
            for j, m in enumerate(keys):
                u = float(y[ng + j])
                if u < 0:
                    if accepted:
                        raise RuntimeError("accepted mineral radius is negative")
                    counters["negative_radius_trials"] += 1
                    u = 0.0  # Only a trial extension while locating the exact event.
                solid[minerals.index(m)] = baseline[minerals.index(m)] * u**3
            return y[:ng] * scale, solid

        def rhs(
            t: float,
            y: np.ndarray,
            keys: tuple[str, ...] = keys,
            factors: dict[str, float] = factors,
            decode: Any = decode,
        ) -> np.ndarray:
            nonlocal last_progress
            counters["rhs_evaluations"] += 1
            _, rates = close(*decode(y))
            if time.monotonic() - last_progress >= 30:
                print(
                    f"joint headspace minerals t={t:.6g}/{duration_s:g}s "
                    f"rhs={counters['rhs_evaluations']} closures={counters['fast_closures']}",
                    flush=True,
                )
                last_progress = time.monotonic()
            return np.array(
                [rates["transfers"][t.reaction_id]["net_mol_s"] / scale for t in spec.transfers]
                + [-factors[m] * rates["minerals"][m]["drive"] for m in keys]
            )

        roots = []
        for j in range(len(keys)):

            def exhausted(t: float, y: np.ndarray, j: int = j) -> float:
                return float(y[ng + j])

            cast(Any, exhausted).terminal = True
            cast(Any, exhausted).direction = -1
            roots.append(exhausted)
        policy = replace(
            spec.ode_policy, dense_output=True, event_names=tuple("exhausted:" + m for m in keys)
        )
        report = solver_backend.solve_ode(
            rhs,
            np.concatenate((extents / scale, np.ones(len(keys)))),
            time_span_s=(start, duration_s),
            policy=policy,
            events=tuple(roots),
        )
        report.raise_for_failure("joint headspace mineral kinetics")
        diagnostics.append(report.diagnostic.to_dict())
        terminal = report.terminal_point
        end = float(report.t[-1]) if terminal is None else terminal[0]
        final_y = report.y[:, -1].copy() if terminal is None else terminal[1].copy()
        if terminal is not None:
            j = terminal[2]
            final_y[ng + j] = 0.0
            events.append({"event": "mineral_exhausted", "mineral_id": keys[j], "time_s": end})
            active.remove(keys[j])
        extents, stocks = decode(final_y, accepted=True)
        segments.append((start, end, report.raw_result.sol, decode))
        if end <= start:
            raise RuntimeError("joint mineral event failed to advance interval")
        start = end
    before = headspace_mineral_properties(spec, initial)
    trajectory = []
    for t in times:
        x, solid = extents, stocks
        for lo, hi, solution, decoder in segments:
            if lo <= t < hi:
                x, solid = decoder(solution(t), accepted=True)
                break
        amounts, rates = close(x, solid)
        props = headspace_mineral_properties(spec, amounts)
        trajectory.append(
            {
                "time_s": t,
                "amounts_mol": amounts,
                "properties": props,
                **rates,
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
            "solver_segments": diagnostics,
            "events": events,
            "energy_scope": (
                "rigid isothermal closed vessel: Q=delta U, external W=0; "
                "gas volume subtracts actual solids"
            ),
            "coordinate_scope": (
                "one shared ODE: gas extents and mineral radii; original phases own stock"
            ),
            "mineral_scope": "original material geometry; no spontaneous nucleation or seed repair",
        },
    )
