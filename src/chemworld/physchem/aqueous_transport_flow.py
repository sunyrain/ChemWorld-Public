"""Finite kinematic throughflow for the original aqueous spatial owner.

All vessels share T/P; no pump, pressure-drop or shaft-work law is implied.
Reservoirs are fast-mixing equilibrium collectors; slow reactions belong to the
declared column cells. Water, solutes and their source enthalpies travel together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum, isfinite
from typing import Any

from chemworld.physchem.aqueous_phase_properties import (
    AqueousPhaseProperties,
    aqueous_phase_properties,
)


@dataclass(frozen=True)
class AqueousColumnFlowSpec:
    feed_vessel_id: str
    collector_vessel_id: str
    flow_m3_s: float
    dispersion_m2_s: float
    feed_capacity_m3: float
    collector_capacity_m3: float
    minimum_feed_volume_m3: float
    source_id: str

    def __post_init__(self) -> None:
        if (
            not all((self.feed_vessel_id, self.collector_vessel_id, self.source_id))
            or self.feed_vessel_id == self.collector_vessel_id
        ):
            raise ValueError("flow needs distinct finite reservoirs and a declared source")
        values = (
            self.flow_m3_s,
            self.dispersion_m2_s,
            self.feed_capacity_m3,
            self.collector_capacity_m3,
            self.minimum_feed_volume_m3,
        )
        if (
            any(isinstance(v, bool) or not isfinite(v) for v in values)
            or self.flow_m3_s < 0
            or self.dispersion_m2_s < 0
            or self.collector_capacity_m3 <= 0
            or not 0 < self.minimum_feed_volume_m3 < self.feed_capacity_m3
        ):
            raise ValueError(
                "flow requires nonnegative q/D, finite capacities and positive feed reserve"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inventory_properties(spec: Any, amounts: Any) -> AqueousPhaseProperties:
    """The exact empty reservoir has zero extensive properties, no concentration."""
    if amounts and all(value == 0 for value in amounts.values()):
        return AqueousPhaseProperties(spec.temperature_K, spec.pressure_Pa, 0.0, 0.0, 0.0, 0.0, 0.0)
    return aqueous_phase_properties(
        spec.equilibrium.database,
        amounts,
        temperature_K=spec.temperature_K,
        pressure_Pa=spec.pressure_Pa,
    )


def water_molar_volume(spec: Any) -> float:
    water = spec.equilibrium.database.solvent_species
    return float(water.standard_state.molar_volume_m3_mol)


def flow_horizon(spec: Any, amounts: Any, requested: float) -> tuple[float, tuple[str, ...]]:
    """Exact constant-q event elimination, from the actual finite water stocks."""
    flow = spec.flow
    if flow is None or flow.flow_m3_s == 0:
        return requested, ()
    water, vw = spec.equilibrium.database.solvent_species.species_id, water_molar_volume(spec)
    bounds = {
        "aqueous_feed_reserve": (
            amounts[flow.feed_vessel_id][water] * vw - flow.minimum_feed_volume_m3
        )
        / flow.flow_m3_s,
        "aqueous_collector_capacity": (
            flow.collector_capacity_m3 - amounts[flow.collector_vessel_id][water] * vw
        )
        / flow.flow_m3_s,
    }
    horizon = min(bounds.values())
    if horizon < -1e-10:
        raise ValueError("finite aqueous flow starts beyond its volume boundary")
    horizon = max(horizon, 0.0)
    if horizon > requested:
        return requested, ()
    return horizon, tuple(k for k, value in bounds.items() if abs(value - horizon) <= 1e-10)


def water_at(spec: Any, initial: Any, time_s: float) -> dict[str, float]:
    water = spec.equilibrium.database.solvent_species.species_id
    result = {v: n[water] for v, n in initial.items()}
    if spec.flow is not None:
        transferred = spec.flow.flow_m3_s * time_s / water_molar_volume(spec)
        result[spec.flow.feed_vessel_id] -= transferred
        result[spec.flow.collector_vessel_id] += transferred
    return result


def add_flow_faces(spec: Any, amounts: Any, molecular_faces: Any) -> dict[str, Any]:
    """Add conservative upwind water/solute advection and common dispersion.

    The inlet's declared total flux is q*Cfeed; outlet has no dispersive return.
    Reservoir concentrations never enter the molecular or dispersion mobility.
    """
    flow, db = spec.flow, spec.equilibrium.database
    standards = db.standard_states(spec.temperature_K, spec.pressure_Pa)
    ids, water = tuple(s.species_id for s in db.species), db.solvent_species.species_id
    charges = {s.species_id: s.charge for s in db.species}
    q, vw = flow.flow_m3_s, water_molar_volume(spec)

    def advective(source: str) -> dict[str, float]:
        volume = amounts[source][water] * vw
        if volume <= 0:
            raise ValueError("a finite upstream liquid volume is required")
        return {k: q * amounts[source][k] / volume for k in ids}

    def row(
        source: str, target: str, adv: Any, dispersion: Any, molecular: Any, original: Any = None
    ) -> dict[str, Any]:
        total = {k: adv[k] + dispersion[k] + molecular[k] for k in ids}
        return {
            **(original or {}),
            "source": source,
            "target": target,
            "advection_amount_rates_mol_s": adv,
            "dispersion_amount_rates_mol_s": dispersion,
            "electrodiffusion_amount_rates_mol_s": molecular,
            "amount_rates_mol_s": total,
            "enthalpy_W": fsum(standards[k].enthalpy_J_mol * total[k] for k in ids),
            "current_molar_charge_rate_mol_s": fsum(charges[k] * total[k] for k in ids),
            "mechanical_model": "same-pressure kinematic flow; no pressure-drop or pump law",
        }

    zero = dict.fromkeys(ids, 0.0)
    faces = {
        "aqueous-inlet": row(
            flow.feed_vessel_id,
            spec.cell_vessel_ids[0],
            advective(flow.feed_vessel_id),
            zero.copy(),
            zero.copy(),
        )
    }
    for i, (name, original) in enumerate(molecular_faces.items()):
        left, right = original["source"], original["target"]
        dx = float(spec.mesh.centers_m[i + 1] - spec.mesh.centers_m[i])
        volumes = spec.mesh.cell_volumes_m3
        dispersion = {
            k: -flow.dispersion_m2_s
            * spec.mesh.area_m2
            / dx
            * (amounts[right][k] / volumes[i + 1] - amounts[left][k] / volumes[i])
            for k in ids
        }
        # Equal solvent concentrations imply exactly zero mechanical water dispersion.
        dispersion[water] = 0.0
        molecular = {water: 0.0, **original["amount_rates_mol_s"]}
        faces[name] = row(left, right, advective(left), dispersion, molecular, original)
    faces["aqueous-outlet"] = row(
        spec.cell_vessel_ids[-1],
        flow.collector_vessel_id,
        advective(spec.cell_vessel_ids[-1]),
        zero.copy(),
        zero.copy(),
    )
    return {
        "faces": faces,
        "boundary_amount_rates_mol_s": {
            "left": faces["aqueous-inlet"]["amount_rates_mol_s"],
            "right": faces["aqueous-outlet"]["amount_rates_mol_s"],
        },
        "boundary_sign_convention": (
            "left enters column; right leaves column; finite tanks included"
        ),
    }
