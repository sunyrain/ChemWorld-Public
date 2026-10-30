"""Finite axial gas/solid contact using original material and sorbent owners.

Neutral ideal gas, rigid compartments and a common isothermal bath. The native
Ergun kernel closes quasi-steady momentum; original conservative transfers,
Langmuir/LDF processes and the coupled solver advance actual inventories.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from math import fsum, isclose, isfinite
from types import MappingProxyType
from typing import Any

import numpy as np

from chemworld.foundation.state_ownership import StateSlot
from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledEvent,
    CoupledProcess,
    CoupledStateView,
    CoupledSystem,
    InternalTransfer,
    ProcessContribution,
    SpeciesInternalEnergy,
)
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.packed_beds import PackedBedSpec, packed_bed_pressure
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.residual_system import (
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)
from chemworld.physchem.solver_backend import ODESolverPolicy
from chemworld.physchem.sorbent_kinetics import (
    SorbentKineticsSpec,
    make_sorbent_slot,
    sorbent_key,
    sorbent_process,
    validate_sorbent_slot,
)
from chemworld.physchem.transport import FluidState

FIXED_BED_POLICY = ODESolverPolicy("native-fixed-bed-v1", "BDF", rtol=1e-8, atol=1e-12)


@dataclass(frozen=True)
class FixedBedSpec:
    bed_id: str
    cell_phase_ids: tuple[str, ...]
    faces_m: tuple[float, ...]
    area_m2: float
    void_fraction: float
    particle_diameter_m: float
    carrier_mass_kg: float
    sorbent: SorbentKineticsSpec
    molecular_weights_kg_mol: Mapping[str, float]
    gas_calorics: Mapping[str, SpeciesInternalEnergy]
    viscosity_Pa_s: float
    minimum_pressure_Pa: float = 20000.0
    maximum_pressure_Pa: float = 160000.0
    maximum_velocity_m_s: float = 0.5
    momentum_root: RootPolicy = field(default_factory=RootPolicy)
    inlet_resistance_Pa_s_m: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell_phase_ids", tuple(self.cell_phase_ids))
        object.__setattr__(self, "faces_m", tuple(self.faces_m))
        object.__setattr__(
            self, "molecular_weights_kg_mol", MappingProxyType(dict(self.molecular_weights_kg_mol))
        )
        object.__setattr__(self, "gas_calorics", MappingProxyType(dict(self.gas_calorics)))
        mesh = self.mesh
        if (
            not self.bed_id
            or len(self.cell_phase_ids) != mesh.cells
            or len(set(self.cell_phase_ids)) != mesh.cells
            or any(not isinstance(k, str) or not k for k in self.cell_phase_ids)
            or not self.molecular_weights_kg_mol
            or set(self.gas_calorics) != set(self.molecular_weights_kg_mol)
            or set(self.sorbent.component_ids) - set(self.gas_calorics)
        ):
            raise ValueError("fixed bed requires explicit cell, gas and sorbent identities")
        if (
            any(
                isinstance(v, bool) or not isfinite(v) or v <= 0
                for v in (
                    *self.molecular_weights_kg_mol.values(),
                    self.particle_diameter_m,
                    self.carrier_mass_kg,
                    self.viscosity_Pa_s,
                    self.minimum_pressure_Pa,
                    self.maximum_pressure_Pa,
                    self.maximum_velocity_m_s,
                )
            )
            or self.minimum_pressure_Pa >= self.maximum_pressure_Pa
            or not 0.05 < self.void_fraction < 0.95
            or any(not isinstance(c, SpeciesInternalEnergy) for c in self.gas_calorics.values())
            or not isinstance(self.momentum_root, RootPolicy)
            or isinstance(self.inlet_resistance_Pa_s_m, bool)
            or not isfinite(self.inlet_resistance_Pa_s_m)
            or self.inlet_resistance_Pa_s_m < 0
        ):
            raise ValueError("fixed bed requires positive geometry, declared calorics and limits")

    @property
    def mesh(self) -> DiffusionMesh:
        return DiffusionMesh("planar", self.faces_m, area_m2=self.area_m2)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(self.molecular_weights_kg_mol)

    @property
    def void_volumes_L(self) -> tuple[float, ...]:  # noqa: N802 - physical unit
        return tuple(float(v * self.void_fraction * 1000) for v in self.mesh.cell_volumes_m3)

    @property
    def carrier_masses_kg(self) -> tuple[float, ...]:
        volumes = self.mesh.cell_volumes_m3
        return tuple(float(self.carrier_mass_kg * v / sum(volumes)) for v in volumes)

    def slot_id(self, cell: int) -> str:
        return f"{self.bed_id}:cell:{cell}:sorbent"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "finite-isothermal-fixed-bed-v1",
            "bed_id": self.bed_id,
            "cell_phase_ids": self.cell_phase_ids,
            "faces_m": self.faces_m,
            "area_m2": self.area_m2,
            "void_fraction": self.void_fraction,
            "particle_diameter_m": self.particle_diameter_m,
            "carrier_mass_kg": self.carrier_mass_kg,
            "sorbent": self.sorbent.to_dict(),
            "molecular_weights_kg_mol": dict(self.molecular_weights_kg_mol),
            "gas_calorics": {k: asdict(c) for k, c in self.gas_calorics.items()},
            "viscosity_Pa_s": self.viscosity_Pa_s,
            "minimum_pressure_Pa": self.minimum_pressure_Pa,
            "maximum_pressure_Pa": self.maximum_pressure_Pa,
            "maximum_velocity_m_s": self.maximum_velocity_m_s,
            "momentum_root": asdict(self.momentum_root),
            "inlet_resistance_Pa_s_m": self.inlet_resistance_Pa_s_m,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FixedBedSpec:
        data = dict(raw)
        if data.pop("model", None) != "finite-isothermal-fixed-bed-v1":
            raise ValueError("unknown fixed-bed declaration")
        data["sorbent"] = SorbentKineticsSpec.from_dict(data["sorbent"])
        data["gas_calorics"] = {
            k: SpeciesInternalEnergy(**v) for k, v in data["gas_calorics"].items()
        }
        data["momentum_root"] = RootPolicy(**data["momentum_root"])
        return cls(**data)


def fixed_bed_slots(
    spec: FixedBedSpec,
    vessel_id: str,
    *,
    initial_loading_mol_kg: Sequence[Mapping[str, float]] | None = None,
) -> tuple[StateSlot, ...]:
    """Author the initial persistent lots once; advance never calls this initializer."""
    loads = (
        tuple({} for _ in spec.cell_phase_ids)
        if initial_loading_mol_kg is None
        else tuple(initial_loading_mol_kg)
    )
    if len(loads) != len(spec.cell_phase_ids):
        raise ValueError("one initial loading declaration is required per actual cell")
    slots = []
    for i, (mass, loading) in enumerate(zip(spec.carrier_masses_kg, loads, strict=True)):
        slot = make_sorbent_slot(
            spec.sorbent,
            owner_id=spec.bed_id,
            location_id=vessel_id,
            mass_kg=mass,
            initial_loading_mol_per_kg=loading,
        )
        slots.append(replace(slot, spec=replace(slot.spec, slot_id=spec.slot_id(i))))
    return tuple(slots)


def fixed_bed_gas_key(phase_id: str, component_id: str) -> str:
    return f"gas:{phase_id}:{component_id}"


def fixed_bed_pressure(
    spec: FixedBedSpec, state: CoupledStateView, phase_id: str, volume_L: float
) -> float:
    return (
        fsum(state.amounts_mol[fixed_bed_gas_key(phase_id, k)] for k in spec.component_ids)
        * R
        * state.temperature_K
        / (volume_L * 0.001)
    )


def fixed_bed_face_velocity(
    spec: FixedBedSpec,
    *,
    distance_m: float,
    density_kg_m3: float,
    pressure_difference_Pa: float,
    series_resistance_Pa_s_m: float = 0.0,
) -> tuple[float, dict[str, Any]]:
    """Invert Ergun plus a declared linear series valve: deltaP = Ergun(u) + R*u.

    Resistance uses superficial bed velocity and is independent of mesh spacing.
    It is bidirectional, has no storage, and zero preserves the original law.
    """
    if (isinstance(series_resistance_Pa_s_m, bool)
            or not isfinite(series_resistance_Pa_s_m) or series_resistance_Pa_s_m < 0):
        raise ValueError("series resistance must be finite and nonnegative")
    if not isfinite(pressure_difference_Pa):
        raise ValueError("face pressure difference must be finite")
    if pressure_difference_Pa == 0:
        return 0.0, {"success": True, "iterations": 0}
    bed = PackedBedSpec("Ergun", distance_m, spec.particle_diameter_m, spec.void_fraction)
    fluid = FluidState(density_kg_m3, spec.viscosity_Pa_s)
    pressure = abs(pressure_difference_Pa)
    scale = max(pressure, 1.0)

    def forward(velocity: float) -> float:
        return (packed_bed_pressure(bed, fluid,
                                   superficial_velocity_m_s=velocity).pressure_drop_Pa
                + series_resistance_Pa_s_m * velocity)

    def residual(point: Any) -> Any:
        return np.array([(forward(float(point[0])) - pressure) / scale])

    seed = pressure / forward(1.0)
    root = solve_root(
        residual,
        [seed],
        jacobian=lambda z: finite_jacobian(residual, z, [max(seed, 0.01)], lower=np.array([0.0])),
        variables=(
            ResidualVariable("superficial_velocity", "m/s", scale=max(seed, 0.01), lower=0.0),
        ),
        residual_scales=[1.0],
        policy=spec.momentum_root,
    )
    if not root.success:
        raise ValueError("fixed-bed momentum root failed: " + str(root.diagnostics))
    return float(root.value[0]) * (1 if pressure_difference_Pa > 0 else -1), dict(root.diagnostics)


def build_fixed_bed_system(
    spec: FixedBedSpec,
    phase_volumes_L: Mapping[str, float],
    slots: Sequence[StateSlot],
    *,
    inlet_phase_id: str | None,
    outlet_phase_id: str | None,
    policy: ODESolverPolicy = FIXED_BED_POLICY,
    maximum_pressure_Pa: float | None = None,
    _phase_temperature: Callable[[CoupledStateView, str], float] | None = None,
    _solid_temperature: Callable[[CoupledStateView, str], float] | None = None,
) -> tuple[CoupledSystem, dict[str, Any]]:
    """Compile selected finite compartments and original sorbent lots, without a clock."""
    phases = tuple(phase_volumes_L)
    cells = spec.cell_phase_ids
    expected = {
        *cells,
        *(() if inlet_phase_id is None else (inlet_phase_id,)),
        *(() if outlet_phase_id is None else (outlet_phase_id,)),
    }
    if (
        set(phases) != expected
        or len(expected)
        != len(cells) + int(inlet_phase_id is not None) + int(outlet_phase_id is not None)
        or any(not isfinite(v) or v <= 0 for v in phase_volumes_L.values())
        or len(slots) != len(cells)
    ):
        raise ValueError("fixed-bed boundaries/cells must be distinct finite owned compartments")
    for i, (phase, volume, slot) in enumerate(zip(cells, spec.void_volumes_L, slots, strict=True)):
        validate_sorbent_slot(slot, spec.sorbent)
        if (
            not isclose(phase_volumes_L[phase], volume, rel_tol=1e-12)
            or slot.spec.slot_id != spec.slot_id(i)
            or slot.spec.owner_id != spec.bed_id
            or slot.spec.owner_kind != "equipment"
            or not isclose(slot.values["mass_kg"], spec.carrier_masses_kg[i], rel_tol=1e-12)
        ):
            raise ValueError("bed cell volume or persistent carrier ownership disagrees")
    keys = {
        phase: tuple(fixed_bed_gas_key(phase, k) for k in spec.component_ids) for phase in phases
    }
    flat = tuple(key for row in keys.values() for key in row)
    if len(set(flat)) != len(flat):
        raise ValueError("gas key identities collide")
    calorics = {
        fixed_bed_gas_key(phase, k): law for phase in phases for k, law in spec.gas_calorics.items()
    }
    for slot in slots:
        for k in spec.sorbent.component_ids:
            gas = spec.gas_calorics[k]
            calorics[sorbent_key(slot, k)] = SpeciesInternalEnergy(
                gas.u_ref_J_mol + spec.sorbent.adsorption_energy_J_mol[k], gas.cv_J_mol_K
            )
    thermal = ConstantCvThermodynamics(
        calorics,
        fsum(s.values["heat_capacity_J_K"] for s in slots),
        spec.sorbent.reference_temperature_K,
    )

    def localized(base: CoupledProcess, volume: float, phase: str) -> CoupledProcess:
        def local(state: CoupledStateView) -> CoupledStateView:
            return replace(state, volume_L=volume, temperature_K=state.temperature_K
                           if _solid_temperature is None else _solid_temperature(state, phase))

        def evaluate(state: CoupledStateView) -> ProcessContribution:
            return base.evaluate(local(state))

        def validate(state: CoupledStateView) -> None:
            if not (base.validate_state is not None):
                raise ValueError("fixed-bed process requires a state validator")
            base.validate_state(local(state))

        return CoupledProcess(
            base.process_id,
            evaluate,
            validate_state=validate,
            inventory_dependencies=base.inventory_dependencies,
        )

    processes = []
    for i, slot in enumerate(slots):
        base = sorbent_process(
            slot,
            spec.sorbent,
            thermodynamics=thermal,
            fluid_keys={k: fixed_bed_gas_key(cells[i], k) for k in spec.sorbent.component_ids},
        )
        volume = phase_volumes_L[cells[i]]
        processes.append(localized(base, volume, cells[i]))
    mesh = spec.mesh
    faces = [
        (cells[i], cells[i + 1], float(mesh.centers_m[i + 1] - mesh.centers_m[i]))
        for i in range(mesh.cells - 1)
    ]
    if inlet_phase_id is not None:
        faces.insert(0, (inlet_phase_id, cells[0], float(mesh.centers_m[0])))
    if outlet_phase_id is not None:
        faces.append((cells[-1], outlet_phase_id, float(mesh.faces_m[-1] - mesh.centers_m[-1])))
    diagnostic: dict[str, Any] = {
        "momentum_evaluations": 0,
        "maximum_root_iterations": 0,
        "maximum_observed_speed_m_s": 0.0,
        "faces": faces,
        "inlet_resistance_Pa_s_m": spec.inlet_resistance_Pa_s_m,
    }
    cache: dict[str, Any] = {}

    def pressure(state: CoupledStateView, phase: str) -> float:
        local = state if _phase_temperature is None else replace(
            state, temperature_K=_phase_temperature(state, phase))
        return fixed_bed_pressure(spec, local, phase, phase_volumes_L[phase])

    def velocities(state: CoupledStateView) -> list[float]:
        temperatures = ((state.temperature_K,) if _phase_temperature is None
                        else tuple(_phase_temperature(state, p) for p in phases))
        signature = (*temperatures, *(state.amounts_mol[k] for k in flat))
        if cache.get("signature") == signature:
            return cache["velocities"]
        pressures = {p: pressure(state, p) for p in phases}
        result = []
        for left, right, distance in faces:
            density = 0.5 * fsum(
                state.amounts_mol[fixed_bed_gas_key(p, k)] * mass / (phase_volumes_L[p] * 0.001)
                for p in (left, right)
                for k, mass in spec.molecular_weights_kg_mol.items()
            )
            velocity, detail = fixed_bed_face_velocity(
                spec,
                distance_m=distance,
                density_kg_m3=density,
                pressure_difference_Pa=pressures[left] - pressures[right],
                series_resistance_Pa_s_m=(
                    spec.inlet_resistance_Pa_s_m if left == inlet_phase_id else 0.0
                ),
            )
            diagnostic["momentum_evaluations"] += 1
            diagnostic["maximum_root_iterations"] = max(
                diagnostic["maximum_root_iterations"], detail.get("iterations", 0)
            )
            diagnostic["maximum_observed_speed_m_s"] = max(
                diagnostic["maximum_observed_speed_m_s"], abs(velocity)
            )
            result.append(velocity)
        cache.update(signature=signature, velocities=result)
        return result

    def transport(state: CoupledStateView) -> ProcessContribution:
        transfers = []
        for i, ((left, right, _), velocity) in enumerate(
            zip(faces, velocities(state), strict=True)
        ):
            source, destination = (left, right) if velocity >= 0 else (right, left)
            for k in spec.component_ids:
                key = fixed_bed_gas_key(source, k)
                transfers.append(
                    InternalTransfer(
                        f"{spec.bed_id}:face:{i}:{k}",
                        key,
                        fixed_bed_gas_key(destination, k),
                        abs(velocity)
                        * spec.area_m2
                        * state.amounts_mol[key]
                        / (phase_volumes_L[source] * 0.001),
                    )
                )
        return ProcessContribution(internal_transfers=tuple(transfers))

    limit = min(spec.maximum_pressure_Pa, maximum_pressure_Pa or spec.maximum_pressure_Pa)

    def pressure_margin(
        p: str, boundary: float, sign: float
    ) -> Callable[[CoupledStateView], float]:
        def value(state: CoupledStateView) -> float:
            return sign * (pressure(state, p) - boundary)

        return value

    events = tuple(
        event
        for p in phases
        for event in (
            CoupledEvent(
                f"bed:minimum_pressure:{p}", pressure_margin(p, spec.minimum_pressure_Pa, 1.0)
            ),
            CoupledEvent(f"bed:maximum_pressure:{p}", pressure_margin(p, limit, -1.0)),
        )
    )
    events += (
        CoupledEvent(
            "bed:maximum_face_velocity",
            lambda state: spec.maximum_velocity_m_s - max(map(abs, velocities(state)), default=0.0),
        ),
    )

    def validate(state: CoupledStateView) -> None:
        for p in phases:
            value = pressure(state, p)
            if not spec.minimum_pressure_Pa - 1e-4 <= value <= limit + 1e-4:
                raise ValueError("actual gas compartment pressure leaves its hardware/domain")
        if max(map(abs, velocities(state)), default=0.0) > spec.maximum_velocity_m_s + 1e-8:
            raise ValueError("fixed-bed face velocity exceeds its declared domain")

    dependencies = {key: set(keys[p]) for p in phases for key in keys[p]}
    for left, right, _ in faces:
        for p, other in ((left, right), (right, left)):
            for key in keys[p]:
                dependencies[key].update(keys[other])
    processes.append(
        CoupledProcess(
            f"{spec.bed_id}:Ergun-upwind",
            transport,
            events,
            validate,
            {k: tuple(sorted(v)) for k, v in dependencies.items()},
        )
    )
    laws = {
        k: {
            **dict.fromkeys((fixed_bed_gas_key(p, k) for p in phases), 1.0),
            **({sorbent_key(s, k): 1.0 for s in slots} if k in spec.sorbent.component_ids else {}),
        }
        for k in spec.component_ids
    }
    scales = {
        fixed_bed_gas_key(p, k): phase_volumes_L[p]
        * 0.001
        * 100000
        / (R * spec.sorbent.reference_temperature_K)
        for p in phases
        for k in spec.component_ids
    }
    scales.update(
        {
            sorbent_key(s, k): max(s.values["sites_mol"], 1e-12)
            for s in slots
            for k in spec.sorbent.component_ids
        }
    )
    return CoupledSystem(
        thermal,
        tuple(processes),
        conservation_laws=laws,
        inventory_scales_mol=scales,
        isothermal=True,
        policy=policy,
    ), diagnostic
