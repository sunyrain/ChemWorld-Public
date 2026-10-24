"""Finite gas/liquid chemistry participates in the original wait transaction."""

from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.aqueous_gas_transfer import (
    AqueousHeadspaceSpec,
    advance_aqueous_headspace,
)
from chemworld.physchem.aqueous_gas_transfer import (
    headspace_pressure as _v1_pressure,
)
from chemworld.physchem.aqueous_gas_transfer import (
    headspace_properties as _v1_properties,
)
from chemworld.physchem.aqueous_gas_transfer import (
    prepare_aqueous_headspace_state as _v1_prepare,
)
from chemworld.physchem.aqueous_headspace_minerals import (
    AqueousHeadspaceMineralSpec,
    advance_aqueous_headspace_minerals,
    headspace_mineral_pressure,
    headspace_mineral_properties,
    headspace_mineral_volumes,
    prepare_aqueous_headspace_mineral_state,
)
from chemworld.physchem.mineral_kinetics import MineralGeometry
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.aqueous_mineral_kinetics import database_mineral_geometry_slot
from chemworld.runtime.kernel_contracts import ModelProviderResult

AQUEOUS_GAS_TRANSFER_TYPE = "finite_aqueous_gas_transfer"
HeadspaceSpec = AqueousHeadspaceSpec | AqueousHeadspaceMineralSpec


def headspace_pressure(spec: HeadspaceSpec, amounts: dict[str, float]) -> float:
    if isinstance(spec, AqueousHeadspaceMineralSpec):
        return headspace_mineral_pressure(spec, amounts)
    return _v1_pressure(spec, amounts)


def headspace_properties(spec: HeadspaceSpec, amounts: dict[str, float]) -> dict[str, Any]:
    if isinstance(spec, AqueousHeadspaceMineralSpec):
        return headspace_mineral_properties(spec, amounts)
    return _v1_properties(spec, amounts)


def prepare_aqueous_headspace_state(
    spec: HeadspaceSpec, amounts: dict[str, float]
) -> dict[str, float]:
    if isinstance(spec, AqueousHeadspaceMineralSpec):
        return prepare_aqueous_headspace_mineral_state(spec, amounts)
    return _v1_prepare(spec, amounts)


def headspace_phase_layout(
    spec: HeadspaceSpec, amounts: dict[str, float]
) -> tuple[tuple[str, str, float, tuple[str, ...]], ...]:
    if isinstance(spec, AqueousHeadspaceMineralSpec):
        volumes = headspace_mineral_volumes(spec, amounts)
        return (
            ("aqueous", "aqueous", spec.liquid_volume_m3, spec.liquid_species_ids),
            ("headspace", "gas", volumes["headspace"], spec.gas_species_ids),
            *(("mineral:" + m, "solid", volumes["mineral:" + m], (m,)) for m in spec.mineral_ids),
        )
    return (
        ("aqueous", "aqueous", spec.liquid_volume_m3, spec.liquid_species_ids),
        ("headspace", "gas", spec.gas_volume_m3, spec.gas_species_ids),
    )


def headspace_container_volume(spec: HeadspaceSpec) -> float:
    if isinstance(spec, AqueousHeadspaceMineralSpec):
        return spec.container_volume_m3
    return spec.liquid_volume_m3 + spec.gas_volume_m3


def headspace_mineral_geometries(
    state: Any, spec: AqueousHeadspaceMineralSpec
) -> dict[str, MineralGeometry]:
    slots = state.state_slots.slots
    if len(slots) != len(spec.mineral_ids):
        raise ValueError("headspace requires exactly one original geometry slot per mineral")
    result = {}
    for m in spec.mineral_ids:
        matching = [s for s in slots if s.spec.mechanism_id == m]
        if len(matching) != 1:
            raise ValueError("mineral material reference missing or duplicated")
        slot = matching[0]
        geometry = MineralGeometry(**slot.values)
        if state.species_amounts[m] > 0 and geometry.reference_amount_mol <= 0:
            raise ValueError("positive mineral stock requires positive material reference amount")
        expected = database_mineral_geometry_slot(state.vessel_id, "mineral:" + m, m, geometry)
        if slot != expected:
            raise ValueError("mineral reference must retain original phase/material ownership")
        result[m] = geometry
    return result


def aqueous_gas_transfer_record(state: Any) -> Any:
    return attached_device_record(state, AQUEOUS_GAS_TRANSFER_TYPE)


def install_headspace(state: Any, spec: HeadspaceSpec, amounts: dict[str, float]) -> Any:
    pressure = headspace_pressure(spec, amounts)
    phases = {}
    for identity, kind, volume, ids in headspace_phase_layout(spec, amounts):
        phases[identity] = PhaseRecord(
            identity,
            state.vessel_id,
            kind,
            volume * 1000,
            {k: amounts[k] for k in ids},
            selected=identity == "aqueous",
            metadata={"carrier_representation": "species_inventory"},
        )
    vessel = state.vessels.vessels[state.vessel_id]
    return state.replace(
        species_amounts=amounts,
        pressure_Pa=pressure,
        phases=PhaseLedger(phases),
        vessels=replace(
            state.vessels,
            vessels={
                **state.vessels.vessels,
                state.vessel_id: replace(
                    vessel,
                    pressure_Pa=pressure,
                ),
            },
        ),
        process=replace(state.process, last_observation_record=None),
    )


class AqueousGasTransferRuntime:
    supported_operations = ("wait",)

    def __init__(self, composition: Any):
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return aqueous_gas_transfer_record(state) is not None

    def specification(self, state: Any) -> HeadspaceSpec:
        record = aqueous_gas_transfer_record(state)
        if record is None:
            raise ValueError("finite gas/liquid owner missing")
        data = record.settings["physical_model"]
        if data.get("model") == "finite-aqueous-headspace-minerals-v2":
            return AqueousHeadspaceMineralSpec.from_dict(data)
        return AqueousHeadspaceSpec.from_dict(data)

    def reconcile_pressure(self, state: Any) -> Any:
        pressure = headspace_pressure(self.specification(state), state.species_amounts)
        vessel = state.vessels.vessels[state.vessel_id]
        return state.replace(
            pressure_Pa=pressure,
            vessels=replace(
                state.vessels,
                vessels={
                    **state.vessels.vessels,
                    state.vessel_id: replace(vessel, pressure_Pa=pressure),
                },
            ),
        )

    @property
    def preserved_source_vessels(self) -> set[str]:
        physics = self.composition.physics
        return (
            set()
            if physics is None
            else {v for v, state in physics.initial_states.items() if self.supports(state)}
        )

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def validate_state(self, state: Any) -> None:
        spec = self.specification(state)
        phases = state.phases.phases
        vessel = state.vessels.vessels[state.vessel_id]
        pressure = headspace_pressure(spec, state.species_amounts)
        layout = headspace_phase_layout(spec, state.species_amounts)
        mineral = isinstance(spec, AqueousHeadspaceMineralSpec)
        if isinstance(spec, AqueousHeadspaceMineralSpec):
            headspace_mineral_geometries(state, spec)
        if (
            state.phase != "fluid"
            or (state.state_slots.slots and not mineral)
            or state.temperature_K != spec.temperature_K
            or not isclose(state.pressure_Pa, pressure, rel_tol=1e-10, abs_tol=1e-8)
            or not isclose(vessel.pressure_Pa, pressure, rel_tol=1e-10, abs_tol=1e-8)
            or not isclose(state.volume_L, headspace_container_volume(spec) * 1000)
            or state.volume_L > vessel.max_volume_L
            or pressure > vessel.max_pressure_Pa
            or spec.temperature_K > vessel.max_temperature_K
            or set(phases) != {p[0] for p in layout}
        ):
            raise ValueError(
                "finite headspace current geometry/T/P/ownership differs from declaration"
            )
        for identity, kind, volume, ids in layout:
            phase = phases[identity]
            if (
                phase.phase_id != identity
                or phase.vessel_id != state.vessel_id
                or phase.phase_type != kind
                or phase.solvents.volume_L != 0
                or phase.species_amounts_mol != {k: state.species_amounts[k] for k in ids}
                or not isclose(phase.volume_L, volume * 1000, rel_tol=1e-10, abs_tol=1e-13)
            ):
                raise ValueError(
                    "gas/liquid inventories must belong to their declared original phases"
                )
        closed = prepare_aqueous_headspace_state(spec, state.species_amounts)
        if any(
            not isclose(closed[k], n, rel_tol=1e-8, abs_tol=1e-13)
            for k, n in state.species_amounts.items()
        ):
            raise ValueError("stored liquid is not at declared fast equilibrium")
        original = self.composition.physics.initial_states.get(state.vessel_id)
        if original is None or self.specification(original).to_dict() != spec.to_dict():
            raise ValueError("headspace declaration differs from original world configuration")
        if mineral and original.state_slots != state.state_slots:
            raise ValueError("headspace original material geometry cannot be reset or replaced")

    def validate_world(self) -> None:
        physics = self.composition.physics
        declared = {
            c.vessel for c in self.composition.components if c.kind == "aqueous_gas_transfer"
        }
        installed = set() if physics is None else self.preserved_source_vessels
        if not declared and not installed:
            return
        if installed != declared or physics is None or physics.network.reactions:
            raise ValueError("headspace owner/component mismatch or competing reaction owner")
        for name in installed:
            state = physics.initial_states[name]
            spec = self.specification(state)
            self.validate_state(state)
            original = {s.species_id: s for s in physics.network.species}
            if set(original) != {s.species_id for s in spec.equilibrium.database.species} or any(
                original[s.species_id].composition != s.composition
                or original[s.species_id].charge != s.charge
                or original[s.species_id].phase
                != ("gas" if s.phase == "gas" else "solid" if s.phase == "mineral" else "liquid")
                for s in spec.equilibrium.database.species
            ):
                raise ValueError("headspace database and original species identities differ")

    def advance(
        self, state: Any, action: dict, *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        if heat or jacket_program is not None or action.get("operation", "wait") != "wait":
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="headspace_requires_isothermal_wait",
            )
        try:
            self.validate_state(state)
            spec = self.specification(state)
            if isinstance(spec, AqueousHeadspaceMineralSpec):
                result = advance_aqueous_headspace_minerals(
                    spec,
                    state.species_amounts,
                    headspace_mineral_geometries(state, spec),
                    float(action["duration_s"]),
                    maximum_pressure_Pa=state.vessels.vessels[state.vessel_id].max_pressure_Pa,
                )
            else:
                result = advance_aqueous_headspace(
                    spec, state.species_amounts, float(action["duration_s"])
                )
            candidate = install_headspace(state, spec, result.final["amounts_mol"])
            self.validate_state(candidate)
            heat_j = result.final["bath_heat_J"]
            candidate = candidate.replace(
                ledger=advance_ledger(
                    state.ledger,
                    result.actual_duration_s,
                    energy_deltas={"energy_jacket_J": heat_j},
                )
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="aqueous_headspace_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=result.actual_duration_s,
            vessel_thermal_deltas={state.vessel_id: {"energy_jacket_J": heat_j}},
            diagnostics={
                "aqueous_headspace": result.to_dict(),
                "current_properties": headspace_properties(spec, result.final["amounts_mol"]),
            },
        )
