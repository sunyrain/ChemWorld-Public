"""Declared mineral kinetics on the original phases, geometry slots and clock."""

from __future__ import annotations

from dataclasses import replace
from math import fsum, isclose
from typing import Any, cast

from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.foundation.state_ownership import StateFieldSpec, StateSlot, StateSlotSpec
from chemworld.physchem.aqueous_mineral_equilibrium import aqueous_mineral_properties
from chemworld.physchem.aqueous_mineral_kinetics import (
    AqueousMineralKineticsSpec,
    advance_aqueous_minerals,
    prepare_aqueous_mineral_state,
)
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.mineral_kinetics import MineralGeometry
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

DATABASE_MINERAL_TYPE = "database_aqueous_mineral_kinetics"
GEOMETRY_TYPE = "declared_mineral_surface_reference"


def database_mineral_record(state: Any) -> Any:
    records = [
        r
        for r in state.equipment.equipment.values()
        if r.equipment_type == DATABASE_MINERAL_TYPE and r.attached_vessel_id == state.vessel_id
    ]
    if len(records) > 1:
        raise ValueError("one declared mineral kinetic owner is allowed per vessel")
    return records[0] if records else None


def database_mineral_geometry_slot(
    vessel_id: str, phase_id: str, mineral_id: str, geometry: MineralGeometry
) -> StateSlot:
    """Reference surface follows its existing material lot, never copied bulk n."""
    return StateSlot(
        StateSlotSpec(
            f"{vessel_id}:{phase_id}:surface",
            "material",
            f"{vessel_id}:{phase_id}:lot",
            GEOMETRY_TYPE,
            (
                StateFieldSpec("reference_area_m2", "m^2"),
                StateFieldSpec("reference_amount_mol", "mol"),
            ),
            mechanism_id=mineral_id,
            phase_id=phase_id,
            time_evolution="static",
        ),
        vessel_id,
        geometry.to_dict(),
    )


def database_mineral_geometries(
    state: Any, specification: AqueousMineralKineticsSpec, phase_ids: dict[str, str]
) -> dict[str, MineralGeometry]:
    slots = [s for s in state.state_slots.slots if s.spec.state_type == GEOMETRY_TYPE]
    if len(slots) != len(specification.equilibrium.mineral_ids):
        raise ValueError("each declared mineral needs exactly one persistent surface reference")
    result = {}
    for mineral in specification.equilibrium.mineral_ids:
        matching = [s for s in slots if s.spec.mechanism_id == mineral]
        if len(matching) != 1:
            raise ValueError("mineral surface identity is missing or duplicated")
        slot = matching[0]
        if (
            slot.spec.owner_kind != "material"
            or slot.spec.phase_id != phase_ids[mineral]
            or slot.location_id != state.vessel_id
            or slot.spec.time_evolution != "static"
        ):
            raise ValueError("mineral reference geometry must belong to its actual material phase")
        result[mineral] = MineralGeometry(**slot.values)
    return result


class DatabaseMineralKineticsRuntime:
    supported_operations = ("wait",)

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return database_mineral_record(state) is not None

    def validate_state(
        self, state: Any
    ) -> tuple[
        AqueousMineralKineticsSpec,
        dict[str, float],
        dict[str, MineralGeometry],
        str,
        dict[str, str],
    ]:
        """Validate both initialization and restored original material ownership."""
        record = database_mineral_record(state)
        if record is None:
            raise ValueError("declared mineral kinetic record is missing")
        spec = AqueousMineralKineticsSpec.from_dict(record.settings["physical_model"])
        liquid_id = record.settings["liquid_phase_id"]
        phase_ids = record.settings["mineral_phase_ids"]
        if set(phase_ids) != set(spec.equilibrium.mineral_ids) or len(
            {liquid_id, *phase_ids.values()}
        ) != 1 + len(phase_ids):
            raise ValueError("declared liquid/mineral phases require distinct original identities")
        for quantity in ("temperature_K", "pressure_Pa"):
            if not isclose(
                getattr(state, quantity), record.settings[quantity], rel_tol=0, abs_tol=1e-8
            ):
                raise ValueError("declared mineral kinetics supports its fixed TP boundary only")
        vessel = state.vessels.vessels[state.vessel_id]
        if (
            state.temperature_K > vessel.max_temperature_K
            or state.pressure_Pa > vessel.max_pressure_Pa
        ):
            raise ValueError("mineral boundary exceeds vessel limits")
        if set(state.phases.phases) != {liquid_id, *phase_ids.values()}:
            raise ValueError(
                "this mineral slice requires its declared liquid and mineral phases only"
            )
        db = spec.equilibrium.database
        liquid_ids = {s.species_id for s in db.species if s.phase != "mineral"}
        liquid = state.phases.phases[liquid_id]
        if (
            liquid.vessel_id != state.vessel_id
            or liquid.phase_type not in {"liquid", "aqueous"}
            or liquid.solvents.volume_L != 0
            or set(liquid.species_amounts_mol) != liquid_ids
        ):
            raise ValueError("declared water and ions must reside in their original liquid phase")
        props = aqueous_phase_properties(
            db,
            liquid.species_amounts_mol,
            temperature_K=state.temperature_K,
            pressure_Pa=state.pressure_Pa,
        )
        if not isclose(liquid.volume_L, 1000 * props.volume_m3, rel_tol=1e-8, abs_tol=1e-12):
            raise ValueError("liquid geometry disagrees with its same-reference volume")
        for species in db.species:
            if species.phase != "mineral":
                continue
            phase = state.phases.phases[phase_ids[species.species_id]]
            if (
                phase.vessel_id != state.vessel_id
                or phase.phase_type != "solid"
                or phase.solvents.volume_L != 0
                or set(phase.species_amounts_mol) != {species.species_id}
            ):
                raise ValueError("pure mineral inventory must retain its original phase owner")
            n = phase.species_amounts_mol[species.species_id]
            volume = n * species.standard_state.molar_volume_m3_mol * 1000
            if not isclose(phase.volume_L, volume, rel_tol=1e-8, abs_tol=1e-12):
                raise ValueError("mineral volume disagrees with its standard state")
        amounts = state.phases.total_amounts_mol()
        geometries = database_mineral_geometries(state, spec, phase_ids)
        if any(amounts[k] > 0 and g.reference_amount_mol <= 0 for k, g in geometries.items()):
            raise ValueError("positive mineral inventory lacks its surface reference")
        current = aqueous_mineral_properties(
            spec.equilibrium,
            amounts,
            temperature_K=state.temperature_K,
            pressure_Pa=state.pressure_Pa,
        )
        if current.volume_m3 * 1000 > vessel.max_volume_L:
            raise ValueError("mineral inventory exceeds finite vessel capacity")
        closed = prepare_aqueous_mineral_state(
            spec, amounts, temperature_K=state.temperature_K, pressure_Pa=state.pressure_Pa
        )
        if any(not isclose(closed[k], amounts[k], rel_tol=1e-8, abs_tol=1e-11) for k in closed):
            raise ValueError("current liquid is not at its declared fast equilibrium")
        return spec, amounts, geometries, liquid_id, phase_ids

    def validate_world(self) -> None:
        if self.composition is None or self.composition.physics is None:
            return
        physics = self.composition.physics
        for state in physics.initial_states.values():
            if not self.supports(state):
                continue
            if physics.network.reactions:
                raise ValueError("declared mineral kinetics rejects another homogeneous rate owner")
            spec, *_ = self.validate_state(state)
            declared = {s.species_id: s for s in spec.equilibrium.database.species}
            if set(physics.network.species_ids) != set(declared) or any(
                s.composition != declared[s.species_id].composition
                or s.charge != declared[s.species_id].charge
                or s.phase != ("solid" if declared[s.species_id].phase == "mineral" else "liquid")
                for s in physics.network.species
            ):
                raise ValueError("native species identities must agree with the declared database")
            if any(
                state.vessel_id in physics.models_of(kind)
                for kind in (
                    "thermal",
                    "flash",
                    "liquid",
                    "column",
                    "concentration",
                    "crystal",
                )
            ):
                raise ValueError("declared mineral slice rejects competing thermal/phase owners")

    def advance(
        self, state: Any, action: dict[str, Any], *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        if heat or jacket_program is not None or action.get("operation", "wait") != "wait":
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="declared_mineral_nonisothermal_or_transfer_unsupported",
            )
        try:
            spec, amounts, geometries, liquid_id, phase_ids = self.validate_state(state)
            report = advance_aqueous_minerals(
                spec,
                amounts,
                geometries,
                float(action["duration_s"]),
                temperature_K=state.temperature_K,
                pressure_Pa=state.pressure_Pa,
                maximum_volume_L=state.vessels.vessels[state.vessel_id].max_volume_L,
            )
            final = report.final
            liquid = {k: n for k, n in final.amounts_mol.items() if k not in phase_ids}
            props = aqueous_phase_properties(
                spec.equilibrium.database,
                liquid,
                temperature_K=state.temperature_K,
                pressure_Pa=state.pressure_Pa,
            )
            updates = {
                liquid_id: replace(
                    state.phases.phases[liquid_id],
                    species_amounts_mol=liquid,
                    volume_L=props.volume_m3 * 1000,
                )
            }
            for species in spec.equilibrium.database.species:
                if species.phase == "mineral":
                    key, n = phase_ids[species.species_id], final.amounts_mol[species.species_id]
                    updates[key] = replace(
                        state.phases.phases[key],
                        species_amounts_mol={species.species_id: n},
                        volume_L=n * cast(float, species.standard_state.molar_volume_m3_mol) * 1000,
                    )
            duty, work = final.heat_duty_J, final.boundary_work_J
            candidate = state.replace(
                phases=PhaseLedger(updates),
                species_amounts=final.amounts_mol,
                volume_L=fsum(p.volume_L for p in updates.values()),
                ledger=advance_ledger(
                    state.ledger,
                    report.actual_duration_s,
                    energy_deltas={
                        "energy_jacket_J": max(duty, 0),
                        "heat_loss_J": max(-duty, 0),
                        "boundary_work_J": work,
                    },
                ),
                process=replace(state.process, last_observation_record=None),
                metadata={**state.metadata, "last_declared_mineral_advance": report.to_dict()},
            )
            self.validate_state(candidate)
        except (ValueError, ArithmeticError, RuntimeError) as exc:
            return ModelProviderResult(
                {},
                diagnostics={"solver_error": str(exc)},
                success=False,
                outcome="numerical_failure",
                failure_reason="declared_mineral_kinetics_failed",
            )
        return ModelProviderResult(
            {},
            diagnostics={"declared_mineral_kinetics": report.to_dict()},
            state=candidate,
            actual_duration_s=report.actual_duration_s,
        )
