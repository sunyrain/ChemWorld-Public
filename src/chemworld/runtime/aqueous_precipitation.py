"""Finite liquid mixing and instantaneous mineral equilibrium on original owners.

The packet contains liquid only. A source mineral remains in its original phase,
then both finite endpoints re-equilibrate before the common transaction commits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import fsum, isclose
from typing import Any

from chemworld.foundation import species_with_added_initial_amounts
from chemworld.foundation.balances import MaterialPacket
from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.physchem.aqueous_mineral_equilibrium import (
    AqueousMineralEquilibriumSpec,
    aqueous_mineral_properties,
)
from chemworld.physchem.aqueous_phase_properties import (
    AqueousPhaseProperties,
    aqueous_phase_properties,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.aqueous_mineral_equilibrium import equilibrate_aqueous_mineral_inventory
from chemworld.runtime.kernel_contracts import ModelProviderResult

PRECIPITATION_TYPE = "database_aqueous_mineral_equilibrium"


class PrecipitationClosureError(ValueError):
    """The declared equilibrium solver returned no qualified candidate."""


@dataclass(frozen=True)
class EmptyAqueousInventoryResult:
    """Exact empty extensive state, explicitly not an aqueous equilibrium solve."""

    initial_properties: AqueousPhaseProperties
    final_properties: AqueousPhaseProperties
    heat_duty_J: float = 0.0
    boundary_work_J: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "inventory_state": "empty",
            "equilibrium_performed": False,
            "initial_properties": self.initial_properties.to_dict(),
            "final_properties": self.final_properties.to_dict(),
            "heat_duty_J": self.heat_duty_J,
            "boundary_work_J": self.boundary_work_J,
        }


def precipitation_properties(state: Any) -> AqueousPhaseProperties:
    """Same-reference extensive properties, including explicitly declared emptiness."""
    spec = precipitation_specification(state)
    if not any(state.species_amounts.values()):
        if precipitation_record(state).settings.get("allow_empty_inventory") is not True:
            raise ValueError("empty aqueous inventory requires an explicit declaration")
        return AqueousPhaseProperties(state.temperature_K, state.pressure_Pa, 0, 0, 0, 0, 0)
    return aqueous_mineral_properties(
        spec,
        state.species_amounts,
        temperature_K=state.temperature_K,
        pressure_Pa=state.pressure_Pa,
    )


def precipitation_record(state: Any) -> Any:
    if state is None:
        return None
    records = [
        r
        for r in state.equipment.equipment.values()
        if r.equipment_type == PRECIPITATION_TYPE and r.attached_vessel_id == state.vessel_id
    ]
    if len(records) > 1:
        raise ValueError("one declared precipitation owner is allowed per vessel")
    return records[0] if records else None


def precipitation_specification(state: Any) -> AqueousMineralEquilibriumSpec:
    record = precipitation_record(state)
    if record is None:
        raise ValueError("finite precipitation requires its declared aqueous model")
    spec = AqueousMineralEquilibriumSpec.from_dict(record.settings["physical_model"])
    if len(spec.mineral_ids) != 1 or set(spec.dissolution_reactions) != set(spec.mineral_ids):
        raise ValueError("this mixing slice requires one declared dissolving pure mineral")
    if len(spec.reaction_ids) > 2:
        raise ValueError("this mixing slice supports at most one additional aqueous reaction")
    return spec


def close_precipitation(state: Any) -> tuple[Any, Any]:
    """Candidate closure; no clock, heat, work or material commit."""
    record = precipitation_record(state)
    spec = precipitation_specification(state)
    liquid_id, phase_ids = record.settings["liquid_phase_id"], record.settings["mineral_phase_ids"]
    if set(phase_ids) != set(spec.mineral_ids) or len({liquid_id, *phase_ids.values()}) != 1 + len(
        phase_ids
    ):
        raise ValueError("precipitation requires distinct original liquid and mineral identities")
    if set(state.phases.phases) != {liquid_id, *phase_ids.values()}:
        raise ValueError("precipitation requires exactly its original liquid/mineral phases")
    for quantity in ("temperature_K", "pressure_Pa"):
        if not isclose(
            getattr(state, quantity), record.settings[quantity], rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError("precipitation requires its declared fixed TP boundary")
    if state.state_slots.slots:
        raise ValueError("mixing with additional material history needs a declared transfer law")
    if set(state.species_amounts) != {s.species_id for s in spec.database.species}:
        raise ValueError("precipitation inventory must match its declared database")
    if not any(state.species_amounts.values()):
        properties = precipitation_properties(state)
        vessel = state.vessels.vessels[state.vessel_id]
        if (
            state.volume_L != 0
            or state.temperature_K != vessel.temperature_K
            or state.pressure_Pa != vessel.pressure_Pa
            or state.temperature_K > vessel.max_temperature_K
            or state.pressure_Pa > vessel.max_pressure_Pa
        ):
            raise ValueError("empty aqueous vessel must retain its actual zero volume and TP")
        spec.database.standard_states(state.temperature_K, state.pressure_Pa)
        expected = {
            liquid_id: {s.species_id for s in spec.database.species if s.phase != "mineral"},
            **{key: {mineral} for mineral, key in phase_ids.items()},
        }
        for key, phase in state.phases.phases.items():
            if (
                phase.phase_id != key
                or phase.vessel_id != state.vessel_id
                or phase.volume_L != 0
                or phase.solvents.volume_L != 0
                or set(phase.species_amounts_mol) != expected[key]
                or any(phase.species_amounts_mol.values())
                or (
                    phase.phase_type not in {"aqueous", "liquid"}
                    if key == liquid_id
                    else phase.phase_type != "solid"
                )
            ):
                raise ValueError("empty aqueous state must retain its original zero phase owners")
        empty_report = EmptyAqueousInventoryResult(properties, properties)
        return state.replace(
            metadata={
                **state.metadata,
                "precipitation_equilibrium": {
                    "inventory_state": "empty",
                    "equilibrium_performed": False,
                },
            }
        ), empty_report
    report = equilibrate_aqueous_mineral_inventory(
        state.phases,
        spec,
        liquid_id,
        phase_ids,
        vessel=state.vessels.vessels[state.vessel_id],
        temperature_K=state.temperature_K,
        pressure_Pa=state.pressure_Pa,
    )
    if not report.success:
        raise PrecipitationClosureError("precipitation closure failed: " + report.failure_reason)
    candidate = state.replace(
        phases=report.ledger,
        species_amounts=report.ledger.total_amounts_mol(),
        volume_L=fsum(p.volume_L for p in report.ledger.phases.values()),
        process=replace(state.process, last_observation_record=None),
        metadata={**state.metadata, "precipitation_equilibrium": report.equilibrium.to_dict()},
    )
    return candidate, report


def validate_precipitation_vessel(state: Any, physics: Any) -> None:
    spec = precipitation_specification(state)
    if physics.network.reactions:
        raise ValueError("precipitation rejects another homogeneous kinetic owner")
    declared = {s.species_id: s for s in spec.database.species}
    if set(physics.network.species_ids) != set(declared) or any(
        s.composition != declared[s.species_id].composition
        or s.charge != declared[s.species_id].charge
        or s.phase != ("solid" if declared[s.species_id].phase == "mineral" else "liquid")
        for s in physics.network.species
    ):
        raise ValueError("native identities must match the precipitation database")
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
        raise ValueError("precipitation rejects competing thermal/phase owners")
    validate_precipitation_state(state)


def validate_precipitation_state(state: Any) -> None:
    """Restore-time validation of the actual current phase owners and TP closure."""
    candidate, _ = close_precipitation(state)
    if any(
        not isclose(candidate.species_amounts[k], n, rel_tol=1e-8, abs_tol=1e-11)
        for k, n in state.species_amounts.items()
    ):
        raise ValueError("initial precipitation inventory must already be equilibrated")


def _thermal(duty: float) -> dict[str, float]:
    return {"energy_jacket_J": max(duty, 0.0), "heat_loss_J": max(-duty, 0.0)}


def advance_precipitation(
    state: Any, action: dict[str, Any], *, heat: bool = False, jacket_program: Any = None
) -> ModelProviderResult:
    if heat or jacket_program is not None or action.get("operation", "wait") != "wait":
        return ModelProviderResult(
            {},
            diagnostics={},
            success=False,
            outcome="rejected",
            failure_reason="precipitation_requires_isothermal_wait",
        )
    try:
        candidate, report = close_precipitation(state)
        duration = float(action["duration_s"])
        thermal = _thermal(report.heat_duty_J)
        candidate = candidate.replace(
            ledger=advance_ledger(
                state.ledger,
                duration,
                energy_deltas={**thermal, "boundary_work_J": report.boundary_work_J},
            )
        )
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return ModelProviderResult(
            {},
            diagnostics={"solver_error": str(exc)},
            success=False,
            failure_reason="precipitation_equilibrium_failed",
        )
    return ModelProviderResult(
        {},
        state=candidate,
        actual_duration_s=duration,
        diagnostics={"precipitation": report.to_dict()},
        vessel_thermal_deltas={state.vessel_id: thermal},
    )


def route_precipitation_packet(
    state: Any, physics: Any, action: dict[str, Any]
) -> ModelProviderResult:
    """Route a fraction of actual source liquid, never a fraction of total solids."""
    from chemworld.runtime.component_network import resolve_material_connection

    edge, fraction = resolve_material_connection(state, action, source_ports={"out"})
    target = state.inactive_vessels[edge["target_vessel"]]
    spec = precipitation_specification(state)
    if precipitation_specification(target) != spec:
        raise ValueError("precipitation endpoints require identical declared chemistry")
    if (
        edge["target"]["port"] != "in"
        or action.get("mixing") != 1
        or fraction >= 1
        or state.temperature_K != target.temperature_K
        or state.pressure_Pa != target.pressure_Pa
    ):
        raise ValueError("liquid mixing requires matching TP, in-port and positive remaining water")
    source_id = precipitation_record(state).settings["liquid_phase_id"]
    liquid = state.phases.phases[source_id]
    moved = {k: n * fraction for k, n in liquid.species_amounts_mol.items()}
    packet = MaterialPacket(state.vessel_id, target.vessel_id, tuple(moved.items()))
    remaining, received = packet.apply(state.species_amounts, target.species_amounts)
    packet_properties = aqueous_phase_properties(
        spec.database,
        moved,
        temperature_K=state.temperature_K,
        pressure_Pa=state.pressure_Pa,
    )

    def endpoint(local: Any, amounts: dict[str, float], sign: int) -> tuple[Any, Any]:
        key = precipitation_record(local).settings["liquid_phase_id"]
        phase = local.phases.phases[key]
        liquid_amounts = {
            k: n + sign * moved.get(k, 0.0) for k, n in phase.species_amounts_mol.items()
        }
        properties = aqueous_phase_properties(
            spec.database,
            liquid_amounts,
            temperature_K=local.temperature_K,
            pressure_Pa=local.pressure_Pa,
        )
        phases = PhaseLedger(
            {
                **local.phases.phases,
                key: replace(
                    phase,
                    species_amounts_mol=liquid_amounts,
                    volume_L=1000 * properties.volume_m3,
                ),
            }
        )
        return close_precipitation(
            local.replace(
                phases=phases,
                species_amounts=amounts,
                volume_L=fsum(p.volume_L for p in phases.phases.values()),
            )
        )

    # Both closures succeed before either original endpoint is installed.
    source, source_report = endpoint(state, remaining, -1)
    destination, target_report = endpoint(target, received, 1)

    def original_properties(local: Any) -> Any:
        return precipitation_properties(local)

    source_before, target_before = original_properties(state), original_properties(target)
    source_after, target_after = source_report.final_properties, target_report.final_properties
    qs = source_after.enthalpy_J - source_before.enthalpy_J + packet_properties.enthalpy_J
    qt = target_after.enthalpy_J - target_before.enthalpy_J - packet_properties.enthalpy_J
    work = state.pressure_Pa * (
        source_after.volume_m3
        + target_after.volume_m3
        - source_before.volume_m3
        - target_before.volume_m3
    )
    local_heat = {state.vessel_id: _thermal(qs), target.vessel_id: _thermal(qt)}
    energy = {
        key: fsum(values[key] for values in local_heat.values())
        for key in ("energy_jacket_J", "heat_loss_J")
    }
    share = fsum(moved.values()) / fsum(state.species_amounts.values())
    reference = {k: n * share for k, n in state.species.initial_amounts_mol.items()}
    source = source.replace(
        species=replace(
            state.species,
            initial_amounts_mol={
                k: n * (1 - share) for k, n in state.species.initial_amounts_mol.items()
            },
        )
    )
    destination = destination.replace(
        species=species_with_added_initial_amounts(target.species, reference)
    )
    receipt = {
        **packet.to_dict(),
        "model": "finite-isothermal-equilibrium-precipitation",
        "source_phase": source_id,
        "volume_L": packet_properties.volume_m3 * 1000,
        "enthalpy_J": packet_properties.enthalpy_J,
        "heat_duty_J": qs + qt,
        "boundary_work_J": work,
        "vessel_heat_duty_J": {state.vessel_id: qs, target.vessel_id: qt},
        "phase_assumption": "instantaneous_equilibrium_not_nucleation_kinetics",
    }
    transfer = {
        "connection": edge["id"],
        "source": state.vessel_id,
        "destination": target.vessel_id,
        "source_port": "out",
        "fraction": fraction,
        "amounts_mol": moved,
        "time_s": state.ledger.time_s,
        "mixing": 1,
    }
    candidate = source.replace(
        inactive_vessels={**state.inactive_vessels, target.vessel_id: destination},
        ledger=advance_ledger(
            state.ledger, 0, cost_delta=0.005, energy_deltas={**energy, "boundary_work_J": work}
        ),
        samples=replace(
            state.samples,
            transfers=(*state.samples.transfers, transfer),
            transfer_tools_used=state.samples.transfer_tools_used + 1,
        ),
        metadata={**source.metadata, "last_material_packet": receipt},
    )
    return ModelProviderResult(
        {},
        state=candidate,
        diagnostics={"precipitation_packet": receipt},
        vessel_thermal_deltas=local_heat,
    )
