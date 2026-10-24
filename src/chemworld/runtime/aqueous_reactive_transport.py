"""One aqueous spatial owner participates in the original shared transaction."""

from __future__ import annotations

from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.foundation.lab_clock import SharedAdvanceSystem
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.aqueous_reactive_transport import (
    AqueousReactiveTransportSpec,
    advance_aqueous_reactive_transport,
    prepare_aqueous_transport_state,
)
from chemworld.physchem.aqueous_transport_flow import inventory_properties
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

AQUEOUS_TRANSPORT_TYPE = "finite_aqueous_reactive_transport"


def aqueous_transport_record(state: Any) -> Any:
    return attached_device_record(state, AQUEOUS_TRANSPORT_TYPE)


def aqueous_transport_members(physics: Any) -> set[str]:
    """Derive sealed mesh members from the original owner declaration only."""
    names: set[str] = set()
    if physics is None:
        return names
    for state in physics.initial_states.values():
        record = aqueous_transport_record(state)
        if record is not None:
            names.update(
                AqueousReactiveTransportSpec.from_dict(
                    record.settings["physical_model"]
                ).member_vessel_ids
            )
    return names


def install_aqueous_cell(state: Any, amounts: dict[str, float], volume_L: float) -> Any:
    old = state.phases.phases.get("aqueous") if state.phases else None
    phase = (
        replace(old, species_amounts_mol=amounts, volume_L=volume_L)
        if old
        else PhaseRecord(
            "aqueous",
            state.vessel_id,
            "aqueous",
            volume_L,
            amounts,
            selected=True,
            metadata={"carrier_representation": "species_inventory"},
        )
    )
    return state.replace(
        species_amounts=amounts,
        volume_L=volume_L,
        phases=PhaseLedger({"aqueous": phase}),
        process=replace(state.process, last_observation_record=None),
    )


def unpack_aqueous_transport(state: Any) -> tuple[AqueousReactiveTransportSpec, dict]:
    record = aqueous_transport_record(state)
    if record is None:
        raise ValueError("aqueous transport owner missing")
    spec = AqueousReactiveTransportSpec.from_dict(record.settings["physical_model"])
    if state.vessel_id != spec.anchor_vessel_id:
        raise ValueError("aqueous device must remain on its declared anchor")
    locals_ = {state.vessel_id: state, **state.inactive_vessels}
    amounts = {}
    for name in spec.member_vessel_ids:
        local = locals_.get(name)
        if local is None or set(local.phases.phases) != {"aqueous"}:
            raise ValueError("every spatial cell needs its original aqueous phase")
        phase = local.phases.phases["aqueous"]
        volume = (
            spec.vessel_capacity_m3(name)
            if spec.flow is None
            else inventory_properties(spec, local.species_amounts).volume_m3
        ) * 1000
        vessel = local.vessels.vessels[name]
        if (
            phase.vessel_id != name
            or phase.phase_id != "aqueous"
            or phase.phase_type not in {"aqueous", "liquid"}
            or phase.solvents.volume_L != 0
            or phase.species_amounts_mol != local.species_amounts
            or not isclose(phase.volume_L, volume, rel_tol=1e-10, abs_tol=1e-13)
            or not isclose(local.volume_L, volume, rel_tol=1e-10, abs_tol=1e-13)
            or local.temperature_K != spec.temperature_K
            or local.pressure_Pa != spec.pressure_Pa
            or volume > vessel.max_volume_L
            or spec.temperature_K > vessel.max_temperature_K
            or spec.pressure_Pa > vessel.max_pressure_Pa
            or local.state_slots.slots
        ):
            raise ValueError("spatial liquid ownership/geometry/fixed TP differs from declaration")
        amounts[name] = dict(local.species_amounts)
    prepared = prepare_aqueous_transport_state(spec, amounts)
    if any(
        not isclose(prepared[v][s], n, rel_tol=1e-8, abs_tol=1e-14)
        for v, ns in amounts.items()
        for s, n in ns.items()
    ):
        raise ValueError("stored spatial liquid is not at its declared fast equilibrium")
    return spec, amounts


class AqueousReactiveTransportRuntime:
    supported_operations = ("wait",)

    def __init__(self, composition: Any):
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return aqueous_transport_record(state) is not None

    def validate_state(self, state: Any) -> None:
        """Validate current snapshot cells, not their historical initial stocks."""
        spec, _ = unpack_aqueous_transport(state)
        physics = self.composition.physics
        original = None if physics is None else physics.initial_states.get(spec.anchor_vessel_id)
        reference = None if original is None else aqueous_transport_record(original)
        current = aqueous_transport_record(state)
        if (
            reference is None
            or current.settings["physical_model"] != reference.settings["physical_model"]
        ):
            raise ValueError(
                "aqueous owner declaration differs from the original world configuration"
            )

    def shared_system(self, state: Any, strategy_id: str) -> SharedAdvanceSystem:
        record = aqueous_transport_record(state)
        if record is None:
            raise ValueError("aqueous shared system requires its installed owner")
        spec = AqueousReactiveTransportSpec.from_dict(record.settings["physical_model"])
        components = {c.id: c for c in self.composition.components}
        edges = {e["id"]: e for e in self.composition.connections}
        for identity, source, target in spec.topology:
            edge = edges.get(identity)
            if (
                edge is None
                or edge["unit"] != "mol"
                or edge.get("scheduling") == "manual"
                or edge["source"]["port"] != "out"
                or edge["target"]["port"] != "in"
                or components[edge["source"]["component"]].vessel != source
                or components[edge["target"]["component"]].vessel != target
            ):
                raise ValueError("aqueous spatial faces must match original material connections")
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            spec.member_vessel_ids,
            tuple(row[0] for row in spec.topology),
        )

    @property
    def preserved_source_vessels(self) -> set[str]:
        return aqueous_transport_members(self.composition.physics)

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def validate_world(self) -> None:
        physics = self.composition.physics
        declared = {
            c.vessel for c in self.composition.components if c.kind == "aqueous_reactive_transport"
        }
        if physics is None:
            if declared:
                raise ValueError("aqueous transport requires original world physics")
            return
        installed = {v for v, s in physics.initial_states.items() if self.supports(s)}
        if not installed and not declared:
            return
        if installed != declared or physics.network.reactions:
            raise ValueError("aqueous owner/component mismatch or competing reaction owner")
        claimed: set[str] = set()
        for name in installed:
            state = physics.initial_states[name].replace(
                inactive_vessels={v: s for v, s in physics.initial_states.items() if v != name}
            )
            spec, _ = unpack_aqueous_transport(state)
            if claimed & set(spec.member_vessel_ids):
                raise ValueError("aqueous spatial owners overlap")
            claimed.update(spec.member_vessel_ids)
            original = {s.species_id: s for s in physics.network.species}
            if set(original) != {s.species_id for s in spec.equilibrium.database.species} or any(
                original[s.species_id].composition != s.composition
                or original[s.species_id].charge != s.charge
                or original[s.species_id].phase != "liquid"
                for s in spec.equilibrium.database.species
            ):
                raise ValueError("aqueous database and original species identities differ")
            self.shared_system(state, "finite-aqueous-reactive-transport")

    def advance(
        self, state: Any, action: dict, *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        if heat or jacket_program is not None or action.get("operation", "wait") != "wait":
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="aqueous_transport_requires_fixed_TP_wait",
            )
        try:
            self.validate_state(state)
            spec, amounts = unpack_aqueous_transport(state)
            result = advance_aqueous_reactive_transport(spec, amounts, float(action["duration_s"]))
            final, dt = result.final, result.actual_duration_s
            locals_ = {state.vessel_id: state, **state.inactive_vessels}
            thermal = {
                v: {
                    "energy_jacket_J": final["cells"][v]["bath_heat_J"],
                    **(
                        {"boundary_work_J": final["cells"][v]["boundary_work_J"]}
                        if spec.flow is not None
                        else {}
                    ),
                }
                for v in spec.member_vessel_ids
            }
            updated = {}
            for name in spec.member_vessel_ids:
                local = install_aqueous_cell(
                    locals_[name],
                    final["amounts_by_vessel"][name],
                    (
                        spec.vessel_capacity_m3(name)
                        if spec.flow is None
                        else inventory_properties(spec, final["amounts_by_vessel"][name]).volume_m3
                    )
                    * 1000,
                )
                updated[name] = local.replace(
                    ledger=advance_ledger(local.ledger, dt, energy_deltas=thermal[name])
                )
            candidate = updated.pop(state.vessel_id)
            candidate = candidate.replace(
                inactive_vessels={**state.inactive_vessels, **updated},
                vessel_elapsed_s={
                    **state.vessel_elapsed_s,
                    **{v: state.vessel_elapsed_s.get(v, 0) + dt for v in spec.member_vessel_ids},
                },
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    energy_deltas={
                        "energy_jacket_J": final["bath_heat_J"],
                        **(
                            {
                                "boundary_work_J": sum(
                                    r["boundary_work_J"] for r in final["cells"].values()
                                )
                            }
                            if spec.flow is not None
                            else {}
                        ),
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="aqueous_transport_solver_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.diagnostics.get("stopping_events") else "success",
            vessel_thermal_deltas={
                v: {k: value for k, value in deltas.items() if k != "boundary_work_J"}
                for v, deltas in thermal.items()
            },
            diagnostics={
                "aqueous_transport": result.to_dict(),
                "energy_scope": (
                    "fixed TPV; single signed bath duty; internal face enthalpies cancel"
                    if spec.flow is None
                    else (
                        "same TP finite reservoirs and fixed cell V; local Q=dH-net H in; "
                        "dU=Q+net H in-PdV; no pump model"
                    )
                ),
            },
        )
