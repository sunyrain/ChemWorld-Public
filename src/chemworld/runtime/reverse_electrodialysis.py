"""Original membrane state and clock adapter for a passive resistive RED load."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.reverse_electrodialysis import (
    ReverseElectrodialysisSpec,
    advance_reverse_electrodialysis,
    reverse_electrodialysis_point,
    validate_red_load_program,
)
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.control_program import electrical_device
from chemworld.runtime.electrodialysis import (
    electrodialysis_vessels,
    install_ed_liquid,
    unpack_periodic_liquid_inventory,
)
from chemworld.runtime.kernel_contracts import ModelProviderResult

REVERSE_ELECTRODIALYSIS_TYPE = "periodic_reverse_electrodialysis"


def reverse_electrodialysis_record(state: Any) -> Any:
    return attached_device_record(state, REVERSE_ELECTRODIALYSIS_TYPE)


def _validate_settings(record: Any, spec: ReverseElectrodialysisSpec) -> ScalarControlProgram:
    settings, pair = record.settings["electrical_control"], spec.cell_pair
    if (
        settings.get("passive_load_modes") != ["open_circuit", "resistance"]
        or settings.get("supported_units") != ["A", "Ohm"]
        or settings.get("limits_V") != [-pair.maximum_voltage_V, pair.maximum_voltage_V]
        or settings.get("limits_A") != [-pair.maximum_current_A, pair.maximum_current_A]
        or settings.get("limits_R_Ohm")
        != [spec.minimum_load_resistance_Ohm, spec.maximum_load_resistance_Ohm]
        or settings.get("staged_control", False)
        or settings.get("named_control_modes")
        or settings.get("load_mode") not in {"open_circuit", "resistance"}
    ):
        raise ValueError("RED load metadata must match its physical limits and passive modes")
    program = ScalarControlProgram(**settings["program"])
    validate_red_load_program(spec, program)
    expected = "A" if settings["load_mode"] == "open_circuit" else "Ohm"
    if program.unit != expected:
        raise ValueError("RED mode and physical load programme disagree")
    return program


def unpack_reverse_electrodialysis(
    state: Any,
) -> tuple[Any, ReverseElectrodialysisSpec, TransportInventory, ScalarControlProgram]:
    record = reverse_electrodialysis_record(state)
    if record is None:
        raise ValueError("no declared passive reverse electrodialysis equipment is installed")
    spec = ReverseElectrodialysisSpec.from_dict(record.settings["physical_model"])
    if electrical_device(state).equipment_id != record.equipment_id:
        raise ValueError("RED must own its unique connected electrical load")
    program = _validate_settings(record, spec)
    return record, spec, unpack_periodic_liquid_inventory(state, spec.cell_pair), program


def configured_red_load_program(
    state: Any,
    record: Any,
    action: dict[str, Any],
) -> tuple[str, ScalarControlProgram]:
    original, spec, _, previous = unpack_reverse_electrodialysis(state)
    if original.equipment_id != record.equipment_id:
        raise ValueError("passive load control must target the current RED equipment")
    mode = action.get("control_mode", "resistance" if "resistance_Ohm" in action else None)
    if mode not in {"open_circuit", "resistance"}:
        raise ValueError("RED load mode must be open_circuit or resistance")
    forbidden = {
        "potential_V",
        "current_A",
        "power_W",
        "current_limit_A",
        "branch_lower_A",
        "branch_upper_A",
    }
    if forbidden & action.keys():
        raise ValueError("passive RED load cannot impose voltage, current or active circuit power")
    duration = action.get("ramp_duration_s", 0.0)
    if isinstance(duration, bool) or not isfinite(duration) or duration < 0:
        raise ValueError("load ramp duration must be finite nonnegative seconds")
    time = state.ledger.time_s
    if mode == "open_circuit":
        if "resistance_Ohm" in action or duration != 0:
            raise ValueError("open circuit is an explicit switch, not a resistance or a ramp")
        return mode, ScalarControlProgram(((time, 0.0),), "A")
    target = action.get("resistance_Ohm")
    if (
        isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not isfinite(target)
        or not spec.minimum_load_resistance_Ohm <= target <= spec.maximum_load_resistance_Ohm
    ):
        raise ValueError(
            "passive resistance requires an explicit Ohm target inside physical limits"
        )
    if duration and previous.unit != "Ohm":
        raise ValueError("a load ramp cannot interpolate from an open circuit")
    points = (
        ((time, previous.value_at(time)), (time + duration, target))
        if duration
        else ((time, target),)
    )
    return mode, ScalarControlProgram(points, "Ohm", mode="linear")


@dataclass(frozen=True)
class REDElectricalReading:
    terminal_potential_V: float
    applied_current_A: float


def read_reverse_electrodialysis_electrical(state: Any) -> REDElectricalReading:
    _, spec, inventory, program = unpack_reverse_electrodialysis(state)
    row = reverse_electrodialysis_point(
        spec,
        inventory.amounts_mol,
        program.value_at(state.ledger.time_s) if program.unit == "Ohm" else None,
    )
    if row["status"] != "converged":
        raise RuntimeError("current RED terminal point did not converge: " + str(row["solver"]))
    return REDElectricalReading(row["voltage_V"], row["current_A"])


class ReverseElectrodialysisRuntime:
    supported_operations = ("configure_electrical_control", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return reverse_electrodialysis_record(state) is not None

    def validate_state(self, state: Any) -> None:
        unpack_reverse_electrodialysis(state)

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = ReverseElectrodialysisSpec.from_dict(
            reverse_electrodialysis_record(state).settings["physical_model"]
        )
        names = electrodialysis_vessels(state, spec.cell_pair)
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            names,
            connection_ids(self.composition, [(names[0], names[1])]),
        )

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return set()

    @property
    def preserved_source_vessels(self) -> set[str]:
        result: set[str] = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    spec = ReverseElectrodialysisSpec.from_dict(
                        reverse_electrodialysis_record(state).settings["physical_model"]
                    )
                    result.update(electrodialysis_vessels(state, spec.cell_pair))
        return result

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        components = {c.id: c for c in self.composition.components}
        for vessel, original in states.items():
            if not self.supports(original):
                continue
            state = original.replace(
                inactive_vessels={v: s for v, s in states.items() if v != vessel}
            )
            _, spec, _, _ = unpack_reverse_electrodialysis(state)
            if self.composition.physics.network.reactions:
                raise ValueError("periodic passive RED excludes homogeneous reactions")
            native = {
                s.species_id: (dict(s.composition), s.charge, s.phase)
                for s in self.composition.physics.network.species
            }
            if any(
                native.get(s.species_id) != (dict(s.composition), s.charge, "liquid")
                for s in spec.properties.database.species
            ):
                raise ValueError("RED database and original liquid identities must agree")
            names = electrodialysis_vessels(state, spec.cell_pair)
            if any(
                capacity > states[name].vessels.vessels[name].max_volume_L
                for name, capacity in zip(names, spec.cell_pair.capacities_L, strict=True)
            ):
                raise ValueError("RED liquid capacity cannot exceed the actual installed vessel")
            if not any(
                components[c["source"]["component"]].vessel == vessel
                and components[c["target"]["component"]].vessel == names[1]
                for c in self.composition.connections
            ):
                raise ValueError("RED requires an actual finite concentrate material connection")

    def configure_flow(self, state: Any, action: dict[str, Any]) -> Any:
        raise ValueError("periodic batch RED has no liquid pump")

    def advance(
        self,
        state: Any,
        action: dict[str, Any],
        *,
        heat: bool = False,
        jacket_program: Any = None,
    ) -> ModelProviderResult:
        # The existing local instrument clock invokes the physical advancer with
        # operation=measure; its approved in-situ acquisition owns the duration.
        if (
            heat
            or jacket_program is not None
            or action.get("operation", "wait") not in {"wait", "measure"}
        ):
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="passive_red_requires_isothermal_wait",
            )
        try:
            record, spec, inventory, program = unpack_reverse_electrodialysis(state)
            result, receipt = advance_reverse_electrodialysis(
                spec, inventory, program, float(action["duration_s"])
            )
            final, dt = result.final_state, result.actual_duration_s
            heat_out = -receipt["bath_heat_into_liquid_J"]
            name = spec.cell_pair.concentrate_vessel_id
            inactive = dict(state.inactive_vessels)
            tank = install_ed_liquid(
                inactive[name], spec.cell_pair, final.amounts_mol[1], final.enthalpies_J[1]
            )
            inactive[name] = tank.replace(
                ledger=advance_ledger(tank.ledger, dt, energy_deltas={"heat_loss_J": heat_out / 2})
            )
            candidate = install_ed_liquid(
                state, spec.cell_pair, final.amounts_mol[0], final.enthalpies_J[0]
            )
            ages = dict(state.vessel_elapsed_s)
            ages[name] = ages.get(name, 0) + dt
            candidate = candidate.replace(
                inactive_vessels=inactive,
                vessel_elapsed_s=ages,
                equipment=upsert_equipment_record(
                    state.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"last_reverse_electrodialysis": receipt},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    cost_delta=0.0,
                    energy_deltas={
                        "flow_electrical_work_J": receipt["electrical_work_into_liquid_J"],
                        "heat_loss_J": heat_out,
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as error:
            return ModelProviderResult(
                {},
                diagnostics={"solver_error": str(error)},
                success=False,
                outcome="numerical_failure",
                failure_reason="passive_red_solver_failed",
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            vessel_thermal_deltas={
                v: {"energy_jacket_J": 0.0, "heat_loss_J": heat_out / 2}
                for v in electrodialysis_vessels(state, spec.cell_pair)
            },
            diagnostics={
                "reverse_electrodialysis": receipt,
                "electrical_control": {
                    "unit": program.unit,
                    "current_A": receipt["final"]["current_A"],
                    "voltage_V": receipt["final"]["voltage_V"],
                },
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
            },
        )
