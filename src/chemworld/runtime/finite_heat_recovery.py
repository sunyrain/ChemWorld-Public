"""Two-stream heat recovery through the original finite-transport dispatcher."""

from math import isclose
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.physchem.finite_heat_recovery import (
    advance_finite_heat_recovery,
    validate_amounts,
)
from chemworld.physchem.finite_heat_recovery_specs import FiniteHeatRecoverySpec, number
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

HEAT_RECOVERY_TYPE = "finite_two_stream_heat_recovery"


def heat_recovery_record(state: Any) -> Any:
    return attached_device_record(state, HEAT_RECOVERY_TYPE)


def heat_recovery_vessels(state: Any, spec: FiniteHeatRecoverySpec) -> tuple[str, ...]:
    names = (
        spec.left_feed_vessel_id,
        state.vessel_id,
        spec.right_feed_vessel_id,
        spec.right_receiver_vessel_id,
    )
    if len(set(names)) != 4:
        raise ValueError("heat recovery requires four distinct finite vessel owners")
    return names


def unpack_heat_recovery(state: Any) -> tuple[Any, FiniteHeatRecoverySpec, TransportInventory]:
    from chemworld.runtime.transport_devices import enthalpy_slots

    record = heat_recovery_record(state)
    if record is None:
        raise ValueError("no finite heat-recovery apparatus is attached")
    spec = FiniteHeatRecoverySpec.from_dict(record.settings["physical_model"])
    amounts, enthalpies = [], []
    for i, name in enumerate(heat_recovery_vessels(state, spec)):
        tank = state if name == state.vessel_id else state.inactive_vessels[name]
        n = dict(tank.species_amounts)
        validate_amounts(spec, n)
        h = spec.enthalpy(i, n, tank.temperature_K)
        fluid_h = spec.properties.enthalpy(n, tank.temperature_K)
        volume = (
            spec.properties.evaluate(n, tank.temperature_K)[2] * 1000 if sum(n.values()) else 0.0
        )
        if (
            not isclose(tank.pressure_Pa, spec.properties.pressure_Pa, abs_tol=1e-6)
            or not isclose(volume, tank.volume_L, rel_tol=1e-8, abs_tol=1e-10)
            or set(tank.phases.phases) != {"liquid"}
            or tank.phases.phases["liquid"].species_amounts_mol != n
        ):
            raise ValueError(
                "heat-recovery original liquid phase, properties and inventory disagree"
            )
        expected = {s.spec.slot_id: s for s in enthalpy_slots(tank, fluid_h, h - fluid_h).slots}
        actual = {s.spec.slot_id: s for s in tank.state_slots.slots}
        if set(expected) != set(actual) or any(
            actual[key].spec != slot.spec
            or actual[key].location_id != name
            or not isclose(
                actual[key].values["enthalpy_J"],
                slot.values["enthalpy_J"],
                rel_tol=1e-8,
                abs_tol=1e-7,
            )
            for key, slot in expected.items()
        ):
            raise ValueError("heat recovery cannot discard or duplicate other owned state slots")
        amounts.append(n)
        enthalpies.append(sum(s.values["enthalpy_J"] for s in actual.values()))
    controls = record.settings.get("flow_controls")
    if not isinstance(controls, dict) or set(controls) != {"left", "right"}:
        raise ValueError("heat recovery requires two independent persisted flow controls")
    for value in controls.values():
        if number(value, "persisted circuit flow") > 20 / 6e7:
            raise ValueError("circuit flow exceeds original action bounds")
    return record, spec, TransportInventory(tuple(amounts), tuple(enthalpies), state.ledger.time_s)


class FiniteHeatRecoveryAdapter:
    supported_operations = ("set_flow_rate", "run_flow", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return heat_recovery_record(state) is not None

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = FiniteHeatRecoverySpec.from_dict(
            heat_recovery_record(state).settings["physical_model"]
        )
        names = heat_recovery_vessels(state, spec)
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            names,
            connection_ids(self.composition, [(names[0], names[1]), (names[2], names[3])]),
        )

    @property
    def preserved_source_vessels(self) -> set[str]:
        names: set[str] = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    spec = FiniteHeatRecoverySpec.from_dict(
                        heat_recovery_record(state).settings["physical_model"]
                    )
                    names.update(heat_recovery_vessels(state, spec))
        return names

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return self.preserved_source_vessels

    def reconcile_initialization(self, transition: Any) -> Any:
        active = transition.state
        locals_ = {active.vessel_id: active, **active.inactive_vessels}
        for state in locals_.values():
            if self.supports(state):
                unpack_heat_recovery(
                    state.replace(
                        inactive_vessels={k: s for k, s in locals_.items() if k != state.vessel_id}
                    )
                )
        return transition

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        installed = {v for v, s in states.items() if self.supports(s)}
        declared = {c.vessel for c in self.composition.components if c.kind == "finite_transport"}
        if not installed <= declared:
            raise ValueError("heat recovery requires the original finite_transport component")
        occupied: set[str] = set()
        for vessel in installed:
            state = states[vessel].replace(
                inactive_vessels={k: v for k, v in states.items() if k != vessel}
            )
            _, spec, _ = unpack_heat_recovery(state)
            names = heat_recovery_vessels(state, spec)
            if occupied.intersection(names):
                raise ValueError("heat-recovery circuits cannot share finite tank ownership")
            occupied.update(names)
            if self.composition.physics.network.reactions:
                raise ValueError("finite heat-recovery slice does not enable reaction kinetics")
            original = {
                s.species_id: (s.formula, s.charge)
                for s in self.composition.physics.network.species
            }
            if any(
                original.get(p.component.identifier) != (p.component.formula, 0)
                for p in spec.properties.packages
            ):
                raise ValueError("original network and heat-recovery component identities disagree")
            self.shared_system(state, "finite-pump-transport")
            for receiver, cap in zip((names[1], names[3]), spec.receiver_capacities_L, strict=True):
                if cap > states[receiver].vessels.vessels[receiver].max_volume_L:
                    raise ValueError("heat-recovery capacity exceeds installed hardware")

    def configure_flow(self, state: Any, action: dict) -> Any:
        record, _, _ = unpack_heat_recovery(state)
        side = action.get("flow_path")
        if side not in {"left", "right"}:
            raise ValueError("select left or right for independent heat-recovery flow control")
        flow = number(action["flow_rate_mL_min"], "requested circuit flow")
        if flow > 20:
            raise ValueError("requested circuit flow exceeds 20 mL/min")
        controls = {**record.settings["flow_controls"], side: flow / 6e7}
        return state.replace(
            equipment=upsert_equipment_record(
                state.equipment,
                equipment_id=record.equipment_id,
                equipment_type=record.equipment_type,
                attached_vessel_id=state.vessel_id,
                batch_reset="preserve",
                settings={"flow_controls": controls, "flow_control": {"configured": True}},
            )
        )

    def advance(
        self, state: Any, action: dict, *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        if heat or jacket_program is not None or "target_temperature_K" in action:
            return ModelProviderResult(
                {},
                diagnostics={},
                success=False,
                outcome="rejected",
                failure_reason="heat_recovery_has_no_external_thermal_control",
            )
        try:
            record, spec, initial = unpack_heat_recovery(state)
            controls = record.settings["flow_controls"]
            result, diagnostic = advance_finite_heat_recovery(
                spec, initial, (controls["left"], controls["right"]), float(action["duration_s"])
            )
            final, dt = result.final_state, result.actual_duration_s
            names = heat_recovery_vessels(state, spec)
            inactive, ages = dict(state.inactive_vessels), dict(state.vessel_elapsed_s)
            from chemworld.runtime.transport_devices import install_tank

            for i, name in enumerate(names):
                if name == state.vessel_id:
                    continue
                tank = install_tank(
                    inactive[name], spec, i, final.amounts_mol[i], final.enthalpies_J[i]
                )
                deltas = (
                    {"heat_loss_J": diagnostic["circuit_receipts"]["right"]["motor_loss_J"]}
                    if i == 3
                    else {}
                )
                inactive[name] = tank.replace(
                    ledger=advance_ledger(tank.ledger, dt, energy_deltas=deltas)
                )
                ages[name] = ages.get(name, 0.0) + dt
            candidate = install_tank(state, spec, 1, final.amounts_mol[1], final.enthalpies_J[1])
            candidate = candidate.replace(
                inactive_vessels=inactive,
                vessel_elapsed_s=ages,
                equipment=upsert_equipment_record(
                    candidate.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"last_heat_recovery": diagnostic},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    cost_delta=diagnostic["shaft_work_J"] / 250000,
                    energy_deltas={
                        "flow_pump_work_J": diagnostic["shaft_work_J"],
                        "heat_loss_J": diagnostic["motor_loss_J"],
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="finite_heat_recovery_failed",
                diagnostics={"solver_error": str(exc), "root": getattr(exc, "diagnostics", {})},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            vessel_thermal_deltas={
                names[0]: {},
                names[1]: {"heat_loss_J": diagnostic["circuit_receipts"]["left"]["motor_loss_J"]},
                names[2]: {},
                names[3]: {"heat_loss_J": diagnostic["circuit_receipts"]["right"]["motor_loss_J"]},
            },
            diagnostics={
                "heat_recovery": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
            },
        )
