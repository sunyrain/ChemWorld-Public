"""Zero-order device adapter of the original finite-transport dispatcher."""

from math import isclose
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.physchem.transport_devices import TransportInventory
from chemworld.physchem.water_treatment_economics import (
    WaterTreatmentEconomicsSpec,
    accumulate_treatment_receipts,
    empty_treatment_receipt,
    evaluate_water_treatment_economics,
    validate_treatment_receipt,
)
from chemworld.physchem.zero_order_treatment import (
    advance_zero_order_treatment,
    validate_treatment_amounts,
)
from chemworld.physchem.zero_order_treatment_specs import ZeroOrderTreatmentSpec
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

TREATMENT_TYPE = "zero_order_water_treatment"


def treatment_record(state: Any) -> Any:
    return attached_device_record(state, TREATMENT_TYPE)


def treatment_vessels(state: Any, spec: ZeroOrderTreatmentSpec) -> tuple[str, ...]:
    names = (state.vessel_id, spec.product_vessel_id, *(s.reject_vessel_id for s in spec.stages))
    if len(names) != len(set(names)):
        raise ValueError("treatment feed/product/concentrates must have unique physical owners")
    return names


def unpack_treatment(state: Any) -> tuple[Any, ZeroOrderTreatmentSpec, TransportInventory]:
    from chemworld.runtime.transport_devices import enthalpy_slots

    record = treatment_record(state)
    if record is None:
        raise ValueError("no empirical water-treatment equipment is attached")
    spec = ZeroOrderTreatmentSpec.from_dict(record.settings["physical_model"])
    amounts, enthalpies = [], []
    for name in treatment_vessels(state, spec):
        tank = state if name == state.vessel_id else state.inactive_vessels[name]
        n = dict(tank.species_amounts)
        validate_treatment_amounts(spec, n)
        h = spec.properties.enthalpy(n, spec.temperature_K)
        volume = (
            spec.properties.evaluate(n, spec.temperature_K)[2] * 1000 if sum(n.values()) else 0.0
        )
        if (
            not isclose(tank.temperature_K, spec.temperature_K, abs_tol=1e-8)
            or not isclose(tank.pressure_Pa, spec.properties.pressure_Pa, abs_tol=1e-5)
            or not isclose(tank.volume_L, volume, rel_tol=1e-8, abs_tol=1e-10)
            or set(tank.phases.phases) != {"liquid"}
            or dict(tank.phases.phases["liquid"].species_amounts_mol) != n
        ):
            raise ValueError("treatment original phase and declared liquid n/T/P/V disagree")
        expected = {s.spec.slot_id: s for s in enthalpy_slots(tank, h, 0.0).slots}
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
            raise ValueError("treatment cannot discard or duplicate other owned state slots")
        amounts.append(n)
        enthalpies.append(h)
    receipt = validate_treatment_receipt(
        record.settings.get(
            "treatment_totals",
            empty_treatment_receipt(spec),
        )
    )
    if set(receipt["reject_volumes_m3"]) != {s.stage_id for s in spec.stages}:
        raise ValueError("device throughput history belongs to a different chain")
    return record, spec, TransportInventory(tuple(amounts), tuple(enthalpies), state.ledger.time_s)


def analyze_treatment_world_economics(state: Any, policy: WaterTreatmentEconomicsSpec) -> dict:
    record, spec, _ = unpack_treatment(state)
    return evaluate_water_treatment_economics(
        record.settings.get("treatment_totals", empty_treatment_receipt(spec)),
        policy,
    )


class ZeroOrderTreatmentAdapter:
    supported_operations = ("set_flow_rate", "run_flow", "wait")

    def __init__(self, composition: Any = None) -> None:
        self.composition = composition

    def supports(self, state: Any) -> bool:
        return treatment_record(state) is not None

    def shared_system(self, state: Any, strategy_id: str) -> Any:
        from chemworld.foundation.lab_clock import SharedAdvanceSystem
        from chemworld.runtime.shared_systems import connection_ids

        spec = ZeroOrderTreatmentSpec.from_dict(treatment_record(state).settings["physical_model"])
        names = treatment_vessels(state, spec)
        return SharedAdvanceSystem(
            state.vessel_id,
            strategy_id,
            names,
            connection_ids(self.composition, [(names[0], n) for n in names[1:]]),
        )

    @property
    def preserved_source_vessels(self) -> set[str]:
        names: set[str] = set()
        if self.composition is not None and self.composition.physics is not None:
            for state in self.composition.physics.initial_states.values():
                if self.supports(state):
                    spec = ZeroOrderTreatmentSpec.from_dict(
                        treatment_record(state).settings["physical_model"],
                    )
                    names.update(treatment_vessels(state, spec))
        return names

    @property
    def protected_lifecycle_vessels(self) -> set[str]:
        return set()

    def reconcile_initialization(self, transition: Any) -> Any:
        active = transition.state
        locals_ = {active.vessel_id: active, **active.inactive_vessels}
        for state in locals_.values():
            if self.supports(state):
                unpack_treatment(
                    state.replace(
                        inactive_vessels={
                            key: local for key, local in locals_.items() if key != state.vessel_id
                        }
                    )
                )
        return transition

    def validate_world(self) -> None:
        states = self.composition.physics.initial_states
        installed = {name for name, state in states.items() if self.supports(state)}
        declared = {c.vessel for c in self.composition.components if c.kind == "finite_transport"}
        if not installed <= declared:
            raise ValueError("treatment equipment requires the original finite_transport component")
        occupied: set[str] = set()
        for name in installed:
            state = states[name].replace(
                inactive_vessels={k: v for k, v in states.items() if k != name}
            )
            _, spec, _ = unpack_treatment(state)
            members = treatment_vessels(state, spec)
            if occupied.intersection(members):
                raise ValueError("treatment chains cannot share finite physical inventory")
            occupied.update(members)
            if self.composition.physics.network.reactions:
                raise ValueError("zero-order split does not implement reaction kinetics")
            original = {
                s.species_id: (s.formula, s.charge)
                for s in self.composition.physics.network.species
            }
            if any(
                original.get(p.component.identifier) != (p.component.formula, 0)
                for p in spec.properties.packages
            ):
                raise ValueError("original network and treatment component identities disagree")
            self.shared_system(state, "finite-pump-transport")
            for vessel, capacity in zip(
                members[1:],
                (spec.product_capacity_L, *(s.reject_capacity_L for s in spec.stages)),
                strict=True,
            ):
                if capacity > states[vessel].vessels.vessels[vessel].max_volume_L:
                    raise ValueError("treatment collector capacity exceeds installed hardware")

    def configure_flow(self, state: Any, action: dict) -> Any:
        from chemworld.runtime.transport_devices import configure_liquid_flow

        record, _, _ = unpack_treatment(state)
        return configure_liquid_flow(state, action, record)

    def advance(
        self,
        state: Any,
        action: dict,
        *,
        heat: bool = False,
        jacket_program: Any = None,
    ) -> ModelProviderResult:
        from chemworld.runtime.transport_devices import install_tank

        try:
            record, spec, inventory = unpack_treatment(state)
            if (
                heat
                or jacket_program is not None
                or action.get("target_temperature_K", spec.temperature_K) != spec.temperature_K
            ):
                return ModelProviderResult(
                    {},
                    diagnostics={},
                    success=False,
                    outcome="rejected",
                    failure_reason="treatment_requires_declared_isotherm",
                )
            flow = record.settings.get("flow_control", {}).get("requested_flow_m3_s", 0.0)
            result, diagnostic = advance_zero_order_treatment(
                spec,
                inventory,
                flow,
                float(action["duration_s"]),
            )
            final, dt = result.final_state, result.actual_duration_s
            inactive, ages = dict(state.inactive_vessels), dict(state.vessel_elapsed_s)
            for i, name in enumerate(treatment_vessels(state, spec)[1:], 1):
                tank = install_tank(
                    inactive[name], spec, i, final.amounts_mol[i], final.enthalpies_J[i]
                )
                inactive[name] = tank.replace(ledger=advance_ledger(tank.ledger, dt))
                ages[name] = ages.get(name, 0.0) + dt
            candidate = install_tank(state, spec, 0, final.amounts_mol[0], final.enthalpies_J[0])
            receipt = diagnostic["receipt"]
            total = accumulate_treatment_receipts(
                record.settings.get("treatment_totals", empty_treatment_receipt(spec)),
                receipt,
            )
            bath = receipt["bath_heat_into_liquid_J"]
            candidate = candidate.replace(
                inactive_vessels=inactive,
                vessel_elapsed_s=ages,
                equipment=upsert_equipment_record(
                    state.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"treatment_totals": total, "last_treatment": diagnostic},
                ),
                ledger=advance_ledger(
                    state.ledger,
                    dt,
                    cost_delta=(receipt["pump_work_J"] + receipt["additional_electrical_work_J"])
                    / 250000,
                    energy_deltas={
                        "flow_pump_work_J": receipt["pump_work_J"],
                        "flow_electrical_work_J": receipt["additional_electrical_work_J"],
                        "energy_jacket_J": max(bath, 0.0),
                        "heat_loss_J": receipt["motor_loss_J"] + max(-bath, 0.0),
                    },
                ),
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="zero_order_treatment_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=dt,
            outcome="stopped" if result.stopping_event else "success",
            diagnostics={
                "treatment": diagnostic,
                "solver": result.solver_diagnostic,
                "stopping_event": result.stopping_event,
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
            },
        )
