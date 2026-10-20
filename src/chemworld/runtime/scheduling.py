"""Finite FIFO scheduling inside the existing transaction, not a second runtime.

One wait commits at most one laboratory time slice. All local strategies see
the same interval; a failed solve discards the complete candidate, including
sampling, queue changes and RNG draws. Completion is the publication boundary.
"""

from copy import deepcopy
from dataclasses import replace
from math import isclose, isfinite

import numpy as np

from chemworld.foundation import Observation, instrument_equipment_id
from chemworld.foundation.balances import MaterialExchange, TransitionBalance
from chemworld.foundation.lab_clock import LabClockState
from chemworld.foundation.solvents import SolventInventory
from chemworld.foundation.state_ledgers import EquipmentLedger
from chemworld.runtime.component_network import _local, reconnect_device
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.solvent_transport import finish_solvent_transition, initialize_solvents

ALLOWED = {"queue_lab_heat", "queue_lab_measure", "wait", "select_vessel", "terminate"}


def validate_lab(composition):
    physics = composition.physics
    spec = physics.lab_clock
    if spec is None:
        if any(c.kind == "lab_schedule" for c in composition.components):
            raise ValueError("lab_schedule requires an explicit shared clock declaration")
        return
    services = {factory.service_id for factory in composition.modules.service_order()}
    if "lab_schedule" not in services:
        raise ValueError("shared clock requires its declared scheduling service")
    if not 1 <= len(physics.initial_states) <= 2 or composition.connections or physics.feed_sources:
        raise ValueError("shared lab supports one or two independent closed vessels")
    kinds = {c.kind for c in composition.components}
    if kinds - {
        "thermal",
        "reaction",
        "surface_reaction",
        "stochastic",
        "observation",
        "lab_schedule",
    }:
        raise ValueError("component has no declared shared-clock advance strategy")
    states = physics.initial_states
    discrete = {key for key, state in states.items() if state.process.stochastic_continuation}
    if set(physics.thermal_models) | discrete != set(states):
        raise ValueError("every shared-clock vessel needs a native thermal or SSA strategy")
    if any(
        (
            physics.flash_models,
            physics.liquid_models,
            physics.column_models,
            physics.concentration_models,
            physics.crystal_models,
            physics.sorbent_models,
        )
    ):
        raise ValueError("shared lab excludes interconnected/open and alternate phase strategies")
    if any(
        state.lab_clock or state.ledger.time_s or state.inactive_vessels
        for state in states.values()
    ):
        raise ValueError("lab initial states must be independent, zero-time physical states")
    equipment = [r for state in states.values() for r in state.equipment.equipment.values()]
    heaters = [r for r in equipment if r.equipment_type == "thermal_device"]
    if [r.equipment_id for r in heaters] != ([] if spec.heater_id is None else [spec.heater_id]):
        raise ValueError("lab declares exactly its one physical heater identity")
    if heaters and heaters[0].settings["thermal_device"].get("enabled", True):
        raise ValueError("shared heater starts disabled; queue owns its activation")
    if spec.analyzer is not None:
        instrument = composition.modules.instrument_specs().get(spec.analyzer)
        if instrument is None:
            raise ValueError("shared analyzer must be an installed native instrument")
        contract = instrument.contract
        if (
            instrument.native_observer is None
            or contract.clock_semantics != "offline_snapshot"
            or contract.host_evolves_during_analysis
            or contract.latency_s
            or not contract.analysis_time_s > 0
            or contract.requires_terminated
        ):
            raise ValueError("lab analyzer needs a nonterminal native offline snapshot contract")
        if sum(r.equipment_id == instrument_equipment_id(spec.analyzer) for r in equipment) != 1:
            raise ValueError("lab analyzer must have exactly one physical owner")


def initialize_lab(state, composition, rng):
    if composition is None or composition.physics is None or composition.physics.lab_clock is None:
        return state
    # Shared systems are all enabled from t=0. Initialize declared carrier
    # stocks for every vessel now, not lazily when it is selected later.
    for local in local_states(state).values():
        state = install_local(state, initialize_solvents(local))
    return state.replace(lab_clock=LabClockState(detector_rng=deepcopy(rng.bit_generator.state)))


def public_lab(state):
    if state.lab_clock is None:
        return {"mode": "local"}
    return {"time_s": state.ledger.time_s, **state.lab_clock.to_dict(include_hidden=False)}


def local_states(state):
    return {state.vessel_id: _local(state), **state.inactive_vessels}


def install_local(state, local):
    if state.vessel_id == local.vessel_id:
        return local.replace(
            ledger=state.ledger,
            lab_clock=state.lab_clock,
            inactive_vessels=state.inactive_vessels,
            vessel_elapsed_s=state.vessel_elapsed_s,
            samples=state.samples,
            solvent_accounting=state.solvent_accounting,
        )
    return state.replace(inactive_vessels={**state.inactive_vessels, local.vessel_id: local})


def fail(reason, *, outcome="rejected", **diagnostics):
    return ModelProviderResult(
        {},
        state=None,
        success=False,
        outcome=outcome,
        failure_reason=reason,
        diagnostics=diagnostics,
    )


class LabScheduleServices:
    def __init__(self, context):
        self.composition = context.composition
        self.spec = (
            None
            if context.composition is None or context.composition.physics is None
            else context.composition.physics.lab_clock
        )
        self.modules = context.modules
        self.constitution = context.constitution
        self.thermal = context.services["reaction_thermal"]
        self.instrument_cost = context.services["instrument_cost"]

    def queue(self, state, action):
        lab = state.lab_clock
        if lab is None or self.spec is None or lab.stopped:
            return fail("shared_lab_not_available")
        if lab.next_sequence >= self.spec.capacity:
            return fail("finite_lab_job_capacity")
        kind = "heat" if action["operation"] == "queue_lab_heat" else "measure"
        job = {
            "sequence": lab.next_sequence,
            "kind": kind,
            "vessel": state.vessel_id,
            "queued_at": state.ledger.time_s,
        }
        if kind == "heat":
            target, duration = float(action["target_temperature_K"]), float(action["duration_s"])
            model = self.composition.physics.thermal_models.get(state.vessel_id)
            if self.spec.heater_id is None or model is None:
                return fail("shared_heater_not_supported")
            if not (
                isfinite(duration)
                and 0 < duration <= 14400
                and isfinite(target)
                and model.minimum_temperature_K <= target <= model.maximum_temperature_K
            ):
                return fail("lab_heat_outside_declared_domain")
            job.update(duration_s=duration, target_temperature_K=target)
        else:
            instrument = self.spec.analyzer
            allowed = self.composition.task.instruments
            available = state.metadata["component_network"]["instruments"][state.vessel_id]
            if (
                instrument is None
                or instrument not in available
                or (allowed is not None and instrument not in allowed)
            ):
                return fail("shared_analyzer_not_allowed")
            job.update(instrument=instrument)
        return state.replace(
            lab_clock=replace(lab, next_sequence=lab.next_sequence + 1, queued=(*lab.queued, job))
        )

    def _acquire(self, state, job):
        """Existing sampler/observer owns physics; queue only owns timing/publication."""
        from chemworld.runtime.material_balance import ServiceMaterialBoundary

        instrument = job["instrument"]
        state = reconnect_device(state, instrument_equipment_id(instrument), job["vessel"])
        before = initialize_solvents(local_states(state)[job["vessel"]])
        action = {"operation": "measure", "instrument": instrument}
        result = self.instrument_cost.apply_measurement_cost(before, action)
        if result.outcome != "success":
            return state, job, result
        after = finish_solvent_transition(before, result.state, action)
        # Reuse the installed trusted receipt resolver, including persistent
        # chromatography carryover and integer solute capture. No balance deficit
        # is ever used to invent an outlet.
        boundary = ServiceMaterialBoundary(self._domains).resolve(before, after, action)
        spec = self.modules.instrument_specs()[instrument]
        rng = np.random.Generator(np.random.PCG64())
        rng.bit_generator.state = deepcopy(state.lab_clock.detector_rng)
        specimen = result.measurement_state.replace(equipment=after.equipment, ledger=after.ledger)
        try:
            reading = spec.native_observer(specimen, action, rng)
            if (
                not isinstance(reading, Observation)
                or reading.instrument_id != instrument
                or reading.cost != spec.definition.cost
                or reading.sample_consumed_L != spec.definition.sample_volume_L
                or not self.constitution.check_observation(reading).passed
            ):
                raise ValueError("queued native observer violates its declared contract")
        except (ValueError, ArithmeticError) as exc:
            # Detector failure is not a failed physical sample. Keep executed
            # consumption, instrument history and analysis occupancy; withhold
            # its invalid result until the same publication boundary.
            reading = None
            job = {**job, "measurement_status": "invalid", "readout_error": str(exc)}
            rng.bit_generator.state = deepcopy(state.lab_clock.detector_rng)
        else:
            job = {**job, "measurement_status": "valid"}
        now = state.ledger.time_s
        duration = spec.contract.analysis_time_s
        job = {**job, "started_at": now, "available_at": now + duration}
        timing = {
            "sample_id": f"lab:{job['sequence']}:{job['vessel']}",
            "sampled_at": {"clock_id": "lab", "time_s": now},
            "available_at": {"clock_id": "lab", "time_s": now + duration},
        }
        if reading is not None:
            native = reading.native_record()
            native.update(timing)
            native["channels"] = [{**channel, **timing} for channel in native["channels"]]
            job["observation"] = native
        equipment = after.equipment.equipment.copy()
        identity = instrument_equipment_id(instrument)
        device = equipment[identity]
        settings = deepcopy(device.settings)
        settings["elapsed_time_s"] -= duration
        acquisition = settings["diagnostics"]["acquisition"]
        acquisition.update(timing, reactor_advance_s=0.0)
        settings["execution_history"][-1]["diagnostics"]["acquisition"] = deepcopy(acquisition)
        equipment[identity] = replace(device, settings=settings, status="busy")
        after = after.replace(
            equipment=EquipmentLedger(equipment),
            ledger=after.ledger.with_updates(analysis_time_s=before.ledger.analysis_time_s),
        )
        state = install_local(state, after)
        state = state.replace(
            solvent_accounting=replace(
                state.solvent_accounting,
                removed=state.solvent_accounting.removed
                + SolventInventory(
                    tuple(
                        a - b
                        for a, b in zip(
                            after.solvent_accounting.removed.volumes_L,
                            before.solvent_accounting.removed.volumes_L,
                            strict=True,
                        )
                    )
                ),
            ),
            ledger=state.ledger.with_updates(
                cost=state.ledger.cost + after.ledger.cost - before.ledger.cost,
                sample_consumed_L=state.ledger.sample_consumed_L + spec.definition.sample_volume_L,
            ),
            lab_clock=replace(
                state.lab_clock,
                detector_rng=deepcopy(rng.bit_generator.state),
                exchanges=(
                    *state.lab_clock.exchanges,
                    *(
                        {
                            **receipt.to_dict(),
                            "exchange_id": f"lab:{job['sequence']}:{receipt.exchange_id}",
                        }
                        for receipt in boundary.exchanges
                    ),
                ),
            ),
        )
        return state, job, None

    def advance(self, state, action):
        from chemworld.runtime.coupled_thermal import reconnect_thermal_device, set_thermal_enabled

        if state.lab_clock.stopped:
            return fail("lab_stopped_requires_full_reset")
        original = state
        running = list(state.lab_clock.running)
        queued = []
        # Start at most one request per device. Global sequence breaks ties;
        # waiting jobs do not reserve specimens from a changing reactor.
        for job in state.lab_clock.queued:
            if any(item["kind"] == job["kind"] for item in running):
                queued.append(job)
                continue
            if job["kind"] == "heat":
                state = reconnect_thermal_device(state, self.spec.heater_id, job["vessel"])
                job = {
                    **job,
                    "started_at": state.ledger.time_s,
                    "available_at": state.ledger.time_s + job["duration_s"],
                }
            else:
                try:
                    state, job, failure = self._acquire(state, job)
                except (ValueError, ArithmeticError) as exc:
                    return fail("lab_acquisition_failed", detail=str(exc))
                if failure is not None:
                    return failure
            running.append(job)
        base = state
        now = state.ledger.time_s
        dt = min(
            float(action["duration_s"]),
            min((job["available_at"] - now for job in running), default=float("inf")),
        )
        locals_ = local_states(state)

        def solve(duration):
            results = {}
            for key, local in locals_.items():
                heat = next(
                    (job for job in running if job["kind"] == "heat" and job["vessel"] == key), None
                )
                interval = {
                    "operation": "heat" if heat else "wait",
                    "duration_s": duration,
                    "stirring_speed_rpm": action.get("stirring_speed_rpm", 600),
                }
                if heat:
                    interval["target_temperature_K"] = heat["target_temperature_K"]
                results[key] = self.thermal.advancer.batch(local, interval, heat=bool(heat))
            return results

        results = solve(dt)
        failure = next(
            (r for r in results.values() if r.outcome in {"rejected", "numerical_failure"}), None
        )
        if failure is not None:
            return replace(failure, state=None, actual_duration_s=0)
        stopped = any(r.outcome != "success" for r in results.values())
        stopping_events = {
            key: dict(result.diagnostics)
            for key, result in results.items()
            if result.outcome != "success"
        }
        earliest = min(r.actual_duration_s for r in results.values())
        if earliest < dt:
            dt = earliest
            results = solve(dt)
            failure = next(
                (r for r in results.values() if r.outcome in {"rejected", "numerical_failure"}),
                None,
            )
            if failure is not None:
                return replace(failure, state=None, actual_duration_s=0)
        if any(
            not isclose(r.actual_duration_s, dt, abs_tol=1e-7, rel_tol=1e-10)
            for r in results.values()
        ):
            return fail("lab_stop_alignment_failed", outcome="numerical_failure")
        ledger = base.ledger.to_dict()
        ledger["time_s"] = now + dt
        for key, result in results.items():
            before, after = locals_[key], result.state
            thermal = after.thermal.vessels[key]
            after = after.replace(
                thermal=replace(
                    after.thermal,
                    vessels={
                        key: replace(
                            thermal,
                            **{
                                name: getattr(after.ledger, name)
                                for name in ("energy_jacket_J", "heat_reaction_J", "heat_loss_J")
                            },
                        )
                    },
                )
            )
            for name in ledger.keys() - {"time_s", "risk"}:
                ledger[name] += getattr(after.ledger, name) - getattr(before.ledger, name)
            state = install_local(state, after)
        state = state.replace(
            ledger=replace(base.ledger, **ledger),
            vessel_elapsed_s=dict.fromkeys(locals_, now + dt),
        )
        # Analysis time is resource occupancy, not an extra physical clock.
        for job in running:
            if job["kind"] == "measure":
                local = local_states(state)[job["vessel"]]
                equipment = local.equipment.equipment.copy()
                identity = instrument_equipment_id(job["instrument"])
                device = equipment[identity]
                equipment[identity] = replace(
                    device,
                    settings={
                        **device.settings,
                        "elapsed_time_s": device.settings["elapsed_time_s"] + dt,
                    },
                )
                state = install_local(state, local.replace(equipment=EquipmentLedger(equipment)))
                state = state.replace(
                    ledger=state.ledger.with_updates(
                        analysis_time_s=state.ledger.analysis_time_s + dt
                    )
                )
        completed, active = list(state.lab_clock.completed), []
        for job in sorted(running, key=lambda j: j["sequence"]):
            if isclose(job["available_at"], now + dt, abs_tol=1e-9, rel_tol=1e-12):
                completed.append({**job, "status": "completed"})
                if job["kind"] == "heat":
                    local = local_states(state)[job["vessel"]]
                    state = install_local(state, set_thermal_enabled(local, False))
                else:
                    local = local_states(state)[job["vessel"]]
                    equipment = dict(local.equipment.equipment)
                    identity = instrument_equipment_id(job["instrument"])
                    equipment[identity] = replace(equipment[identity], status="used")
                    state = install_local(
                        state, local.replace(equipment=EquipmentLedger(equipment))
                    )
            else:
                active.append(job)
        state = state.replace(
            lab_clock=replace(
                state.lab_clock,
                queued=tuple(queued),
                running=tuple(active),
                completed=tuple(completed),
                stopped=stopped,
            )
        )
        return ModelProviderResult(
            {},
            state=state,
            actual_duration_s=dt,
            outcome="stopped" if stopped else "success",
            diagnostics={
                "clock_mode": "shared",
                "stopping_events": stopping_events,
                "vessels_advanced": len(results),
                "local_outcomes": {
                    key: {"outcome": r.outcome, **r.diagnostics} for key, r in results.items()
                },
                "completed_sequences": [
                    job["sequence"] for job in completed[len(original.lab_clock.completed) :]
                ],
            },
        )

    @staticmethod
    def boundary(before, after):
        previous, current = before.lab_clock.exchanges, after.lab_clock.exchanges
        if current[: len(previous)] != previous:
            raise ValueError("shared sampling history was rewritten")
        return TransitionBalance(
            tuple(
                MaterialExchange(**{**record, "amounts": tuple(record["amounts"].items())})
                for record in current[len(previous) :]
            )
        )
