"""Finite FIFO scheduling inside the existing transaction, not a second runtime.

One wait commits at most one laboratory time slice. All local strategies see
the same interval; a failed solve discards the complete candidate, including
sampling, queue changes and RNG draws. Completion is the publication boundary.
"""

from copy import deepcopy
from dataclasses import replace
from math import fsum, isclose, isfinite
from types import SimpleNamespace

import numpy as np

from chemworld.foundation.balances import (
    MaterialExchange,
    TransitionBalance,
    formal_inventory_exclusions,
)
from chemworld.foundation.lab_clock import LabClockState
from chemworld.foundation.solvents import SolventInventory, total_solvents
from chemworld.foundation.state_helpers import INSTRUMENT_DEVICE_SELECTION
from chemworld.foundation.state_ledgers import EquipmentLedger
from chemworld.runtime.component_network import _local, reconnect_device
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.shared_systems import declared_shared_systems
from chemworld.runtime.solvent_transport import finish_solvent_transition, initialize_solvents

ALLOWED = {"queue_lab_heat", "queue_lab_measure", "wait", "select_vessel", "terminate",
           "set_flow_rate", "configure_electrical_control", "queue_electrical_stage",
           "add_solvent", "route_material", "create_container", "transfer_material"}


def shared_operation_allowed(operation, state, modules, action=None):
    if operation in ALLOWED:
        return True
    if operation == "set_model_parameters":
        return state.phase == "formal"
    if operation != "measure" or state.phase != "formal":
        return False
    from chemworld.world.operations import instrument_name

    specs = modules.instrument_specs()
    selected = (None if action is None
                else instrument_name(action.get("instrument"), modules=modules))
    return any(
        (selected is None or identity == selected)
        and spec.contract.input_state_schema.get("required_phase") == "formal"
        and spec.contract.clock_semantics in {"instantaneous_snapshot", "in_situ_end"}
        and spec.contract.sample_consumption_L == 0
        and not spec.contract.destructive
        and (spec.native_observer is not None or spec.observer_factory is not None)
        for identity, spec in specs.items()
    )


def lab_device_choices(state, kind, *, allowed_instruments=None):
    if state.lab_clock is None:
        return ()
    available = state.metadata.get("component_network", {}).get("instruments", {}).get(
        state.vessel_id, (),
    )
    records = [record for local in (state, *state.inactive_vessels.values())
               for record in local.equipment.equipment.values()]
    if kind == "heat":
        return tuple(sorted(r.equipment_id for r in records
                            if r.equipment_type == "thermal_device"))
    return tuple(sorted(
        record.equipment_id for record in records
        if record.settings.get("lab_analyzer_instrument") in available
        and (allowed_instruments is None
             or record.settings.get("lab_analyzer_instrument") in allowed_instruments)
    ))


def lab_device_selected(state, action, kind):
    choices = lab_device_choices(state, kind)
    return (len(choices) == 1 if "device" not in action
            else isinstance(action["device"], str) and action["device"] in choices)


def device_record(state, identity):
    records = [record for local in (state, *state.inactive_vessels.values())
               for record in local.equipment.equipment.values() if record.equipment_id == identity]
    if len(records) != 1:
        raise ValueError("laboratory device must have exactly one physical owner")
    return records[0]


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
    if not 1 <= len(physics.initial_states) <= spec.vessel_capacity:
        raise ValueError("shared lab exceeds its declared vessel capacity")
    if physics.feed_sources:
        raise ValueError("shared lab does not accept undeclared external feed operations")
    states = physics.initial_states
    declared_shared_systems(composition)
    if any(
        state.lab_clock or state.ledger.time_s or state.inactive_vessels
        for state in states.values()
    ):
        raise ValueError("lab initial states must be independent, zero-time physical states")
    equipment = [r for state in states.values() for r in state.equipment.equipment.values()]
    heaters = [r for r in equipment if r.equipment_type == "thermal_device"]
    if sorted(r.equipment_id for r in heaters) != list(spec.heater_ids):
        raise ValueError("lab declares exactly its physical heater identities")
    if any(record.settings["thermal_device"].get("enabled", True) for record in heaters):
        raise ValueError("shared heater starts disabled; queue owns its activation")
    for analyzer in spec.analyzers:
        instrument = composition.modules.instrument_specs().get(analyzer.instrument)
        if instrument is None:
            raise ValueError("shared analyzer must be an installed native instrument")
        contract = instrument.contract
        if (
            instrument.observer_for(composition) is None
            or contract.clock_semantics != "offline_snapshot"
            or contract.host_evolves_during_analysis
            or contract.latency_s
            or not contract.analysis_time_s > 0
            or contract.requires_terminated
        ):
            raise ValueError("lab analyzer needs a nonterminal native offline snapshot contract")
        matches = [r for r in equipment if r.equipment_id == analyzer.device_id]
        if len(matches) != 1 or matches[0].equipment_type != "instrument":
            raise ValueError("lab analyzer must have exactly one physical owner")
        if matches[0].settings.get("instrument_id", analyzer.instrument) != analyzer.instrument:
            raise ValueError("analyzer identity and installed instrument type disagree")


def initialize_lab(state, composition, rng, *, detector_noise=None):
    if composition is None or composition.physics is None or composition.physics.lab_clock is None:
        return state
    # Shared systems are all enabled from t=0. Initialize declared carrier
    # stocks for every vessel now, not lazily when it is selected later.
    roles = {analyzer.device_id: analyzer.instrument
             for analyzer in composition.physics.lab_clock.analyzers}
    for local in local_states(state).values():
        equipment = {
            identity: replace(record, settings={**record.settings,
                                               "lab_analyzer_instrument": roles[identity]})
            if identity in roles else record
            for identity, record in local.equipment.equipment.items()
        }
        local = local.replace(equipment=EquipmentLedger(equipment))
        state = install_local(state, initialize_solvents(local))
    # Establish the one laboratory basis from every actual initial stock.
    # Summing active-first would change its last bit when declaration order
    # chooses a different initially active vessel.
    records = {state.vessel_id: state.replace(inactive_vessels={}), **state.inactive_vessels}
    stocks = [total_solvents(records[key]).volumes_L for key in sorted(records)]
    return state.replace(
        solvent_accounting=replace(
            state.solvent_accounting,
            initial=SolventInventory(tuple(fsum(values[i] for values in stocks) for i in range(4))),
        ),
        lab_clock=LabClockState(
            detector_rng=deepcopy(rng.bit_generator.state),
            detector_noise=detector_noise or {},
        )
    )


def public_lab(state):
    if state.lab_clock is None:
        return {"mode": "local"}
    return {"time_s": state.ledger.time_s, **state.lab_clock.to_dict(include_hidden=False)}


def local_states(state):
    # Numerical assembly and resource accumulation must not depend on the
    # selected vessel or on input/JSON mapping enumeration order.
    states = {state.vessel_id: _local(state), **state.inactive_vessels}
    return {key: states[key] for key in sorted(states)}


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
    # Resource and carrier accounting belongs to the global inventory. Callers
    # consume the local candidate's actual deltas before installation. Retain
    # the same local thermal basis regardless of which vessel is selected.
    local = _local(local.replace(
        vessel_elapsed_s={local.vessel_id: local.ledger.time_s},
    ))
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


def ready_lab_jobs(state):
    """One FIFO/device admission rule, also used by resource preflight."""
    if state.lab_clock is None or state.lab_clock.stopped:
        return ()
    occupied = list(state.lab_clock.running)
    ready = []
    for job in sorted(state.lab_clock.queued, key=lambda item: item["sequence"]):
        if any(item["device"] == job["device"] or (
            item["kind"] == job["kind"] == "heat" and item["vessel"] == job["vessel"]
        ) for item in occupied):
            continue
        ready.append(job)
        occupied.append(job)
    return tuple(ready)


def lab_instrument_reservation(state, action, *, modules=None):
    """Reserve potential starts, without advancing physics or reading an instrument.

    A wait commits one scheduling slice. A timed formal acquisition can span
    several queued boundaries; its FIFO plan uses the same admission rule and
    declared analysis durations. A physical stop can only reduce actual starts.
    """
    if state is None or state.lab_clock is None:
        return {}
    if action.get("operation") == "measure":
        return _acquisition_reservation(state, action, modules)
    if action.get("operation") != "wait":
        return {}
    counts = {}
    for job in ready_lab_jobs(state):
        if job["kind"] == "measure":
            counts[job["instrument"]] = counts.get(job["instrument"], 0) + 1
    return counts


def _acquisition_reservation(state, action, modules):
    from chemworld.world.operations import instrument_name

    if modules is None or state.phase != "formal":
        return {}
    specs = modules.instrument_specs()
    try:
        identity = instrument_name(action.get("instrument"), modules=modules)
        contract = specs[identity].contract
    except (KeyError, ValueError, TypeError):
        return {}  # Invalid actions still go through the normal schema gate.
    if (contract.clock_semantics != "in_situ_end" or not contract.physical_duration_s
            or contract.sample_consumption_L or contract.destructive):
        return {}
    counts, lab = {identity: 1}, state.lab_clock
    now, horizon = state.ledger.time_s, state.ledger.time_s+contract.physical_duration_s
    for _ in range(len(lab.queued)+len(lab.running)+1):
        ready = ready_lab_jobs(SimpleNamespace(lab_clock=lab))
        started = {job["sequence"] for job in ready}
        running = list(lab.running)
        for job in ready:
            if job["kind"] == "measure":
                counts[job["instrument"]] = counts.get(job["instrument"], 0)+1
                duration = specs[job["instrument"]].contract.analysis_time_s
            else:
                duration = job["duration_s"]
            running.append({**job, "available_at": now+duration})
        if not running:
            break
        next_time = min(job["available_at"] for job in running)
        if next_time >= horizon:
            break  # No new acquisition starts after the observation endpoint.
        lab = replace(lab,
            queued=tuple(job for job in lab.queued if job["sequence"] not in started),
            running=tuple(job for job in running if job["available_at"] > next_time))
        now = next_time
    return counts


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
        self.systems = () if self.spec is None else declared_shared_systems(self.composition)

    def queue(self, state, action):
        lab = state.lab_clock
        if lab is None or self.spec is None or lab.stopped:
            return fail("shared_lab_not_available")
        if lab.next_sequence >= self.spec.capacity:
            return fail("finite_lab_job_capacity")
        kind = "heat" if action["operation"] == "queue_lab_heat" else "measure"
        devices = lab_device_choices(state, kind)
        identity = action.get("device", devices[0] if len(devices) == 1 else None)
        if identity not in devices:
            return fail("select_one_installed_lab_device")
        job = {
            "sequence": lab.next_sequence,
            "kind": kind,
            "vessel": state.vessel_id,
            "queued_at": state.ledger.time_s,
            "device": identity,
        }
        if kind == "heat":
            from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

            target, duration = float(action["target_temperature_K"]), float(action["duration_s"])
            selected = self.thermal.advancer.strategy(state, required=True)
            temperature_bounds = getattr(selected[1], "temperature_bounds_k", None)
            bounds = None if temperature_bounds is None else temperature_bounds(state)
            if bounds is None:
                return fail("shared_heater_not_supported")
            equipment = device_record(state, identity)
            device = ThermalDeviceSpec.from_dict(equipment.settings["thermal_device"])
            if not (
                isfinite(duration)
                and 0 < duration <= 14400
                and isfinite(target)
                and bounds[0] <= target <= bounds[1]
                and device.minimum_setpoint_K <= target <= device.maximum_setpoint_K
            ):
                return fail("lab_heat_outside_declared_domain")
            job.update(duration_s=duration, target_temperature_K=target)
        else:
            instrument = device_record(state, identity).settings["lab_analyzer_instrument"]
            available = state.metadata["component_network"]["instruments"][state.vessel_id]
            if (
                instrument is None
                or instrument not in available
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
        identity = job["device"]
        state = reconnect_device(state, identity, job["vessel"])
        local = local_states(state)[job["vessel"]]
        before = initialize_solvents(local.replace(metadata={
            **local.metadata, INSTRUMENT_DEVICE_SELECTION: {
                "device_id": identity, "instrument": instrument,
            },
        }))
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
        noise = state.lab_clock.detector_noise
        keyed = noise.get("mode") == "keyed"
        if keyed:
            from chemworld.runtime.observation_noise import (
                ObservationNoiseCoordinate,
                keyed_noise_provenance,
                keyed_observation_rng,
            )

            # Queue identity is persistent and cannot be affected by polling.
            # The laboratory namespace separates deferred and direct readings.
            coordinate = ObservationNoiseCoordinate(
                namespace=noise["namespace"] + ":lab:" + identity,
                base_observation_seed=noise["seed"],
                experiment_index=noise["experiment_index"],
                operation_type="measure",
                instrument=instrument,
                replicate_index=job["sequence"],
            )
            rng = keyed_observation_rng(coordinate)
            job = {**job, "noise_provenance": keyed_noise_provenance(coordinate)}
        specimen = result.measurement_state.replace(equipment=after.equipment, ledger=after.ledger)
        try:
            reading = spec.observer_for(self.composition)(specimen, action, rng)
            spec.contract.validate_native_readout(reading)
            if not self.constitution.check_observation(reading).passed:
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
        acquisition = {
            **result.diagnostics["acquisition"], **timing,
            "sample_time_s": now, "result_time_s": now + duration, "reactor_advance_s": 0.0,
        }
        if reading is not None:
            native = reading.with_acquisition(acquisition).native_record()
            # Queue envelope fields remain for the existing lab publication API.
            native.update(timing)
            job["observation"] = native
        equipment = after.equipment.equipment.copy()
        device = equipment[identity]
        settings = deepcopy(device.settings)
        settings["elapsed_time_s"] -= duration
        settings["diagnostics"]["acquisition"] = deepcopy(acquisition)
        settings["execution_history"][-1]["diagnostics"]["acquisition"] = deepcopy(acquisition)
        equipment[identity] = replace(device, settings=settings, status="busy")
        after = after.replace(
            equipment=EquipmentLedger(equipment),
            ledger=after.ledger.with_updates(analysis_time_s=before.ledger.analysis_time_s),
            metadata={key: value for key, value in after.metadata.items()
                      if key != INSTRUMENT_DEVICE_SELECTION},
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
                detector_rng=(
                    state.lab_clock.detector_rng if keyed else deepcopy(rng.bit_generator.state)
                ),
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
        ready = ready_lab_jobs(state)
        sequences = {job["sequence"] for job in ready}
        queued = [job for job in state.lab_clock.queued if job["sequence"] not in sequences]
        # Start at most one request per device. Global sequence breaks ties;
        # waiting jobs do not reserve specimens from a changing reactor.
        for job in ready:
            if job["kind"] == "heat":
                state = reconnect_thermal_device(
                    state, job["device"], job["vessel"], park_idle=True,
                )
                local = local_states(state)[job["vessel"]]
                records = dict(local.equipment.equipment)
                records[job["device"]] = replace(records[job["device"]], status="busy")
                state = install_local(state, local.replace(equipment=EquipmentLedger(records)))
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
        groups = {
            system.system_id: locals_[system.system_id].replace(
                inactive_vessels={key: locals_[key] for key in system.vessels
                                  if key != system.system_id},
                vessel_elapsed_s=dict.fromkeys(system.vessels, now),
                lab_clock=None,
            ) for system in self.systems
        }

        def solve(duration):
            results = {}
            for key, local in groups.items():
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
            before, after = groups[key], result.state
            members = {key: after, **after.inactive_vessels}
            system = next(system for system in self.systems if system.system_id == key)
            if set(members) != set(system.vessels):
                return fail("lab_system_candidate_ownership_changed", outcome="numerical_failure")
            thermal_fields = ("energy_jacket_J", "heat_reaction_J", "heat_loss_J")
            thermal_deltas = result.vessel_thermal_deltas
            if thermal_deltas is not None and (
                set(thermal_deltas) != set(members)
                or any(not isclose(
                    fsum(values.get(field, 0.0) for values in thermal_deltas.values()),
                    getattr(after.ledger, field) - getattr(before.ledger, field),
                    abs_tol=1e-7, rel_tol=1e-8,
                ) for field in thermal_fields)
            ):
                return fail("lab_system_local_heat_mismatch", outcome="numerical_failure")
            previous_members = {key: before, **before.inactive_vessels}
            for name in ledger.keys() - {"time_s", "risk"}:
                ledger[name] += getattr(after.ledger, name) - getattr(before.ledger, name)
            for name in sorted(members):
                local = members[name]
                thermal = local.thermal.vessels[name]
                local_heat = (
                    {field: getattr(previous_members[name].thermal.vessels[name], field)
                     + thermal_deltas[name].get(field, 0.0) for field in thermal_fields}
                    if thermal_deltas is not None else
                    {field: getattr(local.ledger, field) for field in thermal_fields}
                )
                local = local.replace(
                    inactive_vessels={}, vessel_elapsed_s={}, lab_clock=None,
                    thermal=replace(local.thermal, vessels={name: replace(
                        thermal, **local_heat,
                    )}),
                )
                state = install_local(state, local)
        state = state.replace(
            ledger=replace(base.ledger, **ledger),
            vessel_elapsed_s=dict.fromkeys(locals_, now + dt),
        )
        # Analysis time is resource occupancy, not an extra physical clock.
        for job in running:
            if job["kind"] == "measure":
                local = local_states(state)[job["vessel"]]
                equipment = local.equipment.equipment.copy()
                identity = job["device"]
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
                    local = set_thermal_enabled(local, False)
                    records = dict(local.equipment.equipment)
                    records[job["device"]] = replace(records[job["device"]], status="used")
                    state = install_local(state, local.replace(equipment=EquipmentLedger(records)))
                else:
                    local = local_states(state)[job["vessel"]]
                    equipment = dict(local.equipment.equipment)
                    identity = job["device"]
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
                "vessels_advanced": len(locals_),
                "systems_advanced": len(results),
                "shared_systems": {
                    system.system_id: {
                        "strategy": system.strategy_id,
                        "vessels": list(system.vessels),
                        "connections": list(system.connections),
                    } for system in self.systems
                },
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
            ), unmodeled_quantities=formal_inventory_exclusions(before)
        )
