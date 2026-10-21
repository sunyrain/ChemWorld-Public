"""Stateless physical preparation shared by partial resets and campaign batches.

Foundation lifecycle owns replacement rules; domain services reclose retained
energy/inventory. The session alone commits the returned transition and history.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace


def initialization_boundaries(payload):
    """Validate an evaluator's after-action schedule, outside the action API."""
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError("initialization boundaries must map action numbers to options")
    result = {}
    for key, options in payload.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            raise ValueError("initialization boundary needs a nonnegative action number")
        try:
            index = int(key)
        except ValueError as exc:
            raise ValueError("initialization boundary needs a nonnegative action number") from exc
        if index < 0 or str(index) != str(key) or index in result:
            raise ValueError("initialization boundary needs a unique nonnegative action number")
        if not isinstance(options, Mapping) or options.get("scope") not in {
            "batch", "material", "vessel", "equipment",
        }:
            raise ValueError("initialization boundary needs explicit entity scope")
        if set(options) - {"scope", "vessel_ids", "equipment_id", "replacement", "slots"}:
            raise ValueError("initialization boundary cannot change world or noise identity")
        result[index] = deepcopy(dict(options))
    return result


def initialization_replay(history):
    """Derive private replay instructions from the sole committed receipt history."""
    import hashlib
    import json

    commands = []
    for receipt in history:
        scope = receipt["scope"]
        options = {"scope": scope}
        if scope == "equipment":
            options.update(
                equipment_id=receipt["removed_device"]["equipment_id"],
                replacement=deepcopy(receipt["introduced_device"]),
                slots=deepcopy(receipt["introduced_slots"]),
            )
        else:
            options["vessel_ids"] = sorted(receipt["vessels"])
        commands.append({
            "after_operation_id": receipt["after_operation_id"],
            "experiment_index": receipt["experiment_index"],
            "options": options,
            "receipt_sha256": hashlib.sha256(json.dumps(
                receipt, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest(),
        })
    return commands


def replay_initializations(session, commands):
    """Apply missing evaluator boundaries, checking automatic batches only once.

    Commands at a later operation are deferred until that operation commits.
    This function never grants an Agent an initialization operation.
    """
    if not isinstance(commands, list):
        raise ValueError("initialization_replay must be an array")
    for index, command in enumerate(commands):
        if not isinstance(command, dict):
            raise ValueError("initialization replay command must be an object")
        if command["after_operation_id"] > session._operation_id:
            break
        if index == len(session._lifecycle_history):
            if command["after_operation_id"] != session._operation_id:
                raise ValueError("initialization replay missed its operation boundary")
            session.reset(options=deepcopy(command["options"]))
        actual = initialization_replay(session._lifecycle_history)[index]
        if command != actual:
            raise ValueError(f"initialization replay receipt mismatch at boundary {index}")
    if len(session._lifecycle_history) > len(commands):
        raise ValueError("initialization replay omitted a committed boundary")


def shared_lifecycle_vessels(current, *, scope, options, systems, protected_vessels=()):
    """Reject only related queued/running systems; never cancel their work."""
    if scope not in {"material", "vessel", "equipment"}:
        raise ValueError("shared initialization requires material, vessel or equipment scope")
    if scope == "equipment":
        identity = options["equipment_id"]
        selected = {local.vessel_id for local in (
            current, *current.inactive_vessels.values(),
        ) if identity in local.equipment.equipment}
        if len(selected) != 1:
            raise ValueError("replacement must identify exactly one installed device")
    else:
        selected = set(options.get("vessel_ids", (current.vessel_id,)))
        if not selected or not selected <= {current.vessel_id, *current.inactive_vessels}:
            raise ValueError("shared initialization requires declared vessel locations")
        if selected & set(protected_vessels):
            raise ValueError("partial initialization cannot replenish a finite supply vessel")
    related = set(selected)
    for system in systems:
        if selected & set(system.vessels):
            related.update(system.vessels)
    devices = {identity for local in (current, *current.inactive_vessels.values())
               if local.vessel_id in related for identity in local.equipment.equipment}
    jobs = (*current.lab_clock.running, *current.lab_clock.queued)
    blocked = [job for job in jobs if job["vessel"] in related or job["device"] in devices or (
        scope == "equipment" and job["device"] == options["equipment_id"]
    )]
    if blocked:
        references = ", ".join(f"{j['sequence']}:{j['device']}:{j['vessel']}" for j in blocked)
        raise ValueError("shared lifecycle blocked by queued/running jobs " + references)
    return selected


def reconcile_shared_initialization(previous, transition):
    """Retain one clock and account explicit replacement carrier inlet/outlet."""
    from chemworld.foundation.solvents import SolventInventory, total_solvents
    from chemworld.runtime.component_network import _local

    state = transition.state
    selected = set(transition.receipt.get("vessels", {}))
    locals_ = {state.vessel_id: state.replace(inactive_vessels={}), **state.inactive_vessels}
    old = {previous.vessel_id: previous, **previous.inactive_vessels}
    introduced, removed = SolventInventory(), SolventInventory()

    def vessel_carriers(local):
        # Sealed historical samples remain in the global ledger. A local
        # material boundary replaces only this vessel's actual phase stocks.
        return total_solvents(local.replace(
            inactive_vessels={}, samples=replace(local.samples, samples={}),
        ))

    for identity in sorted(selected):
        local = locals_[identity]
        introduced += vessel_carriers(local)
        removed += vessel_carriers(old[identity])
        continuation = old[identity].process.stochastic_continuation
        if continuation is not None:
            continuation = deepcopy(continuation)
            continuation.update(time_s=previous.ledger.time_s, pending=None, signature=None)
            local = local.replace(process=replace(
                local.process, stochastic_continuation=continuation,
            ))
        local = local.replace(ledger=replace(local.ledger, time_s=previous.ledger.time_s))
        locals_[identity] = local
    active = locals_.pop(state.vessel_id)
    # The global ledger owns resource history; inactive frames own their local
    # thermal history and physical clock, as in the existing shared scheduler.
    for identity in selected & locals_.keys():
        locals_[identity] = _local(locals_[identity].replace(
            vessel_elapsed_s={identity: previous.ledger.time_s},
        ))
    state = active.replace(
        inactive_vessels=locals_, lab_clock=previous.lab_clock,
        ledger=previous.ledger, vessel_elapsed_s=previous.vessel_elapsed_s,
        solvent_accounting=replace(
            previous.solvent_accounting,
            added=previous.solvent_accounting.added + introduced,
            removed=previous.solvent_accounting.removed + removed,
        ),
    )
    return replace(transition, state=state, receipt={
        **transition.receipt,
        "shared_clock_time_s": previous.ledger.time_s,
        "replacement_carrier_inlet_L": introduced.to_dict(),
        "replacement_carrier_outlet_L": removed.to_dict(),
    })


def prepare_initialization(current, fresh_initial, *, scope, options, physics, advancer,
                           constitution, shared_systems=()):
    from chemworld.foundation.lifecycle import initialize_entities, replace_equipment
    from chemworld.foundation.state_ledgers import EquipmentRecord
    from chemworld.foundation.state_ownership import StateSlots

    shared = current.lab_clock is not None
    if shared:
        selected = shared_lifecycle_vessels(
            current, scope=scope, options=options, systems=shared_systems,
            protected_vessels=(set() if physics is None else physics.finite_supply_vessels)
            | advancer.protected_lifecycle_vessels,
        )
        if scope != "equipment":
            options = {**options, "vessel_ids": selected}

    if scope == "equipment":
        replacement = options["replacement"]
        if isinstance(replacement, Mapping):
            replacement = EquipmentRecord(**replacement)
        slots = options.get("slots", StateSlots())
        if isinstance(slots, Mapping):
            slots = StateSlots.from_dict(dict(slots))
        transition = replace_equipment(
            current,
            options["equipment_id"],
            replacement,
            slots=slots,
        )
    else:
        transition = initialize_entities(
            current,
            fresh_initial,
            scope=scope,
            vessel_ids=options.get("vessel_ids"),
            preserve_vessels=(set() if physics is None else physics.finite_supply_vessels)
            | advancer.preserved_source_vessels,
        )
        if physics is not None:
            from chemworld.runtime.condensed_material_routing import (
                reconcile_thermal_initialization,
            )

            transition = reconcile_thermal_initialization(
                current, fresh_initial, transition, physics
            )
            from chemworld.runtime.crystal_thermal import reconcile_crystal_initialization

            transition = reconcile_crystal_initialization(
                current, fresh_initial, transition, physics
            )
    if (
        scope != "equipment"
        and physics is not None
        and (physics.models_of("column") or physics.models_of("concentration"))
    ):
        # Material replacement retains stationary body H. Re-equilibrate
        # new fluid plus retained apparatus by the same PH closure, without
        # advancing the experiment clock or introducing a duplicate n.
        from chemworld.runtime.concentration import install_concentration, read_concentration
        from chemworld.runtime.dynamic_column import install_column, read_column

        active = transition.state
        locals_ = {
            active.vessel_id: active.replace(inactive_vessels={}),
            **active.inactive_vessels,
        }
        initialized = {}
        models = {**physics.models_of("column"), **physics.models_of("concentration")}
        for name in set(transition.receipt["vessels"]) & set(models):
            local, model = locals_[name], models[name]
            before_h = sum(s.values["enthalpy_J"] for s in local.state_slots.slots)
            if name in physics.models_of("column"):
                local = install_column(local, model, read_column(local, model))
            else:
                local = install_concentration(local, model, read_concentration(local, model))
            locals_[name] = local
            initialized[name] = {
                "total_enthalpy_before_J": before_h,
                "total_enthalpy_after_J": sum(
                    s.values["enthalpy_J"] for s in local.state_slots.slots
                ),
                "temperature_K": local.temperature_K,
            }
        active = locals_.pop(active.vessel_id)
        transition = replace(
            transition,
            state=active.replace(inactive_vessels=locals_),
            receipt={
                **transition.receipt,
                **(
                    {
                        "column_equilibration": {
                            k: v for k, v in initialized.items()
                            if k in physics.models_of("column")
                        }
                    }
                    if physics.models_of("column")
                    else {}
                ),
                **(
                    {
                        "concentration_equilibration": {
                            k: v
                            for k, v in initialized.items()
                            if k in physics.models_of("concentration")
                        }
                    }
                    if physics.models_of("concentration")
                    else {}
                ),
            },
        )
    if shared:
        transition = reconcile_shared_initialization(current, transition)
    transition = advancer.reconcile_initialization(transition)
    advancer.validate_snapshot_state(transition.state)
    report = constitution.check_state(transition.state)
    if not report.passed:
        raise ValueError("entity initialization produced an invalid physical state")
    return transition
