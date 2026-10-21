"""Transport and accounting of four finite carrier components through operations."""

from __future__ import annotations

from dataclasses import replace
from math import isclose, isfinite
from typing import Any

from chemworld.foundation import WorldState, selected_phase_id
from chemworld.foundation.solvents import (
    ORGANIC_DISTRIBUTION_RATIOS,
    RELATIVE_VOLATILITIES,
    SolventAccounting,
    SolventInventory,
    total_solvents,
)
from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.runtime.full_process_contract import sample_domain
from chemworld.world.mixtures import working_solvents


def initialize_solvents(state: WorldState) -> WorldState:
    medium = working_solvents(state)
    phases = {
        k: replace(p, solvents=medium.at_volume(p.volume_L))
        if p.phase_type in {"liquid", "aqueous", "organic"}
        and p.volume_L > 0
        and p.solvents.volume_L == 0
        and p.metadata.get("carrier_representation") != "species_inventory"
        else p
        for k, p in state.phases.phases.items()
    }
    state = state.replace(phases=PhaseLedger(phases))
    accounting = state.solvent_accounting
    if accounting.initial.volume_L == 0 and accounting.added.volume_L == 0:
        state = state.replace(solvent_accounting=replace(accounting, initial=total_solvents(state)))
    return state


def registered_carrier_state(state):
    return (state is not None and state.phases is not None and bool(state.phases.phases)
            and all(p.metadata.get("carrier_representation") == "species_inventory"
                    for p in state.phases.phases.values()))


def internal_carrier_input(before, after, action):
    if str(action["operation"]) != "add_solvent" or before.vessel_id != after.vessel_id:
        return None
    old = before.metadata.get("carrier_packet_history", [])
    new = after.metadata.get("carrier_packet_history", [])
    if new[:len(old)] != old:
        raise ValueError("previous carrier input history was rewritten")
    fresh = new[len(old):]
    if not fresh:
        return None
    if len(fresh) != 1:
        raise ValueError("carrier input requires one executed packet")
    receipt = fresh[0]
    if receipt.get("boundary_kind") != "internal":
        return None
    name, volume = receipt["source"], receipt["volume_L"]
    requested = float(action["volume_L"])
    index = int(action["solvent"])
    if (name not in before.inactive_vessels or name not in after.inactive_vessels
            or not isfinite(volume) or not 0 < volume <= requested + 1e-12
            or not isclose(receipt["requested_volume_L"], requested, abs_tol=1e-12)
            or not isclose(volume, requested, abs_tol=1e-12)
            or receipt["destination"] != before.vessel_id or receipt["solvent_index"] != index):
        raise ValueError("internal carrier receipt disagrees with the executed input")
    source, target = before.inactive_vessels[name], after.inactive_vessels[name]
    if (source.species_amounts != target.species_amounts or any(source.species_amounts.values())
            or source.state_slots != target.state_slots or source.equipment != target.equipment
            or source.temperature_K != target.temperature_K
            or not isclose(source.volume_L - target.volume_L, volume, abs_tol=1e-12)
            or not isclose(after.volume_L - before.volume_L, volume, abs_tol=1e-12)):
        raise ValueError("internal carrier input must debit its actual pure source")
    expected = SolventInventory.pure(index, volume)
    def pool(state):
        result = SolventInventory()
        for phase in state.phases.phases.values():
            result = result + phase.solvents
        return result
    for initial, final, sign in ((pool(source), pool(target), -1),
                                 (pool(before), pool(after), 1)):
        if any(not isclose(b - a, sign * c, abs_tol=1e-12)
               for a,b,c in zip(initial.volumes_L, final.volumes_L,
                                expected.volumes_L, strict=True)):
            raise ValueError("internal carrier component debit differs from its receipt")
    return receipt


def solvent_input(
    operation: str, action: dict[str, Any], *, state=None, after=None
) -> SolventInventory:
    if state is not None and after is not None and internal_carrier_input(state, after, action):
        return SolventInventory()
    if registered_carrier_state(state):
        # All carrier molecules are already in the registered species balance.
        return SolventInventory()
    if operation in {"add_solvent", "resuspend_crystals"}:
        return SolventInventory.pure(int(action["solvent"]), float(action["volume_L"]))
    if operation == "add_extractant":
        return SolventInventory.pure(int(action["extractant"]), float(action["volume_L"]))
    if operation == "add_phase":
        index = 0 if action["phase"] == "aqueous" else 3
        return SolventInventory.pure(index, float(action["volume_L"]))
    if operation == "wash":
        return SolventInventory.pure(0, float(action["wash_volume_L"]))
    return SolventInventory()


def solvent_outputs(
    before: WorldState, after: WorldState, action: dict[str, Any]
) -> dict[str, SolventInventory]:
    """Actual withdrawals from source phases or explicit provider outlets.

    Never derive an output as the residual required to balance the candidate.
    """
    operation = str(action["operation"])
    if registered_carrier_state(before):
        return {}
    if before.phases is None:
        raise ValueError("carrier transport requires phase inventories")
    phases = before.phases.phases
    if operation in {"sample", "measure"}:
        # The instrument/sample service checks this executed debit against its
        # own volume contract before transaction commit.
        volume = after.ledger.sample_consumed_L - before.ledger.sample_consumed_L
        domain = sample_domain(before)
        source = SolventInventory()
        for key in domain:
            source = source + phases[key].solvents
        return {"sample": source.split(volume)[0]} if volume else {}
    if operation == "separate_phase":
        target = str(action.get("target_phase", "organic"))
        return {key: phase.solvents for key, phase in phases.items() if key != target}
    if operation == "evaporate":
        volume = float(after.metadata["last_evaporation_energy"]["removed_carrier_L"])
        source = working_solvents(before)
        return {"vapour": source.split(volume, RELATIVE_VOLATILITIES)[0]}
    if operation in {"wash", "dry", "concentrate", "transfer"}:
        previous = before.metadata.get("removed_phase_inventory_history", [])
        history = after.metadata.get("removed_phase_inventory_history", [])
        fresh = history[len(previous) :]
        outputs: dict[str, SolventInventory] = {}
        for entry in fresh:
            for key, inventory in entry["inventories"].items():
                volume = float(inventory["volume_L"])
                if operation == "concentrate":
                    outputs[key] = SolventInventory(tuple(inventory["carrier_volumes_L"]))
                elif operation in {"wash", "dry"}:
                    outputs[key] = SolventInventory.pure(0, volume)
                else:
                    target = selected_phase_id(before.phases) or "organic"
                    outputs[key] = phases[target].solvents.at_volume(volume)
        return outputs
    return {}


def finish_solvent_transition(
    before: WorldState, after: WorldState, action: dict[str, Any]
) -> WorldState:
    operation = str(action["operation"])
    old, phases = before.phases.phases, after.phases.phases.copy()
    added = solvent_input(operation, action, state=before, after=after)
    pool = SolventInventory()
    for phase in old.values():
        pool = pool + phase.solvents
    # Existing phase volumes may change in a provider. A proportional carrier
    # withdrawal is its default; separation operations below specify a split law.
    inventories = {
        k: (
            p.solvents
            if (p.solvents.volume_L > 0 or p.phase_type not in {"liquid", "aqueous", "organic"}
                or p.metadata.get("carrier_representation") == "species_inventory")
            else old[k].solvents.at_volume(p.volume_L)
            if k in old
            else pool.at_volume(p.volume_L)
        )
        for k, p in phases.items()
    }
    if registered_carrier_state(before):
        inventories = {key: phase.solvents for key, phase in phases.items()}
    elif operation in {"add_phase", "add_extractant"}:
        target = str(action["phase"]) if operation == "add_phase" else "organic"
        inventories[target] = (
            old[target].solvents if target in old else SolventInventory()
        ) + added
    elif operation == "wash":
        target = selected_phase_id(before.phases) or "organic"
        retained_wash = max(phases[target].volume_L - old[target].volume_L, 0.0)
        inventories[target] = old[target].solvents + SolventInventory.pure(0, retained_wash)
    elif operation in {"mix", "settle"} and {"aqueous", "organic"} <= phases.keys():
        inventories["organic"], remaining = pool.split(
            phases["organic"].volume_L, ORGANIC_DISTRIBUTION_RATIOS
        )
        other_volume = sum(p.volume_L for k, p in phases.items() if k != "organic")
        for key, phase in phases.items():
            if key != "organic":
                inventories[key] = (
                    remaining.scale(phase.volume_L / other_volume)
                    if other_volume
                    else SolventInventory()
                )
    elif operation == "distill":
        # The duty-limited column solves explicit carrier-component outlets.
        # Preserve those outlets rather than applying a second fractionation law.
        inventories["distillate"] = phases["distillate"].solvents
        inventories["bottoms"] = phases["bottoms"].solvents
    elif operation == "collect_fraction":
        old_collected = (
            old["collected_fraction"].solvents
            if "collected_fraction" in old
            else SolventInventory()
        )
        volume = phases["collected_fraction"].volume_L - old_collected.volume_L
        withdrawn, retained = old["distillate"].solvents.split(volume)
        inventories["collected_fraction"] = old_collected + withdrawn
        inventories["distillate"] = retained
    elif operation == "evaporate":
        for key, phase in phases.items():
            if key in old:
                evaporated_volume = max(old[key].volume_L - phase.volume_L, 0)
                _, inventories[key] = old[key].solvents.split(
                    evaporated_volume, RELATIVE_VOLATILITIES
                )
    elif operation in {"seed_crystals", "cool_crystallize", "filter_crystals"}:
        for key, phase in phases.items():
            inventories[key] = pool.at_volume(phase.volume_L)
    phases = {k: replace(p, solvents=inventories[k]) for k, p in phases.items()}
    after = after.replace(phases=PhaseLedger(phases))
    start, end = total_solvents(before), total_solvents(after)
    removed = SolventInventory()
    for inventory in solvent_outputs(before, after, action).values():
        removed = removed + inventory
    transition_residual = max(
        abs(a + b - c - d)
        for a, b, c, d in zip(
            start.volumes_L, added.volumes_L, removed.volumes_L, end.volumes_L, strict=True
        )
    )
    if transition_residual > 1e-10:
        raise ValueError(f"Carrier transition does not match actual inputs/outputs: {operation}")
    accounting = SolventAccounting(
        before.solvent_accounting.initial,
        before.solvent_accounting.added + added,
        before.solvent_accounting.removed + removed,
    )
    residual = max(
        abs(a + b - c - d)
        for a, b, c, d in zip(
            accounting.initial.volumes_L,
            accounting.added.volumes_L,
            accounting.removed.volumes_L,
            end.volumes_L,
            strict=True,
        )
    )
    if residual > 1e-10:
        raise ValueError("Cumulative solvent component balance failed")
    return after.replace(
        solvent_accounting=accounting,
        metadata={
            **after.metadata,
            "last_solvent_transition": {
                "operation": operation,
                "added_L": added.to_dict(),
                "removed_L": removed.to_dict(),
                "component_balance_error_L": residual,
            },
        },
    )
