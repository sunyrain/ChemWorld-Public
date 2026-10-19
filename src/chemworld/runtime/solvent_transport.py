"""Transport and accounting of four finite carrier components through operations."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from chemworld.foundation import WorldState, selected_phase_id
from chemworld.foundation.solvents import (
    ORGANIC_DISTRIBUTION_RATIOS,
    RELATIVE_VOLATILITIES,
    SolventAccounting,
    SolventInventory,
)
from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.world.mixtures import working_solvents


def total_solvents(state: WorldState) -> SolventInventory:
    total = SolventInventory()
    if state.phases is not None:
        for phase in state.phases.phases.values():
            total = total + phase.solvents
    for sample in state.samples.samples.values():
        if sample.contents is None:
            total = total + SolventInventory.pure(sample.solvent, sample.volume_L)
        else:
            for phase in sample.contents.phases.phases.values():
                total = total + phase.solvents
    for local in state.inactive_vessels.values():
        total = total + total_solvents(local)
    return total


def initialize_solvents(state: WorldState) -> WorldState:
    assert state.phases is not None
    medium = working_solvents(state)
    phases = {
        k: replace(p, solvents=medium.at_volume(p.volume_L))
        if p.volume_L > 0 and p.solvents.volume_L == 0
        else p
        for k, p in state.phases.phases.items()
    }
    state = state.replace(phases=PhaseLedger(phases))
    accounting = state.solvent_accounting
    if accounting.initial.volume_L == 0 and accounting.added.volume_L == 0:
        state = state.replace(solvent_accounting=replace(accounting, initial=total_solvents(state)))
    return state


def _input(state: WorldState, operation: str, action: dict[str, Any]) -> SolventInventory:
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


def finish_solvent_transition(
    before: WorldState, after: WorldState, action: dict[str, Any]
) -> WorldState:
    assert before.phases is not None and after.phases is not None
    operation = str(action["operation"])
    old, phases = before.phases.phases, after.phases.phases.copy()
    added = _input(before, operation, action)
    pool = SolventInventory()
    for phase in old.values():
        pool = pool + phase.solvents
    # Existing phase volumes may change in a provider. A proportional carrier
    # withdrawal is its default; separation operations below specify a split law.
    inventories = {
        k: (
            p.solvents
            if p.solvents.volume_L > 0
            else old[k].solvents.at_volume(p.volume_L)
            if k in old
            else pool.at_volume(p.volume_L)
        )
        for k, p in phases.items()
    }
    if operation in {"add_phase", "add_extractant"}:
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
    removed_values = tuple(
        a + b - c for a, b, c in zip(start.volumes_L, added.volumes_L, end.volumes_L, strict=True)
    )
    if min(removed_values) < -1e-10:
        raise ValueError(f"Solvent component inventory increased without input: {operation}")
    removed = SolventInventory(tuple(max(v, 0) for v in removed_values))
    if removed.volume_L > 1e-10 and operation not in {
        "sample",
        "measure",
        "evaporate",
        "separate_phase",
        "wash",
        "dry",
        "concentrate",
        "transfer",
    }:
        raise ValueError(f"Closed carrier transition lost solvent: {operation}")
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
