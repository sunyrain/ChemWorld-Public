"""Finite-column strategy of the common physical advance coordinator.

PhaseLedger owns all bulk n. Typed slots own compartment H; the numerical
column DTO is a temporary view, never a second persisted material inventory.
"""

from dataclasses import replace
from math import ceil

import numpy as np

from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger, PhaseRecord
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.dynamic_distillation import (
    ColumnOperatingLimitError,
    DynamicColumnState,
    advance_dynamic_column,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.coupled_thermal import connected_device
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.phase_inventory import (
    CompartmentLayout,
    install_compartments,
    read_compartments,
    validate_compartments,
)

MODEL_ID = "finite_equilibrium_column_v1"
STATE_TYPE = "finite-column-enthalpy-v1"


def _slot_id(index):
    return f"column:cell:{index}"


def column_layout(state, spec):
    return CompartmentLayout(
        MODEL_ID,
        STATE_TYPE,
        "column",
        "column_cell",
        tuple(
            ("distillate", "distillate_gas")
            if i == spec.collector_index
            else (f"column_{i}_liquid", f"column_{i}_gas")
            for i in range(spec.cell_count)
        ),
        "distillate",
        ("collected_fraction",),
        (
            StateSlotSpec(
                "column:collected",
                "material",
                f"{state.vessel_id}:collected",
                STATE_TYPE,
                (StateFieldSpec("enthalpy_J", "J", minimum=None),),
                phase_id="collected_fraction",
                batch_reset="clear",
            ),
        ),
    )


def read_column(state, spec):
    amounts, enthalpies = read_compartments(state, spec, column_layout(state, spec))
    collected = state.phases.phases.get("collected_fraction")
    return DynamicColumnState(
        amounts,
        enthalpies,
        state.ledger.time_s,
        state.ledger.energy_jacket_J,
        state.ledger.heat_loss_J,
        0 if collected is None else sum(collected.species_amounts_mol.values()),
    )


def install_column(state, spec, column):
    return install_compartments(state, spec, column, column_layout(state, spec))


def validate_column_state(state, spec):
    column = read_column(state, spec)
    total = sum(sum(n.values()) for n in column.amounts_mol) + column.withdrawn_product_mol
    if total > spec.equipment.maximum_batch_amount_mol + 1e-12:
        raise ValueError("initial column charge exceeds its equipment capacity")
    collection_limit = min(
        spec.collector_capacity_mol,
        spec.equipment.maximum_distillate_cut_fraction * total - column.withdrawn_product_mol,
    )
    if sum(column.amounts_mol[-1].values()) > collection_limit + 1e-12:
        raise ValueError("column product inventory exceeds its finite collection limit")
    for i, (n, h) in enumerate(zip(column.amounts_mol[:-1], column.enthalpies_J[:-1], strict=True)):
        phase = spec.at_ph(i, n, h)
        floor = spec.equipment.minimum_bottoms_amount_mol if i == 0 else 1e-6
        if sum(phase.liquid_amounts_mol.values()) < floor - 1e-12:
            raise ValueError("column requires positive operating liquid holdup")
        if i and sum(n.values()) > spec.tray_maximum_amount_mol + 1e-12:
            raise ValueError("column starts outside its finite tray capacity")
    validate_compartments(state, spec, column_layout(state, spec), column)


class DynamicColumnRuntime:
    def __init__(self, composition):
        self.composition = composition

    def supports(self, state):
        return (
            self.composition is not None
            and self.composition.physics is not None
            and state.vessel_id in self.composition.physics.column_models
        )

    def model(self, state):
        return self.composition.physics.column_models[state.vessel_id]

    def advance(self, state, action, *, heat, jacket_program=None):
        spec = self.model(state)
        try:
            record, heater = connected_device(state, required=True)
            hardware = next(
                r
                for r in state.equipment.equipment.values()
                if r.equipment_type == "dynamic_distillation_column"
                and r.attached_vessel_id == state.vessel_id
            )
            if (
                jacket_program is not None
                or heater.mode != "finite_power"
                or heater.maximum_power_W > spec.equipment.maximum_reboiler_power_W
            ):
                raise ValueError("column requires its finite-power native control mode")
            if heat:
                heater = replace(
                    heater, setpoint_K=float(action["target_temperature_K"]), enabled=True
                )
            full = action.get("full_reflux", hardware.settings.get("full_reflux", 0))
            if isinstance(full, bool) or full not in (0, 1):
                raise ValueError("full_reflux must be numeric 0 or 1")
            reflux = (
                None
                if full
                else float(action.get("reflux_ratio", hardware.settings["reflux_ratio"]))
            )
            if reflux is not None and not 0 <= reflux <= spec.equipment.maximum_reflux_ratio:
                raise ValueError("reflux leaves the equipment operating range")
        except (KeyError, StopIteration, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="invalid_column_control",
                diagnostics={"control_error": str(exc)},
            )
        try:
            validate_column_state(state, spec)
            initial = read_column(state, spec)
            duration = float(action["duration_s"])
            result = advance_dynamic_column(
                spec,
                initial,
                heater=heater,
                duration_s=duration,
                reflux_ratio=reflux,
                evaluation_times_s=np.linspace(0, duration, ceil(duration / 2) + 1),
                maximum_volume_L=state.vessels.vessels[state.vessel_id].max_volume_L,
                occupied_volume_L=sum(
                    p.volume_L
                    for p in state.phases.phases.values()
                    if p.phase_id == "collected_fraction"
                ),
            )
            equipment = dict(state.equipment.equipment)
            equipment[record.equipment_id] = replace(
                record, settings={**record.settings, "thermal_device": heater.to_dict()}
            )
            equipment[hardware.equipment_id] = replace(
                hardware,
                status="distilled",
                settings={
                    **hardware.settings,
                    "reflux_ratio": 0 if reflux is None else reflux,
                    "full_reflux": int(reflux is None),
                    "distillation_model": MODEL_ID,
                },
            )
            candidate = install_column(state, spec, result.final_state).replace(
                equipment=EquipmentLedger(equipment),
                ledger=advance_ledger(
                    state.ledger,
                    result.actual_duration_s,
                    energy_deltas={
                        "energy_jacket_J": result.final_state.heater_energy_J
                        - initial.heater_energy_J,
                        "heat_loss_J": result.final_state.condenser_energy_J
                        - initial.condenser_energy_J,
                    },
                ),
            )
            history = list(state.metadata.get("column_timed_fractions", ()))
            history.extend(result.fractions)
            candidate = candidate.replace(
                metadata={**candidate.metadata, "column_timed_fractions": history}
            )
        except ColumnOperatingLimitError as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="column_operating_limit",
                diagnostics={"control_error": str(exc)},
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="dynamic_column_solver_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=result.actual_duration_s,
            outcome="stopped" if result.stopping_event else "success",
            diagnostics={
                "model_id": MODEL_ID,
                "stopping_event": result.stopping_event,
                "solver": result.solver_diagnostic,
                "timed_fractions": list(result.fractions),
                "component_residuals_mol": result.component_residuals_mol,
                "enthalpy_residual_J": result.enthalpy_residual_J,
            },
        )

    def collect_fraction(self, state, action):
        spec = self.model(state)
        fraction = float(action["transfer_fraction"])
        if not 0 < fraction <= 1:
            raise ValueError("column fraction collection must lie in (0,1]")
        column = read_column(state, spec)
        phase = spec.at_ph(spec.collector_index, column.amounts_mol[-1], column.enthalpies_J[-1])
        moved = {k: fraction * v for k, v in phase.liquid_amounts_mol.items()}
        packet_h = fraction * phase.liquid_enthalpy_J
        amounts, energies = list(column.amounts_mol), list(column.enthalpies_J)
        amounts[-1] = {k: v - moved[k] for k, v in amounts[-1].items()}
        energies[-1] -= packet_h
        previous = state.phases.phases.get("collected_fraction")
        phases = dict(state.phases.phases)
        source = phases["distillate"]
        phases["distillate"] = replace(
            source,
            species_amounts_mol={k: v - moved[k] for k, v in source.species_amounts_mol.items()},
            volume_L=source.volume_L * (1 - fraction),
        )
        collected = {
            k: moved[k] + (0 if previous is None else previous.species_amounts_mol[k])
            for k in moved
        }
        phases["collected_fraction"] = PhaseRecord(
            "collected_fraction",
            state.vessel_id,
            "liquid",
            sum(collected.values()) * spec.liquid_molar_holdup_L_mol,
            collected,
            settled=True,
            metadata={"source_phase": "distillate", "carrier_representation": "species_inventory"},
        )
        slots = {s.spec.slot_id: s for s in state.state_slots.slots}
        if "column:collected" in slots:
            old = slots["column:collected"]
            slots["column:collected"] = replace(
                old, values={"enthalpy_J": old.values["enthalpy_J"] + packet_h}
            )
        else:
            slots["column:collected"] = StateSlot(
                StateSlotSpec(
                    "column:collected",
                    "material",
                    f"{state.vessel_id}:collected",
                    STATE_TYPE,
                    (StateFieldSpec("enthalpy_J", "J", minimum=None),),
                    phase_id="collected_fraction",
                    batch_reset="clear",
                ),
                state.vessel_id,
                {"enthalpy_J": packet_h},
            )
        source_slot = slots[_slot_id(spec.collector_index)]
        slots[_slot_id(spec.collector_index)] = replace(
            source_slot, values={"enthalpy_J": energies[-1]}
        )
        candidate = state.replace(
            phases=PhaseLedger(phases), state_slots=StateSlots(tuple(slots.values()))
        )
        return install_column(
            candidate,
            spec,
            replace(
                column,
                amounts_mol=tuple(amounts),
                enthalpies_J=tuple(energies),
                withdrawn_product_mol=column.withdrawn_product_mol + sum(moved.values()),
            ),
        )
