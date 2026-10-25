"""A bounded delegate on the original thermal advance/transaction path."""

from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.solvents import SolventInventory
from chemworld.foundation.state_helpers import attached_device_record
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.catalysis_network import (
    CATALYTIC_STATE_TYPE,
    CatalyticMechanismSpec,
    catalytic_inventory_key,
    validate_catalytic_slot,
)
from chemworld.physchem.catalysis_reservoirs import CatalyticReservoirBoundary
from chemworld.physchem.catalysis_thermochemistry import CatalyticThermodynamics
from chemworld.physchem.chemostatted_catalysis import (
    ELECTRICAL_FIELDS,
    ChemostattedElectricalReadout,
    chemostatted_catalytic_system,
    chemostatted_readout,
    pack_chemostatted_state,
    reservoir_ports,
    validate_boundary,
)
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.control_program import electrical_device
from chemworld.runtime.kernel_contracts import ModelProviderResult

CHEMOSTATTED_CATALYSIS_TYPE = "chemostatted_catalysis"
ELECTRICAL_STATE_TYPE = "chemostatted_catalysis_electrical"
EXCHANGE_STATE_TYPE = "chemostatted_catalysis_exchange"


def chemostatted_record(state: Any) -> Any:
    return attached_device_record(state, CHEMOSTATTED_CATALYSIS_TYPE)


def accounting_slots(
    state: Any, record: Any, boundary: CatalyticReservoirBoundary, physical: CoupledState
) -> tuple[StateSlot, StateSlot]:
    electrical = StateSlot(
        StateSlotSpec(
            f"{record.equipment_id}:electrical",
            "equipment",
            record.equipment_id,
            ELECTRICAL_STATE_TYPE,
            tuple(
                StateFieldSpec(k, "C" if k.endswith("_C") else "J", minimum=None)
                for k in ELECTRICAL_FIELDS
            ),
            mechanism_id="constant-SHE-working-electrode-v1",
            batch_reset="preserve",
        ),
        state.vessel_id,
        dict(physical.coordinates),
    )
    fields, values = [], {}
    for port in reservoir_ports(boundary):
        for suffix, unit, value in (
            ("mol", "mol", physical.exchanged_mol.get(port.port_id, 0.0)),
            ("H_J", "J", physical.exchange_energy_J.get(port.port_id, 0.0)),
        ):
            key = f"{port.port_id}:{suffix}"
            fields.append(StateFieldSpec(key, unit, minimum=0 if unit == "mol" else None))
            values[key] = value
    exchange = StateSlot(
        StateSlotSpec(
            f"{record.equipment_id}:reservoir-exchange",
            "equipment",
            record.equipment_id,
            EXCHANGE_STATE_TYPE,
            tuple(fields),
            mechanism_id="real-aqueous-reservoir-exchange-v1",
            batch_reset="preserve",
        ),
        state.vessel_id,
        values,
    )
    return electrical, exchange


def install_chemostatted_material(
    state: Any,
    spec: CatalyticMechanismSpec,
    boundary: CatalyticReservoirBoundary,
    slot: StateSlot,
    physical: CoupledState,
    record: Any,
) -> Any:
    gas = {k: physical.amounts_mol[k] for k in spec.gas_species}
    liquid = {k: physical.amounts_mol[k] for k in spec.reservoir_species}
    catalyst = replace(
        slot,
        values={
            **slot.values,
            **{k: physical.amounts_mol[catalytic_inventory_key(slot, k)] for k in spec.adsorbates},
        },
    )
    return state.replace(
        species_amounts={**gas, **liquid},
        phases=PhaseLedger(
            {
                "gas": PhaseRecord(
                    "gas",
                    state.vessel_id,
                    "gas",
                    boundary.gas_volume_L,
                    gas,
                    settled=True,
                    selected=True,
                    metadata={"carrier_representation": "species_inventory"},
                ),
                "electrolyte": PhaseRecord(
                    "electrolyte",
                    state.vessel_id,
                    "liquid",
                    boundary.liquid_volume_L,
                    liquid,
                    settled=True,
                    metadata={"boundary": "constant-activity-real-solutes"},
                    solvents=SolventInventory.pure(
                        boundary.carrier.solvent_index, boundary.liquid_volume_L
                    ),
                ),
            }
        ),
        state_slots=StateSlots((catalyst, *accounting_slots(state, record, boundary, physical))),
        pressure_Pa=sum(gas.values()) * R * physical.temperature_K / (boundary.gas_volume_L / 1000),
        volume_L=boundary.gas_volume_L + boundary.liquid_volume_L,
        temperature_K=physical.temperature_K,
        process=replace(state.process, last_observation_record=None),
    )


def unpack_chemostatted_catalysis(
    state: Any,
) -> tuple[
    Any, CatalyticMechanismSpec, CatalyticReservoirBoundary, StateSlot, CoupledState, float, float
]:
    record = chemostatted_record(state)
    if record is None:
        raise ValueError("no chemostatted catalytic electrode is installed")
    raw = record.settings["physical_model"]
    spec = CatalyticMechanismSpec.from_dict(raw["mechanism"])
    boundary = CatalyticReservoirBoundary.from_dict(raw["boundary"])
    if electrical_device(state).equipment_id != record.equipment_id:
        raise ValueError("chemostatted catalyst must use its own electrical source")
    control = record.settings["electrical_control"]
    program = ScalarControlProgram(**control["program"])
    if (
        program.unit != "V"
        or len(program.setpoints) != 1
        or not control.get("constant_potential_only")
        or list(control.get("supported_units", [])) != ["V"]
        or control.get("limits_V") != [boundary.minimum_potential_V, boundary.maximum_potential_V]
    ):
        raise ValueError("chemostatted catalysis requires one constant potential vs SHE")
    potential = program.value_at(state.ledger.time_s)
    validate_boundary(spec, boundary, potential)
    slots = {s.spec.state_type: s for s in state.state_slots.slots}
    if len(state.state_slots.slots) != 3 or set(slots) != {
        CATALYTIC_STATE_TYPE,
        ELECTRICAL_STATE_TYPE,
        EXCHANGE_STATE_TYPE,
    }:
        raise ValueError(
            "one original surface and uniquely owned electrical/exchange slots required"
        )
    slot = slots[CATALYTIC_STATE_TYPE]
    validate_catalytic_slot(slot, spec)
    if set(state.phases.phases) != {"gas", "electrolyte"}:
        raise ValueError("chemostatted catalyst requires real gas and electrolyte phases")
    electrical, exchanges = slots[ELECTRICAL_STATE_TYPE], slots[EXCHANGE_STATE_TYPE]
    physical = pack_chemostatted_state(
        spec,
        slot,
        boundary,
        state.phases.phases["gas"].species_amounts_mol,
        liquid_amounts_mol=state.phases.phases["electrolyte"].species_amounts_mol,
        time_s=state.ledger.time_s,
        coordinates=electrical.values,
        exchanged_mol={
            p.port_id: exchanges.values[f"{p.port_id}:mol"] for p in reservoir_ports(boundary)
        },
        exchange_energy_J={
            p.port_id: exchanges.values[f"{p.port_id}:H_J"] for p in reservoir_ports(boundary)
        },
    )
    expected = install_chemostatted_material(state, spec, boundary, slot, physical, record)
    if (
        state.phases.to_dict() != expected.phases.to_dict()
        or dict(state.species_amounts) != dict(expected.species_amounts)
        or not isclose(state.pressure_Pa, expected.pressure_Pa, rel_tol=1e-9, abs_tol=1e-7)
        or not isclose(state.temperature_K, physical.temperature_K, rel_tol=0, abs_tol=1e-9)
        or not isclose(state.volume_L, physical.volume_L, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("original chemostat phase/bulk n/T/P/V disagree")
    for owned, wanted in zip(state.state_slots.slots, expected.state_slots.slots, strict=True):
        if (
            owned.spec != wanted.spec
            or owned.location_id != wanted.location_id
            or set(owned.values) != set(wanted.values)
        ):
            raise ValueError("chemostat original slot ownership/layout disagree")
    return record, spec, boundary, slot, physical, potential, raw["support_heat_capacity_J_K"]


def read_chemostatted_electrical(state: Any) -> ChemostattedElectricalReadout:
    _, spec, boundary, slot, physical, potential, _ = unpack_chemostatted_catalysis(state)
    return chemostatted_readout(spec, slot, boundary, potential, physical)


class ChemostattedCatalysisRuntime:
    supported_operations = ("configure_electrical_control", "wait")

    def __init__(self, composition: Any, modules: Any = None) -> None:
        self.composition, self.modules = composition, modules

    def supports(self, state: Any) -> bool:
        return chemostatted_record(state) is not None

    def reconcile_pressure(self, state: Any) -> Any:
        record = chemostatted_record(state)
        boundary = CatalyticReservoirBoundary.from_dict(
            record.settings["physical_model"]["boundary"]
        )
        gas = state.phases.phases["gas"].species_amounts_mol
        return state.replace(
            pressure_Pa=sum(gas.values()) * R * state.temperature_K / (boundary.gas_volume_L / 1000)
        )

    def validate_state(self, state: Any) -> None:
        if any("thermal_feedback" in r.settings for r in state.equipment.equipment.values()):
            raise ValueError("declared isothermal half-cell does not support thermal feedback")
        record, spec, boundary, slot, initial, potential, support = unpack_chemostatted_catalysis(
            state
        )
        if self.composition is None or self.composition.physics is None:
            raise ValueError("chemostatted catalyst needs original composition")
        physics = self.composition.physics
        original = physics.initial_states[state.vessel_id]
        source = chemostatted_record(original)
        if source is None or source.settings["physical_model"] != record.settings["physical_model"]:
            raise ValueError("catalytic reservoir model differs from its original source")
        source_slot = next(
            s for s in original.state_slots.slots if s.spec.state_type == CATALYTIC_STATE_TYPE
        )
        if slot.spec != source_slot.spec or any(
            slot.values[k] != v for k, v in source_slot.values.items() if k not in spec.adsorbates
        ):
            raise ValueError(
                "surface capacity/support cannot change during the half-cell experiment"
            )
        declared = physics.models_of("surface")[spec.mechanism_id]
        thermal = physics.models_of("thermal")[state.vessel_id]
        expected_thermo = CatalyticThermodynamics(
            spec.species_by_id,
            support + boundary.carrier.volumetric_cv_J_L_K * boundary.liquid_volume_L,
            boundary.carrier.feed_temperature_K,
        )
        if (
            declared.to_dict() != spec.to_dict()
            or thermal.thermodynamics.to_dict() != expected_thermo.to_dict()
        ):
            raise ValueError(
                "surface kinetics and original half-cell calorics must have identical source data"
            )
        if physics.network.reactions or {
            s.species_id: s.to_dict() for s in physics.network.species
        } != {s.species.species_id: s.species.to_dict() for s in spec.species}:
            raise ValueError(
                "half-cell requires its real species identities without a competing bulk driver"
            )
        total_volume = boundary.liquid_volume_L + boundary.gas_volume_L
        if (
            total_volume > state.vessels.vessels[state.vessel_id].max_volume_L
            or boundary.maximum_pressure_Pa > state.vessels.vessels[state.vessel_id].max_pressure_Pa
        ):
            raise ValueError("half-cell geometry/pressure exceeds its original vessel")
        bath_records = [
            r for r in state.equipment.equipment.values() if r.equipment_type == "thermal_device"
        ]
        source_baths = [
            r for r in original.equipment.equipment.values() if r.equipment_type == "thermal_device"
        ]
        if (
            len(bath_records) != 1
            or len(source_baths) != 1
            or bath_records[0].equipment_id != source_baths[0].equipment_id
            or bath_records[0].attached_vessel_id != state.vessel_id
            or bath_records[0].settings.get("lab_role") != "fixed_isothermal_bath"
            or bath_records[0].settings["thermal_device"] != boundary.bath.to_dict()
        ):
            raise ValueError("half-cell must retain its declared isothermal bath")
        chemostatted_catalytic_system(
            spec, slot, boundary, potential, support_heat_capacity_J_K=support
        ).validate_state(initial)

    def advance(
        self, state: Any, action: dict[str, Any], *, heat: bool = False, jacket_program: Any = None
    ) -> ModelProviderResult:
        try:
            self.validate_state(state)
            record, spec, boundary, slot, initial, potential, support = (
                unpack_chemostatted_catalysis(state)
            )
            if (
                heat
                or jacket_program is not None
                or action.get("operation") not in {"wait", "measure"}
            ):
                return ModelProviderResult(
                    {},
                    success=False,
                    outcome="rejected",
                    failure_reason="chemostatted_catalysis_requires_declared_isotherm",
                    diagnostics={},
                )
            system = chemostatted_catalytic_system(
                spec, slot, boundary, potential, support_heat_capacity_J_K=support
            )
            result = system.advance(initial, float(action["duration_s"]))
            final = result.final_state
            candidate = install_chemostatted_material(state, spec, boundary, slot, final, record)
            work = final.coordinates["electrical_work_J"] - initial.coordinates["electrical_work_J"]
            diagnostic = {
                "potential_reference": "SHE",
                "stopping_event": result.stopping_event,
                "energy_scope": (
                    "declared open half-cell with authored single-ion reference; "
                    "not full-cell calorimetry"
                ),
                "electrical_work_J": work,
                "bath_heat_J": final.boundary_heat_J,
                "boundary_exchange_mol": {
                    k: v - initial.exchanged_mol.get(k, 0) for k, v in final.exchanged_mol.items()
                },
                "boundary_exchange_enthalpy_J": {
                    k: v - initial.exchange_energy_J.get(k, 0)
                    for k, v in final.exchange_energy_J.items()
                },
                "energy_balance_residual_J": result.energy_balance_residual_J,
                "conservation_residuals": dict(result.conservation_residuals),
                "solver": dict(result.solver_diagnostic),
            }
            candidate = candidate.replace(
                ledger=advance_ledger(
                    state.ledger,
                    result.actual_duration_s,
                    cost_delta=max(work, 0) / 250000,
                    energy_deltas={
                        "flow_electrical_work_J": work,
                        "energy_jacket_J": final.boundary_heat_J,
                        "heat_loss_J": final.heat_loss_J,
                    },
                ),
                equipment=upsert_equipment_record(
                    candidate.equipment,
                    equipment_id=record.equipment_id,
                    equipment_type=record.equipment_type,
                    attached_vessel_id=state.vessel_id,
                    batch_reset="preserve",
                    settings={"last_chemostatted_catalysis": diagnostic},
                ),
            )
            self.validate_state(candidate)
        except (ArithmeticError, KeyError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="chemostatted_catalysis_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=result.actual_duration_s,
            outcome=result.outcome,
            diagnostics=diagnostic,
        )
