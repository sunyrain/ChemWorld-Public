"""Material-owned calcite surface geometry and one shared-clock kinetic owner."""

from dataclasses import replace
from math import isclose, isfinite

from chemworld.foundation import species_with_added_initial_amounts
from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.mineral_kinetics import (
    MineralGeometry,
    MineralKineticsSpec,
    advance_mineral,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.aqueous_mineral import (
    aqueous_model,
    aqueous_record,
    close_aqueous,
    install_aqueous,
    validate_aqueous_vessel,
)
from chemworld.runtime.kernel_contracts import ModelProviderResult

GEOMETRY_TYPE = "calcite_surface_reference"


def kinetic_record(state):
    record = aqueous_record(state)
    return record if record is not None and "mineral_kinetics" in record.settings else None


def geometry_slot(vessel, geometry):
    spec = StateSlotSpec(
        f"calcite-lot:{vessel}:geometry",
        "material",
        f"calcite-lot:{vessel}",
        GEOMETRY_TYPE,
        (StateFieldSpec("reference_area_m2", "m^2"), StateFieldSpec("reference_amount_mol", "mol")),
        mechanism_id="single-calcite-family-v1",
        phase_id="calcite",
    )
    return StateSlot(spec, vessel, geometry.to_dict())


def mineral_geometry(state):
    if any(
        s.spec.owner_kind == "material" and s.spec.state_type != GEOMETRY_TYPE
        for s in state.state_slots.slots
    ):
        raise ValueError("this mineral slice supports only its declared calcite material state")
    slots = [
        s
        for s in state.state_slots.slots
        if s.spec.state_type == GEOMETRY_TYPE
        and (s.values["reference_area_m2"] > 0 or s.values["reference_amount_mol"] > 0)
    ]
    if len(slots) > 1:
        raise ValueError("this mineral slice supports one evolving material family per vessel")
    if not slots:
        if state.species_amounts["Calcite"] > 0:
            raise ValueError("finite mineral stock has no owned reference geometry")
        return MineralGeometry(0, 0)
    slot = slots[0]
    if (
        slot.spec.owner_kind != "material"
        or slot.spec.phase_id != "calcite"
        or slot.location_id != state.vessel_id
    ):
        raise ValueError("mineral surface must follow its actual solid phase")
    geometry = MineralGeometry(**slot.values)
    if state.species_amounts["Calcite"] > 0 and geometry.reference_amount_mol <= 0:
        raise ValueError("nonempty mineral requires its finite reference geometry")
    return geometry


def without_exhausted_geometry(state):
    if state.species_amounts["Calcite"] > 0:
        return state
    return state.replace(
        state_slots=StateSlots(
            tuple(s for s in state.state_slots.slots if s.spec.state_type != GEOMETRY_TYPE)
        )
    )


class MineralKineticsRuntime:
    supported_operations = ("wait",)

    def __init__(self, composition=None):
        self.composition = composition

    def supports(self, state):
        return kinetic_record(state) is not None

    def validate_world(self):
        if self.composition is None or self.composition.physics is None:
            return
        physics = self.composition.physics
        installed = {v for v, s in physics.initial_states.items() if self.supports(s)}
        declared = {c.vessel for c in self.composition.components if c.kind == "mineral_kinetics"}
        if installed != declared:
            raise ValueError("kinetic mineral compartments require their unique declared owner")
        for vessel in installed:
            state = physics.initial_states[vessel]
            if not aqueous_model(state).allow_calcite:
                raise ValueError("kinetic mineral requires its allowed calcite phase")
            MineralKineticsSpec.from_dict(kinetic_record(state).settings["mineral_kinetics"])
            mineral_geometry(state)
            validate_aqueous_vessel(state, physics)

    def advance(self, state, action, *, heat=False, jacket_program=None):
        if heat or jacket_program is not None:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="mineral_temperature_data_unavailable",
                diagnostics={},
            )
        try:
            model = MineralKineticsSpec.from_dict(
                kinetic_record(state).settings["mineral_kinetics"]
            )
            report = advance_mineral(
                model,
                aqueous_model(state),
                state.species_amounts,
                mineral_geometry(state),
                float(action["duration_s"]),
                temperature_K=state.temperature_K,
                pressure_Pa=state.pressure_Pa,
            )
            candidate = without_exhausted_geometry(install_aqueous(state, report.final_aqueous))
            candidate = candidate.replace(
                ledger=advance_ledger(state.ledger, report.actual_duration_s),
                metadata={**candidate.metadata, "last_mineral_advance": report.to_dict()},
            )
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="mineral_kinetics_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=report.actual_duration_s,
            diagnostics={"mineral_kinetics": report.to_dict()},
        )


def route_mineral_phase(state, physics, action):
    """Move the selected material phase, its original provenance and its geometry."""
    edges = state.metadata["component_network"]["connections"]
    index, fraction = action.get("connection"), action.get("transfer_fraction")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(edges):
        raise ValueError("unknown mineral phase connection")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, int | float)
        or not isfinite(fraction)
        or not 0 < fraction <= 1
    ):
        raise ValueError("mineral phase fraction must be in (0,1]")
    edge = edges[index]
    phase_id = edge["source"]["port"]
    if (
        edge["source_vessel"] != state.vessel_id
        or phase_id not in {"aqueous", "calcite"}
        or edge["target"]["port"] != phase_id
    ):
        raise ValueError("select a matching aqueous or calcite material outlet")
    target = state.inactive_vessels[edge["target_vessel"]]
    if kinetic_record(target) is None or aqueous_model(state) != aqueous_model(target):
        raise ValueError("mineral endpoints require identical declared water chemistry")
    source_geometry, target_geometry = mineral_geometry(state), mineral_geometry(target)
    if (
        kinetic_record(state).settings["mineral_kinetics"]
        != kinetic_record(target).settings["mineral_kinetics"]
    ):
        raise ValueError("phase transfer requires the same intrinsic mineral rate law")
    aqueous_model(state).validate_conditions(state.temperature_K, state.pressure_Pa)
    aqueous_model(target).validate_conditions(target.temperature_K, target.pressure_Pa)
    if state.temperature_K != target.temperature_K or action.get("mixing") != 1:
        raise ValueError("mineral phase mixing requires equal temperature and explicit mixing")
    source_phase = state.phases.phases[phase_id]
    if not any(source_phase.species_amounts_mol.values()):
        raise ValueError("selected material phase is empty")
    if phase_id == "aqueous" and fraction == 1:
        raise ValueError("aqueous phase transfer must retain positive source water")
    moved = {k: n * fraction for k, n in source_phase.species_amounts_mol.items()}
    volume = source_phase.volume_L * fraction
    source_slots, target_slots = state.state_slots, target.state_slots
    if phase_id == "calcite":
        if target.species_amounts["Calcite"] > 0:
            pairs = (
                (
                    state.species_amounts["Calcite"] / source_geometry.reference_amount_mol,
                    target.species_amounts["Calcite"] / target_geometry.reference_amount_mol,
                ),
                (
                    source_geometry.reference_area_m2 / source_geometry.reference_amount_mol,
                    target_geometry.reference_area_m2 / target_geometry.reference_amount_mol,
                ),
            )
            if any(not isclose(a, b, rel_tol=1e-9, abs_tol=1e-14) for a, b in pairs):
                raise ValueError(
                    "mineral portions with different size histories require separate families"
                )
        portion = state.state_slots.material(phase_id="calcite").portion(fraction, target.vessel_id)
        remaining = state.state_slots.material(phase_id="calcite").portion(
            1 - fraction, state.vessel_id
        )
        source_slots = StateSlots((*state.state_slots.stationary().slots, *remaining.slots))
        target_slots = StateSlots(
            (
                *target.state_slots.stationary().slots,
                *target.state_slots.material().merge(portion).slots,
            )
        )

    def candidate(local, sign, slots):
        amounts = {k: n + sign * moved.get(k, 0) for k, n in local.species_amounts.items()}
        phases = PhaseLedger(
            {
                k: replace(
                    p,
                    species_amounts_mol={
                        s: n + sign * moved.get(s, 0) for s, n in p.species_amounts_mol.items()
                    },
                    volume_L=p.volume_L + sign * volume,
                )
                if k == phase_id
                else p
                for k, p in local.phases.phases.items()
            }
        )
        result = without_exhausted_geometry(
            close_aqueous(
                local.replace(
                    species_amounts=amounts,
                    phases=phases,
                    state_slots=slots,
                    volume_L=local.volume_L + sign * volume,
                )
            )
        )
        mineral_geometry(result)
        return result

    source, target_next = candidate(state, -1, source_slots), candidate(target, 1, target_slots)
    share = sum(moved.values()) / sum(state.species_amounts.values())
    reference = {k: n * share for k, n in state.species.initial_amounts_mol.items()}
    source = source.replace(
        species=replace(
            state.species,
            initial_amounts_mol={
                k: n * (1 - share) for k, n in state.species.initial_amounts_mol.items()
            },
        )
    )
    target_next = target_next.replace(
        species=species_with_added_initial_amounts(target.species, reference)
    )
    receipt = {
        "source": state.vessel_id,
        "destination": target.vessel_id,
        "amounts_mol": moved,
        "source_port": phase_id,
        "volume_L": volume,
        "model": "phase_owned_finite_calcite",
        "not_covered": ["thermal_mixing", "PV_work"],
    }
    transfer = {
        "connection": edge["id"],
        "source": state.vessel_id,
        "destination": target.vessel_id,
        "source_port": phase_id,
        "fraction": fraction,
        "amounts_mol": moved,
        "time_s": state.ledger.time_s,
        "mixing": 1,
    }
    return source.replace(
        inactive_vessels={**state.inactive_vessels, target.vessel_id: target_next},
        ledger=advance_ledger(state.ledger, 0, cost_delta=0.005),
        samples=replace(
            state.samples,
            transfers=(*state.samples.transfers, transfer),
            transfer_tools_used=state.samples.transfer_tools_used + 1,
        ),
        metadata={**source.metadata, "last_material_packet": receipt},
    )
