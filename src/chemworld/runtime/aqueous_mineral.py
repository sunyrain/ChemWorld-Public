"""One finite aqueous closure for native preparation, wait and observation.

Species reside only in PhaseLedger. Cached speciation is derived, checked
against those amounts before a measurement, and never used as material stock.
"""

from dataclasses import replace
from math import fsum, isclose

from chemworld.foundation import species_with_added_initial_amounts
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.aqueous_mineral import (
    CHARGES,
    COMPOSITIONS,
    FORMULAS,
    CarbonateEquilibriumResult,
    CarbonateEquilibriumSpec,
    solve_carbonate_equilibrium,
)
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.kernel_contracts import ModelProviderResult

AQUEOUS_TYPE = "carbonate_aqueous_equilibrium"


def aqueous_record(state):
    records = [
        r
        for r in state.equipment.equipment.values()
        if r.equipment_type == AQUEOUS_TYPE and r.attached_vessel_id == state.vessel_id
    ]
    if len(records) > 1:
        raise ValueError("aqueous phase requires one declared chemical model")
    return records[0] if records else None


def aqueous_model(state):
    record = aqueous_record(state)
    if record is None:
        raise ValueError("no native carbonate model is installed")
    return CarbonateEquilibriumSpec.from_dict(record.settings["physical_model"])


def install_aqueous(state, result):
    if not 0 < state.volume_L <= state.vessels.vessels[state.vessel_id].max_volume_L:
        raise ValueError("aqueous geometric holdup must fit the finite vessel")
    liquid = {k: v for k, v in result.amounts_mol.items() if k != "Calcite"}
    phases = {
        "aqueous": PhaseRecord(
            "aqueous",
            state.vessel_id,
            "aqueous",
            state.volume_L,
            liquid,
            selected=True,
            metadata={"carrier_representation": "species_inventory"},
        ),
        "calcite": PhaseRecord(
            "calcite",
            state.vessel_id,
            "solid",
            0,
            {"Calcite": result.calcite_mol},
            metadata={"carrier_representation": "species_inventory", "pure_mineral": True},
        ),
    }
    return state.replace(
        phases=PhaseLedger(phases),
        species_amounts=result.amounts_mol,
        phase="liquid",
        process=replace(state.process, last_observation_record=None),
        metadata={**state.metadata, "aqueous_equilibrium": result.to_dict()},
    )


def close_aqueous(state):
    record = aqueous_record(state)
    if record is None:
        return state
    result = solve_carbonate_equilibrium(
        aqueous_model(state),
        state.species_amounts,
        temperature_K=state.temperature_K,
        pressure_Pa=state.pressure_Pa,
        fixed_calcite_mol=(
            state.species_amounts["Calcite"] if "mineral_kinetics" in record.settings else None
        ),
    )
    return install_aqueous(state, result)


def committed_speciation(state):
    model = aqueous_model(state)
    model.validate_conditions(state.temperature_K, state.pressure_Pa)
    value = state.metadata.get("aqueous_equilibrium")
    if value is None:
        raise ValueError("aqueous measurement requires committed speciation")
    result = CarbonateEquilibriumResult.from_dict(value)
    if result.amounts_mol != state.species_amounts or (
        result.metadata["temperature_K"] != state.temperature_K
        or result.metadata["pressure_Pa"] != state.pressure_Pa
        or result.metadata["model_id"] != model.model_id
    ):
        raise ValueError("aqueous observation cache does not match committed material")
    return result


def exchange_aqueous_packet(source, target, physics, amounts, *, fraction=0, fixed_source=False):
    """Representative finite solution input; a pure mineral is not a liquid outlet."""
    model = aqueous_model(target)
    model.validate_conditions(source.temperature_K, source.pressure_Pa)
    model.validate_conditions(target.temperature_K, target.pressure_Pa)
    if source.vessel_id == target.vessel_id or source.phase != "liquid":
        raise ValueError("aqueous feed requires a distinct finite liquid vessel")
    if set(source.species_amounts) != set(FORMULAS) or source.species_amounts["Calcite"] > 0:
        raise ValueError("representative solution feed cannot include a separate mineral phase")
    available = fsum(source.species_amounts.values())
    moved = fsum(amounts.values())
    fraction = moved / available
    if not 0 <= fraction < 1 or any(
        not isclose(amounts.get(k, 0), fraction * v, rel_tol=1e-10, abs_tol=1e-14)
        for k, v in source.species_amounts.items()
    ):
        raise ValueError("aqueous packet must leave positive finite source water")
    remaining = {k: v - amounts.get(k, 0) for k, v in source.species_amounts.items()}
    received = {k: v + amounts.get(k, 0) for k, v in target.species_amounts.items()}
    source_candidate = source.replace(
        species_amounts=remaining,
        volume_L=source.volume_L * (1 - fraction),
        phases=PhaseLedger(
            {
                k: replace(
                    p,
                    volume_L=p.volume_L * (1 - fraction),
                    species_amounts_mol={
                        s: n * (1 - fraction) for s, n in p.species_amounts_mol.items()
                    },
                )
                for k, p in source.phases.phases.items()
            }
        ),
    )
    target_candidate = close_aqueous(
        target.replace(
            species_amounts=received,
            phases=PhaseLedger(
                {
                    "aqueous": PhaseRecord(
                        "aqueous",
                        target.vessel_id,
                        "aqueous",
                        target.volume_L + source.volume_L * fraction,
                        {k: n for k, n in received.items() if k != "Calcite"},
                        metadata={"carrier_representation": "species_inventory"},
                    ),
                    "calcite": PhaseRecord(
                        "calcite",
                        target.vessel_id,
                        "solid",
                        0,
                        {"Calcite": received["Calcite"]},
                        metadata={"carrier_representation": "species_inventory"},
                    ),
                }
            ),
            volume_L=target.volume_L + source.volume_L * fraction,
        )
    )
    reference = {k: n * fraction for k, n in source.species.initial_amounts_mol.items()}
    target_candidate = target_candidate.replace(
        species=species_with_added_initial_amounts(target.species, reference)
    )
    source_candidate = source_candidate.replace(
        species=replace(
            source.species,
            initial_amounts_mol={
                k: n * (1 - fraction) for k, n in source.species.initial_amounts_mol.items()
            },
        )
    )
    if aqueous_record(source_candidate) is not None:
        source_candidate = close_aqueous(source_candidate)
    return (
        source_candidate,
        target_candidate,
        {
            "source": source.vessel_id,
            "destination": target.vessel_id,
            "amounts_mol": amounts,
            "model": "finite_isothermal_carbonate",
            "not_covered": ["mixing_heat", "PV_work", "mineral_advection"],
        },
    )


def validate_aqueous_vessel(state, physics):
    """The common identity, domain and fast-chemistry contract for both owners."""
    if physics.network.reactions:
        raise ValueError("this bounded aqueous slice has no homogeneous kinetic coupling")
    if set(physics.network.species_ids) != set(FORMULAS) or any(
        s.composition != COMPOSITIONS[s.species_id]
        or s.charge != CHARGES[s.species_id]
        or s.phase != ("solid" if s.species_id == "Calcite" else "liquid")
        for s in physics.network.species
    ):
        raise ValueError("carbonate species identities, charge and phase must match the model")
    if any(
        state.vessel_id in m
        for m in (
            physics.thermal_models,
            physics.flash_models,
            physics.liquid_models,
            physics.column_models,
            physics.concentration_models,
            physics.crystal_models,
        )
    ):
        raise ValueError("carbonate constants do not declare thermal or other phase coupling")
    committed_speciation(state)
    result = close_aqueous(state)
    if any(
        not isclose(result.species_amounts[k], n, rel_tol=1e-8, abs_tol=1e-12)
        for k, n in state.species_amounts.items()
    ):
        raise ValueError("initial native aqueous material must already be equilibrated")


class AqueousMineralRuntime:
    supported_operations = ("wait",)

    def __init__(self, composition=None):
        self.composition = composition

    def supports(self, state):
        record = aqueous_record(state)
        return record is not None and "mineral_kinetics" not in record.settings

    def validate_world(self):
        if self.composition is None or self.composition.physics is None:
            return
        physics = self.composition.physics
        installed = {v for v, s in physics.initial_states.items() if self.supports(s)}
        declared = {c.vessel for c in self.composition.components if c.kind == "aqueous_mineral"}
        if installed != declared:
            raise ValueError("native aqueous compartments require their declared equilibrium owner")
        for vessel in installed:
            validate_aqueous_vessel(physics.initial_states[vessel], physics)

    def advance(self, state, action, *, heat=False, jacket_program=None):
        if heat or jacket_program is not None:
            return ModelProviderResult(
                {},
                success=False,
                outcome="rejected",
                failure_reason="aqueous_temperature_data_unavailable",
                diagnostics={},
            )
        try:
            duration = float(action["duration_s"])
            candidate = close_aqueous(state).replace(ledger=advance_ledger(state.ledger, duration))
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            return ModelProviderResult(
                {},
                success=False,
                outcome="numerical_failure",
                failure_reason="aqueous_equilibrium_failed",
                diagnostics={"solver_error": str(exc)},
            )
        return ModelProviderResult(
            {},
            state=candidate,
            actual_duration_s=duration,
            diagnostics={
                "aqueous_equilibrium": candidate.metadata["aqueous_equilibrium"],
                "not_covered": ["caloric_closure", "temperature_extrapolation", "PV_work"],
            },
        )
