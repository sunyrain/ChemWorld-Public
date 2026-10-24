"""One original phase/H-slot projection shared by molecular finite apparatus."""

from collections.abc import Mapping
from dataclasses import replace
from math import isclose
from typing import Any

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.molecular_mixture import MolecularCaloricState


def molecular_enthalpy_slot(
    state: Any, phase_id: str, enthalpy: float, state_type: str
) -> StateSlots:
    return StateSlots(
        (
            StateSlot(
                StateSlotSpec(
                    f"{state.vessel_id}:material:enthalpy",
                    "material",
                    state.vessel_id,
                    state_type,
                    (StateFieldSpec("enthalpy_J", "J", minimum=None),),
                    phase_id=phase_id,
                    batch_reset="initial",
                ),
                state.vessel_id,
                {"enthalpy_J": enthalpy},
            ),
        )
    )


def install_molecular_tank(
    state: Any,
    closed: MolecularCaloricState,
    amounts: Mapping[str, float],
    enthalpy: float,
    *,
    phase_id: str,
    pressure_Pa: float,
    state_type: str,
) -> Any:
    volume = 1000 * sum(amounts.values()) / closed.molar_density_mol_m3
    return state.replace(
        species_amounts=dict(amounts),
        temperature_K=closed.temperature_K,
        pressure_Pa=pressure_Pa,
        volume_L=volume,
        phase=phase_id,
        phases=PhaseLedger(
            {
                phase_id: PhaseRecord(
                    phase_id,
                    state.vessel_id,
                    phase_id,
                    volume,
                    dict(amounts),
                    settled=True,
                    selected=True,
                    metadata={"carrier_representation": "species_inventory"},
                )
            }
        ),
        state_slots=molecular_enthalpy_slot(state, phase_id, enthalpy, state_type),
        process=replace(state.process, last_observation_record=None),
    )


def read_molecular_tank(
    state: Any,
    closed: MolecularCaloricState,
    *,
    phase_id: str,
    pressure_Pa: float,
    maximum_volume_L: float,
    state_type: str,
) -> float:
    """Read the owned H after verifying EOS, material and exact slot identity."""
    name, n = state.vessel_id, dict(state.species_amounts)
    total = sum(n.values())
    h = total * closed.enthalpy_J_mol
    volume = 1000 * total / closed.molar_density_mol_m3
    if (
        state.phase != phase_id
        or set(state.phases.phases) != {phase_id}
        or state.phases.phases[phase_id].species_amounts_mol != n
        or state.phases.phases[phase_id].vessel_id != name
        or not isclose(state.phases.phases[phase_id].volume_L, volume, rel_tol=1e-8, abs_tol=1e-10)
        or not isclose(state.volume_L, volume, rel_tol=1e-8, abs_tol=1e-10)
        or not isclose(state.pressure_Pa, pressure_Pa, abs_tol=1e-5)
        or volume > maximum_volume_L + 1e-7
    ):
        raise ValueError("molecular tank phase/pressure/volume disagrees with original EOS")
    expected = molecular_enthalpy_slot(state, phase_id, h, state_type).slots[0]
    slots = state.state_slots.slots
    if (
        len(slots) != 1
        or slots[0].spec != expected.spec
        or slots[0].location_id != name
        or set(slots[0].values) != {"enthalpy_J"}
        or not isclose(slots[0].values["enthalpy_J"], h, rel_tol=1e-8, abs_tol=1e-7)
    ):
        raise ValueError("molecular tank H ownership disagrees; other slots cannot be discarded")
    return slots[0].values["enthalpy_J"]
