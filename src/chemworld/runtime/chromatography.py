"""Instrument-owned retained inventory; shared sampling and public-trace calibration.

The effluent is a real external waste outlet, not a second stored sample. The
finite-memory model is lumped: elutable old residue is washed out exponentially;
the specified fraction of a new injection is retained for the next acquisition.
"""

from dataclasses import replace
from math import exp, isfinite

from chemworld.foundation import (
    equipment_settings,
    instrument_equipment_id,
    upsert_equipment_record,
)
from chemworld.foundation.balances import MaterialExchange
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.physchem.chromatography_methods import (
    ChromatographyProgram,
    EmpiricalChromatographyAnalyteSpec,
    fit_detector_response_calibration,
)
from chemworld.runtime.full_process_contract import sample_domain

INSTRUMENT_IDS = ("programmed_hplc", "programmed_gc")
SAMPLE_L = 0.0002
ANALYSIS_S = 60.0
COST = 0.05
MEMORY_TYPE = "chromatography_retained_inventory_v1"


def settings(state, instrument):
    if instrument not in INSTRUMENT_IDS:
        raise ValueError("Unknown programmed chromatography instrument")
    values = equipment_settings(state.equipment, instrument_equipment_id(instrument))
    if "chromatography" not in values:
        raise ValueError("No physical chromatography instance is installed at this vessel")
    return values


def model(state, instrument):
    value = settings(state, instrument)["chromatography"]
    for key in (
        "carryover_fraction",
        "wash_rate_s",
        "gain_per_analysis",
        "baseline_per_analysis",
        "baseline",
        "noise_sd",
        "saturation_AU",
        "initial_gain",
    ):
        if not isfinite(value[key]) or value[key] < 0:
            raise ValueError(f"Invalid instrument parameter: {key}")
    if value["carryover_fraction"] > 1 or min(value["initial_gain"], value["saturation_AU"]) <= 0:
        raise ValueError("Invalid carryover or detector scale")
    if not isfinite(value["baseline_drift"]):
        raise ValueError("Invalid detector baseline drift")
    if not 1 <= len(value["programs"]) <= 16:
        raise ValueError("Install one to sixteen finite programmes")
    analytes = tuple(EmpiricalChromatographyAnalyteSpec(**a) for a in value["analytes"])
    if len(analytes) != 3 or len({a.analyte_id for a in analytes}) != 3:
        raise ValueError("This finite instrument requires three distinct analytes")
    if any(
        a.instrument_id != instrument.removeprefix("programmed_") or a.tailing_factor != 1
        for a in analytes
    ):
        raise ValueError("Finite detector requires matching symmetric analytes")
    return value, analytes


def programme(state, instrument):
    config = settings(state, instrument)
    value, analytes = model(state, instrument)
    index = config.get("program_index", 0)
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(value["programs"])
    ):
        raise ValueError("Programme is not installed")
    result = ChromatographyProgram.from_dict(value["programs"][index])
    if result.duration_s != ANALYSIS_S:
        raise ValueError("This instrument declares a 60 s offline acquisition")
    for analyte in analytes:
        result.evaluate(analyte, 0)
    return result


def initial_memory(instrument, vessel, analytes):
    owner = instrument_equipment_id(instrument)
    spec = StateSlotSpec(
        owner + ":residue",
        "equipment",
        owner,
        MEMORY_TYPE,
        tuple(StateFieldSpec(a.analyte_id, "mol", species_id=a.analyte_id) for a in analytes),
        mechanism_id="lumped-chromatography-v1",
        batch_reset="preserve",
        time_evolution="static",
    )
    return StateSlot(spec, vessel, {a.analyte_id: 0.0 for a in analytes})


def memory(state, instrument):
    owner = instrument_equipment_id(instrument)
    matches = [
        s
        for s in state.state_slots.slots
        if s.spec.owner_id == owner and s.spec.state_type == MEMORY_TYPE
    ]
    if len(matches) != 1 or matches[0].location_id != state.vessel_id:
        raise ValueError("Instrument must have exactly one local retained inventory")
    return matches[0]


def sampled_amounts(state, volume=SAMPLE_L):
    phases = [state.phases.phases[k] for k in sample_domain(state)]
    available = sum(p.volume_L for p in phases)
    if volume > available or available <= 0:
        raise ValueError("Insufficient actual liquid for chromatography")
    result = {}
    for phase in phases:
        for key, amount in phase.species_amounts_mol.items():
            result[key] = result.get(key, 0) + amount * volume / available
    return result


def transport(state, instrument, injected):
    """Unique forward law, also used to independently validate external waste."""
    params, analytes = model(state, instrument)
    program = programme(state, instrument)
    slot = memory(state, instrument)
    if set(injected) - set(slot.values):
        raise ValueError("Instrument specimen contains undeclared analytes")
    retained, outlet, retention, released = {}, {}, {}, {}
    for analyte in analytes:
        key = analyte.analyte_id
        rt = program.retention_time_s(analyte)
        retention[key] = rt
        new = injected.get(key, 0.0)
        old = slot.values[key]
        release = old * (1 - exp(-params["wash_rate_s"] * ANALYSIS_S)) if rt is not None else 0
        released[key] = release
        new_retained = new * params["carryover_fraction"] if rt is not None else new
        retained[key] = old - release + new_retained
        outlet[key] = release + new - new_retained
    return retained, outlet, retention, released


def sample_outlet(before, action, injected):
    name = action["instrument"]
    _, outlet, _, released = transport(before, name, injected)
    return (
        MaterialExchange(
            "sample",
            f"vessel:{before.vessel_id}",
            f"sink:instrument:{name}",
            "out",
            tuple(sorted((k, n - released[k]) for k, n in outlet.items())),
        ),
        MaterialExchange(
            "instrument:carryover",
            instrument_equipment_id(name),
            f"sink:instrument:{name}",
            "out",
            tuple(sorted(released.items())),
        ),
    )


def sample_transform(before, withdrawn, action, seconds):
    instrument = action["instrument"]
    if seconds != ANALYSIS_S:
        raise ValueError("Declared and actual instrument duration disagree")
    slot = memory(before, instrument)
    retained, outlet, retention, _ = transport(before, instrument, sampled_amounts(before))
    changed = replace(slot, values=retained)
    # This record is the executed outlet, not a second current material stock.
    equipment = upsert_equipment_record(
        withdrawn.equipment,
        equipment_id=slot.spec.owner_id,
        equipment_type="instrument",
        attached_vessel_id=before.vessel_id,
        batch_reset="preserve",
        settings={
            "chromatography_acquisition": {
                "effluent_mol": outlet,
                "retention_s": retention,
                "sample_volume_L": SAMPLE_L,
            }
        },
    )
    return withdrawn.replace(
        equipment=equipment,
        state_slots=StateSlots(
            tuple(
                changed if s.spec.slot_id == slot.spec.slot_id else s
                for s in withdrawn.state_slots.slots
            )
        ),
    )


def configure(state, action):
    instrument = action["instrument"]
    config = settings(state, instrument)
    index = action["program_index"]
    if (
        isinstance(index, bool)
        or int(index) != index
        or not 0 <= index < len(config["chromatography"]["programs"])
    ):
        raise ValueError("Choose an installed programme")
    candidate = state.replace(
        equipment=upsert_equipment_record(
            state.equipment,
            equipment_id=instrument_equipment_id(instrument),
            equipment_type="instrument",
            attached_vessel_id=state.vessel_id,
            batch_reset="preserve",
            settings={"program_index": int(index)},
        )
    )
    programme(candidate, instrument)
    return candidate


def calibrate(state, action):
    """Record a declared standard and fit only already acquired public peak areas."""
    instrument = action["instrument"]
    config = settings(state, instrument)
    observation = state.process.last_observation_record
    if observation is None or observation.instrument_id != instrument:
        raise ValueError("Acquire a standard with this instrument before calibrating")
    channels = [observation.channel(f"peak_{i}_area") for i in range(1, 4)]
    if any(c.quality != "valid" or c.value is None for c in channels):
        raise ValueError("Calibration requires three resolved, unsaturated public peaks")
    points = list(config.get("standard_points", []))
    sample_id = channels[0].sample_id
    if any(p["sample_id"] == sample_id for p in points):
        raise ValueError("A standard acquisition cannot be counted twice")
    declared = [float(action[f"standard_{i}_M"]) for i in range(1, 4)]
    if any(not isfinite(c) or c < 0 for c in declared):
        raise ValueError("Known standard concentrations must be finite and nonnegative")
    points.append(
        {
            "sample_id": sample_id,
            "concentrations_M": declared,
            "areas_AU_s": [c.value for c in channels],
        }
    )
    updates = {"standard_points": points}
    if len(points) >= 3:
        updates["calibration"] = [
            fit_detector_response_calibration(
                [p["concentrations_M"][i] for p in points],
                [p["areas_AU_s"][i] for p in points],
                detector_id=f"peak_{i + 1}",
                provenance_id="declared-standards-and-acquired-public-traces",
            ).to_dict()
            for i in range(3)
        ]
    return state.replace(
        equipment=upsert_equipment_record(
            state.equipment,
            equipment_id=instrument_equipment_id(instrument),
            equipment_type="instrument",
            attached_vessel_id=state.vessel_id,
            batch_reset="preserve",
            settings=updates,
        )
    )
