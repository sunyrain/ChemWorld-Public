"""Independent detector noise and explicitly idealized integer molecular capture.

Capture is Bernoulli selection of solute molecules, NOT a liquid aliquot. The
reaction chamber's carrier and constant volume remain unchanged. Instrument
time/cost is charged by the existing runtime, never by these pure callbacks.
"""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from chemworld.foundation import (
    Instrument,
    Observation,
    ObservationChannel,
    equipment_settings,
    instrument_equipment_id,
    upsert_equipment_record,
)
from chemworld.foundation.balances import MaterialExchange
from chemworld.foundation.discrete_inventory import counts_to_mol, mol_to_counts
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.instruments import InstrumentContract

CAPTURE_PROBABILITY = 0.1
NOISE_SD = 1.0


def capture_plan(state):
    continuation = state.process.stochastic_continuation
    rng = np.random.Generator(np.random.PCG64())
    rng.bit_generator.state = deepcopy(continuation["sampling_rng_state"])
    counts = mol_to_counts(state.species_amounts)
    selected = {key: int(rng.binomial(counts[key], CAPTURE_PROBABILITY)) for key in sorted(counts)}
    return selected, deepcopy(rng.bit_generator.state)


def capture_transform(before, sampled, action, seconds):
    selected, sampling_rng = capture_plan(before)
    counts = mol_to_counts(before.species_amounts)
    continuation = deepcopy(sampled.process.stochastic_continuation)
    continuation.update(sampling_rng_state=sampling_rng, pending=None, signature=None)
    equipment = upsert_equipment_record(
        sampled.equipment,
        equipment_id=instrument_equipment_id("molecule_capture"),
        equipment_type="instrument",
        attached_vessel_id=sampled.vessel_id,
        batch_reset="preserve",
        settings={"captured_counts": selected},
    )
    return sampled.replace(
        species_amounts=counts_to_mol({key: n - selected[key] for key, n in counts.items()}),
        process=replace(sampled.process, stochastic_continuation=continuation),
        equipment=equipment,
    )


def capture_outlet(before, action, injected_amounts):
    selected, _ = capture_plan(before)
    return (
        MaterialExchange(
            "sample",
            f"vessel:{before.vessel_id}",
            "sink:instrument:molecule_capture",
            "out",
            tuple(counts_to_mol(selected).items()),
        ),
    )


def read_counts(state, action, rng):
    instrument = action["instrument"]
    if instrument == "molecule_capture":
        counts = equipment_settings(state.equipment, instrument_equipment_id(instrument))[
            "captured_counts"
        ]
    else:
        counts = mol_to_counts(state.species_amounts)
    # Calibrated signed detector response, one count per molecule. Do not clip
    # negative readings, substitute latent counts, or publish reaction events.
    channels = tuple(
        ObservationChannel(
            f"count_{key}", "counts", float(n + rng.normal(0, NOISE_SD)), uncertainty=NOISE_SD
        )
        for key, n in sorted(counts.items())
    )
    return Observation(channels=channels, instrument_id=instrument, cost=0.01, sample_consumed_L=0)


def biochemical_instrument_specs():
    result = []
    for name in ("molecule_counter", "molecule_capture"):
        capture = name == "molecule_capture"
        definition = Instrument(name, "Synthetic calibrated molecular readout", (), 0.01, 0, {})
        contract = InstrumentContract(
            instrument_id=name,
            observable_keys=(),
            input_state_schema={"required_phase": "liquid"},
            axis_contract={},
            raw_signal_schema={"count_species": {"unit": "counts"}},
            processed_estimate_schema={},
            uncertainty_model="independent Gaussian detector noise",
            noise_model={"sd_counts": NOISE_SD},
            calibration_contract={"counts_per_molecule": 1},
            detection_contract={"signed_readout": True},
            saturation_contract={},
            baseline_drift_contract={},
            missingness_contract={},
            cost=0.01,
            latency_s=0,
            sample_consumption_L=0,
            destructive=capture,
            requires_terminated=False,
            calibration_profile="synthetic-low-copy-v1",
            synthetic_boundary="Idealized molecular capture at p=0.1; no carrier withdrawal"
            if capture
            else "Noisily calibrated copy readout, not reaction-event truth",
            clock_semantics="offline_snapshot" if capture else "instantaneous_snapshot",
            analysis_time_s=1 if capture else 0,
        )
        result.append(
            InstrumentSpec(
                definition,
                contract,
                native_observer=read_counts,
                sample_transform=capture_transform if capture else None,
                sample_outlet=capture_outlet if capture else None,
                discrete_sampling=capture,
            )
        )
    return tuple(result)
