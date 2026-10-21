"""Activity pH from committed carbonate speciation and the shared meter model."""

from chemworld.foundation import Instrument, Observation
from chemworld.foundation.observations import ObservationChannel
from chemworld.physchem.equilibrium_chemistry import aqueous_ph_observation
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.instruments import InstrumentContract


def read_aqueous_ph(state, action, rng):
    from chemworld.runtime.aqueous_mineral import committed_speciation

    reading = aqueous_ph_observation(
        committed_speciation(state), instrument_id="aqueous_ph", rng=rng
    )
    calibration = reading.raw_signal["calibration"]
    potential = reading.raw_signal["electrode_mV"]
    measured_pH = calibration["reference_pH"] - potential / calibration["slope_mV_pH"]
    quality = reading.raw_signal["quality"]
    return Observation(
        channels=(
            ObservationChannel(
                "pH",
                "pH",
                measured_pH,
                quality=quality,
                uncertainty=reading.uncertainty["pH_std"],
                role="processed", derived_from=("electrode_potential",),
                acquisition_relation="acquired_trace_fit",
            ),
            ObservationChannel(
                "electrode_potential",
                "mV",
                potential,
                quality=quality,
                uncertainty=reading.uncertainty["pH_std"] * calibration["slope_mV_pH"],
            ),
        ),
        raw_signal={"processing_calibration": calibration},
        instrument_id="aqueous_ph",
        cost=0.02,
        sample_consumed_L=0,
    )


def aqueous_instrument_specs():
    definition = Instrument(
        "aqueous_ph",
        "Synthetic activity pH electrode",
        ("pH", "electrode_potential"),
        0.02,
        0,
        {"pH": 0.02, "electrode_potential": 1.1832},
    )
    contract = InstrumentContract(
        instrument_id=definition.id,
        observable_keys=definition.observable_keys,
        input_state_schema={"required_phase": "liquid", "sample_consumption_L": 0},
        axis_contract={"shape": "scalar"},
        raw_signal_schema={"electrode_potential": {"unit": "mV"}},
        processed_estimate_schema={"pH": {"unit": "pH", "basis": "hydrogen activity"}},
        uncertainty_model="Gaussian pH noise with .01 pH quantization",
        noise_model=definition.noise_std,
        calibration_contract={"pH_offset": 0},
        detection_contract={},
        saturation_contract={"pH_range": [0, 14]},
        baseline_drift_contract={"model": "none"},
        missingness_contract={},
        cost=0.02,
        latency_s=1,
        sample_consumption_L=0,
        destructive=False,
        requires_terminated=False,
        calibration_profile="synthetic-aqueous-ph-v1",
        synthetic_boundary="Activity pH in the declared dilute carbonate system at 298.15 K",
        clock_semantics="in_situ_end",
        analysis_time_s=0,
    )
    return (InstrumentSpec(definition, contract, native_observer=read_aqueous_ph),)
