"""A native terminal voltmeter/ammeter, without hidden concentration profiles."""

from chemworld.foundation import Instrument, Observation
from chemworld.foundation.observations import ObservationChannel
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.instruments import InstrumentContract


def read_electrical(state, action, rng):
    from chemworld.runtime.transient_electrode import read_electrical_state

    reading = read_electrical_state(state)
    return electrical_observation(
        "electrical_meter", reading.terminal_potential_V, reading.applied_current_A
    )


def read_battery(state, action, rng):
    from chemworld.physchem.battery_models import battery_control_current, battery_readout
    from chemworld.runtime.battery import unpack_battery

    _, _, spec, physical, programme = unpack_battery(state)
    reading = battery_readout(spec, physical, battery_control_current(spec, physical, programme))
    return electrical_observation("battery_meter", reading.voltage_V, reading.current_A)


def electrical_observation(identity, potential_v, current_a):
    return Observation(
        channels=(
            ObservationChannel("terminal_potential", "V", potential_v, uncertainty=0),
            ObservationChannel("electrical_current", "A", current_a, uncertainty=0),
        ),
        instrument_id=identity,
        cost=0.01,
        sample_consumed_L=0,
    )


def electrode_instrument_specs():
    return (
        electrical_instrument_spec("electrical_meter", read_electrical),
        electrical_instrument_spec("battery_meter", read_battery),
    )


def electrical_instrument_spec(identity, observer):
    units = {"terminal_potential": "V", "electrical_current": "A"}
    definition = Instrument(
        identity,
        "Synthetic terminal voltmeter and ammeter",
        tuple(units),
        0.01,
        0,
        dict.fromkeys(units, 0.0),
    )
    contract = InstrumentContract(
        instrument_id=definition.id,
        observable_keys=definition.observable_keys,
        input_state_schema={"sample_consumption_L": 0},
        axis_contract={"shape": "scalar"},
        raw_signal_schema={key: {"unit": unit} for key, unit in units.items()},
        processed_estimate_schema={},
        uncertainty_model="noiseless synthetic terminal measurement",
        noise_model=definition.noise_std,
        calibration_contract={"gain": 1, "offset": 0},
        detection_contract={},
        saturation_contract={},
        baseline_drift_contract={"model": "none"},
        missingness_contract={},
        cost=0.01,
        latency_s=1,
        sample_consumption_L=0,
        destructive=False,
        requires_terminated=False,
        calibration_profile="synthetic-electrical-meter-v1",
        synthetic_boundary="Terminal voltage/current only; no concentration or internal potential",
        clock_semantics="in_situ_end",
        analysis_time_s=0,
    )
    return InstrumentSpec(definition, contract, native_observer=observer)
