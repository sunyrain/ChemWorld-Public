"""Explicitly opt-in synthetic model analysis, not a physical sweep acquisition."""

import json

from chemworld.foundation import Instrument, Observation
from chemworld.foundation.observations import ChannelAxis, ObservationChannel
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.instruments import InstrumentContract

THEVENIN_ANALYSIS = "thevenin_small_signal_analysis"
FREQUENCIES_HZ = tuple(10.**(-4+.5*k) for k in range(13))
COST = .02


def read_thevenin_analysis(state, action, rng):
    from chemworld.physchem.thevenin_frequency import thevenin_impedance
    from chemworld.runtime.thevenin import unpack_thevenin

    if {"frequencies_Hz", "frequency_Hz"} & action.keys():
        raise ValueError("analysis frequency grid is fixed by its public instrument contract")
    _, _, spec, physical, program = unpack_thevenin(state)
    if program.unit != "A" or program.value_at(state.ledger.time_s) != 0.:
        raise ValueError("small-signal analysis requires a zero-current controlled rest point")
    result = thevenin_impedance(spec, FREQUENCIES_HZ, state=physical)
    if not result.success:
        raise ValueError("small-signal analysis failed: "
                         +json.dumps(result.to_dict(), allow_nan=False))
    axis = ChannelAxis("frequency", "Hz", FREQUENCIES_HZ)
    return Observation(channels=(
        ObservationChannel("impedance_real", "Ohm",
            tuple(float(z.real) for z in result.impedance_ohm), axis=axis),
        ObservationChannel("impedance_imag", "Ohm",
            tuple(float(z.imag) for z in result.impedance_ohm), axis=axis),
    ), instrument_id=THEVENIN_ANALYSIS, cost=COST, sample_consumed_L=0.)


def thevenin_analysis_instrument_spec():
    keys = ("impedance_real", "impedance_imag")
    definition = Instrument(THEVENIN_ANALYSIS, "Synthetic stationary model small-signal analysis",
                            keys, COST, 0., dict.fromkeys(keys, 0.))
    contract = InstrumentContract(instrument_id=THEVENIN_ANALYSIS, observable_keys=keys,
        input_state_schema={"required_phase": "formal", "required_current_A": 0.},
        axis_contract={"shape": "curve", "points": len(FREQUENCIES_HZ), "unit": "Hz",
                       "values": list(FREQUENCIES_HZ)},
        raw_signal_schema={key: {"unit": "Ohm"} for key in keys}, processed_estimate_schema={},
        uncertainty_model="Deterministic local computation; parameter uncertainty not propagated",
        noise_model=definition.noise_std, calibration_contract={"definition": "Z=-dV/dI"},
        detection_contract={}, saturation_contract={}, baseline_drift_contract={"model": "none"},
        missingness_contract={"failed_frequency": "reject complete analysis; retain failure"},
        cost=COST, latency_s=0., sample_consumption_L=0., destructive=False,
        requires_terminated=False, calibration_profile="thevenin-small-signal-analysis-v1",
        synthetic_boundary=("Explicit model-based stationary local analysis with frozen boundary; "
            "not a physical sinusoidal acquisition or uncertainty estimate; "
            "no hidden state readout"),
        clock_semantics="instantaneous_snapshot", analysis_time_s=0.)
    return InstrumentSpec(definition, contract, native_observer=read_thevenin_analysis)
