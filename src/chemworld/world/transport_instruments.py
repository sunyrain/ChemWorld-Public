"""Dimensional gauges for the installed liquid transport apparatus."""

from chemworld.foundation import Instrument, Observation, ObservationChannel
from chemworld.physchem.transport_devices import transport_rates
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.instruments import InstrumentContract


def read_transport(state, action, rng):
    from chemworld.runtime.transport_devices import unpack_transport

    record, spec, physical = unpack_transport(state)
    temperatures = tuple(
        spec.at_ph(i, n, h)
        for i, (n, h) in enumerate(zip(physical.amounts_mol, physical.enthalpies_J, strict=True))
    )
    flow = record.settings.get("flow_control", {}).get("requested_flow_m3_s", 0)
    if (
        state.inactive_vessels[spec.source_vessel_id].volume_L <= spec.pickup_volume_L + 1e-12
        or state.volume_L >= spec.receiver_capacity_L - 1e-12
    ):
        flow = 0.0
    _, point = transport_rates(
        spec,
        physical,
        temperatures,
        flow,
    )
    margin = point.get("pump_outlet_enthalpy_margin_J_mol")
    if margin is not None and margin <= 0:
        _, point = transport_rates(spec, physical, temperatures, 0.0)
    values = {
        "flow_rate": ("mL/min", point["actual_flow_m3_s"] * 6e7),
        "pump_pressure": ("Pa", point["pressure_rise_Pa"]),
        "shaft_power": ("W", point["shaft_power_W"]),
        "receiver_temperature": ("K", state.temperature_K),
        "source_volume": ("L", state.inactive_vessels[spec.source_vessel_id].volume_L),
    }
    return Observation(
        channels=tuple(ObservationChannel(k, u, v, uncertainty=0) for k, (u, v) in values.items()),
        instrument_id="flow_meter",
        cost=0.01,
        sample_consumed_L=0,
    )


def transport_instrument_spec():
    definition = Instrument("flow_meter", "Synthetic liquid transport gauges", (), 0.01, 0, {})
    contract = InstrumentContract(
        instrument_id="flow_meter",
        observable_keys=(),
        input_state_schema={"sample_consumption_L": 0},
        axis_contract={"shape": "scalar"},
        raw_signal_schema={"flow_rate": {"unit": "mL/min"}},
        processed_estimate_schema={},
        uncertainty_model="noiseless synthetic gauges",
        noise_model={},
        calibration_contract={},
        detection_contract={},
        saturation_contract={},
        baseline_drift_contract={},
        missingness_contract={},
        cost=0.01,
        latency_s=0,
        sample_consumption_L=0,
        destructive=False,
        requires_terminated=False,
        calibration_profile="finite-flow-gauge-v1",
        synthetic_boundary="flow, pressure, power, volume and T",
        clock_semantics="in_situ_end",
        analysis_time_s=0,
    )
    return InstrumentSpec(definition, contract, native_observer=read_transport)
