"""Finite public event language for persistent process controllers."""

from chemworld.foundation.gas import ATMOSPHERES

CONTROL_ACTION_FIELDS = {
    "configure_control": (
        "target_temperature_K",
        "pressure_Pa",
        "atmosphere",
        "ramp_rate_K_s",
        "control_interval_s",
        "headspace_L",
    ),
    "queue_control_stage": (
        "target_temperature_K",
        "pressure_Pa",
        "atmosphere",
        "ramp_rate_K_s",
        "stage_duration_s",
    ),
    "set_control_feedback": (
        "feedback_sensor",
        "feedback_threshold",
        "feedback_direction",
        "feedback_response",
    ),
    "advance_control": ("duration_s",),
    "pause_control": (),
    "resume_control": (),
}
CONTROL_OPERATIONS = tuple(CONTROL_ACTION_FIELDS)
CONTROL_CHOICES = {
    "atmosphere": ATMOSPHERES,
    "feedback_sensor": ("temperature", "pressure", "off"),
    "feedback_direction": ("above", "below"),
    "feedback_response": ("pause", "next_stage"),
}
CONTROL_NUMERIC = {
    "pressure_Pa": (50000.0, 400000.0, "Pa", 101325.0),
    "ramp_rate_K_s": (0.001, 1.0, "K/s", 0.1),
    "control_interval_s": (1.0, 60.0, "s", 10.0),
    "headspace_L": (0.001, 0.1, "L", 0.02),
    "stage_duration_s": (1.0, 14400.0, "s", 60.0),
    "feedback_threshold": (0.0, 550000.0, "sensor_native_unit", 350.0),
}
CONTROL_VECTOR_FIELDS = (*CONTROL_NUMERIC, *CONTROL_CHOICES)

CONTROL_MODEL = {
    "id": "persistent-sampled-process-control-v1",
    "stages_maximum": 32,
    "jacket": "continuous linear ramp; existing 90 W heating / 70 W cooling limits",
    "feedback": "quantized noiseless temperature/pressure; finite pause/next_stage responses",
    "clock": "explicit advancement; persistent cursor; sensors on controller cadence",
    "gas_boundary": "fixed-volume separately thermostatted ideal-gas pressure plenum",
    "gas_domain": "nitrogen/oxygen/argon; well-mixed finite flow; no gas-liquid chemistry",
    "gas_temperature": "tracks vessel temperature during controller advancement",
    "gas_energy": "plenum bath heat and external pump work separate from liquid heat",
    "supply_temperature_K": 298.15,
    "maximum_gas_flow_mol_s": 2e-5,
    "purge_flow_mol_s": 2e-6,
    "validity": "declared finite surrogate equipment model; no external device calibration",
}
