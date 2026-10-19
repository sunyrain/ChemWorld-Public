"""Public continuous-stream event fields."""

STREAM_FIELDS = {
    "set_flow_stream": ("connection", "flow_rate_mL_min"),
    "advance_flow": ("duration_s", "target_temperature_K", "current_mA"),
}
STREAM_OPERATIONS = tuple(STREAM_FIELDS)
