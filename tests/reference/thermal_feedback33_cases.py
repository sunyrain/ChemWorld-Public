"""Fixed synthetic controller/plant source declarations, independent of native code."""

from copy import deepcopy


def case(kind="PI", direction="heat"):
    initial = 300.0 if direction == "heat" else 304.0
    return {
        "id": f"linear-{kind}-{direction}",
        "controller": {
            "model": "finite-thermal-feedback-v1",
            "proportional_gain": 2.0,
            "integral_gain_s_inv": 0.2 if "I" in kind else 0.0,
            "derivative_gain_s": 0.4 if "D" in kind else 0.0,
            "derivative_filter_time_s": 0.5,
            "antiwindup_rate_s_inv": 1.0 if "I" in kind else 0.0,
            "bias_temperature_K": 302.0,
            "minimum_command_K": 290.0,
            "maximum_command_K": 420.0,
            "pause_rule": "freeze-integral-track-measurement",
        },
        "device": {
            "conductance_W_K": 2.0,
            "maximum_power_W": 30.0,
            "maximum_cooling_power_W": 20.0,
            "setpoint_K": 302.0,
            "calibration_factor": 1.0,
            "enabled": True,
            "mode": "finite_power",
            "minimum_setpoint_K": 280.0,
            "maximum_setpoint_K": 430.0,
        },
        "plant": {
            "capacity_J_K": 20.2,
            "ambient_K": 300.0,
            "loss_W_K": 0.2,
            "n_mol": 0.01,
            "volume_L": 1.0,
            "minimum_temperature_K": 280.0,
            "stop_temperature_K": 425.0,
            "maximum_pressure_Pa": 1e6,
        },
        "initial": [initial, 0.0, initial, 0.0, 0.0],
        "segments": [
            {
                "duration_s": 20.0,
                "target_K": 302.0,
                "enabled": True,
                "sample_times_s": [0.1, 1.0, 5.0, 20.0],
            }
        ],
        "linear_reference": True,
        "expected_event": None,
    }


def dynamic_cases():
    cards = [
        case(kind, direction) for kind in ("P", "PI", "PD", "PID") for direction in ("heat", "cool")
    ]
    for limiting in ("heat-power", "cool-power", "command-high", "command-low"):
        c = case("PID")
        c.update(id="saturation-" + limiting, linear_reference=False)
        c["controller"]["bias_temperature_K"] = 350.0
        target = 400.0 if limiting == "heat-power" else 300.0 if limiting == "cool-power" else 350.0
        if limiting in ("cool-power", "command-low"):
            c["initial"] = [400.0, 0.0, 400.0, 0.0, 0.0]
        c["device"]["setpoint_K"] = target
        c["segments"][0]["target_K"] = target
        if limiting.startswith("command"):
            c["device"].update(maximum_power_W=10000.0, maximum_cooling_power_W=10000.0)
            c["controller"][
                "maximum_command_K" if limiting == "command-high" else "minimum_command_K"
            ] = 320.0 if limiting == "command-high" else 380.0
        cards.append(c)
    for antiwindup in (0.0, 1.0):
        c = case("PI")
        c.update(id=f"recovery-antiwindup-{antiwindup}", linear_reference=False)
        c["controller"].update(bias_temperature_K=350.0, antiwindup_rate_s_inv=antiwindup)
        c["device"].update(setpoint_K=350.0, maximum_power_W=2.0)
        c["segments"] = [
            {"duration_s": 20.0, "target_K": 350.0, "enabled": True, "sample_times_s": [1.0, 20.0]},
            {
                "duration_s": 80.0,
                "target_K": 305.0,
                "enabled": True,
                "sample_times_s": [1.0, 20.0, 80.0],
            },
        ]
        cards.append(c)
    for kind in ("PI", "PID"):
        c = case(kind)
        c.update(id="pause-resume-" + kind, linear_reference=False)
        c["segments"] = [
            {"duration_s": dt, "target_K": 302.0, "enabled": active, "sample_times_s": [dt]}
            for dt, active in ((5.0, True), (5.0, False), (10.0, True))
        ]
        cards.append(c)
    for event in ("minimum_temperature", "temperature_limit", "pressure_limit"):
        c = case("PI")
        c.update(id="event-" + event, linear_reference=False, expected_event=event)
        target = 299.0 if event == "minimum_temperature" else 400.0
        c["controller"]["bias_temperature_K"] = target
        c["device"]["setpoint_K"] = target
        c["segments"] = [
            {"duration_s": 10.0, "target_K": target, "enabled": True, "sample_times_s": []}
        ]
        if event == "minimum_temperature":
            c["plant"]["minimum_temperature_K"] = 299.0
            c["plant"]["ambient_K"] = 280.0
            c["controller"]["minimum_command_K"] = 299.0
            c["controller"].update(integral_gain_s_inv=0.0, antiwindup_rate_s_inv=0.0)
        elif event == "temperature_limit":
            c["plant"]["stop_temperature_K"] = 302.0
        else:
            c["plant"]["maximum_pressure_Pa"] = 0.01 * 8.31446261815324 * 301.0 * 1000.0
        cards.append(c)
    c = case("PID")
    c["id"] = "calibrated-conductance-PID"
    c["device"]["calibration_factor"] = 0.75
    cards.append(c)
    return deepcopy(cards)
