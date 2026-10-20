"""CC/CV/rest stages run on the shared controller and persist in snapshots."""

from chemworld.physchem.battery_models import BatterySPMSpec, BatteryThermalSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.battery_world import battery_world_spec


def cycle_stages(spec):
    current = spec.current_at_c_rate(0.5)
    cutoff = spec.current_at_c_rate(0.05)
    return (
        {"current_A": current, "stage_duration_s": 6000, "stop_voltage_below_V": 3.5},
        {"potential_V": 3.5, "stage_duration_s": 6000, "stop_abs_current_A": cutoff},
        {"current_A": 0, "stage_duration_s": 300},
        {"current_A": -current, "stage_duration_s": 6000, "stop_voltage_above_V": 3.9},
        {"potential_V": 3.9, "stage_duration_s": 6000, "stop_abs_current_A": cutoff},
        {"current_A": 0, "stage_duration_s": 300},
    )


def cycle_session():
    spec = BatterySPMSpec(thermal=BatteryThermalSpec(loss_conductance_W_K=0.1))
    session = ChemWorldRuntime(composition=battery_world_spec(spec))
    session.reset(seed=30)
    for stage in cycle_stages(spec) * 2:
        result = session.execute({"operation": "queue_electrical_stage", **stage})
        if result.info.get("transaction_status") != "committed":
            raise ValueError(result.info)
    return session


def main():
    session = cycle_session()
    for index in range(4):
        result = session.execute({"operation": "wait", "duration_s": 14400})
        control = session._state.equipment.equipment["battery"].settings["electrical_control"]
        print(
            {
                "advance": index + 1,
                "time_s": session._state.ledger.time_s,
                "stage": control.get("stage_index"),
                "status": control.get("status"),
                "temperature_K": session._state.temperature_K,
                "outcome": result.info.get("operation_outcome"),
                "transaction": result.info.get("transaction_status"),
                "transitions": result.info.get("outcome_diagnostics", {}).get(
                    "control_transitions"
                ),
            },
            flush=True,
        )
        if control.get("status") == "completed":
            break


if __name__ == "__main__":
    main()
