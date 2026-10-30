"""Ten continuous native cycles; no optional reference library is required."""

from chemworld.examples.native_battery_cycles import cycle_stages
from chemworld.physchem.battery_degradation import SEISpec
from chemworld.physchem.battery_models import BatterySPMSpec, BatteryThermalSpec
from chemworld.runtime import ChemWorldRuntime


def degradation_session():
    from chemworld.world.battery_world import battery_world_spec

    spec = BatterySPMSpec(thermal=BatteryThermalSpec(loss_conductance_W_K=0.1), sei=SEISpec())
    session = ChemWorldRuntime(composition=battery_world_spec(spec, budget=200))
    session.reset(seed=31)
    return spec, session


def run_cycle(session, spec):
    """Submit finite stages and wait; all switching remains in the runtime."""
    for stage in cycle_stages(spec):
        result = session.execute({"operation": "queue_electrical_stage", **stage})
        if result.info["transaction_status"] != "committed":
            raise RuntimeError(
                result.info["error_message"] or str(result.info["outcome_diagnostics"])
            )
    results = []
    for _ in range(2):  # The six declared stages sum to at most 24,600 s.
        result = session.execute({"operation": "wait", "duration_s": 14400})
        results.append(result)
        control = session._state.equipment.equipment["battery"].settings["electrical_control"]
        if result.info["transaction_status"] != "committed" or control["status"] != "running":
            break
    return results


def main() -> None:
    spec, session = degradation_session()
    for cycle in range(1, 11):
        results = run_cycle(session, spec)
        values = session._state.state_slots.slots[0].values
        control = session._state.equipment.equipment["battery"].settings["electrical_control"]
        print(
            {
                "cycle": cycle,
                "time_s": session._state.ledger.time_s,
                "status": control["status"],
                "completed_stages": control["stage_index"],
                "film_m": values["sei:carbonate"]
                * spec.sei.molar_volume_m3_mol
                / spec.negative.area_m2,
                "EC_mol": values["sei:EC"],
                "retained_ethylene_mol": values["sei:ethylene"],
                "outcomes": [r.info["operation_outcome"] for r in results],
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
