"""Finite kinetic calcite, acid input and phase-owned transfer without upstreams."""

from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_mineral import committed_speciation
from chemworld.runtime.mineral_kinetics import mineral_geometry
from chemworld.world.mineral_world import mineral_world_spec


def run_example():
    session = ChemWorldRuntime(composition=mineral_world_spec())
    session.reset(seed=33)
    trace = []
    try:
        for action in (
            {"operation": "wait", "duration_s": 100},
            {"operation": "measure", "instrument": "aqueous_ph"},
            {"operation": "add_component", "component": "HCl-solution", "amount_mol": 0.01},
            {"operation": "wait", "duration_s": 300},
            {
                "operation": "route_material",
                "connection": 0,
                "transfer_fraction": 0.25,
                "mixing": 1,
            },
            {"operation": "route_material", "connection": 1, "transfer_fraction": 0.1, "mixing": 1},
            {"operation": "select_vessel", "vessel": "receiver"},
            {"operation": "wait", "duration_s": 1000},
            {"operation": "sample", "sample_volume_L": 0.001},
            {"operation": "measure", "instrument": "aqueous_ph"},
        ):
            result = session.execute(action)
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(result.info.get("error_message"))
            if action["operation"] == "measure" and result.info["measurement_status"] != "valid":
                raise RuntimeError("finite mineral pH observation failed")
            state = session._state
            trace.append(
                {
                    "action": action,
                    "vessel": state.vessel_id,
                    "global_time_s": state.ledger.time_s,
                    "public_reading": {
                        c.channel_id: c.value for c in result.observation.channels if c.observed
                    },
                }
            )
        state = session._state
        return {
            "mode": "synthetic-development",
            "operations": trace,
            "final_receiver": committed_speciation(state).to_dict(),
            "final_surface_geometry": mineral_geometry(state).to_dict(),
            "sample_consumed_L": state.ledger.sample_consumed_L,
            "vessel_elapsed_s": state.vessel_elapsed_s,
            "actual_cost": state.ledger.cost,
            "retained_source_solid_mol": state.inactive_vessels["reactor"].species_amounts[
                "Calcite"
            ],
            "not_covered": ["thermal_calorics", "mineral_nucleation", "incompatible_family_mixing"],
        }
    finally:
        session.close()


if __name__ == "__main__":
    import json

    print(json.dumps(run_example(), indent=2, allow_nan=False))
