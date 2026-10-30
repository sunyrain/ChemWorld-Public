"""Reuse a heater across material initialization and bind two tasks to one world.

Run with ``python -m chemworld.examples.reuse_native_equipment`` after installation.
The printed chemical readings come from the declared synthetic gas instruments.
"""

import json

from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import compile_world_composition
from chemworld.world.surface_world import surface_world_spec


def _execute(session, **action):
    result = session.execute(action)
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(f"Reuse example could not execute {action['operation']}")
    return result


def _heater(session):
    return next(record for record in session._state.equipment.equipment.values()
                if record.equipment_type == "thermal_device"
                and record.attached_vessel_id == session._state.vessel_id)


def _feed(session):
    _execute(session, operation="add_component", component="CO-feed", amount_mol=2e-5)
    _execute(session, operation="add_component", component="O2-feed", amount_mol=1e-5)


def run_example():
    configurations = []
    for surface, bulk in ((False, False), (False, True), (True, False), (True, True)):
        session = ChemWorldRuntime(composition=surface_world_spec(
            include_surface=surface, include_bulk=bulk,
        ))
        try:
            session.reset(seed=12)
            original = _heater(session)
            readings = []
            for batch in range(2):
                if batch:
                    session.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
                _feed(session)
                _execute(session, operation="heat", target_temperature_K=380,
                         duration_s=2, stirring_speed_rpm=600)
                reading = _execute(session, operation="measure", instrument="gas_composition")
                readings.append([channel.to_dict() for channel in reading.observation.channels])
            retained = _heater(session)
            calibration = original.settings["thermal_device"]["calibration_factor"]
            if (original.equipment_id != retained.equipment_id
                    or retained.settings["thermal_device"]["calibration_factor"] != calibration):
                raise RuntimeError("Material initialization replaced the retained heater")
            configurations.append({
                "surface": surface, "bulk": bulk,
                "heater_id": retained.equipment_id,
                "heater_model": retained.settings["thermal_device"]["mode"],
                "same_heater_after_material_initialization": True,
                "calibration_factor": calibration, "readings": readings,
                "physical_time_s": session._state.ledger.time_s,
            })
        finally:
            session.close()

    world = compile_world_composition(surface_world_spec())
    assay = world.bind_task(surface_world_spec(terminal_assay=True).task)
    left, right = ChemWorldRuntime(composition=world), ChemWorldRuntime(composition=assay)
    try:
        left.reset(seed=12)
        right.reset(seed=12)
        for session in (left, right):
            _feed(session)
            _execute(session, operation="wait", duration_s=2, stirring_speed_rpm=600)
        same_response = (
            left._state.species_amounts == right._state.species_amounts
            and left._state.temperature_K == right._state.temperature_K
            and left._state.pressure_Pa == right._state.pressure_Pa
        )
        immediate_end = _execute(left, operation="terminate").terminated
        frozen = _execute(right, operation="terminate")
        assay_end = _execute(right, operation="measure", instrument="gas_final_assay")
        return {"configurations": configurations, "task_binding": {
            "same_physical_definition": (
                world.spec.physics.to_dict() == assay.spec.physics.to_dict()),
            "same_response_before_task_end": same_response,
            "observation_task_ended": immediate_end,
            "assay_task_waited_for_assay": not frozen.terminated,
            "assay_task_ended": assay_end.terminated,
            "assay_readings": [c.to_dict() for c in assay_end.observation.channels],
        }}
    finally:
        left.close()
        right.close()


def main() -> None:
    print(json.dumps(run_example(), indent=2))


if __name__ == "__main__":
    main()
