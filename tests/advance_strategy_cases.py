"""Fixed short histories for the declaration-dispatch migration."""

from dataclasses import replace

from chemworld.runtime import ChemWorldRuntime

CASES = (
    "inert",
    "bulk",
    "surface",
    "flash",
    "flash-stop",
    "liquid",
    "column",
    "concentration",
    "crystal",
    "battery",
    "ssa",
    "legacy",
    "legacy-stable",
)


def case(name):
    wait = {"operation": "wait", "duration_s": 1, "stirring_speed_rpm": 600}
    heat = {
        "operation": "heat",
        "duration_s": 1,
        "target_temperature_K": 380,
        "stirring_speed_rpm": 600,
    }
    if name in {"inert", "bulk", "surface"}:
        from test_shared_thermal_device import make_env

        return make_env(bulk=name == "bulk", surface=name == "surface"), [heat, wait]
    if name in {"legacy", "legacy-stable"}:
        return ChemWorldRuntime(seed=7), [
            {"operation": "add_reagent", "amount_mol": 0.001},
            {"operation": "add_solvent", "solvent": 0, "volume_L": 0.01},
            {**heat, "target_temperature_K": 300} if name == "legacy-stable" else heat,
            wait,
        ]
    if name == "battery":
        from chemworld.world.battery_world import battery_world_spec

        spec = battery_world_spec()
        actions = [{"operation": "configure_electrical_control", "current_A": 0.02}, wait]
    elif name == "ssa":
        from chemworld.world.biochemical_worlds import biochemical_world

        spec, actions = biochemical_world("reversible", seed=7), [wait, wait]
    elif name == "concentration":
        from test_native_concentration_runtime import make_session

        return make_session()[1], [{"operation": "concentrate", "duration_s": 1}, wait]
    else:
        from test_physics_model_declarations import make_world

        spec = make_world({"column": "distillation", "flash-stop": "flash"}.get(name, name))
        if name == "flash-stop":
            models = {
                **spec.physics.models_of("flash"),
                "reactor": replace(
                    spec.physics.models_of("flash")["reactor"], stop_temperature_K=360
                ),
            }
            spec = replace(
                spec, physics=replace(spec.physics, models={**spec.physics.models, "flash": models})
            )
            actions = [{**heat, "duration_s": 100, "target_temperature_K": 410}]
        elif name == "column":
            actions = [
                {
                    "operation": "distill",
                    "duration_s": 1,
                    "reflux_ratio": 1,
                    "target_temperature_K": 430,
                },
                wait,
            ]
        elif name == "crystal":
            actions = [
                {"operation": "seed_crystals", "seed_mass_g": 0.001},
                {**heat, "target_temperature_K": 320},
                wait,
            ]
        elif name == "liquid":
            actions = [wait, wait]
        else:
            actions = [heat, wait]
    return ChemWorldRuntime(composition=spec, seed=7), actions


def trace(name):
    from chemworld.data.logging import to_builtin

    session, actions = case(name)
    try:
        session.reset(seed=7)
        records = []
        for index, action in enumerate(actions):
            result = session.execute(action)
            expected = "rolled_back" if name == "legacy" and index == 2 else "committed"
            assert result.info["transaction_status"] == expected, result.info
            snapshot = session.snapshot().to_dict()
            records.append(
                to_builtin(
                    {
                        "action": action,
                        "state": session._state.to_dict(),
                        "outcome": result.info["operation_outcome"],
                        "transaction_status": result.info["transaction_status"],
                        "failure_reason": result.info.get("outcome_diagnostics", {}).get(
                            "failure_reason"
                        ),
                        "duration_s": result.info["actual_duration_s"],
                        "observation": result.info["native_observation"],
                        "resource_ledgers": snapshot["resource_ledgers"],
                        "rng_state": snapshot["rng_state"],
                    }
                )
            )
        return records
    finally:
        session.close()
