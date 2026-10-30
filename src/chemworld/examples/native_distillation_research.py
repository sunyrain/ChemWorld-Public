"""Explicit finite-column synthetic data; no real-fluid parameter claim."""

from chemworld.examples.native_flash_research import synthetic_vle_properties
from chemworld.physchem.distillation_units import ShortcutColumnSpec
from chemworld.physchem.dynamic_distillation import DynamicColumnSpec, DynamicColumnState


def synthetic_column_spec(trays=3):
    equipment = ShortcutColumnSpec(
        theoretical_stages=trays,
        stage_efficiency=1,
        maximum_reboiler_power_W=10,
        maximum_condenser_power_W=15,
        maximum_internal_vapor_rate_mol_s=0.01,
        maximum_batch_amount_mol=0.1,
        minimum_bottoms_amount_mol=1e-4,
        maximum_distillate_cut_fraction=0.8,
        minimum_pressure_Pa=5e4,
        maximum_pressure_Pa=2e5,
        maximum_temperature_K=430,
        maximum_duration_s=14400,
        maximum_reflux_ratio=10,
        provenance_id="synthetic-finite-column-development-v1",
    )
    return DynamicColumnSpec(synthetic_vle_properties(), equipment)


def initial_column_state(spec):
    amounts, energies = [], []
    for index in range(spec.cell_count):
        total = 0.02 if index == 0 else 0 if index == spec.collector_index else 0.002
        n = dict.fromkeys(spec.properties.component_ids, total / 2)
        amounts.append(n)
        energies.append(spec.at_tp(index, n, 300).total_enthalpy_j if total else 0)
    return DynamicColumnState(tuple(amounts), tuple(energies))


def run_example():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.dynamic_column_world import dynamic_column_world_spec

    spec = synthetic_column_spec()
    session = ChemWorldRuntime(
        composition=dynamic_column_world_spec(spec, initial_column_state(spec))
    )
    session.reset(seed=20)
    try:
        steps = []
        for action in (
            {
                "operation": "distill",
                "target_temperature_K": 430,
                "duration_s": 120,
                "reflux_ratio": 1,
            },
            {"operation": "measure", "instrument": "phase_state"},
            {"operation": "collect_fraction", "transfer_fraction": 0.5},
            {
                "operation": "distill",
                "target_temperature_K": 430,
                "duration_s": 10,
                "reflux_ratio": 3,
            },
            {"operation": "terminate"},
        ):
            step = session.execute(action)
            if step.info["transaction_status"] != "committed":
                raise RuntimeError(f"Column example failed: {step.info}")
            steps.append(
                {
                    "action": action,
                    "outcome": step.info["operation_outcome"],
                    "actual_duration_s": step.info["actual_duration_s"],
                }
            )
        return {
            "steps": steps,
            "physical_time_s": session._state.ledger.time_s,
            "ended": step.terminated,
            "reading": session.observe().to_dict(),
            "timed_fractions": session._state.metadata["column_timed_fractions"],
        }
    finally:
        session.close()


def main() -> None:
    import json

    print(json.dumps(run_example(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
