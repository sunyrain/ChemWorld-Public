"""Fixed synthetic thermal/SPM calibration through public native measurements."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from chemworld.data.calibration import (
    CalibrationParameter,
    CalibrationReadout,
    NativeCalibration,
    NativeCalibrationExperiment,
)
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.foundation import Observation, ObservationChannel
from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.world.battery_world import battery_world_spec
from chemworld.world.domain_modules import DomainModuleRegistry
from chemworld.world.surface_world import surface_world_spec


def ideal_pressure(state, action, rng):
    """Declared ideal gauge for the synthetic calibration design, without noise."""
    return Observation(
        channels=(ObservationChannel("pressure", "Pa", state.pressure_Pa, uncertainty=0),),
        instrument_id="gas_pressure", cost=.02,
    )


def thermal_world(ambient):
    base = surface_world_spec(include_surface=False, include_bulk=False)
    modules = DomainModuleRegistry(tuple(
        replace(module, instruments=tuple(
            replace(
                item, native_observer=ideal_pressure,
                definition=replace(item.definition, noise_std={"pressure": 0.0}),
                contract=replace(
                    item.contract, noise_model={"pressure": 0.0},
                    uncertainty_model="ideal gauge; separately declared data perturbation",
                ),
            ) if item.definition.id == "gas_pressure" else item for item in module.instruments
        )) for module in base.modules.modules
    ))
    models = dict(base.physics.models_of("thermal"))
    models["reactor"] = replace(models["reactor"], environment_temperature_K=ambient)
    return replace(
        base, modules=modules,
        physics=replace(base.physics, thermal_models=models, modules=modules),
    )


def parameters(kind, *, confounded=False):
    if kind == "thermal":
        base_path = ("physics", "models", "thermal", "bindings", "reactor")
        result = [CalibrationParameter(
            QuantityColumn("heat_loss", "W/K", "parameter", (.01, .2),
                           "conductance to the declared environment"),
            (*base_path, "loss_conductance_W_K"), .12,
        )]
        if confounded:
            result.append(CalibrationParameter(
                QuantityColumn("support_capacity", "J/K", "parameter", (5, 80),
                               "temperature-independent support heat capacity"),
                (*base_path, "thermodynamics", "support_heat_capacity_J_K"), 40.0,
            ))
        return tuple(result)
    if kind != "battery":
        raise ValueError("choose thermal or battery")
    return (CalibrationParameter(
        QuantityColumn("negative_diffusivity", "m^2/s", "parameter", (5e-15, 5e-14),
                       "isothermal negative particle diffusion coefficient"),
        ("physics", "initial_states", "reactor", "equipment", "battery",
         "settings", "physical_model", "negative", "diffusivity_m2_s"),
        4e-14,
    ),)


def fixed_inputs(kind, *, confounded=False):
    source = InputSource(DataSourceProvenance(
        "chemworld:conv17-synthetic", 0,
        "Authored fixed native thermal/SPM protocols in chemworld.examples.native_calibration",
        version="1", license="MIT (ChemWorld-authored synthetic observations)",
    ), "synthetic")
    thermal = kind == "thermal"
    times = (20, 50, 100, 180, 300, 450) if thermal else (30, 90, 180, 300)
    conditions = (300, 310, 320) if thermal else (.2, .4, .3)
    quantity, channel, unit, sigma = (
        ("pressure", "pressure", "Pa", 20.0) if thermal
        else ("voltage", "terminal_potential", "V", 5e-5)
    )
    columns = (
        QuantityColumn("time", "s", "condition", (0, 1000), "local experiment elapsed time"),
        QuantityColumn(quantity, unit, "observation", (1, 500000) if thermal else (0, 6),
                       "absolute pressure" if thermal else "terminal cell voltage"),
    )
    experiments, rows = [], []
    for group_index, condition in enumerate(conditions):
        split = "train" if group_index < 2 else "validation"
        group = f"{kind}-{split}-{group_index}"
        if thermal:
            world = thermal_world(condition)
            actions = [{"operation": "pause_control"}]
        else:
            world = battery_world_spec(BatterySPMSpec(
                negative=ParticleSpec(cells=12), positive=ParticleSpec(cells=12),
            ))
            actions = [{"operation": "configure_electrical_control", "current_A": condition}]
        readouts = []
        elapsed = 0
        for index, time in enumerate(times):
            actions.extend((
                {"operation": "wait", "duration_s": time - elapsed - 1},
                {
                    "operation": "measure",
                    "instrument": "gas_pressure" if thermal else "battery_meter",
                },
            ))
            row_id = f"{group}-{index}"
            readouts.append(CalibrationReadout(row_id, quantity, len(actions) - 1, channel))
            rows.append(ScientificRow(
                row_id, source.provenance.source_id, group, row_id, split,
                {
                    "time": InputValue(float(time), "s", 0, "Exact authored sampling schedule"),
                    quantity: InputValue(
                        100000.0 if thermal else 3.5, unit, sigma,
                        "Fixed declared synthetic standard uncertainty; see experiment note",
                    ),
                },
            ))
            elapsed = time
        experiments.append(NativeCalibrationExperiment(
            group, world, tuple(actions), tuple(readouts), seed=17 + group_index,
        ))
    table = ScientificTable(
        f"conv17-{kind}-synthetic-v1", "measurements", columns, (source,), tuple(rows),
    )
    return table, parameters(kind, confounded=confounded), tuple(experiments)


def generate_data(kind, progress=None):
    table, targets, experiments = fixed_inputs(kind)
    predictor = NativeCalibration(table, targets, experiments, progress=progress)
    truth = {"heat_loss": .06} if kind == "thermal" else {"negative_diffusivity": 1.5e-14}
    predictions = {
        **predictor.forward(truth, "train", stage="synthetic_data"),
        **predictor.forward(truth, "validation", stage="synthetic_data"),
    }
    pattern = (-.25, .15, -.1, .2, -.15, .1) if kind == "thermal" else (.2, -.25, .15, -.1)
    quantity = "pressure" if kind == "thermal" else "voltage"
    rows = []
    for index, row in enumerate(table.rows):
        original = row.values[quantity]
        value = replace(
            original, value=predictions[(row.row_id, quantity)]
            + pattern[index % len(pattern)] * original.uncertainty,
        )
        rows.append(replace(row, values={**row.values, quantity: value}))
    return replace(table, rows=tuple(rows)), experiments, truth, predictor.runs


def run_example(progress=None):
    cases = []
    for kind in ("thermal", "battery"):
        table, experiments, truth, generation = generate_data(kind, progress)
        result = NativeCalibration(
            table, parameters(kind), experiments, progress=progress,
        ).fit()
        cases.append({
            "name": kind, "measurements": table.to_dict(), "known_parameters": truth,
            "generation_runs": generation, "fit": result,
        })
        if kind == "thermal":
            confounded = NativeCalibration(
                table, parameters(kind, confounded=True), experiments, progress=progress,
            ).fit()
            cases.append({
                "name": "thermal_nonidentifiable", "measurement_dataset": table.dataset_id,
                "known_parameters": {**truth, "support_capacity": 20.0}, "fit": confounded,
            })
    failures = []
    for case in cases:
        name, result = case["name"], case["fit"]
        if result["status"] != "fitted":
            failures.append({"case": name, "reason": result["status"]})
            continue
        for split in ("training", "validation"):
            if result[split]["weighted_rmse"] is None or result[split]["weighted_rmse"] > 1:
                failures.append({"case": name, "reason": f"{split}_weighted_rmse"})
        if name == "thermal_nonidentifiable":
            if result["identifiability"]["rank"] >= 2:
                failures.append({"case": name, "reason": "confounding_not_detected"})
        else:
            for parameter, truth in case["known_parameters"].items():
                if abs(result["estimate"][parameter] / truth - 1) > .05:
                    failures.append({"case": name, "reason": f"{parameter}_recovery"})
    unique_rows = [
        row for case in cases if "measurements" in case for row in case["measurements"]["rows"]
    ]
    return {
        "mode": "development", "cases": cases,
        "summary": {
            "cases": len(cases), "unique_measurement_rows": len(unique_rows),
            "training_rows": sum(row["split"] == "train" for row in unique_rows),
            "validation_rows": sum(row["split"] == "validation" for row in unique_rows),
            "status": "failed" if failures else "passed", "failures": failures,
        },
        "boundary": (
            "Fixed synthetic native examples; no empirical material calibration or qualification"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    def progress(record):
        print(json.dumps(record), file=sys.stderr, flush=True)

    result = run_example(progress)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    if result["summary"]["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
