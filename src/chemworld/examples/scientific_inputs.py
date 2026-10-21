"""Import existing property cards, synthetic curves and finite native parameters.

Run with ``uv run --no-sync python -m chemworld.examples.scientific_inputs``.
The curves are authored analytic examples, not laboratory observations or fits.
"""

import json
from dataclasses import replace
from math import exp

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.curated_properties import curated_property_package
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentUncertainty
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.parameter_inputs import ModelParameterTarget, bind_parameter_row
from chemworld.world.surface_world import surface_world_spec


def curated_input(component_id="water"):
    package = curated_property_package(component_id)
    sources = tuple(
        InputSource(
            DataSourceProvenance(
                item.source_id, 0, f"{item.source_path}; table={item.source_table}",
                version="existing-ChemWorld-curated-subset",
                license="upstream table redistribution terms not established by this example",
            ),
            "literature",
        ) for item in package.component.provenance
    )
    return PropertyInputCard(
        package, sources, {
            correlation.correlation_id: ComponentUncertainty(
                field_id=correlation.correlation_id, unit=correlation.output_unit,
                coverage="not_quantified",
                source_id="chemicals:" + correlation.metadata["source_table"],
                note="Existing table regression does not establish measurement uncertainty.",
            )
            for correlation in package.correlations
        },
        CaloricReference(
            phase="gas", temperature_K=298.15, energy_basis="enthalpy", energy_J_mol=0,
            source_note="Authored sensible gas enthalpy zero; not formation enthalpy.",
        ),
        "Gas H(298.15 K)=0 is a convention; Cp and saturation coefficients are unchanged.",
    )


def synthetic_source():
    return InputSource(
        DataSourceProvenance(
            "chemworld:synthetic-input-example", 0,
            "Authored cooling-law and parameter examples in chemworld.examples.scientific_inputs",
            version="1", license="MIT (ChemWorld-authored synthetic examples)",
        ), "synthetic",
    )


def synthetic_curves():
    columns = (
        QuantityColumn("time", "s", "condition", (0, 120), "elapsed experiment time"),
        QuantityColumn("ambient_temperature", "K", "condition", (280, 340),
                       "absolute thermodynamic temperature"),
        QuantityColumn("pressure", "Pa", "observation", (1, 200000), "absolute pressure"),
    )
    rows = []
    # Fixed groups prevent adjacent points from one curve leaking into holdout.
    for split, ambient in (("train", 300.0), ("validation", 310.0)):
        for index, time in enumerate((0.0, 30.0, 60.0)):
            temperature = ambient + (350 - ambient) * exp(-0.1 * time / 20.0)
            pressure = 100000 * temperature / 350
            rows.append(ScientificRow(
                f"{split}-{index}", synthetic_source().provenance.source_id,
                f"cooling-{split}", f"acquisition-{split}-{index}", split, {
                    "time": InputValue(time / 60, "min", 0, "Exact authored time"),
                    "ambient_temperature": InputValue(
                        ambient - 273.15, "degC", 0.2, "Declared synthetic standard uncertainty",
                    ),
                    "pressure": InputValue(
                        pressure / 100000, "bar", 0.001,
                        "Declared 100 Pa standard uncertainty; no random noise added",
                    ),
                },
            ))
    return ScientificTable(
        "synthetic-cooling-curves-v1", "measurements", columns, (synthetic_source(),), tuple(rows),
    )


def parameter_table():
    column = QuantityColumn(
        "heat_loss", "W/K", "parameter", (0, 1),
        "Heat flow per kelvin difference from the declared environment",
    )
    rows = tuple(
        ScientificRow(
            name, synthetic_source().provenance.source_id, "parameter-candidates", name,
            "parameters", {"heat_loss": InputValue(
                value, "mW/K", 0, "Exact authored parameter, not a fitted estimate",
            )},
        ) for name, value in (("low-loss", 10.0), ("high-loss", 100.0))
    )
    return ScientificTable(
        "synthetic-thermal-parameters-v1", "parameters", (column,), (synthetic_source(),), rows,
    )


def parameter_response(table, row_id):
    spec = surface_world_spec(include_surface=False, include_bulk=False)
    target = ModelParameterTarget(
        table.columns[0], "thermal", spec.modules.model_specs()["thermal"].version,
        "reactor", ("loss_conductance_W_K",),
    )
    imported = bind_parameter_row(spec.physics, table, row_id, (target,))
    session = ChemWorldRuntime(composition=replace(spec, physics=imported.physics))
    try:
        session.reset(seed=12)
        history = []
        for action in (
            {"operation": "pause_control"},
            {"operation": "wait", "duration_s": 60, "stirring_speed_rpm": 600},
            {"operation": "measure", "instrument": "gas_pressure"},
        ):
            result = session.execute(action)
            history.append({
                "operation": action["operation"], "outcome": result.info["operation_outcome"],
                "transaction_status": result.info["transaction_status"],
                "actual_duration_s": result.info["actual_duration_s"],
            })
        return {
            "import": imported.receipt, "steps": history,
            "observation": session.observe().native_record(),
        }
    finally:
        session.close()


def run_example():
    card = PropertyInputCard.from_json(json.dumps(curated_input().to_dict()))
    properties = card.evaluate_state(
        temperature_K=400, pressure_Pa=1000, phase="gas",
        quantities=("cp", "enthalpy"), validity_policy="raise",
    )
    table = synthetic_curves()
    metadata = table.to_dict()
    metadata.pop("rows")
    imported = ScientificTable.from_csv(table.to_csv(), metadata)
    parameters = ScientificTable.from_json(json.dumps(parameter_table().to_dict()))
    return {
        "evidence_mode": "development",
        "property_source": card.to_dict()["sources"],
        "property_state": properties.to_dict(),
        "measurements": imported.to_dict(), "measurement_summary": imported.summary(),
        "parameter_responses": [
            parameter_response(parameters, row.row_id) for row in parameters.rows
        ],
        "boundary": (
            "Authored synthetic curves and native response; no fitting or real calibration."
        ),
    }


def main():
    print(json.dumps(run_example(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
