"""Input contracts preserve units/evidence and only bind explicitly allowed model parameters."""

import csv
import io
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.data.scientific_inputs import (
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.examples.scientific_inputs import (
    curated_input,
    parameter_response,
    parameter_table,
    run_example,
    synthetic_curves,
    synthetic_source,
)
from chemworld.foundation.dimensions import dimension_definition
from chemworld.foundation.units import convert_value
from chemworld.physchem.curated_properties import curated_property_cases
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.world.parameter_inputs import ModelParameterTarget, bind_parameter_row
from chemworld.world.surface_world import surface_world_spec


def metadata(table):
    payload = table.to_dict()
    payload.pop("rows")
    return payload


@pytest.mark.parametrize("case", curated_property_cases(), ids=lambda case: case.component_id)
def test_existing_curated_card_import_uses_original_package_equations_and_declared_datum(case):
    original = curated_input(case.component_id)
    card = PropertyInputCard.from_json(json.dumps(original.to_dict()))
    assert card.to_dict() == original.to_dict()
    request = {
        "temperature_K": 400, "pressure_Pa": 1000, "phase": "gas", "quantities": ("cp", "enthalpy"),
    }
    assert card.evaluate_state(**request) == original.package.evaluate_state(
        **request, reference=original.reference,
    )
    assert all(s.data_kind == "literature" for s in card.sources)
    assert all(u.standard_uncertainty is None for u in card.uncertainties.values())
    shifted = replace(card, reference=replace(card.reference, energy_J_mol=1000))
    before, after = card.evaluate_state(**request), shifted.evaluate_state(**request)
    change = after.properties["enthalpy"].value - before.properties["enthalpy"].value
    assert change == pytest.approx(1000)
    assert after.properties["cp"] == before.properties["cp"]
    with pytest.raises(ValueError):
        card.evaluate_state(**{**request, "temperature_K": 10000})


@pytest.mark.parametrize("defect", [
    "source", "license", "domain", "finite_domain", "coefficient_boolean", "coefficient_nan",
    "duplicate_correlation", "unknown_equation", "uncertainty", "reference_missing",
    "unknown_field",
])
def test_property_card_rejects_incomplete_or_incompatible_input(defect):
    payload = curated_input().to_dict()
    first = payload["package"]["correlations"][0]
    if defect == "source":
        payload["sources"] = []
    elif defect == "license":
        payload["sources"][0]["provenance"]["license"] = ""
    elif defect == "domain":
        first["validity_ranges"] = {}
    elif defect == "finite_domain":
        first["validity_ranges"]["temperature"] = [200, float("inf")]
    elif defect.startswith("coefficient"):
        first["coefficients"]["A"] = True if defect.endswith("boolean") else float("nan")
    elif defect == "duplicate_correlation":
        payload["package"]["correlations"].append(deepcopy(first))
    elif defect == "unknown_equation":
        first["equation_id"] = "__import__('os').system"
    elif defect == "uncertainty":
        payload["uncertainties"][first["correlation_id"]]["source_id"] = "unknown"
    elif defect == "reference_missing":
        del payload["reference"]["energy_basis"]
    else:
        first["ignored_coefficients"] = {"a": 2}
    with pytest.raises(ValueError):
        PropertyInputCard.from_dict(payload)


def test_measurement_table_csv_and_json_convert_absolute_values_and_uncertainty_differences():
    table = synthetic_curves()
    payload = table.to_dict()
    # The original input is degC/min/bar; normalized cells use K/s/Pa.
    row = table.rows[1]
    assert row.values["ambient_temperature"].value == 300
    assert row.values["ambient_temperature"].uncertainty == pytest.approx(.2)
    assert row.values["time"].value == 30
    assert row.values["pressure"].uncertainty == pytest.approx(100)
    assert len(table.rows_for("train")) == len(table.rows_for("validation")) == 3
    assert ScientificTable.from_json(json.dumps(payload)).to_dict() == payload
    assert ScientificTable.from_csv(table.to_csv(), metadata(table)).to_dict() == payload
    assert table.summary()["data_kinds"] == {"synthetic": 6}
    with pytest.raises(TypeError):
        row.values["time"] = InputValue(2, "s", 0, "Exact")
    # Mutating the input dictionary cannot rewrite the imported table.
    clone = ScientificTable.from_dict(payload)
    payload["rows"][0]["values"]["pressure"]["value"] = 1
    assert clone.rows[0].values["pressure"].value == 100000


@pytest.mark.parametrize("defect", [
    "missing_column", "unknown_column", "wrong_unit", "missing_source", "license",
    "nan", "boolean", "out_of_domain", "unknown_uncertainty", "negative_uncertainty",
    "duplicate_row", "duplicate_sample", "cross_split_group", "bad_split", "source_kind",
    "duplicate_column", "unknown_version",
])
def test_table_rejects_bad_values_provenance_identity_and_split_leakage(defect):
    payload = synthetic_curves().to_dict()
    row = payload["rows"][0]
    cell = row["values"]["pressure"]
    if defect == "missing_column":
        del row["values"]["pressure"]
    elif defect == "unknown_column":
        row["values"]["hidden_truth"] = deepcopy(cell)
    elif defect == "wrong_unit":
        cell["unit"] = "mol"
    elif defect == "missing_source":
        row["source_id"] = "undefined"
    elif defect == "license":
        del payload["sources"][0]["provenance"]["license"]
    elif defect in {"nan", "boolean", "out_of_domain"}:
        cell["value"] = {"nan": float("nan"), "boolean": True, "out_of_domain": 300000}[defect]
    elif defect == "unknown_uncertainty":
        cell.update(uncertainty=None, uncertainty_note="")
    elif defect == "negative_uncertainty":
        cell["uncertainty"] = -1
    elif defect == "duplicate_row":
        payload["rows"][1]["row_id"] = row["row_id"]
    elif defect == "duplicate_sample":
        payload["rows"][1]["sample_id"] = row["sample_id"]
    elif defect == "cross_split_group":
        payload["rows"][1]["split"] = "validation"
    elif defect == "bad_split":
        row["split"] = "automatic_random"
    elif defect == "source_kind":
        payload["sources"][0]["data_kind"] = "verified_real_data"
    elif defect == "duplicate_column":
        payload["columns"].append(deepcopy(payload["columns"][0]))
    else:
        payload["schema_version"] = "unknown"
    with pytest.raises(ValueError):
        ScientificTable.from_dict(payload)


def test_missing_and_invalid_readings_are_retained_and_not_made_training_targets():
    payload = synthetic_curves().to_dict()
    payload["rows"][0]["values"]["pressure"].update(
        value=None, quality="missing", uncertainty=None, uncertainty_note="Not observed",
    )
    payload["rows"][1]["values"]["pressure"].update(
        value=300000, quality="out_of_domain",
    )
    payload["rows"][2]["values"]["pressure"].update(
        uncertainty=None, uncertainty_note="Uncertainty not supplied by the source",
    )
    table = ScientificTable.from_dict(payload)
    assert table.summary()["rows"] == 6
    assert table.summary()["observation_quality"] == {
        "missing": 1, "out_of_domain": 1, "valid": 4,
    }
    assert len(table.rows_for("train")) == 3
    assert table.rows[0].values["pressure"].value is None
    assert table.rows[2].values["pressure"].uncertainty is None
    assert ScientificTable.from_csv(table.to_csv(), metadata(table)).to_dict() == table.to_dict()


@pytest.mark.parametrize("text", [
    '{"schema_version": "a", "schema_version": "b"}',
    '{"nested": {"value": 1, "value": 2}}',
    '{"value": NaN}',
])
def test_duplicate_json_fields_and_nonstandard_numbers_are_not_silently_accepted(text):
    with pytest.raises(ValueError):
        ScientificTable.from_json(text)


@pytest.mark.parametrize(
    "defect", ["extra_header", "duplicate_header", "extra_cell", "missing_cell"],
)
def test_csv_requires_the_declared_shape(defect):
    table = synthetic_curves()
    rows = list(csv.reader(io.StringIO(table.to_csv())))
    if defect == "extra_header":
        rows[0].append("undeclared")
    elif defect == "duplicate_header":
        rows[0][-1] = rows[0][0]
    elif defect == "extra_cell":
        rows[1].append("extra")
    else:
        rows[1].pop()
    stream = io.StringIO()
    csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError):
        ScientificTable.from_csv(stream.getvalue(), metadata(table))


def test_csv_quotes_uncertainty_notes_without_losing_records():
    payload = synthetic_curves().to_dict()
    payload["rows"][0]["values"]["pressure"]["uncertainty_note"] = (
        'Declared, "synthetic"\nnot fitted'
    )
    table = ScientificTable.from_dict(payload)
    assert ScientificTable.from_csv(table.to_csv(), metadata(table)).to_dict() == table.to_dict()


def binding(physics, column=None, **overrides):
    values = {
        "column": column or parameter_table().columns[0], "model_id": "thermal",
        "model_version": physics.modules.model_specs()["thermal"].version, "binding": "reactor",
        "field_path": ("loss_conductance_W_K",),
    }
    return ModelParameterTarget(**{**values, **overrides})


def test_model_parameter_import_changes_only_the_declared_configuration_and_real_response():
    physics = surface_world_spec(include_surface=False).physics
    before = physics.to_dict()
    table = ScientificTable.from_json(json.dumps(parameter_table().to_dict()))
    imported = bind_parameter_row(physics, table, "high-loss", (binding(physics),))
    assert physics.to_dict() == before
    assert imported.physics.initial_states == physics.initial_states
    assert imported.physics.models_of("thermal")["reactor"].loss_conductance_W_K == .1
    assert imported.physics.models_of("thermal")["receiver"] == (
        physics.models_of("thermal")["receiver"]
    )
    assert imported.receipt["source"]["data_kind"] == "synthetic"
    assert imported.receipt["parameters"][0]["unit"] == "W/K"
    low, high = (parameter_response(table, name) for name in ("low-loss", "high-loss"))
    for response in (low, high):
        assert all(step["transaction_status"] == "committed" for step in response["steps"])
        assert response["observation"]["channels"][0]["unit"] == "Pa"
    assert high["observation"]["channels"][0]["value"] < low["observation"]["channels"][0]["value"]


@pytest.mark.parametrize("defect", [
    "unknown_model", "version", "binding", "missing_path", "initial_state", "string_leaf",
    "reference", "unit", "domain", "extra_target", "measurement_table",
])
def test_model_binding_rejects_unauthorized_or_incompatible_changes_without_mutation(defect):
    physics = surface_world_spec(include_surface=False).physics
    before = physics.to_dict()
    table = parameter_table()
    target = binding(physics)
    if defect == "unknown_model":
        target = replace(target, model_id="not_installed")
    elif defect == "version":
        target = replace(target, model_version="unknown")
    elif defect == "binding":
        target = replace(target, binding="undeclared")
    elif defect == "missing_path":
        target = replace(target, field_path=("not_declared",))
    elif defect == "initial_state":
        target = replace(target, field_path=("initial_states", "reactor", "temperature_K"))
    elif defect == "string_leaf":
        target = replace(target, field_path=("pressure_law",))
    elif defect == "reference":
        target = replace(target, column=replace(target.column, reference_state="different datum"))
    elif defect == "unit":
        target = replace(target, column=replace(target.column, unit="Pa"))
    elif defect == "domain":
        target = replace(target, column=replace(target.column, valid_range=(0, .05)))
    elif defect == "measurement_table":
        table = synthetic_curves()
    targets = (target, target) if defect == "extra_target" else (target,)
    with pytest.raises(ValueError):
        bind_parameter_row(physics, table, "high-loss", targets)
    assert physics.to_dict() == before


def test_parameter_import_runs_model_validation_after_all_coupled_assignments():
    physics = surface_world_spec(include_surface=False).physics
    columns = tuple(
        QuantityColumn(name, "K", "parameter", (200, 600), "absolute temperature")
        for name in ("maximum", "stop")
    )
    def table_for(maximum, stop):
        return ScientificTable("limits", "parameters", columns, (synthetic_source(),), (
            ScientificRow("limits", synthetic_source().provenance.source_id, "limits", "limits",
                          "parameters", {
                              col.quantity_id: InputValue(value, "K", 0, "Authored limit")
                              for col, value in zip(columns, (maximum, stop), strict=True)
                          }),
        ))
    targets = tuple(binding(physics, column=col, field_path=(f"{col.quantity_id}_temperature_K",))
                    for col in columns)
    # Updating maximum alone to 435 would conflict with the old stop=440.
    imported = bind_parameter_row(physics, table_for(435, 430), "limits", targets)
    assert imported.physics.models_of("thermal")["reactor"].maximum_temperature_K == 435
    with pytest.raises(ValueError, match="thermal limits"):
        bind_parameter_row(physics, table_for(435, 450), "limits", targets)
    assert physics.models_of("thermal")["reactor"].maximum_temperature_K == 450


def test_thermal_parameter_units_have_distinct_registered_dimensions():
    assert convert_value(100, "mW/K", "W/K") == .1
    assert convert_value(.02, "kJ/K", "J/K") == 20
    assert dimension_definition("heat_capacity").vector.time == -2
    assert dimension_definition("thermal_conductance").vector.time == -3
    assert dimension_definition("volume_flow").vector.length == 3
    assert dimension_definition("velocity").vector.time == -1
    assert dimension_definition("acidity").canonical_unit == "pH"
    with pytest.raises(ValueError):
        convert_value(7, "pH", "dimensionless")
    with pytest.raises(ValueError):
        convert_value(1, "W/K", "W/(m^2*K)")


def test_unknown_split_and_unrepresentable_numeric_input_are_rejected():
    with pytest.raises(ValueError, match="unknown data split"):
        synthetic_curves().rows_for("training")
    with pytest.raises(ValueError, match="numeric input domain"):
        InputValue(10**500, "Pa", 0, "Invalid oversized number")
    with pytest.raises(ValueError, match="represented exactly"):
        InputValue(2**53 + 1, "counts", 0, "Integer beyond exact binary64 representation")


def test_missing_caloric_datum_does_not_acquire_an_implicit_energy_zero():
    card = replace(curated_input(), reference=None, reference_note="No energy datum supplied")
    request = {"temperature_K": 400, "pressure_Pa": 1000, "phase": "gas"}
    assert card.evaluate_state(**request, quantities=("cp",)).properties["cp"].value > 0
    with pytest.raises(ValueError, match="explicit caloric reference"):
        card.evaluate_state(**request, quantities=("enthalpy",))


@pytest.mark.parametrize("field", ["columns", "sources", "rows"])
def test_table_rejects_nonlist_collections_with_input_diagnostics(field):
    payload = synthetic_curves().to_dict()
    payload[field] = None
    with pytest.raises(ValueError, match="must be a list"):
        ScientificTable.from_dict(payload)


def test_fixed_import_example_preserves_denominators_and_public_responses():
    result = run_example()
    assert result["measurement_summary"]["rows"] == 6
    assert result["measurement_summary"]["groups"] == 2
    assert result["measurement_summary"]["splits"] == {"train": 3, "validation": 3}
    assert len(result["parameter_responses"]) == 2
    assert all(len(response["steps"]) == 3 for response in result["parameter_responses"])
