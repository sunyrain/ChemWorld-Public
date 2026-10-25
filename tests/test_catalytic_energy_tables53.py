"""Table semantics, native fitting/assembly and honest parameter uncertainty."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from chemworld.data.calibration_uncertainty import (
    KNOWN_INDEPENDENT_SIGMA,
    known_sigma_parameter_uncertainty,
)
from chemworld.data.catalysis_inputs import assemble_catalytic_mechanism
from chemworld.data.scientific_inputs import InputValue, ScientificTable
from chemworld.examples.catalytic_energy_tables import (
    energy_calibration,
    energy_table,
    fitted_catalytic_mechanism,
    mechanism_declaration,
)
from reference.catalytic_table53_cases import BOUNDARY_CASES, FIT_CASES, case, rebuild
from reference.catalytic_table53_reference import weighted_reference


@pytest.mark.parametrize("name", FIT_CASES)
def test_original_fitter_matches_independent_physical_wls_and_covariance(name):
    calibration, options = case(name)
    before = calibration.model.to_dict()
    result = calibration.fit(**options)
    reference = weighted_reference(calibration, options.get("fixed_parameters", {}))
    assert result["status"] == "fitted"
    assert result["estimate"] == pytest.approx(reference["estimate"], abs=2e-5)
    assert result["objective"] == pytest.approx(reference["objective"], abs=2e-8)
    uncertainty = result["parameter_uncertainty"]
    assert uncertainty["status"] == "available"
    assert np.asarray(uncertainty["covariance"]) == pytest.approx(reference["covariance"], rel=2e-7)
    assert calibration.model.to_dict() == before
    assert result["input_table"] == calibration.table.to_dict()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("name", BOUNDARY_CASES)
def test_boundary_cases_keep_real_outcomes_without_fake_parameter_precision(name):
    calibration, options = case(name)
    result = calibration.fit(**options)
    status = {"rank": "rank_deficient", "bound": "active_parameter_bound"}.get(
        name, "fit_not_converged"
    )
    assert result["parameter_uncertainty"]["status"] == status
    assert result["parameter_uncertainty"]["covariance"] is None
    assert result["runs"] and result["resources"]["forward_groups"] > 0
    if "budget" in name:
        assert result["failures"] and "fitted_model" not in result
        assert "validation" not in result


def test_table_json_csv_and_missing_observation_preserve_sources_and_units():
    table = energy_table(mixed_units=True)
    assert ScientificTable.from_json(json.dumps(table.to_dict())).to_dict() == table.to_dict()
    csv_text = table.to_csv()
    metadata = {key: value for key, value in table.to_dict().items() if key != "rows"}
    assert ScientificTable.from_csv(csv_text, metadata).to_dict() == table.to_dict()
    row = table.rows[0]
    missing = replace(
        row,
        values={
            **row.values,
            "energy": InputValue(None, "kJ/mol", None, "declared missing", "missing"),
        },
    )
    fit = rebuild(energy_calibration(), table=replace(table, rows=(missing, *table.rows[1:])))
    result = fit.fit(max_nfev=60)
    assert result["training"]["n_valid"] == 7
    assert result["excluded_observations"][0]["row_id"] == row.row_id


def test_holdout_perturbation_cannot_change_parameters_or_covariance():
    first = energy_calibration()
    table = replace(
        first.table,
        rows=tuple(
            replace(
                row, values={**row.values, "energy": replace(row.values["energy"], value=50000.0)}
            )
            if row.split == "validation"
            else row
            for row in first.table.rows
        ),
    )
    a, b = first.fit(max_nfev=60), rebuild(energy_calibration(), table=table).fit(max_nfev=60)
    assert a["estimate"] == b["estimate"]
    assert a["parameter_uncertainty"] == b["parameter_uncertainty"]
    assert b["validation"]["weighted_rmse"] > 100
    assert all(row["split"] == "train" for row in b["jacobian_runs"])


@pytest.mark.parametrize("mutation", ["reference", "uncertain_predictor", "energy_unit"])
def test_input_semantics_are_explicit(mutation):
    fit = energy_calibration()
    table = fit.table
    if mutation == "reference":
        table = replace(
            table, columns=(replace(table.columns[0], reference_state="other"), *table.columns[1:])
        )
    elif mutation == "uncertain_predictor":
        row = table.rows[0]
        table = replace(
            table,
            rows=(
                replace(
                    row,
                    values={
                        **row.values,
                        "binding": replace(row.values["binding"], uncertainty=1.0),
                    },
                ),
                *table.rows[1:],
            ),
        )
    else:
        # eV has particle-energy dimensions; no hidden Avogadro conversion.
        table = replace(
            table,
            columns=(replace(table.columns[0], unit="eV"), *table.columns[1:]),
            rows=tuple(
                replace(r, values={**r.values, "binding": replace(r.values["binding"], unit="eV")})
                for r in table.rows
            ),
        )
    with pytest.raises(ValueError):
        rebuild(fit, table=table)


def test_free_energy_semantics_are_not_interchangeable_with_electronic_scaling():
    from chemworld.data.catalysis_calibration import CatalyticEnergyCalibration
    from chemworld.examples.catalytic_energy_tables import FRAME

    fit = energy_calibration()
    with pytest.raises(ValueError, match="distinct semantics"):
        CatalyticEnergyCalibration(
            fit.table,
            fit.model,
            fit.parameters,
            fitted_source=fit.fitted_source,
            reference_frame=FRAME,
            energy_kind="activation_free_energy",
            noise_model=KNOWN_INDEPENDENT_SIGMA,
        )


def test_all_fixed_fit_does_not_claim_estimated_zero_uncertainty():
    fit = energy_calibration()
    result = fit.fit(fixed_parameters=fit.initial_values)
    assert result["parameter_uncertainty"]["status"] == "no_free_parameters"
    assert result["parameter_uncertainty"]["covariance"] is None
    with pytest.raises(ValueError, match="known sigma"):
        known_sigma_parameter_uncertainty(
            fit,
            result,
            noise_model="unknown",
            parameter_units={p.name: p.unit for p in fit.parameters},
        )


@pytest.mark.parametrize("topology", ["exchange", "dissociation"])
def test_actual_input_holes_generate_two_original_topologies(topology):
    spec, record = fitted_catalytic_mechanism(topology)
    assert len(spec.site_types) == 2
    assert len(spec.species_by_id["bound_A"].adsorbate.electronic_energy.axes) == 2
    assert spec.interaction_matrix_J_mol[0][1] != 0
    assert spec.reactions[1].products["bound_B"] == (1 if topology == "exchange" else 2)
    if topology == "dissociation":
        assert spec.species_by_id["bound_A"].adsorbate.number_of_atoms == 2
    assert spec.to_dict() == record["mechanism"]
    for result in record["fits"].values():
        assert result["parameter_uncertainty"]["status"] == "available"
    json.dumps(record, allow_nan=False)


def test_assembly_preserves_inputs_and_refuses_failed_or_misbound_fits():
    _, record = fitted_catalytic_mechanism()
    declaration = mechanism_declaration()
    before = deepcopy(declaration)
    arguments = {
        "species_fits": {"bound_A": "bound_A", "bound_B": "bound_B"},
        "reaction_fits": {"adsorb": "bep", "exchange": "bep"},
        "descriptors": {"binding": 1000.0, "strain": -500.0},
    }
    assemble_catalytic_mechanism(declaration, record["fits"], **arguments)
    assert declaration == before
    failed, options = case("optimizer_budget")
    bad_fits = {**record["fits"], "bound_A": failed.fit(**options)}
    with pytest.raises(ValueError, match="converged"):
        assemble_catalytic_mechanism(declaration, bad_fits, **arguments)
    with pytest.raises(ValueError, match="undeclared"):
        assemble_catalytic_mechanism(
            declaration,
            record["fits"],
            **{**arguments, "species_fits": {"unknown": "bound_A", "bound_B": "bound_B"}},
        )
    with pytest.raises(ValueError, match="exactly cover"):
        assemble_catalytic_mechanism(
            declaration,
            record["fits"],
            **{**arguments, "descriptors": {**arguments["descriptors"], "extra": 0.0}},
        )
