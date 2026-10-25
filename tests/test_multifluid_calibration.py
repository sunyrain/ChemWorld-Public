"""Shared regression contracts on explicit authored homogeneous models."""

import json
from dataclasses import replace

import pytest

from chemworld.data.multifluid_calibration import MultifluidCalibration, MultifluidFitParameter
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.coupled_system import CoupledStateView
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.molecular_mixture import MolecularIsochoricThermodynamics
from chemworld.physchem.multifluid import MultifluidModel
from test_multifluid import synthetic_model


def inputs(*, pure=False, units=False, rich=False, single_term=False):
    model = synthetic_model(2)
    if single_term:
        model = replace(
            model,
            binary_pairs=(
                replace(
                    model.binary_pairs[0], departure_terms=model.binary_pairs[0].departure_terms[:1]
                ),
            ),
        )
    fields = {"pressure_Pa": "Pa", "residual_enthalpy_J_mol": "J/mol"}
    if rich:
        fields.update(
            {
                "enthalpy_J_mol": "J/mol",
                "cp_J_mol_K": "J/(mol*K)",
                "mu:A": "J/mol",
                "lnphi:B": "dimensionless",
            }
        )
    columns = [
        QuantityColumn(
            "temperature",
            "degC" if units else "K",
            "condition",
            (-200, 2000),
            "absolute temperature",
        ),
        QuantityColumn(
            "molar_density", "mol/L" if units else "mol/m^3", "condition", (0, 1e6), "molar density"
        ),
    ]
    columns.extend(
        QuantityColumn("x:" + k, "dimensionless", "condition", (0, 1), "mole fraction")
        for k in model.component_ids
    )
    columns.extend(
        QuantityColumn(
            k,
            "bar" if units and k == "pressure_Pa" else unit,
            "observation",
            (-1e12, 1e12),
            "synthetic homogeneous state",
        )
        for k, unit in fields.items()
    )
    rows = []
    for i, (t, rho, x) in enumerate(
        (
            (300, 700, 0.2),
            (400, 1500, 0.4),
            (350, 1000, 0.7),
            (450, 1900, 0.8),
            (380, 1200, 0.3),
            (420, 1600, 0.6),
        )
    ):
        xs = (1.0, 0.0) if pure else (x, 1 - x)
        state = model.caloric_state(t, rho, xs) if rich else model.state(t, rho, xs)
        mu = model.residual_chemical_potentials(t, rho, xs) if rich else None
        values = {
            "temperature": InputValue(t, "K", 0, "exact condition"),
            "molar_density": InputValue(rho, "mol/m^3", 0, "exact condition"),
            **{
                "x:" + k: InputValue(v, "dimensionless", 0, "exact condition")
                for k, v in zip(model.component_ids, xs, strict=True)
            },
        }
        for k, unit in fields.items():
            value = (
                mu[0]
                if k == "mu:A"
                else (state.log_fugacity_coefficients[1] if k == "lnphi:B" else getattr(state, k))
            )
            values[k] = InputValue(
                value, unit, 0.01 if unit != "dimensionless" else 1e-5, "declared synthetic sigma"
            )
        rows.append(
            ScientificRow(
                str(i), "authored", f"g{i}", f"s{i}", "train" if i < 4 else "validation", values
            )
        )
    source = InputSource(
        DataSourceProvenance("authored", 0, "Authored synthetic model", version="1", license="MIT"),
        "synthetic",
    )
    table = ScientificTable(
        "multifluid-contract", "measurements", tuple(columns), (source,), tuple(rows)
    )
    return model, table


def parameter(**options):
    return MultifluidFitParameter("binary", ("A", "B"), "F", (0.2, 1.5), 1.1, **options)


def adapter(*, parameters=None, options=None, **kwargs):
    model, table = inputs(**kwargs)
    return MultifluidCalibration(
        table,
        model,
        parameters or (parameter(),),
        fitted_source="synthetic regression",
        **(options or {}),
    )


def test_fit_units_codec_source_immutability_and_original_energy_consumer():
    calibration = adapter(units=True, rich=True)
    before = calibration.model.to_dict()
    report = calibration.fit(max_nfev=30)
    assert report["status"] == "fitted"
    assert report["estimate"][parameter().parameter_id] == pytest.approx(0.7, abs=1e-8)
    assert report["validation"]["weighted_rmse"] < 1e-6
    assert report["validation"]["n_valid"] == 12
    assert report["identifiability"]["status"] == "full_rank_local"
    assert calibration.model.to_dict() == before
    fitted = MultifluidModel.from_dict(json.loads(json.dumps(report["fitted_model"])))
    assert "regression" in fitted.source and "fitted:" in fitted.binary_pairs[0].source
    assert fitted.fluids == calibration.model.fluids
    thermal = MolecularIsochoricThermodynamics(fitted.caloric_model(), 2.0)
    thermal = MolecularIsochoricThermodynamics.from_dict(json.loads(json.dumps(thermal.to_dict())))
    assert thermal.model.residual_model.to_dict() == fitted.to_dict()
    state = CoupledStateView({"A": 1.2, "B": 0.8}, 400.0, 2.0)
    assert thermal.internal_energy_j(state) == pytest.approx(
        2 * fitted.caloric_state(400.0, 1000.0, (0.6, 0.4)).internal_energy_J_mol, rel=1e-12
    )
    assert report["resources"]["state_attempts"] == len(report["runs"])
    assert report["resources"]["chemical_potential_attempts"] == len(report["runs"])
    assert report["resources"]["property_readouts"] == 6 * len(report["runs"])
    json.dumps(report, allow_nan=False)
    with pytest.raises(ValueError, match="fresh"):
        calibration.fit()


def test_holdout_response_changes_cannot_change_training_or_optimization_runs():
    model, table = inputs()
    changed = replace(
        table,
        rows=tuple(
            replace(
                row,
                values={**row.values, "pressure_Pa": replace(row.values["pressure_Pa"], value=1e8)},
            )
            if row.split == "validation"
            else row
            for row in table.rows
        ),
    )
    reports = [
        MultifluidCalibration(t, model, (parameter(),), fitted_source="fit").fit()
        for t in (table, changed)
    ]
    assert reports[0]["estimate"] == reports[1]["estimate"]
    assert reports[0]["training"] == reports[1]["training"]
    assert [r for r in reports[0]["runs"] if r["stage"] != "heldout"] == [
        r for r in reports[1]["runs"] if r["stage"] != "heldout"
    ]
    assert reports[1]["validation"]["weighted_rmse"] > 1e6


def test_confounded_product_is_reported_as_rank_deficient():
    parameters = (
        parameter(),
        MultifluidFitParameter("departure", ("A", "B"), "n", (-0.04, -0.005), -0.012, term_index=0),
    )
    report = adapter(parameters=parameters, single_term=True).fit()
    assert report["status"] == "fitted"
    assert report["training"]["weighted_rmse"] < 1e-6
    assert report["identifiability"]["rank"] == 1
    assert report["identifiability"]["status"] == "rank_deficient"


def test_pure_endpoint_carries_no_information_about_binary_f():
    report = adapter(pure=True).fit()
    assert report["status"] == "fitted"
    assert report["identifiability"]["rank"] == 0
    assert report["estimate"][parameter().parameter_id] == 1.1
    assert report["training"]["weighted_rmse"] == 0


def test_fixed_parameters_and_suppressed_holdout_use_shared_driver():
    p = parameter()
    report = adapter().fit(fixed_parameters={p.parameter_id: 0.7}, evaluate_validation=False)
    assert report["status"] == "fitted"
    assert report["identifiability"]["status"] == "no_free_parameters"
    assert "validation" not in report
    assert all(r["split"] == "train" for r in report["runs"])


def test_active_bound_is_retained_without_replacing_the_best_feasible_estimate():
    p = replace(parameter(), bounds=(0.9, 1.5))
    report = adapter(parameters=(p,)).fit()
    assert report["status"] == "fitted"
    assert report["estimate"][p.parameter_id] == pytest.approx(0.9, abs=1e-8)
    assert report["active_bounds"][p.parameter_id] == -1
    assert report["training"]["weighted_rmse"] > 1


@pytest.mark.parametrize("failure", ["budget", "temperature", "composition", "nonconvergence"])
def test_failures_have_readable_reports_and_actual_work(failure):
    model, table = inputs()
    kwargs = {}
    if failure == "budget":
        kwargs["max_forward_groups"] = 1
    elif failure in {"temperature", "composition"}:
        key, value = ("temperature", 50.0) if failure == "temperature" else ("x:A", 0.1)
        first = table.rows[0]
        table = replace(
            table,
            rows=(
                replace(
                    first, values={**first.values, key: replace(first.values[key], value=value)}
                ),
                *table.rows[1:],
            ),
        )
    report = MultifluidCalibration(table, model, (parameter(),), fitted_source="fit", **kwargs).fit(
        max_nfev=1 if failure == "nonconvergence" else 30
    )
    assert report["status"] == (
        "optimizer_not_converged" if failure == "nonconvergence" else "forward_failed"
    )
    assert report["failures"] and report["resources"]["forward_groups"] >= 1
    assert "fitted_model" not in report
    if failure in {"temperature", "composition"}:
        assert report["runs"][0]["failure"]
    json.dumps(report, allow_nan=False)


def test_missing_observation_is_excluded_and_never_fabricated():
    model, table = inputs()
    first = table.rows[0]
    table = replace(
        table,
        rows=(
            replace(
                first,
                values={
                    **first.values,
                    "pressure_Pa": InputValue(None, "Pa", None, "missing observation", "missing"),
                },
            ),
            *table.rows[1:],
        ),
    )
    report = MultifluidCalibration(table, model, (parameter(),), fitted_source="fit").fit()
    assert report["training"]["n_valid"] == 7
    assert report["training"]["n_excluded"] == 1
    assert len(report["excluded_observations"]) == 1
    assert not any(
        r["quantity_id"] == "pressure_Pa"
        for run in report["runs"]
        if run["row_id"] == "0"
        for r in run["readouts"]
    )


@pytest.mark.parametrize(
    "p",
    [
        MultifluidFitParameter("binary", ("B", "A"), "F", (0.1, 2.0), 0.8),
        MultifluidFitParameter("binary", ("A", "absent"), "F", (0.1, 2.0), 0.8),
        MultifluidFitParameter("departure", ("A", "B"), "n", (-1.0, 1.0), 0.1, term_index=9),
        MultifluidFitParameter("departure", ("A", "B"), "eta", (0.0, 1.0), 0.1, term_index=0),
        MultifluidFitParameter(
            "pure_residual", ("A",), "n", (-1.0, 1.0), 0.1, term_index=0, coefficient_index=5
        ),
    ],
)
def test_missing_or_wrongly_oriented_addresses_are_rejected(p):
    with pytest.raises(ValueError):
        adapter(parameters=(p,))


@pytest.mark.parametrize(
    "changes",
    [
        {"coefficient": "d"},
        {"term_index": 0},
        {"bounds": (1.0, 1.0)},
        {"initial": True},
        {"coordinate": "unknown"},
        {"component_ids": ("A", "A")},
    ],
)
def test_parameter_address_structure_is_explicit(changes):
    with pytest.raises(ValueError):
        replace(parameter(), **changes)


def test_pure_and_departure_candidates_change_only_selected_source_coefficients():
    params = (
        MultifluidFitParameter(
            "pure_residual", ("A",), "n", (-0.5, 0.5), -0.3, term_index=0, coefficient_index=0
        ),
        MultifluidFitParameter("departure", ("A", "B"), "t", (0.1, 2.0), 0.9, term_index=0),
        MultifluidFitParameter("departure", ("A", "B"), "eta", (0.1, 2.0), 0.9, term_index=1),
        MultifluidFitParameter("binary", ("A", "B"), "beta_T", (0.5, 2.0), 1.2, coordinate="log"),
    )
    fit = adapter(parameters=params)
    candidate = fit.candidate(fit.initial_values)
    assert candidate.fluids[0].residual_terms[0].coefficients["n"] == (-0.3, 0.03)
    assert candidate.fluids[1] == fit.model.fluids[1]
    assert candidate.fluids[0].ideal_terms == fit.model.fluids[0].ideal_terms
    assert candidate.binary_pairs[0].departure_terms[0].t == 0.9
    assert candidate.binary_pairs[0].departure_terms[1].eta == 0.9
    assert candidate.binary_pairs[0].beta_T == 1.2
    assert fit.model.binary_pairs[0].beta_T == 1.07
    for p in params:
        assert MultifluidFitParameter.from_dict(json.loads(json.dumps(p.to_dict()))) == p
    with pytest.raises(ValueError, match="whitelist"):
        fit.candidate({**fit.initial_values, "gas_constant": 1.0})


def test_observation_mappings_units_and_condition_completeness_are_strict():
    with pytest.raises(ValueError, match="observable"):
        adapter(
            options={
                "observations": {
                    "pressure_Pa": "not_a_state_field",
                    "residual_enthalpy_J_mol": "residual_enthalpy_J_mol",
                }
            }
        )
    with pytest.raises(ValueError):
        adapter(
            options={
                "observations": {
                    "pressure_Pa": "enthalpy_J_mol",
                    "residual_enthalpy_J_mol": "pressure_Pa",
                }
            }
        )
    with pytest.raises(ValueError, match="every table column"):
        adapter(options={"composition_columns": {"A": "x:A"}})


def test_declared_progress_counts_attempts_and_failures():
    progress = []
    fit = adapter(options={"progress": progress.append, "max_forward_groups": 2})
    report = fit.fit()
    assert report["status"] == "forward_failed"
    assert [p["completed_forward_groups"] for p in progress] == [1, 2]
