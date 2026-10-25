"""Synthetic controls of fitting semantics, separate from measured holdout evidence."""

import json
from dataclasses import replace
from math import sqrt

import numpy as np
import pytest

from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.molecular_interface import (
    MolecularInterfaceResult,
    PolynomialSquareGradientInfluence,
)
from chemworld.physchem.molecular_interface_fitting import (
    InfluenceFitParameter,
    MolecularInterfaceFitCase,
    SurfaceTensionObservation,
    fit_molecular_influence,
)
from reference.run_molecular_interface import QuarticModel


def pure_template(coefficients=(1e-19, 0.)):
    return PolynomialSquareGradientInfluence(("a",), (coefficients,), ((0.,),),
        (290., 340.), "synthetic control", 298.15, 100.)


def pure_prediction(observation, influence):
    c = influence.matrix_at(observation.temperature_K)[0][0]
    return MolecularInterfaceResult(True, .02*sqrt(c/1e-19))


def pure_data():
    true = pure_template((2e-19, .4e-19))
    rows = []
    for index, temperature in enumerate((293.15, 303.15, 313.15, 323.15, 333.15)):
        observation = SurfaceTensionObservation(str(index), temperature, .02, .00002,
            "training" if index % 2 == 0 else "holdout", "synthetic analytic sqrt control")
        rows.append(replace(observation, surface_tension_N_m=
                            pure_prediction(observation, true).surface_tension_N_m))
    return rows


def pure_parameters():
    return (InfluenceFitParameter("c0", "pure_coefficient", (0, 0), .1e-19, 5e-19, 1e-19),
            InfluenceFitParameter("c1", "pure_coefficient", (0, 1), -.5e-19, 1e-19, 1e-19))


def test_pure_coefficients_fit_and_holdout_never_changes_parameters():
    rows = pure_data()
    fitted = fit_molecular_influence(pure_template(), pure_parameters(), rows, pure_prediction)
    assert fitted.success, fitted.to_dict()
    assert [p["value"] for p in fitted.parameters] == pytest.approx((2e-19, .4e-19), rel=2e-6)
    altered = [replace(o, surface_tension_N_m=2*o.surface_tension_N_m)
               if o.partition == "holdout" else o for o in rows]
    other = fit_molecular_influence(pure_template(), pure_parameters(), altered, pure_prediction)
    assert other.parameters == fitted.parameters
    assert other.diagnostics["metrics"]["holdout"]["mean_absolute_relative_error"] > .49
    assert other.success  # Algorithm success is explicitly separate from predictive adequacy.
    assert fitted.diagnostics["jacobian_rank"] == 2
    assert fitted.diagnostics["local_physical_parameter_covariance"] is not None
    json.dumps(fitted.to_dict(), allow_nan=False)


def test_three_component_interactions_and_coefficient_fit():
    template = PolynomialSquareGradientInfluence(("a", "b", "c"), ((1e-19,),)*3,
        ((0., .5, .5), (.5, 0., .5), (.5, .5, 0.)), (290, 340), "synthetic 3-component control")
    truth = replace(template, pure_coefficients_J_m5_mol2=((2e-19,), (1e-19,), (1e-19,)),
                    interaction=((0., .3, .5), (.3, 0., .6), (.5, .6, 0.)))

    def predictor(observation, influence):
        matrix = np.array(influence.matrix_at(observation.temperature_K))/1e-19
        t = (observation.temperature_K-290)/50
        return MolecularInterfaceResult(True, .01*sqrt(
            matrix[0, 0]+(1+t)*matrix[0, 1]+(1+t*t)*matrix[1, 2]))

    rows = []
    for index, temperature in enumerate(np.linspace(291, 339, 9)):
        row = SurfaceTensionObservation(str(index), float(temperature), .02, 2e-5,
            "holdout" if index in (2, 5) else "training", "synthetic ternary sensitivity control")
        rows.append(replace(row, surface_tension_N_m=predictor(row, truth).surface_tension_N_m))
    parameters = (InfluenceFitParameter("c0", "pure_coefficient", (0, 0), .5e-19, 3e-19, 1e-19),
                  InfluenceFitParameter("k01", "interaction", (0, 1), .1, .8, 1),
                  InfluenceFitParameter("k12", "interaction", (1, 2), .1, .8, 1))
    fit = fit_molecular_influence(template, parameters, rows, predictor)
    assert fit.success, fit.to_dict()
    assert [p["value"] for p in fit.parameters] == pytest.approx((2e-19, .3, .6), rel=2e-4)
    assert fit.diagnostics["jacobian_rank"] == 3


def test_native_scalar_forward_adapter_and_influence_result_round_trip():
    model = QuarticModel()
    cases = {}
    rows = []
    for index, temperature in enumerate((295., 305., 315., 325.)):
        states = [molecular_eos_state(model, temperature, sum(c), (1.,))
                  for c in (model.left, model.right)]
        cases[str(index)] = MolecularInterfaceFitCase(model, *states,
                                                      interior_density_grid=(2000., 4000.))
        exact = sqrt(2e-8*2e-18)*4000**3/6
        rows.append(SurfaceTensionObservation(str(index), temperature, exact, 1e-5,
            "holdout" if index == 3 else "training", "synthetic exact quartic functional"))
    template = PolynomialSquareGradientInfluence(model.component_ids, ((1e-18,),), ((0.,),),
        (290., 340.), "synthetic quartic fixture")
    fit = fit_molecular_influence(template,
        (InfluenceFitParameter("c", "pure_coefficient", (0, 0), .2e-18, 3e-18, 1e-18),),
        rows, lambda o, i: cases[o.observation_id].evaluate(o, i))
    assert fit.success, fit.to_dict()
    assert fit.parameters[0]["value"] == pytest.approx(2e-18, rel=1e-6)
    assert PolynomialSquareGradientInfluence.from_dict(
        json.loads(json.dumps(fit.influence.to_dict()))) == fit.influence
    with pytest.raises(ValueError, match="temperatures"):
        cases["0"].evaluate(rows[1], template)


def test_unidentifiable_fit_is_not_accepted():
    result = fit_molecular_influence(pure_template(), pure_parameters(), pure_data(),
        lambda _o, _i: MolecularInterfaceResult(True, .02))
    assert not result.success
    assert result.diagnostics["jacobian_rank"] == 0
    assert result.diagnostics["local_physical_parameter_covariance"] is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_exactly_determined_fit_without_holdout_has_no_residual_variance_estimate():
    rows = [pure_data()[0], pure_data()[-1]]
    fit = fit_molecular_influence(pure_template(), pure_parameters(), rows, pure_prediction)
    assert fit.success, fit.to_dict()
    assert fit.diagnostics["training_degrees_of_freedom"] == 0
    assert fit.diagnostics["jacobian_rank"] == 2
    assert fit.diagnostics["local_physical_parameter_covariance"] is None
    assert fit.diagnostics["metrics"]["holdout"]["total"] == 0
    assert fit.diagnostics["metrics"]["holdout"]["mean_absolute_relative_error"] is None
    assert [p["value"] for p in fit.parameters] == pytest.approx((2e-19, .4e-19), rel=2e-6)


def test_training_only_fit_retains_positive_residual_degrees_of_freedom():
    rows = [o for o in pure_data() if o.partition == "training"]
    fit = fit_molecular_influence(pure_template(), pure_parameters(), rows, pure_prediction)
    assert fit.success, fit.to_dict()
    assert fit.diagnostics["training_degrees_of_freedom"] == 1
    assert fit.diagnostics["local_physical_parameter_covariance"] is not None


def test_failed_holdout_retained_with_exact_denominator():
    def predictor(observation, influence):
        if observation.observation_id == "1":
            return MolecularInterfaceResult(False, failure_reason="deliberate holdout failure")
        return pure_prediction(observation, influence)

    fit = fit_molecular_influence(pure_template(), pure_parameters(), pure_data(), predictor)
    assert not fit.success
    assert len(fit.observations) == 5
    assert fit.diagnostics["metrics"]["holdout"]["total"] == 2
    assert fit.diagnostics["metrics"]["holdout"]["failed"] == 1
    assert fit.diagnostics["metrics"]["holdout"]["metric_denominator"] == 1
    assert fit.diagnostics["failed_evaluations"][0]["observation_id"] == "1"


def test_active_bound_has_no_unconstrained_covariance():
    parameters = (replace(pure_parameters()[0], upper=1.5e-19), pure_parameters()[1])
    fit = fit_molecular_influence(pure_template(), parameters, pure_data(), pure_prediction)
    assert fit.success, fit.to_dict()
    assert "c0" in fit.diagnostics["active_bounds"]
    assert fit.diagnostics["local_physical_parameter_covariance"] is None


def test_domain_positivity_is_checked_between_observation_temperatures():
    with pytest.raises(ValueError, match="positive"):
        PolynomialSquareGradientInfluence(("a",), ((1e-19, -5e-19, 5e-19),),
            ((0.,),), (290, 340), "deliberately negative inside domain", 290, 50)


@pytest.mark.parametrize("change", [{"partition": "test"}, {"source": ""},
                                    {"standard_uncertainty_N_m": 0}, {"temperature_K": True}])
def test_invalid_measurement_declarations(change):
    with pytest.raises(ValueError):
        replace(pure_data()[0], **change)


def test_fit_rejects_duplicate_data_and_unsupported_parameter_indices():
    rows = pure_data()
    with pytest.raises(ValueError, match="unique"):
        fit_molecular_influence(pure_template(), pure_parameters(),
                                [*rows, rows[0]], pure_prediction)
    with pytest.raises(ValueError, match="outside"):
        fit_molecular_influence(pure_template(),
            (replace(pure_parameters()[0], indices=(1, 0)),), rows, pure_prediction)
