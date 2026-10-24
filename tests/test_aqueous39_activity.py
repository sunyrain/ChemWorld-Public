"""Independent analytic electrolyte limits and thermodynamic derivative identities."""

import json
from dataclasses import replace
from math import exp, log, sqrt

import pytest
from scipy.integrate import quad

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.electrolyte_activity import (
    ElectrolyteActivityModel,
    ElectrolyteParameter,
    pitzer_binary_from_cphi,
    pitzer_mixing_J,
)
from reference.aqueous39_fixtures import VALUES, activity, mixed_activity, parameter


@pytest.mark.parametrize("family", ("ideal", "DebyeHuckelLimiting", "DebyeHuckel", "Davies", "SIT"))
@pytest.mark.parametrize("m", (0.001, 0.1, 1.0))
def test_dilute_and_specific_interaction_analytic_coefficients(family, m):
    model = activity(family)
    result = model.evaluate(
        {"Na+": m, "Cl-": m}, temperature_K=298.15, pressure_Pa=1e5, solvent_molar_mass_kg_mol=0.018
    )
    expected = (
        0
        if family == "ideal"
        else -log(10) * 0.509 * sqrt(m)
        if family == "DebyeHuckelLimiting"
        else -log(10) * 0.509 * (sqrt(m) / (1 + sqrt(m)) - 0.3 * m)
        if family == "Davies"
        else -log(10) * 0.509 * sqrt(m) / (1 + 1.5 * sqrt(m))
    )
    if family == "SIT":
        expected += log(10) * 2 * m * VALUES["epsilon"]
    assert result.log_activity_coefficients == pytest.approx(
        {"Na+": expected, "Cl-": expected}, abs=2e-10
    )
    assert result.log_solvent_activity == pytest.approx(-0.018 * 2 * m * result.osmotic_coefficient)


def binary_pitzer_expected(m):
    root, aphi, b = sqrt(m), 0.392, 1.2
    gamma = -aphi * (root / (1 + b * root) + 2 / b * log(1 + b * root)) + 2 * m * VALUES["beta0"]
    phi = 1 - aphi * root / (1 + b * root) + m * VALUES["beta0"]
    for beta, alpha in ((VALUES["beta1"], 2.0), (VALUES["beta2"], 12.0)):
        x = alpha * root
        g = 2 * quad(lambda y, x=x: y * exp(-x * y), 0, 1, epsabs=1e-13)[0]
        gamma += m * beta * (g + exp(-x))
        phi += m * beta * exp(-x)
    x = VALUES["omega"] * root
    h = quad(lambda y: y**3 * exp(-x * y), 0, 1, epsabs=1e-13)[0]
    gamma += m * m * (3 * VALUES["C0"] + 4 * VALUES["C1"] * h + 2 * VALUES["C1"] * exp(-x))
    phi += 2 * m * m * (VALUES["C0"] + VALUES["C1"] * exp(-x))
    return gamma, phi


@pytest.mark.parametrize("m", (0.001, 0.1, 1.0))
def test_binary_pitzer_analytic_mean_activity_and_osmotic(m):
    model = activity()
    result = model.evaluate(
        {"Na+": m, "Cl-": m}, temperature_K=298.15, pressure_Pa=1e5, solvent_molar_mass_kg_mol=0.018
    )
    gamma, phi = binary_pitzer_expected(m)
    assert result.log_activity_coefficients == pytest.approx(
        {"Na+": gamma, "Cl-": gamma}, abs=2e-10
    )
    assert result.osmotic_coefficient == pytest.approx(phi, abs=2e-10)


def test_full_mixed_pitzer_gradients_and_reciprocity():
    model = mixed_activity()
    m = {"Na+": 0.2, "Ca++": 0.1, "Cl-": 0.2, "SO4--": 0.1, "CO2*": 0.05}
    result = model.evaluate(
        m, temperature_K=298.15, pressure_Pa=1e5, solvent_molar_mass_kg_mol=0.018
    )
    step = 1e-5
    for key in m:
        plus, minus = {**m, key: m[key] + step}, {**m, key: m[key] - step}
        fd = (
            model.excess_gibbs_RT_mol_kg(plus, 298.15, 1e5)
            - model.excess_gibbs_RT_mol_kg(minus, 298.15, 1e5)
        ) / (2 * step)
        assert result.log_activity_coefficients[key] == pytest.approx(fd, rel=2e-6, abs=2e-8)

    def energy(values):
        return model.excess_gibbs_RT_mol_kg(values, 298.15, 1e5)

    for left, right in (("Na+", "Ca++"), ("Ca++", "SO4--"), ("Cl-", "CO2*")):
        assert partial_derivative(energy, m, (left, right)) == pytest.approx(
            partial_derivative(energy, m, (right, left)), rel=2e-12
        )

    # Independently differentiate the extensive mixing energy w.r.t. actual water moles.
    def extensive(water_moles):
        water_kg = water_moles * 0.018
        molalities = {k: v / water_kg for k, v in m.items()}
        return water_kg * (
            sum(v * (log(v) - 1) for v in molalities.values())
            + model.excess_gibbs_RT_mol_kg(molalities, 298.15, 1e5)
        )

    nw = 1 / 0.018
    water_mu = (extensive(nw + step) - extensive(nw - step)) / (2 * step)
    assert water_mu == pytest.approx(result.log_solvent_activity, rel=2e-6, abs=2e-8)


def test_tp_parameters_codecs_missing_data_and_domains():
    p = replace(parameter(0.1, "kg/mol"), per_K=0.002, times_K=3, per_log_T=0.4, per_Pa=1e-9)
    for t in (280, 298.15, 330):
        for pressure in (1e5, 1e6):
            expected = (
                0.1
                + 0.002 * (t - 298.15)
                + 3 * (1 / t - 1 / 298.15)
                + 0.4 * log(t / 298.15)
                + 1e-9 * (pressure - 1e5)
            )
            assert p.evaluate(t, pressure) == pytest.approx(expected, abs=1e-14)
    assert ElectrolyteParameter(**json.loads(json.dumps(p.to_dict()))) == p
    model = mixed_activity()
    assert ElectrolyteActivityModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    missing = replace(model, interactions=model.interactions[:-1])
    assert missing.missing_parameters()
    with pytest.raises(ValueError, match="missing electrolyte"):
        missing.evaluate(
            dict.fromkeys(dict(model.charges), 0.1),
            temperature_K=298.15,
            pressure_Pa=1e5,
            solvent_molar_mass_kg_mol=0.018,
        )
    with pytest.raises(ValueError, match="domain"):
        model.evaluate(
            dict.fromkeys(dict(model.charges), 0.1),
            temperature_K=400,
            pressure_Pa=1e5,
            solvent_molar_mass_kg_mol=0.018,
        )
    with pytest.raises(ValueError, match="ionic strength"):
        model.evaluate(
            dict.fromkeys(dict(model.charges), 2.0),
            temperature_K=298.15,
            pressure_Pa=1e5,
            solvent_molar_mass_kg_mol=0.018,
        )
    with pytest.raises(ValueError, match="wrong unit"):
        replace(activity("Davies"), parameters={"A": parameter(0.5, "unknown")})


@pytest.mark.parametrize(
    "family", ("ideal", "DebyeHuckelLimiting", "DebyeHuckel", "Davies", "SIT", "Pitzer")
)
def test_zero_solute_and_finite_absent_ion_insertion(family):
    model = activity(family)
    result = model.evaluate(
        {"Na+": 0, "Cl-": 0}, temperature_K=298.15, pressure_Pa=1e5, solvent_molar_mass_kg_mol=0.018
    )
    assert result.solvent_activity == 1
    assert result.osmotic_coefficient == 1
    assert result.log_activity_coefficients == {"Na+": 0, "Cl-": 0}
    assert result.log_activities == {"Na+": None, "Cl-": None}
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))


def test_explicit_cphi_conversion_and_electrostatic_j():
    p = parameter(0.004, "kg^2/mol^2")
    assert pitzer_binary_from_cphi(cation_charge=2, anion_charge=-1, Cphi=p).value == pytest.approx(
        0.004 / (2 * sqrt(2))
    )
    for x in (0.01, 1.0, 10.0):
        expected = x / (4 + 4.581 * x ** (-0.7237) * exp(-0.012 * x**0.528))
        assert pitzer_mixing_J(x) == pytest.approx(expected, rel=1e-14)


def test_each_mixed_interaction_uses_its_declared_gibbs_monomial():
    model = mixed_activity()
    model = replace(
        model,
        parameters={**model.parameters, "Aphi": replace(model.parameters["Aphi"], value=0)},
        interactions=tuple(
            replace(
                row,
                parameters={
                    k: replace(p, value=0) if k in ("beta1", "beta2", "C1") else p
                    for k, p in row.parameters.items()
                },
            )
            for row in model.interactions
        ),
    )
    m = {"Na+": 0.2, "Ca++": 0.1, "Cl-": 0.2, "SO4--": 0.1, "CO2*": 0.05}
    c0, c1, a0, a1, n = 0.2, 0.1, 0.2, 0.1, 0.05
    z = c0 + 2 * c1 + a0 + 2 * a1
    expected = (
        (c0 + c1) * (a0 + a1) * (2 * VALUES["beta0"] + z * VALUES["C0"])
        + 2 * VALUES["theta"] * (c0 * c1 + a0 * a1)
        + VALUES["psi"] * (c0 * c1 * (a0 + a1) + a0 * a1 * (c0 + c1))
        + 2 * VALUES["lambda"] * (n * n + n * (c0 + c1 + a0 + a1))
        + VALUES["zeta"] * n * (c0 + c1) * (a0 + a1)
        + VALUES["mu"] * n**3
    )
    assert model.excess_gibbs_RT_mol_kg(m, 298.15, 1e5) == pytest.approx(expected, abs=1e-14)
