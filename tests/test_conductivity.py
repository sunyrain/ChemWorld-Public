"""Conductivity input contracts, shared arithmetic and real heat-transfer consumers."""

import json
from dataclasses import replace

import pytest

from chemworld.data.conductivity import CONDUCTIVITY_SOURCES, ConductivityDatabase
from chemworld.physchem.conductivity_kernels import conductivity_derivatives, conductivity_value
from chemworld.physchem.correlation_families import family_value_and_slope
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.transport import internal_heat_transfer_coefficient, jacket_heat_transfer
from chemworld.physchem.transport_properties import gas_thermal_conductivity_dippr9b_report


@pytest.fixture
def database(tmp_path):
    samples = {
        "PERRY_LIQUID_CONDUCTIVITY": "CAS\tChemical\tC1\tC2\tC3\tC4\tC5\tTmin\tTmax\n"
        "71-36-3\tsynthetic\t.3\t-.0003\t0\t0\t0\t250\t400\n"
        "74-82-8\tmissing\t\t-.0003\t0\t0\t0\t250\t400\n",
        "VDI_LIQUID_CONDUCTIVITY": "CAS\tChemical\tA\tB\tC\tD\tE\n"
        "71-36-3\tsynthetic\t.3\t-.0003\t0\t0\t0\n",
    }
    for source in CONDUCTIVITY_SOURCES:
        if source.source_id in samples:
            path = tmp_path / source.relative_path
            path.parent.mkdir(exist_ok=True, parents=True)
            path.write_text(samples[source.source_id], encoding="utf8")
    return ConductivityDatabase(tmp_path)


def test_source_json_property_package_and_actual_heat_transfer(database):
    assert database.loaded_sources == () and len(database.catalog) == 4
    component = ComponentSpec("butanol", "C4H10O", cas_number="71-36-3")
    card = database.property_input("71-36-3", component, method="PERRY_LIQUID_CONDUCTIVITY")
    restored = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
    assert card.to_dict() == restored.to_dict()
    results = []
    for t in (280, 350):
        state = restored.evaluate_state(
            temperature_K=t,
            pressure_Pa=101325,
            phase="liquid",
            quantities=("thermal_conductivity",),
        )
        k = state.properties["thermal_conductivity"].value
        h = internal_heat_transfer_coefficient(
            nusselt=3.66, thermal_conductivity_W_m_K=k, diameter_m=0.01
        )
        heat = jacket_heat_transfer(
            area_m2=0.1,
            overall_u_W_m2_K=h,
            reactor_temperature_K=t,
            jacket_temperature_K=t + 10,
            duration_s=5,
        )
        results.append((k, heat.heat_energy_J))
    assert results[0][1] > results[1][1]
    assert results[0][1] / results[1][1] == pytest.approx(results[0][0] / results[1][0])
    curve = card.package.correlations[0]
    assert curve.metadata["phase"] == "liquid"
    assert curve.metadata["source_record"]["fields"]["C2"] == "-.0003"
    assert correlation_temperature_derivative(curve, temperature_K=300).to(
        "mW/(m*K^2)"
    ).value == pytest.approx(-0.3)
    with pytest.raises(ValueError, match="identity"):
        database.property_input(
            "71-36-3", replace(component, cas_number="74-82-8"), method="PERRY_LIQUID_CONDUCTIVITY"
        )


def test_source_domain_missing_data_and_phase_are_not_filled(database):
    with pytest.raises(ValueError, match="explicit"):
        database.thermal_conductivity("71-36-3", method="VDI_LIQUID_CONDUCTIVITY")
    curve = database.thermal_conductivity(
        "71-36-3", method="VDI_LIQUID_CONDUCTIVITY", temperature_bounds_K=(250, 400)
    )
    assert curve.metadata["source_bounds_K"] is None
    assert curve.metadata["temperature_range_basis"] == "caller_application"
    with pytest.raises(ValueError, match="exceeds"):
        database.thermal_conductivity(
            "71-36-3", method="PERRY_LIQUID_CONDUCTIVITY", temperature_bounds_K=(200, 400)
        )
    with pytest.raises(ValueError, match="missing"):
        database.thermal_conductivity("74-82-8", method="PERRY_LIQUID_CONDUCTIVITY")
    with pytest.raises(ValueError):
        evaluate_correlation(curve, temperature_K=450, validity_policy="raise")


@pytest.mark.parametrize("molecule", ("monoatomic", "linear", "nonlinear"))
def test_original_dippr9b_report_uses_shared_kernel_and_molar_cv(molecule):
    p = {"MW": 28.01, "Cvm": 20.826, "mu": 1.277e-5}
    if molecule == "linear":
        p["Tc"] = 132.92
    report = gas_thermal_conductivity_dippr9b_report(
        temperature_K=200,
        molecular_weight_g_mol=p["MW"],
        molar_cv_J_mol_K=p["Cvm"],
        viscosity_Pa_s=p["mu"],
        critical_temperature_K=132.92,
        molecule_type=molecule,
    )
    assert report.evaluation.value == conductivity_value("DIPPR9B_" + molecule, 200, **p)
    assert 0.01 < report.evaluation.value < 0.04
    higher = conductivity_value("DIPPR9B_" + molecule, 200, **{**p, "mu": 2 * p["mu"]})
    assert higher == pytest.approx(2 * report.evaluation.value)


def test_common_chemsep_kernel_and_new_temperature_correlation():
    p = {"A": -0.12682, "B": -1.5015, "C": -1.0467, "D": -0.00088709, "E": -9.3679e-7}
    assert conductivity_derivatives("Chemsep_16", 300, p) == family_value_and_slope(
        "chemsep16", 300, p
    )
    curve = PropertyCorrelation(
        "synthetic",
        "conductivity_chemsep_16",
        p,
        {"temperature": "K"},
        "W/(m*K)",
        "thermal_conductivity",
        {"temperature": (250, 400)},
        "Synthetic conductivity example",
        {"phase": "liquid"},
    )
    assert evaluate_correlation(curve, temperature_K=300).value == pytest.approx(0.11924904787869)
    assert correlation_temperature_derivative(curve, temperature_K=300).value < 0


@pytest.mark.parametrize(
    "method,parameters",
    [
        ("PPDS8", {"Tc": 500, "a0": 0.1, "a1": 1, "a2": 1, "a3": 1}),
        ("Sato_Riedel", {"Tc": 500, "Tb": 300, "MW": 40}),
        ("kl_Mersmann_Kind", {"Tc": 500, "Vc": 0.0005, "MW": 100, "na": 20}),
    ],
)
def test_critical_value_remains_available_but_fractional_derivative_rejects(method, parameters):
    assert conductivity_value(method, 500, **parameters) > 0
    with pytest.raises(ValueError, match="critical derivative"):
        conductivity_derivatives(method, 500, parameters)
    with pytest.raises(ValueError, match="T <= Tc"):
        conductivity_value(method, 501, **parameters)


def test_invalid_negative_and_missing_parameters_do_not_produce_fake_conductivity():
    for method, t, p in (
        ("unknown", 300, {}),
        ("Eucken", 300, {"MW": 40, "Cvm": 20}),
        ("Eucken", 300, {"MW": 40, "Cvm": -20, "mu": 1e-5}),
        ("Lakshmi_Prasad", 1000, {"MW": 100}),
        ("Sheffy_Johnson", 0, {"MW": 40, "Tm": 250}),
        ("Chemsep_16", 300, {"A": -10, "B": 0, "C": 0, "D": 0, "E": 0}),
    ):
        with pytest.raises(ValueError):
            conductivity_value(method, t, **p)
