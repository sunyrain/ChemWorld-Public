"""Corresponding-state conductivity branches and actual dense-gas volume."""

import json

import pytest

from chemworld.physchem import (
    ComponentPropertyPackage,
    ComponentSpec,
    PropertyCorrelation,
    chung_dense_conductivity,
    conductivity_derivatives,
    conductivity_value,
    correlation_temperature_derivative,
    ely_hanley_dense_conductivity,
    evaluate_correlation,
    stiel_thodos_dense_conductivity,
)
from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.transport import internal_heat_transfer_coefficient, jacket_heat_transfer

ELY = {"MW": 72.151, "Tc": 460.4, "Vc": 3.06e-4, "Zc": 0.267, "omega": 0.227, "Cvm": 135.9}
CHUNG = {
    "MW": 42.081,
    "Tc": 364.9,
    "Vc": 184.6e-6,
    "omega": 0.142,
    "Cvm": 82.67,
    "Vm": 172.1e-6,
    "mu": 1.34e-5,
    "dipole": 0.4,
}
ELY_DENSE = {
    "MW": 42.081,
    "Tc": 364.9,
    "Vc": 1.81e-4,
    "Zc": 0.274,
    "omega": 0.144,
    "Cvm": 82.70,
    "Vm": 1.721e-4,
}
STIEL = {
    "MW": 44.013,
    "Tc": 309.6,
    "Pc": 72.4e5,
    "Vc": 97.4e-6,
    "Zc": 0.274,
    "Vm": 144e-6,
    "kg": 0.0234,
}
GHARAGHEIZI = {"MW": 16.04246, "Tb": 111.66, "Pc": 4599000, "omega": 0.0115478}


def test_ely_methane_units_example_and_shape_cutoff_keep_actual_temperature():
    assert conductivity_value("Eli_Hanley", 373.15, **ELY) == pytest.approx(0.02247951724513664)
    assert conductivity_value("Eli_Hanley", 920.8, **ELY) > 0
    with pytest.raises(ValueError, match="Tr=2"):
        conductivity_derivatives("Eli_Hanley", 920.8, ELY)
    assert conductivity_value("Eli_Hanley", 1151, **ELY) != pytest.approx(
        conductivity_value("Eli_Hanley", 1381.2, **ELY)
    )
    with pytest.raises(ValueError, match="shape"):
        conductivity_value("Eli_Hanley", 1, **ELY)


def test_chung_dense_units_and_actual_volume_reach_heat_consumer():
    assert chung_dense_conductivity(473, **CHUNG) == pytest.approx(0.06160569232570781)
    heat_coefficients = []
    for vm in (172.1e-6, 0.01):
        p = {**CHUNG, "Vm": vm}
        curve = PropertyCorrelation(
            "dense-gas",
            "conductivity_chung_dense",
            p,
            {"temperature": "K"},
            "W/(m*K)",
            "thermal_conductivity",
            {"temperature": (300, 700)},
            "Supplied actual volume, no EOS inference",
            {"phase": "gas"},
        )
        restored = PropertyCorrelation.from_dict(json.loads(json.dumps(curve.to_dict())))
        value = evaluate_correlation(restored, temperature_K=473).value
        assert value == chung_dense_conductivity(473, **p)
        heat_coefficients.append(
            internal_heat_transfer_coefficient(
                nusselt=3.66, thermal_conductivity_W_m_K=value, diameter_m=0.01
            )
        )
    assert heat_coefficients[0] > heat_coefficients[1]
    assert chung_dense_conductivity(473, **{**CHUNG, "association": 0.1}) != pytest.approx(
        chung_dense_conductivity(473, **CHUNG)
    )


@pytest.mark.parametrize(
    "change",
    (
        {"Vm": 0},
        {"Vm": 184.6e-6 / 6},
        {"dipole": -1},
        {"association": -1},
        {"omega": float("nan")},
        {"Cvm": 0},
        {"mu": 0},
        {"MW": True},
    ),
)
def test_dense_invalid_state_and_packing_reject(change):
    with pytest.raises(ValueError):
        chung_dense_conductivity(473, **{**CHUNG, **change})


def test_dense_no_ignored_pressure_or_invented_molar_volume():
    with pytest.raises(ValueError, match="requires"):
        conductivity_value("Chung_dense", 473, **{k: v for k, v in CHUNG.items() if k != "Vm"})
    with pytest.raises(ValueError, match="requires"):
        conductivity_value("Chung_dense", 473, **CHUNG, P=1e7)


def test_completion_examples_and_explicit_ely_correction_scope():
    assert conductivity_value("Gharagheizi_gas", 580, **GHARAGHEIZI) == pytest.approx(
        0.09594861261873211, rel=1e-11
    )
    assert stiel_thodos_dense_conductivity(378.15, **STIEL) == pytest.approx(
        0.041245574404863684, rel=1e-11
    )
    compatible = conductivity_value("Eli_Hanley_dense", 473, **ELY_DENSE)
    assert compatible == pytest.approx(0.06038475754109959, rel=1e-11)
    assert compatible == ely_hanley_dense_conductivity(473, **ELY_DENSE, correction_scope="dilute")
    translational = conductivity_value("Ely_Hanley_dense_translational", 473, **ELY_DENSE)
    assert translational == ely_hanley_dense_conductivity(473, **ELY_DENSE)
    assert translational > compatible
    with pytest.raises(ValueError, match="correction_scope"):
        ely_hanley_dense_conductivity(473, **ELY_DENSE, correction_scope="implicit")


@pytest.mark.parametrize(
    "method,p",
    [
        ("Gharagheizi_gas", GHARAGHEIZI),
        ("Stiel_Thodos_dense", STIEL),
        ("Eli_Hanley_dense", ELY_DENSE),
        ("Ely_Hanley_dense_translational", ELY_DENSE),
    ],
)
def test_completion_json_package_and_real_jacket_heat(method, p):
    heats = []
    for factor in (1, 1.2):
        # Actual density changes dense models; T changes the dilute model.
        parameters = {**p, "Vm": p["Vm"] * factor} if "Vm" in p else p
        t = 473 if "Vm" in p else 473 * factor
        curve = PropertyCorrelation(
            method,
            "conductivity_" + method.lower(),
            parameters,
            {"temperature": "K"},
            "W/(m*K)",
            "thermal_conductivity",
            {"temperature": (300, 800)},
            "Fixed equation qualification, not measured material data",
            {"phase": "gas"},
        )
        curve = PropertyCorrelation.from_dict(json.loads(json.dumps(curve.to_dict())))
        formula = (
            "CH4"
            if method == "Gharagheizi_gas"
            else "N2O"
            if method == "Stiel_Thodos_dense"
            else "C3H6"
        )
        package = ComponentPropertyPackage(
            ComponentSpec("gas", formula, default_phase="gas"), (curve,)
        )
        state = package.evaluate_state(
            temperature_K=t, pressure_Pa=1e7, phase="gas", quantities=("thermal_conductivity",)
        )
        k = state.properties["thermal_conductivity"].value
        assert k == conductivity_value(method, t, **parameters)
        slope = correlation_temperature_derivative(curve, temperature_K=t)
        assert slope.to("mW/(m*K^2)").value == pytest.approx(
            1000 * conductivity_derivatives(method, t, parameters)[1]
        )
        h = internal_heat_transfer_coefficient(
            nusselt=3.66, thermal_conductivity_W_m_K=k, diameter_m=0.01
        )
        heats.append(
            jacket_heat_transfer(
                area_m2=0.1,
                overall_u_W_m2_K=h,
                reactor_temperature_K=t,
                jacket_temperature_K=t + 10,
                duration_s=5,
            ).heat_energy_J
        )
    assert heats[0] != pytest.approx(heats[1])


def test_stiel_density_branches_keep_fixed_input_partials_and_reject_outside():
    for rho in (0.5, 2):
        p = {**STIEL, "Vm": STIEL["Vc"] / rho}
        assert conductivity_derivatives("Stiel_Thodos_dense", 378.15, p)[1] == 0
        with pytest.raises(ValueError, match="boundary"):
            derivative_values(
                lambda vm, p=p: stiel_thodos_dense_conductivity(378.15, **{**p, "Vm": vm}), p["Vm"]
            )
    for rho in (2.8, 3):
        with pytest.raises(ValueError, match="density"):
            stiel_thodos_dense_conductivity(378.15, **{**STIEL, "Vm": STIEL["Vc"] / rho})
    assert (
        derivative_values(
            lambda vm: stiel_thodos_dense_conductivity(378.15, **{**STIEL, "Vm": vm}), STIEL["Vm"]
        )[1]
        < 0
    )


def test_ely_shape_cutoffs_and_internal_term_are_explicit():
    for method in ("Eli_Hanley_dense", "Ely_Hanley_dense_translational"):
        with pytest.raises(ValueError, match="Tr=2"):
            conductivity_derivatives(method, 2 * ELY_DENSE["Tc"], ELY_DENSE)
        for t in (1.8 * ELY_DENSE["Tc"], 2.4 * ELY_DENSE["Tc"]):
            values = []
            for vr in (0.9, 2, 10):
                p = {**ELY_DENSE, "Vm": vr * ELY_DENSE["Vc"]}
                values.append(conductivity_value(method, t, **p))
            assert len(set(values)) == 3  # Vm isn't clipped when the shape is.
            assert conductivity_derivatives(method, t, ELY_DENSE)[1] != 0
        increments = []
        for vm in (1e-4, 1e-3):
            p = {**ELY_DENSE, "Vm": vm}
            increments.append(
                conductivity_value(method, 473, **{**p, "Cvm": 100})
                - conductivity_value(method, 473, **p)
            )
        assert increments[0] == pytest.approx(increments[1])
    with pytest.raises(ValueError, match="Vr=2"):
        derivative_values(
            lambda vm: ely_hanley_dense_conductivity(473, **{**ELY_DENSE, "Vm": vm}),
            2 * ELY_DENSE["Vc"],
        )


@pytest.mark.parametrize(
    "method,p",
    [
        ("Gharagheizi_gas", GHARAGHEIZI),
        ("Stiel_Thodos_dense", STIEL),
        ("Eli_Hanley_dense", ELY_DENSE),
    ],
)
def test_completion_missing_nonfinite_and_irrelevant_inputs_reject(method, p):
    for change in ({"MW": True}, {"MW": 0}, {"Tc": float("nan")}, {"P": 1e7}):
        with pytest.raises(ValueError):
            conductivity_value(method, 473, **{**p, **change})
    with pytest.raises(ValueError, match="requires"):
        conductivity_value(method, 473, **{k: v for k, v in p.items() if k != "MW"})


def test_gharagheizi_gas_singularity_is_not_filled():
    with pytest.raises(ValueError, match="singular"):
        conductivity_value("Gharagheizi_gas", 10, MW=16, Tb=5, Pc=1e5, omega=3.35875)
