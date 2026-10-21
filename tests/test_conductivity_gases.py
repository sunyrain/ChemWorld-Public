"""Corresponding-state conductivity branches and actual dense-gas volume."""

import json

import pytest

from chemworld.physchem import (
    PropertyCorrelation,
    chung_dense_conductivity,
    conductivity_derivatives,
    conductivity_value,
    evaluate_correlation,
)
from chemworld.physchem.transport import internal_heat_transfer_coefficient

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
