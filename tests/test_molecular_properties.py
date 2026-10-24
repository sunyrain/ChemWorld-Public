"""Molecular residual EOS enters existing property, phase and flash consumers."""

import json
from dataclasses import replace
from math import log

import pytest

from chemworld.examples.native_flash_research import synthetic_vle_properties
from chemworld.physchem.eos import R_J_PER_MOL_K as R
from chemworld.physchem.flash_units import VLEPropertyPackage, ph_flash, tp_flash
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.molecular_properties import MolecularEOSSpec
from chemworld.physchem.pcsaft import PCSAFTModel, SAFTSegment
from chemworld.physchem.phase_evaluation import PhasePropertyModel
from chemworld.physchem.property_packages import evaluate_property_state


def molecular_vle():
    base = synthetic_vle_properties()
    model = PCSAFTModel(
        (
            SAFTSegment("light", 1, 3.4e-10, 120, "synthetic"),
            SAFTSegment("heavy", 2, 3.8e-10, 180, "synthetic"),
        ),
        ((0, 0.02), (0.02, 0)),
        (100, 1000),
        "synthetic",
    )
    return replace(base, vapor_eos=MolecularEOSSpec("pcsaft-gas", model, "vapor", (1e-8, 3000)))


def test_properties_entropy_reference_and_volume_responses():
    vle = molecular_vle()
    xs = {"light": 0.3, "heavy": 0.7}
    kwargs = {"temperature_K": 400.0, "pressure_Pa": 1e5, "phase": "gas"}
    quantities = (
        "enthalpy",
        "entropy",
        "cp",
        "cv",
        "internal_energy",
        "molar_volume",
        "dV_dT_P",
        "dV_dP_T",
        "fugacity_coefficient",
        "chemical_potential",
    )
    references = {k: replace(v, entropy_J_mol_K=4) for k, v in vle.gas_references.items()}
    report = evaluate_property_state(
        vle.packages, xs, quantities=quantities, references=references, eos=vle.vapor_eos, **kwargs
    )
    ideal = evaluate_property_state(
        vle.packages, xs, quantities=quantities, references=references, **kwargs
    )
    prop = report.properties
    raw = molecular_eos_state(
        vle.vapor_eos.residual_model, 400, 1 / prop["molar_volume"].value, (0.3, 0.7)
    )
    assert prop["enthalpy"].value - ideal.properties["enthalpy"].value == pytest.approx(
        raw.residual_enthalpy_J_mol
    )
    assert prop["entropy"].value - ideal.properties["entropy"].value == pytest.approx(
        raw.residual_entropy_J_mol_K + R * log(raw.compressibility_factor)
    )
    assert prop["internal_energy"].value == pytest.approx(
        prop["enthalpy"].value - 1e5 * prop["molar_volume"].value
    )
    assert prop["cp"].value - prop["cv"].value == pytest.approx(
        400 * raw.dP_dT_rho_Pa_K**2 / (raw.molar_density_mol_m3**2 * raw.dP_drho_T_Pa_m3_mol)
    )
    for key, step, derivative in (
        ("temperature_K", 0.01, "dV_dT_P"),
        ("pressure_Pa", 10.0, "dV_dP_T"),
    ):
        volumes = [
            evaluate_property_state(
                vle.packages,
                xs,
                quantities=("molar_volume",),
                eos=vle.vapor_eos,
                **{**kwargs, key: kwargs[key] + s * step},
            )
            .properties["molar_volume"]
            .value
            for s in (-1, 1)
        ]
        assert (volumes[1] - volumes[0]) / (2 * step) == pytest.approx(
            prop[derivative].value, rel=2e-7
        )
    # Extensive Gibbs identity catches same-density vs same-pressure reference drift.
    mu = report.component_properties["chemical_potential"]
    assert sum(xs[k] * mu[k].value for k in xs) == pytest.approx(
        prop["enthalpy"].value - 400 * prop["entropy"].value
    )


@pytest.mark.parametrize("temperature", (300.0, 350.0, 450.0))
def test_existing_tp_ph_and_codec(temperature):
    model = molecular_vle()
    restored = VLEPropertyPackage.from_dict(json.loads(json.dumps(model.to_dict())))
    feed = {"light": 0.5, "heavy": 0.5}
    fixed = tp_flash(restored, feed, temperature_K=temperature, pressure_Pa=1e5)
    recovered = ph_flash(restored, feed, pressure_Pa=1e5, enthalpy_J=fixed.total_enthalpy_j)
    assert recovered.temperature_K == pytest.approx(temperature, abs=2e-6)
    for k, amount in feed.items():
        assert recovered.liquid_amounts_mol[k] + recovered.vapor_amounts_mol[k] == pytest.approx(
            amount
        )
    assert recovered.total_enthalpy_j == pytest.approx(fixed.total_enthalpy_j, abs=1e-5)
    assert any(abs(v - 1) > 1e-4 for v in fixed.vapor_fugacity_coefficients.values())


def test_phase_declaration_round_trip_and_local_basin_limits():
    model = molecular_vle()
    phase = PhasePropertyModel(
        model.packages, phase="gas", references=model.gas_references, eos=model.vapor_eos
    )
    restored = PhasePropertyModel.from_dict(json.loads(json.dumps(phase.to_dict())))
    assert restored.to_dict() == phase.to_dict()
    evaluation = restored.evaluate(
        {"light": 0.3, "heavy": 0.7},
        temperature_K=400,
        pressure_Pa=1e5,
        quantities=("enthalpy", "molar_volume"),
    )
    assert evaluation.to_dict()
    eos = model.vapor_eos
    endpoint, _ = eos.evaluate({"light": 1}, temperature_K=400, pressure_Pa=1e5, phase="vapor")
    assert endpoint.fugacity_coefficients["heavy"] > 0
    with pytest.raises(ValueError, match="phase"):
        eos.evaluate({"light": 1}, temperature_K=400, pressure_Pa=1e5, phase="liquid")
    with pytest.raises(ValueError, match="basin"):
        eos.evaluate({"light": 1}, temperature_K=400, pressure_Pa=1e9, phase="vapor")
    with pytest.raises(ValueError, match="same R"):
        replace(eos, residual_model=replace(eos.residual_model, gas_constant_J_mol_K=8.314))
