"""Absolute mixture mixing, endpoint energies and actual open inventory advance."""

import json
from dataclasses import replace
from math import log

import pytest

from chemworld.physchem.coupled_system import (
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    HeatFlux,
    MaterialPort,
)
from chemworld.physchem.eos import R_J_PER_MOL_K as R
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_consumers import MolecularHelmholtzFluid
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.molecular_mixture import (
    MolecularCaloricModel,
    MolecularIsochoricThermodynamics,
    MolecularStream,
)
from chemworld.physchem.pcsaft import PCSAFTModel, SAFTSegment
from chemworld.physchem.solver_backend import ODESolverPolicy


def caloric_model():
    components = (
        SAFTSegment("A", 1.0, 3.4e-10, 120.0, "test"),
        SAFTSegment("B", 2.0, 3.8e-10, 180.0, "test"),
    )
    residual = PCSAFTModel(components, ((0.0, 0.02), (0.02, 0.0)), (100.0, 1000.0), "test")
    ideal = tuple(
        HelmholtzFluid(
            c.component_id,
            "",
            0.04 + 0.02 * i,
            R,
            300.0,
            1000.0,
            100.0,
            1000.0,
            1e9,
            (
                HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 2.0 + i, "a2": 3.0 + i}),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5 + i}),
            ),
            (),
            "explicit test reference",
        )
        for i, c in enumerate(components)
    )
    return MolecularCaloricModel(residual, ideal)


def test_ideal_entropy_mixing_and_pure_endpoint():
    model = caloric_model()
    x = (0.3, 0.7)
    state = model.state(400.0, 500.0, x)
    residual = molecular_eos_state(model.residual_model, 400.0, 500.0, x)
    ideal = [
        ref.state(temperature_K=400.0, molar_density_mol_m3=500.0) for ref in model.ideal_references
    ]
    assert state.entropy_J_mol_K == pytest.approx(
        sum(fraction * value.entropy_J_mol_K for fraction, value in zip(x, ideal, strict=True))
        - R * sum(v * log(v) for v in x)
        + residual.residual_entropy_J_mol_K
    )
    pure = replace(
        model.residual_model, components=model.residual_model.components[:1], interaction=((0.0,),)
    )
    fluid = MolecularHelmholtzFluid.from_reference(pure, model.ideal_references[0])
    endpoint = model.state(400.0, 500.0, (1.0, 0.0))
    expected = fluid.state(temperature_K=400.0, molar_density_mol_m3=500.0)
    for name in (
        "pressure_Pa",
        "internal_energy_J_mol",
        "enthalpy_J_mol",
        "entropy_J_mol_K",
        "cp_J_mol_K",
        "cv_J_mol_K",
        "speed_of_sound_m_s",
    ):
        assert getattr(endpoint, name) == pytest.approx(getattr(expected, name))
    assert state.gibbs_energy_J_mol == pytest.approx(
        state.enthalpy_J_mol - 400 * state.entropy_J_mol_K
    )
    root = model.state_tp(
        400, state.pressure_Pa, x, initial_density_mol_m3=550, density_bounds_mol_m3=(300, 700)
    )
    assert root.molar_density_mol_m3 == pytest.approx(500.0)
    restored = MolecularCaloricModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored == model and restored.state(400, 500, x) == state
    assert state.to_dict()["energy_reference"] == "explicit_component_ideal_plus_residual"


@pytest.mark.parametrize("amounts", [{"A": 0.6, "B": 1.4}, {"A": 2.0, "B": 0.0}])
def test_partial_internal_energies_are_fixed_volume_derivatives(amounts):
    thermal = MolecularIsochoricThermodynamics(caloric_model(), 4.0)
    start = CoupledState(amounts, 400.0, 4.0)
    partials = thermal.model.partial_inventory_energies(400, 0.004, amounts)
    step = 2e-4
    for key, partial in zip(thermal.inventory_keys, partials, strict=True):
        values = [
            thermal.internal_energy_j(
                replace(start, amounts_mol={**amounts, key: amounts[key] + i * step})
            )
            for i in range(5)
        ]
        slope = (
            -25 * values[0] + 48 * values[1] - 36 * values[2] + 16 * values[3] - 3 * values[4]
        ) / (12 * step)
        assert partial == pytest.approx(slope, rel=2e-8, abs=2e-5)
    assert thermal.chemical_energy_rate_w(start, {"B": 0.02}, ()) == pytest.approx(
        0.02 * partials[1]
    )


@pytest.mark.parametrize("form", ["ode", "residual"])
@pytest.mark.parametrize("mode", ["heat", "feed"])
def test_real_mixture_inventory_conserves_mass_and_energy(form, mode):
    model = caloric_model()
    thermal = MolecularIsochoricThermodynamics(model, 4.0, 5.0)
    start = CoupledState({"A": 2.0, "B": 0.0}, 400.0, 4.0)
    options = {
        "numerical_form": form,
        "policy": ODESolverPolicy("molecular-mixture", method="BDF", rtol=1e-11, atol=1e-13),
    }
    if mode == "heat":
        options["heat_flux"] = lambda _: HeatFlux(20.0)
        expected_energy, expected_amounts = 40.0, {"A": 2.0, "B": 0.0}
    else:
        stream = MolecularStream(model.state(500.0, 500.0, (0.3, 0.7)), 0.01)
        contribution = stream.contribution({"A": "feedA", "B": "feedB"})
        options["processes"] = (CoupledProcess("feed", lambda _: contribution),)
        options["ports"] = tuple(
            MaterialPort("feed" + key, key, "source", "vessel", "in") for key in ("A", "B")
        )
        expected_energy, expected_amounts = 2 * stream.enthalpy_rate_W, {"A": 2.006, "B": 0.014}
    result = CoupledSystem(thermal, **options).advance(start, 2.0)
    final = result.final_state
    assert final.amounts_mol == pytest.approx(expected_amounts)
    assert thermal.internal_energy_j(final) - thermal.internal_energy_j(start) == pytest.approx(
        expected_energy, abs=2e-5
    )
    assert result.energy_balance_residual_J == pytest.approx(0.0, abs=2e-5)
    assert final.time_s == 2.0 and start.amounts_mol == {"A": 2.0, "B": 0.0}
    restored = MolecularIsochoricThermodynamics.from_dict(json.loads(json.dumps(thermal.to_dict())))
    assert restored == thermal and restored.internal_energy_j(final) == thermal.internal_energy_j(
        final
    )


def test_explicit_reference_volume_and_stream_boundaries():
    model = caloric_model()
    with pytest.raises(ValueError, match="order"):
        replace(model, ideal_references=model.ideal_references[::-1])
    with pytest.raises(ValueError, match="same R"):
        replace(
            model,
            ideal_references=(
                replace(model.ideal_references[0], gas_constant_J_mol_K=8.0),
                model.ideal_references[1],
            ),
        )
    thermal = MolecularIsochoricThermodynamics(model, 4.0)
    with pytest.raises(ValueError, match="fixed volume"):
        thermal.validate_state(CoupledState({"A": 1.0, "B": 1.0}, 400.0, 5.0))
    stream = MolecularStream(model.state(400, 500, (0.3, 0.7)), 0.01)
    with pytest.raises(ValueError, match="distinct ports"):
        stream.contribution({"A": "feed", "B": "feed"})
