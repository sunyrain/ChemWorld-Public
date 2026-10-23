"""State roots reuse the common solver and preserve explicit physical boundaries."""

import json
from dataclasses import replace

import pytest

from chemworld.data.helmholtz_phase import load_helmholtz_phase_model
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_solvers import (
    HelmholtzAncillary,
    HelmholtzPhaseModel,
    homogeneous_state_derivatives,
)
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.residual_system import RootPolicy


def phase_model():
    fluid = HelmholtzFluid(
        "synthetic", "", .04, 8.3, 300, 1000, 100, 1500, 1e9,
        (HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 0, "a2": 0}),
         HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5})),
        (), "self-authored homogeneous fixture")
    return HelmholtzPhaseModel(fluid, 300, 1000, 1e6, {
        "rhoL": HelmholtzAncillary("rhoLnoexp", 300, 1000, 100, 300, (1,), (1,)),
        "rhoV": HelmholtzAncillary("rhoV", 300, 1000, 100, 300, (-2,), (1,)),
        "pS": HelmholtzAncillary("pV", 300, 1e6, 100, 300, (-2,), (1,)),
    })


@pytest.mark.parametrize("phase", ["auto", "gas", "liquid", "vapor", None])
def test_tp_uses_shared_root_and_does_not_invent_supercritical_phase(phase):
    report = phase_model().solve_tp(temperature_K=500, pressure_Pa=1e5, phase_hint=phase)
    assert report.success and report.state is not None
    assert report.state.molar_density_mol_m3 == pytest.approx(1e5/(8.3*500))
    assert report.diagnostics["algorithm"] == "scaled_projected_newton_with_gradient"
    assert report.diagnostics["selected_phase"] == "supercritical_homogeneous"
    assert report.diagnostics["stability_scope"] == "homogeneous_local_only"


@pytest.mark.parametrize("pair", [("P", "H"), ("P", "S"), ("rho", "U"), ("H", "S")])
def test_independent_caloric_coordinate_pairs_close_the_same_state(pair):
    model = phase_model()
    target = model.fluid.state(temperature_K=500, molar_density_mol_m3=20)
    values = {"P": target.pressure_Pa, "H": target.enthalpy_J_mol,
              "S": target.entropy_J_mol_K, "rho": 20, "U": target.internal_energy_J_mol}
    initial = model.fluid.state(temperature_K=450, molar_density_mol_m3=18)
    report = model.solve_homogeneous_state({k: values[k] for k in pair}, initial_state=initial)
    assert report.success and report.state is not None
    assert report.state.temperature_K == pytest.approx(500)
    assert report.state.molar_density_mol_m3 == pytest.approx(20)
    assert not report.diagnostics["global_phase_selection_performed"]


def test_dependent_pair_and_exhausted_root_do_not_return_success_state():
    model = phase_model()
    initial = model.fluid.state(temperature_K=450, molar_density_mol_m3=18)
    dependent = model.solve_homogeneous_state({"H": 2.5*8.3*500, "U": 1.5*8.3*500},
                                              initial_state=initial)
    assert not dependent.success and dependent.state is None
    exhausted = model.solve_homogeneous_state({"T": 1200, "rho": 100}, initial_state=initial,
                                              policy=RootPolicy(maximum_iterations=1))
    assert not exhausted.success and exhausted.state is None


def test_phase_model_json_roundtrip_and_source_units():
    model = phase_model()
    restored = HelmholtzPhaseModel.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored == model
    eos = {"gas_constant": 8.3, "molar_mass": .04, "T_max": 1500, "p_max": 1e9,
           "pseudo_pure": False,
           "STATES": {"reducing": {"T": 300, "rhomolar": 1000},
                      "sat_min_liquid": {"T": 100}},
           "alpha0": [t.to_dict() for t in model.fluid.ideal_terms], "alphar": []}
    source = {"INFO": {"NAME": "synthetic"}, "EOS": [eos], "ANCILLARIES": {
        key: {"type": row.kind, "T_r": row.reducing_temperature_K,
              "reducing_value": row.reducing_value, "Tmin": row.minimum_temperature_K,
              "Tmax": row.maximum_temperature_K, "n": list(row.n), "t": list(row.t)}
        for key, row in model.ancillaries.items()},
        "STATES": {"critical": {"T": 300, "rhomolar": 1000, "p": 1e6}}}
    imported = load_helmholtz_phase_model(source, source="fixture")
    assert imported.solve_tp(temperature_K=500, pressure_Pa=1e5).success
    source["STATES"]["critical"]["p_units"] = "bar"
    with pytest.raises(ValueError, match="native SI"):
        load_helmholtz_phase_model(source, source="fixture")


def test_pseudo_pure_and_critical_saturation_are_explicit_boundaries():
    model = phase_model()
    pseudo = replace(model, fluid=replace(model.fluid, pseudo_pure=True))
    with pytest.raises(ValueError, match="pseudo-pure"):
        pseudo.saturation(200)
    with pytest.raises(ValueError, match="pseudo-pure"):
        pseudo.solve_tp(temperature_K=500, pressure_Pa=1e5)
    with pytest.raises(ValueError, match="Tcritical"):
        model.saturation(300)
    with pytest.raises(ValueError, match="phase_hint"):
        model.solve_tp(temperature_K=500, pressure_Pa=1e5, phase_hint="solid")
    with pytest.raises(ValueError, match="domain"):
        model.ancillaries["rhoL"].value(400)
    initial = model.fluid.state(temperature_K=500, molar_density_mol_m3=20)
    with pytest.raises(ValueError, match="must be positive"):
        model.solve_homogeneous_state({"T": 500, "P": -1}, initial_state=initial)


def test_caloric_density_derivative_is_not_lost_at_fixed_volume():
    model = phase_model()
    fluid = replace(model.fluid, residual_terms=(HelmholtzTerm("ResidualHelmholtzPower",
        {"n": [.05], "d": [2], "t": [1], "l": [0]}),))
    state = fluid.state(temperature_K=500, molar_density_mol_m3=100)
    derivatives = homogeneous_state_derivatives(fluid, state)
    expected = 2*8.3*300*.05*100/1000**2
    assert derivatives["U"][1] == pytest.approx(expected)
    # d(n*u(T,n/V))/dn = u + rho*u_rho, including the density dependence.
    assert state.internal_energy_J_mol+100*derivatives["U"][1] == pytest.approx(
        1.5*8.3*500+3*8.3*300*.05*(100/1000)**2)
    with pytest.raises(ValueError, match="different fluid"):
        homogeneous_state_derivatives(replace(fluid, fluid_id="other"), state)
