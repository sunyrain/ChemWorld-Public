"""Molecular residuals actually reach existing phase and energy consumers."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.corresponding_states import (
    CorrespondingStatesComponent,
    CubicShapeComponent,
    ExtendedCorrespondingStatesModel,
    HuberElyModel,
    LeeKeslerPlockerModel,
    MBWRReference,
)
from chemworld.physchem.coupled_system import CoupledState, CoupledSystem, HeatFlux
from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.eos import R_J_PER_MOL_K as R
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_flash import HelmholtzTwoPhaseState, flash_at_pressure
from chemworld.physchem.helmholtz_processes import HelmholtzPressureStage, HelmholtzStream
from chemworld.physchem.helmholtz_solvers import HelmholtzAncillary, HelmholtzPhaseModel
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.helmholtz_thermal import HelmholtzIsochoricThermodynamics
from chemworld.physchem.lj_spline import LJSplineBHModel, LJSplineUVModel, LJSplineWCAModel
from chemworld.physchem.molecular_consumers import (
    MolecularHelmholtzFluid,
    molecular_model_from_dict,
)
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.molecular_potential_extensions import (
    ChaparroMieModel,
    PohlMieModel,
    TwoCenterLJModel,
)
from chemworld.physchem.molecular_potentials import PurePotentialModel, SoftSAFTModel
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL, PCSAFTModel, SAFTSegment
from chemworld.physchem.quantum_molecular import QuantumMieSegment, SAFTVRQMieModel
from chemworld.physchem.quantum_molecular_cubic import QuantumCubicComponent, QuantumCubicModel
from chemworld.physchem.saftvrmie import MieSegment, SAFTVRMieModel
from chemworld.physchem.solver_backend import ODESolverPolicy

RHO = 1/(AVOGADRO_PER_MOL*3.4e-10**3)


def models():
    segment = SAFTSegment("A", 1, 3.4e-10, 120, "synthetic test scale")
    cpa = CPAModel((CPAComponent("A", .2, 2e-5, 160, .4, "test"),),
                   ((0.,),), (60, 1200), "test")
    critical = CorrespondingStatesComponent("A", 160, 1e-4, .2, "test")
    shape = CubicShapeComponent("A", .2, 2e-5, 160, "soave", (.4,), "test")
    return [PCSAFTModel((segment,), ((0.,),), (60, 1200), "test"),
            cpa,
            SAFTVRMieModel((MieSegment(**segment.__dict__),), ((0.,),), (60, 1200), "test"),
            PurePotentialModel(segment, "LJ-Thol-2016", (60, 1200), "Thol2016"),
            SoftSAFTModel((segment,), (60, 1200), "test"),
            *[cls("A", 3.4e-10, 120, (60, 1200), "test")
              for cls in (TwoCenterLJModel, PohlMieModel, ChaparroMieModel)],
            SAFTVRQMieModel((QuantumMieSegment("A", 1, 3.4e-10, 120, "test", 12., 6.,
                                               .04/AVOGADRO_PER_MOL),),
                           ((0.,),), (60, 1200), "test"),
            QuantumCubicModel((QuantumCubicComponent("A", .2, 2e-5, 160, (.4, .9, 1.1),
                                                     1., 2., 1e-6, "test"),),
                              ((0.,),), (60, 1200), "test"),
            *[cls("A", 3.4e-10, 120, (60, 1200), "test")
              for cls in (LJSplineBHModel, LJSplineWCAModel, LJSplineUVModel)],
            LeeKeslerPlockerModel((critical,), ((1.,),), (60, 1200), "test"),
            HuberElyModel(critical, .28, critical, .28, (.2, -.1), (.3, .1),
                          cpa, (60, 1200), "test"),
            ExtendedCorrespondingStatesModel((shape,), ((0.,),), shape, cpa, (60, 1200), "test"),
            MBWRReference("A", 19, (.1,)+(0.,)*18, 1., (60, 1200), "test")]


def reference():
    return HelmholtzFluid("A", "", .04, R, 158.4, .31*RHO, 84, 1200, 1e9,
        (HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 2., "a2": 3.}),
         HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5})), (),
        "explicit synthetic monatomic reference with nonzero H/S offsets")


def phase_model():
    fluid = MolecularHelmholtzFluid.from_reference(models()[3], reference())
    tc, rc, pc = 158.4, .31*RHO, .13006*R*120*RHO
    return HelmholtzPhaseModel(fluid, tc, rc, pc, {
        "rhoL": HelmholtzAncillary("rhoLnoexp", tc, rc, 84, tc, (2.,), (.35,)),
        "rhoV": HelmholtzAncillary("rhoV", tc, rc, 84, tc, (-5.,), (.5,), True),
        "pS": HelmholtzAncillary("pV", tc, pc, 84, tc, (-5.,), (1.,), True),
    })


@pytest.mark.parametrize("model", models(), ids=lambda m: type(m).__name__)
def test_absolute_reference_and_json_flow_through_original_consumers(model):
    ideal = reference()
    fluid = MolecularHelmholtzFluid.from_reference(model, ideal)
    residual = molecular_eos_state(model, 480, 1000, (1.,))
    state = fluid.state(temperature_K=480, molar_density_mol_m3=1000)
    zero = ideal.state(temperature_K=480, molar_density_mol_m3=1000)
    for absolute, departure in (("enthalpy_J_mol", "residual_enthalpy_J_mol"),
                                ("internal_energy_J_mol", "residual_internal_energy_J_mol"),
                                ("entropy_J_mol_K", "residual_entropy_J_mol_K"),
                                ("cp_J_mol_K", "residual_cp_J_mol_K"),
                                ("cv_J_mol_K", "residual_cv_J_mol_K")):
        assert getattr(state, absolute) == pytest.approx(
            getattr(zero, absolute)+getattr(residual, departure))
    assert state.pressure_Pa == pytest.approx(residual.pressure_Pa)
    thermal = HelmholtzIsochoricThermodynamics(fluid, "A", 2.)
    restored = HelmholtzIsochoricThermodynamics.from_dict(json.loads(json.dumps(thermal.to_dict())))
    assert restored == thermal
    assert HelmholtzFluid.from_dict(json.loads(json.dumps(fluid.to_dict()))) == fluid


def test_reference_identity_mixture_and_duplicate_residual_rejections():
    model = models()[3]
    for ref in (replace(reference(), fluid_id="other"),
                replace(reference(), gas_constant_J_mol_K=8.),
                replace(reference(), ideal_prefactor=2.), replace(reference(), pseudo_pure=True),
                replace(reference(), residual_terms=(HelmholtzTerm("ResidualHelmholtzPower",
                    {"n": [.1], "d": [1], "t": [1], "l": [0]}),))):
        with pytest.raises(ValueError):
            MolecularHelmholtzFluid.from_reference(model, ref)
    mix = SoftSAFTModel((model.component, replace(model.component, component_id="B")),
                       (60, 1200), "test")
    with pytest.raises(ValueError, match="exactly"):
        MolecularHelmholtzFluid.from_reference(mix, reference())
    with pytest.raises(ValueError, match="unknown"):
        molecular_model_from_dict({"kind": "arbitrary.module"})
    fluid = MolecularHelmholtzFluid.from_reference(model, reference())
    with pytest.raises(ValueError, match="already bound"):
        MolecularHelmholtzFluid.from_reference(model, fluid)


def test_actual_lj_coexistence_and_two_phase_ph_ps():
    model = phase_model()
    saturation = model.saturation(120.)
    assert saturation.success, saturation.failure_reason
    state = HelmholtzTwoPhaseState(saturation.liquid, saturation.gas, .35)
    for coordinate, value in (("H", state.enthalpy_J_mol), ("S", state.entropy_J_mol_K)):
        flash = flash_at_pressure(model, pressure_Pa=state.pressure_Pa,
                                  coordinate=coordinate, value=value)
        assert flash.success, flash.failure_reason
        assert flash.state.temperature_K == pytest.approx(120., rel=1e-9)
        assert flash.state.vapor_fraction == pytest.approx(.35, abs=1e-8)


@pytest.mark.parametrize("mode", ["throttle", "compressor", "expander"])
def test_existing_pressure_stage_uses_molecular_absolute_reference(mode):
    model = phase_model()
    upstream = model.solve_tp(temperature_K=360, pressure_Pa=3*model.critical_pressure_Pa).state
    device = HelmholtzPressureStage(model, mode,
        (4 if mode == "compressor" else 2)*model.critical_pressure_Pa,
        1 if mode == "throttle" else .8)
    result = device.evaluate(HelmholtzStream(upstream, .01))
    assert result.success, result.failure_reason
    assert result.diagnostics["energy_residual_W"] == pytest.approx(0, abs=1e-6)
    assert result.diagnostics["entropy_generation_W_K"] >= -1e-10
    assert HelmholtzPressureStage.from_dict(json.loads(json.dumps(device.to_dict()))) == device


@pytest.mark.parametrize("numerical_form", ["ode", "residual"])
def test_existing_finite_inventory_heating_reuses_molecular_energy(numerical_form):
    fluid = phase_model().fluid
    thermal = HelmholtzIsochoricThermodynamics(fluid, "A", 2., 5.)
    start = CoupledState({"A": 2.}, 360, 2.)
    result = CoupledSystem(thermal, heat_flux=lambda _: HeatFlux(20.),
                           numerical_form=numerical_form,
                           policy=ODESolverPolicy("molecular-heating", method="BDF",
                                                  rtol=1e-11, atol=1e-13)).advance(start, 2.)
    assert result.final_state.amounts_mol == start.amounts_mol
    assert thermal.internal_energy_j(result.final_state)-thermal.internal_energy_j(start) == (
        pytest.approx(40., abs=2e-5))
    assert result.energy_balance_residual_J == pytest.approx(0., abs=2e-5)
