import json
import math
from types import SimpleNamespace

import pytest
from tests.reference.humid_air_official import G8_INPUTS

from chemworld.physchem import (
    humid_air_derivatives,
    humid_air_helmholtz,
    humid_air_state,
    humid_air_tp,
    iapws95_water,
    iapws_dry_air,
)
from chemworld.physchem.helmholtz import HelmholtzFluid

PROCESS_INPUTS = tuple((A, T, rho) for T in (200, 300, 450)
                       for A in ((.99999999, .9999999999) if T == 200 else (.99, .9999))
                       for rho in (.001, 1))


@pytest.mark.parametrize("A,T,rho", (*G8_INPUTS, *PROCESS_INPUTS))
def test_pressure_inverse_and_same_potential_identities(A, T, rho):
    state = humid_air_state(A, T, rho)
    recovered = humid_air_tp(A, T, state.pressure_Pa)
    assert recovered.mass_density_kg_m3 == pytest.approx(rho, rel=2e-9)
    assert state.enthalpy_J_kg == pytest.approx(
        state.internal_energy_J_kg+state.pressure_Pa/rho, rel=2e-10)
    assert state.gibbs_energy_J_kg == pytest.approx(
        state.enthalpy_J_kg-T*state.entropy_J_kg_K, rel=2e-10)
    assert state.gibbs_energy_J_kg == pytest.approx(
        A*state.air_chemical_potential_J_kg+(1-A)*state.water_chemical_potential_J_kg, rel=2e-10)
    assert state.cp_J_kg_K-state.cv_J_kg_K == pytest.approx(
        T*state.dP_dT_rho_Pa_K**2/(rho*rho*state.dP_drho_T_Pa_m3_kg), rel=2e-10)
    assert state.composition_stiffness_J_kg > 0
    assert state.humidity_ratio_kg_water_kg_dry_air == pytest.approx((1-A)/A)
    assert state.enthalpy_J_kg_dry_air == pytest.approx(state.enthalpy_J_kg/A)
    assert json.loads(json.dumps(state.to_dict(), allow_nan=False))["dry_air_mass_fraction"] == A


@pytest.mark.parametrize("A,T,rho", G8_INPUTS)
def test_shared_hessian_matches_centered_derivatives(A, T, rho):
    d = humid_air_derivatives(A, T, rho)
    coordinates = {"A": A, "T": T, "rho": rho}
    increments = {"A": 1e-6, "T": .001, "rho": rho*1e-5}
    for key, step in increments.items():
        low, high = dict(coordinates), dict(coordinates)
        low[key] -= step
        high[key] += step
        left, right = [tuple(point[k] for k in ("A", "T", "rho")) for point in (low, high)]
        numerical = (humid_air_helmholtz(*right)-humid_air_helmholtz(*left))/(2*step)
        assert d["f_"+key] == pytest.approx(numerical, rel=2e-5)
        dl, dr = humid_air_derivatives(*left), humid_air_derivatives(*right)
        for first in ("A", "T", "rho"):
            if (first, key) in (("T", "A"), ("rho", "A"), ("rho", "T")):
                label = "f_"+key+first
            else:
                label = "f_"+first+key
            numerical = (dr["f_"+first]-dl["f_"+first])/(2*step)
            assert d[label] == pytest.approx(numerical, rel=2e-5, abs=0)


@pytest.mark.parametrize("A,model", [(0, iapws95_water), (1, iapws_dry_air)])
def test_pure_limits_keep_existing_fluid_and_absent_species_semantics(A, model):
    fluid, T, rho = model(), 400, .1
    humid = humid_air_state(A, T, rho)
    pure = fluid.state(temperature_K=T, molar_density_mol_m3=rho/fluid.molar_mass_kg_mol)
    assert humid.pressure_Pa == pytest.approx(pure.pressure_Pa, rel=2e-12)
    assert humid.enthalpy_J_kg == pytest.approx(pure.enthalpy_J_mol/fluid.molar_mass_kg_mol)
    assert humid.water_mole_fraction == 1-A
    if A:
        assert humid.water_chemical_potential_J_kg is None
    else:
        assert humid.air_chemical_potential_J_kg is None
        assert humid.humidity_ratio_kg_water_kg_dry_air is None
    with pytest.raises(ValueError, match="singular"):
        humid_air_derivatives(A, T, rho)
    recovered = HelmholtzFluid.from_dict(json.loads(json.dumps(fluid.to_dict())))
    assert recovered.to_dict() == fluid.to_dict()


@pytest.mark.parametrize("args", [(-.01, 300, 1), (1.01, 300, 1), (True, 300, 1),
                                  (.5, 192.9, 1), (.5, 473.1, 1), (.5, math.nan, 1),
                                  (.5, True, 1), (.5, 300, -1), (.5, 300, 0),
                                  (.5, 300, math.inf), (.5, 300, True), (.5, 300, 1000)])
def test_invalid_and_unstable_states_rejected(args):
    with pytest.raises(ValueError):
        humid_air_state(*args)


@pytest.mark.parametrize("p", [True, 0, -1, math.nan, math.inf, 5e6+1])
def test_pressure_domain_refusal(p):
    with pytest.raises(ValueError):
        humid_air_tp(.99, 300, p)


def test_nonconvergence_is_not_replaced_with_an_ideal_gas(monkeypatch):
    from chemworld.physchem import humid_air

    monkeypatch.setattr(humid_air, "solve_root", lambda *a, **k:
                        SimpleNamespace(success=False, failure_reason="forced failure"))
    with pytest.raises(ValueError, match="forced failure"):
        humid_air_tp(.99, 300, 101325)
