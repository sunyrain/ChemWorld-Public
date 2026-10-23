import json
from dataclasses import replace
from math import exp, log

import pytest

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.cubic_mixing import (
    GibbsExcessMixingSpec,
    cubic_helmholtz_fugacities,
    cubic_mixing_expression,
    cubic_residual_helmholtz,
)
from chemworld.physchem.eos import (
    CubicEOSSpec,
    EOSComponentSpec,
    cubic_mixture_parameters,
    cubic_state_derivatives,
    evaluate_cubic_eos,
)
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.regular_solution import regular_solution_model
from chemworld.physchem.unifac import UNIFACSubgroup, unifac_activity_model

RULES = ("huron_vidal", "mhv", "mhv1", "mhv2", "wong_sandler")


def case(rule, model="peng_robinson", n=3):
    ids = tuple("ABC"[:n])
    activity = ActivityModelSpec("mix-activity", ids, "nrtl", {
        parameter: value for i, left in enumerate(ids) for j, right in enumerate(ids) if i != j
        for parameter, value in ((f"tau_b:{left}|{right}", 70+60*i-40*j),
                                  (f"alpha:{left}|{right}", .3))})
    coefficients = ((-.53 if model == "peng_robinson" else -.593,) if rule == "mhv1" else
                    ((-.4347, -.003654) if model == "peng_robinson" else (-.478, -.0047))
                    if rule == "mhv2" else ())
    pairs = {f"{a}|{b}": .04 for i, a in enumerate(ids) for b in ids[i+1:]}
    ge = GibbsExcessMixingSpec(rule, activity, coefficients,
                               pairs if rule == "wong_sandler" else {})
    components = tuple(EOSComponentSpec(key, tc, pc, omega) for key, tc, pc, omega in
                        zip(ids, (500,580,650), (4e6,3.5e6,3e6), (.2,.4,.6), strict=False))
    return CubicEOSSpec("GE", model, components, ge_mixing=ge)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("model", ("peng_robinson", "srk"))
def test_all_rules_original_eos_caloric_and_pure_limits(rule, model):
    spec = case(rule, model)
    composition = {"A": .2, "B": .3, "C": .5}
    state = evaluate_cubic_eos(spec, composition, temperature_K=340, pressure_Pa=1e5)
    assert state.molar_residual_gibbs_J_mol == pytest.approx(
        state.molar_residual_enthalpy_J_mol-340*state.molar_residual_entropy_J_mol_K, abs=1e-7)
    assert state.molar_residual_gibbs_J_mol == pytest.approx(
        8.31446261815324*340*sum(composition[k]*log(v)
                               for k,v in state.fugacity_coefficients.items()), abs=1e-7)
    pure = evaluate_cubic_eos(spec, {"A":1., "B":0., "C":0.}, temperature_K=340, pressure_Pa=1e5)
    classic = evaluate_cubic_eos(replace(spec, ge_mixing=None), {"A":1., "B":0., "C":0.},
                                  temperature_K=340, pressure_Pa=1e5)
    assert pure.compressibility_factor == pytest.approx(classic.compressibility_factor, abs=2e-10)
    assert pure.fugacity_coefficients["A"] == pytest.approx(classic.fugacity_coefficients["A"])
    payload = json.loads(json.dumps(spec.to_dict()))
    restored = CubicEOSSpec.from_dict(payload)
    assert restored.to_dict() == spec.to_dict()
    assert evaluate_cubic_eos(restored, composition, temperature_K=340, pressure_Pa=1e5) == state


def test_ws_temperature_dependent_covolume_and_original_mechanical_response():
    spec = case("wong_sandler")
    x = {"A":.2,"B":.3,"C":.5}
    mix = cubic_mixture_parameters(spec, x, temperature_K=340)
    assert mix.db_mix_dT != 0
    derivative = cubic_state_derivatives(spec, x, temperature_K=340, pressure_Pa=1e5)
    high = evaluate_cubic_eos(spec, x, temperature_K=340.001, pressure_Pa=1e5)
    low = evaluate_cubic_eos(spec, x, temperature_K=339.999, pressure_Pa=1e5)
    assert derivative["d_molar_volume_dT_P"] == pytest.approx(
        (high.molar_volume_m3_mol-low.molar_volume_m3_mol)/.002, rel=2e-7)


@pytest.mark.parametrize("model", ("peng_robinson", "srk"))
def test_same_residual_potential_reproduces_original_classical_mixing(model):
    spec = replace(case("mhv1", model), ge_mixing=None, binary_interaction={"A|B":.04})
    xs = [.2,.3,.5]
    x = dict(zip(spec.component_ids,xs,strict=True))
    state = evaluate_cubic_eos(spec, x, temperature_K=340, pressure_Pa=1e5)
    lnphi = cubic_helmholtz_fugacities(spec, xs, 340., state.molar_volume_m3_mol,
                                     state.compressibility_factor)
    assert [exp(v) for v in lnphi] == pytest.approx(list(state.fugacity_coefficients.values()))
    A = cubic_residual_helmholtz(spec, xs, 340., state.molar_volume_m3_mol)
    dA = partial_derivative(lambda v: cubic_residual_helmholtz(
        spec,xs,v['T'],state.molar_volume_m3_mol), {'T':340.}, ('T',))
    assert A-340*dA+8.31446261815324*340*(state.compressibility_factor-1) == pytest.approx(
        state.molar_residual_enthalpy_J_mol)


def test_incomplete_and_incompatible_declarations_reject():
    spec = case("wong_sandler")
    with pytest.raises(ValueError, match="every explicit"):
        replace(spec.ge_mixing, binary_interaction={})
    with pytest.raises(ValueError, match="negative q"):
        GibbsExcessMixingSpec("mhv2", spec.ge_mixing.activity)
    with pytest.raises(ValueError, match="cannot be combined"):
        replace(spec, binary_interaction={"A|B":.2})
    with pytest.raises(ValueError, match="zero-pressure"):
        cubic_mixing_expression(case("mhv"), [.2,.3,.5], 1000.)


def activity_variants():
    result = [ActivityModelSpec(name, ("A","B"), name, params) for name,params in (
        ("ideal", {}), ("margules", {"A:A|B":.2,"A:B|A":.4}),
        ("wilson", {"lambda:A|B":.8,"lambda:B|A":.6}),
        ("nrtl", {"tau_b:A|B":70.,"tau_b:B|A":-50.,"alpha:A|B":.3,"alpha:B|A":.3}),
        ("uniquac", {"r:A":2.,"r:B":1.,"q:A":1.5,"q:B":1.2,
                     "tau_b:A|B":80.,"tau_b:B|A":-100.}),
        ("hansen", {"V:A":90e-6,"V:B":110e-6,"alpha":.6,
                    "delta_d:A":16000.,"delta_d:B":15000.,"delta_p:A":4000.,
                    "delta_p:B":2000.,"delta_h:A":6000.,"delta_h:B":3000.}),
    )]
    result.extend(regular_solution_model(name,("A","B"),model=name,
        molar_volumes_m3_mol={"A":90e-6,"B":110e-6},
        solubility_parameters_Pa_half={"A":18000.,"B":16000.},
        interaction_parameters={("A","B"):.02,("B","A"):.04})
        for name in ("regular_solution","flory_huggins"))
    result.extend(unifac_activity_model(f"unifac-{version}",
        {"A":{1:2,2:4},"B":{1:1,2:1,18:1}}, version=version,
        subgroups={1:UNIFACSubgroup(1,1,.9011,.848),2:UNIFACSubgroup(2,1,.6744,.540),
                   18:UNIFACSubgroup(18,9,1.6724,1.488)},
        interactions={(1,9):(476.4,0.,0.),(9,1):(26.76,0.,0.)},
        provenance={"source":"explicit interface exercise"}) for version in range(6))
    return result


@pytest.mark.parametrize("activity", activity_variants(), ids=lambda v:v.model_id)
@pytest.mark.parametrize("rule", RULES)
def test_every_ge_family_and_six_unifac_variants_reach_original_eos(activity, rule):
    spec = case(rule,n=2)
    spec = replace(spec, ge_mixing=replace(spec.ge_mixing,activity=activity))
    state = evaluate_cubic_eos(spec,{"A":.3,"B":.7},temperature_K=340,pressure_Pa=1e5)
    assert state.compressibility_factor > 0
    assert state.molar_residual_gibbs_J_mol == pytest.approx(
        8.31446261815324*340*(.3*log(state.fugacity_coefficients['A'])
                            +.7*log(state.fugacity_coefficients['B'])), abs=1e-7)
