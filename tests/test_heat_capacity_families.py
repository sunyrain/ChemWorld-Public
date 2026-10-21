"""Caloric formula domains, piecewise thermodynamics and real native consumers."""

import json
import subprocess
import sys
from dataclasses import replace
from math import log

import pytest

from chemworld.physchem.correlation_families import CORRELATION_FAMILIES
from chemworld.physchem.heat_capacity import (
    PiecewiseHeatCapacity,
    temperature_from_caloric_change,
    validate_capacity_curve,
)
from chemworld.physchem.heat_capacity_kernels import vibrational_temperature
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
    integrate_correlation,
)
from chemworld.physchem.property_packages import (
    PropertyPackageThermodynamics,
    SpeciesCaloricModel,
    evaluate_caloric_correlation,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation, property_equation_contracts

# Published equation examples, not claims of experimental material accuracy.
CASES = {
    "poling_cp": ({"a": 4.568, "b": -.008975, "c": 3.631e-5, "d": -3.407e-8, "e": 1.091e-11},
                  "ideal_gas", 300, 35.850973388425),
    "shomate_cp": ({"A": 30.092, "B": .006832514, "C": 6.793435e-6,
                    "D": -2.534480e-9, "E": 82139}, "ideal_gas", 500, 35.21836175),
    "trc_cp": (dict(zip((f"a{i}" for i in range(8)),
                        (4, 7.65e5, 720, 3.565, -.052, -1.55e6, 52, 201), strict=True)),
               "ideal_gas", 300, 42.065271080974654),
    "ppds2_cp": ({"Ts": 462.493, "C_low": 4.54115, "C_inf": 9.96847, "a1": -103.419,
                  "a2": 695.484, "a3": -2006.1, "a4": 2476.84, "a5": -1186.47},
                 "ideal_gas", 350, 136.46338956689),
    "ppds15_cp": ({"Tc": 562.05, "a0": .198892, "a1": 24.1389, "a2": -20.2301,
                   "a3": 5.72481, "a4": 4.43613e-7, "a5": -3.10751e-7},
                  "liquid", 400, 161.8983143509),
    "tde_cs_cp": ({"Tc": 778, "b": .626549, "a1": 120.705, "a2": .255987,
                   "a3": .000381027, "a4": -3.03077e-7}, "liquid", 550, 328.472042686),
    "rowlinson_poling_cp": ({"Tc": 435.5, "omega": .203, "Cpgm": 91.21},
                            "liquid", 350, 143.80196224081436),
    "rowlinson_bondi_cp": ({"Tc": 535.55, "omega": .323, "Cpgm": 119.342},
                           "liquid", 373.28, 175.3976263003074),
    "zabransky_cubic_cp": ({"a1": 20.9634, "a2": -10.1344, "a3": 2.8253, "a4": -.256738},
                           "liquid", 298.15, 75.31465144297),
    "zabransky_quasi_cp": ({"Tc": 591.79, "a1": -3.12743, "a2": .0857315, "a3": 13.7282,
                           "a4": 1.28971, "a5": 6.42297, "a6": 4.10989},
                          "liquid", 330, 165.472878778683),
    "perry151_cp": ({"a": 4.8, "b": .00322, "c": 0, "d": 0}, "solid", 300, 24.124944),
    "lastovka_shaw_cp": ({"similarity_variable": .22491, "MW": 142.28, "cyclic_aliphatic": 0},
                         "ideal_gas", 1000, 530.7443465580366),
    "dadgostar_shaw_cp": ({"similarity_variable": .139, "MW": 100},
                          "liquid", 355.6, 180.25291501191516),
    "lastovka_solid_cp": ({"similarity_variable": .2139, "MW": 100},
                          "solid", 300, 168.20637469909211),
    "statistical_cpg": ({"linear": 0, **dict(zip((f"theta_{i}" for i in range(6)),
                           (1360, 2330, 2330, 4800, 4880, 4880), strict=True))},
                        "ideal_gas", 300, 35.55983440173097),
}


def card(equation):
    coefficients, phase, temperature, _ = CASES[equation]
    return PropertyCorrelation(equation, equation, dict(coefficients), {"temperature": "K"},
        "J/(mol*K)", phase + "_heat_capacity", {"temperature": (.8*temperature, 1.2*temperature)},
        "Published formula example; equation reproduction only")


@pytest.mark.parametrize("equation", CASES)
def test_published_examples_and_native_caloric_contract(equation):
    curve = card(equation)
    temperature, expected = CASES[equation][2:]
    actual = evaluate_correlation(curve, temperature_K=temperature).value
    assert actual == pytest.approx(expected, rel=2e-10)
    assert PropertyCorrelation.from_dict(json.loads(json.dumps(curve.to_dict()))) == curve
    assert property_equation_contracts()[equation]["coefficient_output_unit"] == "J/(mol*K)"
    validate_capacity_curve(curve, .9*temperature, 1.1*temperature)
    forward = integrate_correlation(curve, initial_temperature_K=.9*temperature,
                                    final_temperature_K=1.1*temperature)
    reverse = integrate_correlation(curve, initial_temperature_K=1.1*temperature,
                                    final_temperature_K=.9*temperature)
    assert forward.value == -reverse.value and forward.value > 0


@pytest.mark.parametrize("defect", ["unit", "missing", "extra", "boolean", "MW", "shape",
                                    "theta_gap", "theta_alias", "theta_zero"])
def test_bad_caloric_parameters_are_not_guessed(defect):
    curve = card("lastovka_shaw_cp")
    if defect.startswith("theta"):
        curve = card("statistical_cpg")
    c = dict(curve.coefficients)
    if defect == "unit":
        with pytest.raises(ValueError, match="coefficients require"):
            replace(curve, output_unit="kJ/(mol*K)")
        return
    if defect == "missing":
        del c["MW"]
    elif defect == "extra":
        c["invented"] = 1
    elif defect == "boolean":
        c["MW"] = True
    elif defect == "MW":
        c["MW"] = -1
    elif defect == "shape":
        c["cyclic_aliphatic"] = 2
    elif defect == "theta_gap":
        del c["theta_2"]
    elif defect == "theta_alias":
        c["theta_01"] = c.pop("theta_1")
    else:
        c["theta_0"] = 0
    with pytest.raises(ValueError):
        replace(curve, coefficients=c)


def test_critical_poles_interior_negative_capacity_and_vibration_limits():
    for equation in ("ppds15_cp", "zabransky_quasi_cp", "tde_cs_cp", "rowlinson_bondi_cp"):
        curve = card(equation)
        with pytest.raises(ValueError, match="T < Tc"):
            evaluate_correlation(curve, temperature_K=curve.coefficients["Tc"],
                                 validity_policy="ignore")
    curve = replace(card("shomate_cp"), coefficients={
        "A": 333**2 - 1e-4, "B": -666, "C": 1, "D": 0, "E": 0,
    }, validity_ranges={"temperature": (250, 400)})
    assert all(evaluate_correlation(curve, temperature_K=T).value > 0 for T in (250, 400))
    with pytest.raises(ValueError, match="minimum"):
        integrate_correlation(curve, initial_temperature_K=250, final_temperature_K=400)
    assert vibrational_temperature(667) == pytest.approx(959.6641613636505)
    vibration = replace(card("statistical_cpg"), coefficients={"linear": 1, "theta_0": 3000},
                        validity_ranges={"temperature": (1, 1e10)})
    assert evaluate_correlation(vibration, temperature_K=1).value == pytest.approx(3.5*R)
    assert evaluate_correlation(vibration, temperature_K=1e10).value == pytest.approx(4.5*R)
    assert correlation_temperature_derivative(vibration, temperature_K=1).value == 0


def segments():
    first = PropertyCorrelation("low", "poling_cp", dict(zip("abcde", (4, .001, 0, 0, 0),
                                                            strict=True)),
        {"temperature": "K"}, "J/(mol*K)", "ideal_gas_heat_capacity",
        {"temperature": (200, 400)}, "Synthetic first range")
    second = replace(first, correlation_id="high", equation_id="shomate_cp",
        coefficients={"A": 5*R, "B": .002*R, "C": 0, "D": 0, "E": 0},
        validity_ranges={"temperature": (400, 600)}, source_note="Synthetic second range")
    return first, second


def test_piecewise_integrals_inverses_derivative_and_json_keep_same_reference():
    curve = PiecewiseHeatCapacity("joined", segments(), "Synthetic same-phase ranges")
    restored = PiecewiseHeatCapacity.from_dict(json.loads(json.dumps(curve.to_dict())))
    assert restored == curve
    assert curve.segment_at(400).correlation_id == "high"
    derivative = correlation_temperature_derivative(curve, temperature_K=400)
    assert derivative.value == pytest.approx(.002*R) and "right-sided" in derivative.warnings[0]
    H = R*(4*100 + .001/2*(400**2-300**2) + 5*100 + .002/2*(500**2-400**2))
    S = R*(4*log(400/300) + .001*100 + 5*log(500/400) + .002*100)
    ref = CaloricReference(phase="gas", temperature_K=300, energy_J_mol=123,
                           entropy_J_mol_K=10)
    result = evaluate_caloric_correlation(curve, reference=ref, temperature_K=500,
        quantities=("cp", "cv", "enthalpy", "entropy", "internal_energy"))
    assert result["enthalpy"].value == pytest.approx(H + 123)
    assert result["entropy"].value == pytest.approx(S + 10)
    assert result["internal_energy"].value == pytest.approx(H + 123 - R*500)
    assert result["cp"].value - result["cv"].value == pytest.approx(R)
    for entropy, change in ((False, H), (True, S)):
        assert temperature_from_caloric_change(curve, change, reference_temperature_K=300,
                                               entropy=entropy) == pytest.approx(500)
        with pytest.raises(ValueError, match="outside"):
            temperature_from_caloric_change(curve, 1e10, reference_temperature_K=300,
                                             entropy=entropy)
    for policy in ("raise", "warn", "ignore"):
        with pytest.raises(ValueError, match="outside piecewise"):
            evaluate_correlation(curve, temperature_K=199, validity_policy=policy)


@pytest.mark.parametrize("kind", ["gap", "overlap", "phase", "Cv", "reverse"])
def test_piecewise_rejects_ambiguous_coverage_or_phase_basis(kind):
    first, second = segments()
    if kind == "gap":
        second = replace(second, validity_ranges={"temperature": (401, 600)})
    elif kind == "overlap":
        second = replace(second, validity_ranges={"temperature": (399, 600)})
    elif kind == "phase":
        second = replace(second, property_id="liquid_heat_capacity")
    elif kind == "Cv":
        second = PropertyCorrelation("Cv", "cp_polynomial", {"a": 20}, {"temperature": "K"},
            "J/(mol*K)", "isochoric_heat_capacity", {"temperature": (400, 600)})
    else:
        first, second = second, first
    with pytest.raises(ValueError):
        PiecewiseHeatCapacity("invalid", (first, second), "Invalid fixture")


def test_inventory_admission_checks_entire_cv_interval_and_roundtrips():
    curve = PiecewiseHeatCapacity("joined", segments(), "Synthetic gas")
    reference = CaloricReference(phase="gas", temperature_K=300, energy_J_mol=0)
    model = SpeciesCaloricModel(curve, reference)
    model.validate_interval(250, 550)
    assert SpeciesCaloricModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    # Positive Cp everywhere, but Cv has a narrow negative pocket.
    dippr = PropertyCorrelation("bad-Cv", "dippr100", {"A": 333**2 + R - .01, "B": -666,
        "C": 1}, {"temperature": "K"}, "J/(mol*K)", "ideal_gas_heat_capacity",
        {"temperature": (250, 400)})
    model = SpeciesCaloricModel(dippr, reference)
    with pytest.raises(ValueError):
        model.validate_interval(250, 400)


def test_piecewise_zero_cv_retains_gas_entropy_conversion():
    curve = PropertyCorrelation("zero", "cp_polynomial", {"a": 0}, {"temperature": "K"},
        "J/(mol*K)", "isochoric_heat_capacity", {"temperature": (200, 400)})
    joined = PiecewiseHeatCapacity("zero-cv", (curve,), "Zero Cv limit")
    ref = CaloricReference(phase="gas", energy_basis="internal_energy", temperature_K=250,
                           entropy_J_mol_K=10)
    result = evaluate_caloric_correlation(joined, reference=ref, temperature_K=350,
                                         quantities=("cp", "cv", "entropy", "internal_energy"))
    assert result["internal_energy"].value == result["cv"].value == 0
    assert result["entropy"].value == pytest.approx(10 + R*log(350/250))


def test_generic_capacity_cannot_hide_a_formula_phase_mismatch():
    from chemworld.physchem.enthalpy import phase_sensible_enthalpy_report

    gas = replace(card("poling_cp"), property_id="heat_capacity")
    for curve in (gas, PiecewiseHeatCapacity("generic-gas", (gas,), "Gas formula")):
        with pytest.raises(ValueError, match="formula phase"):
            evaluate_caloric_correlation(curve, reference=CaloricReference(phase="liquid"),
                                         temperature_K=300, quantities=("cp",))
        with pytest.raises(ValueError, match="formula phase"):
            phase_sensible_enthalpy_report(component_id="test", phase="liquid",
                heat_capacity_correlation=curve, initial_temperature_K=290, final_temperature_K=310)
    # No metadata is required to detect incompatible gas/liquid formula families.
    liquid = replace(card("zabransky_cubic_cp"), property_id="heat_capacity",
                     validity_ranges={"temperature": (360, 600)})
    with pytest.raises(ValueError, match="phase/basis"):
        PiecewiseHeatCapacity("cross-phase", (gas, liquid), "Invalid implicit transition")


def test_real_thermal_runtime_accepts_piecewise_gas_cp_and_replays():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.surface_world import surface_world_spec

    world = surface_world_spec(include_surface=False)
    physics = world.physics
    thermal = physics.models_of("thermal")["reactor"]
    curve = PiecewiseHeatCapacity("synthetic-gas", segments(), "Synthetic native thermal input")
    thermo = PropertyPackageThermodynamics({key: SpeciesCaloricModel(curve,
        CaloricReference(phase="gas", temperature_K=350, energy_J_mol=value.u_ref_J_mol + R*350))
        for key, value in thermal.thermodynamics.species.items()}, 20, 350)
    models = {key: dict(value) for key, value in physics.models.items()}
    models["thermal"]["reactor"] = replace(thermal, thermodynamics=thermo)
    updated = replace(world, physics=replace(physics, models=models))
    session = ChemWorldRuntime(composition=updated)
    replay = None
    try:
        session.reset(seed=23)
        action = {"operation": "heat", "target_temperature_K": 400, "duration_s": 2,
                  "stirring_speed_rpm": 600}
        assert session.execute(action).info["transaction_status"] == "committed"
        snapshot = json.loads(json.dumps(session.snapshot().to_dict()))
        replay = ChemWorldRuntime.from_snapshot(snapshot)
        for current in (session, replay):
            result = current.execute({"operation": "wait", "duration_s": 2})
            assert result.info["transaction_status"] == "committed"
        assert session.snapshot().to_dict() == replay.snapshot().to_dict()
        assert session._state.temperature_K > 350
    finally:
        session.close()
        if replay:
            replay.close()


def test_public_facade_and_default_runtime_need_no_reference_library():
    families = [family for family in CORRELATION_FAMILIES.values() if family.reference_method]
    assert len(families) == 15
    subprocess.run([sys.executable, "-c", """
import sys
from chemworld.physchem import PiecewiseHeatCapacity, temperature_from_caloric_change
from chemworld.physchem import vibrational_temperature
assert vibrational_temperature(1) > 0
assert 'chemicals' not in sys.modules
"""], check=True, capture_output=True, text=True)
