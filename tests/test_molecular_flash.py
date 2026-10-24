"""Independent binodal, true cubic VLE and inventory/failure semantics."""

import json
from dataclasses import dataclass
from math import log

import pytest
from scipy.optimize import brentq

from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.molecular_flash import (
    MolecularDensityBasin,
    molecular_tp_flash,
    molecular_tp_stability,
)


@dataclass(frozen=True)
class ExcessModel:
    amplitude: float = 3
    component_ids: tuple[str, ...] = ("a", "b")
    gas_constant_J_mol_K: float = 8.31446261815324

    def alphar(self, temperature, density, fractions):
        return self.amplitude*fractions[0]*fractions[1]


BASIN = MolecularDensityBasin("fluid", (1, 100), 40)


@pytest.mark.parametrize("amplitude", [0., 1., 3.])
def test_independent_symmetric_binodal_and_lever_rule(amplitude):
    model = ExcessModel(amplitude)
    result = molecular_tp_flash(model, 300, 1e5, {"a": .6, "b": 1.4}, basins=[BASIN])
    assert result.success, result.to_dict()
    assert result.material_balance_error_mol < 1e-12
    assert result.pressure_relative_error < 2e-10
    json.dumps(result.to_dict(), allow_nan=False)
    if amplitude < 2:
        assert result.phase_status == "single_phase"
        assert result.phases[0].component_amounts_mol == {"a": .6, "b": 1.4}
        return
    endpoint = brentq(lambda x: log(x/(1-x))+amplitude*(1-2*x), 1e-8, .49)
    assert result.phase_status == "two_phase"
    assert result.phases[0].state.mole_fractions[0] == pytest.approx(endpoint, abs=2e-7)
    assert result.phases[1].state.mole_fractions[0] == pytest.approx(1-endpoint, abs=2e-7)
    assert result.phases[0].amount_mol == pytest.approx(
        2*(1-endpoint-.3)/(1-2*endpoint), abs=2e-6)
    assert result.chemical_potential_error_RT < 2e-7
    assert all(s.complete and not s.unstable for s in result.post_split_stability)
    assert sum(p.volume_m3 for p in result.phases) == pytest.approx(2*model.gas_constant_J_mol_K
                                                                *300/1e5, rel=1e-12)


@pytest.mark.parametrize("trace", [0., 1e-15])
def test_exact_zero_and_positive_trace_inventory(trace):
    result = molecular_tp_flash(ExcessModel(0), 300, 1e5, {"a": 1., "b": trace}, basins=[BASIN])
    assert result.success, result.to_dict()
    assert result.phases[0].component_amounts_mol["b"] == trace
    assert (result.phases[0].state.mole_fractions[1] > 0) == (trace > 0)


def test_phase_disappearance_below_binodal():
    result = molecular_tp_flash(ExcessModel(), 300, 1e5, {"a": .01, "b": .99}, basins=[BASIN])
    assert result.success and result.phase_status == "single_phase", result.to_dict()


@pytest.mark.parametrize("trace", [0., 1e-15])
def test_two_phase_absent_and_positive_trace_support(trace):
    model = ExcessModel(component_ids=("a", "b", "c"))
    result = molecular_tp_flash(model, 300, 1e5, {"a": .3, "b": .7, "c": trace}, basins=[BASIN])
    assert result.success and result.phase_status == "two_phase", result.to_dict()
    assert sum(p.component_amounts_mol["c"] for p in result.phases) == pytest.approx(trace,
                                                                                   rel=1e-12, abs=0)
    assert all((p.component_amounts_mol["c"] > 0) == (trace > 0) for p in result.phases)
    assert result.chemical_potential_error_RT < 2e-7


def test_unresolved_start_and_no_density_root_are_reported():
    failed = molecular_tp_flash(ExcessModel(), 300, 1e5, {"a": .3, "b": .7}, basins=[BASIN],
                                max_iterations=1)
    assert not failed.success and not failed.stability.complete
    assert any(not s["success"] for s in failed.stability.starts)
    missing = molecular_tp_flash(ExcessModel(), 300, 1e5, {"a": .3, "b": .7},
        basins=[MolecularDensityBasin("missing", (100, 200), 150)])
    assert not missing.success and missing.failure_reason == "no_reference_pressure_root"
    assert not missing.stability.root_attempts[0]["success"]


def test_invalid_declarations_and_absent_start_support():
    with pytest.raises(ValueError):
        MolecularDensityBasin("bad", (1, 100), 100)
    for feed in ({"alien": 1}, {"a": -1}, {"a": 0}, {"a": float("nan")}):
        with pytest.raises(ValueError):
            molecular_tp_flash(ExcessModel(), 300, 1e5, feed, basins=[BASIN])
    with pytest.raises(ValueError, match="support"):
        molecular_tp_stability(ExcessModel(), 300, 1e5, (1, 0), basins=[BASIN],
                                composition_starts=[(.5, .5)])


def cubic_model():
    R = 8.31446261815324
    components = []
    for name, tc, pc, omega in (("propane", 369.83, 4.248e6, .152),
                                ("butane", 425.12, 3.796e6, .2)):
        components.append(CPAComponent(name, .45724*R*R*tc*tc/pc, .07780*R*tc/pc, tc,
            .37464+1.54226*omega-.26992*omega*omega, "declared PR test parameters"))
    return CPAModel(tuple(components), ((0, 0), (0, 0)), (200, 500), "explicit binary PR",
                     cubic_model="peng_robinson")


def test_actual_cubic_vapor_liquid_split():
    model = cubic_model()
    basins = (MolecularDensityBasin("vapor", (1, 2000), 250),
              MolecularDensityBasin("liquid", (7500, 13500), 10000))
    result = molecular_tp_flash(model, 300, 6e5, {"propane": .5, "butane": .5}, basins=basins)
    assert result.success and result.phase_status == "two_phase", result.to_dict()
    assert {p.basin for p in result.phases} == {"vapor", "liquid"}
    vapor = next(p for p in result.phases if p.basin == "vapor")
    liquid = next(p for p in result.phases if p.basin == "liquid")
    assert vapor.state.mole_fractions[0] > liquid.state.mole_fractions[0]
    assert vapor.state.molar_density_mol_m3 < liquid.state.molar_density_mol_m3
    assert result.chemical_potential_error_RT < 2e-7
    assert result.material_balance_error_mol < 1e-12
    json.dumps(result.to_dict(), allow_nan=False)
