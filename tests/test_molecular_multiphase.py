"""General phase counts, simultaneous state pairs and exact material support."""

import json
from dataclasses import dataclass, replace
from math import log

import pytest
from scipy.optimize import brentq

from chemworld.physchem.caloric_math import logarithm
from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_mixture import MolecularCaloricModel
from chemworld.physchem.molecular_multiphase import (
    MolecularMultiphaseSeed,
    MolecularMultiphaseSpec,
    solve_molecular_multiphase,
)

R = 8.31446261815324


@dataclass(frozen=True)
class Regular:
    component_ids: tuple[str, ...] = ("a", "b", "c")
    omega: float = 11224.524534506875
    gas_constant_J_mol_K: float = R

    def alphar(self, temperature, density, fractions):
        return (
            self.omega
            / (R * temperature)
            * sum(
                fractions[i] * fractions[j]
                for i in range(len(fractions))
                for j in range(i + 1, len(fractions))
            )
        )


def regular_spec(omega=11224.524534506875):
    model = Regular(omega=omega)
    refs = tuple(
        HelmholtzFluid(
            k,
            "",
            0.04 + 0.01 * i,
            R,
            300.0,
            40.0,
            100.0,
            1000.0,
            1e8,
            (
                HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 80 / R - 1, "a2": -80 / R}),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 80 / R - 1}),
            ),
            (),
            "authored ideal Cp",
        )
        for i, k in enumerate(model.component_ids)
    )
    return MolecularMultiphaseSpec(
        MolecularCaloricModel(model, refs),
        (MolecularDensityBasin("fluid", (1.0, 500.0), 40.0),),
        (200.0, 700.0),
        (1e4, 1e6),
        phase_kind_by_basin={"fluid": "liquid"},
    )


def solve(spec=None, feed=None, **kwargs):
    return solve_molecular_multiphase(
        spec or regular_spec(),
        feed or {"a": 1.0, "b": 1.0, "c": 1.0},
        **({"inputs": {"T": 300.0, "P": 1e5}} | kwargs),
    )


def test_three_phase_tp_independent_symmetric_binodal_and_material():
    spec = regular_spec()
    result = solve(spec)
    assert result.success, result.to_dict()
    assert len(result.phases) == 3
    # Symmetric ternary coexistence: (r,s,s) and permutations, r+2s=1.
    a = spec.model.residual_model.omega / (R * 300)
    small = brentq(lambda s: log((1 - 2 * s) / s) - a * (1 - 3 * s), 1e-5, 0.1)
    for phase in result.phases:
        assert sorted(phase.state.mole_fractions) == pytest.approx(
            [small, small, 1 - 2 * small], abs=2e-7
        )
        assert phase.amount_mol == pytest.approx(1.0, abs=2e-7)
    assert result.material_balance_error_mol < 1e-14
    assert result.chemical_potential_error_RT < 1e-8
    assert [a["after"] for a in result.diagnostics["phase_updates"] if a["action"] == "add"] == [
        2,
        3,
    ]
    json.dumps(result.to_dict(), allow_nan=False)


def test_phase_limit_does_not_hide_remaining_instability():
    result = solve(replace(regular_spec(), max_phases=2))
    assert not result.success
    assert "instability" in result.failure_reason
    assert any(s.unstable for s in result.stability)


@pytest.mark.parametrize(
    "inputs",
    [
        {"P": 1e5, "H": "H"},
        {"P": 1e5, "S": "S"},
        {"U": "U", "V": "V"},
        {"T": 300.0, "V": "V"},
        {"P": 1e5, "mass_density": "mass_density"},
        {"P": 1e5, "h_mass": "h_mass"},
    ],
)
def test_simultaneous_state_pair_roundtrip(inputs):
    spec = regular_spec()
    anchor = solve(spec)
    assert anchor.success, anchor.to_dict()
    values = {
        "H": anchor.enthalpy_J,
        "S": anchor.entropy_J_K,
        "U": anchor.internal_energy_J,
        "V": anchor.volume_m3,
        "mass_density": anchor.mass_kg / anchor.volume_m3,
        "h_mass": anchor.enthalpy_J / anchor.mass_kg,
    }
    actual = solve(
        spec,
        inputs={k: values[v] if isinstance(v, str) else v for k, v in inputs.items()},
        seed=MolecularMultiphaseSeed(305.0, 1.01e5, anchor.phases),
    )
    assert actual.success, actual.to_dict()
    assert actual.temperature_K == pytest.approx(300.0, abs=2e-5)
    assert actual.pressure_Pa == pytest.approx(1e5, rel=2e-7)
    assert len(actual.phases) == 3
    assert actual.internal_energy_J == pytest.approx(
        actual.enthalpy_J - actual.pressure_Pa * actual.volume_m3, abs=1e-4
    )


def test_three_to_one_phase_with_changed_enthalpy():
    spec = regular_spec()
    low = solve(spec)
    high = solve(spec, inputs={"T": 650.0, "P": 1e5})
    assert low.success and high.success
    assert len(high.phases) == 1
    actual = solve(
        spec,
        inputs={"P": 1e5, "H": high.enthalpy_J},
        seed=MolecularMultiphaseSeed(300.0, 1e5, low.phases),
    )
    assert actual.success, actual.to_dict()
    assert len(actual.phases) == 1
    assert actual.temperature_K == pytest.approx(650.0, abs=1e-5)


@pytest.mark.parametrize("trace", [0.0, 1e-15])
def test_absent_and_positive_trace_are_not_floored(trace):
    feed = {"a": 1.0, "b": 1.0, "c": trace}
    result = solve(feed=feed)
    assert result.success, result.to_dict()
    assert sum(p.component_amounts_mol["c"] for p in result.phases) == pytest.approx(
        trace, abs=0, rel=1e-12
    )
    assert all((p.component_amounts_mol["c"] > 0) == (trace > 0) for p in result.phases)


def test_invalid_state_pairs_and_budget():
    for inputs in ({"V": 1.0, "rho": 2.0}, {"P": 1e5, "Q": 0.5}, {"T": 300.0, "Q": 0.0}):
        with pytest.raises(ValueError):
            solve(inputs=inputs)
    result = solve(replace(regular_spec(), maximum_phase_updates=1))
    assert not result.success and "phase_update_limit" in result.failure_reason


def test_real_cubic_quality_mass_codec_and_unqualified_trial():
    from reference.molecular28_caloric_cases import caloric_spec

    old = caloric_spec("PR")
    spec = MolecularMultiphaseSpec(
        old.model,
        old.basins,
        (280.0, 330.0),
        (4e5, 8e5),
        phase_kind_by_basin={"vapor": "gas", "liquid": "liquid"},
    )
    encoded = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
    restored = MolecularMultiphaseSpec.from_dict(encoded)
    assert restored.to_dict() == spec.to_dict()
    feed = {"propane": 0.6, "butane": 0.4}
    anchor = solve(restored, feed, inputs={"T": 300.0, "P": 6e5})
    assert anchor.success, anchor.to_dict()
    assert len(anchor.phases) == 2
    assert 0 < anchor.vapor_fraction < 1
    restored_state = solve(
        restored,
        feed,
        inputs={"P": 6e5, "Q": anchor.vapor_fraction},
        seed=MolecularMultiphaseSeed(301.0, 6e5, anchor.phases),
    )
    assert restored_state.success, restored_state.to_dict()
    assert restored_state.temperature_K == pytest.approx(300.0, abs=1e-5)
    assert restored_state.mass_kg == pytest.approx(0.6 * 0.04409562 + 0.4 * 0.0581222)
    trial = solve(
        restored,
        feed,
        inputs={"P": 6e5, "H": anchor.enthalpy_J},
        seed=MolecularMultiphaseSeed(301.0, 6e5, anchor.phases),
        verify_stability=False,
    )
    assert trial.success and not trial.stability_checked
    assert trial.to_dict()["scope"] == "unqualified_numerical_trial"
    assert not trial.stability
    json.dumps(restored_state.to_dict(), allow_nan=False)


def vlle_spec():
    gas = 8.314
    data = (
        ("methane", 190.564, 45.992e5, 0.01142, 0.016043),
        ("butane", 425.12, 37.96e5, 0.2, 0.0581222),
        ("water", 647.096, 220.64e5, 0.3443, 0.01801528),
    )
    residual = CPAModel(
        tuple(
            CPAComponent(
                k,
                0.4572355289213822 * gas * gas * tc * tc / pc,
                0.07779607390388846 * gas * tc / pc,
                tc,
                0.37464 + 1.54226 * w - 0.26992 * w * w,
                "explicit three-phase PR control",
            )
            for k, tc, pc, w, _ in data
        ),
        ((0.0, 0.0, 0.5), (0.0, 0.0, 0.5), (0.5, 0.5, 0.0)),
        (250.0, 500.0),
        "explicit same-equation three-phase control",
        cubic_model="peng_robinson",
        gas_constant_J_mol_K=gas,
    )
    refs = tuple(
        HelmholtzFluid(
            k,
            "",
            mass,
            gas,
            300.0,
            40.0,
            250.0,
            500.0,
            1e10,
            (
                HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 80 / gas - 1, "a2": -80 / gas}),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 80 / gas - 1}),
            ),
            (),
            "authored ideal Cp",
        )
        for k, _, _, _, mass in data
    )
    return MolecularMultiphaseSpec(
        MolecularCaloricModel(residual, refs),
        (
            MolecularDensityBasin("gas", (1.0, 7000.0), 1000.0),
            MolecularDensityBasin("hydrocarbon_liquid", (5000.0, 20000.0), 10000.0),
            MolecularDensityBasin("water_liquid", (25000.0, 52000.0), 47000.0),
        ),
        (250.0, 500.0),
        (1e5, 1e7),
        phase_kind_by_basin={
            "gas": "gas",
            "hydrocarbon_liquid": "liquid",
            "water_liquid": "liquid",
        },
    )


@pytest.mark.parametrize(
    "starts", [None, ((0.005, 0.005, 0.99), (0.1, 0.89, 0.01), (0.98, 0.01, 0.01))]
)
def test_actual_cubic_three_phase_has_different_density_domains(starts):
    result = solve(
        vlle_spec(),
        {"methane": 0.3, "butane": 0.3, "water": 0.4},
        inputs={"T": 300.0, "P": 3e6},
        composition_starts=starts,
    )
    assert result.success, result.to_dict()
    assert len(result.phases) == 3
    assert result.material_balance_error_mol < 1e-13
    assert result.chemical_potential_error_RT < 2e-8
    assert all(s.complete and not s.unstable for s in result.stability)


@dataclass(frozen=True)
class PureDoubleWell:
    component_ids: tuple[str, ...] = ("a",)
    gas_constant_J_mol_K: float = R

    def alphar(self, temperature, density, fractions):
        volume = 1 / density
        free = 1e12 * (volume - 0.01) ** 2 * (volume - 0.001) ** 2
        return (free - 1e5 * volume) / (R * temperature) - logarithm(density)


def test_pure_tp_coexistence_amount_remains_underdetermined():
    template = regular_spec()
    model = MolecularCaloricModel(PureDoubleWell(), template.model.ideal_references[:1])
    spec = replace(
        template,
        model=model,
        basins=(
            MolecularDensityBasin("gas", (50.0, 150.0), 100.0),
            MolecularDensityBasin("liquid", (800.0, 1200.0), 1000.0),
        ),
        phase_kind_by_basin={"gas": "gas", "liquid": "liquid"},
        molar_masses_kg_mol=None,
    )
    result = solve(spec, {"a": 1.0})
    assert not result.success
    assert result.failure_reason == "pure_coexistence_amount_underdetermined"


def test_single_gas_seed_can_cool_into_two_phases():
    from reference.molecular28_caloric_cases import caloric_spec

    old = caloric_spec("PR")
    spec = MolecularMultiphaseSpec(
        old.model,
        old.basins,
        (280.0, 330.0),
        (4e5, 8e5),
        phase_kind_by_basin={"vapor": "gas", "liquid": "liquid"},
    )
    feed = {"propane": 0.006, "butane": 0.004}
    hot = solve(spec, feed, inputs={"T": 320.0, "P": 6e5})
    cold = solve(spec, feed, inputs={"T": 300.0, "P": 6e5})
    assert hot.success and len(hot.phases) == 1
    assert cold.success and len(cold.phases) == 2
    actual = solve(
        spec,
        feed,
        inputs={"P": 6e5, "H": cold.enthalpy_J},
        seed=MolecularMultiphaseSeed(320.0, 6e5, hot.phases),
    )
    assert actual.success, actual.to_dict()
    assert len(actual.phases) == 2
    assert actual.temperature_K == pytest.approx(300.0, abs=1e-5)
