"""Query-private opposed numerical opt-ins on authored networks only."""

import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import replace
from math import fsum, inf, nextafter
from pathlib import Path

import numpy as np
import pytest

import chemworld.physchem.gas_opposed_flow as opposed
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.gas_polynomial_transport import GasTransportPolynomialSpec
from chemworld.physchem.gas_query_thermo import GasQueryThermoMemo
from chemworld.physchem.gas_transport import GasTransportSpec
from chemworld.physchem.lennard_jones import LennardJonesSpec
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.residual_sparse_difference import sparse_finite_jacobian
from chemworld.physchem.residual_system import ResidualDomainError, bounds, finite_jacobian
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.thermochemistry import NASA7SpeciesThermo

FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures/gas_query36_legacy"
_spec = importlib.util.spec_from_file_location(
    "chemworld_test_legacy_opposed36", FIXTURE_DIRECTORY / "gas_opposed_flow.py"
)
baseline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = baseline
_spec.loader.exec_module(baseline)
_authored_spec = importlib.util.spec_from_file_location(
    "chemworld_test_authored_opposed36", FIXTURE_DIRECTORY / "authored_opposed.py"
)
_authored_fixture = importlib.util.module_from_spec(_authored_spec)
sys.modules[_authored_spec.name] = _authored_fixture
_authored_spec.loader.exec_module(_authored_fixture)
non_nasa_case = _authored_fixture.authored
OPTS = ((64, False), (None, True), (64, True))


def authored_case(*, size=2, reacting=False, cells=5, model="mixture-averaged", kinetics=None):
    species = [
        SpeciesSpec("A", "H2", phase="gas"),
        SpeciesSpec("B", "H2" if reacting else "O2", phase="gas"),
    ]
    if size == 3:
        species.append(SpeciesSpec("C", "He", phase="gas"))
    declarations = {}
    for i, item in enumerate(species):
        coeff = [4.0 + 0.3 * i, 0, 0, 0, 0, -1000.0 * i / R, 2.0]
        declarations[item.species_id] = {
            "thermo": {
                "model": "NASA7",
                "temperature-ranges": [300.0, 1000.0, 2000.0],
                "data": [coeff, coeff.copy()],
                "note": "Authored synthetic query fixture",
            },
            "composition": item.composition,
            "reference_pressure_Pa": 101325.0,
            "units": {},
        }
    reactions = ()
    if reacting:
        params = {"k": 0.1}
        law, equation = "mass_action", "A -> B"
        if kinetics is not None:
            params = {
                "mechanism_schema": "cantera-gas-v2",
                "reference_pressure_Pa": 101325.0,
                "reverse_basis": "ideal_gas_thermo",
                "default_efficiency": 0.0,
                "third_body_efficiencies": {"A": 2.0, "B": 0.5},
            }
            if kinetics == "third_body_arrhenius":
                params.update(A=2.0, b=0.2, Ea_J_per_mol=100.0)
            else:
                params.update(
                    low_A=30.0,
                    high_A=10.0,
                    troe_a=0.5,
                    troe_T3=1000.0,
                    troe_T1=10000.0,
                    troe_T2=5000.0,
                )
            law, equation = kinetics, "A <=> B"
        reactions = (
            ReactionSpec.from_equation(
                reaction_id="isomer",
                equation=equation,
                rate_law=RateLawSpec("authored-opposed-" + law, law, params),
            ),
        )
    network = ReactionNetworkSpec(
        "authored-opposed-query36",
        tuple(species),
        reactions,
        metadata={"gas_thermochemistry": declarations},
    )
    ids = network.species_ids
    masses = {
        s.species_id: ComponentSpec(s.species_id, s.formula).molecular_weight_g_mol / 1000
        for s in species
    }
    transport = GasTransportPolynomialSpec(
        ids,
        {s.species_id: s.to_dict() for s in species},
        declarations,
        masses,
        {s: (0.002 + i * 0.0001, 0, 0, 0, 0) for i, s in enumerate(ids)},
        dict.fromkeys(ids, (0.001, 0, 0, 0, 0)),
        {s: dict.fromkeys(ids, (0.0001, 0, 0, 0, 0)) for s in ids},
        DataSourceProvenance(
            "authored-opposed-query36", 0, "Authored synthetic", "v1", "CC0", "0" * 64
        ),
        "authored-synthetic",
        (300.0, 1200.0),
        200000.0,
        "off",
    )
    x = {"A": 0.8, "B": 0.2} if size == 2 else {"A": 0.7, "B": 0.2, "C": 0.1}
    mesh = DiffusionMesh("planar", tuple(np.linspace(0, 1, cells + 1) ** 1.2 * 0.02), area_m2=0.023)
    rho = 1e5 / (R * 600.0) * fsum(x[s] * masses[s] for s in ids)
    mdot = rho * 2 * mesh.faces_m[-1]
    spec = opposed.GasOpposedFlowSpec(
        mesh,
        transport,
        1e5,
        600.0,
        600.0,
        x,
        x,
        mdot,
        mdot,
        ids[-1],
        (300.0, 1200.0),
        2.0,
        2.0,
        model,
    )
    initial = opposed.opposed_flow_initial_profile(
        spec,
        network,
        temperatures_K=[600.0] * cells,
        mole_fractions=[x] * cells,
        spread_rates_per_s=[2.0] * cells,
        radial_pressure_gradient_Pa_m2=-rho * 4,
    )
    return spec, network, initial


def assembly(spec, network, initial, cap=None, maximum=20000, progress=None):
    return opposed._OpposedResidual(
        spec, network, initial, maximum, progress, thermo_memo_max_entries=cap
    )


def stripped(report):
    return {
        k: v
        for k, v in report.items()
        if k not in {"thermo_memo_statistics", "residual_only_trials"}
    }


def varied(spec, network):
    spec = replace(
        spec,
        left_inlet_temperature_K=450.0,
        right_inlet_temperature_K=900.0,
        left_spread_rate_per_s=1.5,
        right_spread_rate_per_s=-0.8,
    )
    fractions = []
    for i in range(spec.mesh.cells):
        a = 0.75 - 0.06 * i
        fractions.append(
            {"A": a, "B": 1 - a} if len(network.species) == 2 else {"A": a, "B": 0.9 - a, "C": 0.1}
        )
    initial = opposed.opposed_flow_initial_profile(
        spec,
        network,
        temperatures_K=[590.0, nextafter(1000.0, -inf), 1000.0, nextafter(1000.0, inf), 710.0],
        mole_fractions=fractions,
        spread_rates_per_s=[1.0, 2.0, -1.0, 3.0, -0.4],
        face_mass_fluxes_kg_m2_s=[0.001, 0.0007, 0.0002, -0.0001, -0.0006, -0.001],
        radial_pressure_gradient_Pa_m2=-0.01,
    )
    return spec, initial


@pytest.mark.parametrize("size", (2, 3))
@pytest.mark.parametrize("model", ("mixture-averaged", "stefan-maxwell"))
def test_defaults_preserve_complete_baseline_report_and_validator_once(size, model, monkeypatch):
    spec, network, initial = authored_case(size=size, model=model)
    oldspec = baseline.GasOpposedFlowSpec.from_dict(spec.to_dict())
    expected = baseline.solve_opposed_gas_flow(oldspec, network, initial)
    original = deepcopy((spec.to_dict(), network.to_dict(), initial))
    calls = []
    validate = GasTransportPolynomialSpec.validate_network

    def counted(self, private):
        calls.append(private)
        return validate(self, private)

    def forbidden(*_args, **_kwargs):
        pytest.fail("default must not construct memo owner")

    monkeypatch.setattr(GasTransportPolynomialSpec, "validate_network", counted)
    monkeypatch.setattr(GasQueryThermoMemo, "__init__", forbidden)
    actual = opposed.solve_opposed_gas_flow(
        spec, network, initial, thermo_memo_max_entries=None, residual_only_trials=False
    )
    assert actual == expected and actual["success"]
    assert len(actual["physical_checks"]) == 18 and all(actual["physical_checks"].values())
    assert len(calls) == 1 and calls[0] is not network
    assert (spec.to_dict(), network.to_dict(), initial) == original


@pytest.mark.parametrize("size", (2, 3))
@pytest.mark.parametrize("model", ("mixture-averaged", "stefan-maxwell"))
@pytest.mark.parametrize(("cap", "light"), OPTS)
def test_all_options_keep_complete_nonuniform_signed_f_and_fresh_profile_exact(
    size, model, cap, light
):
    spec, network, _ = authored_case(size=size, reacting=True, model=model)
    spec, initial = varied(spec, network)
    original = deepcopy((spec.to_dict(), network.to_dict(), initial))
    old = baseline._OpposedResidual(
        baseline.GasOpposedFlowSpec.from_dict(spec.to_dict()), network, initial, 20000, None
    )
    expected, details = old.evaluate(old.initial)
    unit = assembly(spec, network, initial, cap)
    actual, trial = unit.evaluate(unit.initial, retain_details=not light)
    np.testing.assert_array_equal(actual, expected)
    assert trial is None if light else trial == details
    fresh, profile = unit.evaluate(unit.initial)
    np.testing.assert_array_equal(fresh, expected)
    assert profile == details
    assert unit.variables == old.variables and unit.equations == old.equations
    assert (
        profile["face_mass_fluxes_kg_m2_s"][0] > 0 and profile["face_mass_fluxes_kg_m2_s"][-1] < 0
    )
    assert min(profile["spread_rates_per_s"]) < 0 < max(profile["spread_rates_per_s"])
    assert min(profile["radial_mass_outflows_kg_s"]) < 0 < max(profile["radial_mass_outflows_kg_s"])
    assert profile["pressure_momentum_terms_kg_s2"]
    assert any(v != 0 for row in profile["chemical_sources_mol_s"] for v in row.values())
    for face, temperature in ((0, 450.0), (-1, 900.0)):
        point = unit.response(
            temperature, profile[("left" if face == 0 else "right") + "_inlet_trace_mole_fractions"]
        )
        part = profile["face_diffusive_fluxes"][face]
        h = fsum(
            point.partial_enthalpies_J_mol[s] * part["molar_fluxes_mol_m2_s"][s]
            for s in network.species_ids
        )
        assert part["diffusion_enthalpy_flux_W_m2"] == h
        assert (
            profile["face_transport_and_advective_states"][face]["advective_temperature_K"]
            == temperature
        )
    assert (
        profile["global_enthalpy_balance_residual_W"]
        == details["global_enthalpy_balance_residual_W"]
    )
    assert (spec.to_dict(), network.to_dict(), initial) == original


@pytest.mark.parametrize("size", (2, 3))
@pytest.mark.parametrize("model", ("mixture-averaged", "stefan-maxwell"))
@pytest.mark.parametrize(("cap", "light"), OPTS)
def test_dense_and_colored_full_jacobians_match_original_including_signed_g_v_lambda(
    size, model, cap, light
):
    spec, network, _ = authored_case(size=size, reacting=True, model=model)
    spec, initial = varied(spec, network)
    old = baseline._OpposedResidual(
        baseline.GasOpposedFlowSpec.from_dict(spec.to_dict()), network, initial, 20000, None
    )
    unit = assembly(spec, network, initial, cap=cap)
    scales = [v.scale for v in old.variables]
    lower, upper = bounds(old.variables)
    expected = finite_jacobian(
        lambda y: old.evaluate(y)[0], old.initial, scales, lower=lower, upper=upper
    )

    def function(y):
        return unit.evaluate(y, retain_details=not light)[0]

    dense = finite_jacobian(function, unit.initial, scales, lower=lower, upper=upper)
    colored = sparse_finite_jacobian(
        function, unit.initial, scales, lower=lower, upper=upper, sparsity=unit.state_sparsity()
    )
    np.testing.assert_array_equal(dense, expected)
    np.testing.assert_array_equal(colored.toarray(), expected)
    np.testing.assert_array_equal(
        np.flatnonzero(dense[:, unit.lambda_col]),
        np.arange(unit.n_stop + unit.count, unit.n_stop + 2 * unit.count),
    )


@pytest.mark.parametrize("size", (2, 3))
@pytest.mark.parametrize("strategy", ("dense", "sparse-colored"))
@pytest.mark.parametrize(("cap", "light"), OPTS)
def test_shared_authored_root_and_complete_18_gate_report_stay_exact(size, strategy, cap, light):
    spec, network, initial = authored_case(size=size)
    initial["temperatures_K"] = [601.0, 602.0, 601.0, 599.0, 600.5]
    initial["spread_rates_per_s"] = [1.8, 2.2, 1.9, 2.1, 1.95]
    initial["radial_pressure_gradient_Pa_m2"] *= 1.1
    oldspec = baseline.GasOpposedFlowSpec.from_dict(spec.to_dict())
    expected = baseline.solve_opposed_gas_flow(
        oldspec, network, initial, jacobian_strategy=strategy
    )
    actual = opposed.solve_opposed_gas_flow(
        spec,
        network,
        initial,
        jacobian_strategy=strategy,
        thermo_memo_max_entries=cap,
        residual_only_trials=light,
    )
    assert expected["success"] and actual["success"]
    assert stripped(actual) == expected
    assert len(actual["physical_checks"]) == 18 and all(actual["physical_checks"].values())
    assert actual["experimental_elapsed_s"] == actual["live_operations_executed"] == 0
    assert actual["numerical_derivatives"] == "shared finite_difference"
    assert actual["derivative_diagnostics"]["derivative_mode"] == "finite_difference"
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize(("cap", "light"), OPTS)
def test_missing_chemistry_ledgers_on_trials_cannot_hide_omitted_species_failure(
    monkeypatch, cap, light
):
    spec, network, initial = authored_case()

    def faulty(self, amounts, *, volume_L, temperature_K, species_thermo=None):
        return {"A": 0.0, "B": 1e-5}

    monkeypatch.setattr(ReactionNetworkSpec, "amount_derivatives", faulty)
    actual = opposed.solve_opposed_gas_flow(
        spec, network, initial, thermo_memo_max_entries=cap, residual_only_trials=light
    )
    assert not actual["success"] and actual["profile"] is None
    assert len(actual["physical_checks"]) == 18
    profile = actual["candidate_profile"]
    assert not actual["physical_checks"]["chemical_mass"]
    assert not actual["physical_checks"]["chemical_elements"]
    assert not actual["physical_checks"]["global_chemical_mass"]
    assert not actual["physical_checks"]["global_chemical_elements"]
    assert not actual["physical_checks"]["all_species_including_omitted_balance"]
    assert all(
        key in profile
        for key in (
            "global_mass_balance_residual_kg_s",
            "global_mass_pde_balance_residual_kg_s",
            "global_chemical_mass_residual_kg_s",
            "global_element_balance_residuals_mol_atoms_s",
            "global_element_pde_balance_residuals_mol_atoms_s",
            "global_chemical_element_residuals_mol_atoms_s",
            "global_enthalpy_balance_residual_W",
        )
    )


@pytest.mark.parametrize("cap", (True, False, 0, -1, 1.0, "8", np.int64(8)))
def test_capacity_type_validation_is_explicit_and_does_not_coerce(cap):
    spec, network, initial = authored_case()
    with pytest.raises(ValueError, match="positive Python int"):
        opposed.solve_opposed_gas_flow(spec, network, initial, thermo_memo_max_entries=cap)


@pytest.mark.parametrize("flag", (None, 0, 1, "false", np.bool_(True)))
def test_trial_flag_type_is_explicit(flag):
    spec, network, initial = authored_case()
    with pytest.raises(ValueError, match="explicitly boolean"):
        opposed.solve_opposed_gas_flow(spec, network, initial, residual_only_trials=flag)


@pytest.mark.parametrize("light", (False, True))
def test_non_nasa_legacy_capability_survives_and_only_memo_refuses(light):
    spec, network, initial = non_nasa_case()
    expected = baseline.solve_opposed_gas_flow(
        baseline.GasOpposedFlowSpec.from_dict(spec.to_dict()), network, initial
    )
    actual = opposed.solve_opposed_gas_flow(spec, network, initial, residual_only_trials=light)
    assert actual["success"] and stripped(actual) == expected
    with pytest.raises(ValueError, match="exactly ID-bound NASA7"):
        opposed.solve_opposed_gas_flow(
            spec, network, initial, thermo_memo_max_entries=16, residual_only_trials=light
        )


@pytest.mark.parametrize("light", (False, True))
def test_lj_default_and_light_remain_original_while_memo_refuses(light):
    spec, network, initial = authored_case()
    transport = GasTransportSpec(
        (
            LennardJonesSpec("A", 120.0, 3.2e-10, "Authored LJ"),
            LennardJonesSpec("B", 160.0, 3.5e-10, "Authored LJ"),
        ),
        {"A": "linear", "B": "linear"},
        "off",
    )
    spec = replace(spec, transport=transport)
    oldspec = baseline.GasOpposedFlowSpec.from_dict(spec.to_dict())
    expected = baseline.solve_opposed_gas_flow(oldspec, network, initial)
    actual = opposed.solve_opposed_gas_flow(spec, network, initial, residual_only_trials=light)
    assert actual["success"] and stripped(actual) == expected
    with pytest.raises(ValueError, match="currently requires polynomial transport"):
        opposed.solve_opposed_gas_flow(
            spec, network, initial, thermo_memo_max_entries=16, residual_only_trials=light
        )


def test_one_shared_owner_serves_both_feed_preflight_transport_chemistry_scales_and_final_mach(
    monkeypatch,
):
    spec, network, initial = authored_case(size=3)
    instances, consumers, post_fresh = [], [], []
    initialize = opposed._OpposedResidual.__init__
    evaluate = opposed._OpposedResidual.evaluate
    derivatives = ReactionNetworkSpec.amount_derivatives

    def counted_init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        instances.append(self)

    def counted_chemistry(self, *args, **kwargs):
        consumers.append((self, kwargs.get("species_thermo"), kwargs["temperature_K"]))
        return derivatives(self, *args, **kwargs)

    def counted_evaluate(self, vector, *, retain_details=True):
        result = evaluate(self, vector, retain_details=retain_details)
        if retain_details:
            post_fresh.append(self.thermo_memo.statistics())
        return result

    monkeypatch.setattr(opposed._OpposedResidual, "__init__", counted_init)
    monkeypatch.setattr(opposed._OpposedResidual, "evaluate", counted_evaluate)
    monkeypatch.setattr(ReactionNetworkSpec, "amount_derivatives", counted_chemistry)
    actual = opposed.solve_opposed_gas_flow(
        spec, network, initial, thermo_memo_max_entries=16, residual_only_trials=True
    )
    assert actual["success"] and len(instances) == 1 and len(post_fresh) == 1
    unit = instances[0]
    assert unit.bound_response.args[2] is unit.thermo_views
    assert (
        consumers[0][2] == spec.left_inlet_temperature_K
        and consumers[1][2] == spec.right_inlet_temperature_K
    )
    assert all(
        private is unit.network and views is unit.thermo_views for private, views, _ in consumers
    )
    assert actual["thermo_memo_statistics"] == unit.thermo_memo.statistics()
    assert actual["thermo_memo_statistics"]["requests"] > post_fresh[0]["requests"]
    assert all(
        type(m) is NASA7SpeciesThermo for m in unit.network.imported_gas_thermochemistry.values()
    )
    assert not any("memo" in name or "view" in name for name in unit.network.__dict__)


def test_private_complete_snapshot_remains_bound_against_caller_mutation(monkeypatch):
    spec, network, initial = authored_case(size=3)
    expected = opposed.solve_opposed_gas_flow(
        spec, network, initial, thermo_memo_max_entries=64, residual_only_trials=True
    )
    changed = []

    def progress(_event):
        if not changed:
            network.metadata["gas_thermochemistry"]["A"]["thermo"]["data"][0][0] = 9.0
            network.imported_gas_thermochemistry.clear()
            changed.append(True)

    actual = opposed.solve_opposed_gas_flow(
        spec,
        network,
        initial,
        thermo_memo_max_entries=64,
        residual_only_trials=True,
        progress=progress,
    )
    assert actual == expected and changed


@pytest.mark.parametrize("law", ("third_body_arrhenius", "troe_falloff"))
def test_only_thermo_is_cached_while_composition_and_collider_rates_change(law):
    spec, network, initial = authored_case(reacting=True, size=3, kinetics=law)
    unit = assembly(spec, network, initial, cap=16)
    rows = (
        {"A": 0.03, "B": 0.01, "C": 0.01},
        {"A": 0.01, "B": 0.03, "C": 0.01},
        {"A": 0.06, "B": 0.02, "C": 0.02},
    )
    rates = []
    for row in rows:
        args = {"volume_L": 1.0, "temperature_K": 800.0}
        plain = unit.network.reaction_rates(row, **args)
        memo = unit.network.reaction_rates(row, **args, species_thermo=unit.thermo_views)
        assert memo == plain
        assert unit.network.amount_derivatives(
            row, **args, species_thermo=unit.thermo_views
        ) == unit.network.amount_derivatives(row, **args)
        rates.append(memo["isomer"])
    assert len(set(rates)) == len(rows)
    assert unit.thermo_memo.statistics()["hits"] > 0


def test_bounded_total_capacity_evicts_without_state_rounding_or_shared_query_cache():
    spec, network, initial = authored_case(size=3)
    first, second = assembly(spec, network, initial, cap=2), assembly(spec, network, initial, cap=2)
    other_stats = second.thermo_memo.statistics()
    for temperature in (700.0, nextafter(1000.0, -inf), 1000.0, nextafter(1000.0, inf)):
        first.response(temperature, spec.left_inlet_mole_fractions)
    stats = first.thermo_memo.statistics()
    assert stats["current_entries"] <= 2 and stats["peak_entries"] <= 2 and stats["evictions"] > 0
    assert second.thermo_memo.statistics() == other_stats
    plain = assembly(spec, network, initial)
    expected = plain.evaluate(plain.initial)
    actual = first.evaluate(first.initial)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


def test_original_thermo_failure_is_not_swallowed_or_cached(monkeypatch):
    spec, network, initial = authored_case()
    unit = assembly(spec, network, initial, cap=16)
    error = RuntimeError("authored NASA error")
    calls = []
    original = NASA7SpeciesThermo.evaluate

    def changed(self, temperature):
        if temperature == 777.0:
            calls.append(temperature)
            raise error
        return original(self, temperature)

    monkeypatch.setattr(NASA7SpeciesThermo, "evaluate", changed)
    before = unit.thermo_memo.statistics()
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            unit.response(777.0, spec.left_inlet_mole_fractions)
        assert caught.value is error
    after = unit.thermo_memo.statistics()
    assert len(calls) == 2 and after["misses"] == before["misses"] + 2
    assert after["current_entries"] == before["current_entries"]


@pytest.mark.parametrize(
    "kind", ("negative-amount", "empty-trace", "temperature", "overflow-V", "overflow-G")
)
@pytest.mark.parametrize("light", (False, True))
def test_original_residual_domain_guards_and_error_messages_stay_exact(kind, light):
    spec, network, initial = authored_case()
    old = baseline._OpposedResidual(
        baseline.GasOpposedFlowSpec.from_dict(spec.to_dict()), network, initial, 20000, None
    )
    unit = assembly(spec, network, initial, cap=16)
    point = old.initial.copy()
    if kind == "negative-amount":
        point[0] = -1.0
    elif kind == "empty-trace":
        point[old.trace_start : old.trace_start + old.size] = 0.0
    elif kind == "temperature":
        point[old.t_start] = 1201.0
    elif kind == "overflow-V":
        point[old.v_start] = 1.7e308
    else:
        point[old.g_start] = 1.7e308
    with pytest.raises(ResidualDomainError) as expected:
        old.evaluate(point)
    with pytest.raises(ResidualDomainError) as actual:
        unit.evaluate(point, retain_details=not light)
    assert actual.value.args == expected.value.args


@pytest.mark.parametrize(("cap", "light"), OPTS)
def test_root_and_fresh_budget_failures_preserve_raw_root_and_reason(cap, light):
    spec, network, initial = authored_case()
    oldspec = baseline.GasOpposedFlowSpec.from_dict(spec.to_dict())
    for exact in (False, True):
        guess = deepcopy(initial)
        if not exact:
            guess["spread_rates_per_s"] = [1.7] * spec.mesh.cells
        expected = baseline.solve_opposed_gas_flow(oldspec, network, guess, maximum_evaluations=1)
        actual = opposed.solve_opposed_gas_flow(
            spec,
            network,
            guess,
            maximum_evaluations=1,
            thermo_memo_max_entries=cap,
            residual_only_trials=light,
        )
        assert stripped(actual) == expected and not actual["success"]
        assert actual["profile"] is None and actual["initial_value"]
        if exact:
            assert actual["root_value"] and actual["failure_reason"].startswith(
                "fresh_check_failed"
            )


@pytest.mark.parametrize(("cap", "light"), OPTS)
def test_original_root_exception_is_preserved_without_retry_or_candidate_substitution(
    monkeypatch, cap, light
):
    spec, network, initial = authored_case()
    error = RuntimeError("authored original root failure")
    calls = []

    def failed(*args, **kwargs):
        calls.append((args, kwargs))
        raise error

    monkeypatch.setattr(opposed, "solve_steady_residual", failed)
    actual = opposed.solve_opposed_gas_flow(
        spec, network, initial, thermo_memo_max_entries=cap, residual_only_trials=light
    )
    assert len(calls) == 1 and not actual["success"]
    assert actual["failure_reason"] == "root_exception"
    assert actual["root_exception"]["message"] == str(error)
    assert actual["root_exception"]["type"] == "builtins.RuntimeError"
    assert "raise error" in actual["root_exception"]["traceback"]
    assert actual["root_value"] is None and actual["profile"] is None
    assert "candidate_profile" not in actual and actual["initial_value"]


def test_light_trials_skip_only_profile_serialization_while_full_fresh_still_requests_it(
    monkeypatch,
):
    from chemworld.physchem.gas_transport_fluxes import GasTransportFluxResult

    spec, network, initial = authored_case(reacting=True)
    spec, initial = varied(spec, network)
    unit = assembly(spec, network, initial, cap=16)
    expected = unit.evaluate(unit.initial)[0]

    def forbidden(_self):
        raise RuntimeError("diagnostic serialization called")

    monkeypatch.setattr(GasTransportFluxResult, "to_dict", forbidden)
    actual, details = unit.evaluate(unit.initial, retain_details=False)
    np.testing.assert_array_equal(actual, expected)
    assert details is None
    with pytest.raises(RuntimeError, match="diagnostic serialization"):
        unit.evaluate(unit.initial)


def test_distinct_feed_preflights_and_both_cp_scales_use_the_same_exact_temperature_owner(
    monkeypatch,
):
    spec, network, _ = authored_case(size=3, reacting=True, kinetics="troe_falloff")
    spec, initial = varied(spec, network)
    consumers = []
    original = ReactionNetworkSpec.amount_derivatives

    def counted(self, *args, **kwargs):
        consumers.append((self, kwargs.get("species_thermo"), kwargs["temperature_K"]))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ReactionNetworkSpec, "amount_derivatives", counted)
    unit = assembly(spec, network, initial, cap=64)
    assert [record[2] for record in consumers] == [450.0, 900.0]
    assert all(
        private is unit.network and views is unit.thermo_views for private, views, _ in consumers
    )
    # Two three-species transport states, two A/B reverse-thermo preflights,
    # then two three-species Cp scales, all through this single owner.
    assert unit.thermo_memo.statistics()["requests"] == 16
    assert unit.thermo_memo.statistics()["misses"] == 6
    assert unit.thermo_memo.statistics()["hits"] == 10
