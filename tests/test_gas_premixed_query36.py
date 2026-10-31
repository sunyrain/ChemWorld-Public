"""Authored query compatibility; no material, Cantera or saved source."""

import importlib.util
import json
import sys
from copy import deepcopy
from dataclasses import replace
from math import inf, nextafter
from pathlib import Path

import numpy as np
import pytest

import chemworld.physchem.gas_premixed_flow as flow
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.gas_polynomial_transport import GasTransportPolynomialSpec
from chemworld.physchem.gas_query_thermo import GasQueryThermoMemo
from chemworld.physchem.gas_transport import gas_transport_response
from chemworld.physchem.reaction_network import (
    R_J_PER_MOL_K as R,
)
from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.residual_sparse_difference import sparse_finite_jacobian
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    RootPolicy,
    bounds,
    finite_jacobian,
)
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.thermochemistry import NASA7SpeciesThermo

FIXTURE_DIRECTORY = Path(__file__).resolve().parent / "fixtures/gas_query36_legacy"
BASELINE_PATH = FIXTURE_DIRECTORY / "gas_premixed_flow.py"
_spec = importlib.util.spec_from_file_location("chemworld_test_legacy_premixed36", BASELINE_PATH)
baseline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = baseline
_spec.loader.exec_module(baseline)


def authored_case(
    *, nasa=True, energy=False, cells=4, reacting=True, advection="upwind", kinetics=None
):
    species = tuple(SpeciesSpec(s, "H2", phase="gas") for s in ("A", "B"))
    if kinetics is None:
        law = RateLawSpec("authored-isomer", "mass_action", {"k": 2.0})
        equation = "A -> B"
    else:
        params = {
            "mechanism_schema": "cantera-gas-v2",
            "reference_pressure_Pa": 101325.0,
            "reverse_basis": "ideal_gas_thermo",
            "third_body_efficiencies": {"A": 2.0, "B": 0.5},
            "default_efficiency": 0.0,
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
        law = RateLawSpec("authored-reversible-" + kinetics, kinetics, params)
        equation = "A <=> B"
    reaction = ReactionSpec.from_equation(reaction_id="isomer", equation=equation, rate_law=law)
    declarations = {}
    for species_spec in species:
        s = species_spec.species_id
        heat = 0.0 if s == "A" else -9000.0
        if nasa:
            # Authored isomers; identical constant Cp and explicit chemical H offset.
            coeff = [4.0, 0.0, 0.0, 0.0, 0.0, heat / R, 2.0]
            thermo = {
                "model": "NASA7",
                "temperature-ranges": [300.0, 1000.0, 2000.0],
                "data": [coeff, coeff.copy()],
                "note": "authored synthetic, not a real material fit",
            }
        else:
            thermo = {
                "model": "constant-cp",
                "T0": 600.0,
                "h0": f"{heat} J/mol",
                "s0": "30 J/mol/K",
                "cp0": "30 J/mol/K",
                "T-min": 200.0,
                "T-max": 2000.0,
            }
        declarations[s] = {
            "thermo": thermo,
            "composition": species_spec.composition,
            "reference_pressure_Pa": 101325.0,
            "units": {},
        }
    network = ReactionNetworkSpec(
        "isolated-authored-candidate36",
        species,
        (reaction,) if reacting else (),
        metadata={"gas_thermochemistry": declarations},
    )
    ids = network.species_ids
    transport = GasTransportPolynomialSpec(
        ids,
        {s.species_id: s.to_dict() for s in species},
        declarations,
        {
            s.species_id: ComponentSpec(s.species_id, s.formula).molecular_weight_g_mol / 1000
            for s in species
        },
        dict.fromkeys(ids, (0.002, 0, 0, 0, 0)),
        dict.fromkeys(ids, (0.001, 0, 0, 0, 0)),
        {s: dict.fromkeys(ids, (0.0001, 0, 0, 0, 0)) for s in ids},
        DataSourceProvenance(
            "authored36", 0, "Authored synthetic coefficients", "v1", "CC0", "0" * 64
        ),
        "authored-synthetic",
        (300.0, 1200.0),
        200000.0,
        "off",
    )
    spec = flow.GasPremixedFlowSpec(
        DiffusionMesh.planar(0.02, cells, area_m2=0.023),
        transport,
        100000.0,
        600.0,
        {"A": 0.8, "B": 0.2},
        "B",
        (300.0, 1200.0),
        energy,
        0.005,
        advection_scheme=advection,
    )
    initial = flow.premixed_flow_initial_profile(
        spec,
        network,
        temperatures_K=[600.0] * cells,
        mole_fractions=[dict(spec.inlet_mole_fractions)] * cells,
    )
    return spec, network, initial


def assembly(spec, network, initial, cap=None, progress=None):
    return flow._PremixedResidual(
        spec,
        network,
        initial,
        20000,
        progress,
        thermo_memo_max_entries=cap,
    )


def stripped(report):
    return {
        k: v
        for k, v in report.items()
        if k not in {"thermo_memo_statistics", "residual_only_trials"}
    }


def test_query_imports_use_one_selected_source_or_installed_package():
    import chemworld

    package_root = Path(chemworld.__file__).resolve().parent
    assert Path(flow.__file__).resolve().is_relative_to(package_root)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name == "chemworld" or name.startswith("chemworld."):
            assert path is not None and Path(path).resolve().is_relative_to(package_root), (
                name,
                path,
            )
    # A combined suite may already have imported an optional backend. These
    # authored tests never call it; its global presence is not a query contract.


@pytest.mark.parametrize("nasa", (False, True))
@pytest.mark.parametrize("energy", (False, True))
@pytest.mark.parametrize("advection", ("upwind", "linear-upwind"))
def test_disabled_defaults_preserve_full_old_spec_profile_report_and_validation(
    nasa, energy, advection, monkeypatch
):
    spec, network, initial = authored_case(nasa=nasa, energy=energy, advection=advection)
    old_spec = baseline.GasPremixedFlowSpec.from_dict(spec.to_dict())
    old_network = ReactionNetworkSpec.from_dict(deepcopy(network.to_dict()))
    before = deepcopy((spec.to_dict(), network.to_dict(), initial))
    expected = baseline.solve_premixed_gas_flow(old_spec, old_network, initial)
    validations = []
    original_validate = GasTransportPolynomialSpec.validate_network

    def validate(self, bound):
        validations.append(bound)
        return original_validate(self, bound)

    def forbidden(*_args, **_kwargs):
        pytest.fail("disabled query must not construct a memo")

    monkeypatch.setattr(GasTransportPolynomialSpec, "validate_network", validate)
    monkeypatch.setattr(GasQueryThermoMemo, "__init__", forbidden)
    result = flow.solve_premixed_gas_flow(
        spec,
        network,
        initial,
        thermo_memo_max_entries=None,
        residual_only_trials=False,
    )
    assert result == expected and result["success"]
    assert before == (spec.to_dict(), network.to_dict(), initial)
    assert len(validations) == 1 and validations[0] is not network
    assert result["numerical_derivatives"] == "shared finite_difference"
    assert "thermo_memo_statistics" not in result and "residual_only_trials" not in result


@pytest.mark.parametrize("cap", (True, False, 0, -1, 1.0, "8", np.int64(8)))
def test_explicit_capacity_validation_does_not_coerce(cap):
    spec, network, initial = authored_case()
    with pytest.raises(ValueError, match="positive Python int"):
        flow.solve_premixed_gas_flow(spec, network, initial, thermo_memo_max_entries=cap)


@pytest.mark.parametrize("flag", (None, 0, 1, "false", np.bool_(True)))
def test_residual_only_flag_requires_exact_boolean(flag):
    spec, network, initial = authored_case()
    with pytest.raises(ValueError, match="explicitly boolean"):
        flow.solve_premixed_gas_flow(spec, network, initial, residual_only_trials=flag)


def test_non_nasa_is_rejected_only_when_opted_in():
    spec, network, initial = authored_case(nasa=False)
    assert flow.solve_premixed_gas_flow(spec, network, initial)["success"]
    with pytest.raises(ValueError, match="exactly ID-bound NASA7"):
        flow.solve_premixed_gas_flow(spec, network, initial, thermo_memo_max_entries=8)


@pytest.mark.parametrize(
    "temperature", (600.0, nextafter(1000.0, -inf), 1000.0, nextafter(1000.0, inf))
)
def test_exact_residual_complete_profile_transport_and_nextafter_are_unchanged(temperature):
    spec, network, initial = authored_case(energy=True, cells=5)
    plain, memo = assembly(spec, network, initial), assembly(spec, network, initial, cap=128)
    point = plain.initial.copy()
    point[plain.count * plain.size : plain.count * (plain.size + 1)] = temperature
    expected, details = plain.evaluate(point)
    actual, profile = memo.evaluate(point)
    np.testing.assert_array_equal(actual, expected)
    assert profile == details
    light, omitted = memo.evaluate(point, retain_details=False)
    np.testing.assert_array_equal(light, expected)
    assert omitted is None
    kwargs = {
        "temperature_K": temperature,
        "pressure_Pa": spec.pressure_Pa,
        "mole_fractions": {"A": 0.7, "B": 0.3},
    }
    assert (
        memo.response(temperature, kwargs["mole_fractions"]).to_dict()
        == gas_transport_response(spec.transport, network, **kwargs).to_dict()
    )
    assert memo.variables == plain.variables and memo.equations == plain.equations
    assert memo.heat_scale == plain.heat_scale
    assert memo.thermo_memo.statistics()["hits"] > 0


@pytest.mark.parametrize("law", ("third_body_arrhenius", "troe_falloff"))
def test_thermo_only_memo_never_caches_composition_pressure_colliders_or_rates(law):
    spec, network, initial = authored_case(kinetics=law)
    memo = assembly(spec, network, initial, cap=16)
    views = memo.thermo_views
    original_network = deepcopy(memo.network.to_dict())
    original_models = memo.network.imported_gas_thermochemistry
    model_dict = {s: model.to_dict() for s, model in original_models.items()}
    rows = ({"A": 0.03, "B": 0.01}, {"A": 0.01, "B": 0.03}, {"A": 0.06, "B": 0.02})
    rates = []
    for row in rows:
        arguments = {"volume_L": 1.0, "temperature_K": 800.0}
        expected = memo.network.reaction_rates(row, **arguments)
        actual = memo.network.reaction_rates(row, **arguments, species_thermo=views)
        assert actual == expected
        assert memo.network.amount_derivatives(
            row, **arguments, species_thermo=views
        ) == memo.network.amount_derivatives(row, **arguments)
        rates.append(actual["isomer"])
    assert len(set(rates)) == len(rows)
    assert memo.thermo_memo.statistics()["hits"] >= 6
    assert memo.network.to_dict() == original_network
    assert memo.network.imported_gas_thermochemistry is original_models
    assert {s: model.to_dict() for s, model in original_models.items()} == model_dict
    assert all(type(model) is NASA7SpeciesThermo for model in original_models.values())
    json.dumps(memo.network.to_dict())
    # Distinct composition changes the full BVP source/profile at identical cell T.
    point = memo.initial.copy()
    _, first = memo.evaluate(point)
    point[0], point[1] = point[1], point[0]
    _, second = memo.evaluate(point)
    assert first["chemical_sources_mol_s"][0] != second["chemical_sources_mol_s"][0]


@pytest.mark.parametrize("advection", ("upwind", "linear-upwind"))
@pytest.mark.parametrize(("free", "energy"), ((False, False), (False, True), (True, True)))
def test_dense_and_colored_complete_jacobians_are_exact_for_plain_memo_and_light(
    advection, free, energy
):
    spec, network, _ = authored_case(cells=7, energy=energy, advection=advection)
    mesh = DiffusionMesh("planar", tuple(np.linspace(0, 1, 8) ** 1.2 * 0.02), area_m2=0.023)
    spec = replace(spec, mesh=mesh)
    if free:
        spec = replace(spec, mass_flux_kg_m2_s=None, phase=flow.GasFlamePhaseSpec(0.011, 650.0))
    initial = flow.premixed_flow_initial_profile(
        spec,
        network,
        temperatures_K=np.linspace(610, 720, 7),
        mole_fractions=[{"A": v, "B": 1 - v} for v in np.linspace(0.75, 0.65, 7)],
        inlet_trace_mole_fractions={"A": 0.79, "B": 0.21},
        mass_flux_kg_m2_s=0.002 if free else spec.mass_flux_kg_m2_s,
    )
    plain, memo = assembly(spec, network, initial), assembly(spec, network, initial, cap=1024)
    lower, upper = bounds(plain.variables)
    scales = [v.scale for v in plain.variables]
    dense = finite_jacobian(
        lambda y: plain.evaluate(y)[0], plain.initial, scales, lower=lower, upper=upper
    )
    for retain in (True, False):

        def callback(y, retain_details=retain):
            return memo.evaluate(y, retain_details=retain_details)[0]

        observed = finite_jacobian(callback, memo.initial, scales, lower=lower, upper=upper)
        colored = sparse_finite_jacobian(
            callback, memo.initial, scales, lower=lower, upper=upper, sparsity=memo.state_sparsity()
        )
        np.testing.assert_array_equal(observed, dense)
        np.testing.assert_array_equal(colored.toarray(), dense)


@pytest.mark.parametrize("strategy", ("dense", "sparse-colored"))
@pytest.mark.parametrize("energy", (False, True))
@pytest.mark.parametrize(("cap", "light"), ((None, True), (1024, False), (1024, True)))
def test_real_authored_roots_and_final_full_reports_are_exact_under_both_jacobians(
    strategy, energy, cap, light
):
    spec, network, initial = authored_case(energy=energy, cells=5)
    expected = flow.solve_premixed_gas_flow(spec, network, initial, jacobian_strategy=strategy)
    actual = flow.solve_premixed_gas_flow(
        spec,
        network,
        initial,
        jacobian_strategy=strategy,
        thermo_memo_max_entries=cap,
        residual_only_trials=light,
    )
    assert expected["success"] and actual["success"]
    assert stripped(actual) == expected
    assert all(actual["physical_checks"].values())
    assert actual["derivative_diagnostics"]["derivative_mode"] == "finite_difference"
    if cap is not None:
        assert actual["thermo_memo_statistics"]["hits"] > actual["thermo_memo_statistics"]["misses"]
        assert all(type(v) is int for v in actual["thermo_memo_statistics"].values())
    else:
        assert "thermo_memo_statistics" not in actual
    assert actual.get("residual_only_trials", False) is light
    assert actual["experimental_elapsed_s"] == actual["live_operations_executed"] == 0


@pytest.mark.parametrize("strategy", ("dense", "sparse-colored"))
def test_real_authored_free_mass_eigenvalue_keeps_identical_final_fresh_profile(strategy):
    spec, network, initial = authored_case(energy=True, cells=5)
    fixed = flow.solve_premixed_gas_flow(spec, network, initial, jacobian_strategy=strategy)
    assert fixed["success"]
    profile = fixed["profile"]
    target = float(np.interp(0.01, spec.mesh.centers_m, profile["temperatures_K"]))
    spec = replace(spec, mass_flux_kg_m2_s=None, phase=flow.GasFlamePhaseSpec(0.01, target))
    guess = {k: deepcopy(profile[k]) for k in initial}
    guess["mass_flux_kg_m2_s"] *= 1.1
    expected = flow.solve_premixed_gas_flow(spec, network, guess, jacobian_strategy=strategy)
    actual = flow.solve_premixed_gas_flow(
        spec,
        network,
        guess,
        jacobian_strategy=strategy,
        thermo_memo_max_entries=1024,
        residual_only_trials=True,
    )
    assert expected["success"] and actual["success"]
    assert stripped(actual) == expected
    assert actual["profile"]["mass_flux_kg_m2_s"] == pytest.approx(0.005, rel=1e-6)


def test_residual_only_skips_only_diagnostic_serialization_and_final_remains_full(monkeypatch):
    spec, network, initial = authored_case(energy=True)
    unit = assembly(spec, network, initial, cap=16)
    _, complete = unit.evaluate(unit.initial)
    from chemworld.physchem.gas_transport_fluxes import GasTransportFluxResult

    def forbidden(_self):
        raise RuntimeError("diagnostic serialization reached")

    monkeypatch.setattr(GasTransportFluxResult, "to_dict", forbidden)
    residual, details = unit.evaluate(unit.initial, retain_details=False)
    assert details is None and np.all(np.isfinite(residual))
    with pytest.raises(RuntimeError, match="diagnostic serialization"):
        unit.evaluate(unit.initial)
    assert complete["interior_diffusive_fluxes"]


def test_public_trial_option_forwards_false_and_performs_one_original_full_fresh_pass(monkeypatch):
    spec, network, initial = authored_case(energy=True)
    original = flow._PremixedResidual.evaluate
    retained = []

    def counted(self, vector, *, retain_details=True):
        retained.append(retain_details)
        return original(self, vector, retain_details=retain_details)

    monkeypatch.setattr(flow._PremixedResidual, "evaluate", counted)
    result = flow.solve_premixed_gas_flow(
        spec, network, initial, thermo_memo_max_entries=64, residual_only_trials=True
    )
    assert result["success"]
    assert retained[-1] is True and sum(retained) == 1 and len(retained) > 2
    assert result["profile"]["interior_diffusive_fluxes"]
    assert result["profile"]["Mach_inlet_trace_cells_and_faces"]


def test_capacity_eviction_and_repeated_domain_failures_remain_query_local():
    spec, network, initial = authored_case()
    first, second = assembly(spec, network, initial, cap=3), assembly(spec, network, initial, cap=3)
    before = second.thermo_memo.statistics()
    for temperature in (650.0, 700.0, 750.0, 800.0):
        first.response(temperature, {"A": 0.6, "B": 0.4})
    stats = first.thermo_memo.statistics()
    assert stats["current_entries"] <= 3 and stats["peak_entries"] <= 3 and stats["evictions"] > 0
    assert second.thermo_memo.statistics() == before
    for _ in range(2):
        with pytest.raises(ResidualDomainError, match="source-fit domain"):
            first.response(299.0, {"A": 0.6, "B": 0.4})
    assert first.thermo_memo.statistics() == stats


def test_runtime_thermo_exception_is_not_cached_or_changed(monkeypatch):
    spec, network, initial = authored_case()
    memo = assembly(spec, network, initial, cap=16)
    original = NASA7SpeciesThermo.evaluate
    error = RuntimeError("authored thermo failure")
    calls = []

    def failing(self, temperature):
        if temperature == 777.0:
            calls.append((self.species_id, temperature))
            raise error
        return original(self, temperature)

    monkeypatch.setattr(NASA7SpeciesThermo, "evaluate", failing)
    before = memo.thermo_memo.statistics()
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            memo.response(777.0, {"A": 0.6, "B": 0.4})
        assert caught.value is error
    after = memo.thermo_memo.statistics()
    assert len(calls) == 2 and after["misses"] == before["misses"] + 2
    assert after["current_entries"] == before["current_entries"]


def test_transport_state_validation_runs_on_every_hit_and_never_caches_response():
    spec, network, initial = authored_case()
    memo = assembly(spec, network, initial, cap=16)
    first = memo.response(800.0, {"A": 0.8, "B": 0.2})
    second = memo.response(800.0, {"A": 0.2, "B": 0.8})
    assert first.mole_fractions != second.mole_fractions
    assert first is not second
    stats = memo.thermo_memo.statistics()
    for bad in ({"A": -0.1, "B": 1.1}, {"A": 0.4, "B": 0.4}, {"A": 1.0}):
        with pytest.raises(ResidualDomainError):
            memo.response(800.0, bad)
    assert memo.thermo_memo.statistics() == stats


def test_opt_in_final_fresh_mach_uses_frozen_query_models_despite_caller_progress_mutation():
    spec, network, initial = authored_case(energy=True)
    expected = flow.solve_premixed_gas_flow(
        spec, network, initial, thermo_memo_max_entries=128, residual_only_trials=True
    )
    original = deepcopy(network.to_dict())
    changed = []

    def mutate(_event):
        if not changed:
            # Change both caller declaration and its cached model mapping; neither
            # is the complete private network snapshot or query memo owner.
            model = network.imported_gas_thermochemistry["A"]
            coeff = list(model.segments[0].coefficients)
            coeff[0] = 10.0
            network.imported_gas_thermochemistry["A"] = replace(
                model,
                segments=(replace(model.segments[0], coefficients=tuple(coeff)),),
            )
            network.metadata["gas_thermochemistry"]["A"]["thermo"]["data"][0][0] = 10.0
            network.reactions[0].rate_law.parameters["k"] = 100.0
            changed.append(True)

    result = flow.solve_premixed_gas_flow(
        spec,
        network,
        initial,
        thermo_memo_max_entries=128,
        residual_only_trials=True,
        progress=mutate,
    )
    assert changed and result == expected
    assert network.to_dict() != original
    with pytest.raises(ValueError, match="binding mismatch"):
        gas_transport_response(
            spec.transport,
            network,
            temperature_K=600.0,
            pressure_Pa=100000.0,
            mole_fractions={"A": 0.8, "B": 0.2},
        )


def test_failure_reports_keep_original_failure_plus_only_opt_in_numeric_fields():
    spec, network, initial = authored_case(energy=True)
    policy = RootPolicy(maximum_iterations=1)
    expected = flow.solve_premixed_gas_flow(spec, network, initial, root_policy=policy)
    actual = flow.solve_premixed_gas_flow(
        spec,
        network,
        initial,
        root_policy=policy,
        thermo_memo_max_entries=16,
        residual_only_trials=True,
    )
    assert not expected["success"] and not actual["success"]
    assert stripped(actual) == expected
    assert actual["profile"] is None
    assert actual["thermo_memo_statistics"]["current_entries"] <= 16


def test_all_bound_consumers_share_one_view_without_network_model_replacement(monkeypatch):
    spec, network, initial = authored_case(kinetics="third_body_arrhenius")
    original = ReactionNetworkSpec.amount_derivatives
    consumers = []

    def counted(self, *args, **kwargs):
        consumers.append((self, kwargs.get("species_thermo")))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ReactionNetworkSpec, "amount_derivatives", counted)
    memo = assembly(spec, network, initial, cap=16)
    memo.evaluate(memo.initial)
    assert memo.bound_transport_response.args[2] is memo.thermo_views
    assert len(consumers) == spec.mesh.cells + 1
    assert all(bound is memo.network and views is memo.thermo_views for bound, views in consumers)
    assert memo.network is not network
    assert all(
        type(model) is NASA7SpeciesThermo
        for model in memo.network.imported_gas_thermochemistry.values()
    )
    assert not any("memo" in key or "view" in key for key in memo.network.__dict__)


def test_repeated_state_reduces_real_thermo_evaluations_but_revalidates_every_cache_hit(
    monkeypatch,
):
    spec, network, initial = authored_case(kinetics="troe_falloff")
    plain, memo = assembly(spec, network, initial), assembly(spec, network, initial, cap=16)
    calls, validations = [], []
    original_evaluate = NASA7SpeciesThermo.evaluate
    original_select = NASA7SpeciesThermo.select_segment

    def counted(self, temperature):
        calls.append((self.species_id, temperature))
        return original_evaluate(self, temperature)

    def checked(self, temperature):
        validations.append((self.species_id, temperature))
        return original_select(self, temperature)

    monkeypatch.setattr(NASA7SpeciesThermo, "evaluate", counted)
    monkeypatch.setattr(NASA7SpeciesThermo, "select_segment", checked)
    expected = plain.evaluate(plain.initial)
    assert len(calls) > 20
    calls.clear()
    validations.clear()
    actual = memo.evaluate(memo.initial)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]
    assert calls == []
    assert len(validations) > 20  # No skip of the original thermo domain/segment validator.


def test_original_fresh_gate_remains_authoritative_after_residual_only_trials(monkeypatch):
    spec, network, initial = authored_case(energy=True)
    original = flow._PremixedResidual.evaluate
    full_calls = []

    def changed_fresh(self, vector, *, retain_details=True):
        residual, profile = original(self, vector, retain_details=retain_details)
        if retain_details:
            full_calls.append(profile)
            profile["EOS_residuals_Pa"][0] = 2e-4
        return residual, profile

    monkeypatch.setattr(flow._PremixedResidual, "evaluate", changed_fresh)
    actual = flow.solve_premixed_gas_flow(
        spec, network, initial, thermo_memo_max_entries=16, residual_only_trials=True
    )
    assert len(full_calls) == 1 and actual["profile"] is full_calls[0]
    assert not actual["success"] and not actual["physical_checks"]["EOS"]
    assert actual["failure_reason"] == "fresh_physical_balance_failed"


def test_residual_only_retains_non_nasa_original_capability_and_no_cache_metadata():
    spec, network, initial = authored_case(nasa=False, energy=True)
    expected = flow.solve_premixed_gas_flow(spec, network, initial)
    actual = flow.solve_premixed_gas_flow(spec, network, initial, residual_only_trials=True)
    assert expected["success"] and actual["success"]
    assert stripped(actual) == expected
    assert "thermo_memo_statistics" not in actual


@pytest.mark.parametrize("damage", ("negative-material", "empty-trace", "outside-transport"))
@pytest.mark.parametrize("retain", (False, True))
def test_domain_failure_type_and_message_are_preserved_for_both_detail_modes(damage, retain):
    spec, network, initial = authored_case()
    plain, memo = assembly(spec, network, initial), assembly(spec, network, initial, cap=16)
    point = plain.initial.copy()
    if damage == "negative-material":
        point[0] = -1.0
    elif damage == "empty-trace":
        start = plain.count * (plain.size + 1)
        point[start : start + plain.size] = 0.0
    else:
        point[plain.count * plain.size] = 1201.0
    with pytest.raises(ResidualDomainError) as expected:
        plain.evaluate(point)
    with pytest.raises(ResidualDomainError) as actual:
        memo.evaluate(point, retain_details=retain)
    assert type(actual.value) is type(expected.value)
    assert actual.value.args == expected.value.args


def authored_soret_case():
    """Reuse authored Soret SI/C* patterns with authored NASA7 thermo only."""
    from chemworld.physchem.gas_soret_transport import (
        BOLTZMANN_J_K,
        DEBYE_C_M,
        GasMixtureSoretSpec,
    )

    ids = ("N2", "O2", "AR")
    formulas = {"N2": "N2", "O2": "O2", "AR": "Ar"}
    cp = {"N2": 30.0, "O2": 40.0, "AR": 25.0}
    h0 = {"N2": -1000.0, "O2": 1200.0, "AR": 700.0}
    species, declarations, si = [], {}, {}
    for i, name in enumerate(ids):
        raw = {
            "model": "gas",
            "geometry": "atom" if name == "AR" else "linear",
            "diameter": 3.0 + 0.2 * i,
            "well-depth": 100 + 30 * i,
            "dipole": 0.5 if name == "O2" else 0.0,
            "polarizability": 0.8 + 0.1 * i,
            "note": "Authored artificial mathematical parameters, no material fit",
        }
        species_spec = SpeciesSpec(
            name,
            formulas[name],
            phase="gas",
            metadata={"source_declaration": {"transport": raw}},
        )
        species.append(species_spec)
        coeff = [cp[name] / R, 0.0, 0.0, 0.0, 0.0, h0[name] / R, 2.0]
        declarations[name] = {
            "thermo": {
                "model": "NASA7",
                "temperature-ranges": [300.0, 1000.0, 2000.0],
                "data": [coeff, coeff.copy()],
                "note": "Authored integration fixture",
            },
            "composition": species_spec.composition,
            "reference_pressure_Pa": 101325.0,
            "units": {},
        }
        si[name] = {
            "diameter_m": 1e-10 * raw["diameter"],
            "well_depth_J": BOLTZMANN_J_K * raw["well-depth"],
            "dipole_C_m": DEBYE_C_M * raw["dipole"],
            "polarizability_m3": 1e-30 * raw["polarizability"],
        }
    source = DataSourceProvenance(
        "authored-soret-query36",
        0,
        "Authored SI/Cstar fixture",
        "v1",
        "CC0",
        "a" * 64,
    )
    base = GasTransportPolynomialSpec(
        ids,
        {s.species_id: s.to_dict() for s in species},
        declarations,
        {name: ComponentSpec(name, formulas[name]).molecular_weight_g_mol / 1000 for name in ids},
        {name: (0.002 + 0.0005 * ids.index(name), 0, 0, 0, 0) for name in ids},
        dict.fromkeys(ids, (0.001, 0, 0, 0, 0)),
        {
            name: {other: (0.0001 * (i + j + 1), 0, 0, 0, 0) for j, other in enumerate(ids)}
            for i, name in enumerate(ids)
        },
        source,
        "authored-synthetic",
        (300.0, 1200.0),
        200000.0,
        "off",
    )
    soret = GasMixtureSoretSpec(
        parent_transport_declaration=base.to_dict(),
        species_transport_SI=si,
        cstar_coefficients={
            name: {other: (1.1 + 0.02 * (i + j), *([0] * 8)) for j, other in enumerate(ids)}
            for i, name in enumerate(ids)
        },
        fit_temperature_range_K=(300.0, 1200.0),
        source_data_kind="authored-synthetic",
        source=source,
    )
    transport = replace(base, thermal_diffusion="cantera-3.2-mixture-soret", soret=soret)
    network = ReactionNetworkSpec(
        "authored-soret-query-integration36",
        tuple(species),
        (),
        metadata={"gas_thermochemistry": declarations},
    )
    spec = flow.GasPremixedFlowSpec(
        DiffusionMesh("planar", (0.0, 0.002, 0.009, 0.02), area_m2=0.023),
        transport,
        100000.0,
        350.0,
        {"N2": 0.5, "O2": 0.3, "AR": 0.2},
        "N2",
        (300.0, 1200.0),
        True,
        0.005,
    )
    initial = flow.premixed_flow_initial_profile(
        spec,
        network,
        temperatures_K=[500.0, 750.0, 1000.0],
        mole_fractions=[
            {"N2": 0.4, "O2": 0.35, "AR": 0.25},
            {"N2": 0.3, "O2": 0.45, "AR": 0.25},
            {"N2": 0.2, "O2": 0.5, "AR": 0.3},
        ],
    )
    return spec, network, initial, base


@pytest.mark.parametrize("energy", (False, True))
def test_lj_explicit_memo_refusal_and_none_false_original_contract(energy, monkeypatch):
    from chemworld.physchem.gas_transport import GasTransportSpec
    from chemworld.physchem.lennard_jones import LennardJonesSpec

    spec, network, initial = authored_case(reacting=False, energy=energy)
    transport = GasTransportSpec(
        (
            LennardJonesSpec("A", 120.0, 3.2e-10, "Authored synthetic LJ fixture"),
            LennardJonesSpec("B", 160.0, 3.5e-10, "Authored synthetic LJ fixture"),
        ),
        {"A": "linear", "B": "linear"},
        "off",
    )
    spec = replace(spec, transport=transport)
    original = deepcopy((spec.to_dict(), network.to_dict(), initial))
    with pytest.raises(ValueError, match="currently requires polynomial transport"):
        flow.solve_premixed_gas_flow(spec, network, initial, thermo_memo_max_entries=16)
    old_spec = baseline.GasPremixedFlowSpec.from_dict(spec.to_dict())
    old = baseline._PremixedResidual(old_spec, network, initial, 10000, None)
    new = assembly(spec, network, initial)
    for point in (old.initial, old.initial.copy()):
        point[old.count * old.size : old.count * (old.size + 1)] = [600.0, 650.0, 700.0, 750.0]
        expected, old_profile = old.evaluate(point)
        actual, profile = new.evaluate(point)
        np.testing.assert_array_equal(actual, expected)
        assert profile == old_profile
    assert (
        new.response(700.0, {"A": 0.7, "B": 0.3}).to_dict()
        == old.response(700.0, {"A": 0.7, "B": 0.3}).to_dict()
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("None/False LJ path must not create a thermo owner")

    monkeypatch.setattr(GasQueryThermoMemo, "__init__", forbidden)
    expected = baseline.solve_premixed_gas_flow(old_spec, network, initial)
    actual = flow.solve_premixed_gas_flow(
        spec, network, initial, thermo_memo_max_entries=None, residual_only_trials=False
    )
    assert actual == expected and actual["success"]
    assert (spec.to_dict(), network.to_dict(), initial) == original
    assert "thermo_memo_statistics" not in actual and "residual_only_trials" not in actual


@pytest.mark.parametrize(("cap", "light"), ((64, False), (None, True), (64, True)))
def test_soret_nonzero_temperature_and_composition_gradients_keep_original_complete_physics(
    cap, light
):
    from math import fsum

    spec, network, initial, off_transport = authored_soret_case()
    original = deepcopy((spec.to_dict(), network.to_dict(), initial))
    old_spec = baseline.GasPremixedFlowSpec.from_dict(spec.to_dict())
    old = baseline._PremixedResidual(old_spec, network, initial, 10000, None)
    expected, old_profile = old.evaluate(old.initial)
    new = assembly(spec, network, initial, cap=cap)
    trial, trial_profile = new.evaluate(new.initial, retain_details=not light)
    np.testing.assert_array_equal(trial, expected)
    if light:
        assert trial_profile is None
    else:
        assert trial_profile == old_profile
    fresh, profile = new.evaluate(new.initial)
    np.testing.assert_array_equal(fresh, expected)
    assert profile == old_profile
    off = assembly(replace(spec, transport=off_transport), network, initial)
    _, off_profile = off.evaluate(off.initial)
    assert any(
        on["molar_fluxes_mol_m2_s"] != disabled["molar_fluxes_mol_m2_s"]
        for on, disabled in zip(
            profile["interior_diffusive_fluxes"],
            off_profile["interior_diffusive_fluxes"],
            strict=True,
        )
    )
    assert profile["face_enthalpy_flows_W"] != off_profile["face_enthalpy_flows_W"]
    for face, part in enumerate(profile["interior_diffusive_fluxes"]):
        left_temperature = (
            spec.inlet_temperature_K if face == 0 else initial["temperatures_K"][face - 1]
        )
        right_temperature = initial["temperatures_K"][face]
        distance = float(
            spec.mesh.centers_m[0]
            if face == 0
            else spec.mesh.centers_m[face] - spec.mesh.centers_m[face - 1]
        )
        force = (
            2
            * (right_temperature - left_temperature)
            / ((left_temperature + right_temperature) * distance)
        )
        assert force != 0.0 and part["thermal_diffusion"]["temperature_log_gradient_per_m"] == force
        assert any(v != 0.0 for v in part["thermal_diffusion"]["coefficients_kg_m_s"].values())
        assert any(v != 0.0 for v in part["driving_forces_per_m"].values())
        mid = profile["face_transport_and_advective_states"][2 * face]
        midpoint = new.response(mid["temperature_K"], mid["mole_fractions"])
        assert part["thermal_diffusion"]["coefficients_kg_m_s"] == dict(
            midpoint.thermal_diffusion_coefficients_kg_m_s
        )
        assert (
            part["fourier_heat_flux_W_m2"]
            == -midpoint.conductivity_W_m_K * (right_temperature - left_temperature) / distance
        )
        enthalpy_temperature = spec.inlet_temperature_K if face == 0 else midpoint.temperature_K
        models = new.network.imported_gas_thermochemistry
        diffusion_h = fsum(
            models[s].evaluate(enthalpy_temperature).enthalpy_J_mol
            * part["molar_fluxes_mol_m2_s"][s]
            for s in network.species_ids
        )
        assert part["diffusion_enthalpy_flux_W_m2"] == diffusion_h
        assert part["total_heat_flux_W_m2"] == part["fourier_heat_flux_W_m2"] + diffusion_h
    assert (spec.to_dict(), network.to_dict(), initial) == original
    assert (new.thermo_views is None) is (cap is None)
