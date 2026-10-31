"""Authored function and seven real shared-root cases; no material/Cantera data."""

import json
from copy import deepcopy
from dataclasses import replace
from math import fsum
from types import SimpleNamespace

import numpy as np
import pytest

from chemworld.examples.native_imported_surface import SOURCE, authored_declaration
from chemworld.physchem import gas_impinging_surface as module
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.gas_polynomial_transport import GasTransportPolynomialSpec
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.residual_sparse_difference import sparse_finite_jacobian
from chemworld.physchem.residual_system import ResidualDomainError, bounds, finite_jacobian
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.surface_mechanism_import import convert_surface_mechanism


def authored(*, area=0.023, occupancy=1, wall=600, closure="B", model="mixture-averaged"):
    payload = authored_declaration(site_occupancy=occupancy)
    payload["reactions"] = [
        {
            "equation": f"A + {'' if occupancy == 1 else str(occupancy) + ' '}Pt => HA",
            "rate-constant": {"A": 0.01, "b": 0, "Ea": 0},
        },
        {
            "equation": f"HA => B + {'' if occupancy == 1 else str(occupancy) + ' '}Pt",
            "rate-constant": {"A": 2, "b": 0, "Ea": 0},
        },
    ]
    imported = convert_surface_mechanism(
        payload,
        surface_phase="surface",
        source=SOURCE,
        minimum_temperature_K=300,
        maximum_temperature_K=1900,
    )
    gas = imported.gas_network
    ids = gas.species_ids
    masses = {
        s.species_id: ComponentSpec(s.species_id, s.formula).molecular_weight_g_mol / 1000
        for s in gas.species
    }
    transport = GasTransportPolynomialSpec(
        ids,
        {s.species_id: s.to_dict() for s in gas.species},
        gas.metadata["gas_thermochemistry"],
        masses,
        dict.fromkeys(ids, (0.002, 0, 0, 0, 0)),
        dict.fromkeys(ids, (0.001, 0, 0, 0, 0)),
        {s: dict.fromkeys(ids, (0.0001, 0, 0, 0, 0)) for s in ids},
        replace(SOURCE, checksum="0" * 64),
        "authored-synthetic",
        (300, 1200),
        200000,
        "off",
    )
    mesh = DiffusionMesh("planar", (0, 0.001, 0.004, 0.011, 0.02), area_m2=area)
    x = {"A": 0.8, "B": 0.2}
    rho = 1e5 / (R * 600) * fsum(x[k] * masses[k] for k in ids)
    a = 100
    spec = module.GasImpingingSurfaceSpec(
        mesh,
        transport,
        imported.surface,
        1e5,
        600,
        wall,
        x,
        rho * a * 0.02**2,
        closure,
        (300, 1200),
        a * 0.02,
        model,
    )
    if occupancy == 1:
        adsorption = 0.01 * 1e5 * 0.8 / (R * wall)
        theta = {
            "Pt": 0.9 * 2 / (2 + adsorption),
            "HA": 0.9 * adsorption / (2 + adsorption),
            "HB": 0.1,
        }
    else:
        theta = {"Pt": 0.4, "HA": 0.4, "HB": 0.2, "HAA": 0}
    initial = module.impinging_surface_initial_profile(
        spec,
        gas,
        temperatures_K=[600] * mesh.cells,
        mole_fractions=[x] * mesh.cells,
        spread_rates_per_s=(a * (0.02 - np.asarray(mesh.centers_m))).tolist(),
        face_mass_fluxes_kg_m2_s=(rho * a * (0.02 - np.asarray(mesh.faces_m)) ** 2).tolist(),
        surface_coverages=theta,
        inlet_trace_mole_fractions=x,
        wall_trace_mole_fractions=x,
    )
    return spec, gas, initial


def assembly(spec, gas, initial):
    return module._ImpingingResidual(spec, gas, initial, 100000, None)


def test_complete_codec_chart_dimension_and_single_wall_boundary():
    spec, gas, initial = authored()
    assert module.GasImpingingSurfaceSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    a = assembly(spec, gas, initial)
    assert (
        len(a.variables)
        == len(a.equations)
        == spec.mesh.cells * (len(gas.species) + 3) + 2 * len(gas.species) + 2 + a.chart.rank
    )
    assert a.chart.rank == 1 and len(a.chart.species) == 3  # Includes a disconnected HB pool.
    residual, details = a.evaluate(a.initial)
    assert len(residual) == len(a.variables)
    assert details["G_boundary_residuals_kg_s"] == [0, 0]
    assert max(map(abs, details["surface_rates_mol_s"].values())) < 1e-20
    assert details["surface_coverages"]["HB"] == pytest.approx(0.1)
    assert details["surface_gas_rates_mol_s"]["A"] < 0 < details["surface_gas_rates_mol_s"]["B"]
    assert details["inlet_species_residuals_mol_s"][1]["A"] == pytest.approx(
        details["surface_gas_rates_mol_s"]["A"]
    )
    assert abs(details["global_mass_balance_residual_kg_s"]) < 1e-18
    assert abs(details["global_enthalpy_balance_residual_W"]) < 1e-12
    assert (
        max(abs(v) for row in details["species_balance_residuals_mol_s"] for v in row.values())
        < 1e-14
    )


@pytest.mark.parametrize("volume", (0.001, 0.3, 1, 8))
def test_surface_probe_volume_does_not_change_actual_original_rates(volume):
    spec, gas, initial = authored()
    a = assembly(spec, gas, initial)
    theta = a.decode(a.initial)[-1]
    x = initial["wall_trace_mole_fractions"]
    reference, _ = a.wall_rates(theta, x)
    observed, _ = a.wall_rates(theta, x, probe_volume_L=volume)
    assert observed.amount_rates_mol_s == pytest.approx(
        reference.amount_rates_mol_s, rel=3e-14, abs=1e-22
    )


@pytest.mark.parametrize("occupancy", (1, 2, 3))
def test_area_and_occupancy_use_actual_molecular_mol_without_double_site_factors(occupancy):
    spec, gas, initial = authored(occupancy=occupancy)
    other, gas2, initial2 = authored(area=2 * spec.mesh.area_m2, occupancy=occupancy)
    a, b = assembly(spec, gas, initial), assembly(other, gas2, initial2)
    _, left = a.evaluate(a.initial)
    _, right = b.evaluate(b.initial)
    capacity = spec.mesh.area_m2 * spec.surface.site_densities_mol_m2["surface"]
    assert initial["surface_amounts_mol"]["HA"] == pytest.approx(
        capacity * left["surface_coverages"]["HA"] / occupancy
    )
    for key in gas.species_ids:
        assert right["surface_gas_rates_mol_s"][key] == pytest.approx(
            2 * left["surface_gas_rates_mol_s"][key]
        )
        assert right["surface_gas_rates_mol_m2_s"][key] == pytest.approx(
            left["surface_gas_rates_mol_m2_s"][key]
        )
    assert left["surface_coverage_rates_per_s"] == pytest.approx(
        right["surface_coverage_rates_per_s"]
    )


@pytest.mark.parametrize("model", ("mixture-averaged", "stefan-maxwell"))
def test_real_wall_temperature_enthalpy_once_and_no_second_chemical_heat(model):
    spec, gas, initial = authored(wall=700, model=model)
    initial["wall_trace_mole_fractions"] = {"A": 0.2, "B": 0.8}
    a = assembly(spec, gas, initial)
    _, d = a.evaluate(a.initial)
    face = d["face_diffusive_fluxes"][-1]
    wall = a.response(700, initial["wall_trace_mole_fractions"])
    diffusion = fsum(
        wall.partial_enthalpies_J_mol[k] * face["molar_fluxes_mol_m2_s"][k] for k in gas.species_ids
    )
    expected = spec.mesh.area_m2 * (face["fourier_heat_flux_W_m2"] + diffusion)
    assert d["face_transport_and_advective_states"][-1]["advective_temperature_K"] == 700
    assert d["axial_face_enthalpy_flows_W"][-1] == pytest.approx(expected)
    assert d["wall_external_compensation_W"] == pytest.approx(-expected)
    assert face["diffusion_enthalpy_flux_W_m2"] == pytest.approx(diffusion)


@pytest.mark.parametrize("model", ("mixture-averaged", "stefan-maxwell"))
def test_dense_and_colored_full_jacobian_include_all_wall_trace_and_q_dependencies(model):
    spec, gas, initial = authored(model=model)
    initial["wall_trace_mole_fractions"] = {"A": 0.3, "B": 0.7}
    a = assembly(spec, gas, initial)
    scales = np.asarray([v.scale for v in a.variables])
    low, high = bounds(a.variables)

    def fun(y):
        return a.evaluate(y)[0]

    dense = finite_jacobian(fun, a.initial, scales, lower=low, upper=high, relative_step=1e-6)
    sparse = sparse_finite_jacobian(
        fun,
        a.initial,
        scales,
        lower=low,
        upper=high,
        relative_step=1e-6,
        sparsity=a.state_sparsity(),
    ).toarray()
    np.testing.assert_allclose(sparse, dense, atol=1e-10, rtol=1e-10)
    assert np.any(dense[-a.chart.rank :, a.trace_start + a.size : a.q_start] != 0)
    assert np.any(dense[:, a.q_start :] != 0)


def test_chart_preserves_all_pools_and_rejects_negative_reconstructed_inventory():
    spec, gas, initial = authored()
    a = assembly(spec, gas, initial)
    y = a.initial.copy()
    y[a.q_start] = 0.01
    _, d = a.evaluate(y)
    assert d["surface_coverages"]["HB"] == pytest.approx(0.1)
    assert a.chart.same_class(d["surface_coverages"])
    y[a.q_start] = 100
    with pytest.raises(ResidualDomainError, match="negative"):
        a.evaluate(y)


def test_off_eos_trial_mass_identity_uses_original_stock_and_never_projects_wall_gas_rates():
    spec, gas, initial = authored()
    a = assembly(spec, gas, initial)
    y = a.initial.copy()
    y[: a.n_stop] *= 1.2
    y[a.q_start] = 0.01
    _, d = a.evaluate(y)
    assert max(map(abs, d["EOS_residuals_Pa"])) > 1000
    for i, row in enumerate(d["species_balance_residuals_mol_s"]):
        observed = fsum(row[s] * a.masses[s] for s in gas.species_ids)
        assert observed == pytest.approx(
            d["continuity_residuals_kg_s"][i] - d["chemical_mass_residuals_kg_s"][i], abs=1e-18
        )
        mass = fsum(y[i * a.size + j] * a.masses[s] for j, s in enumerate(a.ids))
        assert d["radial_mass_outflows_kg_s"][i] == pytest.approx(
            2 * d["spread_rates_per_s"][i] * mass
        )
    original_gas_mass = fsum(d["surface_gas_rates_mol_s"][s] * a.masses[s] for s in gas.species_ids)
    assert abs(original_gas_mass) > 1e-12
    assert abs(d["wall_chemical_mass_residual_kg_s"]) < 1e-20
    assert max(map(abs, d["wall_chemical_element_residuals_mol_atoms_s"].values())) < 1e-18
    observed = fsum(d["inlet_species_residuals_mol_s"][1][s] * a.masses[s] for s in gas.species_ids)
    assert observed == pytest.approx(original_gas_mass, rel=1e-12, abs=1e-20)
    assert d["surface_original_conserved_totals_mol"] == pytest.approx(
        d["surface_actual_conserved_totals_mol"]
    )
    assert max(map(abs, d["surface_site_residuals_mol_sites"].values())) < 1e-22


def test_signed_internal_reverse_flow_is_not_replaced_by_a_positive_velocity_model():
    spec, gas, initial = authored()
    initial["face_mass_fluxes_kg_m2_s"][2] = -0.001
    initial["spread_rates_per_s"][1] = -2
    a = assembly(spec, gas, initial)
    _, d = a.evaluate(a.initial)
    assert d["face_mass_fluxes_kg_m2_s"][2] == -0.001
    assert d["radial_mass_outflows_kg_s"][1] < 0


def test_query_private_clones_survive_mutations_of_caller_network_initial_and_surface_payload():
    spec, gas, initial = authored()
    a = assembly(spec, gas, initial)
    before = a.evaluate(a.initial)[0]
    gas.metadata["gas_thermochemistry"]["A"]["thermo"]["h0"] = "99999 J/mol"
    initial["surface_amounts_mol"]["HB"] = 999
    initial["temperatures_K"][0] = 1000
    np.testing.assert_array_equal(a.evaluate(a.initial)[0], before)
    assert a.chart.initial_coverages[-1] == pytest.approx(0.1)


def test_conserved_element_formation_reference_shift_does_not_add_a_second_wall_heat():
    spec, gas, initial = authored()
    initial["wall_trace_mole_fractions"] = {"A": 0.2, "B": 0.8}
    base = assembly(spec, gas, initial)
    _, first = base.evaluate(base.initial)
    original_response = base.response

    def shifted(t, x):
        point = original_response(t, x)
        return replace(
            point,
            partial_enthalpies_J_mol={
                s: h + 2000 for s, h in point.partial_enthalpies_J_mol.items()
            },
        )

    base.response = shifted
    _, second = base.evaluate(base.initial)
    # Both authored gases contain the same two H atoms and have equal MW.
    # Their full mass-frame diffusive molar sum is zero, so this elemental
    # reference shift contributes no wall heat; no second reaction heat appears.
    assert second["wall_external_compensation_W"] == pytest.approx(
        first["wall_external_compensation_W"], abs=1e-12
    )
    for i in range(spec.mesh.cells):
        assert second["energy_balance_residuals_W"][i] == pytest.approx(
            first["energy_balance_residuals_W"][i], abs=1e-12
        )


@pytest.mark.parametrize(
    "mutation", ("species", "thermo", "missing_stock", "temperature", "unknown_codec")
)
def test_missing_original_binding_or_domain_declaration_is_rejected(mutation):
    spec, gas, initial = authored()
    if mutation == "unknown_codec":
        value = spec.to_dict()
        value["unrecognized"] = True
        with pytest.raises(ValueError):
            module.GasImpingingSurfaceSpec.from_dict(value)
    elif mutation == "temperature":
        with pytest.raises(ValueError):
            replace(spec, wall_temperature_K=250)
    elif mutation == "missing_stock":
        del initial["surface_amounts_mol"]["HB"]
        with pytest.raises(ValueError):
            assembly(spec, gas, initial)
    elif mutation == "thermo":
        altered = deepcopy(gas.to_dict())
        altered["metadata"]["gas_thermochemistry"]["A"]["thermo"]["h0"] = "123 J/mol"
        with pytest.raises(ValueError):
            assembly(spec, type(gas).from_dict(altered), initial)
    else:
        altered = deepcopy(gas.to_dict())
        altered["species"][0]["formula"] = "O2"
        with pytest.raises(ValueError):
            assembly(spec, type(gas).from_dict(altered), initial)


@pytest.mark.parametrize("failure", ("exception", "nonconvergence", "fresh_failure"))
def test_query_failure_preserves_complete_public_raw_and_never_calls_a_material_root(
    monkeypatch, failure
):
    spec, gas, initial = authored()
    original = (spec.to_dict(), gas.to_dict(), deepcopy(initial))
    seen = []

    def fake_root(system, y, **kwargs):
        seen.append((system, y.copy(), kwargs))
        if failure == "exception":
            raise RuntimeError("authored original root error")
        return SimpleNamespace(
            root=SimpleNamespace(
                success=failure == "fresh_failure",
                value=y.copy(),
                failure_reason="authored nonconvergence",
                diagnostics={"authored": True},
            ),
            diagnostics={"authored": True},
        )

    monkeypatch.setattr(module, "solve_steady_residual", fake_root)
    report = module.solve_impinging_surface_gas_flow(spec, gas, initial)
    assert len(seen) == 1 and not report["success"] and report["profile"] is None
    assert report["live_operations_executed"] == report["experimental_elapsed_s"] == 0
    assert report["initial_value"] == seen[0][1].tolist()
    if failure == "exception":
        assert report["root_exception"]["message"] == "authored original root error"
    else:
        assert report["root_value"] == seen[0][1].tolist()
    assert (spec.to_dict(), gas.to_dict(), initial) == original
    json.dumps(report, allow_nan=False)


def fake_converged(system, y, **kwargs):
    return SimpleNamespace(
        root=SimpleNamespace(
            success=True, value=y.copy(), failure_reason="", diagnostics={"authored": True}
        ),
        diagnostics={"authored": True},
    )


def test_fresh_checks_include_an_omitted_inert_surface_rate_not_only_q_coordinates(monkeypatch):
    spec, gas, initial = authored()
    original_kernel = module.evaluate_catalytic_rates

    def broken_kernel(view, slot, surface, **kwargs):
        result = original_kernel(view, slot, surface, **kwargs)
        key = module.catalytic_inventory_key(slot, "HB")
        result.amount_rates_mol_s[key] += 1e-6
        return result

    monkeypatch.setattr(module, "evaluate_catalytic_rates", broken_kernel)
    monkeypatch.setattr(module, "solve_steady_residual", fake_converged)
    report = module.solve_impinging_surface_gas_flow(spec, gas, initial)
    candidate = report["candidate_profile"]
    assert max(map(abs, candidate["surface_reduced_residual_per_s"])) < 1e-12
    assert candidate["surface_rates_mol_s"]["HB"] == 1e-6
    assert report["physical_checks"]["all_original_surface_rates"] is False
    assert report["physical_checks"]["all_original_surface_coverage_rates"] is False
    assert report["physical_checks"]["wall_chemical_elements"] is False
    assert not report["success"] and report["profile"] is None


def test_post_root_fresh_exception_keeps_original_public_vector_and_candidate(monkeypatch):
    spec, gas, initial = authored()
    original_evaluate = module._ImpingingResidual.evaluate
    original_response = module._ImpingingResidual.response
    flags = {"completed_profile": False}

    def evaluate(self, vector, **kwargs):
        value = original_evaluate(self, vector, **kwargs)
        if value[1] is not None:
            flags["completed_profile"] = True
        return value

    def response(self, t, x):
        if flags["completed_profile"]:
            raise RuntimeError("authored fresh Mach readout failure")
        return original_response(self, t, x)

    monkeypatch.setattr(module, "solve_steady_residual", fake_converged)
    monkeypatch.setattr(module._ImpingingResidual, "evaluate", evaluate)
    monkeypatch.setattr(module._ImpingingResidual, "response", response)
    report = module.solve_impinging_surface_gas_flow(spec, gas, initial)
    assert not report["success"] and report["profile"] is None
    assert report["root_value"] == report["initial_value"]
    assert report["fresh_exception"]["message"] == "authored fresh Mach readout failure"
    assert report["candidate_profile"]["surface_coverages"]["HB"] == pytest.approx(0.1)


def test_first_full_fresh_exception_returns_preserved_raw_report_without_retry(monkeypatch):
    spec, gas, initial = authored()
    original = RuntimeError("authored first full fresh kernel failure")
    calls = []

    def failed_fresh(self, vector, **kwargs):
        calls.append((vector.copy(), kwargs))
        raise original

    monkeypatch.setattr(module, "solve_steady_residual", fake_converged)
    monkeypatch.setattr(module._ImpingingResidual, "evaluate", failed_fresh)
    report = module.solve_impinging_surface_gas_flow(spec, gas, initial)
    assert len(calls) == 1
    assert not report["success"] and report["profile"] is None
    assert report["root_value"] == report["initial_value"] == calls[0][0].tolist()
    assert report["root_diagnostics"] == {"authored": True}
    assert report["fresh_exception"]["message"] == str(original)
    assert report["fresh_exception"]["type"] == "builtins.RuntimeError"
    assert "raise original" in report["fresh_exception"]["traceback"]
    assert "candidate_profile" not in report


FUNCTIONAL_ROOT_CASES = (
    (550, "mixture-averaged", "dense"),
    (550, "stefan-maxwell", "dense"),
    (600, "mixture-averaged", "dense"),
    (600, "stefan-maxwell", "dense"),
    (650, "mixture-averaged", "dense"),
    (650, "stefan-maxwell", "dense"),
    (600, "mixture-averaged", "sparse-colored"),
)


@pytest.mark.parametrize("wall,model,strategy", FUNCTIONAL_ROOT_CASES)
def test_actual_shared_root_authored_fixed_functional_block(tmp_path, wall, model, strategy):
    import os
    import time
    from pathlib import Path

    from chemworld.physchem.residual_system import RootPolicy

    spec, gas, initial = authored(wall=wall, model=model, occupancy=1)
    before = (spec.to_dict(), gas.to_dict(), deepcopy(initial))
    identity = f"Tw{wall}-{model}-{strategy}"
    options = {
        "root_policy": RootPolicy(tolerance=1e-10, maximum_iterations=40, maximum_backtracks=24),
        "derivative_step": 1e-6,
        "maximum_unknowns": 512,
        "maximum_evaluations": 20000,
        "jacobian_strategy": strategy,
    }
    started = time.monotonic()

    def progress(event):
        if event.get("event") == "root-iteration":
            print(json.dumps({"functional_case": identity, **event}), flush=True)

    result = module.solve_impinging_surface_gas_flow(
        spec, gas, initial, progress=progress, **options
    )
    receipt = {
        "case": identity,
        "wall_temperature_K": wall,
        "flux_model": model,
        "jacobian_strategy": strategy,
        "elapsed_s": time.monotonic() - started,
        "original_spec": before[0],
        "original_network": before[1],
        "original_initial_profile": before[2],
        "result": result,
        "scope": "authored functional only; no material/Cantera/source qualification",
    }
    output = (
        Path(os.environ["CHEMWORLD_IMPINGING_FUNCTIONAL_REPORT_DIR"])
        if "CHEMWORLD_IMPINGING_FUNCTIONAL_REPORT_DIR" in os.environ
        else tmp_path
    )
    path = output / (identity + ".json")
    with path.open("x") as stream:
        json.dump(receipt, stream, allow_nan=False)
    assert result["success"] is True, (identity, result["failure_reason"])
    assert (spec.to_dict(), gas.to_dict(), initial) == before
    a = assembly(spec, gas, initial)
    fresh_residual, d = a.evaluate(np.asarray(result["root_value"]))
    scale = np.asarray([e.scale for e in a.equations])
    independent = {
        "maximum_scaled_residual": float(np.max(abs(fresh_residual / scale))),
        "open_mass_kg_s": d["axial_face_mass_flows_kg_s"][-1]
        - d["axial_face_mass_flows_kg_s"][0]
        + fsum(
            2
            * d["spread_rates_per_s"][i]
            * fsum(d["amounts_mol"][i][s] * a.masses[s] for s in a.ids)
            for i in range(spec.mesh.cells)
        ),
        "open_enthalpy_W": d["axial_face_enthalpy_flows_W"][-1]
        - d["axial_face_enthalpy_flows_W"][0]
        + fsum(
            2
            * d["spread_rates_per_s"][i]
            * fsum(
                d["amounts_mol"][i][s]
                * gas.imported_gas_thermochemistry[s]
                .evaluate(d["temperatures_K"][i])
                .enthalpy_J_mol
                for s in a.ids
            )
            for i in range(spec.mesh.cells)
        ),
        "full_surface_coverage_rate_per_s": max(map(abs, d["surface_coverage_rates_per_s"])),
        "all_wall_species_mol_s": max(map(abs, d["inlet_species_residuals_mol_s"][1].values())),
        "all_original_pools": a.chart.same_class(d["surface_coverages"]),
        "all_chemical_mass_kg_s": d["wall_chemical_mass_residual_kg_s"],
    }
    with (output / (identity + "-independent.json")).open("x") as stream:
        json.dump(independent, stream, allow_nan=False)
    assert independent["maximum_scaled_residual"] <= 1e-10
    assert abs(independent["open_mass_kg_s"]) <= 1e-10
    assert abs(independent["open_enthalpy_W"]) <= 1e-6
    assert independent["full_surface_coverage_rate_per_s"] <= 1e-8
    assert independent["all_wall_species_mol_s"] <= 1e-9
    assert independent["all_original_pools"] is True
    assert abs(independent["all_chemical_mass_kg_s"]) <= 1e-10
