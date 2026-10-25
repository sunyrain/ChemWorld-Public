"""Pinned-source build/short contracts; these are not the 99-route block."""

import json
from copy import deepcopy
from itertools import pairwise

import numpy as np
import pytest
from battery_half_cell_dfn45_cases import (
    CASE_IDS,
    RESPONSE_ORDER,
    SOURCE_F_C_MOL,
    SOURCE_VARIANTS,
    XU2019_PROJECTION,
    case_data,
    native_initial_state,
    native_spec,
    planned_denominators,
    routes,
)
from battery_half_cell_dfn45_pybamm import (
    build_reference,
    initial_state_record,
    reference_time,
    source_contract,
    stage_inputs,
)

SMALL_GRID = {"x": (3, 3), "r": 4}


@pytest.mark.parametrize("variant", SOURCE_VARIANTS)
def test_observation_grid_is_not_a_sequence_of_internal_source_restarts(monkeypatch, variant):
    import pybamm

    data = case_data("H02")
    data["purpose"] = "sampling API regression; not a formal reference history"
    data["stages"] = (
        {"duration_s": 0.02, "target": 0.0024, "unit": "A", "samples": 5, "expected": "complete"},
        {"duration_s": 0.02, "target": 0.0, "unit": "A", "samples": 5, "expected": "complete"},
    )
    original_step = pybamm.Simulation.step
    calls = []

    def observe_step(simulation, duration, *args, **kwargs):
        calls.append((duration, deepcopy(kwargs)))
        return original_step(simulation, duration, *args, **kwargs)

    monkeypatch.setattr(pybamm.Simulation, "step", observe_step)
    trace = reference_time(data, SMALL_GRID, variant)
    assert len(calls) == len(trace["segments"]) == 2
    for index, (duration, kwargs) in enumerate(calls):
        # IDAKLU restarts at each t_eval. Only stage boundaries are physical
        # discontinuities; requested observation times belong in t_interp.
        np.testing.assert_array_equal(kwargs["t_eval"], [0.0, duration])
        np.testing.assert_array_equal(kwargs["t_interp"], np.linspace(0.0, duration, 5))
        segment = trace["segments"][index]
        requested_start = segment["requested_start_time_s"]
        expected_times = requested_start + np.linspace(0.0, duration, 5)
        if index:
            expected_times[0] = np.nextafter(requested_start, np.inf)
            rhs_size = trace["contract"]["rhs_size"]
            assert segment["boundary_t0"]["source_y"][:rhs_size] == trace["segments"][
                index - 1
            ]["records"][-1]["source_y"][:rhs_size]
        np.testing.assert_array_equal(segment["original_reported_times_s"], expected_times)
        np.testing.assert_array_equal(
            np.concatenate(segment["original_all_ts"]), expected_times
        )
        assert segment["boundary_t0"]["time_s"] == requested_start
        assert segment["actual_end_time_s"] == requested_start + duration
        assert segment["termination"] == "final time"
        assert len(segment["records"]) == 5


def test_prospective_matrix_keeps_exact_route_and_stage_denominators():
    counts = planned_denominators()
    assert counts == {
        "units": 6,
        "histories": 99,
        "requested_stages": 189,
        "requested_sample_frames_before_events": 4329,
        "by_implementation": {
            "native": 18,
            "modular-full-Xu2019": 18,
            "modular-constant-transport": 18,
            "basic-full-raw-metal": 15,
            "basic-full-pure-metal": 15,
            "basic-constant-transport-pure-metal": 15,
        },
    }
    assert all(
        route["unit"] != "H04" for route in routes() if route["implementation"].startswith("basic")
    )


@pytest.mark.parametrize("identity", CASE_IDS)
def test_real_xu_projection_keeps_geometry_kinetics_ocp_barrier_and_unique_state(identity):
    from chemworld.physchem.battery_half_cell_dfn import validate_half_cell_dfn_state

    data = case_data(identity)
    spec = native_spec(data, SMALL_GRID)
    initial = native_initial_state(data, spec)
    validate_half_cell_dfn_state(spec, initial)
    assert spec.cv_monotone_certified
    assert spec.positive.maximum_concentration_mol_m3 == 48230.0
    assert spec.initial_positive_fraction == 4631 / 48230
    assert spec.positive_ocv.endpoint_policy.barrier_V == 1e-6
    assert spec.positive_ocv.endpoint_policy.stoichiometry_clip == 1e-10
    assert spec.electrolyte.positive_solid_conductivity_S_m == pytest.approx(100 * 0.518**1.5)
    assert spec.metal.native_parameters["exchange_current_density_A_m2"] == pytest.approx(
        70.59419558726486, abs=1e-12
    )
    assert spec.metal.native_parameters["electrolyte_concentration_mol_m3"] == 1000.0
    assert spec.working_reference_concentration_mol_m3 == 1000.0
    assert spec.metal.native_parameters["thickness_m"] == (1e-7 if identity == "H05" else 0.0007)
    assert sum(initial.amounts_mol[k] for k in spec.electrolyte.keys) == pytest.approx(
        sum(spec.electrolyte.initial_amounts().values()), abs=1e-15
    )


@pytest.mark.parametrize("variant", SOURCE_VARIANTS)
def test_actual_source_variants_keep_all_66_released_entries_and_original_events(variant):
    built = build_reference(case_data("H02"), SMALL_GRID, variant)
    contract = source_contract(built)
    assert json.loads(json.dumps(contract))["variant"] == variant
    assert len(contract["source_parameter_manifest"]) == 66
    assert contract["source_parameter_manifest"]["Positive electrode OCP [V]"]["function_id"] == (
        "nmc_ocp_Xu2019"
    )
    assert "4.3452" in contract["source_parameter_manifest"]["Positive electrode OCP [V]"]["source"]
    assert (
        "does not claim to be representative"
        not in contract["source_parameter_manifest"]["Positive electrode OCP [V]"]["source"]
    )
    assert contract["original_ocp_and_barrier_preserved"]
    assert contract["endpoint_contract"]["clip"] == 1e-10
    originals = {(e["name"], e["type"]) for e in contract["original_events"]}
    assert originals <= {(e["name"], e["type"]) for e in contract["all_events"]}
    assert len(contract["all_events"]) == len(originals) + 4
    updates = contract["material_parameter_updates"]
    assert "Positive electrode OCP [V]" not in updates
    expected = set()
    if "constant-transport" in variant:
        expected.update(("Electrolyte diffusivity [m2.s-1]", "Electrolyte conductivity [S.m-1]"))
    if variant.startswith("basic") and variant.endswith("pure-metal"):
        expected.add("Exchange-current density for lithium metal electrode [A.m-2]")
    assert set(updates) == expected
    assert contract["mesh"]["separator"]["cells"] == 3
    assert contract["mesh"]["positive electrode"]["cells"] == 3
    assert contract["mesh"]["positive particle"]["cells"] == 4
    assert contract["mesh"]["separator"]["original_edges_m"][0] == 0.0007
    assert contract["mesh"]["separator"]["local_edges_m"][0] == 0.0
    policy = contract["algebraic_conditioning"]
    assert policy["root_method"] == "casadi"
    assert "not unmodified source default" in policy["status"]
    assert sum(block["rows"] for block in policy["blocks"]) == contract["algebraic_size"]
    for block in policy["blocks"]:
        expected_factor = (6.7e-5 / 0.000767) ** 2 if block["charge_equation"] else 1.0
        assert block["factor"] == pytest.approx(expected_factor, rel=2e-15)
        assert block["factor"] > 0
        assert block["actual_row_factors"] == [block["factor"]] * block["rows"]
        assert block["original_continuous_expression"]
        assert block["conditioned_discretized_expression"]
    if variant.startswith("basic"):
        assert contract["work_contract"]["added_observer"]
        assert contract["work_contract"]["no_feedback"]
    else:
        assert not contract["work_contract"]["added_observer"]
        assert {"Discharge energy [W.h]", "Throughput energy [W.h]"} <= set(
            contract["original_rhs_names"]
        )


def test_basic_cross_reference_rejects_cv_instead_of_substituting_a_modular_model():
    with pytest.raises(ValueError, match="CC/rest/reverse"):
        build_reference(case_data("H04"), SMALL_GRID, "basic-full-pure-metal")


@pytest.mark.parametrize(
    "variant", ["basic-full-raw-metal", "basic-full-pure-metal", "modular-full-Xu2019"]
)
def test_actual_t0_metal_j0_uses_the_recorded_basic_or_pure_metal_argument(variant):
    data = case_data("H02")
    built = build_reference(data, SMALL_GRID, variant)
    record = initial_state_record(built, data["stages"][0])
    row = dict(zip(RESPONSE_ORDER, record["row"], strict=True))
    assert record["time_s"] == 0.0
    assert record["differential_state_unchanged_by_boundary_root"]
    assert row["charge_C"] == row["work_J"] == 0.0
    assert row["current_A"] == pytest.approx(0.0024, abs=1e-10)
    cmetal = 48230.0 if variant == "basic-full-raw-metal" else 1 / 1.3e-5
    assert row["metal_j0_A_m2"] == pytest.approx(
        3.5e-8 * SOURCE_F_C_MOL * cmetal**0.7 * row["metal_boundary_ce_mol_m3"] ** 0.3, rel=2e-12
    )
    assert max(abs(x) for x in record["source_algebraic"]) < 1e-8
    assert len(record["source_algebraic_original"]) == len(record["source_algebraic"])
    assert record["physical_current_closure"]["maximum_absolute_A"] < 1e-8
    for theta, actual in zip(
        record["fields"]["theta_surface"], record["fields"]["positive_ocp_V"], strict=True
    ):
        raw = float(built["parameter_values"]["Positive electrode OCP [V]"](theta))
        assert actual == pytest.approx(raw + 1e-6 * (1 / theta + 1 / (theta - 1)), abs=2e-12)
    assert len(record["fields"]["ionic_current_A_m2"]) == sum(SMALL_GRID["x"]) + 1
    assert len(record["fields"]["solid_current_A_m2"]) == SMALL_GRID["x"][1] + 1
    assert np.asarray(record["fields"]["cs_mol_m3_x_r"]).shape == (3, 4)
    assert len(record["source_y"]) == built["model"].len_rhs + built["model"].len_alg


def test_nonuniform_salt_is_a_declared_state_not_a_changed_reference_parameter():
    data = case_data("H06")
    built = build_reference(data, SMALL_GRID)
    record = initial_state_record(built, data["stages"][0])
    spec = native_spec(data, SMALL_GRID)
    initial = native_initial_state(data, spec)
    assert built["parameter_values"]["Initial concentration in electrolyte [mol.m-3]"] == 1000.0
    assert len(built["initial_condition_updates"]) == 1
    assert data["initial_salt_direction"] == -1
    assert data["initial_salt_span_mol_m3"] == 500.0
    np.testing.assert_allclose(
        record["fields"]["ce_mol_m3"], spec.electrolyte.concentrations(initial), atol=1e-10
    )
    coefficients = record["native_projection_difference"]
    assert coefficients["D_e_range_m2_s"][0] < coefficients["D_e_range_m2_s"][1]
    assert coefficients["kappa_range_S_m"][0] < coefficients["kappa_range_S_m"][1]
    projected = build_reference(data, SMALL_GRID, "modular-constant-transport")
    projected_t0 = initial_state_record(projected, data["stages"][0])
    assert (
        projected_t0["native_projection_difference"]["D_e_range_m2_s"]
        == [XU2019_PROJECTION["frozen_D_m2_s"]] * 2
    )


@pytest.mark.parametrize("variant", ["modular-full-Xu2019", "basic-constant-transport-pure-metal"])
def test_micro_source_advance_keeps_t0_full_y_events_mesh_and_signed_work(variant):
    data = case_data("H02")
    data["purpose"] = "short functional contract; not a formal route"
    data["stages"] = (
        {"duration_s": 0.02, "target": 0.0024, "unit": "A", "samples": 3, "expected": "complete"},
    )
    trace = reference_time(data, SMALL_GRID, variant)
    segment = trace["segments"][0]
    assert segment["termination"] == "final time"
    assert segment["actual_end_time_s"] == 0.02
    assert segment["boundary_t0"]["time_s"] == 0.0
    assert segment["boundary_t0"]["differential_state_unchanged_by_boundary_root"]
    assert len(segment["records"]) == 3
    rows = np.asarray([r["row"] for r in segment["records"]])
    assert rows[-1, 2] == pytest.approx(0.0024 * 0.02, abs=1e-10)
    assert rows[-1, 3] > 0.0
    assert rows[-1, 4] - rows[0, 4] == pytest.approx(-rows[-1, 2] / 96485.33212, abs=1e-12)
    assert np.asarray(segment["original_all_ys"][0]).shape[1] == 3
    assert all(
        len(r["source_events"]) == len(trace["contract"]["all_events"]) for r in segment["records"]
    )
    assert all(
        r["physical_current_closure"]["maximum_absolute_A"] < 1e-8 for r in segment["records"]
    )
    json.dumps(trace)


def test_micro_mixed_current_rest_cv_keeps_each_original_stage_boundary():
    data = case_data("H04")
    data["purpose"] = "short functional mixed-mode contract; not a formal route"
    data["stages"] = tuple(dict(stage, duration_s=0.01, samples=2) for stage in data["stages"])
    trace = reference_time(data, SMALL_GRID, "modular-constant-transport")
    assert trace["actual_stages"] == 3
    for previous, following in pairwise(trace["segments"]):
        boundary = following["boundary_t0"]
        assert boundary["time_s"] == previous["actual_end_time_s"]
        assert boundary["differential_state_unchanged_by_boundary_root"]
        nrhs = trace["contract"]["rhs_size"]
        assert boundary["source_y"][:nrhs] == previous["records"][-1]["source_y"][:nrhs]
    final = dict(zip(RESPONSE_ORDER, trace["segments"][-1]["records"][-1]["row"], strict=True))
    assert abs(final["current_A"]) <= data["maximum_current_A"] + 1e-10


def test_failure_keeps_the_completed_prefix_and_exact_failed_boundary(monkeypatch):
    import battery_half_cell_dfn45_pybamm as adapter

    data = case_data("H03")
    data["purpose"] = "short functional failure-retention contract"
    data["stages"] = tuple(dict(s, duration_s=0.01, samples=2) for s in data["stages"][:2])
    original = adapter.initial_state_record
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("injected second-boundary failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "initial_state_record", fail_second)
    with pytest.raises(RuntimeError, match="injected second-boundary") as caught:
        reference_time(deepcopy(data), SMALL_GRID)
    partial = caught.value.partial_history
    assert len(partial["segments"]) == 1
    assert partial["failed_stage"] == 1
    assert partial["last_completed_time_s"] == 0.01
    assert partial["segments"][0]["records"][-1]["source_y"]


def test_extraction_failure_keeps_the_entire_original_return_before_field_derivation(monkeypatch):
    import battery_half_cell_dfn45_pybamm as adapter

    data = case_data("H02")
    data["purpose"] = "short extraction-failure retention contract"
    data["stages"] = (dict(data["stages"][0], duration_s=0.01, samples=2),)
    original, calls, progress = adapter.state_record, [], []

    def fail_extraction(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("injected post-return extraction failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "state_record", fail_extraction)
    with pytest.raises(RuntimeError, match="post-return extraction") as caught:
        reference_time(data, SMALL_GRID, progress_callback=progress.append)
    partial = caught.value.partial_history
    returned = partial["failed_source_return"]
    assert partial["segments"] == []
    assert partial["failed_boundary_t0"]["source_y"]
    assert returned["times_s"] == [0.0, 0.01]
    assert len(returned["source_y"][0]) == 2
    assert returned["all_ys"]
    assert returned["termination"] == "final time"
    assert progress[-1]["phase"] == "source-solution-returned"


def test_failed_source_root_preserves_the_raw_input_without_inventing_a_returned_trial(monkeypatch):
    data = case_data("H02")
    built = build_reference(data, SMALL_GRID)

    def fail(*args, **kwargs):
        raise RuntimeError("injected upstream root failure")

    monkeypatch.setattr(built["simulation"].solver, "calculate_consistent_state", fail)
    with pytest.raises(RuntimeError, match="upstream root") as caught:
        initial_state_record(built, data["stages"][0])
    attempt = caught.value.source_boundary_attempt
    assert len(attempt["initial_source_y"]) == built["model"].len_rhs + built["model"].len_alg
    assert len(attempt["initial_source_algebraic"]) == built["model"].len_alg
    assert attempt["root_tol"] == 1e-8
    assert attempt["failed_final_trial"] == "not exposed by the upstream failed root solver"
    json.dumps(attempt)


def test_native_ocp_uses_the_exact_released_polynomial_exponential_and_original_barrier():
    data = case_data("H02")
    spec = native_spec(data, SMALL_GRID)
    built = build_reference(data, SMALL_GRID)
    parameters, p = built["parameter_values"], built["model"].param
    for theta in (-0.2, 0.0, 0.1, 0.7, 1.0, 1.2):
        actual_source = float(parameters.evaluate(p.p.prim.U(theta, p.T_init)))
        assert spec.positive_ocv.derivatives(theta, trial=True)[0] == pytest.approx(
            actual_source, rel=2e-13, abs=2e-12
        )
    assert "does not claim to be representative" in built["released_parameter_factory_source"]


@pytest.mark.parametrize("variant", ["modular-full-Xu2019", "basic-full-pure-metal"])
def test_actual_mesh_inventory_and_pde_distinguish_lx_conditioning_from_physical_metal_ir(variant):
    data = case_data("H06")
    data["stages"] = (dict(data["stages"][0], target=0.0),)
    built = []
    for lm in (0.0007, 1e-7):
        other = dict(data, metal_thickness_m=lm)
        source = build_reference(other, SMALL_GRID, variant)
        initial = initial_state_record(source, other["stages"][0])
        contract = source_contract(source)
        geometry = contract["inventory_geometry_contract"]
        physical = dict(zip(RESPONSE_ORDER, initial["row"], strict=True))["electrolyte_salt_mol"]
        reported = initial["original_source_reported_inventory"][
            "source_reported_electrolyte_Li_mol"
        ]
        assert reported / physical == pytest.approx(
            geometry["source_reported_to_physical_salt_factor"], rel=2e-14
        )
        assert geometry["actual_salt_domain_length_m"] == pytest.approx(6.7e-5, abs=1e-18)
        assert physical == pytest.approx(3.642408e-6, abs=1e-18)
        assert contract["F_C_mol"] == SOURCE_F_C_MOL
        assert contract["F_C_mol"] != contract["native_F_C_mol"]
        assert geometry["source_concentration_scale_mol_m3"] == 1000.0
        built.append((source, initial, contract))
    raw = np.asarray(built[0][1]["source_y"]).reshape(-1, 1)
    # Curved potentials keep all cell divergence residuals far from zero, so
    # the relative operator check does not divide a cancellation by itself.
    for variable, slices in built[0][0]["model"].y_slices.items():
        if "potential" in variable.name and slices:
            for sl in slices:
                raw[sl] += (
                    0.0002 + 0.0008 * np.linspace(0.0, 1.0, sl.stop - sl.start) ** 2
                ).reshape(-1, 1)
    evaluated = []
    for source, _, contract in built:
        inputs = stage_inputs(data["stages"][0], variant)

        def evaluate(expression, inputs=inputs):
            return np.asarray(expression.evaluate(t=0, y=raw, inputs=inputs), dtype=float)

        evaluated.append(
            (
                evaluate(source["model"].concatenated_rhs),
                {
                    block["name"]: evaluate(source["model"].variables[block["original_alias"]])
                    / (contract["inventory_geometry_contract"]["source_Lx_m"] ** 2)
                    for block in source["algebraic_conditioning"]
                    if block["charge_equation"]
                },
            )
        )
    np.testing.assert_allclose(evaluated[0][0], evaluated[1][0], rtol=1e-9, atol=1e-10)
    for name, residual in evaluated[0][1].items():
        assert np.max(np.abs(residual)) > 1e6
        np.testing.assert_allclose(residual, evaluated[1][1][name], rtol=1e-9, atol=1e-5)
