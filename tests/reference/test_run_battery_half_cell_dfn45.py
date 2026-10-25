"""Tool fixtures are synthetic mechanics, never independent battery evidence."""

import json
import os
import signal
from copy import deepcopy

import numpy as np
import pytest
import run_battery_half_cell_dfn45 as runner
from battery_half_cell_dfn45_cases import case_data, routes


@pytest.fixture
def comparison_fixture(monkeypatch):
    """Mock route physics assessment so graph/aggregation tests never simulate99 routes.

    Actual domain/raw-data gates are separately tested against tiny genuine
    native/source histories below. These artificial views test only comparison
    bookkeeping, norm units, endpoints and preservation of the full denominator.
    """
    views = {}
    for route in routes():
        data = case_data(route["unit"])
        metal = runner.AREA * data["metal_thickness_m"] / 1.3e-5
        positive = runner.AREA * runner.LP * 0.518 * 4631
        salt = 3.642408e-6
        time_s, charge, segments = 0.0, 0.0, []
        for stage in data["stages"]:
            duration = stage["duration_s"] if stage["expected"] == "complete" else 47.625
            current = stage["target"] if stage["unit"] == "A" else 0.001
            times = np.asarray([time_s, time_s + duration])
            q = charge + current * (times - time_s)
            voltage = stage["target"] if stage["unit"] == "V" else 4.0
            rows = np.tile(
                [voltage, current, 0, 0, metal, positive, salt, 1000, 70, 0.001, 0], (2, 1)
            ).astype(float)
            rows[:, 2], rows[:, 3] = q, 4 * q
            rows[:, 4], rows[:, 5] = metal - q / runner.F, positive + q / runner.F
            profile_values = {
                "surface_fraction": 0.2,
                "radial_fraction": 0.2,
                "salt_mol_m3": 1000.0,
                "electrolyte_potential_V": -0.01,
                "solid_potential_V": 4.0,
                "positive_ocp_V": 4.01,
                "positive_eta_V": -0.02,
                "positive_j0_A_m2": 3.0,
                "local_current_A": -current / 10,
                "ionic_face_current_A": current,
                "solid_face_current_A": current,
                "diffusive_salt_equivalent_A": current,
                "cation_flux_equivalent_A": current,
            }
            segments.append(
                {
                    "times": times,
                    "rows": rows,
                    "profiles": {
                        name: np.full((2, 2), value) for name, value in profile_values.items()
                    },
                    "stop": stage["expected"],
                    "raw": {"termination": stage["expected"]},
                    "contract": {
                        "F_C_mol": runner.F,
                        "variant": route["implementation"],
                        "case": data,
                        "finite_metal_contract": {"initial_mol": metal},
                        "inventory_geometry_contract": {
                            "actual_salt_pore_volumes_m3": [salt / 1000]
                        },
                    },
                }
            )
            charge += current * duration
            time_s += duration
        views[runner.route_id(route)] = segments

    def pass_route(route, trace):
        key = runner.route_id(route)
        checks = [runner._check(key + "/" + name, True) for name in runner.ROUTE_CHECKS]
        checks += [
            runner._check(f"{key}/stage-{i}/{name}", True)
            for i, _ in enumerate(case_data(route["unit"])["stages"])
            for name in runner.STAGE_CHECKS
        ]
        return {
            "route": route,
            "checks": checks,
            "passed": True,
            "physical_initial_source_stop": False,
            "available_physical_stages": len(views[key]),
        }, views[key]

    monkeypatch.setattr(runner, "_route_assessment", pass_route)
    return views


def test_fixed_comparison_graph_and_all_missing_denominator_are_exact():
    plan = runner.plan()
    assert (
        plan["histories"],
        plan["requested_stages"],
        plan["requested_sample_frames_before_events"],
    ) == (99, 189, 4329)
    assert (
        plan["strict_cross_edges"],
        plan["strict_self_refinement_edges"],
        plan["strict_comparison_stages"],
    ) == (48, 66, 216)
    assert (plan["model_discrepancy_edges"], plan["model_discrepancy_stages"]) == (51, 99)
    assessment = runner.assess_histories({})
    assert assessment["evaluated_checks"] == assessment["failed_checks"] == 20988
    assert len({check["name"] for check in assessment["checks"]}) == 20988
    assert not assessment["all_passed"]


def test_synthetic_comparison_graph_keeps_complete_success_denominator(comparison_fixture):
    result = runner.assess_histories({})
    assert result["all_passed"]
    assert result["evaluated_checks"] == 20988
    assert len(result["comparisons"]) == 114
    assert sum(len(c["stages"]) for c in result["comparisons"]) == 216
    assert len(result["model_discrepancies"]) == 51


def test_coarse_cross_failure_cannot_be_canceled_by_fine_or_self_reports(comparison_fixture):
    comparison_fixture["H06/native/coarse"][0]["rows"][:, 0] += 0.01
    result = runner.assess_histories({})
    coarse = next(
        c
        for c in result["comparisons"]
        if c["unit"] == "H06" and c["left"] == ("native", "coarse") and c["kind"] == "cross"
    )
    fine = next(
        c
        for c in result["comparisons"]
        if c["unit"] == "H06" and c["left"] == ("native", "fine") and c["kind"] == "cross"
    )
    assert not result["all_passed"] and not coarse["passed"] and fine["passed"]
    assert result["evaluated_checks"] == 20988


def test_self_refinement_gate_is_half_absolute_and_relative_not_an_error_ratio(comparison_fixture):
    comparison_fixture["H02/native/middle"][0]["rows"][:, 0] += 0.0016
    result = runner.assess_histories({})
    cross = next(
        c
        for c in result["comparisons"]
        if c["unit"] == "H02" and c["kind"] == "cross" and c["left"] == ("native", "middle")
    )
    self_edge = next(
        c
        for c in result["comparisons"]
        if c["unit"] == "H02"
        and c["kind"] == "self-refinement"
        and c["left"] == ("native", "coarse")
    )
    assert cross["passed"] and not self_edge["passed"]
    norm = next(c for c in self_edge["stages"][0]["checks"] if c["name"].endswith("t0:voltage_V"))
    assert norm["absolute_gate"] == 0.001 and norm["relative_gate"] == 0.0001


def test_full_transport_and_raw_metal_differences_remain_descriptive(comparison_fixture):
    for key, stages in comparison_fixture.items():
        if "/basic-full-raw-metal/" in key:
            for stage in stages:
                stage["rows"][:, 8] += 1000
    result = runner.assess_histories({})
    assert result["all_passed"]
    discrepancy = next(
        c for c in result["model_discrepancies"] if c["left"] == ("basic-full-raw-metal", "coarse")
    )
    assert discrepancy["passed"] is None
    metric = discrepancy["stages"][0]["descriptive_metrics"]["t0:metal_j0_A_m2"]
    assert metric["maximum_absolute"] == 1000 and metric["passed"] is False
    assert discrepancy["stages"][0]["checks"] == []


def test_true_tail_endpoint_and_stop_identity_remain_in_strict_comparisons(comparison_fixture):
    view = comparison_fixture["H05/native/fine"][0]
    view["times"][-1] += 2
    view["rows"][-1, 3] += 10
    view["stop"] = "minimum-voltage"
    result = runner.assess_histories({})
    pair = next(
        c
        for c in result["comparisons"]
        if c["unit"] == "H05" and c["kind"] == "cross" and c["left"] == ("native", "fine")
    )["stages"][0]
    assert pair["tail_audit"]["left"]["unpaired_duration_s"] == 2
    assert any(not c["passed"] and c["name"].endswith("endpoint:work_J") for c in pair["checks"])
    assert any(not c["passed"] and c["name"].endswith("termination-kind") for c in pair["checks"])
    assert result["evaluated_checks"] == 20988


@pytest.mark.parametrize("message", [123, [], {}, None])
def test_malformed_source_exception_cannot_abort_the_fixed_denominator(message):
    trace = {
        "exception": {"type": "SolverError", "message": message},
        "partial_history": {
            "failed_boundary_t0": {},
            "contract": {},
            "failed_stage": 0,
            "failed_request": {},
        },
    }
    result = runner.assess_histories({"H01/modular-full-Xu2019/coarse": trace})
    assert result["evaluated_checks"] == 20988
    assert not result["all_passed"]
    assert runner._view(trace)[2] is False


@pytest.fixture(scope="module")
def short_source():
    from battery_half_cell_dfn45_pybamm import reference_time

    data = case_data("H02")
    data["purpose"] = "development runner micro-test, not a formal route"
    data["stages"] = (dict(data["stages"][0], duration_s=0.02, samples=3),)
    return data, reference_time(data, {"x": (6, 8), "r": 8}, "modular-constant-transport")


def test_real_source_raw_layout_full_y_and_physical_checks(short_source):
    data, trace = short_source
    contract, segment = trace["contract"], trace["segments"][0]
    assert runner._source_complete(segment["boundary_t0"], contract, boundary=True)
    view = runner._stage_view(segment, contract)
    checks = runner._stage_checks(view, None, view["rows"][0], data["stages"][0], 0, contract, data)
    assert all(checks.values()), {key: value for key, value in checks.items() if not value}
    assert runner._raw_retained(view, contract, False)


@pytest.mark.parametrize(
    "damage",
    ["event-ordinal", "raw-y", "raw-rhs", "ce-field", "raw-original-residual", "time-nonfinite"],
)
def test_source_complete_rejects_data_loss_or_tampering(short_source, damage):
    _, trace = short_source
    record, contract = deepcopy(trace["segments"][0]["boundary_t0"]), trace["contract"]
    if damage == "event-ordinal":
        record["source_events"][0]["ordinal"] += 1
    elif damage == "raw-y":
        record["source_y"] = record["source_y"][:-1]
    elif damage == "raw-rhs":
        record["source_rhs"][0] = float("nan")
    elif damage == "ce-field":
        record["fields"]["ce_mol_m3"][0] += 1
    elif damage == "raw-original-residual":
        del record["source_algebraic_original"]
    else:
        record["time_s"] = float("nan")
    assert not runner._source_complete(record, contract, boundary=True)


def test_original_all_y_matrices_must_match_every_exposed_record(short_source):
    _, trace = short_source
    segment = deepcopy(trace["segments"][0])
    segment["original_all_ys"][0][0][1] += 1e-5
    assert not runner._raw_retained(
        runner._stage_view(segment, trace["contract"]), trace["contract"], False
    )


@pytest.fixture(scope="module", params=["modular-constant-transport", "basic-full-pure-metal"])
def multistage_source(request):
    from battery_half_cell_dfn45_pybamm import reference_time

    data = case_data("H02")
    data["purpose"] = "bounded genuine multistage time-identity regression, not qualification"
    cc = dict(data["stages"][0], duration_s=1.0, samples=3)
    data["stages"] = (cc, dict(cc, target=0.0), dict(cc))
    return data, reference_time(data, {"x": (6, 8), "r": 8}, request.param)


def test_real_multistage_source_preserves_same_time_boundary_and_nextafter(multistage_source):
    data, trace = multistage_source
    contract, previous = trace["contract"], None
    for index, segment in enumerate(trace["segments"]):
        view = runner._stage_view(segment, contract)
        boundary, first = segment["boundary_t0"], segment["records"][0]
        if previous is not None:
            assert boundary["time_s"] == previous["raw"]["records"][-1]["time_s"]
            assert first["time_s"] == np.nextafter(boundary["time_s"], np.inf)
            assert np.array_equal(
                boundary["source_y"][: contract["rhs_size"]],
                np.asarray(previous["raw"]["original_all_ys"][-1])[: contract["rhs_size"], -1],
            )
        else:
            assert first["time_s"] == boundary["time_s"] == 0
        checks = runner._stage_checks(
            view,
            previous,
            trace["segments"][0]["boundary_t0"]["row"],
            data["stages"][index],
            float(index),
            contract,
            data,
        )
        assert checks["raw-state-retention"]
        assert checks["boundary-continuity"]
        previous = view


def _source_record_from_original_y(data, contract, segment, record):
    """Re-evaluate a test mutation; never root, integrate or repair an archive."""
    from battery_half_cell_dfn45_pybamm import state_record

    ns, np_ = (contract["mesh"][key]["cells"] for key in ("separator", "positive electrode"))
    built, _ = runner._source_evaluator(
        contract["variant"],
        ns,
        np_,
        contract["mesh"]["positive particle"]["cells"],
        data["metal_thickness_m"],
        json.dumps(runner.json_value(data), sort_keys=True),
    )
    return dict(
        record, **state_record(built, record["time_s"], record["source_y"], segment["inputs"])
    )


def _replace_source_first_record(segment, record):
    segment["records"][0] = record
    segment["original_all_ts"][0][0] = record["time_s"]
    segment["original_reported_times_s"][0] = record["time_s"]
    segment["actual_start_time_s"] = record["time_s"]
    for row, value in zip(segment["original_all_ys"][0], record["source_y"], strict=True):
        row[0] = value


def test_exact_same_time_source_first_state_is_also_supported(multistage_source):
    data, trace = multistage_source
    contract, segment = trace["contract"], deepcopy(trace["segments"][1])
    previous = runner._stage_view(trace["segments"][0], contract)
    # Contract fixture: represent the independently retained actual boundary at
    # exact t0, instead of the pinned nextafter return. It remains separate from
    # the unmodified real multistage regression and is not qualification data.
    _replace_source_first_record(segment, deepcopy(segment["boundary_t0"]))
    assert runner._raw_retained(
        runner._stage_view(segment, contract), contract, False, data, previous=previous
    )


@pytest.mark.parametrize("damage", ["extra-ulp", "backward-ulp", "finite-gap", "initial-nextafter"])
def test_source_first_time_rejects_any_undeclared_gap(multistage_source, damage):
    data, trace = multistage_source
    contract = trace["contract"]
    index = 0 if damage == "initial-nextafter" else 1
    segment = deepcopy(trace["segments"][index])
    previous = runner._stage_view(trace["segments"][0], contract) if index else None
    first = deepcopy(segment["records"][0])
    t0 = segment["boundary_t0"]["time_s"]
    first["time_s"] = (
        np.nextafter(np.nextafter(t0, np.inf), np.inf)
        if damage == "extra-ulp"
        else np.nextafter(t0, -np.inf)
        if damage == "backward-ulp"
        else t0 + 1e-10
        if damage == "finite-gap"
        else np.nextafter(t0, np.inf)
    )
    first = _source_record_from_original_y(data, contract, segment, first)
    _replace_source_first_record(segment, first)
    assert not runner._raw_retained(
        runner._stage_view(segment, contract), contract, False, data, previous=previous
    )


@pytest.mark.parametrize(
    "damage", ["boundary-y", "previous-y", "previous-time", "missing-previous"]
)
def test_source_boundary_identity_uses_original_same_time_endpoint(multistage_source, damage):
    data, trace = multistage_source
    contract, segment = trace["contract"], deepcopy(trace["segments"][1])
    previous = runner._stage_view(deepcopy(trace["segments"][0]), contract)
    if damage == "boundary-y":
        boundary = segment["boundary_t0"]
        i = next(
            v["slices"][0][0]
            for v in contract["state_layout"]
            if v["name"] == "Discharge capacity [A.h]"
        )
        boundary["source_y"][i] = float(np.nextafter(boundary["source_y"][i], np.inf))
        segment["boundary_t0"] = _source_record_from_original_y(data, contract, segment, boundary)
        # The preserved True marker alone must not certify a changed state.
        assert runner._source_complete(
            segment["boundary_t0"], contract, boundary=True, inputs=segment["inputs"], data=data
        )
    elif damage == "previous-y":
        previous["raw"]["original_all_ys"][-1][0][-1] += 1e-10
    elif damage == "previous-time":
        previous["raw"]["original_all_ts"][-1][-1] = float(
            np.nextafter(segment["boundary_t0"]["time_s"], -np.inf)
        )
    else:
        previous = None
    assert not runner._raw_retained(
        runner._stage_view(segment, contract), contract, False, data, previous=previous
    )


@pytest.mark.parametrize("exact_time", [False, True])
def test_source_first_state_rejects_self_consistent_but_discontinuous_raw_y(
    multistage_source, exact_time
):
    data, trace = multistage_source
    contract, segment = trace["contract"], deepcopy(trace["segments"][1])
    previous = runner._stage_view(trace["segments"][0], contract)
    first = deepcopy(segment["boundary_t0"] if exact_time else segment["records"][0])
    i = next(
        v["slices"][0][0]
        for v in contract["state_layout"]
        if v["name"] == "Discharge capacity [A.h]"
    )
    first["source_y"][i] += 1e-8
    first = _source_record_from_original_y(data, contract, segment, first)
    _replace_source_first_record(segment, first)
    assert runner._source_complete(first, contract, inputs=segment["inputs"], data=data)
    assert not runner._raw_retained(
        runner._stage_view(segment, contract), contract, False, data, previous=previous
    )


def test_only_actual_complete_original_initial_events_are_classified():
    from battery_half_cell_dfn45_pybamm import reference_time

    data = case_data("H06")
    data["purpose"] = "reproduce the preserved invalid positive-gradient candidate; no advance"
    data["initial_salt_direction"] = 1
    data["stages"] = (dict(data["stages"][0], duration_s=0.002, samples=2),)
    with pytest.raises(Exception, match="initial conditions") as caught:
        reference_time(data, {"x": (6, 8), "r": 8}, "modular-full-Xu2019")
    failed = {
        "exception": {"type": type(caught.value).__name__, "message": str(caught.value)},
        "partial_history": caught.value.partial_history,
    }
    _, stages, classified = runner._view(failed)
    assert classified and stages[0]["physical_initial_stop"]
    assert runner.raw_saved_counts([failed])["saved_raw_stages"] == 0
    assert stages[0]["original_reported_times_s"] == []
    broken = deepcopy(failed)
    broken["partial_history"]["failed_boundary_t0"]["source_y"].pop()
    assert not runner._view(broken)[2]
    failed["exception"]["type"] = "RouteTimeout"
    assert not runner._view(failed)[2]


@pytest.mark.parametrize(
    "damage",
    [
        "conditioned-residual",
        "original-residual",
        "raw-current",
        "missing-current-layout",
        "theta-profile",
        "source-manifest",
    ],
)
def test_review_false_positive_witnesses_are_rejected(short_source, damage):
    data, trace = short_source
    contract = deepcopy(trace["contract"])
    record = deepcopy(trace["segments"][0]["boundary_t0"])
    if damage == "conditioned-residual":
        record["source_algebraic"] = [1e50] * contract["algebraic_size"]
    elif damage == "original-residual":
        record["source_algebraic_original"] = [1e50] * contract["algebraic_size"]
    elif damage == "raw-current":
        entry = next(
            item for item in contract["state_layout"] if item["name"] == "Current variable [A]"
        )
        record["source_y"][entry["slices"][0][0]] = 0.5
    elif damage == "missing-current-layout":
        contract["state_layout"] = [
            item for item in contract["state_layout"] if item["name"] != "Current variable [A]"
        ]
    elif damage == "theta-profile":
        record["fields"]["theta_x_r"][0][0] += 0.01
    else:
        contract["source_parameter_manifest"]["Positive electrode conductivity [S.m-1]"][
            "value"
        ] = 999
    assert not runner._source_complete(
        record, contract, boundary=True, data=data, inputs=trace["segments"][0]["inputs"]
    )


@pytest.mark.parametrize("damage", ["missing", "string", "wrong-time", "wrong-state"])
def test_actual_source_endpoint_arrays_are_required_even_at_final_time(short_source, damage):
    data, trace = short_source
    segment = deepcopy(trace["segments"][0])
    if damage == "missing":
        segment["original_t_event"] = segment["original_y_event"] = None
    elif damage == "string":
        segment["original_y_event"] = "not the original y"
    elif damage == "wrong-time":
        segment["original_t_event"][0] += 1
    else:
        segment["original_y_event"][0][0] += 1
    view = runner._stage_view(segment, trace["contract"])
    try:
        retained = runner._raw_retained(view, trace["contract"], False, data)
    except (TypeError, ValueError):
        retained = False
    assert not retained


def test_missing_first_stage_cannot_be_replaced_by_a_valid_later_stage(short_source):
    _, trace = short_source
    malformed = {"contract": trace["contract"], "segments": [{}, trace["segments"][0]]}
    route = next(
        r
        for r in routes()
        if r["unit"] == "H06" and r["implementation"] == "modular-constant-transport"
    )
    report, _ = runner._route_assessment(route, malformed)
    assert len(report["checks"]) == len(runner.ROUTE_CHECKS) + 2 * len(runner.STAGE_CHECKS)
    assert not report["passed"]


@pytest.mark.parametrize("mixed", [False, True])
def test_real_native_micro_cc_or_mixed_finite_cv_retains_reports_and_physics(mixed):
    data = case_data("H04" if mixed else "H02")
    data["purpose"] = "development native micro-test, not a formal history"
    data["stages"] = tuple(dict(stage, duration_s=0.002, samples=2) for stage in data["stages"])
    trace = runner.native_time(data, {"x": (6, 8), "r": 8})
    previous, original = None, None
    for index, segment in enumerate(trace["segments"]):
        view = runner._stage_view(segment, trace["contract"])
        original = view["rows"][0] if index == 0 else original
        checks = runner._stage_checks(
            view, previous, original, data["stages"][index], index * 0.002, trace["contract"], data
        )
        assert all(checks.values()), {key: value for key, value in checks.items() if not value}
        assert segment["original_residual_reports"][0]["points"]
        previous = view
    restored = json.loads(json.dumps(runner.json_value(trace)))
    assert restored == runner.json_value(trace)
    route = {
        "unit": data["id"],
        "implementation": "native",
        "grid_level": "tool-micro",
        "grid": {"x": (6, 8), "r": 8},
    }
    report, _ = runner._route_assessment(route, restored)
    assert next(check for check in report["checks"] if check["name"].endswith("/identity"))[
        "passed"
    ]
    # A CV report with finite fake rate values must not pass as port evidence.
    if mixed:
        segment = deepcopy(trace["segments"][-1])
        raw = segment["original_residual_reports"][0]
        index = next(
            i
            for i, variable in enumerate(raw["variables"])
            if variable["name"] == "delivered_work_J"
        )
        raw["points"][0]["ydot"][index] += 1
        view = runner._stage_view(segment, trace["contract"])
        assert not runner._charge_work_rates(view, trace["contract"], data["stages"][-1])


def test_source_archive_roundtrip_assessment_and_fresh_eval_are_canonical(short_source, tmp_path):
    data, trace = short_source
    path = tmp_path / "actual-micro.json.gz"
    runner.archive_new(path, trace)
    restored = runner.read_archive(path)
    assert restored == runner.json_value(trace)
    route = {
        "unit": "H02",
        "implementation": "modular-constant-transport",
        "grid_level": "tool-micro",
        "grid": {"x": (6, 8), "r": 8},
    }
    # Duration remains a declared nonformal subset, so full-case slot checks
    # honestly fail; decoded data must reproduce exactly the same assessment.
    before = runner._route_assessment(route, trace)[0]
    after = runner._route_assessment(route, restored)[0]
    assert runner.json_value(before) == runner.json_value(after)
    assert not before["passed"]
    complete = runner._source_complete(
        restored["segments"][0]["boundary_t0"],
        restored["contract"],
        boundary=True,
        data=data,
        inputs=restored["segments"][0]["inputs"],
    )
    assert complete


@pytest.fixture(scope="module")
def short_native():
    data = case_data("H02")
    data["purpose"] = "bounded native raw-layout regression, not a formal route"
    data["stages"] = (dict(data["stages"][0], duration_s=0.002, samples=2),)
    return data, runner.native_time(data, {"x": (6, 8), "r": 8})


@pytest.mark.parametrize(
    "damage",
    [
        "truncated-layout",
        "equations",
        "indices",
        "inventory-y",
        "inventory-rate",
        "native-snapshot",
        "physical-profile",
    ],
)
def test_complete_native_packer_and_raw_point_binding_reject_review_witnesses(short_native, damage):
    data, original = short_native
    trace = deepcopy(original)
    segment, contract = trace["segments"][0], trace["contract"]
    report = segment["original_residual_reports"][0]
    if damage == "truncated-layout":
        report["variables"] = report["variables"][-7:]
        report["equations"] = []
        for point in report["points"]:
            point["y"], point["ydot"] = point["y"][-7:], point["ydot"][-7:]
    elif damage == "equations":
        report["equations"] = []
    elif damage == "indices":
        report["differential_indices"].pop()
    elif damage == "inventory-y":
        report["points"][-1]["y"][0] += 1e-5
    elif damage == "inventory-rate":
        report["points"][-1]["ydot"][0] += 1e-5
    elif damage == "native-snapshot":
        state = segment["records"][-1]["native_state"]
        key = next(iter(state["amounts_mol"]))
        state["amounts_mol"][key] += 1e-5
    else:
        segment["records"][-1]["fields"]["theta_x_r"][0][0] += 0.01
    view = runner._stage_view(segment, contract)
    assert not runner._raw_retained(view, contract, True, data)
    assert not runner._charge_work_rates(view, contract, data["stages"][0])


def test_signed_discrepancies_record_direction_and_global_inventory_reference(comparison_fixture):
    for level in runner.LEVELS:
        comparison_fixture[f"H03/basic-full-raw-metal/{level}"][1]["rows"][:, 8] += 2
    result = runner.assess_histories({})
    discrepancy = next(
        c
        for c in result["model_discrepancies"]
        if c["unit"] == "H03" and c["left"] == ("basic-full-raw-metal", "coarse")
    )
    stage = discrepancy["stages"][1]
    metric = stage["descriptive_metrics"]["t0:metal_j0_A_m2"]
    assert metric["signed_difference_min"] == metric["signed_difference_max"] == 2
    assert stage["sample_metadata"]["signed_direction"] == "left-minus-right"
    assert "whole-history" in stage["sample_metadata"]["inventory_delta_reference"]
    view = comparison_fixture["H03/native/coarse"][1]
    sampled = runner._sample(view, [view["times"][0]])
    assert abs(sampled["metal_Li_mol"][0]) > 0.1  # Previous CC did not reset the inventory metric.


def test_worker_retains_raw_return_and_prefix_when_post_return_extraction_fails(
    short_source, tmp_path, monkeypatch
):
    import battery_half_cell_dfn45_pybamm as source

    data, original = short_source
    route = {
        "unit": "H02",
        "implementation": "modular-constant-transport",
        "grid_level": "tool-micro",
        "grid": {"x": (6, 8), "r": 8},
    }
    request = tmp_path / "request.json"
    runner.write_new(request, {"route": route, "case": data})
    monkeypatch.setattr(source, "build_reference", lambda *args, **kwargs: {"fixture": True})
    monkeypatch.setattr(source, "source_contract", lambda built: original["contract"])

    def fail(*args, progress_callback, **kwargs):
        source.build_reference()
        payload = {
            "segments": original["segments"],
            "phase": "source-solution-returned",
            "pending_source_solution": {"source_y": [[1, 2]], "all_ys": [[[1, 2]]]},
        }
        progress_callback(payload)
        error = RuntimeError("injected post-return extraction failure")
        error.partial_history = {
            "contract": original["contract"],
            "case": data,
            "segments": original["segments"],
            "failed_source_return": payload,
        }
        raise error

    monkeypatch.setattr(source, "reference_time", fail)
    assert runner._worker(request, tmp_path) == 1
    final = runner.read_archive(tmp_path / "final.json.gz")
    assert final["partial_history"]["segments"][0]["records"]
    assert final["partial_history"]["failed_source_return"]["pending_source_solution"][
        "source_y"
    ] == [[1, 2]]
    checkpoint, unreadable = runner.transport._latest_partial(tmp_path)
    assert not unreadable and checkpoint["contract"]
    assert checkpoint["pending_source_solution"]["all_ys"] == [[[1, 2]]]


def test_verified_exclusive_archives_and_nonfinite_failure_representation_are_reused(tmp_path):
    assert runner.archive_new is runner.transport.archive_new
    path = tmp_path / "failure.json.gz"
    runner.archive_new(path, {"partial": [np.nan, np.inf]})
    saved = path.read_bytes()
    assert runner.read_archive(path)["partial"] == [
        {"__nonfinite_float__": "nan"},
        {"__nonfinite_float__": "inf"},
    ]
    with pytest.raises(FileExistsError):
        runner.archive_new(path, {})
    assert path.read_bytes() == saved


@pytest.mark.skipif(os.name != "posix", reason="route process-group budget is POSIX-only")
def test_posix_budget_kills_the_full_group_and_retains_latest_complete_prefix(
    tmp_path, monkeypatch
):
    directory = tmp_path / "route"
    signals = []

    class Process:
        pid = 987654321
        returncode = None

        def __init__(self, command, **kwargs):
            assert kwargs["start_new_session"] is True
            assert "run_battery_half_cell_dfn45.py" in command[2]
            runner.archive_new(
                directory / "progress-0000.json.gz",
                {"segments": [{"records": []}], "pending_source_solution": {"source_y": [[1, 2]]}},
            )
            (directory / "progress-0001.json.gz").write_bytes(b"corrupt checkpoint")

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            return self.returncode

    def kill(pid, sig):
        assert pid == Process.pid
        signals.append(sig)
        Process.returncode = -int(sig)

    monkeypatch.setattr(runner.subprocess, "Popen", Process)
    monkeypatch.setattr(runner.transport.os, "killpg", kill)
    trace, artifact = runner.execute_route(routes()[0], directory, 0, progress=lambda event: None)
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert artifact["timed_out"] and trace["exception"]["type"] == "RouteTimeout"
    assert trace["partial_history"]["pending_source_solution"]["source_y"] == [[1, 2]]
    assert trace["unreadable_checkpoints"]
    with pytest.raises(FileExistsError):
        runner.execute_route(routes()[0], directory, 0, progress=lambda event: None)


def test_no_execute_prints_plan_and_requires_no_protocol_or_source_solver(capsys):
    assert runner.main([]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["histories"] == 99 and not printed["execution_authorized_by_this_plan"]
    with pytest.raises(ValueError, match="fixed experiment note"):
        runner.main(["--execute"])
    with pytest.raises(ValueError, match="unknown or duplicate"):
        runner.main(["--route", "missing"])
