"""Fixed coverage, independent model construction and failure bookkeeping; no block execution."""

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import run_battery_thevenin_block as runner
from battery_thevenin_cases import CASES, TIMES, native_case
from battery_thevenin_traces import BASE_BOUNDS, build_reference, response_bounds, response_names

from chemworld.physchem.battery_thevenin import TheveninSpec


def test_fixed_fourteen_units_and_unweakened_boundaries():
    assert len(CASES) == 14
    assert [(c["rc_count"], c["target"]) for c in CASES[:8]] == [
        (n, i) for n in (0, 1, 2, 4) for i in (.5, -.5)]
    assert [c["target"] for c in CASES[8:]] == [3.5, 3.9, .5, -.5, 3.3, 4.]
    assert [c["feedback"] for c in CASES] == [False]*10+[True]*2+[False]*2
    assert [c["limit_A"] for c in CASES] == [10.]*12+[.1]*2
    assert np.array_equal(TIMES, np.arange(0, 10.5, .5))
    assert BASE_BOUNDS == (1e-6, 1e-6, 1e-7, 1e-5, 1e-5, 1e-5, 5e-5,
                           1e-5, 5e-5, 1e-5, 1e-5)


@pytest.mark.parametrize("index", range(1, 15))
def test_authored_inputs_roundtrip_and_state_domain(index):
    spec, program = native_case(index)
    p = spec.native_parameters
    assert TheveninSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).to_dict() == (
        spec.to_dict())
    assert len(spec.rc_elements) == CASES[index-1]["rc_count"]
    assert p["initial_cell_temperature_K"] == p["initial_jig_temperature_K"] == 310.
    assert p["capacity_C"] == 100. and p["initial_soc"] == .7
    assert program.value_at(0.) == CASES[index-1]["target"]
    assert program.unit == CASES[index-1]["unit"]
    assert len(response_names(index)) == len(response_bounds(index)) == 11+len(spec.rc_elements)
    assert spec.parameters.sources[0].provenance.source_id == (
        "chemworld:thevenin-qualification-synthetic")
    if CASES[index-1]["feedback"]:
        point = {"soc": .31, "temperature_K": 319., "current_A": -.4}
        assert spec.series_resistance.value(point) == pytest.approx(
            .08+.02*.31+.0001*(319-298.15)+.002*(-.4), abs=1e-14)
        assert len(spec.series_resistance.table.rows) == 27


@pytest.mark.parametrize("index", range(1, 15))
def test_released_model_processes_only_explicit_authored_parameters(index):
    pybamm = pytest.importorskip("pybamm")
    assert pybamm.__version__ == "25.4.2"
    model, parameters = build_reference(index)
    assert parameters["Cell capacity [A.h]"] == 100/3600
    assert parameters["Initial SoC"] == .7
    assert parameters["Initial temperature [K]"] == 310.
    assert len(model.rhs) == CASES[index-1]["rc_count"]+9
    assert len(model.algebraic) == int(CASES[index-1]["unit"] == "V")
    assert sum(k.startswith("Qualification integral: ") for k in model.variables) == 6
    parameters.process_model(model, inplace=False).check_well_posedness()


def trace(index=3):
    rows = np.zeros((21, len(response_names(index))))
    rows[:, 0], rows[:, 2], rows[:, 3:5] = 3.7, .7, 310.
    return {"times_s": TIMES.tolist(), "rows": rows.tolist(), "outcome": "success",
            "end_time_s": 10., "maximum_charge_balance_C": 0.,
            "maximum_node_heat_balance_J": 0., "chemical_total_energy_residual_J": None}


def record():
    return {"native": [trace(), trace()], "reference": [trace(), trace()]}


def test_complete_fixture_and_each_original_response_bound_are_enforced():
    original = record()
    runner.assess(original, 3)
    assert original["passed"]
    for column, bound in enumerate(response_bounds(3)):
        changed = record()
        changed["reference"][1]["rows"][4][column] += bound*1.1
        runner.assess(changed, 3)
        assert not changed["passed"] and not changed["checks"]["native_reference"]


@pytest.mark.parametrize("variant", ("native", "reference"))
def test_precision_checks_cannot_be_replaced_by_cross_model_accuracy(variant):
    data = record()
    data[variant][0]["rows"][4][0] += .8e-6
    runner.assess(data, 3)
    assert data["checks"]["native_reference"]
    assert not data["passed"] and not data["checks"][variant+"_precision"]


@pytest.mark.parametrize("defect", ("missing", "exception", "time", "nonfinite", "charge",
                                     "heat", "chemical", "current", "temperature", "voltage"))
def test_incomplete_or_invalid_runs_never_count_as_passes(defect):
    data = record()
    row = data["native"][0]
    if defect == "missing":
        data["reference"].pop()
    elif defect == "exception":
        data["reference"][0] = {"exception": "retained failure"}
    elif defect == "time":
        row["times_s"].pop()
    elif defect == "nonfinite":
        row["rows"][2][0] = float("nan")
    elif defect == "charge":
        row["maximum_charge_balance_C"] = 2e-8
    elif defect == "heat":
        row["maximum_node_heat_balance_J"] = 2e-6
    elif defect == "chemical":
        row["chemical_total_energy_residual_J"] = 0.
    elif defect == "current":
        row["rows"][1][1] = 11.
    elif defect == "voltage":
        row["rows"][1][0] = 11.
    else:
        row["rows"][1][3] = 401.
    runner.assess(data, 3)
    assert not data["passed"]


def test_four_runs_attempted_after_the_first_solver_failure(monkeypatch):
    calls, checkpoints = [], []

    def native(index, tight):
        calls.append(("native", tight))
        if not tight:
            raise RuntimeError("synthetic retained failure")
        return trace(index)

    def reference(index, tight):
        calls.append(("reference", tight))
        return trace(index)

    monkeypatch.setattr(runner, "native_trace", native)
    monkeypatch.setattr(runner, "reference_trace", reference)
    data = {}
    runner.run_unit(data, 3, lambda _: None, lambda: checkpoints.append(True))
    assert calls == [("native", False), ("native", True), ("reference", False), ("reference", True)]
    assert len(checkpoints) == 4 and not data["passed"]
    assert "synthetic retained failure" in data["native"][0]["exception"]


def test_cli_retains_full_denominator_and_rejects_output_replacement(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pybamm", SimpleNamespace(__version__="25.4.2"))
    output = tmp_path / "contract-fixture-only.json"
    monkeypatch.setattr(sys, "argv", ["thevenin", str(output)])

    def run(data, index, progress, checkpoint):
        data.update(passed=index != 2, fixture_only=True)
        checkpoint()

    monkeypatch.setattr(runner, "run_unit", run)
    assert runner.main() == 1
    data = json.loads(output.read_text())
    assert data["denominator"] == len(data["units"]) == 14
    assert data["passed"] == 13 and "active_unit" not in data
    saved = deepcopy(data)
    with pytest.raises(FileExistsError):
        runner.main()
    assert json.loads(output.read_text()) == saved
