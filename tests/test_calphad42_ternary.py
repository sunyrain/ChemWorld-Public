"""Finite ternary nodes, no invented boundaries, and actual original-stock consumers."""

import copy
import json
from dataclasses import replace

import pytest

from chemworld.examples.calphad_ternary import build_calphad_ternary_example, ternary_example_spec
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.calphad_equilibrium import CalphadEquilibriumResult
from chemworld.physchem.calphad_ternary_diagram import (
    CalphadTernaryPath,
    CalphadTernaryResult,
    CalphadTernarySpec,
    scan_calphad_ternary_diagram,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.calphad_phase_inventory import CalphadPhaseTarget
from chemworld.runtime.calphad_ternary_inventory import select_calphad_ternary_inventory_point
from chemworld.world.calphad_ternary_analysis import analyze_calphad_ternary_world


@pytest.fixture(scope="module")
def diagrams():
    return {
        family: scan_calphad_ternary_diagram(ternary_example_spec(family))
        for family in ("preferred", "ideal", "regular", "excluded")
    }


@pytest.mark.parametrize("family", ["preferred", "ideal", "regular", "excluded"])
def test_exact_grid_requested_and_paths(diagrams, family):
    result = diagrams[family]
    assert result.complete
    assert len(result.points) == 11
    assert len(result.geometry["grid_cells"]) == 9
    assert len(result.geometry["grid_edges"]) == 18
    assert all(not e["continuous_segment_certified"] for e in result.geometry["grid_edges"])
    assert result.points[10]["composition"] == [0.3, 0.3, 0.4]
    path = result.geometry["paths"][0]
    assert path["complete"] and len(path["edges"]) == 2
    assert 10 in path["point_indices"]
    for p in result.points:
        eq = p["equilibrium"]
        for e, fraction in zip(result.specification.components, p["composition"], strict=True):
            actual = sum(
                row["fraction"] * row["element_fractions"].get(e, 0) for row in eq["phases"]
            )
            assert actual == pytest.approx(fraction, abs=2e-9)
            assert (e in eq["element_potentials_J_mol"]) == (fraction > 0)


def test_ties_preserve_multiplicity_and_exclusion(diagrams):
    preferred = diagrams["preferred"]
    assert len(preferred.points[10]["equilibrium"]["phases"]) == 3
    assert {g["kind"] for g in preferred.geometry["tie_geometry"]} == {"tie_line", "tie_triangle"}
    assert preferred.geometry["boundary_brackets"]
    assert not diagrams["ideal"].geometry["tie_geometry"]
    regular = diagrams["regular"]
    assert any(p["phase_signature"] == ["ALPHA", "ALPHA"] for p in regular.points)
    assert all("CR_PREF" not in p["phase_signature"] for p in diagrams["excluded"].points)
    for result in diagrams.values():
        for tie in result.geometry["tie_geometry"]:
            assert len({v["instance"] for v in tie["vertices"]}) == len(tie["vertices"])
            x = [
                sum(v["fraction"] * v["composition"][i] for v in tie["vertices"]) for i in range(3)
            ]
            assert x == pytest.approx(result.points[tie["point_index"]]["composition"], abs=2e-9)


@pytest.mark.parametrize(
    "change",
    [
        {"grid_intervals": 0},
        {"grid_intervals": True},
        {"grid_intervals": 101},
        {"components": ("FE", "NI", "FE")},
        {"components": ("FE", "NI", "CU")},
        {"requested_compositions": ((0.3, 0.3, 0.3),)},
        {"requested_compositions": ((float("nan"), 0, 1),)},
        {"requested_compositions": ((-0.1, 0.1, 1),)},
        {"temperature_K": float("inf")},
        {"pressure_Pa": 0},
        {"maximum_evaluations": 0},
    ],
)
def test_explicit_scope_and_finite_declaration(change):
    with pytest.raises(ValueError):
        replace(ternary_example_spec(), **change)


def test_path_declaration_and_nearby_points_are_not_rounded_away():
    with pytest.raises(ValueError):
        CalphadTernaryPath("bad", ((1.0, 0.0, 0.0),))
    base = ternary_example_spec("ideal")
    with pytest.raises(ValueError):
        replace(base, paths=(base.paths[0], base.paths[0]))
    result = scan_calphad_ternary_diagram(
        replace(base, requested_compositions=((0.3, 0.3, 0.4), (0.3 + 1e-13, 0.3 - 1e-13, 0.4)))
    )
    assert len(result.points) == 12
    assert result.points[10]["composition"] != result.points[11]["composition"]


def test_json_validates_all_nodes_and_derived_geometry(diagrams):
    original = diagrams["preferred"].to_dict()
    data = json.loads(json.dumps(original, allow_nan=False))
    clone = CalphadTernaryResult.from_dict(data)
    assert clone.to_dict() == original
    assert CalphadTernarySpec.from_dict(data["specification"]).to_dict() == data["specification"]
    for key, value in (
        ("physical_time_s", 1),
        ("complete", False),
        ("scope", "global"),
        ("grid_edges", []),
        ("points", data["points"][:-1]),
    ):
        with pytest.raises(ValueError):
            CalphadTernaryResult.from_dict({**data, key: value})
    changed = copy.deepcopy(data)
    changed["points"][0]["composition"] = [1, 0, 0]
    with pytest.raises(ValueError):
        CalphadTernaryResult.from_dict(changed)
    with pytest.raises(ValueError):
        CalphadTernarySpec.from_dict({**data["specification"], "untracked": 1})


def test_failed_grid_and_path_nodes_are_never_bridged(monkeypatch):
    import chemworld.physchem.calphad_ternary_diagram as api

    original = api.calphad_tp_equilibrium

    def fail(spec, t, p, amounts, **kwargs):
        if 0 < amounts["FE"] < 1 and 0 < amounts["NI"] < 1 and amounts["CR"] > 0:
            return CalphadEquilibriumResult(False, failure_reason="injected_interior_failure")
        return original(spec, t, p, amounts, **kwargs)

    monkeypatch.setattr(api, "calphad_tp_equilibrium", fail)
    progress = []
    result = scan_calphad_ternary_diagram(ternary_example_spec("ideal"), progress=progress.append)
    assert not result.complete and len(progress) == len(result.points)
    assert progress[-1]["completed"] == progress[-1]["total"] == 11
    failed = {p["index"] for p in result.points if not p["success"]}
    assert len(failed) == 2
    assert all(
        not failed.intersection(e["point_indices"]) for e in result.geometry["boundary_brackets"]
    )
    assert not result.geometry["paths"][0]["complete"]
    assert all(not e["success"] for e in result.geometry["paths"][0]["edges"])
    for edge in result.geometry["grid_edges"]:
        assert edge["success"] == (not failed.intersection(edge["point_indices"]))
    assert CalphadTernaryResult.from_dict(result.to_dict()).to_dict() == result.to_dict()


def stock():
    return PhaseLedger(
        {
            "feed": PhaseRecord(
                "feed", "reactor", "solid", 0.00002, {"FE": 0.0006, "NI": 0.0006, "CR": 0.0008}
            ),
            "other": PhaseRecord("other", "elsewhere", "solid", 0.01, {"CU": 0.1}),
        }
    )


def select(ledger, diagram, index=10, source=("feed",)):
    return select_calphad_ternary_inventory_point(
        ledger,
        diagram,
        index,
        vessel_id="reactor",
        source_phase_ids=source,
        phase_targets=tuple(
            CalphadPhaseTarget(m.phase_name.lower(), m.phase_name, 0.01)
            for m in diagram.specification.equilibrium.models
        ),
    )


def test_actual_inventory_is_recomputed_and_original_ownership_preserved(diagrams):
    data = json.loads(json.dumps(diagrams["preferred"].to_dict()))
    clone = CalphadTernaryResult.from_dict(data)
    clone.points[10]["equilibrium"]["phases"][0]["amount_mol_atoms"] = 1e12
    clone.points[10]["equilibrium"]["phases"][0]["fraction"] = 1e12
    ledger = stock()
    before = ledger.to_dict()
    selected = select(ledger, clone)
    assert selected.success
    assert selected.ledger.total_amounts_mol() == pytest.approx(
        ledger.total_amounts_mol(), abs=1e-12
    )
    assert selected.ledger.phases["other"] == ledger.phases["other"]
    assert ledger.to_dict() == before
    assert (
        selected.to_dict()["energy_scope"] == "TP_material_boundary; no_heat_duty_or_time_advance"
    )
    with pytest.raises(ValueError, match="composition"):
        select(ledger, clone, 0)
    with pytest.raises(ValueError, match="distinct"):
        select(ledger, clone, source=("feed", "feed"))
    with pytest.raises(ValueError, match="owner"):
        select(ledger, clone, source=("other",))


def test_exhausted_budget_retains_every_failure_and_rejects_materialization():
    result = scan_calphad_ternary_diagram(replace(ternary_example_spec(), maximum_evaluations=1))
    assert len(result.points) == 11 and sum(p["success"] for p in result.points) == 1
    assert all(
        p["equilibrium"]["failure_reason"] == "ternary_TP_budget_exhausted"
        for p in result.points[1:]
    )
    with pytest.raises(ValueError, match="unresolved"):
        select(stock(), result)


def test_original_thermal_world_query_real_heat_and_snapshot():
    composition = build_calphad_ternary_example()
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=42)
    replay = None
    try:
        before = session._state.to_dict()
        report = analyze_calphad_ternary_world(composition, session._state, grid_intervals=1)
        assert session._state.to_dict() == before
        query = report["world_query"]
        point = report["diagram"]["points"][query["actual_point_index"]]
        assert point["composition"] == query["actual_composition"]
        assert point["success"] and not query["world_advanced"]
        assert query["actual_element_amounts_mol"] == pytest.approx(
            {"FE": 0.0003, "NI": 0.0003, "CR": 0.0004}
        )
        action = session.execute(
            {"operation": "heat", "target_temperature_K": 800.0, "duration_s": 1.0}
        )
        assert action.info["transaction_status"] == "committed", action.info
        assert session._state.ledger.time_s == 1
        assert session._state.temperature_K > before["temperature_K"]
        assert session._state.ledger.energy_jacket_J == pytest.approx(0.05, abs=1e-8)
        replay = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        for runtime in (session, replay):
            action = runtime.execute({"operation": "wait", "duration_s": 1.0})
            assert action.info["transaction_status"] == "committed", action.info
        assert session._state.to_dict() == replay._state.to_dict()
        after = analyze_calphad_ternary_world(composition, session._state, grid_intervals=1)
        assert after["world_query"]["state_time_s"] == 2
        assert after["diagram"]["specification"]["temperature_K"] == session._state.temperature_K
    finally:
        session.close()
        if replay is not None:
            replay.close()
