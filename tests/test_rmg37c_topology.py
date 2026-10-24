"""Original-path topology constraints must survive generation and real consumers."""

import json
from dataclasses import replace
from math import exp

import pytest

pytest.importorskip("cantera", reason="optional independent reference package")

from chemworld.examples.native_intramolecular_reactions import example_graphs, example_world
from chemworld.examples.native_parameter_estimation import example_parameter_inputs
from chemworld.physchem.molecular_graph import find_pattern_matches
from chemworld.physchem.reaction_family import (
    ReactionFamily,
    apply_reaction_recipe,
    elementary_reaction_families,
    extended_radical_reaction_families,
)
from chemworld.physchem.reaction_family_topology import ReactionPathConstraint
from chemworld.physchem.reaction_generation import generate_reactions
from chemworld.physchem.thermo_group_additivity import estimate_group_thermochemistry
from chemworld.runtime import ChemWorldRuntime
from reference.rmg37c_cases import cases, graph
from reference.rmg37c_reference import R, maps_valid, paths_valid, product_signatures, signature


@pytest.mark.parametrize("case", cases()[:18], ids=lambda c: c["name"])
def test_complete_independent_structures_and_atom_maps(case):
    family = extended_radical_reaction_families()[case["family"]]
    report = generate_reactions(case["seeds"], (family,))
    assert report.complete
    assert product_signatures(report) == sorted(signature(g) for g in case["expected"])
    assert len(report.reactions) == len(case["expected"])
    assert maps_valid(report)
    assert paths_valid(report, case["family"])
    if "paths" in case:
        assert sum(len(r.paths) for r in report.reactions) == case["paths"]
    if case.get("shorter_route"):
        assert any("shorter original-graph route" in r["reason"] for r in report.rejected)


def test_manual_recipe_application_cannot_bypass_original_ring_constraint():
    family = extended_radical_reaction_families()[2]
    # Two equal shortest routes across an existing eight-membered ring are not
    # eligible acyclic 1,5 migration paths, despite their correct length.
    ring = graph(
        "cyclooctyl", [(1, 1), *[(2, 0)] * 7], [(i, i + 1, 1) for i in range(1, 8)] + [(8, 1, 1)]
    )
    matches = list(find_pattern_matches(ring, family.reactant_patterns[0]))
    assert matches
    for match in matches:
        with pytest.raises(ValueError, match="pre-existing ring bond"):
            apply_reaction_recipe(family, (ring,), (match,))
    report = generate_reactions({"ring": ring}, (family,))
    assert report.complete and not report.reactions
    assert len(report.rejected) == len(matches)


def test_family_codec_keeps_old_three_family_contract_and_new_exact_paths():
    old = elementary_reaction_families()
    assert len(old) == 3
    for family in (*old, *extended_radical_reaction_families()):
        encoded = json.loads(json.dumps(family.to_dict()))
        assert ReactionFamily.from_dict(encoded) == family
    assert all("path_constraints" not in f.to_dict() for f in old)
    assert [
        len(f.path_constraints[0].bond_orders) for f in extended_radical_reaction_families()[1:]
    ] == [3, 4, 4]


def test_template_cannot_hide_different_path_order_or_asymmetric_swap():
    family = extended_radical_reaction_families()[3]
    with pytest.raises(ValueError, match="identical exact bonds"):
        replace(
            family,
            path_constraints=(
                ReactionPathConstraint(
                    ("*1", "*2", "*3", "*4", "*5"), (1, 2, 1, 1), exocyclic_double_bond=("*5", "*6")
                ),
            ),
        )
    with pytest.raises(ValueError):
        replace(family, unordered_reactants=True)
    with pytest.raises(ValueError, match=r"outside|leave"):
        ReactionPathConstraint(("a", "b", "c"), (1, 1), exocyclic_double_bond=("c", "a"))


@pytest.mark.parametrize("name", ["hexenyl", "cyclopentylmethyl"])
def test_new_graph_families_do_not_silently_expand_37b_thermo_domain(name):
    _, _, thermo, _ = example_parameter_inputs()
    result = estimate_group_thermochemistry(example_graphs()[name], name, thermo)
    assert not result.success
    assert "saturated" in result.failure_reason or "acyclic" in result.failure_reason


@pytest.mark.parametrize("temperature", [500.0, 900.0])
@pytest.mark.parametrize("isothermal", [True, False])
def test_generated_world_commits_and_exactly_resumes_with_topology(temperature, isothermal):
    runtime = ChemWorldRuntime(
        composition=example_world(temperature_K=temperature, isothermal=isothermal)
    )
    restored = None
    try:
        runtime.reset(seed=37)
        first = runtime.execute({"operation": "wait", "duration_s": 0.25})
        assert first.info["transaction_status"] == "committed", first.info.get(
            "outcome_diagnostics"
        )
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for world in (runtime, restored):
            result = world.execute({"operation": "wait", "duration_s": 0.25})
            assert result.info["transaction_status"] == "committed", result.info.get(
                "outcome_diagnostics"
            )
            assert world._state.ledger.time_s == 0.5
        assert restored.snapshot().to_dict() == runtime.snapshot().to_dict()
        if isothermal:
            expected = 0.02 * exp(-2 * exp(-2000 / (R * temperature)) * 0.5)
            assert runtime._state.species_amounts["hexenyl"] == pytest.approx(
                expected, rel=3e-6, abs=2e-10
            )
    finally:
        runtime.close()
        if restored is not None:
            restored.close()
