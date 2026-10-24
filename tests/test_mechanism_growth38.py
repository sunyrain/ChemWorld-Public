"""Actual graph growth, passive edge rates, all conditions and restart semantics."""

import json
from dataclasses import replace
from math import exp

import pytest

from chemworld.examples.native_mechanism_growth import example_growth_inputs, run_example
from chemworld.physchem.mechanism_growth import grow_mechanism
from chemworld.physchem.mechanism_growth_specs import (
    CoreEdgeGrowthPolicy,
    GrowthCondition,
    GrowthResourceLimits,
    MechanismGrowthCheckpoint,
    MechanismGrowthLibrary,
)
from chemworld.physchem.reaction_generation import GenerationPolicy


def run(*, library=None, families=None, conditions=None, **kwargs):
    supplied, recipes, tests = example_growth_inputs()
    return grow_mechanism(
        library or supplied, families or recipes, ("ethyl",), conditions or tests, **kwargs
    )


def test_real_multiround_growth_and_all_conditions():
    result = run()
    assert result.complete, result.reasons
    assert len(result.network.species) == 6
    assert len(result.network.reactions) == 4
    history = result.checkpoint.state["history"]
    assert len(history) == 3
    assert history[0]["core_ids"] == ["ethyl"]
    assert set(history[0]["decisions"]["promoted_species"]) == {"H", "ethene", "butane"}
    assert set(history[1]["decisions"]["promoted_species"]) == {"H2", "ethane"}
    assert history[2]["decisions"]["promoted_species"] == []
    for step in history:
        assert len(step["conditions"]) == 4
        assert all(len(case["samples"]) == 9 for case in step["conditions"])
    assert result.checkpoint.state["condition_solves"] == 12
    assert result.checkpoint.state["nfev"] > 0


def test_edge_observation_does_not_drain_core_or_create_initial_stock():
    library, families, conditions = example_growth_inputs()
    before = json.dumps([c.to_dict() for c in conditions], sort_keys=True)
    result = grow_mechanism(library, families, ("ethyl",), conditions)
    assert before == json.dumps([c.to_dict() for c in conditions], sort_keys=True)
    for step in result.checkpoint.state["history"]:
        for case in step["conditions"]:
            initial = case["samples"][0]["core_amounts_mol"]
            assert all(v == 0 for s, v in initial.items() if s != "ethyl")
            if step["round"] == 1:
                assert all(row["core_amounts_mol"] == initial for row in case["samples"])
                assert all(row["characteristic_rate_mol_L_s"] == 0 for row in case["samples"])


def test_first_round_analytic_rates_zero_reference_and_stoichiometry():
    result = run(resources=GrowthResourceLimits(maximum_rounds=1))
    first = result.checkpoint.state["history"][0]
    _, _, conditions = example_growth_inputs()
    lookup = {c.condition_id: c for c in conditions}
    for case in first["conditions"]:
        c = lookup[case["condition_id"]]
        amount = c.initial_amounts_mol["ethyl"]
        q = 0.4 * exp(-3000 / (8.31446261815324 * c.temperature_K)) * amount
        for row in case["samples"]:
            assert row["edge_production_mol_L_s"]["H"] == pytest.approx(q, rel=1e-12)
            assert row["edge_production_mol_L_s"]["ethene"] == pytest.approx(q, rel=1e-12)
            assert row["edge_production_mol_L_s"]["butane"] == pytest.approx(0.04 * amount**2)
    for witness in first["decisions"]["witnesses"].values():
        assert witness["normalized_flux"] is None
        assert witness["threshold_mol_L_s"] == 1e-6


@pytest.mark.parametrize(
    "limits",
    [
        GrowthResourceLimits(maximum_condition_solves=1),
        GrowthResourceLimits(maximum_rounds=1),
        GrowthResourceLimits(maximum_core_species=1),
        GrowthResourceLimits(maximum_edge_candidates=1),
    ],
)
def test_resource_stops_and_exact_json_resume(limits):
    uninterrupted = run()
    stopped = run(resources=limits)
    assert stopped.status == "resource_limit"
    restored = MechanismGrowthCheckpoint.from_dict(
        json.loads(json.dumps(stopped.checkpoint.to_dict(), allow_nan=False))
    )
    resumed = run(checkpoint=restored)
    assert resumed.complete
    assert resumed.network.to_dict() == uninterrupted.network.to_dict()
    assert resumed.checkpoint.state["history"] == uninterrupted.checkpoint.state["history"]
    assert resumed.checkpoint.state["condition_solves"] == 12
    assert len(resumed.checkpoint.state["resource_history"]) == 2
    assert resumed.checkpoint.state["stops"][0]["status"] == "resource_limit"


def test_no_partial_promotion_when_atomic_products_exceed_budget():
    stopped = run(resources=GrowthResourceLimits(maximum_core_species=2))
    assert stopped.status == "resource_limit"
    assert stopped.network.species_ids == ("ethyl",)
    assert stopped.checkpoint.state["core_ids"] == ["ethyl"]
    assert set(stopped.checkpoint.state["pending"]["proposed_decisions"]["promoted_species"]) == {
        "H",
        "ethene",
        "butane",
    }


def test_missing_rates_are_unknown_not_zero_or_silent_completion():
    library, _, _ = example_growth_inputs()
    missing = replace(library, reactions=library.reactions[1:])
    result = run(library=missing)
    assert result.status == "missing_data"
    assert any(r["kind"] == "kinetics_binding" for r in result.reasons)
    assert result.checkpoint.state["condition_solves"] == 0
    assert result.checkpoint.state["last_generation_attempt"]["reactions"]
    assert not result.network.metadata["growth_complete"]


def test_generated_species_outside_library_stays_visible():
    library, _, _ = example_growth_inputs()
    limited = replace(
        library,
        species=tuple(b for b in library.species if b.species.species_id != "butane"),
        reactions=tuple(r for r in library.reactions if "butane" not in r.products),
    )
    result = run(library=limited)
    assert result.status == "missing_data"
    assert any(r["kind"] == "species_binding" and r["graph"] for r in result.reasons)
    assert result.checkpoint.state["last_generation_attempt"]["complete"]


def test_truncated_enumeration_cannot_be_exported_complete():
    result = run(policy=CoreEdgeGrowthPolicy(generation=GenerationPolicy(maximum_matches=1)))
    assert result.status == "generation_truncated"
    assert result.reasons
    assert not result.checkpoint.state["last_generation_attempt"]["complete"]
    assert result.checkpoint.state["condition_solves"] == 0


def test_large_threshold_prunes_only_edge_and_preserves_discovery_records():
    result = run(
        policy=CoreEdgeGrowthPolicy(promotion_relative_threshold=10, keep_relative_threshold=5)
    )
    assert result.complete
    assert set(result.network.species_ids) == {"ethyl", "H", "ethene", "butane"}
    last = result.checkpoint.state["history"][-1]
    assert len(last["decisions"]["pruned_edge_reactions"]) == 2
    assert last["edge_network"]["reactions"]
    assert last["decisions"]["witnesses"]
    assert all(
        set(h["core_ids"]) <= set(result.network.species_ids)
        for h in result.checkpoint.state["history"]
    )


def test_checkpoint_rejects_changed_science_and_reduced_budgets():
    result = run(resources=GrowthResourceLimits(maximum_condition_solves=1))
    with pytest.raises(ValueError, match="cannot change"):
        run(
            checkpoint=result.checkpoint,
            policy=CoreEdgeGrowthPolicy(promotion_relative_threshold=0.1),
        )
    with pytest.raises(ValueError, match="increase"):
        run(checkpoint=result.checkpoint, resources=GrowthResourceLimits(maximum_core_species=5))
    payload = result.checkpoint.to_dict()
    payload["state"]["completed_rounds"] = -1
    with pytest.raises(ValueError, match="counters"):
        run(checkpoint=MechanismGrowthCheckpoint.from_dict(payload))


def test_library_json_roundtrip_keeps_stable_species_and_reaction_keys():
    library, _, _ = example_growth_inputs()
    restored = MechanismGrowthLibrary.from_dict(json.loads(json.dumps(library.to_dict())))
    assert restored.to_dict() == library.to_dict()
    assert run(library=restored).network.to_dict() == run().network.to_dict()


def test_species_flux_aggregates_distinct_generated_channels():
    library, families, conditions = example_growth_inputs()
    scission = next(f for f in families if f.family_id == "beta_scission")
    second = replace(scission, family_id="second_declared_scission")
    duplicate = replace(library.reactions[0], family_id=second.family_id)
    changed = replace(library, reactions=(*library.reactions, duplicate))
    result = grow_mechanism(
        changed,
        (*families, second),
        ("ethyl",),
        conditions,
        resources=GrowthResourceLimits(maximum_rounds=1),
    )
    for case in result.checkpoint.state["history"][0]["conditions"]:
        for row in case["samples"]:
            parts = row["edge_reaction_contributions_mol_L_s"]["H"]
            assert len(parts) == 2
            assert row["edge_production_mol_L_s"]["H"] == pytest.approx(sum(parts.values()))


def test_empty_core_reaction_set_can_close_without_fake_reactions():
    library, families, _ = example_growth_inputs()
    condition = GrowthCondition("inert", {"ethane": 0.1}, 500, 1, (0, 1))
    result = grow_mechanism(library, families, ("ethane",), (condition,))
    assert result.complete
    assert result.network.reactions == ()
    assert result.network.species_ids == ("ethane",)


def test_progress_callback_cannot_change_fixed_execution_inputs_or_resume():
    library, families, conditions = example_growth_inputs()
    before = json.dumps(library.to_dict(), sort_keys=True)

    def mutate_callers_dictionaries(progress):
        for condition in conditions:
            condition.initial_amounts_mol["ethyl"] = 999.0
        library.reactions[0].kinetics.rate_law.parameters["A"] = 17.0
        library.species[0].thermo_declaration["thermo"]["data"][0][5] = 1e8

    stopped = grow_mechanism(
        library,
        families,
        ("ethyl",),
        conditions,
        resources=GrowthResourceLimits(maximum_condition_solves=1),
        progress=mutate_callers_dictionaries,
    )
    assert json.dumps(library.to_dict(), sort_keys=True) != before
    assert json.dumps(stopped.checkpoint.design["library"], sort_keys=True) == before
    original = run()
    resumed = run(checkpoint=stopped.checkpoint)
    assert resumed.network.to_dict() == original.network.to_dict()
    assert resumed.checkpoint.state["history"] == original.checkpoint.state["history"]


def test_original_world_consumes_grown_network_and_resumes_snapshot():
    result = run_example()
    assert result["snapshot_replay_equal"]
    assert result["world_time_s"] == 1
    assert result["world_amounts_mol"]["ethyl"] < 0.1
    assert result["world_amounts_mol"]["H2"] > 0


def test_pruned_zero_flux_edges_reactivate_when_enlarged_core_produces_reactant():
    library, families, _ = example_growth_inputs()
    condition = GrowthCondition("zero-H", {"ethyl": 0.1, "H": 0}, 500, 1, (0, 0.5, 1, 1.5, 2))
    result = grow_mechanism(library, families, ("ethyl", "H"), (condition,))
    assert result.complete
    history = result.checkpoint.state["history"]
    archived = set(history[0]["decisions"]["pruned_edge_reactions"])
    assert len(archived) == 2
    assert set(history[1]["decisions"]["reactivated_edge_reactions"]) == archived
    assert history[2]["decisions"]["reactivated_edge_reactions"] == []
    assert result.checkpoint.state["dormant_edge_keys"] == []


def test_solver_failure_retains_condition_and_does_not_skip_to_favorable_case(monkeypatch):
    from chemworld.physchem.reaction_network import ReactionNetworkSpec

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate solver failure")

    monkeypatch.setattr(ReactionNetworkSpec, "integrate_batch", fail)
    result = run()
    assert result.status == "integration_failed"
    assert result.reasons[0]["condition_id"] == "T400.0-n0.1"
    assert "deliberate solver failure" in result.reasons[0]["message"]
    assert result.checkpoint.state["condition_solves"] == 1
    assert result.checkpoint.state["pending"]["conditions"] == []
    assert not result.complete


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CoreEdgeGrowthPolicy(keep_relative_threshold=1),
        lambda: GrowthCondition("bad", {"ethyl": 1}, 500, 1, (0, 0.5, 0.5)),
        lambda: GrowthResourceLimits(maximum_rounds=0),
        lambda: CoreEdgeGrowthPolicy(generation=GenerationPolicy(allylic_resonance=True)),
    ],
)
def test_explicit_invalid_semantics_rejected(factory):
    with pytest.raises(ValueError):
        factory()
