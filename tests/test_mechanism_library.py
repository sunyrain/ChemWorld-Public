from __future__ import annotations

import pytest

from chemworld.physchem import (
    MechanismScenarioCard,
    get_mechanism_card,
    list_mechanism_cards,
    list_mechanism_paths,
    load_library_mechanism,
    validate_mechanism_library,
)
from chemworld.runtime import compile_mechanism, validate_mechanism_file
from chemworld.schemas import (
    MECHANISM_SCHEMA,
    MECHANISM_SCHEMA_VERSION,
    load_schema_file,
    validate_mechanism_schema,
)

EXPECTED_MECHANISMS = {
    "autocatalytic_reaction",
    "catalyst_deactivation",
    "cstr_multiplicity",
    "electrochemical_conversion",
    "parallel_series_reaction",
    "pfr_hotspot",
    "reaction_extraction",
    "reactive_distillation_lite",
    "reversible_reaction",
    "simple_batch_reaction",
}


def test_curated_mechanism_library_is_complete_and_valid() -> None:
    paths = {path.stem for path in list_mechanism_paths()}
    cards = list_mechanism_cards()
    report = validate_mechanism_library()

    assert paths == EXPECTED_MECHANISMS
    assert {card.mechanism_id for card in cards} == EXPECTED_MECHANISMS
    assert report.passed, report.to_dict()
    assert report.cards_checked == len(EXPECTED_MECHANISMS)
    assert report.mechanisms_checked == len(EXPECTED_MECHANISMS)


def test_mechanism_schema_contract_is_loadable_and_matches_runtime_constant() -> None:
    schema = load_schema_file("mechanism")

    assert schema["title"] == MECHANISM_SCHEMA["title"]
    assert schema["properties"]["schema_version"]["const"] == MECHANISM_SCHEMA_VERSION
    assert schema["properties"]["reactions"]["items"]["properties"]["rate_law"][
        "properties"
    ]["equation_id"]["enum"] == MECHANISM_SCHEMA["properties"]["reactions"]["items"][
        "properties"
    ]["rate_law"]["properties"]["equation_id"]["enum"]


@pytest.mark.parametrize("card", list_mechanism_cards())
def test_every_mechanism_exports_a_replay_manifest(card: MechanismScenarioCard) -> None:
    report = validate_mechanism_file(card.resolved_mechanism_path)
    compiled = compile_mechanism(card)
    manifest = compiled.manifest.to_dict()

    assert report.passed
    assert manifest["mechanism_id"] == card.mechanism_id
    assert manifest["mechanism_hash"] == compiled.mechanism_hash
    assert manifest["species_count"] == len(compiled.network.species)
    assert manifest["reaction_count"] == len(compiled.network.reactions)
    assert manifest["validation_report"]["passed"]
    assert manifest["score_spec"] == compiled.score_spec.to_dict()
    assert manifest["initial_amount_policy"] == card.initial_amounts_mol


def test_mechanism_schema_rejects_executable_or_unknown_rate_laws() -> None:
    payload = {
        "schema_version": MECHANISM_SCHEMA_VERSION,
        "network_id": "bad_runtime_mechanism",
        "species": [
            {"species_id": "A", "formula": "H2"},
            {"species_id": "B", "formula": "H2"},
        ],
        "reactions": [
            {
                "reaction_id": "dangerous",
                "equation": "A => B",
                "rate_law": {
                    "rate_law_id": "dangerous_python",
                    "equation_id": "eval",
                    "parameters": {"code": "__import__('os').system('echo no')"},
                },
            }
        ],
    }
    result = validate_mechanism_schema(payload)

    assert not result.valid
    assert any("unsupported" in error for error in result.errors)
    assert any("cannot execute code" in error for error in result.errors)


def test_mechanism_schema_rejects_duplicate_species_and_reactions() -> None:
    payload = {
        "schema_version": MECHANISM_SCHEMA_VERSION,
        "network_id": "duplicate_mechanism",
        "species": [
            {"species_id": "A", "formula": "H2"},
            {"species_id": "A", "formula": "H2"},
        ],
        "reactions": [
            {
                "reaction_id": "r1",
                "equation": "A => A",
                "rate_law": {
                    "rate_law_id": "k1",
                    "equation_id": "mass_action",
                    "parameters": {"k": 1.0},
                },
            },
            {
                "reaction_id": "r1",
                "equation": "A => A",
                "rate_law": {
                    "rate_law_id": "k2",
                    "equation_id": "mass_action",
                    "parameters": {"k": 1.0},
                },
            },
        ],
    }
    result = validate_mechanism_schema(payload)

    assert not result.valid
    assert "duplicate species_id values: ['A']" in result.errors
    assert "duplicate reaction_id values: ['r1']" in result.errors


@pytest.mark.parametrize("card", list_mechanism_cards())
def test_every_mechanism_card_loads_a_balanced_network(card: MechanismScenarioCard) -> None:
    network = load_library_mechanism(card)
    species = set(network.species_ids)

    assert network.network_id == card.mechanism_id
    assert network.check_element_balance()
    assert set(card.initial_amounts_mol).issubset(species)
    assert set(card.target_species).issubset(species)
    assert set(card.impurity_species).issubset(species)
    assert card.expected_qualitative_behavior
    assert card.to_dict()["mechanism_id"] == card.mechanism_id


def test_get_mechanism_card_accepts_card_or_mechanism_id() -> None:
    by_card = get_mechanism_card("simple-batch-reaction-card")
    by_mechanism = get_mechanism_card("simple_batch_reaction")

    assert by_card == by_mechanism
    assert by_card.resolved_mechanism_path.exists()


def test_autocatalytic_mechanism_exposes_seed_sensitivity() -> None:
    network = load_library_mechanism("autocatalytic_reaction")
    unseeded = network.reaction_rates(
        {"A": 1.0, "P": 0.0},
        volume_L=1.0,
        temperature_K=335.0,
    )
    seeded = network.reaction_rates(
        {"A": 1.0, "P": 0.05},
        volume_L=1.0,
        temperature_K=335.0,
    )

    assert unseeded["autocatalytic_growth"] == 0.0
    assert seeded["autocatalytic_growth"] > seeded["slow_initiation"]


def test_catalyst_deactivation_mechanism_loses_active_catalyst_over_time() -> None:
    network = load_library_mechanism("catalyst_deactivation")
    initial = {
        "A": 1.0,
        "P": 0.0,
        "B": 0.0,
        "D": 0.0,
        "Cat_active": 0.015,
        "Cat_dead": 0.0,
    }
    short = network.integrate_batch(
        initial,
        volume_L=1.0,
        temperature_K=370.0,
        duration_s=2000.0,
    )
    long = network.integrate_batch(
        initial,
        volume_L=1.0,
        temperature_K=370.0,
        duration_s=18000.0,
    )

    assert long.final_amounts_mol["Cat_dead"] > short.final_amounts_mol["Cat_dead"]
    assert long.final_amounts_mol["Cat_active"] < short.final_amounts_mol["Cat_active"]


def test_reaction_extraction_mechanism_prefers_product_in_organic_phase() -> None:
    network = load_library_mechanism("reaction_extraction")
    result = network.integrate_batch(
        {
            "A": 1.0,
            "P_aq": 0.0,
            "P_org": 0.0,
            "B_aq": 0.0,
            "B_org": 0.0,
            "D": 0.0,
            "E": 0.0,
        },
        volume_L=1.0,
        temperature_K=350.0,
        duration_s=18000.0,
    )

    assert result.final_amounts_mol["P_org"] > result.final_amounts_mol["B_org"]
    assert result.final_amounts_mol["P_org"] > 0.0


def test_reactive_distillation_mechanism_forms_product_and_vapor_species() -> None:
    network = load_library_mechanism("reactive_distillation_lite")
    result = network.integrate_batch(
        {
            "Acid": 1.0,
            "Alcohol": 1.25,
            "Ester": 0.0,
            "Water": 0.0,
            "Ether": 0.0,
            "Ester_vapor": 0.0,
            "Alcohol_vapor": 0.0,
            "Water_vapor": 0.0,
        },
        volume_L=1.0,
        temperature_K=370.0,
        duration_s=25000.0,
    )

    assert result.final_amounts_mol["Ester"] > 0.0
    assert result.final_amounts_mol["Alcohol_vapor"] > 0.0
