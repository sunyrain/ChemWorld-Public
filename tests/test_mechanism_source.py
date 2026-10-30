"""Declared mechanism selections preserve equations, unit scopes and replay."""

import copy
import hashlib
import json

import pytest
import yaml

from chemworld.physchem.mechanism_import import (
    convert_mechanism,
    import_mechanism,
    mechanism_to_cantera_document,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec


def document():
    return {
        "phases": [{"name": "gas", "thermo": "ideal-gas", "kinetics": "gas",
                    "species": ["A", "B"], "elements": ["H"]}],
        "species": [
            {"name": name, "composition": {"H": 2}, "thermo": {
                "model": "constant-cp", "cp0": "30 J/mol/K", "h0": "-1 kJ/mol",
                "s0": "20 J/mol/K", "T-min": 200, "T-max": 2000,
            }} for name in ("A", "B")
        ],
        "reactions": [{"equation": "A => B", "rate-constant": {"A": 2}}],
    }


def split_document():
    original = document()
    root = {"phases": copy.deepcopy(original["phases"])}
    root["phases"][0].update(
        species=[{"data.yaml/molecules": ["B", "A"]}],
        reactions=[{"data.yaml/rates": "all"}],
    )
    source = {"molecules": original["species"], "rates": original["reactions"]}
    return root, source


def test_named_sections_and_species_all_preserve_selected_order_and_input():
    raw = document()
    raw["molecules"] = raw.pop("species")
    raw["kinetics"] = raw.pop("reactions")
    raw["phases"][0].update(species=[{"molecules": "all"}], reactions=["kinetics"])
    before = copy.deepcopy(raw)
    network = convert_mechanism(raw)
    assert network.species_ids == ("A", "B")
    assert network.reaction_rates({"A": .2, "B": .3}, volume_L=1, temperature_K=700) == {
        "reaction-001": .4,
    }
    assert raw == before
    provenance = network.metadata["source_resolution"]
    assert provenance["documents"][0]["sha256"] == hashlib.sha256(
        json.dumps(before, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    assert provenance["selected_phase"]["reactions"] == ["kinetics"]


def test_in_memory_external_sources_bind_file_list_species_and_thermo_units():
    root, source = split_document()
    root["units"] = {"quantity": "kmol", "energy": "J", "time": "s"}
    source["units"] = {"quantity": "mol", "energy": "cal", "time": "s"}
    for entry in source["molecules"]:
        entry["thermo"].update(cp0=30, h0=-100, s0=20)
    source["molecules"][0]["thermo"]["units"] = {"energy": "kJ"}
    source["rates"].insert(0, {"units": {"time": "min"}})
    source["rates"][1]["rate-constant"].update(A=120, Ea=3)
    source_before = copy.deepcopy(source)
    network = convert_mechanism(root, source_documents={"data.yaml": source})
    assert network.species_ids == ("B", "A")
    assert network.reactions[0].rate_law.parameters["A"] == pytest.approx(2)
    assert network.reactions[0].rate_law.parameters["Ea_J_per_mol"] == pytest.approx(12.552)
    thermo = network.imported_gas_thermochemistry
    assert thermo["A"].evaluate(298.15).cp_J_mol_K == pytest.approx(30000)
    assert thermo["B"].evaluate(298.15).cp_J_mol_K == pytest.approx(125.52)
    restored = ReactionNetworkSpec.from_dict(json.loads(json.dumps(network.to_dict())))
    assert restored.to_dict() == network.to_dict()
    assert source == source_before


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_explicit_local_search_and_relative_priority(tmp_path, suffix):
    root, source = split_document()
    source_name = "data" + suffix
    root["phases"][0].update(species=[{source_name + "/molecules": "all"}],
                             reactions=[source_name + "/rates"])
    path = tmp_path / "main.yaml"
    path.write_text(yaml.safe_dump(root), encoding="utf-8")
    search = tmp_path / "library"
    search.mkdir()
    encoded = json.dumps(source) if suffix == ".json" else yaml.safe_dump(source)
    (search / source_name).write_text(encoded, encoding="utf-8")
    with pytest.raises(ValueError, match="not found"):
        import_mechanism(path)
    network = import_mechanism(path, search_paths=[search])
    assert network.reactions[0].rate_law.parameters["A"] == 2
    source["rates"][0]["rate-constant"]["A"] = 3
    (tmp_path / source_name).write_text(
        json.dumps(source) if suffix == ".json" else yaml.safe_dump(source), encoding="utf-8",
    )
    assert import_mechanism(path, search_paths=[search]).reactions[0].rate_law.parameters["A"] == 3


def test_declared_selection_omissions_are_explicit_and_all_keeps_errors():
    raw = document()
    raw["species"].append({"name": "C", "composition": {"O": 2},
                           "thermo": copy.deepcopy(raw["species"][0]["thermo"])})
    raw["reactions"].append({"equation": "C => C", "rate-constant": {"A": 1}})
    raw["phases"][0].update(species="all", reactions="declared-species",
                             **{"skip-undeclared-elements": True})
    network = convert_mechanism(raw)
    provenance = network.metadata["source_resolution"]
    assert network.species_ids == ("A", "B")
    assert len(network.reactions) == 1
    assert provenance["omitted_species"][0]["name"] == "C"
    assert provenance["omitted_reactions"][0]["undeclared_species"] == ["C"]
    raw["phases"][0]["reactions"] = "all"
    with pytest.raises(ValueError, match="unknown species"):
        convert_mechanism(raw)


def test_undeclared_efficiencies_require_source_permission():
    raw = document()
    raw["phases"][0]["species"] = "all"
    raw["reactions"] = [{"equation": "A + M => B + M", "type": "three-body",
                         "rate-constant": {"A": 2}, "efficiencies": {"C": 3}}]
    with pytest.raises(ValueError, match="unknown species"):
        convert_mechanism(raw)
    raw["phases"][0]["skip-undeclared-third-bodies"] = True
    network = convert_mechanism(raw)
    assert network.metadata["source_resolution"]["reactions"][0]["omitted_efficiencies"] == {
        "C": 3,
    }
    assert network.reaction_rates({"A": .2, "B": .3}, volume_L=1, temperature_K=700) == {
        "reaction-001": .2,
    }


def test_explicit_nonreacting_gas_keeps_thermochemistry_and_native_zero_derivative():
    raw = document()
    raw["phases"][0].update(species="all", reactions="none")
    raw.pop("reactions")
    network = convert_mechanism(raw)
    assert network.reactions == ()
    assert network.imported_gas_thermochemistry["A"].evaluate(500).cp_J_mol_K == 30
    assert network.amount_derivatives({"A": .2, "B": .3}, volume_L=1, temperature_K=500) == {
        "A": 0, "B": 0,
    }


@pytest.mark.parametrize("selection", [[{"species": ["A", "A"]}],
                                       [{"species": ["A", "missing"]}],
                                       [{"species": ["A"]}, {"species": ["A"]}]])
def test_missing_or_repeated_species_never_silently_merge(selection):
    raw = document()
    raw["phases"][0]["species"] = selection
    with pytest.raises(ValueError):
        convert_mechanism(raw)


@pytest.mark.parametrize("suffix,text", [
    (".yaml", "phases: []\nphases: []\n"),
    (".json", '{"phases": [], "phases": []}'),
])
def test_ambiguous_file_keys_reject(tmp_path, suffix, text):
    path = tmp_path / ("input" + suffix)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"[Dd]uplicate"):
        import_mechanism(path)


def test_no_implicit_external_lookup_in_memory_and_no_mutable_provenance():
    root, source = split_document()
    with pytest.raises(ValueError, match="external"):
        convert_mechanism(root)
    network = convert_mechanism(root, source_documents={"data.yaml": source})
    before = network.to_dict()
    source["rates"][0]["rate-constant"]["A"] = 99
    root["phases"][0]["name"] = "changed"
    assert network.to_dict() == before


def test_external_source_reaches_real_thermal_world_and_exact_continuation(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "source34_probe", Path(__file__).parent / "reference/probe_mechanism_source34_install.py",
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert probe.run(tmp_path / "world-inputs")["exact_continuation"]


def test_imported_network_exports_to_a_reimportable_cantera_document():
    network = convert_mechanism(document(), source="synthetic-source.yaml")
    exported = mechanism_to_cantera_document(network)
    assert exported["generator"] == "chemworld.mechanism_import.export"
    assert exported["phases"][0]["species"] == ["A", "B"]
    assert [item["name"] for item in exported["species"]] == ["A", "B"]
    restored = convert_mechanism(exported, source="roundtrip.yaml")
    assert restored.species_ids == network.species_ids
    assert restored.reaction_ids == network.reaction_ids
    assert restored.reaction_rates(
        {"A": 0.2, "B": 0.3}, volume_L=1, temperature_K=700
    ) == network.reaction_rates({"A": 0.2, "B": 0.3}, volume_L=1, temperature_K=700)
    assert json.loads(json.dumps(exported, allow_nan=False)) == exported


def test_export_rejects_native_network_without_complete_source_declarations():
    network = ReactionNetworkSpec.from_dict(
        {
            "network_id": "native",
            "species": [
                {"species_id": "A", "formula": "H2"},
                {"species_id": "B", "formula": "H2"},
            ],
            "reactions": [
                {
                    "reaction_id": "reaction-001",
                    "equation": "A => B",
                    "stoichiometry": {"A": -1, "B": 1},
                    "reversible": False,
                    "rate_law": {
                        "rate_law_id": "r",
                        "equation_id": "arrhenius",
                        "parameters": {"A": 1, "b": 0, "Ea_J_per_mol": 0},
                    },
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="complete imported Cantera gas source"):
        mechanism_to_cantera_document(network)
