"""Explicit source identities, immutable declarations and deterministic selection."""

import copy
import json
from itertools import combinations

import pytest
from tests.test_multifluid import R, source_payload

from chemworld.data.multifluid_catalog import (
    MultifluidSourceCatalog,
    load_multifluid_catalog,
    read_multifluid_catalog,
)
from chemworld.physchem.multifluid import MultifluidModel


def catalog_payload(count=4):
    base = source_payload()
    fluids = []
    for i in range(count):
        fluid = copy.deepcopy(base["fluids"][i % 2])
        fluid["INFO"] = {"NAME": chr(65 + i), "CAS": f"synthetic-{i + 1}"}
        fluid["EOS"][0]["STATES"]["reducing"]["T"] = 200.0 + 30 * i
        fluid["EOS"][0]["STATES"]["reducing"]["rhomolar"] = 9000.0 + 1000 * i
        fluids.append(fluid)
    pairs = []
    for i, j in combinations(range(count), 2):
        pair = copy.deepcopy(base["binary_pairs"][0])
        pair.update(
            Name1=chr(65 + i),
            Name2=chr(65 + j),
            CAS1=f"synthetic-{i + 1}",
            CAS2=f"synthetic-{j + 1}",
        )
        pairs.append(pair)
    return {
        "model": "chemworld-multifluid-catalog-v1",
        "catalog_id": "synthetic-declared",
        "revision": "one",
        "source": "explicit synthetic fixtures; not material data",
        "fluids": fluids,
        "binary_pairs": pairs,
        "departure_functions": base["departure_functions"],
    }


def selected(catalog, cases=None, **kwargs):
    return catalog.select(
        cases or [c["cas"] for c in catalog.components],
        gas_constant_J_mol_K=R,
        gas_constant_convention="source_equal",
        **kwargs,
    )


def test_catalogue_json_selection_and_source_identity_survive_roundtrip(tmp_path):
    payload = catalog_payload()
    catalog = load_multifluid_catalog(payload)
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(payload))
    restored = read_multifluid_catalog(path)
    assert restored == catalog and catalog.catalog_id == "synthetic-declared"
    assert catalog.revision == "one" and len(catalog.declaration_sha256) == 64
    model = selected(restored)
    assert model.component_ids == ("A", "B", "C", "D")
    assert len(model.binary_pairs) == 6
    assert MultifluidModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    assert '"revision": "one"' in model.source
    assert '"CAS1": "synthetic-1"' in model.source
    assert '"Name1": "A"' in model.source
    assert catalog.declaration_sha256 in model.source
    assert model.caloric_state(400.0, 100.0, (0.25,) * 4).pressure_Pa > 0
    payload["fluids"][0]["INFO"]["NAME"] = "mutated"
    exported = catalog.to_dict()
    exported["binary_pairs"].clear()
    assert catalog.components[0]["name"] == "A" and len(selected(catalog).binary_pairs) == 6


def test_source_record_orientation_and_requested_component_order_are_preserved():
    catalog = load_multifluid_catalog(catalog_payload())
    normal = selected(catalog)
    reverse = selected(catalog, [c["cas"] for c in catalog.components][::-1])
    assert reverse.component_ids == normal.component_ids[::-1]
    assert {p.component_ids: p.beta_T for p in reverse.binary_pairs} == {
        p.component_ids: p.beta_T for p in normal.binary_pairs
    }
    assert reverse.state(400.0, 100.0, (0.1, 0.2, 0.3, 0.4)).pressure_Pa == pytest.approx(
        normal.state(400.0, 100.0, (0.4, 0.3, 0.2, 0.1)).pressure_Pa, rel=1e-12
    )


def test_full_source_tables_select_only_relevant_pairs_and_departures():
    catalog = load_multifluid_catalog(catalog_payload())
    assert len(selected(catalog, ["synthetic-1", "synthetic-3"]).binary_pairs) == 1
    pure = selected(catalog, ["synthetic-2"])
    assert not pure.binary_pairs and pure.component_ids == ("B",)


@pytest.mark.parametrize(
    "change",
    ["revision", "source", "unknown", "duplicate_cas", "duplicate_name", "no_cas", "no_eos", "nan"],
)
def test_source_envelope_rejects_untraceable_or_non_json_records(change):
    data = catalog_payload()
    if change in ("revision", "source"):
        data[change] = ""
    elif change == "unknown":
        data["implicit_network_discovery"] = True
    elif change == "duplicate_cas":
        data["fluids"][1]["INFO"]["CAS"] = data["fluids"][0]["INFO"]["CAS"]
    elif change == "duplicate_name":
        data["fluids"][1]["INFO"]["NAME"] = "A"
    elif change == "no_cas":
        del data["fluids"][0]["INFO"]["CAS"]
    elif change == "no_eos":
        data["fluids"][0]["EOS"] = []
    else:
        data["fluids"][0]["EOS"][0]["gas_constant"] = float("nan")
    with pytest.raises(ValueError):
        load_multifluid_catalog(data)


@pytest.mark.parametrize("cases", [("missing",), ("A",), ("synthetic-1", "synthetic-1"), ()])
def test_select_requires_unique_exact_cas(cases):
    with pytest.raises(ValueError):
        load_multifluid_catalog(catalog_payload()).select(
            cases, gas_constant_J_mol_K=R, gas_constant_convention="source_equal"
        )


def test_duplicate_binary_needs_explicit_complete_indices_and_reason():
    data = catalog_payload(2)
    duplicate = copy.deepcopy(data["binary_pairs"][0])
    duplicate["gammaT"] *= 1.02
    duplicate["BibTeX"] = "Alternative source declaration"
    data["binary_pairs"].append(duplicate)
    catalog = load_multifluid_catalog(data)
    with pytest.raises(ValueError, match="ambiguous"):
        selected(catalog)
    with pytest.raises(ValueError, match="reason"):
        selected(catalog, pair_indices=(1,))
    model = selected(catalog, pair_indices=(1,), selection_reason="Explicit alternate citation")
    assert model.binary_pairs[0].gamma_T == duplicate["gammaT"]
    assert '"index": 1' in model.source and "Explicit alternate citation" in model.source
    assert "Alternative source declaration" in model.source


@pytest.mark.parametrize("indices", [(), (0, 0), (-1,), (100,), (True,), (0, 1)])
def test_invalid_incomplete_or_extraneous_explicit_indices_reject(indices):
    with pytest.raises(ValueError):
        selected(
            load_multifluid_catalog(catalog_payload()),
            pair_indices=indices,
            selection_reason="invalid test selection",
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing_pair",
        "bad_name",
        "missing_departure",
        "duplicate_departure",
        "unsupported_departure",
        "bad_eos",
    ],
)
def test_selected_source_errors_remain_errors_in_native_importer(change):
    data = catalog_payload(2)
    if change == "missing_pair":
        data["binary_pairs"].clear()
    elif change == "bad_name":
        data["binary_pairs"][0]["Name1"] = "wrong source identity"
    elif change == "missing_departure":
        data["departure_functions"].clear()
    elif change == "duplicate_departure":
        data["departure_functions"] *= 2
    elif change == "unsupported_departure":
        data["departure_functions"][0]["type"] = "unimplemented"
    else:
        data["fluids"][0]["EOS"] *= 2
    with pytest.raises(ValueError):
        selected(load_multifluid_catalog(data))


def test_eos_selection_uses_cas_and_survives_provenance():
    data = catalog_payload(2)
    data["fluids"][0]["EOS"] *= 2
    catalog = load_multifluid_catalog(data)
    model = selected(catalog, eos_indexes={"synthetic-1": 1})
    assert model.fluids[0].eos_index == 1 and '"eos_index": 1' in model.source
    for indexes in ({"alien": 0}, {"synthetic-1": True}, {"synthetic-1": 2}):
        with pytest.raises(ValueError):
            selected(catalog, eos_indexes=indexes)


def test_bounded_file_ingestion(tmp_path):
    path = tmp_path / "oversized.json"
    with path.open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="limit"):
        read_multifluid_catalog(path)


@pytest.mark.parametrize("key", ["revision", "CAS", "gammaT"])
def test_file_reader_rejects_duplicate_source_json_keys(tmp_path, key):
    payload = catalog_payload()
    text = json.dumps(payload)
    value = {
        "revision": payload["revision"],
        "CAS": payload["fluids"][0]["INFO"]["CAS"],
        "gammaT": payload["binary_pairs"][0]["gammaT"],
    }[key]
    field = json.dumps(key) + ": " + json.dumps(value)
    # Put the intended value last: a permissive reader silently drops the
    # conflicting declaration and still returns a seemingly valid catalogue.
    conflicting = 1.23 if key == "gammaT" else "different-source-identity"
    text = text.replace(field, json.dumps(key) + ": " + json.dumps(conflicting) + ", " + field, 1)
    path = tmp_path / "duplicate-key.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="duplicate JSON key: " + key):
        read_multifluid_catalog(path)


def test_direct_constructor_and_native_codec_require_the_same_envelope():
    data = catalog_payload()
    kwargs = {k: v for k, v in data.items() if k != "model"}
    assert MultifluidSourceCatalog(**kwargs) == MultifluidSourceCatalog.from_dict(data)


@pytest.mark.parametrize("reason", ["", " ", 1, [], {}])
def test_optional_selection_reason_still_requires_nonempty_text(reason):
    with pytest.raises(ValueError, match="reason"):
        selected(load_multifluid_catalog(catalog_payload()), selection_reason=reason)


@pytest.mark.parametrize("field", ["CAS1", "CAS2", "function"])
def test_explicit_binary_selection_rejects_nontext_source_identity(field):
    data = catalog_payload(2)
    data["binary_pairs"][0][field] = ["not", "an", "identity"]
    with pytest.raises(ValueError):
        selected(
            load_multifluid_catalog(data), pair_indices=(0,), selection_reason="invalid source"
        )
