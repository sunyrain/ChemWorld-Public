"""Authored NASA7/counting fixtures only; no materials, source backend or root."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from math import inf, isnan, nextafter

import numpy as np
import pytest

from chemworld.physchem.gas_query_thermo import GasQueryThermoMemo
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.thermochemistry import (
    NASA7SpeciesThermo,
    NASA7TemperatureSegment,
)


def authored(species_id="A", *, cp=4.0, gap=False, mutable_coefficients=False):
    lower = [cp, 1e-5, -1e-9, 1e-13, 0.0, 17.0, 2.0]
    upper = [cp + 1.0, -1e-5, 1e-9, -1e-13, 0.0, -23.0, 7.0]
    return NASA7SpeciesThermo(
        species_id,
        (
            NASA7TemperatureSegment(
                300.0,
                900.0 if gap else 1000.0,
                lower if mutable_coefficients else tuple(lower),
                "lower",
            ),
            NASA7TemperatureSegment(
                1100.0 if gap else 1000.0,
                2000.0,
                upper if mutable_coefficients else tuple(upper),
                "upper",
            ),
        ),
        reference_pressure_Pa=101325.0,
        composition={"H": 2.0},
        source_note="authored synthetic polynomial, no source material",
    )


def count_evaluations(monkeypatch):
    original, calls = NASA7SpeciesThermo.evaluate, []

    def counted(self, temperature):
        calls.append((self.species_id, temperature, type(temperature)))
        return original(self, temperature)

    monkeypatch.setattr(NASA7SpeciesThermo, "evaluate", counted)
    return calls


def test_repeated_exact_float_returns_original_frozen_state_and_numeric_stats(monkeypatch):
    model = authored()
    expected = model.evaluate(800.0).to_dict()
    calls = count_evaluations(monkeypatch)
    memo = GasQueryThermoMemo({"A": model}, maximum_states=4)
    first = memo.views["A"].evaluate(800.0)
    second = memo.views["A"].evaluate(800.0)
    assert first is second and first.to_dict() == expected
    assert len(calls) == 1
    with pytest.raises(FrozenInstanceError):
        first.enthalpy_J_mol = 0
    assert memo.statistics() == {
        "requests": 2,
        "hits": 1,
        "misses": 1,
        "delegated_calls": 0,
        "uncacheable_results": 0,
        "evictions": 0,
        "current_entries": 1,
        "peak_entries": 1,
        "maximum_entries": 4,
    }
    stats = memo.statistics()
    assert all(type(value) is int for value in stats.values())
    stats["hits"] = 999
    assert memo.statistics()["hits"] == 1


def test_original_segment_validator_runs_again_on_every_hit(monkeypatch):
    original = NASA7SpeciesThermo.select_segment
    calls = []
    rejected = RuntimeError("authored validator refusal")
    reject = [False]

    def validator(self, temperature):
        calls.append(temperature)
        if reject[0]:
            raise rejected
        return original(self, temperature)

    monkeypatch.setattr(NASA7SpeciesThermo, "select_segment", validator)
    memo = GasQueryThermoMemo({"A": authored()}, maximum_states=4)
    memo.views["A"].evaluate(800.0)
    before = len(calls)
    memo.views["A"].evaluate(800.0)
    assert len(calls) == before + 1
    reject[0] = True
    with pytest.raises(RuntimeError) as caught:
        memo.views["A"].evaluate(800.0)
    assert caught.value is rejected
    assert memo.statistics()["hits"] == 1


def test_nextafter_split_and_endpoints_keep_exact_original_branches(monkeypatch):
    model = authored()
    temperatures = (300.0, nextafter(1000.0, -inf), 1000.0, nextafter(1000.0, inf), 2000.0)
    expected = [model.evaluate(t).to_dict() for t in temperatures]
    calls = count_evaluations(monkeypatch)
    memo = GasQueryThermoMemo({"A": model}, maximum_states=10)
    observed = [memo.views["A"].evaluate(t).to_dict() for t in temperatures]
    assert observed == expected and len(calls) == 5
    assert [row["segment_label"] for row in observed] == [
        "lower",
        "lower",
        "lower",
        "upper",
        "upper",
    ]
    assert memo.views["A"].evaluate(1000.0).to_dict() == expected[2]
    assert memo.statistics()["current_entries"] == 5 and len(calls) == 5


@pytest.mark.parametrize("temperature", (0.0, -1.0, float("nan"), inf, -inf, 299.999, 2000.001))
def test_invalid_float_domain_matches_original_without_cached_failure(temperature):
    model = authored()
    with pytest.raises(Exception) as expected:
        model.evaluate(temperature)
    memo = GasQueryThermoMemo({"A": model}, maximum_states=2)
    for _ in range(2):
        with pytest.raises(type(expected.value)) as caught:
            memo.views["A"].evaluate(temperature)
        assert str(caught.value) == str(expected.value)
    assert memo.statistics()["current_entries"] == memo.statistics()["hits"] == 0


def test_outer_bounds_cannot_hide_a_declared_segment_gap():
    model = authored(gap=True)
    memo = GasQueryThermoMemo({"A": model}, maximum_states=3)
    for temperature in (950.0, 1000.0, 1050.0):
        with pytest.raises(ValueError, match="no NASA7 segment"):
            memo.views["A"].evaluate(temperature)
    assert memo.views["A"].evaluate(900.0).segment_label == "lower"
    assert memo.views["A"].evaluate(1100.0).segment_label == "upper"
    assert memo.statistics()["current_entries"] == 2


@pytest.mark.parametrize(
    "temperature", (800, True, False, "800", Decimal("800"), np.float64(800.0))
)
def test_non_float_is_delegated_without_coercion_or_memoization(temperature, monkeypatch):
    model = authored()
    try:
        expected = model.evaluate(temperature)
    except Exception as exc:
        expected = exc
    calls = count_evaluations(monkeypatch)
    memo = GasQueryThermoMemo({"A": model}, maximum_states=3)
    for _ in range(2):
        if isinstance(expected, Exception):
            with pytest.raises(type(expected)) as caught:
                memo.views["A"].evaluate(temperature)
            assert str(caught.value) == str(expected)
        else:
            assert memo.views["A"].evaluate(temperature).to_dict() == expected.to_dict()
    assert len(calls) == 2 and all(t is temperature for _, t, _ in calls)
    assert memo.statistics()["delegated_calls"] == 2
    assert memo.statistics()["current_entries"] == memo.statistics()["hits"] == 0


def test_original_evaluation_exception_is_neither_replaced_nor_cached(monkeypatch):
    sentinel = RuntimeError("authored evaluate failure")
    calls = []

    def failed(_self, temperature):
        calls.append(temperature)
        raise sentinel

    monkeypatch.setattr(NASA7SpeciesThermo, "evaluate", failed)
    memo = GasQueryThermoMemo({"A": authored()}, maximum_states=2)
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            memo.views["A"].evaluate(800.0)
        assert caught.value is sentinel
    assert calls == [800.0, 800.0]
    assert memo.statistics()["current_entries"] == 0


@pytest.mark.parametrize(
    "defect",
    (
        "nan",
        "inf",
        "huge-int",
        "dict",
        "mutable-field",
        "boolean-field",
        "wrong-species",
        "wrong-T",
    ),
)
def test_only_immutable_finite_original_state_is_cached(defect, monkeypatch):
    original = NASA7SpeciesThermo.evaluate
    returns = []

    def invalid(self, temperature):
        state = original(self, temperature)
        value = {
            "nan": lambda: replace(state, cp_J_mol_K=float("nan")),
            "inf": lambda: replace(state, entropy_J_mol_K=inf),
            "huge-int": lambda: replace(state, gibbs_J_mol=10**1000),
            "dict": lambda: {"not": "immutable state"},
            "mutable-field": lambda: replace(state, segment_label=[]),
            "boolean-field": lambda: replace(state, cp_J_mol_K=True),
            "wrong-species": lambda: replace(state, species_id="OTHER"),
            "wrong-T": lambda: replace(state, temperature_K=801.0),
        }[defect]()
        returns.append(value)
        return value

    monkeypatch.setattr(NASA7SpeciesThermo, "evaluate", invalid)
    memo = GasQueryThermoMemo({"A": authored()}, maximum_states=3)
    assert memo.views["A"].evaluate(800.0) is returns[0]
    assert memo.views["A"].evaluate(800.0) is returns[1]
    assert memo.statistics()["current_entries"] == 0
    assert memo.statistics()["uncacheable_results"] == 2
    if defect == "nan":
        assert isnan(returns[0].cp_J_mol_K)


def test_total_lru_capacity_and_recency_across_species(monkeypatch):
    models = {"A": authored("A"), "B": authored("B", cp=5.0)}
    expected = {
        s: {t: model.evaluate(t).to_dict() for t in (700.0, 800.0, 900.0)}
        for s, model in models.items()
    }
    calls = count_evaluations(monkeypatch)
    memo = GasQueryThermoMemo(models, maximum_states=3)
    for species, temperature in (
        ("A", 700.0),
        ("A", 800.0),
        ("B", 700.0),
        ("A", 700.0),
        ("B", 900.0),
        ("A", 800.0),
    ):
        assert memo.views[species].evaluate(temperature).to_dict() == expected[species][temperature]
        assert memo.statistics()["current_entries"] <= 3
    assert len(calls) == 5 and calls[-1][:2] == ("A", 800.0)
    stats = memo.statistics()
    assert stats["hits"] == 1 and stats["evictions"] == 2
    assert stats["peak_entries"] == stats["current_entries"] == 3


@pytest.mark.parametrize("capacity", (None, True, False, 0, -1, 1.5, "3", np.int64(3)))
def test_capacity_is_an_explicit_positive_python_integer(capacity):
    with pytest.raises(ValueError, match="positive Python int"):
        GasQueryThermoMemo({"A": authored()}, maximum_states=capacity)


@pytest.mark.parametrize("kind", ("empty", "none", "list", "wrong-id", "blank-id", "unsupported"))
def test_only_complete_id_bound_original_nasa7_mapping_is_accepted(kind):
    model = authored()
    models = {
        "empty": {},
        "none": None,
        "list": [model],
        "wrong-id": {"B": model},
        "blank-id": {"": model},
        "unsupported": {"A": object()},
    }[kind]
    with pytest.raises(ValueError):
        GasQueryThermoMemo(models, maximum_states=2)


def test_read_only_metadata_and_defensive_declarations_have_no_cache_schema():
    model = authored(mutable_coefficients=True)
    before = deepcopy(model.to_dict())
    memo = GasQueryThermoMemo({"A": model}, maximum_states=2)
    view = memo.views["A"]
    assert (
        view.species_id,
        view.reference_pressure_Pa,
        view.min_temperature_K,
        view.max_temperature_K,
    ) == ("A", 101325.0, 300.0, 2000.0)
    assert view.source_note == model.source_note and view.boundary_policy == model.boundary_policy
    assert dict(view.composition) == {"H": 2.0}
    with pytest.raises(TypeError):
        memo.views["A"] = model
    with pytest.raises(AttributeError, match="read-only"):
        view.reference_pressure_Pa = 1
    with pytest.raises(TypeError):
        view.composition["H"] = 9
    with pytest.raises(AttributeError):
        memo.views = {}
    declaration = view.to_dict()
    declaration["segments"][0]["coefficients"][0] = 999
    declaration["composition"]["H"] = 9
    assert view.to_dict() == before
    # Even mutable containers accepted by the original model cannot alias the query.
    model.composition["H"] = 7
    model.segments[0].coefficients[0] = 99
    assert view.to_dict() == before
    json.dumps(view.to_dict(), allow_nan=False)


def test_two_queries_are_isolated_and_composition_is_not_a_thermo_key(monkeypatch):
    models = {"A": authored("A", cp=4.0), "B": authored("B", cp=5.0)}
    calls = count_evaluations(monkeypatch)
    first = GasQueryThermoMemo(models, maximum_states=4)
    second = GasQueryThermoMemo(models, maximum_states=4)
    mixtures = []
    for x in ({"A": 0.2, "B": 0.8}, {"A": 0.8, "B": 0.2}):
        mixtures.append(sum(x[s] * first.views[s].evaluate(800.0).cp_J_mol_K for s in x))
    assert mixtures[0] != mixtures[1] and len(calls) == 2
    state = second.views["A"].evaluate(800.0)
    assert len(calls) == 3 and state == first.views["A"].evaluate(800.0)
    assert second.statistics()["hits"] == 0 and first.statistics()["hits"] == 3
    different = GasQueryThermoMemo({"A": authored("A", cp=7.0)}, maximum_states=2)
    assert different.views["A"].evaluate(800.0) != state


def test_no_network_cached_property_or_serialized_payload_is_modified():
    species = (SpeciesSpec("A", "H2", phase="gas"),)
    network = ReactionNetworkSpec(
        "authored-query-memo",
        species,
        (),
        metadata={
            "gas_thermochemistry": {
                "A": {
                    "thermo": {
                        "model": "NASA7",
                        "temperature-ranges": [300.0, 1000.0, 2000.0],
                        "data": [[4.0, 0, 0, 0, 0, 17.0, 2.0], [5.0, 0, 0, 0, 0, -23.0, 7.0]],
                    },
                    "composition": {"H": 2.0},
                    "reference_pressure_Pa": 101325.0,
                }
            },
        },
    )
    original = network.imported_gas_thermochemistry
    original_model = original["A"]
    payload = json.dumps(network.to_dict(), sort_keys=True, allow_nan=False)
    memo = GasQueryThermoMemo(original, maximum_states=2)
    memo.views["A"].evaluate(800.0)
    assert network.imported_gas_thermochemistry is original
    assert network.imported_gas_thermochemistry["A"] is original_model
    assert type(original_model) is NASA7SpeciesThermo
    assert json.dumps(network.to_dict(), sort_keys=True, allow_nan=False) == payload
    restored = ReactionNetworkSpec.from_dict(json.loads(payload))
    assert restored.to_dict() == network.to_dict()


def test_subclass_callback_adapter_is_not_implicitly_enabled():
    class StatefulSubclass(NASA7SpeciesThermo):
        pass

    original = authored()
    alternate = StatefulSubclass("A", original.segments)
    with pytest.raises(ValueError, match="exactly ID-bound NASA7"):
        GasQueryThermoMemo({"A": alternate}, maximum_states=2)


def test_metadata_note_container_cannot_alias_the_private_model():
    original = replace(authored(), source_note={"note": ["authored"]})
    memo = GasQueryThermoMemo({"A": original}, maximum_states=2)
    view = memo.views["A"]
    returned_note = view.source_note
    returned_note["note"].append("modified copy")
    assert view.source_note == {"note": ["authored"]}
    original.source_note["note"].append("modified caller")
    assert view.source_note == {"note": ["authored"]}


def test_already_continuous_energy_model_is_copied_without_recorrecting(monkeypatch):
    raw = authored()
    corrected = replace(raw, boundary_policy="continuous-energy")
    assert corrected.segments[0].coefficients == raw.segments[0].coefficients
    assert corrected.segments[1].coefficients[:5] == raw.segments[1].coefficients[:5]
    assert corrected.segments[1].coefficients[5:] != raw.segments[1].coefficients[5:]
    declaration = deepcopy(corrected.to_dict())
    temperatures = (nextafter(1000.0, -inf), 1000.0, nextafter(1000.0, inf))
    expected = {
        temperature: corrected.evaluate(temperature).to_dict() for temperature in temperatures
    }

    def no_second_correction(*_args):
        pytest.fail("query snapshot repeated the model's continuous-energy correction")

    monkeypatch.setattr(
        NASA7SpeciesThermo, "_continuous_energy_segments", staticmethod(no_second_correction)
    )
    memo = GasQueryThermoMemo({"A": corrected}, maximum_states=3)
    view = memo.views["A"]
    assert view.to_dict() == declaration
    assert set(view.to_dict()) == set(declaration)
    assert view.to_dict()["model"] == "NASA7"
    assert view.to_dict()["boundary_policy"] == "continuous-energy"
    for temperature in temperatures:
        first = view.evaluate(temperature)
        hit = view.evaluate(temperature)
        assert first is hit
        assert first.to_dict() == hit.to_dict() == expected[temperature]
    assert memo.statistics()["misses"] == memo.statistics()["hits"] == 3
    assert view.to_dict() == declaration == corrected.to_dict()
