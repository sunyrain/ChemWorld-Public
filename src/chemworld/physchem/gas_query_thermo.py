"""Explicit query-owned exact-temperature memo for immutable NASA7 states.

This helper owns private model copies; it never changes a reaction network,
its cached properties, or an existing consumer. A caller must opt in
and retain the owner for its query. No composition, rate, residual, inventory,
or experimental work is cached. Unsupported model families are rejected here,
without changing their existing uncached public paths.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from math import isfinite
from types import MappingProxyType
from typing import Any

from chemworld.physchem.thermochemistry import NASA7SpeciesThermo, SpeciesThermoState

_MISSING = object()
_NUMERIC_STATE_FIELDS = (
    "temperature_K",
    "cp_over_R",
    "h_over_RT",
    "s_over_R",
    "cp_J_mol_K",
    "enthalpy_J_mol",
    "entropy_J_mol_K",
    "gibbs_J_mol",
)


def _cacheable(state, species_id, temperature):
    # Never introduce a new failure merely while inspecting a raw return.
    # Unsupported or nonfinite returns pass through unchanged and are not cached.
    if type(state) is not SpeciesThermoState:
        return False
    if (
        type(state.species_id) is not str
        or type(state.segment_label) is not str
        or state.species_id != species_id
        or type(state.temperature_K) is not float
        or state.temperature_K != temperature
    ):
        return False
    try:
        return all(
            type(value := getattr(state, name)) in (int, float) and isfinite(value)
            for name in _NUMERIC_STATE_FIELDS
        )
    except (ArithmeticError, TypeError, ValueError):
        return False


class _ExactStateLRU:
    """One total state cap across every species; no references back to views."""

    __slots__ = (
        "capacity",
        "delegated_calls",
        "entries",
        "evictions",
        "hits",
        "misses",
        "peak_entries",
        "requests",
        "uncacheable_results",
    )

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.requests = self.hits = self.misses = self.delegated_calls = 0
        self.uncacheable_results = self.evictions = self.peak_entries = 0

    def evaluate(self, model, temperature):
        self.requests += 1
        if type(temperature) is not float:
            # In particular, do not coerce bool/int/numpy/Decimal/string inputs.
            self.delegated_calls += 1
            return model.evaluate(temperature)
        # Re-run the original domain/segment validator even on a cache hit.
        # Outer min/max alone would incorrectly permit a declared segment gap.
        model.select_segment(temperature)
        key = (model.species_id, temperature)
        state = self.entries.get(key, _MISSING)
        if state is not _MISSING:
            self.hits += 1
            self.entries.move_to_end(key)
            return state
        self.misses += 1
        state = model.evaluate(temperature)  # exceptions propagate and are never cached
        if not _cacheable(state, model.species_id, temperature):
            self.uncacheable_results += 1
            return state
        # Evict before insertion: even the temporary cached-entry count is bounded.
        if len(self.entries) >= self.capacity:
            self.entries.popitem(last=False)
            self.evictions += 1
        self.entries[key] = state
        self.peak_entries = max(self.peak_entries, len(self.entries))
        return state

    def statistics(self):
        return {
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "delegated_calls": self.delegated_calls,
            "uncacheable_results": self.uncacheable_results,
            "evictions": self.evictions,
            "current_entries": len(self.entries),
            "peak_entries": self.peak_entries,
            "maximum_entries": self.capacity,
        }


class _QueryThermoView:
    """Read-only metadata and evaluate protocol; not a new serialized model type."""

    __slots__ = ("__memo", "__model")

    def __init__(self, model, memo):
        object.__setattr__(self, "_QueryThermoView__model", model)
        object.__setattr__(self, "_QueryThermoView__memo", memo)

    def __setattr__(self, name, value):
        raise AttributeError("query thermo metadata is read-only")

    @property
    def species_id(self):
        return self.__model.species_id

    @property
    def reference_pressure_Pa(self):  # noqa: N802 -- existing thermo protocol
        return self.__model.reference_pressure_Pa

    @property
    def min_temperature_K(self):  # noqa: N802 -- existing thermo protocol
        return self.__model.min_temperature_K

    @property
    def max_temperature_K(self):  # noqa: N802 -- existing thermo protocol
        return self.__model.max_temperature_K

    @property
    def composition(self):
        return MappingProxyType(self.__model.composition)

    @property
    def source_note(self):
        return deepcopy(self.__model.source_note)

    @property
    def boundary_policy(self):
        return self.__model.boundary_policy

    def evaluate(self, temperature_K):
        return self.__memo.evaluate(self.__model, temperature_K)

    def to_dict(self):
        # The original schema only: no memo, proxy or query state is persisted.
        return deepcopy(self.__model.to_dict())


class GasQueryThermoMemo:
    """Own explicit NASA7 model snapshots and a bounded exact-float-T LRU.

    ``maximum_states`` is one positive Python-int cap for all species states.
    ``views`` supports the existing .evaluate(T)/reference-pressure protocol;
    it is deliberately not a NASA7 subclass or an arbitrary callback adapter.
    Model declarations are copied without reconstructing their coefficients.
    One owner serves sequential evaluations within one query; it is not a
    shared cache for concurrent queries or arbitrary stateful model callbacks.
    None/disabled handling belongs to the caller, not this helper.
    """

    __slots__ = ("_memo", "_views")

    def __init__(self, models: Mapping[str, NASA7SpeciesThermo], *, maximum_states: int):
        if type(maximum_states) is not int or maximum_states < 1:
            raise ValueError("maximum_states must be an explicit positive Python int")
        if not isinstance(models, Mapping) or not models:
            raise ValueError("query thermo memo requires a nonempty NASA7 mapping")
        owned = {}
        for species_id, model in models.items():
            if (
                type(species_id) is not str
                or not species_id.strip()
                or type(model) is not NASA7SpeciesThermo
                or model.species_id != species_id
            ):
                raise ValueError("query thermo memo supports exactly ID-bound NASA7 models")
            owned[species_id] = deepcopy(model)
        self._memo = _ExactStateLRU(maximum_states)
        self._views = MappingProxyType(
            {species_id: _QueryThermoView(model, self._memo) for species_id, model in owned.items()}
        )

    @property
    def views(self) -> Mapping[str, Any]:
        return self._views

    def statistics(self) -> dict[str, int]:
        """Copy numeric cache counters only; these are not physical work or time."""
        return self._memo.statistics()
