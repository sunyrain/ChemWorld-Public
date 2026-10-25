"""Explicit finite parameter libraries and replayable core-edge growth contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from math import isfinite
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.molecular_graph import MolecularGraph, molecular_isomorphic
from chemworld.physchem.reaction_generation import (
    GenerationPolicy,
    KineticsBinding,
    SpeciesDataBinding,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reaction_network_specs import RateLawSpec, species_from_dict


def reaction_library_key(
    family_id: str, reactants: Mapping[str, int], products: Mapping[str, int]
) -> str:
    """A stable structural identity uses bound species, never per-pass numbering."""
    if not family_id or not reactants or not products:
        raise ValueError("reaction library key requires family and both sides")
    for side in (reactants, products):
        if any(not k or type(v) is not int or v < 1 for k, v in side.items()):
            raise ValueError("library stoichiometry requires named species and positive integers")
    return json.dumps(
        [family_id, sorted(reactants.items()), sorted(products.items())],
        ensure_ascii=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class ReactionLibraryEntry:
    family_id: str
    reactants: Mapping[str, int]
    products: Mapping[str, int]
    kinetics: KineticsBinding

    def __post_init__(self) -> None:
        reaction_library_key(self.family_id, self.reactants, self.products)

    @property
    def key(self) -> str:
        return reaction_library_key(self.family_id, self.reactants, self.products)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "reactants": dict(self.reactants),
            "products": dict(self.products),
            "kinetics": {
                **asdict(self.kinetics),
                "rate_law": self.kinetics.rate_law.to_dict(),
                "source": self.kinetics.source.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReactionLibraryEntry:
        values = dict(payload["kinetics"])
        values["rate_law"] = RateLawSpec(**values["rate_law"])
        values["source"] = DataSourceProvenance.from_dict(values["source"])
        return cls(
            payload["family_id"],
            dict(payload["reactants"]),
            dict(payload["products"]),
            KineticsBinding(**values),
        )


@dataclass(frozen=True)
class MechanismGrowthLibrary:
    species: tuple[SpeciesDataBinding, ...]
    reactions: tuple[ReactionLibraryEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "species", tuple(self.species))
        object.__setattr__(self, "reactions", tuple(self.reactions))
        names = [s.species.species_id for s in self.species]
        if not names or len(set(names)) != len(names):
            raise ValueError("finite library requires unique species identities")
        if len({r.key for r in self.reactions}) != len(self.reactions):
            raise ValueError("finite library reaction keys must be unique")
        for i, binding in enumerate(self.species):
            if any(molecular_isomorphic(binding.graph, old.graph) for old in self.species[:i]):
                raise ValueError("isomorphic graphs cannot have competing library identities")
        for reaction in self.reactions:
            if (reaction.reactants.keys() | reaction.products.keys()) - set(names):
                raise ValueError("library kinetic entry references a missing declared species")

    @property
    def species_by_id(self) -> dict[str, SpeciesDataBinding]:
        return {binding.species.species_id: binding for binding in self.species}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "finite-graph-parameter-library-v1",
            "species": [
                {
                    "graph": b.graph.to_dict(),
                    "species": b.species.to_dict(),
                    "thermo_declaration": dict(b.thermo_declaration),
                    "source": b.source.to_dict(),
                }
                for b in self.species
            ],
            "reactions": [r.to_dict() for r in self.reactions],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MechanismGrowthLibrary:
        if payload.get("model") != "finite-graph-parameter-library-v1":
            raise ValueError("unsupported growth library schema")
        return cls(
            tuple(
                SpeciesDataBinding(
                    MolecularGraph.from_dict(row["graph"]),
                    species_from_dict(row["species"]),
                    row["thermo_declaration"],
                    DataSourceProvenance.from_dict(row["source"]),
                )
                for row in payload["species"]
            ),
            tuple(ReactionLibraryEntry.from_dict(row) for row in payload["reactions"]),
        )


@dataclass(frozen=True)
class GrowthCondition:
    condition_id: str
    initial_amounts_mol: Mapping[str, float]
    temperature_K: float
    volume_L: float
    sample_times_s: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_times_s", tuple(self.sample_times_s))
        if not self.condition_id or not self.initial_amounts_mol:
            raise ValueError("growth condition requires identity and explicit initial inventory")
        if any(not isfinite(x) or x <= 0 for x in (self.temperature_K, self.volume_L)):
            raise ValueError("condition temperature and volume must be finite positive")
        if any(not k or not isfinite(v) or v < 0 for k, v in self.initial_amounts_mol.items()):
            raise ValueError("condition initial amounts must be finite nonnegative")
        times = self.sample_times_s
        if (
            len(times) < 2
            or times[0] != 0
            or any(not isfinite(t) for t in times)
            or any(a >= b for a, b in pairwise(times))
        ):
            raise ValueError("growth samples must start at zero and increase strictly")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "initial_amounts_mol": dict(self.initial_amounts_mol),
            "sample_times_s": list(self.sample_times_s),
        }


@dataclass(frozen=True)
class CoreEdgeGrowthPolicy:
    promotion_relative_threshold: float = 0.001
    promotion_absolute_threshold_mol_L_s: float = 1e-6
    keep_relative_threshold: float = 0.0005
    keep_absolute_threshold_mol_L_s: float = 5e-7
    reference_temperature_K: float = 500.0
    generation: GenerationPolicy = field(default_factory=GenerationPolicy)

    def __post_init__(self) -> None:
        values = (
            self.promotion_relative_threshold,
            self.promotion_absolute_threshold_mol_L_s,
            self.keep_relative_threshold,
            self.keep_absolute_threshold_mol_L_s,
            self.reference_temperature_K,
        )
        if any(not isfinite(v) or v <= 0 for v in values):
            raise ValueError("growth thresholds and reference temperature must be finite positive")
        if (
            self.keep_relative_threshold > self.promotion_relative_threshold
            or self.keep_absolute_threshold_mol_L_s > self.promotion_absolute_threshold_mol_L_s
        ):
            raise ValueError("edge keep thresholds must not exceed promotion thresholds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GrowthResourceLimits:
    maximum_rounds: int = 8
    maximum_condition_solves: int = 40
    maximum_core_species: int = 12
    maximum_edge_candidates: int = 32

    def __post_init__(self) -> None:
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("growth resource limits must be positive integers")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class MechanismGrowthCheckpoint:
    design: dict[str, Any]
    state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                {
                    "model": "native-core-edge-checkpoint-v1",
                    "design": self.design,
                    "state": self.state,
                },
                allow_nan=False,
            )
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MechanismGrowthCheckpoint:
        if set(payload) != {"model", "design", "state"}:
            raise ValueError("invalid growth checkpoint fields")
        if payload["model"] != "native-core-edge-checkpoint-v1":
            raise ValueError("unsupported growth checkpoint schema")
        copy = json.loads(json.dumps(payload, allow_nan=False))
        return cls(copy["design"], copy["state"])


@dataclass(frozen=True)
class MechanismGrowthResult:
    status: str
    network: ReactionNetworkSpec
    reasons: tuple[dict[str, Any], ...]
    checkpoint: MechanismGrowthCheckpoint

    @property
    def complete(self) -> bool:
        return self.status == "threshold_satisfied"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "native-core-edge-growth-v1",
            "status": self.status,
            "complete": self.complete,
            "network": self.network.to_dict(),
            "reasons": list(self.reasons),
            "checkpoint": self.checkpoint.to_dict(),
        }
