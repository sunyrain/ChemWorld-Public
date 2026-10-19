"""Private multi-entity intervention for the bounded equilibrium world."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from chemworld.world.scenario import ScenarioInstance

EQUILIBRIUM_ENTITY_PANEL_VERSION = "chemworld-equilibrium-entity-panel-0.2"


@dataclass(frozen=True)
class EquilibriumEntityProfileSpec:
    selector: int
    pka_shift: float
    log10_ksp: float
    cation_fraction: float
    activity_coefficient_ratio: float

    def __post_init__(self) -> None:
        values = (
            self.pka_shift,
            self.log10_ksp,
            self.cation_fraction,
            self.activity_coefficient_ratio,
        )
        if self.selector not in {0, 1, 2}:
            raise ValueError("EQ-E selector must be 0, 1, or 2")
        if not all(isfinite(value) for value in values):
            raise ValueError("EQ-E entity properties must be finite")
        if not 0.0 < self.cation_fraction < 1.0:
            raise ValueError("EQ-E cation fraction must be in (0, 1)")
        if self.activity_coefficient_ratio <= 0.0:
            raise ValueError("EQ-E activity ratio must be positive")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EquilibriumEntityProfileSpec:
        expected = {
            "selector",
            "pka_shift",
            "log10_ksp",
            "cation_fraction",
            "activity_coefficient_ratio",
        }
        if set(payload) != expected:
            raise ValueError("EQ-E entity profile fields changed")
        return cls(
            selector=int(payload["selector"]),
            pka_shift=float(payload["pka_shift"]),
            log10_ksp=float(payload["log10_ksp"]),
            cation_fraction=float(payload["cation_fraction"]),
            activity_coefficient_ratio=float(payload["activity_coefficient_ratio"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selector": self.selector,
            "pka_shift": self.pka_shift,
            "log10_ksp": self.log10_ksp,
            "cation_fraction": self.cation_fraction,
            "activity_coefficient_ratio": self.activity_coefficient_ratio,
        }


@dataclass(frozen=True)
class EquilibriumEntityPanelIntervention:
    base_pka: float
    profiles: tuple[EquilibriumEntityProfileSpec, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.base_pka):
            raise ValueError("EQ-E base pKa must be finite")
        if len(self.profiles) != 3 or {row.selector for row in self.profiles} != {0, 1, 2}:
            raise ValueError("EQ-E panel requires selectors 0, 1, and 2 exactly once")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EquilibriumEntityPanelIntervention:
        expected = {"kind", "version", "base_pka", "profiles"}
        if set(payload) != expected:
            raise ValueError("EQ-E panel intervention fields changed")
        if payload["kind"] != "equilibrium_entity_panel":
            raise ValueError("invalid EQ-E intervention kind")
        if payload["version"] != "eq-e-v0.2":
            raise ValueError("unsupported EQ-E intervention version")
        profiles = payload["profiles"]
        if not isinstance(profiles, list):
            raise ValueError("EQ-E profiles must be a list")
        return cls(
            base_pka=float(payload["base_pka"]),
            profiles=tuple(
                EquilibriumEntityProfileSpec.from_dict(dict(row)) for row in profiles
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "equilibrium_entity_panel",
            "version": "eq-e-v0.2",
            "base_pka": self.base_pka,
            "profiles": [row.to_dict() for row in self.profiles],
        }


def equilibrium_entity_panel_hash(
    intervention: EquilibriumEntityPanelIntervention,
) -> str:
    payload = {
        "contract_version": EQUILIBRIUM_ENTITY_PANEL_VERSION,
        "intervention": intervention.to_dict(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def apply_equilibrium_entity_panel(
    instance: ScenarioInstance,
    intervention: EquilibriumEntityPanelIntervention,
) -> ScenarioInstance:
    """Attach a three-medium property panel without changing reaction topology."""

    if instance.spec.scenario_id != "equilibrium-characterization":
        raise ValueError("EQ-E entity panels apply only to equilibrium-characterization")
    contract_hash = equilibrium_entity_panel_hash(intervention)
    metadata = {
        **instance.initial_state.metadata,
        "equilibrium_entity_panel_version": EQUILIBRIUM_ENTITY_PANEL_VERSION,
        "equilibrium_entity_panel_hash": contract_hash,
        "equilibrium_entity_base_pka": intervention.base_pka,
        "equilibrium_entity_profiles": {
            str(row.selector): row.to_dict() for row in intervention.profiles
        },
        "equilibrium_mechanism_family": "direct_free_ion_precipitation",
        "equilibrium_aqueous_intermediate_present": False,
    }
    parameters = replace(
        instance.parameters,
        world_id=f"{instance.parameters.world_id}:eq-entity-{contract_hash[:12]}",
        provider=f"{instance.parameters.provider}+equilibrium-entity-panel",
    )
    return replace(
        instance,
        parameters=parameters,
        initial_state=instance.initial_state.replace(metadata=metadata),
    )


__all__ = [
    "EQUILIBRIUM_ENTITY_PANEL_VERSION",
    "EquilibriumEntityPanelIntervention",
    "EquilibriumEntityProfileSpec",
    "apply_equilibrium_entity_panel",
    "equilibrium_entity_panel_hash",
]
