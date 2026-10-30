"""Declared ideal wet-cake/filtrate recovery using existing aqueous phase owners."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Any

from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.physchem.aqueous_mineral_equilibrium import AqueousMineralEquilibriumSpec
from chemworld.runtime.aqueous_filtration import AQUEOUS_FILTER_TYPE
from chemworld.world.aqueous_precipitation_world import build_aqueous_precipitation_world
from chemworld.world.composition import WorldCompositionSpec


@dataclass(frozen=True)
class AqueousFilterSpec:
    filter_id: str
    source_vessel: str
    cake_vessel: str
    filtrate_vessel: str
    solid_retention_fraction: float
    cake_liquid_fraction: float

    def __post_init__(self) -> None:
        if (
            any(
                not isinstance(v, str) or not v
                for v in (
                    self.filter_id,
                    self.source_vessel,
                    self.cake_vessel,
                    self.filtrate_vessel,
                )
            )
            or len({self.source_vessel, self.cake_vessel, self.filtrate_vessel}) != 3
        ):
            raise ValueError("filter requires an identity and three distinct finite vessels")
        for value in (self.solid_retention_fraction, self.cake_liquid_fraction):
            if isinstance(value, bool) or not isfinite(value):
                raise ValueError("filter retention fractions must be finite explicit numbers")
        if not 0 <= self.solid_retention_fraction <= 1 or not 0 < self.cake_liquid_fraction < 1:
            raise ValueError("solid retention must be in [0,1] and liquid retention in (0,1)")

    def to_dict(self) -> dict[str, Any]:
        return {"model": "declared-instantaneous-aqueous-filtration-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AqueousFilterSpec":
        if raw.get("model") != "declared-instantaneous-aqueous-filtration-v1":
            raise ValueError("unknown aqueous filtration declaration")
        return cls(**{k: v for k, v in raw.items() if k != "model"})


def build_aqueous_filtration_world(
    spec: AqueousMineralEquilibriumSpec,
    inventories: Mapping[str, Mapping[str, float]],
    filters: Sequence[AqueousFilterSpec],
    connections: Sequence[tuple[str, str, str]] = (),
    *,
    temperature_K: float = 298.15,
    pressure_Pa: float = 1e5,
    capacities_L: Mapping[str, float] | None = None,
    budget: int = 100,
) -> WorldCompositionSpec:
    """Both filter outlets are compiled connections to distinct original empty receivers."""
    if (
        not filters
        or len({f.filter_id for f in filters}) != len(filters)
        or len({f.source_vessel for f in filters}) != len(filters)
    ):
        raise ValueError("declare unique filter identities, with one filter per source")
    members = {v for f in filters for v in (f.source_vessel, f.cake_vessel, f.filtrate_vessel)}
    if members - set(inventories):
        raise ValueError("all filtration endpoints must be installed finite vessels")
    for f in filters:
        if any(inventories[f.cake_vessel].values()) or any(inventories[f.filtrate_vessel].values()):
            raise ValueError("filter receivers must begin as actual empty finite vessels")
    base = build_aqueous_precipitation_world(
        spec,
        inventories,
        connections,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        capacities_L=capacities_L,
        allow_empty_vessels=tuple(sorted(members)),
        budget=budget,
    )
    if not (base.physics is not None):
        raise ValueError("aqueous filtration requires composition physics")
    states = dict(base.physics.initial_states)
    payload = base.to_dict()
    for f in filters:
        state = states[f.source_vessel]
        if not (state.equipment is not None):
            raise ValueError("aqueous filtration source requires equipment state")
        record = EquipmentRecord(
            f.filter_id,
            AQUEOUS_FILTER_TYPE,
            f.source_vessel,
            batch_reset="preserve",
            settings={"filter_specification": f.to_dict()},
        )
        states[f.source_vessel] = state.replace(
            equipment=EquipmentLedger(
                {
                    **state.equipment.equipment,
                    record.equipment_id: record,
                }
            )
        )
        payload["components"].append(
            {
                "id": f.filter_id,
                "kind": "aqueous_filter",
                "vessel": f.source_vessel,
                "parameters": {},
            }
        )
        for port, target in (("cake", f.cake_vessel), ("filtrate", f.filtrate_vessel)):
            payload["connections"].append(
                {
                    "id": f.filter_id + ":" + port,
                    "unit": "mol",
                    "source": {"component": f.filter_id, "port": port},
                    "target": {"component": target + "-solution", "port": "in"},
                }
            )
    payload["physics"] = replace(base.physics, initial_states=states).to_dict()
    payload["task"]["operations"].append("filter_crystals")
    payload["task"]["objective"] = "finite precipitation, wet-cake recovery and finite washing"
    payload["composition_id"] = "declared-aqueous-filtration"
    return WorldCompositionSpec.from_dict(payload)
