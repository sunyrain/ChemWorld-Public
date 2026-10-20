"""Typed execution receipts for the material boundary of a world transition.

Receipts are produced by runtime services, never accepted as agent action fields.
Inventories inside the boundary (vessels and retained samples) are counted from
state; only actual external sources and sinks belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class MaterialExchange:
    exchange_id: str
    source: str
    destination: str
    direction: str
    amounts: tuple[tuple[str, float], ...]
    unit: str = "mol"
    inventory_kind: str = "species"

    def __post_init__(self) -> None:
        if not self.exchange_id or not self.source or not self.destination:
            raise ValueError("material exchange requires identity, source and destination")
        if self.source == self.destination or self.direction not in {"in", "out"}:
            raise ValueError("invalid material exchange endpoints or direction")
        units = {"species": "mol", "carrier": "L"}
        if self.inventory_kind == "entity":
            if not isinstance(self.unit,str) or not self.unit:
                raise ValueError("entity exchanges require their explicit declared unit")
        elif self.inventory_kind not in units or self.unit != units[self.inventory_kind]:
            raise ValueError("species exchanges require mol; carrier exchanges require L")
        if len(dict(self.amounts)) != len(self.amounts):
            raise ValueError("duplicate species in material exchange")
        if any(not key or not isfinite(value) or value < 0 for key, value in self.amounts):
            raise ValueError("material exchange requires finite nonnegative quantities")

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange_id": self.exchange_id,
            "source": self.source,
            "destination": self.destination,
            "direction": self.direction,
            "amounts": dict(self.amounts),
            "unit": self.unit,
            "inventory_kind": self.inventory_kind,
        }


@dataclass(frozen=True)
class TransitionBalance:
    """One fixed boundary: registered species in all vessels and retained samples.

    The legacy carrier model has volumes, not molecular compositions. Its existing
        volume bookkeeping is not an elemental/energy proof. Do not report
    unmodelled quantities as zero-residual checks.
    """

    exchanges: tuple[MaterialExchange, ...] = ()
    stock_consumption: tuple[tuple[str, float], ...] = ()
    unmodeled_quantities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len({item.exchange_id for item in self.exchanges}) != len(self.exchanges):
            raise ValueError("duplicate material exchange receipt")
        if len(dict(self.stock_consumption)) != len(self.stock_consumption):
            raise ValueError("duplicate stock consumption")
        if any(not isfinite(value) or value < 0 for _, value in self.stock_consumption):
            raise ValueError("stock consumption must be finite and nonnegative")

    def to_dict(self, *, include_hidden: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "boundary_id": "registered-species:all-vessels-and-retained-samples:v1",
            "quantities": ["elements_mol", "carrier_volumes_L", "declared_entity_quantities"],
            "exchange_count": len(self.exchanges),
            "stock_consumption": dict(self.stock_consumption),
            "not_covered": [
                "carrier_element_inventory",
                "general_energy",
                "general_charge",
                *self.unmodeled_quantities,
            ],
        }
        if include_hidden:
            payload["exchanges"] = [item.to_dict() for item in self.exchanges]
        return payload
