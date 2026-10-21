"""Typed execution receipts for the material boundary of a world transition.

Receipts are produced by runtime services, never accepted as agent action fields.
Inventories inside the boundary (vessels and retained samples) are counted from
state; only actual external sources and sinks belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isclose, isfinite
from typing import Any


@dataclass(frozen=True)
class MaterialPacket:
    """An executed molecular packet, not a second inventory or a balance repair.

    Domain services select the phase and physical withdrawal law. This value
    applies the same explicitly selected amounts to both endpoints and exposes
    them to accounting. Discrete molecule capture uses its integer contract.
    """

    source: str
    destination: str
    amounts_mol: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, str) or not isinstance(self.destination, str)
            or not self.source or not self.destination or self.source == self.destination
        ):
            raise ValueError("material packet requires distinct explicit endpoints")
        values = tuple(tuple(item) for item in self.amounts_mol)
        if len(dict(values)) != len(values) or any(
            not isinstance(key, str) or not key or isinstance(value, bool)
            or not isinstance(value, int | float) or not isfinite(value) or value < 0
            for key, value in values
        ):
            raise ValueError("material packet requires unique finite nonnegative molar amounts")
        object.__setattr__(self, "amounts_mol", tuple(sorted(values)))

    @classmethod
    def from_dict(cls, receipt: dict[str, Any]) -> MaterialPacket:
        amounts = receipt.get("amounts_mol")
        if not isinstance(amounts, dict):
            raise ValueError("executed material packet requires explicit molecular amounts")
        return cls(receipt.get("source"), receipt.get("destination"), tuple(amounts.items()))

    @classmethod
    def representative(cls, source, destination, inventory, *, amount_mol):
        available = fsum(inventory.values())
        if (
            isinstance(amount_mol, bool) or not isinstance(amount_mol, int | float)
            or not isfinite(amount_mol) or not 0 <= amount_mol <= available or available <= 0
        ):
            raise ValueError("feed amount exceeds finite source inventory")
        return cls(source, destination, tuple(
            (key, value / available * amount_mol) for key, value in inventory.items()
        ))

    @property
    def total_mol(self) -> float:
        return fsum(value for _, value in self.amounts_mol)

    def validate_representative(self, inventory, *, requested_mol):
        """Authenticate a receipt from the source law, never a candidate deficit."""
        expected = self.representative(
            self.source, self.destination, inventory, amount_mol=requested_mol
        )
        if self.amounts_mol != expected.amounts_mol:
            raise ValueError("executed packet differs from the declared representative feed")

    def apply(self, source, destination):
        amounts = dict(self.amounts_mol)
        if amounts.keys() - source.keys() or any(
            amount > source[key] for key, amount in self.amounts_mol
        ):
            raise ValueError("material packet exceeds its actual source inventory")
        remaining = {key: value - amounts.get(key, 0) for key, value in source.items()}
        received = dict(destination)
        for key, amount in self.amounts_mol:
            received[key] = received.get(key, 0) + amount
        return remaining, received

    def check_debit(self, before, after):
        amounts = dict(self.amounts_mol)
        for key in before.keys() | after.keys() | amounts.keys():
            if not isclose(
                before.get(key, 0) - after.get(key, 0), amounts.get(key, 0),
                rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise ValueError("finite source debit disagrees with executed feed")

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "destination": self.destination,
                "amounts_mol": dict(self.amounts_mol)}


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


def formal_inventory_exclusions(state):
    """Registered chemical balances make no claims about unformulated entities."""
    if any(local.phase == "formal" for local in (state, *state.inactive_vessels.values())):
        return ("formal_entity_element_inventory", "formal_entity_charge_inventory",
                "formal_entity_energy", "formal_compartment_carrier_mass")
    return ()


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
