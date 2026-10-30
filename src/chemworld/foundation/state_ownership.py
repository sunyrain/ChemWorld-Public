"""Entity-owned state with explicit units, movement and persistence rules.

Bulk species remain in PhaseLedger. Slots hold only additional entity state
(for example bound adsorbates or instrument drift), never a bulk inventory copy.
Material identity denotes a lot; split portions retain that identity and acquire
distinct locations. Device and vessel identities cannot be split with a sample.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Any, Literal, cast


@dataclass(frozen=True)
class StateFieldSpec:
    name: str
    unit: str
    mixing: Literal["sum", "equal", "weighted"] = "sum"
    weight_field: str | None = None
    minimum: float | None = 0.0
    public: bool = False
    conserved_quantity: str | None = None
    species_id: str | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.unit or self.mixing not in {"sum", "equal", "weighted"}:
            raise ValueError("State fields need a name, unit and explicit mixing rule")
        if (self.mixing == "weighted") != (self.weight_field is not None):
            raise ValueError("Weighted mixing requires exactly one extensive weight field")
        if (self.conserved_quantity or self.species_id) and self.mixing != "sum":
            raise ValueError("Conserved quantities must be extensive")
        if self.species_id and self.unit != "mol":
            raise ValueError("Bound species inventories use mol")
        if self.minimum is not None and not isfinite(self.minimum):
            raise ValueError("State bounds must be finite")
        if self.maximum is not None and not isfinite(self.maximum):
            raise ValueError("State bounds must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.maximum < self.minimum
        ):
            raise ValueError("State field maximum must be at least its minimum")


@dataclass(frozen=True)
class CapacityConstraint:
    capacity_field: str
    occupied_fields: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "occupied_fields", tuple(tuple(p) for p in self.occupied_fields))
        if len(dict(self.occupied_fields)) != len(self.occupied_fields) or any(
            not isfinite(n) or n <= 0 for _, n in self.occupied_fields
        ):
            raise ValueError("Capacity occupancy needs unique fields and positive coefficients")


@dataclass(frozen=True)
class InventoryInteractionSpec:
    """Temperature-independent 1/2 n.epsilon.n / sum(capacity), in J.

    All amounts refer to this entity's existing fields. The normalizer moves
    with its carrier, so splitting preserves extensive energy without storing
    a second energy inventory.
    """

    fields: tuple[str, ...]
    capacity_fields: tuple[str, ...]
    coefficients_J_mol: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "capacity_fields", tuple(self.capacity_fields))
        matrix = tuple(tuple(row) for row in self.coefficients_J_mol)
        object.__setattr__(self, "coefficients_J_mol", matrix)
        size = len(self.fields)
        if (
            not size or len(set(self.fields)) != size
            or not self.capacity_fields
            or len(set(self.capacity_fields)) != len(self.capacity_fields)
            or len(matrix) != size or any(len(row) != size for row in matrix)
            or any(not isfinite(v) for row in matrix for v in row)
        ):
            raise ValueError("interaction energy needs named fields and a finite square matrix")
        if any(matrix[i][j] != matrix[j][i] for i in range(size) for j in range(size)):
            raise ValueError("interaction free energy requires a symmetric matrix")


@dataclass(frozen=True)
class StateSlotSpec:
    slot_id: str
    owner_kind: Literal["material", "vessel", "equipment"]
    owner_id: str
    state_type: str
    fields: tuple[StateFieldSpec, ...]
    mechanism_id: str = ""
    phase_id: str | None = None
    # A change of batch clears material slots, but preserves devices and vessels.
    batch_reset: Literal["clear", "preserve", "initial"] | None = None
    heat_capacity_field: str | None = None
    capacities: tuple[CapacityConstraint, ...] = ()
    # Static state changes only through explicit operations (for example a sealed
    # supply debit), not by advancing the connected physical system's clock.
    time_evolution: Literal["coupled", "static"] = "coupled"
    energy_interactions: tuple[InventoryInteractionSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "capacities", tuple(self.capacities))
        object.__setattr__(self, "energy_interactions", tuple(self.energy_interactions))
        if self.batch_reset is None:
            object.__setattr__(
                self, "batch_reset", "clear" if self.owner_kind == "material" else "preserve"
            )
        if not all((self.slot_id, self.owner_id, self.state_type)):
            raise ValueError("State slots require stable slot, owner and type identities")
        if self.owner_kind not in {"material", "vessel", "equipment"}:
            raise ValueError("Unknown state owner kind")
        if self.batch_reset not in {"clear", "preserve", "initial"}:
            raise ValueError("Unknown batch reset rule")
        if self.time_evolution not in {"coupled", "static"}:
            raise ValueError("Unknown state time-evolution rule")
        fields = {f.name: f for f in self.fields}
        if len(fields) != len(self.fields):
            raise ValueError("Duplicate state field")
        for term in self.energy_interactions:
            if any(
                key not in fields or fields[key].unit != "mol" or fields[key].mixing != "sum"
                or (minimum := fields[key].minimum) is None or minimum < 0
                for key in (*term.fields, *term.capacity_fields)
            ):
                raise ValueError("interaction fields must be nonnegative extensive mol inventories")
            if any(not fields[key].species_id for key in term.fields):
                raise ValueError("interaction energy acts on declared bound species")
            if any(not fields[key].conserved_quantity for key in term.capacity_fields):
                raise ValueError("interaction normalizers must be conserved carrier inventories")
        for spec in self.fields:
            if spec.weight_field and (
                spec.weight_field not in fields
                or fields[spec.weight_field].mixing != "sum"
                or (minimum := fields[spec.weight_field].minimum) is None
                or minimum < 0
            ):
                raise ValueError("Mixing weights must refer to an extensive field")
        if self.heat_capacity_field and (
            self.heat_capacity_field not in fields
            or fields[self.heat_capacity_field].unit != "J/K"
            or fields[self.heat_capacity_field].mixing != "sum"
        ):
            raise ValueError("Thermal mixing requires an extensive heat capacity in J/K")
        for capacity in self.capacities:
            names = (capacity.capacity_field, *(name for name, _ in capacity.occupied_fields))
            if any(name not in fields or fields[name].mixing != "sum" for name in names):
                raise ValueError("Capacity constraints require declared extensive fields")
            if len({fields[name].unit for name in names}) != 1:
                raise ValueError("Capacity and occupied quantities must have the same unit")
            if any((minimum := fields[name].minimum) is None or minimum < 0 for name in names):
                raise ValueError("Capacity inventories must be nonnegative")


@dataclass(frozen=True)
class StateSlot:
    spec: StateSlotSpec
    location_id: str
    values: dict[str, float]
    initial_values: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if not self.location_id:
            raise ValueError("State slots require a location")
        for attr in ("values", "initial_values"):
            values = self.values if getattr(self, attr) is None else getattr(self, attr)
            if set(values) != {f.name for f in self.spec.fields}:
                raise ValueError("State payload does not match its declared fields")
            for spec in self.spec.fields:
                value = values[spec.name]
                if isinstance(value, bool) or not isfinite(value):
                    raise ValueError("State payload must contain finite numbers")
                if spec.minimum is not None and value < spec.minimum:
                    raise ValueError(f"State field {spec.name} is below its lower bound")
                if spec.maximum is not None and value > spec.maximum:
                    raise ValueError(f"State field {spec.name} is above its upper bound")
            object.__setattr__(self, attr, dict(values))
            for constraint in self.spec.capacities:
                occupied = sum(values[k] * n for k, n in constraint.occupied_fields)
                capacity = values[constraint.capacity_field]
                if occupied > capacity + 1e-12 * max(capacity, 1e-20):
                    raise ValueError("Occupied inventory exceeds the declared capacity")

    def coverages(self, capacity_field: str) -> dict[str, float]:
        constraint = next(c for c in self.spec.capacities if c.capacity_field == capacity_field)
        capacity = self.values[capacity_field]
        result = {
            key: n * self.values[key] / capacity if capacity else 0.0
            for key, n in constraint.occupied_fields
        }
        return {**result, "vacant": max(0.0, 1 - sum(result.values())) if capacity else 0.0}

    def portion(self, fraction: float, location_id: str) -> StateSlot:
        if self.spec.owner_kind != "material":
            raise ValueError("Only material state can be split or carried by a sample")
        if isinstance(fraction, bool) or not isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Material fraction must be in [0, 1]")

        def scaled(values: dict[str, float]) -> dict[str, float]:
            return {
                f.name: values[f.name] * fraction if f.mixing == "sum" else values[f.name]
                for f in self.spec.fields
            }

        return replace(
            self,
            location_id=location_id,
            values=scaled(self.values),
            # __post_init__ materializes the optional constructor input.
            initial_values=scaled(cast(dict[str, float], self.initial_values)),
        )

    def move(self, location_id: str) -> StateSlot:
        """Relocate a whole entity; vessel identities themselves cannot move."""
        if self.spec.owner_kind == "vessel" and location_id != self.location_id:
            raise ValueError("Vessel state remains with its vessel")
        return replace(self, location_id=location_id)

    def merge(self, other: StateSlot) -> StateSlot:
        if self.spec != other.spec or self.location_id != other.location_id:
            raise ValueError("Merging slots requires matching ownership, mechanism and location")
        if self.spec.owner_kind != "material":
            raise ValueError("Equipment and vessel state cannot be merged with material")

        def mixed(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
            values = {}
            for spec in self.spec.fields:
                a, b = left[spec.name], right[spec.name]
                if spec.mixing == "sum":
                    values[spec.name] = a + b
                elif spec.mixing == "equal":
                    if a != b:
                        raise ValueError(f"Incompatible material field {spec.name}")
                    values[spec.name] = a
                else:
                    # StateFieldSpec requires a weight exactly for weighted fields.
                    weight = cast(str, spec.weight_field)
                    wa, wb = left[weight], right[weight]
                    if wa + wb == 0 and a != b:
                        raise ValueError("Undefined intensive mixing with zero capacity")
                    values[spec.name] = (wa * a + wb * b) / (wa + wb) if wa + wb else a
            return values

        return replace(
            self,
            values=mixed(self.values, other.values),
            initial_values=mixed(
                cast(dict[str, float], self.initial_values),
                cast(dict[str, float], other.initial_values),
            ),
        )

    def reset(self, *, full: bool = False) -> StateSlot | None:
        if full or self.spec.batch_reset == "initial":
            return replace(self, values=cast(dict[str, float], self.initial_values))
        return None if self.spec.batch_reset == "clear" else self

    def to_dict(self, *, include_hidden: bool = True) -> dict[str, Any]:
        if include_hidden:
            payload = asdict(self)
            # Adding an optional bound must not rewrite every existing slot's
            # serialized declaration or its original continuation bindings.
            for field in payload["spec"]["fields"]:
                if field["maximum"] is None:
                    del field["maximum"]
            return payload
        return {
            f.name: {"value": self.values[f.name], "unit": f.unit}
            for f in self.spec.fields
            if f.public
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StateSlot:
        spec = dict(payload["spec"])
        spec["fields"] = tuple(StateFieldSpec(**f) for f in spec["fields"])
        spec["capacities"] = tuple(CapacityConstraint(**c) for c in spec.get("capacities", ()))
        spec["energy_interactions"] = tuple(
            InventoryInteractionSpec(**item) for item in spec.get("energy_interactions", ())
        )
        return cls(
            StateSlotSpec(**spec),
            payload["location_id"],
            payload["values"],
            payload.get("initial_values"),
        )


@dataclass(frozen=True)
class StateSlots:
    slots: tuple[StateSlot, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", deepcopy(tuple(self.slots)))
        if len({s.spec.slot_id for s in self.slots}) != len(self.slots):
            raise ValueError("Duplicate slot identity in one location")

    def material(self, *, phase_id: str | None = None) -> StateSlots:
        return StateSlots(
            tuple(
                s
                for s in self.slots
                if s.spec.owner_kind == "material"
                and (phase_id is None or s.spec.phase_id == phase_id)
            )
        )

    def stationary(self) -> StateSlots:
        return StateSlots(tuple(s for s in self.slots if s.spec.owner_kind != "material"))

    def has_inventory(self) -> bool:
        return any(
            s.values[f.name] > 0 for s in self.slots for f in s.spec.fields if f.mixing == "sum"
        )

    def portion(self, fraction: float, location_id: str) -> StateSlots:
        return StateSlots(tuple(s.portion(fraction, location_id) for s in self.slots))

    def split_material(
        self, fraction: float, source: str, destination: str, *,
        phase_id: str | None = None, destination_phase_id: str | None = None,
        phase_ids: tuple[str | None, ...] | None = None,
    ) -> tuple[StateSlots, StateSlots]:
        """Split only the declared material selection; stationary owners stay put.

        Unselected phases retain their exact state. Phase rebinding belongs to
        this ownership boundary; field scaling and mixing use the slot rules.
        """
        if (
            isinstance(fraction, bool) or not isfinite(fraction) or not 0 <= fraction <= 1
            or not source or not destination or source == destination
        ):
            raise ValueError("material split requires a valid fraction and distinct locations")
        retained, moved = [], []
        for slot in self.slots:
            if slot.spec.owner_kind != "material" or (
                phase_id is not None and slot.spec.phase_id != phase_id
            ) or (
                phase_ids is not None and slot.spec.phase_id not in phase_ids
            ):
                retained.append(slot)
                continue
            if slot.location_id != source:
                raise ValueError("material selection is not owned by the source location")
            retained.append(slot.portion(1 - fraction, source))
            portion = slot.portion(fraction, destination)
            if destination_phase_id is not None:
                portion = replace(
                    portion, spec=replace(portion.spec, phase_id=destination_phase_id)
                )
            moved.append(portion)
        return StateSlots(tuple(retained)), StateSlots(tuple(moved))

    def receive_material(self, incoming: StateSlots) -> StateSlots:
        """Merge carried material while retaining each stationary entity once."""
        return StateSlots((*self.stationary().slots, *self.material().merge(incoming).slots))

    def merge(self, other: StateSlots) -> StateSlots:
        mechanisms: dict[str, str] = {}
        result = {s.spec.slot_id: s for s in self.slots}
        for s in (*self.slots, *other.slots):
            if s.spec.owner_kind != "material":
                raise ValueError("Only material slots belong in material mixtures")
            if (
                s.spec.state_type in mechanisms
                and mechanisms[s.spec.state_type] != s.spec.mechanism_id
            ):
                raise ValueError("Material mechanisms are incompatible")
            mechanisms[s.spec.state_type] = s.spec.mechanism_id
        for s in other.slots:
            result[s.spec.slot_id] = (
                result[s.spec.slot_id].merge(s) if s.spec.slot_id in result else s
            )
        return StateSlots(tuple(result[k] for k in sorted(result)))

    def conserved(self) -> dict[tuple[str, str], float]:
        result: dict[tuple[str, str], float] = {}
        for slot in self.slots:
            for spec in slot.spec.fields:
                if spec.conserved_quantity:
                    key = (spec.conserved_quantity, spec.unit)
                    result[key] = result.get(key, 0) + slot.values[spec.name]
        return result

    def species_amounts(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for slot in self.slots:
            for spec in slot.spec.fields:
                if spec.species_id:
                    result[spec.species_id] = (
                        result.get(spec.species_id, 0) + slot.values[spec.name]
                    )
        return result

    def heat_capacity(self) -> float:
        return sum(
            s.values[s.spec.heat_capacity_field] for s in self.slots if s.spec.heat_capacity_field
        )

    def coverages(self, state_type: str, capacity_field: str) -> dict[str, float]:
        slots = [s for s in self.slots if s.spec.state_type == state_type]
        if len({(s.spec.mechanism_id, s.spec.capacities) for s in slots}) > 1:
            raise ValueError("Coverage aggregation requires one compatible site mechanism")
        capacity = sum(s.values[capacity_field] for s in slots)
        result: dict[str, float] = {}
        for slot in slots:
            for name, value in slot.coverages(capacity_field).items():
                result[name] = result.get(name, 0) + value * slot.values[capacity_field]
        return {name: value / capacity if capacity else 0.0 for name, value in result.items()}

    def reset(self, *, full: bool = False) -> StateSlots:
        return StateSlots(
            tuple(value for s in self.slots if (value := s.reset(full=full)) is not None)
        )

    def to_dict(self, *, include_hidden: bool = True) -> dict[str, Any]:
        return {
            s.spec.slot_id: value
            for s in self.slots
            if (value := s.to_dict(include_hidden=include_hidden))
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any], *, order: list[str] | tuple[str, ...] | None = None,
    ) -> StateSlots:
        if order is not None and (
            not isinstance(order, list | tuple) or any(not isinstance(key, str) for key in order)
        ):
            raise ValueError("serialized material slot order must be an identity sequence")
        keys = tuple(payload) if order is None else tuple(order)
        if len(set(keys)) != len(keys) or set(keys) != set(payload):
            raise ValueError("serialized material slot order must match its declared identities")
        slots = tuple(StateSlot.from_dict(payload[key]) for key in keys)
        if set(payload) != {s.spec.slot_id for s in slots}:
            raise ValueError("Serialized slot identities disagree")
        return cls(slots)
