"""Source-declared extensive balances without invented molecular identities.

Empirical masses and equivalents live in the original owned slots. Reaction
progress is a receipt, not an alternative inventory. A model adapter must bind
the static declaration to its original physical specification on installation
and restoration; this layer checks every committed material transition.
"""

from collections.abc import Mapping
from math import fsum, isfinite

from chemworld.foundation.units import unit_spec

EMPIRICAL_BALANCE_SCHEMA = "declared-empirical-reaction-balance-v1"
EMPIRICAL_ABSOLUTE_TOLERANCE = 1e-10
EMPIRICAL_RELATIVE_TOLERANCE = 1e-8


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def validate_empirical_basis(basis):
    """Validate units, named budget rows and declared linear invariants."""
    fields = {
        "schema",
        "mechanism_id",
        "member_vessel_ids",
        "quantities",
        "reactions",
        "inventory_fields",
        "extent_state_type",
        "invariants",
    }
    if (
        not isinstance(basis, Mapping)
        or set(basis) != fields
        or basis["schema"] != EMPIRICAL_BALANCE_SCHEMA
    ):
        raise ValueError("empirical balance requires its complete declared schema")
    for key in ("mechanism_id", "extent_state_type"):
        if not isinstance(basis[key], str) or not basis[key]:
            raise ValueError("empirical balance requires named mechanism and progress types")
    members = basis["member_vessel_ids"]
    if (
        not isinstance(members, (list, tuple))
        or not members
        or any(not isinstance(v, str) or not v for v in members)
        or len(set(members)) != len(members)
    ):
        raise ValueError("empirical balance requires unique finite member vessels")
    quantities, reactions = basis["quantities"], basis["reactions"]
    if not isinstance(quantities, Mapping) or not quantities:
        raise ValueError("empirical balance has no declared quantities")
    if not isinstance(reactions, Mapping) or not reactions:
        raise ValueError("empirical balance has no declared reactions")
    for definitions, extra in ((quantities, set()), (reactions, {"stoichiometry"})):
        for key, definition in definitions.items():
            if (
                not isinstance(key, str)
                or not key
                or not isinstance(definition, Mapping)
                or set(definition) != {"unit", "basis"} | extra
                or not isinstance(definition["basis"], str)
                or not definition["basis"]
            ):
                raise ValueError("empirical quantity or reaction lacks an explicit unit/basis")
            unit_spec(definition["unit"])
    for reaction in reactions.values():
        row = reaction["stoichiometry"]
        if (
            not isinstance(row, Mapping)
            or not row
            or set(row) - set(quantities)
            or any(not _number(v) for v in row.values())
        ):
            raise ValueError("empirical stoichiometry must use finite declared budget rows")
    mappings = basis["inventory_fields"]
    if not isinstance(mappings, Mapping) or not mappings or basis["extent_state_type"] in mappings:
        raise ValueError("empirical inventory and progress need distinct slot types")
    covered = set()
    for slot_type, row in mappings.items():
        if (
            not isinstance(slot_type, str)
            or not slot_type
            or not isinstance(row, Mapping)
            or not row
            or any(not isinstance(k, str) or not k for k in row)
            or any(q not in quantities for q in row.values())
        ):
            raise ValueError("empirical slot fields must map to known quantities")
        covered.update(row.values())
    if covered != set(quantities):
        raise ValueError("every empirical quantity must have an actual owned inventory field")
    invariants = basis["invariants"]
    if not isinstance(invariants, Mapping):
        raise ValueError("empirical linear invariants must be a named mapping")
    for name, invariant in invariants.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(invariant, Mapping)
            or set(invariant) != {"unit", "coefficients"}
        ):
            raise ValueError("invalid empirical invariant declaration")
        unit_spec(invariant["unit"])
        coefficients = invariant["coefficients"]
        if (
            not isinstance(coefficients, Mapping)
            or not coefficients
            or set(coefficients) - set(quantities)
            or any(not _number(v) for v in coefficients.values())
            or any(quantities[q]["unit"] != invariant["unit"] for q in coefficients)
        ):
            raise ValueError("empirical invariant requires finite coefficients in one unit")
        for reaction in reactions.values():
            terms = [v * reaction["stoichiometry"].get(q, 0.0) for q, v in coefficients.items()]
            if abs(fsum(terms)) > 1e-12 * max(1.0, fsum(abs(v) for v in terms)):
                raise ValueError("empirical reaction violates a declared linear invariant")


def validate_empirical_transition(before, after, before_extents, after_extents, basis):
    """Check a closed finite system's budgets in each row's own physical unit."""
    validate_empirical_basis(basis)
    quantities, reactions = basis["quantities"], basis["reactions"]
    for values, expected in (
        (before, quantities),
        (after, quantities),
        (before_extents, reactions),
        (after_extents, reactions),
    ):
        if (
            not isinstance(values, Mapping)
            or set(values) != set(expected)
            or any(not _number(v) or v < 0 for v in values.values())
        ):
            raise ValueError(
                "empirical inventory/progress must be complete finite nonnegative values"
            )
    if any(after_extents[r] < before_extents[r] for r in reactions):
        raise ValueError("empirical irreversible reaction progress cannot decrease")
    residuals = {}
    for q in quantities:
        source = fsum(
            reactions[r]["stoichiometry"].get(q, 0.0) * (after_extents[r] - before_extents[r])
            for r in reactions
        )
        residual = fsum((after[q], -before[q], -source))
        tolerance = EMPIRICAL_ABSOLUTE_TOLERANCE + EMPIRICAL_RELATIVE_TOLERANCE * max(
            before[q], after[q], abs(source)
        )
        if not isfinite(residual) or abs(residual) > tolerance:
            raise ValueError(
                f"empirical quantity {q} does not close against original reaction progress"
            )
        residuals[q] = residual
    for name, invariant in basis["invariants"].items():
        weights = invariant["coefficients"]
        old = fsum(c * before[q] for q, c in weights.items())
        new = fsum(c * after[q] for q, c in weights.items())
        tolerance = EMPIRICAL_ABSOLUTE_TOLERANCE + EMPIRICAL_RELATIVE_TOLERANCE * max(
            abs(old), abs(new)
        )
        if not isfinite(new) or abs(new - old) > tolerance:
            raise ValueError(f"empirical invariant {name} does not close")
    return residuals


def _collect(states, basis):
    quantities = {q: [] for q in basis["quantities"]}
    progress = {r: [] for r in basis["reactions"]}
    slots, represented = {}, set()
    for vessel in basis["member_vessel_ids"]:
        if vessel not in states:
            raise ValueError("empirical finite member vessel is missing")
        state = states[vessel]
        for slot in state.state_slots.slots:
            if slot.spec.mechanism_id != basis["mechanism_id"]:
                continue
            key = (vessel, slot.spec.slot_id)
            if key in slots or slot.location_id != vessel:
                raise ValueError("empirical inventory must have one actual local owner")
            fields = {f.name: f for f in slot.spec.fields}
            if any(
                f.mixing != "sum" or f.species_id or f.conserved_quantity for f in fields.values()
            ):
                raise ValueError(
                    "empirical fields are extensive budgets, not molecular/conserved aliases"
                )
            if slot.spec.state_type == basis["extent_state_type"]:
                if set(fields) != set(progress) or slot.spec.owner_kind != "equipment":
                    raise ValueError(
                        "empirical progress must belong to the actual reacting equipment"
                    )
                record = state.equipment.equipment.get(slot.spec.owner_id)
                if record is None or record.attached_vessel_id != vessel:
                    raise ValueError("empirical progress owner is not installed at this vessel")
                for r, value in slot.values.items():
                    if fields[r].unit != basis["reactions"][r]["unit"]:
                        raise ValueError("empirical reaction progress unit differs from its source")
                    progress[r].append(value)
            else:
                mapping = basis["inventory_fields"].get(slot.spec.state_type)
                if (
                    mapping is None
                    or set(fields) != set(mapping)
                    or slot.spec.owner_kind != "material"
                ):
                    raise ValueError(
                        "empirical material slot differs from its declared budget basis"
                    )
                for field, q in mapping.items():
                    if fields[field].unit != basis["quantities"][q]["unit"]:
                        raise ValueError("empirical material field unit differs from its source")
                    quantities[q].append(slot.values[field])
                    represented.add(q)
            slots[key] = slot
    if represented != set(quantities) or any(not v for v in progress.values()):
        raise ValueError("empirical system omits material or reaction-progress inventory")
    return (
        {q: fsum(v) for q, v in quantities.items()},
        {r: fsum(v) for r, v in progress.items()},
        slots,
    )


def check_empirical_transitions(before, after, balance):
    """Consume the original slots at the existing all-vessels transaction boundary."""
    old_states = {before.vessel_id: before, **before.inactive_vessels}
    new_states = {after.vessel_id: after, **after.inactive_vessels}

    def declarations(states):
        found = {}
        for vessel, state in states.items():
            for identity, record in state.equipment.equipment.items():
                if record.attached_vessel_id != vessel:
                    continue
                basis = record.settings.get("empirical_reaction_balance")
                if basis is not None:
                    found[(vessel, identity)] = basis
        return found

    original, candidate = declarations(old_states), declarations(new_states)
    if original != candidate:
        raise ValueError("empirical source declaration changed across a material transaction")
    claimed = set()
    for basis in original.values():
        validate_empirical_basis(basis)
        members = set(basis["member_vessel_ids"])
        if claimed & members:
            raise ValueError("empirical systems cannot claim the same material vessel")
        claimed.update(members)
        endpoints = {f"vessel:{v}" for v in members} | members
        if any(
            e.inventory_kind == "entity"
            and (
                e.source in endpoints
                or e.destination in endpoints
                or any(q in basis["quantities"] for q, _ in e.amounts)
            )
            for e in balance.exchanges
        ):
            raise ValueError("empirical reaction sources cannot be disguised as external exchanges")
        old, old_progress, old_slots = _collect(old_states, basis)
        new, new_progress, new_slots = _collect(new_states, basis)
        if set(old_slots) != set(new_slots) or any(
            old_slots[k].spec != new_slots[k].spec for k in old_slots
        ):
            raise ValueError("empirical inventory/progress ownership changed during finite advance")
        for key, slot in old_slots.items():
            if slot.spec.state_type == basis["extent_state_type"] and any(
                new_slots[key].values[r] < value for r, value in slot.values.items()
            ):
                raise ValueError("individual reactor progress cannot decrease")
        validate_empirical_transition(old, new, old_progress, new_progress, basis)
