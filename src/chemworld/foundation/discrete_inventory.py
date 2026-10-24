"""Exact molecular interpretation of the existing mol inventory, not a second ledger."""

from math import fsum, isfinite, ulp

AVOGADRO = 6.02214076e23
MAX_COUNT = 2**40


def validate_counts(counts):
    if any(type(n) is not int or not 0 <= n <= MAX_COUNT for n in counts.values()):
        raise ValueError("inventory requires bounded nonnegative integer counts")
    return dict(counts)


def counts_to_mol(counts):
    return {key: n / AVOGADRO for key, n in validate_counts(counts).items()}


def mol_to_counts(amounts):
    counts = {}
    for key, amount in amounts.items():
        if isinstance(amount, bool) or not isfinite(amount) or amount < 0:
            raise ValueError("amount must represent nonnegative integer molecules")
        value = amount * AVOGADRO
        if value > MAX_COUNT:
            raise ValueError("count exceeds exact ledger representability bound")
        integer = round(value)
        if abs(value - integer) > 4 * ulp(value):
            raise ValueError("noninteger molecular dosing")
        counts[key] = integer
    return counts


def integer_element_residual(before, after, balance, substances):
    """Return an exact atom residual when the complete boundary is discrete."""
    locals_before = (before, *before.inactive_vessels.values())
    locals_after = (after, *after.inactive_vessels.values())
    discrete = {s.vessel_id for s in locals_before if s.process.stochastic_continuation is not None}
    mixed = len(discrete) != len(locals_before)
    if not discrete or (mixed and before.lab_clock is None):
        return None
    after_discrete = {
        s.vessel_id for s in locals_after if s.process.stochastic_continuation is not None
    }
    if discrete != after_discrete:
        raise ValueError("discrete inventory lost its declared process state")
    hybrid = _check_hybrid_partitions(locals_before, locals_after, balance, substances)
    residual = {}

    def accumulate(amounts, sign):
        for key, count in mol_to_counts(amounts).items():
            for atom, coefficient in substances[key].formula.items():
                if not float(coefficient).is_integer():
                    raise ValueError("discrete chemistry requires integer atomic formulas")
                residual[atom] = residual.get(atom, 0) + sign * count * int(coefficient)

    for states, sign in ((locals_before, -1), (locals_after, 1)):
        for state in states:
            if state.vessel_id not in discrete or state.vessel_id in hybrid:
                continue
            accumulate(state.species_amounts, sign)
            accumulate(state.state_slots.species_amounts(), sign)
            accumulate(state.samples.total_amounts_mol(), sign)
    for exchange in balance.exchanges:
        if mixed and not {exchange.source, exchange.destination}.intersection(
            f"vessel:{name}" for name in discrete
        ):
            continue
        if exchange.inventory_kind == "species":
            accumulate(dict(exchange.amounts), -1 if exchange.direction == "in" else 1)
    return max(map(abs, residual.values()), default=0)


def _check_hybrid_partitions(before, after, balance, substances):
    """Check a fixed mixed partition without weakening old all-integer boundaries.

    The runtime independently verifies this state declaration against its actual
    physical model and authored closed stock, including at snapshot restore.
    Here the before/after partition must agree and each subset closes separately.
    """
    previous = {state.vessel_id: state for state in before}
    following = {state.vessel_id: state for state in after}
    hybrid = set()

    def partition(state):
        continuation = state.process.stochastic_continuation
        if continuation is None or continuation.get("algorithm") != "integrated-hazard-pdmp-v1":
            return None
        keys = continuation.get("discrete_species")
        if (not isinstance(keys, list) or not keys or len(set(keys)) != len(keys)
                or any(not isinstance(k, str) for k in keys)
                or not set(keys) < state.species_amounts.keys()):
            raise ValueError("hybrid integer subset is invalid")
        return tuple(keys)

    hybrid_identities = {state.vessel_id for state in (*before, *after)
                        if state.process.stochastic_continuation is not None
                        and state.process.stochastic_continuation.get("algorithm")
                        == "integrated-hazard-pdmp-v1"}
    for identity in hybrid_identities:
        if identity not in previous or identity not in following:
            raise ValueError("stochastic vessel inventory was added or removed")
        old, new = previous[identity], following[identity]
        keys = partition(old)
        if keys != partition(new):
            raise ValueError("hybrid fixed integer subset changed during the transition")
        if keys is None:
            continue
        hybrid.add(identity)
        if set(old.species_amounts) != set(new.species_amounts):
            raise ValueError("hybrid species inventory declaration changed")
        if any(exchange.inventory_kind == "species" and
               f"vessel:{identity}" in {exchange.source, exchange.destination}
               for exchange in balance.exchanges):
            raise ValueError("hybrid first slice has no material boundary exchange")
        integer_before = mol_to_counts({k: old.species_amounts[k] for k in keys})
        integer_after = mol_to_counts({k: new.species_amounts[k] for k in keys})
        continuous = set(old.species_amounts) - set(keys)
        atoms = {atom for key in old.species_amounts for atom in substances[key].formula}
        for atom in atoms:
            if any(not float(substances[k].formula.get(atom, 0)).is_integer() for k in keys):
                raise ValueError("hybrid discrete chemistry requires integer atomic formulas")
            delta = sum((integer_after[k] - integer_before[k]) *
                        int(substances[k].formula.get(atom, 0)) for k in keys)
            if delta != 0:
                raise ValueError("hybrid integer atom balance does not close")
            totals = [fsum(state.species_amounts[k] * substances[k].formula.get(atom, 0)
                           for k in continuous) for state in (old, new)]
            if abs(totals[0] - totals[1]) > max(1e-30, 1e-9 * max(map(abs, totals))):
                raise ValueError("hybrid continuous atom balance does not close")
    return hybrid
