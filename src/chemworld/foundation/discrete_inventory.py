"""Exact molecular interpretation of the existing mol inventory, not a second ledger."""

from math import isfinite, ulp

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
    residual = {}

    def accumulate(amounts, sign):
        for key, count in mol_to_counts(amounts).items():
            for atom, coefficient in substances[key].formula.items():
                if not float(coefficient).is_integer():
                    raise ValueError("discrete chemistry requires integer atomic formulas")
                residual[atom] = residual.get(atom, 0) + sign * count * int(coefficient)

    for states, sign in ((locals_before, -1), (locals_after, 1)):
        for state in states:
            if state.vessel_id not in discrete:
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
