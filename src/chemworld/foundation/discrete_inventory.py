"""Exact molecular interpretation of the existing mol inventory, not a second ledger."""

from math import fsum, isfinite, ulp

AVOGADRO = 6.02214076e23
MAX_COUNT = 2**40
CROSS_PARTITION_ALGORITHM = "stoichiometric-jump-pdmp-v1"
CROSS_PARTITION_TRANSFER_SCHEMA = "hybrid-internal-jump-transfer-v1"
CROSS_PARTITION_ATOM_TOLERANCE = 1e-4
CONTINUOUS_SDE_ALGORITHM = "ri5-ito-reaction-sde-v1"
CONTINUOUS_ADAPTIVE_ALGORITHM = "hysteresis-next-reaction-ode-v1"
CONTINUOUS_REACTION_ALGORITHMS = frozenset({
    CONTINUOUS_SDE_ALGORITHM, CONTINUOUS_ADAPTIVE_ALGORITHM,
})


def validate_cross_partition_transition(before_amounts, after_amounts, discrete_species,
                                        formulae, before_receipt, after_receipt):
    """Check internal integer jump receipts against the one real mol inventory.

    Continuous stocks need not be integers. Their atomic error and floating-point
    resolution must nevertheless remain much smaller than one molecule.
    Receipts are event counts, never an alternative inventory or adjustable tolerance.
    """
    species = set(before_amounts)
    discrete = set(discrete_species)
    if (species != set(after_amounts) or not discrete < species or not discrete
            or not species <= formulae.keys()):
        raise ValueError("cross-partition transition requires the same fixed real species")
    continuous = species - discrete
    atoms = sorted({atom for key in species for atom in formulae[key]})
    for key in species:
        for coefficient in formulae[key].values():
            if (isinstance(coefficient, bool) or not isfinite(coefficient)
                    or not float(coefficient).is_integer() or coefficient < 0):
                raise ValueError("cross-partition chemistry requires integer atomic formulas")
        for values in (before_amounts, after_amounts):
            amount = values[key]
            if isinstance(amount, bool) or not isfinite(amount) or amount < 0:
                raise ValueError("cross-partition inventory must be finite and nonnegative")

    def receipt(value):
        if (not isinstance(value, dict)
                or set(value) != {"schema", "channel_stoichiometry", "firings"}
                or value["schema"] != CROSS_PARTITION_TRANSFER_SCHEMA):
            raise ValueError("invalid cross-partition internal transfer receipt")
        channels, firings = value["channel_stoichiometry"], value["firings"]
        if (not isinstance(channels, dict) or not channels or not isinstance(firings, dict)
                or channels.keys() != firings.keys()):
            raise ValueError("cross-partition receipt needs every declared jump count")
        for identity, stoichiometry in channels.items():
            if (not isinstance(identity, str) or not identity
                    or not isinstance(stoichiometry, dict) or not stoichiometry
                    or not stoichiometry.keys() <= species
                    or not stoichiometry.keys() & discrete
                    or not stoichiometry.keys() & continuous
                    or any(type(n) is not int or n == 0 for n in stoichiometry.values())
                    or type(firings[identity]) is not int or firings[identity] < 0):
                raise ValueError("cross-partition receipt has invalid stoichiometry or count")
            if any(sum(n * int(formulae[key].get(atom, 0))
                       for key, n in stoichiometry.items()) != 0 for atom in atoms):
                raise ValueError("cross-partition jump does not conserve atoms")
        return channels, firings

    channels, old_firings = receipt(before_receipt)
    new_channels, new_firings = receipt(after_receipt)
    if channels != new_channels or any(new_firings[k] < old_firings[k] for k in channels):
        raise ValueError("cross-partition receipt changed its declaration or lost events")
    old_counts = mol_to_counts({k: before_amounts[k] for k in discrete})
    new_counts = mol_to_counts({k: after_amounts[k] for k in discrete})
    residuals = {}
    for atom in atoms:
        transfer = sum((new_firings[channel] - old_firings[channel])
                       * sum(n * int(formulae[k].get(atom, 0))
                             for k, n in stoichiometry.items() if k in discrete)
                       for channel, stoichiometry in channels.items())
        delta = sum((new_counts[k] - old_counts[k]) * int(formulae[k].get(atom, 0))
                    for k in discrete)
        if delta != transfer:
            raise ValueError("cross-partition discrete atom transfer does not close")
        resolution = 0.0
        for key in continuous:
            coefficient = int(formulae[key].get(atom, 0))
            if not coefficient:
                continue
            values = (before_amounts[key], after_amounts[key])
            if any(not isfinite(AVOGADRO * value) for value in values):
                raise ValueError("continuous molecular quantum cannot be represented")
            resolution += abs(coefficient) * max(
                max(ulp(AVOGADRO * value), AVOGADRO * ulp(value)) for value in values
            )
        if resolution > CROSS_PARTITION_ATOM_TOLERANCE / 16:
            raise ValueError("continuous molecular quantum is not numerically resolved")
        delta_continuous = AVOGADRO * fsum(
            int(formulae[k].get(atom, 0)) * (after_amounts[k] - before_amounts[k])
            for k in continuous
        )
        error = delta_continuous + transfer
        if not isfinite(error) or abs(error) > CROSS_PARTITION_ATOM_TOLERANCE:
            raise ValueError("cross-partition continuous atom transfer does not close")
        residuals[atom] = error
    return residuals


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
    continuous = _check_sde_inventories(locals_before, locals_after, balance, substances)
    discrete = {s.vessel_id for s in locals_before
                if s.process.stochastic_continuation is not None} - continuous
    mixed = len(discrete) != len(locals_before)
    if not discrete or (mixed and before.lab_clock is None):
        return None
    after_discrete = {
        s.vessel_id for s in locals_after
        if s.process.stochastic_continuation is not None and s.vessel_id not in continuous
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


def _check_sde_inventories(before, after, balance, substances):
    """Continuous SDE does not weaken the integer checks on any other vessel.

    Actual immutable model binding is checked by the original advance strategy,
    including on restore. Here require the same explicit algorithm across the
    transition and check tiny mol inventories against real boundary receipts.
    """
    previous = {s.vessel_id: s for s in before}
    following = {s.vessel_id: s for s in after}

    def identities(states):
        return {s.vessel_id for s in states
                if s.process.stochastic_continuation is not None
                and s.process.stochastic_continuation.get("algorithm")
                in CONTINUOUS_REACTION_ALGORITHMS}

    continuous = identities(before)
    if continuous != identities(after):
        raise ValueError("continuous SDE process declaration changed during a transition")
    for identity in continuous:
        old, new = previous[identity], following[identity]
        if (old.process.stochastic_continuation['algorithm']
                != new.process.stochastic_continuation['algorithm']):
            raise ValueError("continuous reaction algorithm changed during a transition")
        if set(old.species_amounts) != set(new.species_amounts):
            raise ValueError("continuous SDE species declaration changed")
        atoms = {a for key in old.species_amounts for a in substances[key].formula}
        for atom in atoms:
            totals = [fsum(s.species_amounts[k] * substances[k].formula.get(atom, 0)
                           for k in s.species_amounts) for s in (old, new)]
            packets = []
            for exchange in balance.exchanges:
                if exchange.inventory_kind != "species":
                    continue
                sign = ((exchange.destination == f"vessel:{identity}")
                        - (exchange.source == f"vessel:{identity}"))
                if sign:
                    packets.append(sign * fsum(amount * substances[k].formula.get(atom, 0)
                                               for k, amount in exchange.amounts))
            error = fsum([totals[1], -totals[0], *(-v for v in packets)])
            scale = max(*map(abs, totals), fsum(map(abs, packets)))
            tolerance = 1e-10 * scale
            if old.process.stochastic_continuation['algorithm'] == CONTINUOUS_ADAPTIVE_ALGORITHM:
                tolerance = min(tolerance, 1e-4 / AVOGADRO)
            if not isfinite(error) or abs(error) > max(1e-300, tolerance):
                raise ValueError("continuous SDE atom inventory does not close")
    return continuous


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
        if continuation is None or continuation.get("algorithm") not in {
                "integrated-hazard-pdmp-v1", CROSS_PARTITION_ALGORITHM}:
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
                        in {"integrated-hazard-pdmp-v1", CROSS_PARTITION_ALGORITHM}}
    for identity in hybrid_identities:
        if identity not in previous or identity not in following:
            raise ValueError("stochastic vessel inventory was added or removed")
        old, new = previous[identity], following[identity]
        keys = partition(old)
        if keys != partition(new):
            raise ValueError("hybrid fixed integer subset changed during the transition")
        if keys is None:
            continue
        old_memory, new_memory = (s.process.stochastic_continuation for s in (old, new))
        if old_memory["algorithm"] != new_memory["algorithm"]:
            raise ValueError("hybrid algorithm changed during the transition")
        hybrid.add(identity)
        if set(old.species_amounts) != set(new.species_amounts):
            raise ValueError("hybrid species inventory declaration changed")
        if any(exchange.inventory_kind == "species" and
               f"vessel:{identity}" in {exchange.source, exchange.destination}
               for exchange in balance.exchanges):
            raise ValueError("hybrid first slice has no material boundary exchange")
        if old_memory["algorithm"] == CROSS_PARTITION_ALGORITHM:
            old_receipt = old_memory.get("partition_transfer")
            new_receipt = new_memory.get("partition_transfer")
            validate_cross_partition_transition(
                old.species_amounts, new.species_amounts, keys,
                {key: substances[key].formula for key in old.species_amounts},
                old_receipt, new_receipt,
            )
            totals = [sum(r["firings"].values()) for r in (old_receipt, new_receipt)]
            events = [m.get("events_committed") for m in (old_memory, new_memory)]
            if (any(type(e) is not int or e < n for e, n in zip(events, totals, strict=True))
                    or events[1] - events[0] < totals[1] - totals[0]):
                raise ValueError("cross-partition receipts exceed committed jump events")
            continue
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
