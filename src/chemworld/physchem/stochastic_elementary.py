"""Elementary chemical channels shared by declared stochastic approximations."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ElementaryChannel:
    reaction_id: str
    changes: tuple[tuple[str, int], ...]
    orders: tuple[tuple[str, int], ...]
    k: float

    @property
    def participants(self) -> frozenset[str]:
        return frozenset(key for key, _ in (*self.changes, *self.orders))


def compile_elementary_channels(network: object) -> tuple[ElementaryChannel, ...]:
    """Compile integer full reactants, including regenerated species and catalysts."""
    from chemworld.physchem.reaction_network import ReactionNetworkSpec

    if not isinstance(network, ReactionNetworkSpec) or not network.reactions:
        raise ValueError("elementary chemistry requires a nonempty original reaction network")
    if any(not float(n).is_integer() for s in network.species for n in s.composition.values()):
        raise ValueError("elementary stochastic chemistry requires integer atomic formulas")
    channels = []
    for reaction in network.reactions:
        orders = reaction.kinetic_forward_orders
        if (
            reaction.reversible
            or reaction.rate_law.equation_id != "mass_action"
            or reaction.rate_law.uses_activities
            or set(reaction.rate_law.parameters) != {"k"}
            or any(not float(n).is_integer() for n in reaction.stoichiometry.values())
            or any(not float(n).is_integer() or n <= 0 for n in orders.values())
            or sum(orders.values()) > 3
            # Products are full reactants plus the net stoichiometric change.
            # A reactant may therefore be regenerated or have a positive net
            # coefficient (autocatalysis). Every net consumption still needs
            # at least that many explicitly declared reactants.
            or any(orders.get(key, 0) < n for key, n in reaction.reactants.items())
        ):
            raise ValueError(
                "elementary chemistry needs irreversible integer mass action and full reactants"
            )
        k = reaction.rate_law.parameters["k"]
        if isinstance(k, bool) or not isfinite(k) or k < 0:
            raise ValueError("elementary mass-action constant must be finite and nonnegative")
        channels.append(
            ElementaryChannel(
                reaction.reaction_id,
                tuple(sorted((key, int(n)) for key, n in reaction.stoichiometry.items())),
                tuple(sorted((key, int(n)) for key, n in orders.items())),
                float(k),
            )
        )
    return tuple(channels)
