"""Pure trigger dependencies for a future event-aware integer step policy.

This module derives immutable channel sets. It has no process, random stream,
clock or event queue and is not connected to the advancing coordinator yet.
"""

from chemworld.physchem.mathml import MathExpression
from chemworld.physchem.sbml_jump_triggers import SBMLJumpTriggers


def guard_species(triggers):
    """Find source species used by guards, including derived reaction quantities.

    Trigger ASTs already inline source assignment rules and lexical functions.
    Reaction-rate symbols still refer to frame recipes; their local bindings
    must shadow global names when following transitive dependencies.
    """
    if not isinstance(triggers, SBMLJumpTriggers):
        raise ValueError("integer trigger dependencies require compiled jump guards")
    frame = triggers.frame
    pending = set().union(*(MathExpression._symbols(node)
                            for _, _, node, _ in triggers._guards))
    seen, species = set(), set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in frame._species:
            species.add(name)
        elif name in frame._recipes:
            pending.update(frame._recipes[name].dependencies)
    return frozenset(species)


def guard_critical_reactions(triggers, channels):
    """Conservatively mark reactions whose net updates can change a guard.

    Full reactants are not net updates: a regenerated catalyst cannot change
    its own stock guard. Source-held species likewise have no channel update.
    """
    species = guard_species(triggers)
    return frozenset(channel.reaction_id for channel in channels
                     if any(name in species and change != 0 for name, change in channel.changes))
