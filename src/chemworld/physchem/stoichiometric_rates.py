"""One pure stoichiometric sum shared by chemical and formal kinetic networks."""


def stoichiometric_amount_rates(species_ids, reaction_stoichiometries, rates, *, rate_scale=1.):
    """Apply extent rates without inventing elemental or thermochemical metadata.

    A scale converts source rates to amount/time. Keep coefficient*rate*scale
    grouping and source iteration order identical to the chemical network's
    historical arithmetic; this extraction does not alter its numerical model.
    """
    derivatives = dict.fromkeys(species_ids, 0.)
    for reaction_id, coefficients in reaction_stoichiometries:
        rate = rates[reaction_id]
        for species_id, coefficient in coefficients.items():
            derivatives[species_id] += coefficient*rate*rate_scale
    return derivatives
