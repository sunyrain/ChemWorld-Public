"""Pure SUBG/SUBQ energies on the common chain arithmetic.

The source-index topology is immutable. These local moments are disposable
expressions of the supplied quadruplet fractions, never a second inventory.
"""

from itertools import combinations_with_replacement, product
from math import prod

from chemworld.physchem.caloric_math import FirstDerivative, logarithm
from chemworld.physchem.mqmqa_topology import canonical_quad
from chemworld.physchem.tdb_expressions import TDB_GAS_CONSTANT_J_MOL_K as R
from chemworld.physchem.tdb_expressions import scalar_value


def _log_term(weight, argument):
    if scalar_value(weight) == 0:
        if isinstance(weight, FirstDerivative):
            raise ValueError("MQMQA derivative requires positive ion/pair/quadruplet amounts")
        return 0.
    return weight*logarithm(argument())


def energies(model, fractions, temperature, pressure):
    topology, source = model._topology, model._source
    cations, anions = topology.cations, topology.anions
    q = {quad: fractions[0][label] for label, quad in zip(
        topology.labels, topology.quadruplets, strict=True)}

    def fraction(a, b, x, y):
        return q[canonical_quad((a, b, x, y))]

    n = topology.ion_amounts(fractions[0])
    pair_n = {(i, k): sum(value*quad.count(i)*quad.count(k) for quad, value in q.items())
              for i, k in product(cations, anions)}
    pair_star_n = {p: count/topology.pairs[p].zeta for p, count in pair_n.items()}
    pair_x = {p: count/sum(pair_n.values()) for p, count in pair_n.items()}
    pair_star_x = {p: count/sum(pair_star_n.values()) for p, count in pair_star_n.items()}
    equivalent = {i: sum(value*quad.count(i)/2 for quad, value in q.items()) for i in n}
    fpair = pair_x if source.phase_type == "SUBG" else pair_star_x
    coordination_fraction = {i: sum(value for p, value in fpair.items() if i in p) for i in n}
    total_cat, total_an = sum(n[i] for i in cations), sum(n[k] for k in anions)
    entropy = sum(_log_term(count, lambda i=i, count=count: count/(
        total_cat if i in cations else total_an)) for i, count in n.items())
    for (i, k), pair in fpair.items():
        weight = pair_n[i, k] if source.phase_type == "SUBG" else pair_star_n[i, k]
        entropy += _log_term(weight, lambda i=i, k=k, pair=pair:
            pair/(coordination_fraction[i]*coordination_fraction[k]))
    alpha, beta = (1, 1) if source.phase_type == "SUBG" else (.75, .5)
    for (a, b, x, y), value in q.items():
        def argument(a=a, b=b, x=x, y=y, value=value):
            multiplicity = (2-(a == b))*(2-(x == y))
            pairs = ((a, x), (a, y), (b, x), (b, y))
            return (value*prod(equivalent[i] for i in (a, b, x, y))**beta
                    /(multiplicity*prod(pair_x[p] for p in pairs)**alpha))

        entropy += _log_term(value, argument)
    reference = 0.
    for parameter in model._parameters:
        if parameter.source.kind != "MQMG":
            continue
        member = source.endmembers[parameter.source.source_index]
        c, a = source.mqmqa.pair_indices[parameter.source.source_index]
        c, a = c-1, len(source.mqmqa.cations)+a-1
        amount = sum(value*quad.count(c)*quad.count(a)/(2*topology.z(c, quad))
                     for quad, value in q.items() if c in quad and a in quad)
        reference += amount*model.database.evaluate_parameter(parameter.source, temperature,
            pressure, extrapolation=model.extrapolation)/member.pair_stoichiometry[0]

    def group(i, j):
        lattice = cations if i in cations else anions
        return (i, *(m for m in lattice if m not in {i, j}
                     and topology.groups[m] == topology.groups[i]
                     and topology.groups[m] != topology.groups[j]))

    def xi(i, j, spectator):
        return sum(pair_n[m, spectator] if i in cations else pair_n[spectator, m]
                   for m in group(i, j))/4

    def chi(i, j, spectator):
        def selected(members):
            result = 0.
            for a, b in combinations_with_replacement(members, 2):
                if source.phase_type == "SUBG":
                    result += (fraction(a, b, spectator, spectator) if i in cations else
                               fraction(spectator, spectator, a, b))
                else:
                    other = anions if i in cations else cations
                    for x, y in product(other, repeat=2):
                        value = (fraction(a, b, x, y) if i in cations else fraction(x, y, a, b))
                        result += value*((x == spectator)+(y == spectator))/2
            return result

        return selected(group(i, j))/selected((*group(i, j), *group(j, i)))

    def mixing(item, a, b, x, y):
        p, s, r_cat, r_an = item.exponents
        if a != b and x != y:
            m, k = item.additional_constituent-1, item.additional_anion_constituent-1
            exponent = next(v for v in item.exponents if v != 0)
            return fraction(m, m, k, k)**exponent
        if a == b and x == y:
            return 1.
        i, j, spectator = (a, b, x) if a != b else (x, y, a)
        first, second = xi(i, j, spectator), xi(j, i, spectator)
        value = (chi(i, j, spectator)**p*chi(j, i, spectator)**s if item.mixing_code == "G"
                 else first**p*second**s/(first+second)**(p+s))
        extra = item.additional_constituent if a != b else item.additional_anion_constituent
        exponent = r_cat if a != b else r_an
        if extra:
            extra -= 1
            extra_pair = ((extra, spectator) if i in cations else (spectator, extra))
            value *= pair_n[extra_pair]/4
            # Xi/Chi groups include their principal ion; nu/gamma do not.
            if extra in group(j, i)[1:]:
                own_pair = ((j, spectator) if i in cations else (spectator, j))
                value *= (1-pair_n[own_pair]/(4*second))**(exponent-1)/second
            elif extra in group(i, j)[1:]:
                own_pair = ((i, spectator) if i in cations else (spectator, i))
                value *= (1-pair_n[own_pair]/(4*first))**(exponent-1)/first
            else:
                value *= (1-first-second)**(exponent-1)
        return value

    excess = 0.
    for parameter in model._parameters:
        if parameter.source.kind != "MQMX":
            continue
        item = source.interactions[parameter.source.source_index]
        a, b, x, y = (i-1 for i in item.indices)
        quad = (a, b, x, y)
        weight = fraction(*quad)
        if a == b:
            weight += topology.z(a, quad)/2*sum(
                fraction(a, m, x, y)/topology.z(a, (a, m, x, y)) for m in cations if m != a)
        if x == y:
            weight += topology.z(x, quad)/2*sum(
                fraction(a, b, x, m)/topology.z(x, (a, b, x, m)) for m in anions if m != x)
        if scalar_value(weight) == 0 and not isinstance(weight, FirstDerivative):
            continue
        coefficient = model.database.evaluate_parameter(parameter.source, temperature,
            pressure, extrapolation=model.extrapolation)
        excess += .5*coefficient*weight*mixing(item, a, b, x, y)
    normalization, _ = model._composition(fractions)
    return {"reference": reference/normalization, "ideal": R*temperature*entropy/normalization,
            "excess": excess/normalization}
