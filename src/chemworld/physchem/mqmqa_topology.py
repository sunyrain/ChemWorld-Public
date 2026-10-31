"""Immutable MQMQA coordination and atom maps over original source indices."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, replace
from functools import cache
from itertools import combinations_with_replacement, product
from math import isclose, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, overload

from chemworld.physchem.tdb_expressions import TdbScalar

if TYPE_CHECKING:
    from chemworld.data.chem_sage import ChemSageDatabase
    from chemworld.data.chem_sage_records import ChemSageEndmember
    from chemworld.data.tdb import TdbSpecies


def canonical_quad(quad: tuple[int, ...]) -> tuple[int, ...]:
    return (*sorted(quad[:2]), *sorted(quad[2:]))


@dataclass(frozen=True)
class MQMQATopology:
    cations: tuple[int, ...]
    anions: tuple[int, ...]
    labels: tuple[str, ...]
    quadruplets: tuple[tuple[int, ...], ...]
    charges: Mapping[int, float]
    groups: Mapping[int, int]
    ions: Mapping[int, TdbSpecies]
    coordinations: Mapping[tuple[int, ...], tuple[float, ...]]
    pairs: Mapping[tuple[int, int], ChemSageEndmember]

    def z(self, species: int, quad: tuple[int, ...]) -> float:
        quad = canonical_quad(quad)
        return self.coordinations[quad][quad.index(species)]

    @overload
    def ion_amounts(self, fractions: Mapping[str, float]) -> Mapping[int, float]: ...

    @overload
    def ion_amounts(self, fractions: Mapping[str, TdbScalar]) -> Mapping[int, TdbScalar]: ...

    def ion_amounts(self, fractions: Mapping[str, TdbScalar]) -> Mapping[int, TdbScalar]:
        return {i: sum(fractions[label]*quad.count(i)/self.z(i, quad)
                       for label, quad in zip(self.labels, self.quadruplets, strict=True)
                       if i in quad) for i in self.ions}

    def restrict(self, eligible: Collection[str]) -> MQMQATopology:
        cations = tuple(i for i in self.cations if self.ions[i].name in eligible)
        anions = tuple(i for i in self.anions if self.ions[i].name in eligible)
        if not cations or not anions:
            raise ValueError("MQMQA active ion topology needs both cations and anions")
        active = {*cations, *anions}
        selected = [(label, quad) for label, quad in zip(self.labels, self.quadruplets, strict=True)
                    if set(quad) <= active]
        return replace(self, cations=cations, anions=anions,
            labels=tuple(label for label, _ in selected),
            quadruplets=tuple(quad for _, quad in selected),
            ions=MappingProxyType({i: self.ions[i] for i in active}),
            pairs=MappingProxyType({p: m for p, m in self.pairs.items() if set(p) <= active}))


def compile_topology(database: ChemSageDatabase, phase_name: str) -> MQMQATopology:
    source = database.phase_records[phase_name]
    raw = source.mqmqa
    if raw is None or source.phase_type not in {"SUBG", "SUBQ"}:
        raise ValueError("MQMQA requires original SUBG/SUBQ declarations")
    lattice = database.phases[phase_name].constituents
    nc, na = len(lattice[0]), len(lattice[1])
    cations, anions = tuple(range(nc)), tuple(range(nc, nc+na))
    ions = {i: database.species[name] for i, name in enumerate((*lattice[0], *lattice[1]))}
    charges = {i: abs(s.charge) for i, s in ions.items()}
    groups = dict(enumerate((*raw.cation_groups, *raw.anion_groups)))
    quads = tuple((*c, *a) for c, a in product(combinations_with_replacement(cations, 2),
                                              combinations_with_replacement(anions, 2)))
    pairs = {(c-1, nc+a-1): member for (c, a), member in zip(
        raw.pair_indices, source.endmembers, strict=True)}
    for (c, a), member in pairs.items():
        stoich = member.pair_stoichiometry
        if (len(stoich) != 5 or stoich[0] <= 0 or stoich[1] <= 0 or any(stoich[2:])
                or member.zeta is None or member.zeta <= 0):
            raise ValueError("MQMQA needs positive pair stoichiometry/zeta and no extra pair terms")
        derived: dict[str, float] = {}
        for i, count in ((c, stoich[0]), (a, stoich[1])):
            for e, number in ions[i].constituents:
                if e not in {"VA", "/-"}:
                    derived[e] = derived.get(e, 0.)+number*count
        if any(not isclose(derived.get(e, 0), n, abs_tol=1e-9, rel_tol=1e-8)
               for e, n in zip(database.declarations.header.components, member.stoichiometry,
                               strict=True) if e in database.elements):
            raise ValueError("MQMQA pair source atoms disagree with declared ion stoichiometry")
        # A vacancy's formula coefficient is a source energy basis, not a
        # real ionic stock. Actual ion counts/charge follow quadruplet Z.
        real_pair = not any(e == "VA" for i in (c, a) for e, _ in ions[i].constituents)
        if (real_pair and not isclose(stoich[0]*charges[c], stoich[1]*charges[a],
                                     abs_tol=1e-9, rel_tol=1e-8)):
            raise ValueError("MQMQA pair stoichiometry must be charge neutral")
    explicit: dict[tuple[int, ...], tuple[float, ...]] = {}
    for indices, values in raw.coordinations:
        indices = tuple(i-1 for i in indices)
        if (any(i not in cations for i in indices[:2])
                or any(i not in anions for i in indices[2:])):
            raise ValueError("MQMQA coordination indices must select cations then anions")
        ordered = (*sorted(zip(indices[:2], values[:2], strict=True)),
                   *sorted(zip(indices[2:], values[2:], strict=True)))
        quad, zs = tuple(i for i, _ in ordered), tuple(v for _, v in ordered)
        if quad in explicit or any(not isfinite(v) or v <= 0 for v in zs):
            raise ValueError("MQMQA coordination declarations must be unique and positive")
        if any(zs[j] != zs[quad.index(i)] for j, i in enumerate(quad)):
            raise ValueError("MQMQA repeated ions require equal coordination numbers")
        explicit[quad] = zs

    @cache
    def coordination(i: int, quad: tuple[int, ...]) -> float:
        a, b, x, y = canonical_quad(quad)
        quad = (a, b, x, y)
        if quad in explicit:
            return explicit[quad][quad.index(i)]
        if a == b and x == y:
            raise ValueError(f"MQMQA pure pair {quad} needs explicit coordination")
        if a != b and x != y:
            f = (charges[a]/coordination(a, (a, a, x, y))
                 +charges[b]/coordination(b, (b, b, x, y))
                 +charges[x]/coordination(x, (a, b, x, x))
                 +charges[y]/coordination(y, (a, b, y, y)))/8
            partners = (x, y) if i in cations else (a, b)
            inverse = 0.
            for j in partners:
                adjacent = (a, b, j, j) if i in cations else (j, j, x, y)
                inverse += coordination(j, adjacent)/(charges[j]*coordination(i, adjacent))
            return 1/(f*inverse)
        if a != b:
            if i in cations:
                return coordination(i, (i, i, x, x))
            return 2*charges[i]/(charges[a]/coordination(a, (a, a, i, i))
                                 +charges[b]/coordination(b, (b, b, i, i)))
        if i in anions:
            return coordination(i, (a, a, i, i))
        return 2*charges[i]/(charges[x]/coordination(x, (i, i, x, x))
                             +charges[y]/coordination(y, (i, i, y, y)))

    coordinations: dict[tuple[int, ...], tuple[float, ...]] = {
        quad: tuple(coordination(i, quad) for i in quad) for quad in quads}
    for quad, values in coordinations.items():
        if any(not isfinite(v) or v <= 0 for v in values):
            raise ValueError("MQMQA derived coordination must be positive and finite")
        charge = sum((1 if i in cations else -1)*charges[i]/z
                     for i, z in zip(quad, values, strict=True))
        if abs(charge) > 1e-8*max(1., sum(charges[i]/z for i, z in zip(quad, values, strict=True))):
            raise ValueError("MQMQA quadruplet coordination violates charge neutrality")
    return MQMQATopology(cations, anions, tuple("Q"+"_".join(str(i+1) for i in q) for q in quads),
        quads, MappingProxyType(charges), MappingProxyType(groups), MappingProxyType(ions),
        MappingProxyType(coordinations), MappingProxyType(pairs))
