"""Kohler/Toop constitution weights on the common CEF Gibbs parameter sum."""

from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING, cast

from chemworld.data.tdb import TdbPhase
from chemworld.physchem.gibbs_phase import JetSiteFractions
from chemworld.physchem.tdb_expressions import TdbScalar, scalar_value

if TYPE_CHECKING:
    from chemworld.physchem.calphad import _Parameter


def validate_qkt(parameter: _Parameter, phase: TdbPhase) -> None:
    mixing = [subl for subl in parameter.constituents if len(subl) > 1]
    if (len(mixing) != 1 or len(mixing[0]) not in {2, 3}
            or len(parameter.source.exponents) != len(mixing[0])):
        raise ValueError("QKTO requires one binary/ternary mixing array with matching exponents")
    groups = dict(cast(tuple[tuple[str, int], ...], phase.model_hints.get("chemical_groups", ())))
    if any(s not in groups for subl in phase.constituents for s in subl):
        raise ValueError("QKTO needs declared chemical groups for its active constituents")


def qkt_weight(parameter: _Parameter, fractions: JetSiteFractions, phase: TdbPhase) -> TdbScalar:
    mixing = next(i for i, subl in enumerate(parameter.constituents) if len(subl) > 1)
    array = parameter.constituents[mixing]
    exponents = parameter.source.exponents
    subl = fractions[mixing]
    groups = dict(cast(tuple[tuple[str, int], ...], phase.model_hints["chemical_groups"]))
    prefactor = prod(fractions[i][s] for i, species in enumerate(parameter.constituents)
                     for s in species)
    if len(array) == 2:
        i, j = array
        xi = subl[i]+sum(y for s, y in subl.items() if s not in {i, j}
                         and groups[s] == groups[i] and groups[s] != groups[j])
        xj = subl[j]+sum(y for s, y in subl.items() if s not in {i, j}
                         and groups[s] == groups[j] and groups[s] != groups[i])
        sigma = 1-sum(y for s, y in subl.items() if s not in {i, j}
                      and groups[s] == groups[i] == groups[j])
        if scalar_value(sigma) == 0:
            if scalar_value(prefactor) == 0 and min(exponents) >= 0:
                return 0.
            raise ValueError("QKTO binary extrapolation has no finite declared denominator")
        first = (1+(xi-xj)/sigma)/2
        second = (1+(xj-xi)/sigma)/2
        return prefactor*first**exponents[0]*second**exponents[1]
    total = sum(subl[s] for s in array)
    if scalar_value(total) == 0:
        if scalar_value(prefactor) == 0 and min(exponents) >= 0:
            return 0.
        raise ValueError("QKTO ternary extrapolation has no finite declared denominator")
    return prefactor*prod(subl[s]**p for s, p in zip(array, exponents, strict=True))/(
        total**sum(exponents))
