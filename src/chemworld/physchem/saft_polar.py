"""Explicit dipole, quadrupole and cross perturbations on shared SAFT arithmetic.

The default Gross/Vrabec convention uses full precision tables and separately
capped segment counts. The optional teqp-0.23.2 convention records its different
rounding, mixed-chain cap, quadrupole third-order table and prefactors; it is
version compatibility, not a silent correction of the published model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product
from math import isfinite, pi, prod, sqrt
from typing import Any

from chemworld.physchem.caloric_math import compensated_sum, derivative_depth, scalar_value
from chemworld.physchem.saft_polar_coefficients import GROSS_VRABEC_POLAR, TEQP_0232_POLAR


@dataclass(frozen=True)
class SAFTMultipole:
    component_id: str
    reduced_dipole_squared: float
    reduced_quadrupole_squared: float
    source: str
    dipole_count: float = 1
    quadrupole_count: float = 1

    def __post_init__(self) -> None:
        if not self.component_id or not self.source or any(
            isinstance(v, bool) or not isfinite(v) or v < 0 for v in (
                self.reduced_dipole_squared, self.reduced_quadrupole_squared,
                self.dipole_count, self.quadrupole_count)
        ):
            raise ValueError("multipoles require identity, provenance and nonnegative finite data")


@dataclass(frozen=True)
class SAFTPolar:
    moments: tuple[SAFTMultipole, ...]
    source: str
    convention: str = "Gross-Vrabec"
    contributions: tuple[str, ...] = ("DD", "QQ", "DQ")

    def __post_init__(self) -> None:
        object.__setattr__(self, "moments", tuple(self.moments))
        object.__setattr__(self, "contributions", tuple(self.contributions))
        if (not self.moments or not self.source
                or len({m.component_id for m in self.moments}) != len(self.moments)):
            raise ValueError("polar parameters require unique components and provenance")
        if self.convention not in ("Gross-Vrabec", "teqp-0.23.2"):
            raise ValueError("unknown polar convention")
        if (not self.contributions or len(set(self.contributions)) != len(self.contributions)
                or not set(self.contributions) <= {"DD", "QQ", "DQ"}):
            raise ValueError("polar contributions must explicitly select unique DD, QQ or DQ terms")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SAFTPolar:
        values = dict(data)
        values["moments"] = tuple(SAFTMultipole(**m) for m in data["moments"])
        return cls(**values)


def _polynomial(table: Sequence[Sequence[float]], m: float, eta: Any) -> Any:
    weights = (1, (m-1)/m, (m-1)*(m-2)/m**2)
    result = 0
    for power in reversed(range(len(table[0]))):
        result = result*eta+sum(weight*row[power] for weight, row in zip(
            weights, table, strict=False))
    return result


def polar_helmholtz_contributions(
    spec: SAFTPolar, temperature_K: Any, number_density_A3: Any, eta: Any,
    component_ids: Sequence[str], mole_fractions: Sequence[Any], segment_counts: Sequence[float],
    diameters_A: Sequence[float], well_depths_K: Sequence[float],
    interaction: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Dimensionless residual terms; lengths and number density are in Angstroms.

    Factor density before the Pade quotient to preserve vacuum derivatives.
    Singleton polar populations are also factored before division, preserving
    their exact absent-component insertion limit. Mixed derivatives at a face
    with several absent cross-polar species may not be uniquely analytic.
    """
    if tuple(m.component_id for m in spec.moments) != tuple(component_ids):
        raise ValueError("polar moment ordering must match EOS component ordering")
    x, m, s, e, rho, t = (mole_fractions, segment_counts, diameters_A, well_depths_K,
                          number_density_A3, temperature_K)
    dipoles = tuple(p.reduced_dipole_squared*p.dipole_count for p in spec.moments)
    quads = tuple(p.reduced_quadrupole_squared*p.quadrupole_count for p in spec.moments)
    groups = {"D": tuple(i for i, a in enumerate(dipoles) if a),
              "Q": tuple(i for i, a in enumerate(quads) if a)}
    moments = {"D": dipoles, "Q": quads}
    compat = spec.convention == "teqp-0.23.2"
    tables = TEQP_0232_POLAR if compat else GROSS_VRABEC_POLAR
    result = {}
    for kind in spec.contributions:
        first, second = kind[0], kind[1]
        if not groups[first] or not groups[second]:
            result[kind] = 0
            continue
        factors = tuple(groups[g][0] for g in kind if len(groups[g]) == 1)
        common = prod(x[i] for i in factors)

        def composition(indices: Sequence[int], factors: tuple[int, ...] = factors) -> Any:
            remaining = list(indices)
            for index in factors:
                remaining.remove(index)
            return prod(x[i] for i in remaining)

        def mean_chain(indices: Sequence[int]) -> float:
            if compat:
                return min(prod(m[i] for i in indices)**(1/len(indices)), 2)
            return prod(min(m[i], 2) for i in indices)**(1/len(indices))

        pairs = []
        for i, j in product(groups[first], groups[second]):
            sigma = (s[i]+s[j])/2
            geometry = (s[i]*s[j]/sigma)**3 if kind == "DD" else (
                (s[i]*s[j])**5/sigma**7 if kind == "QQ" else s[i]**3*s[j]**5/sigma**5)
            eps_pair = sqrt(e[i]*e[j])*(1 if compat else 1-interaction[i][j])
            chain = mean_chain((i, j))
            integral = (_polynomial(tables[kind][0], chain, eta)
                        +eps_pair/t*_polynomial(tables[kind][1], chain, eta))
            pairs.append(composition((i, j))*moments[first][i]*moments[second][j]
                         *e[i]*e[j]/t**2*geometry*integral)
        a2 = -pi*{"DD": 1, "QQ": .75**2, "DQ": 2.25}[kind]*compensated_sum(pairs)
        triplets = []
        combinations = ("DDD",) if kind == "DD" else ("QQQ",) if kind == "QQ" else ("DDQ", "DQQ")
        for term in combinations:
            for i, j, k in product(*(groups[g] for g in term)):
                cross = (s[i]+s[j])*(s[i]+s[k])*(s[j]+s[k])/8
                geometry = ((s[i]*s[j]*s[k])**3/cross if kind == "DD" else
                            (s[i]*s[j]*s[k])**5/cross**3 if kind == "QQ" else
                            (s[i]*s[j]*s[k])**4/cross**2)
                # teqp 0.23.2 QQ calls its DD third-order polynomial, explicitly versioned here.
                table = tables["DD" if compat and kind == "QQ" else kind][2]
                integral = _polynomial(table, mean_chain((i, j, k)), eta)
                polar = moments[term[0]][i]*moments[term[1]][j]*moments[term[2]][k]
                triplets.append(composition((i, j, k))*polar*geometry*e[i]*e[j]*e[k]/t**3
                                *integral*((1.19374 if compat else 1.193735)
                                           if term == "DQQ" else 1))
        prefactor = (-4*pi*pi/3 if kind == "DD" else
                     -4*pi*pi/3*.75**3 if compat and kind == "QQ" else
                     (.75*pi)**2 if kind == "QQ" else -1 if compat else -pi*pi)
        a3 = prefactor*compensated_sum(triplets)
        denominator = a2-rho*a3
        if scalar_value(denominator) == 0:
            if scalar_value(a2) or scalar_value(a3):
                raise ValueError("singular polar Pade denominator")
            if (kind == "DQ" and any(any(scalar_value(x[i]) for i in groups[g]) for g in kind)
                    and any(derivative_depth(v) for v in x)):
                raise ValueError("cross-polar composition face has no unique mixed Taylor limit")
            result[kind] = common*(rho*a2+rho*rho*a3)
        else:
            result[kind] = common*rho*a2*a2/denominator
    return result
