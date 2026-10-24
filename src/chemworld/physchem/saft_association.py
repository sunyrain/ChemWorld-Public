"""Wertheim first-order site association on the common residual solver.

The site topology and pair energies/volumes are explicit data. Solve the
physical site fractions once; implicit Taylor lifting uses that root Jacobian
and existing analytic jets through order three, without finite differences or
a separate nonlinear/linear solver.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

import numpy as np

from chemworld.physchem.caloric_math import (
    compensated_sum,
    derivative_depth,
    exponential_minus_one,
    logarithm,
    scalar_value,
)
from chemworld.physchem.residual_system import (
    ResidualVariable,
    RootPolicy,
    linear_solve,
    solve_root,
)


@dataclass(frozen=True)
class AssociationSite:
    component_id: str
    site_id: str
    count: int = 1

    def __post_init__(self) -> None:
        if (not self.component_id or not self.site_id
                or type(self.count) is not int or self.count < 1):
            raise ValueError("association site needs component, label and positive integer count")


@dataclass(frozen=True)
class GeometricAssociationPair:
    """Elliott cross strength: factor*sqrt(Delta_left*Delta_right).

    Indices refer to the declared site list. References are explicit unmixed
    self-component bonds, so distinct site chemistries are never inferred.
    """

    pair: tuple[int, int]
    reference_pairs: tuple[tuple[int, int], tuple[int, int]]
    factor: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair", tuple(self.pair))
        object.__setattr__(self, "reference_pairs", tuple(tuple(p) for p in self.reference_pairs))
        if (len(self.pair) != 2 or len(self.reference_pairs) != 2
                or any(len(p) != 2 for p in self.reference_pairs)
                or any(type(i) is not int or i < 0
                       for p in (self.pair, *self.reference_pairs) for i in p)
                or isinstance(self.factor, bool) or not isfinite(self.factor) or self.factor < 0):
            raise ValueError("geometric association needs site-index pairs and nonnegative factor")


@dataclass(frozen=True)
class SAFTAssociation:
    sites: tuple[AssociationSite, ...]
    energy_K: tuple[tuple[float, ...], ...]
    bonding_volume_m3: tuple[tuple[float, ...], ...]
    source: str
    geometric_pairs: tuple[GeometricAssociationPair, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sites", tuple(self.sites))
        object.__setattr__(self, "geometric_pairs", tuple(self.geometric_pairs))
        for field in ("energy_K", "bonding_volume_m3"):
            object.__setattr__(self, field, tuple(tuple(row) for row in getattr(self, field)))
        size = len(self.sites)
        if (not size or not self.source
                or len({(s.component_id, s.site_id) for s in self.sites}) != size):
            raise ValueError("association requires unique sites and parameter provenance")
        for matrix in (self.energy_K, self.bonding_volume_m3):
            if len(matrix) != size or any(len(row) != size for row in matrix):
                raise ValueError("explicit square site interaction matrices required")
            for i in range(size):
                for j in range(size):
                    value = matrix[i][j]
                    if (isinstance(value, bool) or not isfinite(value) or value < 0
                            or value != matrix[j][i]):
                        raise ValueError(
                            "association pairs must be symmetric finite nonnegative data")
        mixed = {tuple(sorted(p.pair)) for p in self.geometric_pairs}
        if len(mixed) != len(self.geometric_pairs):
            raise ValueError("duplicate geometric association pair")
        for mixing in self.geometric_pairs:
            pairs = (mixing.pair, *mixing.reference_pairs)
            if any(i >= size for p in pairs for i in p):
                raise ValueError("geometric association references an undeclared site")
            i, j = mixing.pair
            if self.energy_K[i][j] != 0 or self.bonding_volume_m3[i][j] != 0:
                raise ValueError("geometric pair must not also declare explicit energy/volume")
            for site, (a, b) in zip((i, j), mixing.reference_pairs, strict=True):
                if (tuple(sorted((a, b))) in mixed
                        or self.sites[a].component_id != self.sites[site].component_id
                        or self.sites[b].component_id != self.sites[site].component_id
                        or self.energy_K[a][b] <= 0 or self.bonding_volume_m3[a][b] <= 0):
                    raise ValueError("geometric reference must be a positive unmixed self bond")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SAFTAssociation:
        return cls(tuple(AssociationSite(**s) for s in data["sites"]), data["energy_K"],
                   data["bonding_volume_m3"], data["source"],
                   tuple(GeometricAssociationPair(**p) for p in data.get("geometric_pairs", ())))


def association_site_fractions(
    populations: Sequence[Any], coupling: Sequence[Sequence[Any]],
) -> tuple[Any, ...]:
    """X_i (1 + sum_j population_j coupling_ij X_j) = 1, with 0<X_i<=1.

    coupling already includes number density and association strength. The
    primal Jacobian is inverted using the shared linear solver. Each lift fixes
    the next Taylor degree; no iteration count based on derivative outcomes.
    """
    size = len(populations)
    if not size or len(coupling) != size or any(len(r) != size for r in coupling):
        raise ValueError("site populations and coupling matrix dimensions must agree")
    if any(not isfinite(scalar_value(v)) or scalar_value(v) < 0
           for v in (*populations, *(v for row in coupling for v in row))):
        raise ValueError("site populations and coupling must be finite and nonnegative")
    matrix = [[populations[j]*coupling[i][j] for j in range(size)] for i in range(size)]
    primal = np.asarray([[scalar_value(a) for a in row] for row in matrix], dtype=float)
    if np.any(~np.isfinite(primal)) or np.any(primal < 0):
        raise ValueError("association coupling must be finite and nonnegative")

    def jacobian(x: np.ndarray) -> np.ndarray:
        return np.diag(1+primal@x)+x[:, None]*primal

    initial = 2/(1+np.sqrt(1+4*np.sum(primal, axis=1)))
    root = solve_root(
        lambda x: x*(1+primal@x)-1, initial, jacobian=jacobian,
        variables=tuple(ResidualVariable(f"site_{i}", "dimensionless", role="algebraic",
                                         lower=1e-15, upper=1) for i in range(size)),
        residual_scales=np.ones(size),
        policy=RootPolicy(tolerance=2e-13, maximum_iterations=80),
    )
    if not root.success:
        raise ValueError("SAFT association solve failed: "+root.failure_reason)
    fractions: list[Any] = list(root.value)
    depth = max(derivative_depth(a) for row in matrix for a in row)
    if depth:
        inverse = linear_solve(jacobian(root.value), np.eye(size))
        for _ in range(depth):
            residual = [fractions[i]*(1+compensated_sum(
                matrix[i][j]*fractions[j] for j in range(size)))-1 for i in range(size)]
            fractions = [fractions[i]-compensated_sum(
                float(inverse[i, j])*residual[j] for j in range(size)) for i in range(size)]
    return tuple(fractions)


def association_helmholtz(
    spec: SAFTAssociation, temperature_K: Any, number_density_m3: Any,
    component_ids: Sequence[str], mole_fractions: Sequence[Any],
    contact_values: Sequence[Sequence[Any]],
) -> Any:
    """Dimensionless residual association A/(nRT) for an explicit contact law."""
    n = len(component_ids)
    if (len(mole_fractions) != n or len(contact_values) != n
            or any(len(row) != n for row in contact_values)):
        raise ValueError("association contact law must cover every component pair")
    if (scalar_value(temperature_K) <= 0 or scalar_value(number_density_m3) < 0
            or not isfinite(scalar_value(temperature_K))
            or not isfinite(scalar_value(number_density_m3))):
        raise ValueError("association requires positive temperature and nonnegative density")
    indices = []
    for site in spec.sites:
        if site.component_id not in component_ids:
            raise ValueError("association site refers to undeclared component")
        indices.append(component_ids.index(site.component_id))
    populations = [mole_fractions[i]*site.count
                   for i, site in zip(indices, spec.sites, strict=True)]
    strengths = [[spec.bonding_volume_m3[i][j]
               *exponential_minus_one(spec.energy_K[i][j]/temperature_K)
               *contact_values[left][right]
               for j, right in enumerate(indices)] for i, left in enumerate(indices)]
    for mixing in spec.geometric_pairs:
        i, j = mixing.pair
        (a, b), (c, d) = mixing.reference_pairs
        strengths[i][j] = strengths[j][i] = mixing.factor*(strengths[a][b]*strengths[c][d])**.5
    # Mix Delta before multiplying by density: sqrt(rho^2) would lose the
    # analytic vacuum limit and its density derivatives.
    matrix = [[number_density_m3*v for v in row] for row in strengths]
    fractions = association_site_fractions(populations, matrix)
    return compensated_sum(population*(logarithm(fraction)-fraction/2+0.5)
                           for population, fraction in zip(populations, fractions, strict=True))
