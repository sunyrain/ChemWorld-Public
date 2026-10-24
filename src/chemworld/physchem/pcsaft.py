"""PC-SAFT and simplified PC-SAFT with explicit site association, in SI units.

The hard-sphere/chain and dispersion free energy is evaluated once for all
responses. Molecular parameters and symmetric interactions are explicit inputs.
Association uses a shared implicit site balance. Declared polar parameters
select DD/QQ/DQ contributions and an explicit Gross/Vrabec convention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, pi, sqrt
from typing import Any

from chemworld.physchem.caloric_math import (
    compensated_sum,
    exponential,
    logarithm,
    scalar_value,
)
from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.pcsaft_coefficients import PCSAFT_DISPERSION_COEFFICIENTS
from chemworld.physchem.saft_association import SAFTAssociation, association_helmholtz
from chemworld.physchem.saft_polar import SAFTPolar, polar_helmholtz_contributions

AVOGADRO_PER_MOL = 6.02214076e23


@dataclass(frozen=True)
class SAFTSegment:
    component_id: str
    segment_count: float
    diameter_m: float
    well_depth_K: float
    source: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.source:
            raise ValueError("SAFT parameters require component identity and source")
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (
            self.segment_count, self.diameter_m, self.well_depth_K
        )):
            raise ValueError("SAFT segment parameters must be finite and positive")


def hard_sphere_chain(
    number_density: Any, xs: Sequence[Any], segments: Sequence[float], diameters: Sequence[Any],
    *, simplified: bool = False,
) -> tuple[Any, Any]:
    """BMCSL spheres and tangent-chain connectivity; lengths match density units.

    Cancelling density in moment ratios preserves the zero-density derivatives.
    No division by density or packing fraction remains.
    """
    moments = [compensated_sum(x*m*d**k for x, m, d in zip(
        xs, segments, diameters, strict=True)) for k in range(4)]
    zeta = [pi/6*number_density*q for q in moments]
    eta = zeta[3]
    if not 0 <= scalar_value(eta) < 1:
        raise ValueError("SAFT hard-sphere packing fraction must lie in [0, 1)")
    free = 1-eta
    q0, q1, q2, q3 = moments
    if simplified:
        return q0*(4*eta-3*eta**2)/free**2-(q0-1)*logarithm((1-eta/2)/free**3), eta
    spheres = (3*q1/q0*zeta[2]/free + q2*q2/(q3*q0)*zeta[2]/free**2
               + (q2**3/(q3*q3*q0)-1)*logarithm(free))
    connectivity = compensated_sum(
        x*(m-1)*logarithm(hard_sphere_contact(zeta[2], eta, d, d))
        for x, m, d in zip(xs, segments, diameters, strict=True)
    )
    return q0*spheres-connectivity, eta


def hard_sphere_contact(zeta2: Any, eta: Any, diameter_i: Any, diameter_j: Any) -> Any:
    """BMCSL contact shared by chain connectivity and association."""
    reduced = diameter_i*diameter_j/(diameter_i+diameter_j)
    free = 1-eta
    return 1/free+3*reduced*zeta2/free**2+2*reduced**2*zeta2**2/free**3


@dataclass(frozen=True)
class PCSAFTModel:
    components: tuple[SAFTSegment, ...]
    interaction: tuple[tuple[float, ...], ...]
    temperature_range_K: tuple[float, float]
    source: str
    coefficient_set: str = "Gross-Sadowski-2001"
    gas_constant_J_mol_K: float = R_J_PER_MOL_K
    association: SAFTAssociation | None = None
    simplified: bool = False
    polar: SAFTPolar | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "interaction", tuple(tuple(r) for r in self.interaction))
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        n = len(self.components)
        if not n or len(set(self.component_ids)) != n or not self.source:
            raise ValueError("PC-SAFT needs unique component identities and model provenance")
        if type(self.simplified) is not bool:
            raise ValueError("simplified PC-SAFT selection must be boolean")
        if self.polar and tuple(p.component_id for p in self.polar.moments) != self.component_ids:
            raise ValueError("polar parameters must match PC-SAFT component ordering")
        if self.association and any(
            site.component_id not in self.component_ids for site in self.association.sites
        ):
            raise ValueError("association site does not belong to a PC-SAFT component")
        if len(self.interaction) != n or any(len(r) != n for r in self.interaction):
            raise ValueError("explicit square PC-SAFT interaction matrix required")
        for i, row in enumerate(self.interaction):
            for j, value in enumerate(row):
                if (isinstance(value, bool) or not isfinite(value) or value >= 1
                        or value != self.interaction[j][i] or (i == j and value != 0)):
                    raise ValueError(
                        "interaction matrix needs finite symmetric k<1 and zero diagonal")
        if self.coefficient_set not in PCSAFT_DISPERSION_COEFFICIENTS:
            raise ValueError("unknown PC-SAFT dispersion coefficient set")
        if (len(self.temperature_range_K) != 2 or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (*self.temperature_range_K, self.gas_constant_J_mol_K)
        ) or self.temperature_range_K[0] >= self.temperature_range_K[1]):
            raise ValueError("PC-SAFT requires an ordered positive temperature domain and R")

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(c.component_id for c in self.components)

    def validate(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> None:
        t, r = scalar_value(temperature), scalar_value(density)
        if (isinstance(t, bool) or not isfinite(t)
                or not self.temperature_range_K[0] <= t <= self.temperature_range_K[1]):
            raise ValueError("temperature outside declared PC-SAFT validity")
        if isinstance(r, bool) or not isfinite(r) or r < 0:
            raise ValueError("molar density must be finite and nonnegative")
        if len(fractions) != len(self.components):
            raise ValueError("PC-SAFT composition does not match declared components")
        values = tuple(scalar_value(x) for x in fractions)
        if (any(isinstance(x, bool) or not isfinite(x) or x < 0 for x in values)
                or abs(sum(values)-1) > 1e-10):
            raise ValueError("PC-SAFT mole fractions must be nonnegative and normalized")

    def contributions(self, temperature: Any, density: Any,
                      fractions: Sequence[Any]) -> dict[str, Any]:
        self.validate(temperature, density, fractions)
        x = tuple(fractions)
        m = tuple(c.segment_count for c in self.components)
        # Condition the geometry in Angstroms; public inputs and outputs stay SI.
        sigma = tuple(c.diameter_m*1e10 for c in self.components)
        epsilon = tuple(c.well_depth_K for c in self.components)
        d = tuple(s*(1-0.12*exponential(-3*e/temperature))
                  for s, e in zip(sigma, epsilon, strict=True))
        rho = density*AVOGADRO_PER_MOL*1e-30
        chain, eta = hard_sphere_chain(rho, x, m, d, simplified=self.simplified)
        mean_m = compensated_sum(xi*mi for xi, mi in zip(x, m, strict=True))
        ratio1, ratio2 = (mean_m-1)/mean_m, (mean_m-2)/mean_m
        coefficients = PCSAFT_DISPERSION_COEFFICIENTS[self.coefficient_set]
        integrals = []
        for offset in (0, 3):
            polynomial = 0
            for k in reversed(range(7)):
                polynomial = polynomial*eta + coefficients[offset][k] + ratio1*(
                    coefficients[offset+1][k]+ratio2*coefficients[offset+2][k])
            integrals.append(polynomial)
        pair_sums = []
        for power in (1, 2):
            pair_sums.append(compensated_sum(
                x[i]*x[j]*m[i]*m[j]*((sigma[i]+sigma[j])/2)**3
                *(sqrt(epsilon[i]*epsilon[j])*(1-self.interaction[i][j])/temperature)**power
                for i in range(len(x)) for j in range(len(x))
            ))
        correction = 1/(1+mean_m*(8*eta-2*eta**2)/(1-eta)**4
                        +(1-mean_m)*(20*eta-27*eta**2+12*eta**3-2*eta**4)
                        /((1-eta)*(2-eta))**2)
        dispersion = -pi*rho*(2*integrals[0]*pair_sums[0]
                              + mean_m*correction*integrals[1]*pair_sums[1])
        result = {"hard_chain": chain, "dispersion": dispersion}
        if self.polar:
            result.update(polar_helmholtz_contributions(
                self.polar, temperature, rho, eta, self.component_ids, x, m, sigma, epsilon,
                self.interaction))
        if self.association:
            zeta2 = pi*rho/6*compensated_sum(xi*mi*di**2 for xi, mi, di in zip(
                x, m, d, strict=True))
            free = 1-eta
            contacts = [[(1-eta/2)/free**3 if self.simplified
                         else hard_sphere_contact(zeta2, eta, di, dj)
                         for dj in d] for di in d]
            result["association"] = association_helmholtz(
                self.association, temperature, density*AVOGADRO_PER_MOL,
                self.component_ids, x, contacts)
        return result

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        return compensated_sum(self.contributions(temperature, density, fractions).values())

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld-pcsaft-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PCSAFTModel:
        data = dict(payload)
        if data.pop("model", None) != "chemworld-pcsaft-v1":
            raise ValueError("unknown PC-SAFT model encoding")
        data["components"] = tuple(SAFTSegment(**c) for c in data["components"])
        if data.get("association") is not None:
            data["association"] = SAFTAssociation.from_dict(data["association"])
        if data.get("polar") is not None:
            data["polar"] = SAFTPolar.from_dict(data["polar"])
        return cls(**data)
