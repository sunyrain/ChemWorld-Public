"""Lafitte SAFT-VR-Mie on common hard spheres, analytic jets and EOS responses.

All molecular inputs are SI. Geometry is conditioned internally in Angstroms.
The Barker-Henderson integral and its analytic temperature derivatives use a
fixed Gauss-Legendre rule, independent of the reference backend's quadrature.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, log, pi, sqrt, tanh
from typing import TYPE_CHECKING, Any

import numpy as np

from chemworld.physchem.caloric_math import (
    compensated_sum,
    derivative_depth,
    exponential,
    exponential_minus_one,
    logarithm,
    scalar_value,
)
from chemworld.physchem.dufal_association import DufalAssociation
from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL, SAFTSegment, hard_sphere_chain
from chemworld.physchem.saft_polar import SAFTPolar, polar_helmholtz_contributions

if TYPE_CHECKING:
    from chemworld.physchem.mie_reference_options import MieReferenceOptions
    from chemworld.physchem.saft_polar_extensions import CorrelationPolar

# Lafitte et al., J Chem Phys139,154504(2013), TablesII/III and AppendixA.
_EFFECTIVE_PACKING = ((.81096, 1.7888, -37.578, 92.284),
                      (1.0205, -19.341, 151.26, -463.50),
                      (-1.9057, 22.845, -228.14, 973.92),
                      (1.0885, -6.1962, 106.98, -677.64))
_PHI = ((7.5365557, -359.44, 1550.9, -1.19932, -1911.28, 9236.9),
        (-37.60463, 1825.6, -5070.1, 9.063632, 21390.175, -129430),
        (71.745953, -3168.0, 6534.6, -17.9482, -51320.7, 357230),
        (-46.83552, 1884.2, -3288.7, 11.34027, 37064.54, -315530),
        (-2.467982, -.82376, -2.7171, 20.52142, 1103.742, 1390.2),
        (-.50272, -3.1935, 2.0883, -56.6377, -3264.61, -4518.2),
        (8.0956883, 3.7090, 0., 40.53683, 2556.181, 4241.6))
_NODES, _RAW_WEIGHTS = np.polynomial.legendre.leggauss(128)
_RADII, _WEIGHTS = (_NODES+1)/2, _RAW_WEIGHTS/2


@dataclass(frozen=True)
class MieSegment(SAFTSegment):
    repulsive_exponent: float = 12.0
    attractive_exponent: float = 6.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if (any(isinstance(v, bool) or not isfinite(v)
                for v in (self.repulsive_exponent, self.attractive_exponent))
                or not self.repulsive_exponent > self.attractive_exponent > 3):
            raise ValueError("Mie exponents must be finite with repulsive > attractive > 3")


def mie_prefactor(repulsive: float, attractive: float) -> float:
    return repulsive/(repulsive-attractive)*(repulsive/attractive)**(
        attractive/(repulsive-attractive))


def mie_effective_diameter(component: MieSegment, temperature: Any) -> Any:
    """Barker-Henderson diameter in meters, with analytic derivatives through3.

    Taylor composition preserves the existing nested jets and SecondOrderPair.
    Saturated Boltzmann factors below floating underflow have zero derivatives.
    """
    t = scalar_value(temperature)
    if not isfinite(t) or t <= 0:
        raise ValueError("Mie diameter requires finite positive temperature")
    lr, la = component.repulsive_exponent, component.attractive_exponent
    log_u_over_t = (log(mie_prefactor(lr, la)*component.well_depth_K/t)
                   -lr*np.log(_RADII)+np.log1p(-_RADII**(lr-la)))
    active = log_u_over_t < log(745.)
    v = np.exp(log_u_over_t[active])
    weights = _WEIGHTS[active]
    ev = np.exp(-v)
    diameter = component.diameter_m*(1-float(np.dot(weights, ev)))
    depth = derivative_depth(temperature)
    if depth > 3:
        raise ValueError("Mie diameter supports analytic derivatives through order three")
    if depth:
        delta = temperature-t
        polys = (-v, 2*v-v*v, -6*v+6*v*v-v**3)
        for order in range(1, depth+1):
            derivative = component.diameter_m*float(np.dot(weights, ev*polys[order-1]))/t**order
            diameter += derivative*delta**order/(1 if order == 1 else 2 if order == 2 else 6)
    return diameter


def _perturbation_term(exponent: float, ratio: Any, packing: Any,
                       integration_ratio: Any | None = None) -> tuple[Any, Any]:
    """A10-A18: F_lambda and F_lambda+zeta*dF_lambda/dzeta.

    Density is cancelled algebraically, so the chain contact has a vacuum limit.
    An explicit integration limit permits quantum effective diameters while
    retaining the original sigma/d power prefactor. Classical default unchanged.
    """
    cs = tuple(sum(a/exponent**k for k, a in enumerate(row)) for row in _EFFECTIVE_PACKING)
    effective = sum(c*packing**(k+1) for k, c in enumerate(cs))
    slope = sum((k+1)*c*packing**k for k, c in enumerate(cs))
    limit = ratio if integration_ratio is None else integration_ratio
    integral_i = (1-limit**(3-exponent))/(exponent-3)
    integral_j = (logarithm(limit)+1/limit-1 if exponent == 4 else
        (1-(exponent-3)*limit**(4-exponent)+(exponent-4)*limit**(3-exponent))
        /((exponent-3)*(exponent-4)))
    free = 1-packing
    bhat = ((1-packing/2)*integral_i-4.5*packing*(1+packing)*integral_j)/free**3
    dbhat = ((2.5-packing)*integral_i
             -4.5*(1+4*packing+packing**2)*integral_j)/free**4
    ahat = -(1-effective/2)/((exponent-3)*(1-effective)**3)
    dahat = (effective-2.5)*slope/((exponent-3)*(1-effective)**4)
    factor = ratio**exponent
    return factor*(bhat+ahat), factor*(bhat+ahat+packing*(dbhat+dahat))


def mie_correction_coefficients(alpha: Any) -> tuple[Any, ...]:
    """Six Lafitte rational corrections, shared by classical and quantum Mie."""
    return tuple(sum(_PHI[k][q]*alpha**k for k in range(4))
                 /(1+sum(_PHI[k][q]*alpha**(k-3) for k in range(4, 7))) for q in range(6))


@dataclass(frozen=True)
class SAFTVRMieModel:
    components: tuple[MieSegment, ...]
    interaction: tuple[tuple[float, ...], ...]
    temperature_range_K: tuple[float, float]
    source: str
    energy_mixing: str = "Lafitte"
    gas_constant_J_mol_K: float = R_J_PER_MOL_K
    polar: SAFTPolar | CorrelationPolar | None = None
    association: DufalAssociation | None = None
    reference_options: MieReferenceOptions | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "interaction", tuple(tuple(r) for r in self.interaction))
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        n = len(self.components)
        if not n or len(set(self.component_ids)) != n or not self.source:
            raise ValueError("SAFT-VR-Mie needs unique components and provenance")
        if self.energy_mixing not in ("Lafitte", "Lorentz-Berthelot"):
            raise ValueError("unknown Mie energy mixing rule")
        if len(self.interaction) != n or any(len(r) != n for r in self.interaction):
            raise ValueError("explicit square Mie interaction matrix required")
        for i, row in enumerate(self.interaction):
            for j, v in enumerate(row):
                if (isinstance(v, bool) or not isfinite(v) or v >= 1
                        or v != self.interaction[j][i] or (i == j and v != 0)):
                    raise ValueError("Mie interactions need finite symmetric k<1, zero diagonal")
        if (len(self.temperature_range_K) != 2 or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (*self.temperature_range_K, self.gas_constant_J_mol_K)
        ) or self.temperature_range_K[0] >= self.temperature_range_K[1]):
            raise ValueError("Mie requires ordered positive temperature domain and R")
        if self.polar and tuple(p.component_id for p in self.polar.moments) != self.component_ids:
            raise ValueError("polar parameters must match Mie component ordering")
        if self.association and self.association.component_ids != self.component_ids:
            raise ValueError("Mie association component ordering must match")
        if self.reference_options is not None:
            self.reference_options.bind(self)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(c.component_id for c in self.components)

    def _classical_contributions(self, temperature: Any, density: Any,
                      fractions: Sequence[Any]) -> dict[str, Any]:
        t, rho = scalar_value(temperature), scalar_value(density)
        values = tuple(scalar_value(x) for x in fractions)
        if (not isfinite(t) or not self.temperature_range_K[0] <= t <= self.temperature_range_K[1]
                or not isfinite(rho) or rho < 0 or len(values) != len(self.components)
                or any(not isfinite(x) or x < 0 for x in values) or abs(sum(values)-1) > 1e-10):
            raise ValueError("Mie state outside declared temperature/density/composition domain")
        n, x = len(values), fractions
        m = tuple(c.segment_count for c in self.components)
        sigma = tuple(c.diameter_m*1e10 for c in self.components)
        epsilon = tuple(c.well_depth_K for c in self.components)
        ds = tuple(mie_effective_diameter(c, temperature)*1e10 for c in self.components)
        mbar = compensated_sum(xi*mi for xi, mi in zip(x, m, strict=True))
        xs = tuple(xi*mi/mbar for xi, mi in zip(x, m, strict=True))
        rhon = density*AVOGADRO_PER_MOL*1e-30
        rhos = rhon*mbar
        hs, _eta = hard_sphere_chain(rhos, xs, (1.,)*n, ds)
        packing = pi*rhos/6*compensated_sum(xs[i]*xs[j]*((ds[i]+ds[j])/2)**3
                                         for i in range(n) for j in range(n))
        packing_bar = pi*rhos/6*compensated_sum(xs[i]*xs[j]*((sigma[i]+sigma[j])/2)**3
                                             for i in range(n) for j in range(n))
        free = 1-packing
        kden = 1+4*packing+4*packing**2-4*packing**3+packing**4
        khs = free**4/kden
        rho_dk = -4*(packing-1)**3*(packing**2-5*packing-2)*packing/kden**2
        k0 = -logarithm(free)+(42*packing-39*packing**2+9*packing**3-2*packing**4)/(6*free**3)
        k1 = (packing**4+6*packing**2-12*packing)/(2*free**3)
        k2 = -3*packing**2/(8*free**2)
        k3 = (-packing**4+3*packing**2+3*packing)/(6*free**3)
        first, second, third, chain = [], [], [], []
        for i, left in enumerate(self.components):
            for j in range(i, n):
                right = self.components[j]
                la = 3+sqrt((left.attractive_exponent-3)*(right.attractive_exponent-3))
                lr = 3+sqrt((left.repulsive_exponent-3)*(right.repulsive_exponent-3))
                sij, dij = (sigma[i]+sigma[j])/2, (ds[i]+ds[j])/2
                eij = (1-self.interaction[i][j])*sqrt(epsilon[i]*epsilon[j])
                if self.energy_mixing == "Lafitte":
                    eij *= (sigma[i]*sigma[j])**1.5/sij**3
                prefactor = mie_prefactor(lr, la)
                alpha = prefactor*(1/(la-3)-1/(lr-3))
                fs = mie_correction_coefficients(alpha)
                ratio = sij/dij
                fa, dfa = _perturbation_term(la, ratio, packing)
                fr, dfr = _perturbation_term(lr, ratio, packing)
                faa, dfaa = _perturbation_term(2*la, ratio, packing)
                far, dfar = _perturbation_term(la+lr, ratio, packing)
                frr, dfrr = _perturbation_term(2*lr, ratio, packing)
                pair = xs[i]*xs[j]*(1 if i == j else 2)
                scale = 2*pi*rhos*dij**3
                chi = fs[0]*packing_bar+fs[1]*packing_bar**5+fs[2]*packing_bar**8
                first.append(pair*scale*eij*prefactor*(fa-fr))
                second.append(pair*scale*eij**2*prefactor**2*khs*(1+chi)*(faa-2*far+frr)/2)
                third.append(-pair*eij**3*fs[3]*packing_bar*exponential(
                    fs[4]*packing_bar+fs[5]*packing_bar**2))
                if i == j:
                    log_gd = k0+k1*ratio+k2*ratio**2+k3*ratio**3
                    g1 = prefactor*(3*(dfa-dfr)-la*fa+lr*fr)
                    g2mca = prefactor**2*(1.5*(rho_dk*(faa-2*far+frr)
                        +khs*(dfaa-2*dfar+dfrr))-khs*(la*faa-(la+lr)*far+lr*frr))
                    be = eij/temperature
                    gamma = 10*(1-tanh(10*(.57-alpha)))*packing_bar*exponential_minus_one(be)*(
                        exponential(-6.7*packing_bar-8*packing_bar**2))
                    # Stay in log-contact form: differentiating 1/exp(log_gd)
                    # can overflow intermediate inverse powers at dense states.
                    log_gmie = log_gd+(be*g1+be**2*(1+gamma)*g2mca)*exponential(-log_gd)
                    chain.append(-x[i]*(m[i]-1)*log_gmie)
        result = {"hard_sphere": mbar*hs, "dispersion1": mbar*compensated_sum(first)/temperature,
                  "dispersion2": mbar*compensated_sum(second)/temperature**2,
                  "dispersion3": mbar*compensated_sum(third)/temperature**3,
                  "chain": compensated_sum(chain)}
        return result

    def contributions(self, temperature: Any, density: Any,
                      fractions: Sequence[Any]) -> dict[str, Any]:
        if self.reference_options is None:
            result = self._classical_contributions(temperature, density, fractions)
        else:
            result = self.reference_options.bind(self).contributions(
                temperature, density, fractions)
        if self.polar is None and self.association is None:
            return result
        x = fractions
        m = tuple(c.segment_count for c in self.components)
        sigma = tuple(c.diameter_m*1e10 for c in self.components)
        epsilon = tuple(c.well_depth_K for c in self.components)
        ds = tuple(mie_effective_diameter(c, temperature)*1e10 for c in self.components)
        rhon = density*AVOGADRO_PER_MOL*1e-30
        eta = pi*rhon/6*compensated_sum(
            xi*mi*di**3 for xi, mi, di in zip(x, m, ds, strict=True))
        if isinstance(self.polar, SAFTPolar):
            result.update(polar_helmholtz_contributions(self.polar, temperature, rhon, eta,
                self.component_ids, x, m, sigma, epsilon, self.interaction))
        elif self.polar is not None:
            from chemworld.physchem.saft_polar_extensions import (
                correlation_polar_helmholtz_contributions,
            )

            result.update(correlation_polar_helmholtz_contributions(
                self.polar, temperature, rhon, eta, self.component_ids, x,
                sigma, epsilon, self.interaction, energy_mixing=self.energy_mixing))
        if self.association:
            result['association'] = self.association.alphar(temperature, density, x)
        return result

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        return compensated_sum(self.contributions(temperature, density, fractions).values())

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld-saftvrmie-v1", **asdict(self),
                "polar": self.polar.to_dict() if self.polar else None,
                "reference_options": self.reference_options.to_dict()
                if self.reference_options else None}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SAFTVRMieModel:
        data = dict(payload)
        if data.pop("model", None) != "chemworld-saftvrmie-v1":
            raise ValueError("unknown SAFT-VR-Mie model encoding")
        data["components"] = tuple(MieSegment(**c) for c in data["components"])
        if data.get("polar") is not None:
            if data["polar"].get("kind") == "correlation-polar":
                from chemworld.physchem.saft_polar_extensions import CorrelationPolar

                data["polar"] = CorrelationPolar.from_dict(data["polar"])
            else:
                data["polar"] = SAFTPolar.from_dict(data["polar"])
        if data.get("reference_options") is not None:
            from chemworld.physchem.mie_reference_options import MieReferenceOptions

            data["reference_options"] = MieReferenceOptions.from_dict(data["reference_options"])
        if data.get("association") is not None:
            data["association"] = DufalAssociation.from_dict(data["association"])
        return cls(**data)
