"""Pure model potentials and soft-SAFT on the common residual EOS protocol.

Inputs are explicit SI molecular scales. Reduced potentials are distinct
scientific models, not interchangeable fitted material property packages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import factorial, isfinite, pi, sqrt
from typing import Any

from chemworld.physchem.caloric_math import (
    compensated_sum,
    exponential,
    exponential_minus_one,
    logarithm,
    scalar_value,
)
from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_potential_coefficients import (
    EXP6_COEFFICIENTS,
    JOHNSON_COEFFICIENTS,
    KOLAFA_DIAMETER,
    KOLAFA_RESIDUAL,
    KOLAFA_VIRIAL,
    LJ_CONTACT_COEFFICIENTS,
    SW_GAMMA,
    SW_PHI,
    SW_THETA,
    THOL_TERMS,
)
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL, SAFTSegment, hard_sphere_chain

POTENTIAL_FAMILIES = ('LJ-Johnson-1993', 'LJ-Kolafa-1994', 'LJ-Thol-2016',
                      'PeTS-2018', 'exp6-Kataoka-1992', 'square-well-2009')
_THOL = tuple(HelmholtzTerm.from_dict(t) for t in THOL_TERMS)
_PETS_A = (.690603404, 1.189317012, 1.265604153, -24.34554201,
           93.67300357, -157.8773415, 96.93736697)
_PETS_B = (.664852128, 2.10733079, -9.597951213, -17.37871193,
           30.17506222, 209.3942909, -353.2743581)


def _johnson_integrals(density: Any) -> tuple[Any, ...]:
    """Integral0^rho exp(-3s²)s^(2i-1)ds, i1..6; stable at vacuum.

    Near zero the source recurrence subtracts nearly equal quantities. Its
    integrated exponential series preserves value and analytic derivatives.
    Sixty-four fixed terms at rho<=1 leave truncation far below double error.
    """
    if scalar_value(density) <= 1:
        z = -3*density*density
        return tuple(density**(2*i)*compensated_sum(
            z**k/(factorial(k)*(2*i+2*k)) for k in range(64)) for i in range(1, 7))
    decay = exponential(-3*density*density)
    terms = [-exponential_minus_one(-3*density*density)/6]
    for i in range(2, 7):
        terms.append((2*(i-1)*terms[-1]-decay*density**(2*i-2))/6)
    return tuple(terms)


def _johnson(temperature: Any, density: Any) -> Any:
    x, t, r = JOHNSON_COEFFICIENTS, temperature, density
    a = (x[1]*t+x[2]*t**.5+x[3]+x[4]/t+x[5]/t**2,
         x[6]*t+x[7]+x[8]/t+x[9]/t**2, x[10]*t+x[11]+x[12]/t, x[13],
         x[14]/t+x[15]/t**2, x[16]/t, x[17]/t+x[18]/t**2, x[19]/t**2)
    b = (x[20]/t**2+x[21]/t**3, x[22]/t**2+x[23]/t**4,
         x[24]/t**2+x[25]/t**3, x[26]/t**2+x[27]/t**4,
         x[28]/t**2+x[29]/t**3, x[30]/t**2+x[31]/t**3+x[32]/t**4)
    return (compensated_sum(v*r**i/i for i, v in enumerate(a, 1))
            +compensated_sum(v*g for v, g in zip(b, _johnson_integrals(r), strict=True)))/t


def _kolafa(t: Any, r: Any) -> Any:
    diameter = -.063920968*logarithm(t)+compensated_sum(
        c*t**(i/2) for i, c in KOLAFA_DIAMETER)
    eta = pi/6*r*diameter**3
    if scalar_value(diameter) <= 0 or not 0 <= scalar_value(eta) < 1:
        raise ValueError('Kolafa diameter/packing outside positive physical domain')
    spheres = 5/3*logarithm(1-eta)+eta*(34-33*eta+4*eta**2)/(6*(1-eta)**2)
    virial = compensated_sum(c*t**(i/2) for i, c in KOLAFA_VIRIAL)
    return spheres+r*exponential(-1.92907278*r*r)*virial+compensated_sum(
        c*t**(i/2-1)*r**int(j) for i, j, c in KOLAFA_RESIDUAL)


def _pets(t: Any, r: Any) -> Any:
    # Heier2018 DOI10.1080/00268976.2018.1447153, truncated/shifted2.5sigma.
    diameter = 1-.127112544*exponential(-3.052785558/t)
    spheres, eta = hard_sphere_chain(r, (1,), (1,), (diameter,))
    i1 = compensated_sum(a*eta**k for k, a in enumerate(_PETS_A))
    i2 = compensated_sum(b*eta**k for k, b in enumerate(_PETS_B))
    compressibility = 1/(1+(8*eta-2*eta**2)/(1-eta)**4)
    return spheres-pi*r*(2*i1/t+compressibility*i2/t**2)


@lru_cache(maxsize=128)
def _square_well_parameters(width: float) -> tuple[Any, ...]:
    w = width
    def polynomials(v: float) -> tuple[float, ...]:
        return (v**6-18*v**4+32*v**3-15,
                -2*v**6+36*v**4-32*v**3-18*v**2+16,
                6*v**6-18*v**4+18*v**2-6,
                32*v**3-18*v**2-48, 5*v**6-32*v**3+18*v**2+26)
    p1, p2, p3, p4, p5 = polynomials(w)
    second = tuple(-2*pi/(3*factorial(i))*(w**3-1) for i in range(1, 5))
    # Rational coefficients, including lambda>2: integer truncation in the
    # pinned C++ source is not part of the published free-energy expansion.
    third = (tuple((pi/6)**2*v for v in (-p1, p2-p1/2, p2-p1/6-p3,
                                        -p1/24+7*p2/12-3*p3/2)) if w <= 2 else
             tuple((pi/6)**2*v for v in (-17, -17/2+p4, -17/6+p4-p5,
                                        -17/24+7*p4/12-3*p5/2)))
    phi = tuple(sum(c*w**k for k, c in enumerate(row)) for row in SW_PHI)
    theta = tuple((sum(row[k]*w**k for k in range(1, 5)),
                   sum(row[k]*w**(k-4) for k in range(5, 8))) for row in SW_THETA)
    z = w**3-1
    gamma = tuple(row[1]*w+row[2]*w*w
        +(row[3]+sum(row[k]*z**(k-2) for k in range(4, 9)))
        /(row[9]+sum(row[k]*z**(k-7) for k in range(10, 15))) for row in SW_GAMMA)
    return second, third, phi, theta, gamma


def _square_well(t: Any, r: Any, width: float) -> Any:
    second, third, phi, theta, gamma = _square_well_parameters(width)
    spheres, _ = hard_sphere_chain(r, (1,), (1,), (1,))
    first = second[0]*r+third[0]*r*r+compensated_sum(
        v*r**k for k, v in enumerate(gamma, 3))
    quadratic = second[1]*r*(1-r*r/1.5129)*exponential(
        third[1]/second[1]*r+phi[0]*r**3+phi[1]*r**4)
    higher = [second[i]*r*exponential(third[i]/second[i]*r
              +theta[i-2][0]*r*r/(1+theta[i-2][1]*r)) for i in (2, 3)]
    return spheres+first/t+quadratic/t**2+higher[0]/t**3+higher[1]/t**4


def _validate_declaration(components: Sequence[SAFTSegment], temperature_range: Sequence[float],
                          source: str, gas_constant: float) -> None:
    if (not components or len({c.component_id for c in components}) != len(components)
            or not source):
        raise ValueError('potential EOS needs unique components and provenance')
    if (len(temperature_range) != 2 or any(isinstance(v, bool) or not isfinite(v) or v <= 0
        for v in (*temperature_range, gas_constant))
            or temperature_range[0] >= temperature_range[1]):
        raise ValueError('potential EOS needs positive ordered T domain and R')


def _validate_state(temperature_range: Sequence[float], size: int, t: Any, rho: Any,
                    x: Sequence[Any]) -> None:
    tv, rv = scalar_value(t), scalar_value(rho)
    if (not isfinite(tv) or not temperature_range[0] <= tv <= temperature_range[1]
            or not isfinite(rv) or rv < 0):
        raise ValueError('potential EOS state outside declared T/rho domain')
    xs = tuple(scalar_value(v) for v in x)
    if len(xs) != size or any(not isfinite(v) or v < 0 for v in xs) or abs(sum(xs)-1) > 1e-10:
        raise ValueError('potential EOS needs normalized composition matching components')


@dataclass(frozen=True)
class PurePotentialModel:
    """Pure potential with SI scales; shape is alpha(exp6) or lambda(square-well).

    Declared T range is caller supplied; mathematical guards do not establish
    a material fit or global stability. No undeclared mixture rule is inferred.
    """

    component: SAFTSegment
    family: str
    temperature_range_K: tuple[float, float]
    source: str
    shape_parameter: float | None = None
    gas_constant_J_mol_K: float = R_J_PER_MOL_K

    def __post_init__(self) -> None:
        object.__setattr__(self, 'temperature_range_K', tuple(self.temperature_range_K))
        _validate_declaration((self.component,), self.temperature_range_K, self.source,
                              self.gas_constant_J_mol_K)
        if self.family not in POTENTIAL_FAMILIES or self.component.segment_count != 1:
            raise ValueError('pure potential requires a supported family and monomer component')
        minimum = {'exp6-Kataoka-1992': 8, 'square-well-2009': 1}.get(self.family)
        if minimum is None:
            if self.shape_parameter is not None:
                raise ValueError('shape parameter is not used by this potential')
        elif (self.shape_parameter is None or isinstance(self.shape_parameter, bool)
              or not isfinite(self.shape_parameter) or self.shape_parameter <= minimum):
            raise ValueError('potential shape is outside its real-valued domain')

    @property
    def component_ids(self) -> tuple[str, ...]:
        return (self.component.component_id,)

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        _validate_state(self.temperature_range_K, 1, temperature, density, fractions)
        t = temperature/self.component.well_depth_K
        r = density*AVOGADRO_PER_MOL*self.component.diameter_m**3
        if self.family == 'LJ-Johnson-1993':
            return _johnson(t, r)
        if self.family == 'LJ-Kolafa-1994':
            return _kolafa(t, r)
        if self.family == 'LJ-Thol-2016':
            return compensated_sum(term.evaluate(r/.31, 1.32/t, 1.32) for term in _THOL)
        if self.family == 'PeTS-2018':
            return _pets(t, r)
        if self.shape_parameter is None:
            raise AssertionError("invariant violated: self.shape_parameter is None")
        if self.family == 'exp6-Kataoka-1992':
            shape = (self.shape_parameter-8)/10
            return compensated_sum(c*r**int(p)*t**(-q)*shape**v
                                   for p, q, v, c in EXP6_COEFFICIENTS)
        return _square_well(t, r, self.shape_parameter)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PurePotentialModel:
        data = dict(payload)
        data['component'] = SAFTSegment(**data['component'])
        return cls(**data)


@dataclass(frozen=True)
class SoftSAFTModel:
    """Johnson LJ monomers, vdW one-fluid mixing and common LJ-chain contact."""

    components: tuple[SAFTSegment, ...]
    temperature_range_K: tuple[float, float]
    source: str
    gas_constant_J_mol_K: float = R_J_PER_MOL_K

    def __post_init__(self) -> None:
        object.__setattr__(self, 'components', tuple(self.components))
        object.__setattr__(self, 'temperature_range_K', tuple(self.temperature_range_K))
        _validate_declaration(self.components, self.temperature_range_K, self.source,
                              self.gas_constant_J_mol_K)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(c.component_id for c in self.components)

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        _validate_state(self.temperature_range_K, len(self.components), temperature,
                        density, fractions)
        mbar = compensated_sum(x*c.segment_count for x, c in zip(
            fractions, self.components, strict=True))
        weights = tuple(x*c.segment_count/mbar for x, c in zip(
            fractions, self.components, strict=True))
        volume_terms, energy_terms = [], []
        for i, a in enumerate(self.components):
            for j, b in enumerate(self.components):
                term = weights[i]*weights[j]*((a.diameter_m+b.diameter_m)/2)**3
                volume_terms.append(term)
                energy_terms.append(term*sqrt(a.well_depth_K*b.well_depth_K))
        volume = compensated_sum(volume_terms)
        t = temperature*volume/compensated_sum(energy_terms)
        r = density*AVOGADRO_PER_MOL*mbar*volume
        contact = 1+compensated_sum(c*r**i*t**(-j) for i, row in
            enumerate(LJ_CONTACT_COEFFICIENTS, 1) for j, c in enumerate(row))
        if scalar_value(contact) <= 0 or not isfinite(scalar_value(contact)):
            raise ValueError('soft-SAFT LJ-chain contact outside positive finite domain')
        return mbar*_johnson(t, r)+(1-mbar)*logarithm(contact)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SoftSAFTModel:
        data = dict(payload)
        data['components'] = tuple(SAFTSegment(**c) for c in data['components'])
        return cls(**data)
