"""Feynman-Hibbs SAFT-VRQ-Mie on shared molecular EOS arithmetic.

The quantum potential is a finite inverse-power sum. Its zero, well minimum,
and BH diameter retain temperature derivatives through order three. The
dispersion terms reuse the classical Mie Sutherland integrals and empirical
correction functions. Explicit chain, hard-sphere and truncation controls retain
the selected public source equations, including their documented limitations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from math import factorial, isfinite, pi, sqrt
from typing import Any

import numpy as np
from scipy.optimize import brentq

from chemworld.physchem.caloric_math import (
    compensated_sum,
    derivative_depth,
    exponential,
    partial_derivative,
    scalar_value,
)
from chemworld.physchem.dufal_association import DufalAssociation
from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.mie_reference_options import mie_reference_contributions
from chemworld.physchem.mie_reference_options import santos_hard_sphere as santos_hard_sphere
from chemworld.physchem.pcsaft import (
    AVOGADRO_PER_MOL,
)
from chemworld.physchem.saftvrmie import (
    MieSegment,
    mie_prefactor,
)

_NODES, _WEIGHTS_RAW = np.polynomial.legendre.leggauss(128)
_RADII, _WEIGHTS = (_NODES+1)/2, _WEIGHTS_RAW/2


@dataclass(frozen=True)
class QuantumMieSegment(MieSegment):
    molecular_mass_kg: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if (isinstance(self.molecular_mass_kg, bool) or not isfinite(self.molecular_mass_kg)
                or self.molecular_mass_kg <= 0):
            raise ValueError("quantum Mie requires explicit positive molecular mass in kg")


@dataclass(frozen=True)
class QuantumMiePair:
    """A declared unlike or pure pair; reduced radius is r/sigma."""

    diameter_m: float
    well_depth_K: float
    repulsive_exponent: float
    attractive_exponent: float
    reduced_mass_kg: float
    feynman_hibbs_order: int = 1
    planck_J_s: float = 6.62607015e-34
    boltzmann_J_K: float = 1.380649e-23

    def __post_init__(self) -> None:
        if (any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (
                self.diameter_m, self.well_depth_K, self.reduced_mass_kg, self.boltzmann_J_K))
                or any(isinstance(v, bool) or not isfinite(v) for v in (
                    self.repulsive_exponent, self.attractive_exponent, self.planck_J_s))
                or self.planck_J_s < 0
                or not self.repulsive_exponent > self.attractive_exponent > 3
                or type(self.feynman_hibbs_order) is not int
                or self.feynman_hibbs_order not in (0, 1, 2)):
            raise ValueError("quantum pair requires positive SI inputs and FH order 0,1,2")

    def powers(self, temperature: Any) -> tuple[tuple[float, Any], ...]:
        """Coefficients in u/(C*epsilon)=sum(c_l*(sigma/r)^l)."""
        if not isfinite(scalar_value(temperature)) or scalar_value(temperature) <= 0:
            raise ValueError("quantum potential temperature must be positive")
        d = self.planck_J_s**2/(96*pi*pi*self.reduced_mass_kg*self.boltzmann_J_K
                              *self.diameter_m**2*temperature)
        terms = []
        for exponent, sign in ((self.repulsive_exponent, 1), (self.attractive_exponent, -1)):
            terms.append((exponent, sign))
            if self.feynman_hibbs_order >= 1:
                terms.append((exponent+2, sign*d*exponent*(exponent-1)))
            if self.feynman_hibbs_order >= 2:
                terms.append((exponent+4,
                    sign*d*d*exponent*(exponent-1)*(exponent+1)*(exponent+2)/2))
        return tuple(terms)

    def potential_K(  # noqa: N802 - explicit SI unit
        self, temperature: Any, reduced_radius: Any, radial_order: int = 0
    ) -> Any:
        """Potential/kB or its derivative w.r.t. dimensionless r/sigma."""
        if (not isfinite(scalar_value(reduced_radius)) or scalar_value(reduced_radius) <= 0
                or type(radial_order) is not int or not 0 <= radial_order <= 2):
            raise ValueError("potential radius positive; radial derivative order in 0..2")
        return self.well_depth_K*mie_prefactor(self.repulsive_exponent,
            self.attractive_exponent)*compensated_sum(
                coefficient*(1 if radial_order == 0 else -exponent if radial_order == 1
                    else exponent*(exponent+1))*reduced_radius**(-exponent-radial_order)
                for exponent, coefficient in self.powers(temperature))

    def geometry(self, temperature: Any, *, use_cache: bool = True) -> tuple[Any, Any, Any]:
        """Effective sigma/sigma, effective epsilon/K and BH diameter/sigma."""
        t, depth = scalar_value(temperature), derivative_depth(temperature)
        if not isfinite(t) or t <= 0 or depth > 3:
            raise ValueError("quantum geometry needs positive T and derivative order <=3")
        coefficients = (_geometry_coefficients(self, float(t), depth) if use_cache else
                        _geometry_coefficients.__wrapped__(self, float(t), depth))
        delta = temperature-t
        return tuple(compensated_sum(row[k]*delta**k/factorial(k) for k in range(depth+1))
                     for row in coefficients)


def _pair_geometry(pair: QuantumMiePair, temperature: Any) -> tuple[Any, Any, Any]:
    """Scalar roots lifted by existing jet Newton arithmetic (no differencing)."""
    t = scalar_value(temperature)
    zero = brentq(lambda r: pair.potential_K(t, r), .5, 64., xtol=1e-14)
    minimum = brentq(lambda r: pair.potential_K(t, r, 1), zero, 128., xtol=1e-14)
    # A converged scalar root gains one exact jet order per Newton iteration.
    for _ in range(derivative_depth(temperature)):
        zero -= pair.potential_K(temperature, zero)/pair.potential_K(temperature, zero, 1)
        minimum -= pair.potential_K(temperature, minimum, 1)/pair.potential_K(
            temperature, minimum, 2)
    depth = -pair.potential_K(temperature, minimum)
    boltzmann = []
    for node, weight in zip(_RADII, _WEIGHTS, strict=True):
        argument = pair.potential_K(temperature, zero*float(node))/temperature
        if scalar_value(argument) < 745:
            boltzmann.append(float(weight)*exponential(-argument))
    diameter = zero*(1-compensated_sum(boltzmann))
    return zero, depth, diameter


@lru_cache(maxsize=2048)
def _geometry_coefficients(pair: QuantumMiePair, temperature: float,
                           depth: int) -> tuple[tuple[float, ...], ...]:
    values = _pair_geometry(pair, temperature)
    return tuple((float(values[i]), *(float(partial_derivative(
        lambda v, index=i: _pair_geometry(pair, v["T"])[index],
        {"T": temperature}, ("T",)*k))
        for k in range(1, depth+1))) for i in range(3))


@dataclass(frozen=True)
class SAFTVRQMieModel:
    components: tuple[QuantumMieSegment, ...]
    interaction: tuple[tuple[float, ...], ...]
    temperature_range_K: tuple[float, float]
    source: str
    feynman_hibbs_order: int = 1
    hard_sphere_reference: str = "nonadditive-Santos"
    energy_mixing: str = "Lafitte"
    diameter_interaction: tuple[tuple[float, ...], ...] | None = None
    repulsion_interaction: tuple[tuple[float, ...], ...] | None = None
    gas_constant_J_mol_K: float = R_J_PER_MOL_K
    planck_J_s: float = 6.62607015e-34
    boltzmann_J_K: float = 1.380649e-23
    exact_binary_dhs: bool | None = None
    enable_hs_extra: bool | None = None
    enable_hard_sphere: bool = True
    enable_a1: bool = True
    enable_a2: bool = True
    enable_a3: bool = True
    enable_chain: bool = True
    enable_truncation_correction: bool = False
    enable_shift_correction: bool = False
    reduced_radius_cut: float = 3.5
    enable_temperature_cache: bool = True
    association: DufalAssociation | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        n = len(self.components)
        if not n or len(set(self.component_ids)) != n or not self.source:
            raise ValueError("SAFT-VRQ-Mie requires unique components and provenance")
        for field in ("interaction", "diameter_interaction", "repulsion_interaction"):
            raw = getattr(self, field)
            matrix = tuple(tuple(r) for r in raw) if raw is not None else ((0.,)*n,)*n
            if len(matrix) != n or any(len(r) != n for r in matrix):
                raise ValueError("quantum Mie needs square interaction matrices")
            for i, row in enumerate(matrix):
                for j, v in enumerate(row):
                    if (isinstance(v, bool) or not isfinite(v) or v >= 1
                            or v != matrix[j][i] or (i == j and v != 0)):
                        raise ValueError("quantum Mie interactions symmetric <1, zero diagonal")
            object.__setattr__(self, field, matrix)
        if (type(self.feynman_hibbs_order) is not int or self.feynman_hibbs_order not in (0, 1, 2)
                or self.hard_sphere_reference not in (
                    "Lafitte", "additive", "additive-exact", "nonadditive-Santos")
                or self.energy_mixing not in ("Lafitte", "Lorentz-Berthelot")
                or len(self.temperature_range_K) != 2
                or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (
                    *self.temperature_range_K, self.gas_constant_J_mol_K, self.boltzmann_J_K))
                or self.temperature_range_K[0] >= self.temperature_range_K[1]
                or isinstance(self.planck_J_s, bool) or not isfinite(self.planck_J_s)
                or self.planck_J_s < 0):
            raise ValueError("invalid quantum Mie order, HS reference, constants or T domain")
        flags = (self.enable_hard_sphere, self.enable_a1, self.enable_a2, self.enable_a3,
                 self.enable_chain, self.enable_truncation_correction,
                 self.enable_shift_correction, self.enable_temperature_cache)
        if (any(type(v) is not bool for v in flags)
                or any(v is not None and type(v) is not bool
                       for v in (self.exact_binary_dhs, self.enable_hs_extra))
                or isinstance(self.reduced_radius_cut, bool)
                or not isfinite(self.reduced_radius_cut) or self.reduced_radius_cut <= 0):
            raise ValueError("quantum Mie controls require booleans and a positive cutoff")
        if self.association is not None and self.association.component_ids != self.component_ids:
            raise ValueError("quantum Mie association component ordering must match")
        for i in range(n):
            for j in range(i, n):
                self.pair(i, j)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(c.component_id for c in self.components)

    def pair(self, i: int, j: int) -> QuantumMiePair:
        left, right = self.components[i], self.components[j]
        assert self.diameter_interaction is not None and self.repulsion_interaction is not None
        sigma = (left.diameter_m+right.diameter_m)/2*(1-self.diameter_interaction[i][j])
        epsilon = sqrt(left.well_depth_K*right.well_depth_K)*(1-self.interaction[i][j])
        if self.energy_mixing == "Lafitte":
            epsilon *= (left.diameter_m*right.diameter_m)**1.5/sigma**3
        repulsion = 3+(1-self.repulsion_interaction[i][j])*sqrt(
            (left.repulsive_exponent-3)*(right.repulsive_exponent-3))
        attraction = 3+sqrt((left.attractive_exponent-3)*(right.attractive_exponent-3))
        mass = 1/(1/left.molecular_mass_kg+1/right.molecular_mass_kg)
        return QuantumMiePair(sigma, epsilon, repulsion, attraction, mass,
            self.feynman_hibbs_order, self.planck_J_s, self.boltzmann_J_K)

    def de_boer_parameter(self, i: int) -> float:
        component = self.components[i]
        return self.planck_J_s/(component.diameter_m*sqrt(component.molecular_mass_kg
            *component.well_depth_K*self.boltzmann_J_K))

    def with_de_boer_parameter(self, i: int, value: float) -> SAFTVRQMieModel:
        """Return a new, serializable model with the requested molecular mass."""
        if isinstance(value, bool) or not isfinite(value) or value <= 0 or self.planck_J_s <= 0:
            raise ValueError("mass adjustment requires positive de Boer parameter and Planck h")
        component = self.components[i]
        mass = (self.planck_J_s/(value*component.diameter_m))**2/(
            component.well_depth_K*self.boltzmann_J_K)
        components = tuple(replace(c, molecular_mass_kg=mass) if j == i else c
                           for j, c in enumerate(self.components))
        return replace(self, components=components)

    def de_broglie_wavelength(self, i: int, temperature: Any) -> Any:
        if not isfinite(scalar_value(temperature)) or scalar_value(temperature) <= 0:
            raise ValueError("thermal wavelength requires positive temperature")
        return self.planck_J_s/(2*pi*self.components[i].molecular_mass_kg
                               *self.boltzmann_J_K*temperature)**.5

    def potential_K(  # noqa: N802 - explicit SI unit
        self, temperature: Any, i: int, j: int, reduced_radius: Any, radial_order: int = 0
    ) -> Any:
        """Pair potential/kB with this model's declared cutoff and optional shift.

        Derivatives use reduced radius. At the cutoff the zero right-hand branch
        is returned; the unshifted potential and the force can be discontinuous.
        """
        pair = self.pair(i, j)
        value = pair.potential_K(temperature, reduced_radius, radial_order)
        if self.enable_truncation_correction:
            if scalar_value(reduced_radius) >= self.reduced_radius_cut:
                return 0.*value
            if self.enable_shift_correction and radial_order == 0:
                value -= pair.potential_K(temperature, self.reduced_radius_cut)
        return value

    def alpha(self, temperature: Any) -> tuple[tuple[Any, ...], ...]:
        """Dimensionless integrated effective attraction for every component pair."""
        if self.enable_truncation_correction and self.feynman_hibbs_order > 0:
            raise ValueError("source truncated-alpha helper is unavailable for quantum potentials")
        rows = []
        for i in range(len(self.components)):
            row = []
            for j in range(len(self.components)):
                pair = self.pair(i, j)
                sigma, epsilon, _ = pair.geometry(temperature,
                    use_cache=self.enable_temperature_cache)
                c = mie_prefactor(pair.repulsive_exponent, pair.attractive_exponent)
                if self.enable_truncation_correction:
                    cutoff = self.reduced_radius_cut
                    shift = pair.potential_K(temperature, cutoff) if self.enable_shift_correction\
                        else 0.
                    if self.enable_shift_correction:
                        minimum = (pair.repulsive_exponent/pair.attractive_exponent)**(
                            1/(pair.repulsive_exponent-pair.attractive_exponent))
                        sigma = brentq(lambda r, p=pair, s=shift: float(scalar_value(
                            p.potential_K(temperature, r)-s)), .5, minimum)
                        epsilon += shift
                    integral = c*pair.well_depth_K*compensated_sum(
                        a*(sigma**(3-p)-cutoff**(3-p))/(p-3)
                        for p, a in pair.powers(temperature))-shift*(cutoff**3-sigma**3)/3
                    row.append(-integral/(epsilon*sigma**3))
                else:
                    row.append(-c*pair.well_depth_K/epsilon*compensated_sum(
                        a*sigma**(-p)/(p-3) for p, a in pair.powers(temperature)))
            rows.append(tuple(row))
        return tuple(rows)

    def _truncation(self, temperature: Any, density: Any) -> Any:
        if not self.enable_truncation_correction:
            return 0.
        if len(self.components) != 1:
            raise ValueError("source quantum truncation EOS correction supports only pure fluids")
        pair = self.pair(0, 0)
        rc = self.reduced_radius_cut
        integral = compensated_sum(a*rc**(3-p)/(p-3) for p, a in pair.powers(temperature))
        if self.enable_shift_correction:
            integral += compensated_sum(a*rc**(3-p)/3 for p, a in pair.powers(temperature))
        # Source correction uses molecular density, including for m != 1.
        return -2*pi*density*AVOGADRO_PER_MOL*pair.diameter_m**3*pair.well_depth_K*\
            mie_prefactor(pair.repulsive_exponent, pair.attractive_exponent)*integral/temperature

    def contributions(self, temperature: Any, density: Any,
                      fractions: Sequence[Any]) -> dict[str, Any]:
        result = mie_reference_contributions(self, temperature, density, fractions)
        if self.association is not None:
            result['association'] = self.association.alphar(temperature, density, fractions)
        return result

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        return compensated_sum(self.contributions(temperature, density, fractions).values())

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld-saftvrqmie-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SAFTVRQMieModel:
        data = dict(payload)
        if data.pop("model", None) != "chemworld-saftvrqmie-v1":
            raise ValueError("unknown quantum Mie model encoding")
        data["components"] = tuple(QuantumMieSegment(**c) for c in data["components"])
        if data.get("association") is not None:
            data["association"] = DufalAssociation.from_dict(data["association"])
        return cls(**data)
