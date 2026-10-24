"""Declared fixed-site hindered adsorbates, from the published partition model.

Two equivalent surface translations and one perpendicular rotation replace three
explicit modes. Remaining modes are harmonic. This is the Sprowl--Campbell--
Arnadottir approximation, not a diffusion solver or an exact quantum rotor.
The lattice mixing entropy belongs to the original catalytic activities. No
temperature-dependent standard-concentration entropy is included here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from math import expm1, fsum, isfinite, log, pi, sqrt
from types import MappingProxyType
from typing import Any

from scipy.special import i0e, i1e

from chemworld.physchem.catalysis_descriptors import (
    LinearEnergyScaling,
    finite_number,
    validate_source,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.thermochemistry import SpeciesThermoState

AVOGADRO_PER_MOL = 6.02214076e23
PLANCK_J_S = 6.62607015e-34


def harmonic_mode_totals(
    quanta_J_mol: Sequence[float], temperature_K: float, initial_energy_J_mol: float = 0.0
) -> tuple[float, float, float]:
    """U (including ZPE), S and Cv of already validated harmonic modes."""
    energy, entropy, capacity = initial_energy_J_mol, 0.0, 0.0
    for quantum in quanta_J_mol:
        x = quantum / (R * temperature_K)
        if x > 700:
            energy += quantum * 0.5
            continue
        occupation = 1 / expm1(x)
        energy += quantum * (0.5 + occupation)
        entropy += R * (x * occupation - log(-expm1(-x)))
        capacity += R * x * x * occupation * (1 + occupation)
    return energy, entropy, capacity


def _hindered_mode(
    quantum: float, barrier: float, temperature: float
) -> tuple[float, float, float]:
    """Derivatives of one fixed-cell partition function, in molar SI units.

    ln q = -ln(1-exp(-b))-b/2+b/(2+16r)
           +ln(sqrt(2*pi*x)*exp(-x)*I0(x)),
    where b=quantum/RT, r=barrier/quantum, x=barrier/(2RT).
    The Bessel term is evaluated with scaled functions, without exp(x).
    """
    b = quantum / (R * temperature)
    r = barrier / quantum
    x = barrier / (2 * R * temperature)
    occupation = 0.0 if b > 700 else 1 / expm1(b)
    # x^2 d(I1/I0)/dx is the potential-energy variance. Its usual
    # subtraction loses precision in the deep-well limit.
    if x > 100:
        inverse = 1 / x
        # log of the scaled Bessel prefactor and both derivatives of that
        # SAME asymptotic series. Avoid x*(1-I1/I0), whose subtraction rounds
        # to zero in deep wells and would break the caloric identity.
        log_bessel = inverse * (
            1 / 8 + inverse * (1 / 16 + inverse * (25 / 384 + inverse * (13 / 128)))
        )
        shifted_ratio = inverse * (
            1 / 8 + inverse * (1 / 8 + inverse * (25 / 128 + inverse * (13 / 32)))
        )
        variance = 0.5 + inverse * (
            0.25 + inverse * (0.375 + inverse * (25 / 32 + inverse * (65 / 32)))
        )
    else:
        ratio = float(i1e(x) / i0e(x))
        shifted_ratio = x * (1 - ratio) - 0.5
        log_bessel = 0.5 * (log(2 * pi) + log(x)) + log(float(i0e(x)))
        variance = x * x * (1 - ratio * ratio) - x * ratio
    energy = quantum * (0.5 + occupation - 1 / (2 + 16 * r)) + R * temperature * shifted_ratio
    entropy = R * (b * occupation - log(-expm1(-b)) + shifted_ratio + log_bessel)
    harmonic_capacity = 0.0 if b > 700 else b * b * occupation * (1 + occupation)
    capacity = R * (harmonic_capacity - 0.5 + variance)
    return energy, entropy, capacity


@dataclass(frozen=True)
class HinderedAdsorbateThermo:
    """Explicit molecular mode partition at one fixed lattice-site standard.

    Energies and electronic scaling are J/mol; mass is kg/mol; inertia is
    kg m^2 per molecule; site density is mol of sites/m^2. Mode IDs partition
    exactly 3N coordinates, without sorting or silently dropping frequencies.
    The rotor uses the source model's one-period convention; rotational_minima
    enters its curvature. An additional full-circle degeneracy is not inferred.
    """

    electronic_energy: LinearEnergyScaling
    descriptors: Mapping[str, float]
    vibrational_energies_J_mol: tuple[float, ...]
    harmonic_mode_ids: tuple[str, ...]
    hindered_mode_ids: tuple[str, str, str]
    number_of_atoms: int
    translation_barrier_J_mol: float
    rotation_barrier_J_mol: float
    site_density_mol_m2: float
    molar_mass_kg_mol: float
    moment_of_inertia_kg_m2: float
    rotational_minima: int
    symmetry_number: int
    minimum_temperature_K: float
    maximum_temperature_K: float
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", MappingProxyType(dict(self.descriptors)))
        for name in ("vibrational_energies_J_mol", "harmonic_mode_ids", "hindered_mode_ids"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        validate_source(self.source)
        self.electronic_energy.evaluate(self.descriptors)
        for name in ("number_of_atoms", "rotational_minima", "symmetry_number"):
            value = getattr(self, name)
            if type(value) is not int or value < (2 if name == "number_of_atoms" else 1):
                raise ValueError("hindered model requires positive integer counts and N >= 2")
        all_ids = (*self.harmonic_mode_ids, *self.hindered_mode_ids)
        if (
            len(self.vibrational_energies_J_mol) != 3 * self.number_of_atoms - 3
            or len(self.harmonic_mode_ids) != len(self.vibrational_energies_J_mol)
            or len(self.hindered_mode_ids) != 3
            or any(not isinstance(key, str) or not key.strip() for key in all_ids)
            or len(set(all_ids)) != 3 * self.number_of_atoms
        ):
            raise ValueError(
                "explicit unique mode partition must contain 3N-3 harmonic + 3 hindered"
            )
        for name in (
            "translation_barrier_J_mol",
            "rotation_barrier_J_mol",
            "site_density_mol_m2",
            "molar_mass_kg_mol",
            "moment_of_inertia_kg_m2",
            "minimum_temperature_K",
            "maximum_temperature_K",
        ):
            value = getattr(self, name)
            finite_number(value, name)
            if value <= 0:
                raise ValueError(name + " must be positive")
        for value in self.vibrational_energies_J_mol:
            if isinstance(value, complex):
                raise ValueError("harmonic modes must be real and positive; no imaginary modes")
            finite_number(value, "harmonic quantum")
            if value <= 0:
                raise ValueError("harmonic modes must be positive; no zero/imaginary modes")
        if self.minimum_temperature_K >= self.maximum_temperature_K:
            raise ValueError("hindered model temperature interval must be ordered")
        if any(not isfinite(v) or v <= 0 for v in self.hindered_quanta_J_mol):
            raise ValueError("hindered parameters must give finite positive frequencies")

    @cached_property
    def hindered_quanta_J_mol(self) -> tuple[float, float]:  # noqa: N802 - explicit SI property
        area = 1 / (AVOGADRO_PER_MOL * self.site_density_mol_m2)
        trans = sqrt(self.translation_barrier_J_mol / (2 * self.molar_mass_kg_mol * area))
        rotor = (
            self.rotational_minima
            / (2 * pi)
            * sqrt(
                self.rotation_barrier_J_mol / (2 * AVOGADRO_PER_MOL * self.moment_of_inertia_kg_m2)
            )
        )
        return AVOGADRO_PER_MOL * PLANCK_J_S * trans, AVOGADRO_PER_MOL * PLANCK_J_S * rotor

    @property
    def zero_point_energy_J_mol(self) -> float:  # noqa: N802 - explicit SI property
        """Source model ZPE convention; anharmonic shift remains in U(T)."""
        trans, rotor = self.hindered_quanta_J_mol
        return 0.5 * fsum((*self.vibrational_energies_J_mol, trans, trans, rotor))

    def evaluate(
        self, temperature_K: float, *, species_id: str = "adsorbate"
    ) -> SpeciesThermoState:
        finite_number(temperature_K, "temperature")
        if not self.minimum_temperature_K <= temperature_K <= self.maximum_temperature_K:
            raise ValueError("temperature outside declared hindered domain")
        trans, rotor = self.hindered_quanta_J_mol
        vt = _hindered_mode(trans, self.translation_barrier_J_mol, temperature_K)
        vr = _hindered_mode(rotor, self.rotation_barrier_J_mol, temperature_K)
        vh = harmonic_mode_totals(self.vibrational_energies_J_mol, temperature_K)
        energy, entropy, capacity = (2 * vt[k] + vr[k] + vh[k] for k in range(3))
        energy += self.electronic_energy.evaluate(self.descriptors)
        entropy -= R * log(self.symmetry_number)
        gibbs = energy - temperature_K * entropy
        if not all(isfinite(v) for v in (energy, entropy, capacity, gibbs)) or capacity < 0:
            raise ValueError("hindered thermochemistry is outside finite positive-Cv domain")
        return SpeciesThermoState(
            species_id,
            temperature_K,
            capacity / R,
            energy / (R * temperature_K),
            entropy / R,
            capacity,
            energy,
            entropy,
            gibbs,
            "hindered-adsorbate-fixed-site-v1",
        )

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in self.__dataclass_fields__}
        result.update(
            model="hindered-adsorbate-fixed-site-v1",
            electronic_energy=self.electronic_energy.to_dict(),
            descriptors=dict(self.descriptors),
            source=self.source.to_dict(),
        )
        for name in ("vibrational_energies_J_mol", "harmonic_mode_ids", "hindered_mode_ids"):
            result[name] = list(result[name])
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> HinderedAdsorbateThermo:
        values = dict(raw)
        if values.pop("model") != "hindered-adsorbate-fixed-site-v1":
            raise ValueError("unknown hindered adsorbate model")
        values["electronic_energy"] = LinearEnergyScaling.from_dict(values["electronic_energy"])
        values["source"] = DataSourceProvenance.from_dict(values["source"])
        return cls(**values)
