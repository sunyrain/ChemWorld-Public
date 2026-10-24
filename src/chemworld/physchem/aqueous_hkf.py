"""Revised HKF on stable IF97 region-1 water, from one differentiable Gibbs G.

Equations: Johnson, Oelkers & Helgeson (1992), Computers & Geosciences 18,
899-947, equations 45-59 and Tables 2-3. No upstream implementation is used.
IF97/IAPWS97 is an explicit solvent choice, not SUPCRT92's HGK84/JN91 closure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cached_property
from math import isfinite
from types import MappingProxyType
from typing import Any, cast

from chemworld.physchem.caloric_math import logarithm, nonnegative_power, scalar_value
from chemworld.physchem.elements import element_matrix
from chemworld.physchem.second_order_pair import SecondOrderPair
from chemworld.physchem.water_dielectric import water_relative_permittivity
from chemworld.physchem.water_if97 import (
    water_if97_liquid_density,
    water_if97_liquid_gibbs,
)

_TR, _PR, _THETA, _PSI = 298.15, 100000.0, 228.0, 2600e5


@dataclass(frozen=True)
class AqueousReferenceBasis:
    """Explicit entropy gauge per mol of atoms, including the ionic convention.

    G_native=Hf-T*S_standard. Gf=G_native+T*(sum atoms*S_element+z*S_charge).
    Missing elemental/charge references are errors, never inferred from a species name.
    """

    basis_id: str
    element_entropy_J_mol_K: Mapping[str, float]
    charge_entropy_J_mol_K: float
    source_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "element_entropy_J_mol_K", MappingProxyType(dict(self.element_entropy_J_mol_K))
        )
        if not self.basis_id or not self.source_id or not self.element_entropy_J_mol_K:
            raise ValueError("aqueous reference needs identity, elements and source")
        if any(
            isinstance(v, bool) or not isfinite(v)
            for v in (*self.element_entropy_J_mol_K.values(), self.charge_entropy_J_mol_K)
        ):
            raise ValueError("reference entropies must be explicit finite numbers")

    def entropy_shift(self, composition: Mapping[str, float], charge: int) -> float:
        element_matrix([dict(composition)])
        if set(composition) - self.element_entropy_J_mol_K.keys():
            raise ValueError("missing elemental entropy reference")
        if type(charge) is not int:
            raise ValueError("charge reference requires an explicit integer charge")
        return sum(n * self.element_entropy_J_mol_K[e] for e, n in composition.items()) + (
            charge * self.charge_entropy_J_mol_K
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis_id": self.basis_id,
            "element_entropy_J_mol_K": dict(self.element_entropy_J_mol_K),
            "charge_entropy_J_mol_K": self.charge_entropy_J_mol_K,
            "source_id": self.source_id,
        }


@dataclass(frozen=True)
class AqueousGibbsState:
    cp_J_mol_K: float
    enthalpy_J_mol: float
    entropy_J_mol_K: float
    gibbs_J_mol: float
    molar_volume_m3_mol: float


def _state(function: Any, temperature_K: float, pressure_Pa: float) -> AqueousGibbsState:
    value = function(SecondOrderPair(temperature_K, dx=1), SecondOrderPair(pressure_Pa, dy=1))
    if not isinstance(value, SecondOrderPair):
        value = SecondOrderPair(value)
    result = AqueousGibbsState(
        -temperature_K * value.dxx,
        value.value - temperature_K * value.dx,
        -value.dx,
        value.value,
        value.dy,
    )
    if not all(isfinite(v) for v in asdict(result).values()):
        raise ValueError("aqueous Gibbs response is nonfinite")
    return result


def hkf_solvent_g(temperature_K: Any, pressure_Pa: Any) -> Any:
    """Shock solvent g, angstrom; published density branch and f correction."""
    rho = water_if97_liquid_density(temperature_K, pressure_Pa) / 1000
    if scalar_value(rho) < 0.35:
        raise ValueError("HKF solvent g requires density >= 0.35 g/cm3")
    if scalar_value(rho) >= 1:
        return 0.0
    t, p = temperature_K - 273.15, pressure_Pa / 1e5
    a = -2.037662 + 0.005747 * t - 6.557892e-6 * t * t
    b = 6.107361 - 0.01074377 * t + 1.268348e-5 * t * t
    g = a * nonnegative_power(1 - rho, b)
    if 155 < scalar_value(t) <= 355 and scalar_value(p) < 1000:
        x, y = (t - 155) / 300, 1000 - p
        g -= (x**4.8 + 36.66666 * x**16) * (-1.504956e-10 * y**3 + 5.017997e-14 * y**4)
    return g


@dataclass(frozen=True)
class RevisedHKFSpeciesThermo:
    """HKF coefficients in SI; a=(J/mol/Pa,J/mol,J*K/mol/Pa,J*K/mol).

    c=(J/mol/K,J*K/mol), omega=J/mol. Source Hf, Gf and S are preserved.
    At least Hf or Gf must be supplied. If both exist they must agree under the
    explicit gauge within a caller-declared source precision bound.
    """

    species_id: str
    composition: Mapping[str, float]
    charge: int
    reference_basis: AqueousReferenceBasis
    entropy_ref_J_mol_K: float
    formation_enthalpy_ref_J_mol: float | None
    formation_gibbs_ref_J_mol: float | None
    a: tuple[float, float, float, float]
    c: tuple[float, float]
    omega_ref_J_mol: float
    source_id: str
    reference_consistency_tolerance_J_mol: float
    reference_pressure_Pa: float = _PR
    reference_temperature_K: float = _TR

    def __post_init__(self) -> None:
        object.__setattr__(self, "composition", MappingProxyType(dict(self.composition)))
        object.__setattr__(self, "a", tuple(self.a))
        object.__setattr__(self, "c", tuple(self.c))
        if not isinstance(self.reference_basis, AqueousReferenceBasis):
            raise ValueError("HKF needs an explicit elemental and charge reference basis")
        shift = self.reference_basis.entropy_shift(self.composition, self.charge)
        if (
            not self.species_id
            or not self.source_id
            or len(self.a) != 4
            or len(self.c) != 2
            or self.reference_temperature_K != _TR
            or self.reference_pressure_Pa != _PR
            or self.reference_consistency_tolerance_J_mol < 0
        ):
            raise ValueError("HKF requires complete coefficients, source and 298.15 K/1 bar datum")
        h, g = self.formation_enthalpy_ref_J_mol, self.formation_gibbs_ref_J_mol
        if h is None and g is None:
            raise ValueError("HKF needs formation enthalpy or Gibbs energy")
        if any(
            isinstance(v, bool) or not isfinite(v)
            for v in (
                *self.a,
                *self.c,
                self.omega_ref_J_mol,
                self.entropy_ref_J_mol_K,
                self.reference_consistency_tolerance_J_mol,
                *(v for v in (h, g) if v is not None),
            )
        ):
            raise ValueError("HKF parameters must be finite numbers")
        if h is not None and g is not None:
            residual = h - _TR * (self.entropy_ref_J_mol_K - shift) - g
            if abs(residual) > self.reference_consistency_tolerance_J_mol:
                raise ValueError(
                    f"inconsistent formation H/G/standard S: residual={residual:.9g} J/mol"
                )
        if self.charge:
            denominator = self.omega_ref_J_mol / (166027 * 4.184) + self.charge / 3.082
            if denominator <= 0:
                raise ValueError("HKF reference effective radius must be positive")

    @property
    def enthalpy_ref_J_mol(self) -> float:  # noqa: N802 - property units follow the data contract
        if self.formation_enthalpy_ref_J_mol is not None:
            return self.formation_enthalpy_ref_J_mol
        return cast(float, self.formation_gibbs_ref_J_mol) + _TR * (
            self.entropy_ref_J_mol_K
            - self.reference_basis.entropy_shift(self.composition, self.charge)
        )

    def omega(self, temperature_K: Any, pressure_Pa: Any) -> Any:
        if not self.charge:
            return self.omega_ref_J_mol
        eta, z = 166027 * 4.184, self.charge
        radius = z * z / (self.omega_ref_J_mol / eta + z / 3.082)
        g = hkf_solvent_g(temperature_K, pressure_Pa)
        if scalar_value(radius + abs(z) * g) <= 0 or scalar_value(3.082 + g) <= 0:
            raise ValueError("HKF effective ionic radius became nonpositive")
        return eta * (z * z / (radius + abs(z) * g) - z / (3.082 + g))

    def _born(self, T: Any, P: Any) -> Any:
        epsilon = water_relative_permittivity(T, water_if97_liquid_density(T, P))
        return self.omega(T, P) * (1 / epsilon - 1)

    @cached_property
    def _born_reference(self) -> tuple[float, float]:
        value = self._born(SecondOrderPair(_TR, dx=1), _PR)
        return value.value, value.dx

    def gibbs(self, temperature_K: Any, pressure_Pa: Any) -> Any:
        T, P = temperature_K, pressure_Pa
        water_if97_liquid_gibbs(T, P)
        a1, a2, a3, a4 = self.a
        c1, c2 = self.c
        dH = c1 * (T - _TR) + c2 * (1 / (_TR - _THETA) - 1 / (T - _THETA))

        def integral(x: Any) -> Any:
            return logarithm(x / (x - _THETA)) / _THETA**2 - 1 / (_THETA * (x - _THETA))

        dS = c1 * logarithm(T / _TR) + c2 * (integral(T) - integral(_TR))
        pressure = (a1 + a3 / (T - _THETA)) * (P - _PR) + (a2 + a4 / (T - _THETA)) * logarithm(
            (_PSI + P) / (_PSI + _PR)
        )
        q0, qt0 = self._born_reference
        return (
            self.enthalpy_ref_J_mol
            - T * self.entropy_ref_J_mol_K
            + dH
            - T * dS
            + pressure
            + self._born(T, P)
            - q0
            - qt0 * (T - _TR)
        )

    def evaluate_tp(self, temperature_K: float, pressure_Pa: float) -> AqueousGibbsState:
        return _state(self.gibbs, temperature_K, pressure_Pa)

    def evaluate(self, temperature_K: float) -> AqueousGibbsState:
        return self.evaluate_tp(temperature_K, self.reference_pressure_Pa)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "revised-HKF-IF97",
            "species_id": self.species_id,
            "composition": dict(self.composition),
            "charge": self.charge,
            "reference_basis": self.reference_basis.to_dict(),
            "entropy_ref_J_mol_K": self.entropy_ref_J_mol_K,
            "formation_enthalpy_ref_J_mol": self.formation_enthalpy_ref_J_mol,
            "formation_gibbs_ref_J_mol": self.formation_gibbs_ref_J_mol,
            "a": list(self.a),
            "c": list(self.c),
            "omega_ref_J_mol": self.omega_ref_J_mol,
            "source_id": self.source_id,
            "reference_consistency_tolerance_J_mol": self.reference_consistency_tolerance_J_mol,
            "reference_pressure_Pa": self.reference_pressure_Pa,
            "reference_temperature_K": self.reference_temperature_K,
        }


@dataclass(frozen=True)
class IF97AqueousWaterThermo:
    species_id: str
    reference_basis: AqueousReferenceBasis
    enthalpy_ref_J_mol: float
    entropy_ref_J_mol_K: float
    source_id: str
    molar_mass_kg_mol: float = 0.018015268
    reference_pressure_Pa: float = _PR
    reference_temperature_K: float = _TR

    def __post_init__(self) -> None:
        if not isinstance(self.reference_basis, AqueousReferenceBasis):
            raise ValueError("water needs an explicit elemental and charge reference basis")
        self.reference_basis.entropy_shift({"H": 2, "O": 1}, 0)
        if (
            not self.species_id
            or not self.source_id
            or self.molar_mass_kg_mol <= 0
            or self.reference_temperature_K != _TR
            or self.reference_pressure_Pa != _PR
            or any(
                isinstance(v, bool) or not isfinite(v)
                for v in (self.molar_mass_kg_mol, self.enthalpy_ref_J_mol, self.entropy_ref_J_mol_K)
            )
        ):
            raise ValueError("water requires finite explicit H/S datum, mass and source")

    @property
    def composition(self) -> dict[str, int]:
        return {"H": 2, "O": 1}

    @cached_property
    def _offsets(self) -> tuple[float, float]:
        ref = _state(lambda T, P: self.molar_mass_kg_mol * water_if97_liquid_gibbs(T, P), _TR, _PR)
        return (
            self.enthalpy_ref_J_mol - ref.enthalpy_J_mol,
            self.entropy_ref_J_mol_K - ref.entropy_J_mol_K,
        )

    def gibbs(self, temperature_K: Any, pressure_Pa: Any) -> Any:
        dh, ds = self._offsets
        return (
            self.molar_mass_kg_mol * water_if97_liquid_gibbs(temperature_K, pressure_Pa)
            + dh
            - temperature_K * ds
        )

    def evaluate_tp(self, temperature_K: float, pressure_Pa: float) -> AqueousGibbsState:
        return _state(self.gibbs, temperature_K, pressure_Pa)

    def evaluate(self, temperature_K: float) -> AqueousGibbsState:
        return self.evaluate_tp(temperature_K, self.reference_pressure_Pa)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "aqueous-IF97-water",
            "species_id": self.species_id,
            "reference_basis": self.reference_basis.to_dict(),
            "enthalpy_ref_J_mol": self.enthalpy_ref_J_mol,
            "entropy_ref_J_mol_K": self.entropy_ref_J_mol_K,
            "source_id": self.source_id,
            "molar_mass_kg_mol": self.molar_mass_kg_mol,
            "reference_pressure_Pa": self.reference_pressure_Pa,
            "reference_temperature_K": self.reference_temperature_K,
        }


def aqueous_gibbs_thermo_from_dict(
    payload: Mapping[str, Any],
) -> RevisedHKFSpeciesThermo | IF97AqueousWaterThermo:
    data = dict(payload)
    model = data.pop("model")
    data["reference_basis"] = AqueousReferenceBasis(**data["reference_basis"])
    if model == "revised-HKF-IF97":
        return RevisedHKFSpeciesThermo(**data)
    if model == "aqueous-IF97-water":
        return IF97AqueousWaterThermo(**data)
    raise ValueError("unknown aqueous Gibbs standard-state model")
