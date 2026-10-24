"""Aasen quantum cubic PR with explicit Twu, beta and volume-shift inputs.

The empirical beta is a fitted covolume correction, not a Feynman-Hibbs
expansion. All residual responses use the common molecular Helmholtz protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from chemworld.physchem.caloric_math import compensated_sum, logarithm, scalar_value
from chemworld.physchem.cubic_alpha import cubic_alpha_value
from chemworld.physchem.cubic_mixing import (
    GibbsExcessMixingSpec,
    cubic_residual_from_parameters,
    gibbs_excess_cubic_mixing,
    quadratic_cubic_mixing,
)
from chemworld.physchem.eos import R_J_PER_MOL_K


@dataclass(frozen=True)
class QuantumCubicComponent:
    component_id: str
    attraction_Pa_m6_mol2: float
    covolume_m3_mol: float
    critical_temperature_K: float
    twu_coefficients: tuple[float, float, float]
    beta_xi_K: float
    beta_temperature_shift_K: float
    volume_translation_m3_mol: float
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "twu_coefficients", tuple(self.twu_coefficients))
        positive = (self.attraction_Pa_m6_mol2, self.covolume_m3_mol,
                    self.critical_temperature_K)
        finite = (*self.twu_coefficients, self.beta_xi_K,
                  self.beta_temperature_shift_K, self.volume_translation_m3_mol)
        if (not self.component_id or not self.source or len(self.twu_coefficients) != 3
                or any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in positive)
                or any(isinstance(v, bool) or not isfinite(v) for v in finite)
                or self.beta_xi_K < 0 or self.beta_temperature_shift_K < 0):
            raise ValueError("quantum cubic component needs explicit finite SI fit parameters")

    def covolume(self, temperature: Any) -> Any:
        """b(Tc)*[(1+xi/(T+shift))/(1+xi/(Tc+shift))]^3."""
        reference = 1+self.beta_xi_K/(self.critical_temperature_K+self.beta_temperature_shift_K)
        return self.covolume_m3_mol*((1+self.beta_xi_K/(
            temperature+self.beta_temperature_shift_K))/reference)**3


@dataclass(frozen=True)
class QuantumCubicModel:
    components: tuple[QuantumCubicComponent, ...]
    interaction: tuple[tuple[float, ...], ...]
    temperature_range_K: tuple[float, float]
    source: str
    gas_constant_J_mol_K: float = R_J_PER_MOL_K
    covolume_interaction: tuple[tuple[float, ...], ...] | None = None
    ge_mixing: GibbsExcessMixingSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "interaction", tuple(tuple(r) for r in self.interaction))
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        n = len(self.components)
        if not n or len(set(self.component_ids)) != n or not self.source:
            raise ValueError("quantum cubic needs unique components and provenance")
        if len(self.interaction) != n or any(len(r) != n for r in self.interaction):
            raise ValueError("quantum cubic needs an explicit square interaction matrix")
        for i, row in enumerate(self.interaction):
            for j, v in enumerate(row):
                if (isinstance(v, bool) or not isfinite(v) or v >= 1
                        or v != self.interaction[j][i] or (i == j and v != 0)):
                    raise ValueError("quantum cubic interaction symmetric k<1, zero diagonal")
        raw = self.covolume_interaction
        matrix = tuple(tuple(r) for r in raw) if raw is not None else ((0.,)*n,)*n
        if len(matrix) != n or any(len(r) != n for r in matrix):
            raise ValueError("quantum cubic covolume interactions need a square matrix")
        for i, row in enumerate(matrix):
            for j, v in enumerate(row):
                if (isinstance(v, bool) or not isfinite(v) or v >= 1
                        or v != matrix[j][i] or (i == j and v != 0)):
                    raise ValueError("quantum cubic covolume interactions symmetric <1, diagonal0")
        object.__setattr__(self, "covolume_interaction", matrix)
        if (len(self.temperature_range_K) != 2 or any(isinstance(v, bool)
                or not isfinite(v) or v <= 0 for v in (
                    *self.temperature_range_K, self.gas_constant_J_mol_K))
                or self.temperature_range_K[0] >= self.temperature_range_K[1]):
            raise ValueError("quantum cubic needs positive ordered temperature domain and R")
        if self.ge_mixing is not None:
            if self.ge_mixing.activity.component_ids != self.component_ids:
                raise ValueError("quantum cubic and GE component order must agree")
            if any(v != 0 for rows in (self.interaction, matrix) for row in rows for v in row):
                raise ValueError("quantum cubic GE requires zero classical kij/lij")

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(c.component_id for c in self.components)

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        t, rho = scalar_value(temperature), scalar_value(density)
        xs = tuple(scalar_value(v) for v in fractions)
        if (not isfinite(t) or not self.temperature_range_K[0] <= t <= self.temperature_range_K[1]
                or not isfinite(rho) or rho < 0 or len(xs) != len(self.components)
                or any(not isfinite(x) or x < 0 for x in xs) or abs(sum(xs)-1) > 1e-10):
            raise ValueError("quantum cubic state outside declared T/rho/composition domain")
        attraction = tuple(c.attraction_Pa_m6_mol2*cubic_alpha_value(
            temperature, c.critical_temperature_K, model="twu91", coefficients=c.twu_coefficients)
            for c in self.components)
        covolume = tuple(c.covolume(temperature) for c in self.components)
        a, b = (quadratic_cubic_mixing(fractions, attraction, covolume, self.interaction,
                covolume_interactions=self.covolume_interaction) if self.ge_mixing is None
                else gibbs_excess_cubic_mixing(self.ge_mixing, "peng_robinson", self.component_ids,
                    fractions, temperature, attraction, covolume, self.gas_constant_J_mol_K))
        shift = compensated_sum(x*c.volume_translation_m3_mol
            for x, c in zip(fractions, self.components, strict=True))
        volume_ratio = 1+density*shift
        if scalar_value(volume_ratio) <= 0:
            raise ValueError("translated cubic volume must be positive")
        eos_density = density/volume_ratio
        return cubic_residual_from_parameters("peng_robinson", a, b, temperature,
            eos_density, self.gas_constant_J_mol_K)/(self.gas_constant_J_mol_K*temperature
            )-logarithm(volume_ratio)

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld-quantum-cubic-v1", "components": [asdict(c)
                for c in self.components], "interaction": self.interaction,
                "temperature_range_K": self.temperature_range_K, "source": self.source,
                "gas_constant_J_mol_K": self.gas_constant_J_mol_K,
                "covolume_interaction": self.covolume_interaction,
                "ge_mixing": self.ge_mixing.to_dict() if self.ge_mixing else None}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> QuantumCubicModel:
        data = dict(payload)
        if data.pop("model", None) != "chemworld-quantum-cubic-v1":
            raise ValueError("unknown quantum cubic model encoding")
        data["components"] = tuple(QuantumCubicComponent(**c) for c in data["components"])
        if data.get("ge_mixing") is not None:
            data["ge_mixing"] = GibbsExcessMixingSpec.from_dict(data["ge_mixing"])
        return cls(**data)
