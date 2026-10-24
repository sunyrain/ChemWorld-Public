"""Cubic-plus-association on the existing cubic and Wertheim expressions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

from chemworld.physchem.caloric_math import scalar_value
from chemworld.physchem.cubic_alpha import cubic_alpha_value
from chemworld.physchem.cubic_mixing import (
    GibbsExcessMixingSpec,
    cubic_residual_from_parameters,
    gibbs_excess_cubic_mixing,
    quadratic_cubic_mixing,
)
from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL
from chemworld.physchem.saft_association import SAFTAssociation, association_helmholtz


@dataclass(frozen=True)
class CPAComponent:
    component_id: str
    attraction_Pa_m6_mol2: float
    covolume_m3_mol: float
    critical_temperature_K: float
    alpha_coefficient: float
    source: str

    def __post_init__(self) -> None:
        if (not self.component_id or not self.source or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (
                self.attraction_Pa_m6_mol2, self.covolume_m3_mol, self.critical_temperature_K)
        ) or isinstance(self.alpha_coefficient, bool) or not isfinite(self.alpha_coefficient)):
            raise ValueError(
                "CPA components need positive SI parameters, finite alpha and provenance")


@dataclass(frozen=True)
class CPAModel:
    components: tuple[CPAComponent, ...]
    interaction: tuple[tuple[float, ...], ...]
    temperature_range_K: tuple[float, float]
    source: str
    association: SAFTAssociation | None = None
    cubic_model: str = "srk"
    radial_distribution: str = "Kontogeorgis"
    gas_constant_J_mol_K: float = R_J_PER_MOL_K
    ge_mixing: GibbsExcessMixingSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "interaction", tuple(tuple(row) for row in self.interaction))
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        n = len(self.components)
        if not n or len(set(self.component_ids)) != n or not self.source:
            raise ValueError("CPA requires unique component identities and provenance")
        if (self.cubic_model not in ("peng_robinson", "srk")
                or self.radial_distribution not in ("Kontogeorgis", "Carnahan-Starling")):
            raise ValueError("unknown CPA cubic or radial-distribution model")
        if len(self.interaction) != n or any(len(row) != n for row in self.interaction):
            raise ValueError("explicit square CPA interaction matrix required")
        for i, row in enumerate(self.interaction):
            for j, value in enumerate(row):
                if (isinstance(value, bool) or not isfinite(value) or value >= 1
                        or value != self.interaction[j][i] or (i == j and value != 0)):
                    raise ValueError("CPA interactions require symmetric finite k<1, zero diagonal")
        if (len(self.temperature_range_K) != 2 or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (*self.temperature_range_K, self.gas_constant_J_mol_K)
        ) or self.temperature_range_K[0] >= self.temperature_range_K[1]):
            raise ValueError("CPA requires an ordered positive temperature domain and R")
        if self.association and any(s.component_id not in self.component_ids
                                    for s in self.association.sites):
            raise ValueError("CPA association site refers to an undeclared component")
        if self.ge_mixing is not None:
            if self.ge_mixing.activity.component_ids != self.component_ids:
                raise ValueError("CPA and GE activity component order must agree")
            if any(v != 0 for row in self.interaction for v in row):
                raise ValueError("CPA GE uses its declared mixing; classical kij must be zero")

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(c.component_id for c in self.components)

    def contributions(self, temperature: Any, density: Any,
                      fractions: Sequence[Any]) -> dict[str, Any]:
        t, rho = scalar_value(temperature), scalar_value(density)
        xs = tuple(scalar_value(x) for x in fractions)
        if (not isfinite(t) or not self.temperature_range_K[0] <= t <= self.temperature_range_K[1]
                or not isfinite(rho) or rho < 0 or len(xs) != len(self.components)
                or any(not isfinite(x) or x < 0 for x in xs) or abs(sum(xs)-1) > 1e-10):
            raise ValueError(
                "CPA state outside declared temperature, density or composition domain")
        ai = tuple(c.attraction_Pa_m6_mol2*cubic_alpha_value(
            temperature, c.critical_temperature_K, model="soave",
            coefficients=(c.alpha_coefficient,))
            for c in self.components)
        bi = tuple(c.covolume_m3_mol for c in self.components)
        a, b = (quadratic_cubic_mixing(fractions, ai, bi, self.interaction)
                if self.ge_mixing is None else gibbs_excess_cubic_mixing(
                    self.ge_mixing, self.cubic_model, self.component_ids, fractions,
                    temperature, ai, bi, self.gas_constant_J_mol_K))
        result = {"cubic": cubic_residual_from_parameters(self.cubic_model, a, b, temperature,
                   density, self.gas_constant_J_mol_K)/(self.gas_constant_J_mol_K*temperature)}
        if self.association:
            eta = density*b/4
            contact = (1/(1-1.9*eta) if self.radial_distribution == "Kontogeorgis"
                       else (1-eta/2)/(1-eta)**3)
            n = len(xs)
            result["association"] = association_helmholtz(
                self.association, temperature, density*AVOGADRO_PER_MOL,
                self.component_ids, fractions, [[contact]*n for _ in range(n)])
        return result

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        return sum(self.contributions(temperature, density, fractions).values())

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld-cpa-v1", "components": [asdict(c) for c in self.components],
                "interaction": self.interaction, "temperature_range_K": self.temperature_range_K,
                "source": self.source, "cubic_model": self.cubic_model,
                "radial_distribution": self.radial_distribution,
                "gas_constant_J_mol_K": self.gas_constant_J_mol_K,
                "association": self.association.to_dict() if self.association else None,
                "ge_mixing": self.ge_mixing.to_dict() if self.ge_mixing else None}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CPAModel:
        data = dict(payload)
        if data.pop("model", None) != "chemworld-cpa-v1":
            raise ValueError("unknown CPA model encoding")
        data["components"] = tuple(CPAComponent(**c) for c in data["components"])
        if data.get("association") is not None:
            data["association"] = SAFTAssociation.from_dict(data["association"])
        if data.get("ge_mixing") is not None:
            data["ge_mixing"] = GibbsExcessMixingSpec.from_dict(data["ge_mixing"])
        return cls(**data)
