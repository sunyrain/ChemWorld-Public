"""Planar fundamental-measure hard spheres on the common analytic algebra.

Weighted densities are number-based SI: (n0 [m^-3], n1 [m^-2], n2 [m^-1],
n3 [-], nV1 [m^-2], nV2 [m^-1]). Vectors are their signed planar components.
This local functional consumes supplied weighted densities; it does not perform
spatial convolution, minimize a density profile, or introduce an evolution clock.
Equations: Rosenfeld1989; Roth et al.2002; Hansen-Goos/Roth2006; Yu/Wu2002.
ThermoPack source equations were inspected; no unseen-source claim is made.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, pi
from numbers import Real
from typing import Any

import numpy as np

from chemworld.physchem.caloric_math import logarithm, partial_derivative, scalar_value
from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL


def _polynomial(x: Any, coefficients: Sequence[float]) -> Any:
    result = x * 0
    for value in reversed(coefficients):
        result = result * x + value
    return result


def _coefficients(eta: Any, model: str) -> tuple[Any, Any]:
    """Pair and cubic coefficients, with analytic vacuum continuations."""
    inverse = 1 / (1 - eta)
    if model == "RF":
        return inverse, inverse**2 / (24 * pi)
    if scalar_value(eta) < 0.05:
        w = _polynomial(eta, [1.5] + [-2 / (j * (j + 1) * (j + 2)) for j in range(1, 26)])
        p2 = _polynomial(eta, [0, 0] + [2 / (3 * j * (j + 1)) for j in range(2, 26)])
        p3 = _polynomial(eta, [0, 4 / 9] + [-4 / (3 * j * (j + 1) * (j + 2)) for j in range(2, 26)])
    else:
        logarithm_free_volume = logarithm(1 - eta)
        w = (eta + (1 - eta) ** 2 * logarithm_free_volume) / eta**2
        p2 = (2 - eta + 2 * (1 - eta) * logarithm_free_volume / eta) / 3
        p3 = (2 * w - 3 + 2 * eta) / 3
    if model == "WB":
        return inverse, w * inverse**2 / (36 * pi)
    if model == "WBII":
        return (1 + p2) * inverse, (1 - p3) * inverse**2 / (24 * pi)
    raise ValueError("FMT model must be RF, WB or WBII")


def _weighted(values: Sequence[Any]) -> tuple[Any, ...]:
    if len(values) != 6:
        raise ValueError("planar FMT needs exactly six weighted densities")
    raw = tuple(scalar_value(v) for v in values)
    if any(not isfinite(v) for v in raw) or any(v < 0 for v in raw[:4]):
        raise ValueError("FMT scalar weighted densities must be finite and nonnegative")
    if raw[3] >= 1 or abs(raw[4]) > raw[1] or abs(raw[5]) > raw[2]:
        raise ValueError("FMT packing/vector weights outside physical domain")
    return tuple(values)


@dataclass(frozen=True)
class FundamentalMeasureFunctional:
    model: str = "WB"

    def __post_init__(self) -> None:
        if self.model not in {"RF", "WB", "WBII"}:
            raise ValueError("FMT model must be RF, WB or WBII")

    def energy_density(self, weighted_densities: Sequence[Any]) -> Any:
        """Phi = A_res/(kB*T*volume) in m^-3, not energy in joules."""
        n0, n1, n2, eta, v1, v2 = _weighted(weighted_densities)
        pair, cubic = _coefficients(eta, self.model)
        return (
            -n0 * logarithm(1 - eta) + (n1 * n2 - v1 * v2) * pair + (n2**3 - 3 * n2 * v2**2) * cubic
        )

    def evaluate_many(
        self,
        rows: Sequence[Sequence[float]],
        *,
        gradient: bool = False,
        hessian: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """Batched planar values and optional derivatives in weighted SI coordinates."""
        if type(gradient) is not bool or type(hessian) is not bool:
            raise ValueError("FMT derivative controls must be booleans")
        data = np.asarray(rows, dtype=float)
        if data.ndim != 2 or data.shape[1] != 6:
            raise ValueError("FMT batch shape must be (number of cells, 6)")
        values, gradients, hessians = [], [], []
        keys = tuple(f"n{i}" for i in range(6))
        for row in data:
            values.append(float(self.energy_density(row)))
            coordinates = dict(zip(keys, row, strict=True))

            def energy(v: Mapping[str, Any]) -> Any:
                return self.energy_density(tuple(v[k] for k in keys))

            if gradient:
                gradients.append([partial_derivative(energy, coordinates, (k,)) for k in keys])
            if hessian:
                hessians.append(
                    [[partial_derivative(energy, coordinates, (i, j)) for j in keys] for i in keys]
                )
        return (
            (np.asarray(values),)
            + ((np.asarray(gradients).reshape((-1, 6)),) if gradient else ())
            + ((np.asarray(hessians).reshape((-1, 6, 6)),) if hessian else ())
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld-planar-fmt-v1",
            "functional": self.model,
            "weighted_density_units": ["m^-3", "m^-2", "m^-1", "1", "m^-2", "m^-1"],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FundamentalMeasureFunctional:
        model = cls(str(payload["functional"]))
        if dict(payload) != model.to_dict():
            raise ValueError("unknown FMT schema or weighted-density units")
        return model


def fmt_contact_value(weighted_densities: Sequence[Any], pair_length_m: Any) -> Any:
    """Yu--Wu BMCSL contact; pair_length=d_i*d_j/(d_i+d_j), including jets."""
    _, _, n2, eta, _, v2 = _weighted(weighted_densities)
    if not isfinite(scalar_value(pair_length_m)) or scalar_value(pair_length_m) <= 0:
        raise ValueError("contact pair length must be finite and positive")
    if scalar_value(n2) == 0:
        if not all(isinstance(v, Real) for v in weighted_densities):
            raise ValueError("contact has no full weighted-density derivative at n2=0")
        return 1 / (1 - eta)
    xi = 1 - (v2 / n2) ** 2
    inverse = 1 / (1 - eta)
    return inverse + xi * n2 * pair_length_m * (
        inverse**2 / 2 + n2 * pair_length_m * inverse**3 / 18
    )


@dataclass(frozen=True)
class FundamentalMeasureBulkModel:
    """Uniform hard-sphere mixture consumes the same functional coefficients."""

    component_ids: tuple[str, ...]
    diameters_m: tuple[float, ...]
    source: str
    functional: str = "WB"
    gas_constant_J_mol_K: float = R_J_PER_MOL_K

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_ids", tuple(self.component_ids))
        object.__setattr__(self, "diameters_m", tuple(self.diameters_m))
        FundamentalMeasureFunctional(self.functional)
        if (
            not self.source
            or not self.component_ids
            or not all(self.component_ids)
            or len(set(self.component_ids)) != len(self.component_ids)
            or len(self.diameters_m) != len(self.component_ids)
            or any(
                isinstance(v, bool) or not isfinite(v) or v <= 0
                for v in (*self.diameters_m, self.gas_constant_J_mol_K)
            )
        ):
            raise ValueError("FMT bulk needs unique components, positive SI diameters and source")

    def _moments(self, density: Any, fractions: Sequence[Any]) -> tuple[Any, Any, Any, Any]:
        if len(fractions) != len(self.component_ids):
            raise ValueError("FMT bulk composition order must match components")
        xs = tuple(scalar_value(x) for x in fractions)
        if (
            not isfinite(scalar_value(density))
            or scalar_value(density) < 0
            or any(not isfinite(x) or x < 0 for x in xs)
            or abs(sum(xs) - 1) > 1e-10
        ):
            raise ValueError("FMT bulk needs nonnegative density and normalized composition")
        radii = tuple(d / 2 for d in self.diameters_m)
        h1 = sum(x * r for x, r in zip(fractions, radii, strict=True))
        h2 = sum(x * 4 * pi * r * r for x, r in zip(fractions, radii, strict=True))
        h3 = sum(x * 4 * pi * r**3 / 3 for x, r in zip(fractions, radii, strict=True))
        n = density * AVOGADRO_PER_MOL
        if scalar_value(n * h3) >= 1:
            raise ValueError("FMT bulk packing fraction must be less than one")
        return n, h1, h2, h3

    def weighted_densities(self, density: Any, fractions: Sequence[Any]) -> tuple[Any, ...]:
        n, h1, h2, h3 = self._moments(density, fractions)
        return n, n * h1, n * h2, n * h3, n * 0, n * 0

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        if not isfinite(scalar_value(temperature)) or scalar_value(temperature) <= 0:
            raise ValueError("FMT temperature must be finite and positive")
        n, h1, h2, h3 = self._moments(density, fractions)
        eta = n * h3
        pair, cubic = _coefficients(eta, self.functional)
        # Cancel n analytically to retain the zero-density Taylor coefficients.
        return -logarithm(1 - eta) + n * h1 * h2 * pair + n * n * h2**3 * cubic

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld-fmt-bulk-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FundamentalMeasureBulkModel:
        values = dict(payload)
        if values.pop("model", None) != "chemworld-fmt-bulk-v1":
            raise ValueError("unknown FMT bulk model schema")
        return cls(**values)
