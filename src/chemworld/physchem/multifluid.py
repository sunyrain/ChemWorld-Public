"""Explicit homogeneous multifluid Helmholtz mixtures on the common EOS API.

GERG reducing/departure expressions own parameters only. State, chemical
potential and density inversion use the existing molecular EOS consumer.
A constant R convention is compulsory; source R values are never overwritten.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from math import isfinite, sqrt
from sys import float_info
from typing import Any

from chemworld.physchem.caloric_math import (
    compensated_sum,
    derivative_depth,
    exponential,
    partial_derivative,
    scalar_value,
)
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.molecular_eos import (
    MolecularEOSState,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
)
from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularCaloricState


def _finite(value: Any, label: str, *, positive: bool = False) -> None:
    v = scalar_value(value)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v):
        raise ValueError(f"{label} must be a finite real number")
    if positive and v <= 0:
        raise ValueError(f"{label} must be positive")


def _strict(data: Mapping[str, Any], fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(data, Mapping) or set(data) != fields:
        raise ValueError(f"{label} requires exactly {sorted(fields)}")
    return dict(data)


@dataclass(frozen=True)
class MultifluidDepartureTerm:
    """One power/exponential or GERG density-Gaussian departure contribution.

    GERG's last exponent is linear in delta-gamma, unlike the pure-fluid
    Gaussian term whose second exponent is quadratic in tau-gamma.
    """

    kind: str
    n: float
    d: float
    t: float
    l: float = 0.0  # noqa: E741 - published departure damping exponent
    eta: float = 0.0
    epsilon: float = 0.0
    beta: float = 0.0
    gamma: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in {"power", "gerg"}:
            raise ValueError("unsupported multifluid departure kind")
        for k, v in asdict(self).items():
            if k != "kind":
                _finite(v, k)
        if self.d <= 0 or self.d != int(self.d) or self.l < 0 or self.l != int(self.l):
            raise ValueError(
                "departure density power must be positive integer; l nonnegative integer"
            )
        if self.kind == "power" and any((self.eta, self.epsilon, self.beta, self.gamma)):
            raise ValueError("power departure has no GERG coefficients")
        if self.kind == "gerg" and self.l:
            raise ValueError("GERG departure has no power damping l")

    def evaluate(self, delta: Any, tau: Any) -> Any:
        exponent = (
            (-(delta ** int(self.l)) if self.l else 0.0)
            if self.kind == "power"
            else (-self.eta * (delta - self.epsilon) ** 2 - self.beta * (delta - self.gamma))
        )
        return self.n * delta ** int(self.d) * tau**self.t * exponential(exponent)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultifluidDepartureTerm:
        return cls(**_strict(data, set(cls.__dataclass_fields__), "departure term"))


@dataclass(frozen=True)
class MultifluidBinaryPair:
    """An oriented source pair; reversing identities requires reciprocal betas."""

    component_ids: tuple[str, str]
    beta_T: float
    gamma_T: float
    beta_v: float
    gamma_v: float
    F: float
    departure_terms: tuple[MultifluidDepartureTerm, ...]
    source: str

    def __post_init__(self) -> None:
        ids = tuple(self.component_ids)
        if len(ids) != 2 or len(set(ids)) != 2 or any(not isinstance(s, str) or not s for s in ids):
            raise ValueError("binary pair requires two distinct component identities")
        for k in ("beta_T", "gamma_T", "beta_v", "gamma_v"):
            _finite(getattr(self, k), k, positive=True)
        _finite(self.F, "F")
        terms = tuple(self.departure_terms)
        if any(not isinstance(t, MultifluidDepartureTerm) for t in terms):
            raise ValueError("binary departure terms must be native declarations")
        if self.F != 0 and not terms:
            raise ValueError("nonzero F requires explicit departure terms")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("binary pair requires provenance")
        object.__setattr__(self, "component_ids", ids)
        object.__setattr__(self, "departure_terms", terms)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["component_ids"] = list(self.component_ids)
        result["departure_terms"] = [t.to_dict() for t in self.departure_terms]
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultifluidBinaryPair:
        values = _strict(data, set(cls.__dataclass_fields__), "binary pair")
        values["departure_terms"] = tuple(
            MultifluidDepartureTerm.from_dict(t) for t in values["departure_terms"]
        )
        return cls(**values)


def _pair_weight(x: Any, y: Any, beta: float) -> Any:
    if beta == 1:
        return x * y
    denominator = beta * beta * x + y
    if scalar_value(denominator) == 0:
        if max(derivative_depth(x), derivative_depth(y)) > 1:
            raise ValueError(
                "GERG second composition derivative at two absent components is undefined"
            )
        # Degree two, hence value and first insertion derivative both vanish.
        return 0.0 * x * y
    return x * y * (x + y) / denominator


@dataclass(frozen=True)
class MultifluidModel:
    fluids: tuple[HelmholtzFluid, ...]
    binary_pairs: tuple[MultifluidBinaryPair, ...]
    gas_constant_J_mol_K: float
    gas_constant_convention: str
    source: str

    def __post_init__(self) -> None:
        fluids, pairs = tuple(self.fluids), tuple(self.binary_pairs)
        if not fluids or any(type(f) is not HelmholtzFluid or f.pseudo_pure for f in fluids):
            raise ValueError(
                "multifluid requires explicit pure Helmholtz fluids, not pseudo-pure models"
            )
        ids = tuple(f.fluid_id for f in fluids)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate multifluid component identity")
        if any(not isinstance(p, MultifluidBinaryPair) for p in pairs):
            raise ValueError("invalid multifluid binary declaration")
        actual = [frozenset(p.component_ids) for p in pairs]
        required = {frozenset((a, b)) for i, a in enumerate(ids) for b in ids[i + 1 :]}
        if len(set(actual)) != len(actual) or set(actual) != required:
            raise ValueError(
                "exactly one explicit binary pair is required for every component pair"
            )
        _finite(self.gas_constant_J_mol_K, "gas constant", positive=True)
        if self.gas_constant_convention not in {"source_equal", "common_constant_override"}:
            raise ValueError(
                "explicit constant gas-constant convention required; weighted R unsupported"
            )
        if self.gas_constant_convention == "source_equal" and any(
            f.gas_constant_J_mol_K != self.gas_constant_J_mol_K for f in fluids
        ):
            raise ValueError(
                "source gas constants differ; an explicit common_constant_override is required"
            )
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("multifluid source is required")
        if max(f.minimum_temperature_K for f in fluids) >= min(
            f.maximum_temperature_K for f in fluids
        ):
            raise ValueError("pure-fluid temperature domains do not intersect")
        object.__setattr__(self, "fluids", fluids)
        object.__setattr__(self, "binary_pairs", pairs)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(f.fluid_id for f in self.fluids)

    def _fractions(self, fractions: Sequence[Any]) -> tuple[Any, ...]:
        xs = tuple(fractions)
        if len(xs) != len(self.fluids):
            raise ValueError("mole-fraction length differs from component count")
        for x in xs:
            _finite(x, "mole fraction")
            if scalar_value(x) < 0:
                raise ValueError("mole fractions must be nonnegative")
        if abs(sum(scalar_value(x) for x in xs) - 1) > 1e-12:
            raise ValueError("mole fractions must sum to one; no implicit normalization")
        return xs

    def reducing(self, fractions: Sequence[Any]) -> tuple[Any, Any]:
        """Return reducing T [K] and reducing molar density [mol/m3]."""
        xs = self._fractions(fractions)
        t_terms = [x * x * f.reducing_temperature_K for x, f in zip(xs, self.fluids, strict=True)]
        v_terms = [
            x * x / f.reducing_molar_density_mol_m3 for x, f in zip(xs, self.fluids, strict=True)
        ]
        indexes = {k: i for i, k in enumerate(self.component_ids)}
        for p in self.binary_pairs:
            i, j = (indexes[k] for k in p.component_ids)
            fi, fj = self.fluids[i], self.fluids[j]
            t_terms.append(
                2
                * p.beta_T
                * p.gamma_T
                * sqrt(fi.reducing_temperature_K * fj.reducing_temperature_K)
                * _pair_weight(xs[i], xs[j], p.beta_T)
            )
            volume = (
                fi.reducing_molar_density_mol_m3 ** (-1 / 3)
                + fj.reducing_molar_density_mol_m3 ** (-1 / 3)
            ) ** 3 / 8
            v_terms.append(2 * p.beta_v * p.gamma_v * volume * _pair_weight(xs[i], xs[j], p.beta_v))
        return compensated_sum(t_terms), 1 / compensated_sum(v_terms)

    def _inputs(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> tuple[Any, ...]:
        _finite(temperature, "temperature", positive=True)
        _finite(density, "density", positive=True)
        if (
            not max(f.minimum_temperature_K for f in self.fluids)
            <= scalar_value(temperature)
            <= min(f.maximum_temperature_K for f in self.fluids)
        ):
            raise ValueError("temperature outside common pure-source validity")
        return self._fractions(fractions)

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        xs = self._inputs(temperature, density, fractions)
        tr, rhor = self.reducing(xs)
        delta, tau = density / rhor, tr / temperature
        terms = [x * f.alpha("residual", delta, tau) for x, f in zip(xs, self.fluids, strict=True)]
        indexes = {k: i for i, k in enumerate(self.component_ids)}
        for p in self.binary_pairs:
            i, j = (indexes[k] for k in p.component_ids)
            terms.append(
                xs[i]
                * xs[j]
                * p.F
                * compensated_sum(term.evaluate(delta, tau) for term in p.departure_terms)
            )
        return compensated_sum(terms)

    def state(
        self,
        temperature_K: float,
        molar_density_mol_m3: float,
        mole_fractions: Sequence[float],
        *,
        require_stable: bool = True,
    ) -> MolecularEOSState:
        return molecular_eos_state(
            self, temperature_K, molar_density_mol_m3, mole_fractions, require_stable=require_stable
        )

    def validate_state(self, state: MolecularEOSState) -> None:
        """Source pressure-domain guard called by every common state consumer."""
        self.validate_pressure(state.pressure_Pa)

    def validate_pressure(self, pressure_Pa: float) -> None:
        """Apply the source ceiling even when a critical response has singular Cp."""
        _finite(pressure_Pa, "pressure")
        if pressure_Pa > min(f.maximum_pressure_Pa for f in self.fluids):
            raise ValueError("pressure outside common pure-source validity")

    def composition_derivative(
        self,
        temperature_K: float,
        molar_density_mol_m3: float,
        mole_fractions: Sequence[float],
        coordinates: Sequence[str],
        *,
        dependent_component: str | None = None,
    ) -> float:
        """Simplex derivatives of alpha_r at fixed T/rho, order 1..3.

        Each coordinate replaces an equal fraction of the dependent component
        (default last). They are not unconstrained mole-fraction derivatives.
        """
        xs = self._inputs(temperature_K, molar_density_mol_m3, mole_fractions)
        dependent = dependent_component or self.component_ids[-1]
        coords = tuple(coordinates)
        if (
            dependent not in self.component_ids
            or not 1 <= len(coords) <= 3
            or any(c not in self.component_ids or c == dependent for c in coords)
        ):
            raise ValueError("simplex derivative requires 1..3 independent component coordinates")
        values = dict(zip(self.component_ids, xs, strict=True))
        values.pop(dependent)

        def fn(v: Mapping[str, Any]) -> Any:
            all_values = {**v, dependent: 1 - sum(v.values())}
            return self.alphar(
                temperature_K,
                molar_density_mol_m3,
                tuple(all_values[k] for k in self.component_ids),
            )

        return partial_derivative(fn, values, coords)

    def residual_chemical_potential_jacobian(
        self,
        temperature_K: float,
        molar_density_mol_m3: float,
        mole_fractions: Sequence[float],
    ) -> tuple[tuple[float, ...], ...]:
        """d mu_i^r / d rho_j at fixed T and other component densities [J m3/mol2]."""
        xs = self._inputs(temperature_K, molar_density_mol_m3, mole_fractions)
        values = {k: x * molar_density_mol_m3 for k, x in zip(self.component_ids, xs, strict=True)}

        def density_energy(v: Mapping[str, Any]) -> Any:
            rho = sum(v.values())
            return (
                rho
                * self.gas_constant_J_mol_K
                * temperature_K
                * self.alphar(temperature_K, rho, tuple(v[k] / rho for k in self.component_ids))
            )

        return tuple(
            tuple(partial_derivative(density_energy, values, (i, j)) for j in self.component_ids)
            for i in self.component_ids
        )

    def caloric_model(self) -> MolecularCaloricModel:
        """Reuse the common absolute-state/inventory consumer with explicit ideal refs.

        Original pure source constants stay in ``fluids`` and the codec. Only
        these derived ideal references apply the declared common-R convention.
        Their mathematical ideal pressure is not a physical source-state
        pressure; the residual model validates the actual mixture pressure.
        The original source T domain and finite-state checks still apply.
        Nonunit ideal prefactors remain unsupported by the common consumer.
        """
        references = tuple(
            replace(
                fluid,
                residual_terms=(),
                gas_constant_J_mol_K=self.gas_constant_J_mol_K,
                maximum_pressure_Pa=float_info.max,
                source=(
                    f"{fluid.source}; derived ideal reference; "
                    f"{self.gas_constant_convention}; "
                    f"original R={fluid.gas_constant_J_mol_K!r}; "
                    f"common R={self.gas_constant_J_mol_K!r}; "
                    f"original p_max={fluid.maximum_pressure_Pa!r}; "
                    "mathematical ideal contribution only, not a complete source fluid; "
                    "actual mixture pressure checked by residual model"
                ),
            )
            for fluid in self.fluids
        )
        return MolecularCaloricModel(self, references)

    def caloric_state(
        self,
        temperature_K: float,
        molar_density_mol_m3: float,
        mole_fractions: Sequence[float],
    ) -> MolecularCaloricState:
        """Absolute state of a locally stable homogeneous mixture, via shared consumer."""
        return self.caloric_model().state(temperature_K, molar_density_mol_m3, mole_fractions)

    def residual_chemical_potentials(
        self, temperature_K: float, molar_density_mol_m3: float, mole_fractions: Sequence[float]
    ) -> tuple[float, ...]:
        return molecular_residual_chemical_potentials(
            self, temperature_K, molar_density_mol_m3, mole_fractions
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld_multifluid_v1",
            "fluids": [f.to_dict() for f in self.fluids],
            "binary_pairs": [p.to_dict() for p in self.binary_pairs],
            "gas_constant_J_mol_K": self.gas_constant_J_mol_K,
            "gas_constant_convention": self.gas_constant_convention,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultifluidModel:
        values = _strict(data, set(cls.__dataclass_fields__) | {"model"}, "multifluid model")
        if values.pop("model") != "chemworld_multifluid_v1":
            raise ValueError("unknown multifluid encoding")
        fluids = []
        for declaration in values["fluids"]:
            fluid = HelmholtzFluid.from_dict(declaration)
            for key in ("ideal_terms", "residual_terms"):
                for raw, parsed in zip(declaration[key], getattr(fluid, key), strict=True):
                    if set(raw) - (set(parsed.to_dict()) | {"reference"}):
                        raise ValueError("unknown canonical pure-fluid term parameter")
            fluids.append(fluid)
        values["fluids"] = tuple(fluids)
        values["binary_pairs"] = tuple(
            MultifluidBinaryPair.from_dict(p) for p in values["binary_pairs"]
        )
        return cls(**values)
