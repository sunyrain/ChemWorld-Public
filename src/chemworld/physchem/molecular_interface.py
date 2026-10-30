"""Planar square-gradient interfaces on the common molecular free energy.

Bulk endpoints must coexist. Interior profiles may cross spinodals: their free
energy is evaluated directly, without homogeneous stability filtering. Constant
or temperature-dependent influence matrices are explicit. Declared semidefinite
factors use algebraic stationarity in their exact nullspace; no epsilon is added.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from math import fsum, isfinite, log, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.linalg import eig_banded
from scipy.optimize import brentq, minimize
from scipy.sparse import bmat, diags, lil_matrix

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.molecular_eos import (
    MolecularEOSState,
    ResidualHelmholtzModel,
    molecular_residual_chemical_potentials,
)
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    solve_root,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SquareGradientInfluence:
    component_ids: tuple[str, ...]
    coefficients_J_m5_mol2: tuple[tuple[float, ...], ...]
    temperature_range_K: tuple[float, float]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_ids", tuple(self.component_ids))
        object.__setattr__(
            self, "coefficients_J_m5_mol2", tuple(tuple(row) for row in self.coefficients_J_m5_mol2)
        )
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        c = np.asarray(self.coefficients_J_m5_mol2)
        n = len(self.component_ids)
        if (
            not n
            or len(set(self.component_ids)) != n
            or not self.source
            or c.shape != (n, n)
            or not np.all(np.isfinite(c))
            or not np.allclose(c, c.T, rtol=1e-13, atol=0)
            or np.min(np.linalg.eigvalsh(c)) <= 0
        ):
            raise ValueError("influence matrix must be finite, symmetric and positive definite")
        if (
            len(self.temperature_range_K) != 2
            or not 0 < self.temperature_range_K[0] < self.temperature_range_K[1]
            or not all(isfinite(v) for v in self.temperature_range_K)
        ):
            raise ValueError("influence parameters require a positive temperature domain")

    def to_dict(self) -> dict[str, Any]:
        return {"model": "constant-square-gradient-influence-v1", **asdict(self)}

    def matrix_at(self, temperature_K: float) -> tuple[tuple[float, ...], ...]:
        if not self.temperature_range_K[0] <= temperature_K <= self.temperature_range_K[1]:
            raise ValueError("influence temperature is outside its declared domain")
        return self.coefficients_J_m5_mol2

    @property
    def declared_rank(self) -> int:
        return len(self.component_ids)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SquareGradientInfluence:
        values = dict(data)
        if values.pop("model", None) != "constant-square-gradient-influence-v1":
            raise ValueError("unknown influence declaration")
        return cls(**values)


@dataclass(frozen=True)
class SingularSquareGradientInfluence:
    """Exact C=F F.T with an explicitly declared nonzero, deficient rank.

    F has n component rows and r independent columns, 0<r<n. Its units are the
    square root of J m5 mol-2. The missing gradient directions are algebraic
    thermodynamic degrees of freedom, not an omitted contribution to C.
    """

    component_ids: tuple[str, ...]
    gradient_factors_sqrt_J_m5_mol2: tuple[tuple[float, ...], ...]
    temperature_range_K: tuple[float, float]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_ids", tuple(self.component_ids))
        object.__setattr__(
            self,
            "gradient_factors_sqrt_J_m5_mol2",
            tuple(tuple(row) for row in self.gradient_factors_sqrt_J_m5_mol2),
        )
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        f = np.asarray(self.gradient_factors_sqrt_J_m5_mol2, dtype=float)
        n = len(self.component_ids)
        if (
            not n
            or len(set(self.component_ids)) != n
            or not self.source
            or f.ndim != 2
            or f.shape[0] != n
            or not 0 < f.shape[1] < n
            or not np.all(np.isfinite(f))
            or np.linalg.matrix_rank(f) != f.shape[1]
        ):
            raise ValueError("singular influence requires n by r independent factors, 0<r<n")
        if (
            len(self.temperature_range_K) != 2
            or not all(isfinite(v) for v in self.temperature_range_K)
            or not 0 < self.temperature_range_K[0] < self.temperature_range_K[1]
        ):
            raise ValueError("influence parameters require a positive temperature domain")

    @property
    def declared_rank(self) -> int:
        return len(self.gradient_factors_sqrt_J_m5_mol2[0])

    def matrix_at(self, temperature_K: float) -> tuple[tuple[float, ...], ...]:
        if not self.temperature_range_K[0] <= temperature_K <= self.temperature_range_K[1]:
            raise ValueError("influence temperature is outside its declared domain")
        f = np.asarray(self.gradient_factors_sqrt_J_m5_mol2, dtype=float)
        return tuple(tuple(map(float, row)) for row in f @ f.T)

    def to_dict(self) -> dict[str, Any]:
        return {"model": "factor-singular-square-gradient-influence-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SingularSquareGradientInfluence:
        values = dict(data)
        if values.pop("model", None) != "factor-singular-square-gradient-influence-v1":
            raise ValueError("unknown singular influence declaration")
        return cls(**values)


@dataclass(frozen=True)
class PolynomialSquareGradientInfluence:
    """ci(T)=sum a_ik theta**k and Cij=(1-kij)*sqrt(ci*cj).

    theta=(T-reference_temperature_K)/temperature_scale_K is dimensionless;
    coefficient rows are ascending powers, all in J m5 mol-2. Declared rank
    None requires positive definiteness. A deficient rank is an explicit
    semidefinite model declaration, checked without diagonal regularization.
    """

    component_ids: tuple[str, ...]
    pure_coefficients_J_m5_mol2: tuple[tuple[float, ...], ...]
    interaction: tuple[tuple[float, ...], ...]
    temperature_range_K: tuple[float, float]
    source: str
    reference_temperature_K: float = 0.0
    temperature_scale_K: float = 1.0
    declared_rank: int | None = None

    def __post_init__(self) -> None:
        for name in ("component_ids", "temperature_range_K"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("pure_coefficients_J_m5_mol2", "interaction"):
            object.__setattr__(self, name, tuple(tuple(row) for row in getattr(self, name)))
        n = len(self.component_ids)
        interaction = np.asarray(self.interaction, dtype=float)
        rank = n if self.declared_rank is None else self.declared_rank
        if (
            not n
            or len(set(self.component_ids)) != n
            or not self.source
            or len(self.pure_coefficients_J_m5_mol2) != n
            or any(
                not row or not all(isfinite(v) for v in row)
                for row in self.pure_coefficients_J_m5_mol2
            )
            or interaction.shape != (n, n)
            or not np.all(np.isfinite(interaction))
            or not np.allclose(interaction, interaction.T, rtol=0, atol=1e-14)
            or np.any(np.diag(interaction) != 0)
            or type(rank) is not int
            or not 1 <= rank <= n
        ):
            raise ValueError(
                "polynomial influence requires explicit coefficients, symmetric kij and rank"
            )
        if (
            len(self.temperature_range_K) != 2
            or not all(isfinite(v) for v in self.temperature_range_K)
            or not 0 < self.temperature_range_K[0] < self.temperature_range_K[1]
            or not isfinite(self.reference_temperature_K)
            or not isfinite(self.temperature_scale_K)
            or self.temperature_scale_K <= 0
        ):
            raise ValueError("polynomial influence temperature coordinates are invalid")
        self._check_matrix(1 - interaction, rank)
        low, high = (
            (t - self.reference_temperature_K) / self.temperature_scale_K
            for t in self.temperature_range_K
        )
        for row in self.pure_coefficients_J_m5_mol2:
            derivative = np.polynomial.polynomial.polyder(row)
            roots = np.polynomial.polynomial.polyroots(derivative)
            if not np.all(np.isfinite(roots)):
                raise ValueError("influence polynomial stationary points are unresolved")
            points = [low, high]
            points.extend(
                float(root.real)
                for root in roots
                if abs(root.imag) <= 1e-9 * max(1.0, abs(root.real)) and low <= root.real <= high
            )
            scale = sum(abs(a) * max(1.0, abs(low), abs(high)) ** k for k, a in enumerate(row))
            margin = 128 * np.finfo(float).eps * scale
            if (
                min(float(np.polynomial.polynomial.polyval(point, row)) for point in points)
                <= margin
            ):
                raise ValueError(
                    "pure influence must remain strictly positive over its temperature domain"
                )

    @staticmethod
    def _check_matrix(matrix: FloatArray, rank: int) -> None:
        values = np.linalg.eigvalsh(matrix)
        scale = float(np.max(np.abs(values)))
        tolerance = 64 * np.finfo(float).eps * len(values) * scale
        if rank == len(values):
            if values[0] <= 0:
                raise ValueError("undeclared singular influence is not positive definite")
        elif (
            values[-rank] <= tolerance
            or np.max(np.abs(values[:-rank])) > tolerance
            or values[0] < -tolerance
        ):
            raise ValueError(
                "influence matrix does not have the explicitly declared semidefinite rank"
            )

    def matrix_at(self, temperature_K: float) -> tuple[tuple[float, ...], ...]:
        if not self.temperature_range_K[0] <= temperature_K <= self.temperature_range_K[1]:
            raise ValueError("influence temperature is outside its declared domain")
        theta = (temperature_K - self.reference_temperature_K) / self.temperature_scale_K
        pure = np.array(
            [
                np.polynomial.polynomial.polyval(theta, row)
                for row in self.pure_coefficients_J_m5_mol2
            ]
        )
        if not np.all(np.isfinite(pure)) or np.any(pure <= 0):
            raise ValueError(
                "pure influence polynomial must be positive at the requested temperature"
            )
        matrix = (1 - np.asarray(self.interaction)) * np.sqrt(pure[:, None] * pure[None, :])
        self._check_matrix(matrix, len(pure) if self.declared_rank is None else self.declared_rank)
        return tuple(tuple(map(float, row)) for row in matrix)

    def to_dict(self) -> dict[str, Any]:
        return {"model": "polynomial-square-gradient-influence-v1", **asdict(self)}

    def gradient_factors_at(self, temperature_K: float) -> FloatArray:
        matrix = np.asarray(self.matrix_at(temperature_K))
        pure = np.sqrt(np.diag(matrix))
        values, vectors = np.linalg.eigh(1 - np.asarray(self.interaction))
        rank = len(pure) if self.declared_rank is None else self.declared_rank
        return pure[:, None] * vectors[:, -rank:] * np.sqrt(values[-rank:])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolynomialSquareGradientInfluence:
        values = dict(data)
        if values.pop("model", None) != "polynomial-square-gradient-influence-v1":
            raise ValueError("unknown polynomial influence declaration")
        return cls(**values)


MolecularInfluence = (
    SquareGradientInfluence | SingularSquareGradientInfluence | PolynomialSquareGradientInfluence
)


def molecular_influence_from_dict(data: dict[str, Any]) -> MolecularInfluence:
    """Decode only the three explicit influence declarations."""
    kind = data.get("model")
    if kind == "constant-square-gradient-influence-v1":
        return SquareGradientInfluence.from_dict(data)
    if kind == "factor-singular-square-gradient-influence-v1":
        return SingularSquareGradientInfluence.from_dict(data)
    if kind == "polynomial-square-gradient-influence-v1":
        return PolynomialSquareGradientInfluence.from_dict(data)
    raise ValueError("unknown molecular influence declaration")


@dataclass(frozen=True)
class MolecularInterfaceBulk:
    helmholtz_density_J_m3: float
    chemical_potentials_J_mol: tuple[float | None, ...]
    pressure_Pa: float


@dataclass(frozen=True)
class MolecularInterfaceResult:
    success: bool
    surface_tension_N_m: float | None = None
    positions_m: tuple[float, ...] = ()
    component_densities_mol_m3: tuple[tuple[float, ...], ...] = ()
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        scope = (
            "planar_semidefinite_differential_algebraic_square_gradient"
            if self.diagnostics.get("gradient_theory") == "semidefinite_differential_algebraic"
            else "planar_constant_positive_definite_square_gradient"
        )
        return {
            **asdict(self),
            "scope": scope,
            "profile_orientation": "rows=positions; columns=model_component_order",
        }


def _density(
    model: ResidualHelmholtzModel, temperature: float, concentrations: Sequence[float]
) -> FloatArray:
    c = np.asarray(concentrations, dtype=float)
    if (
        c.shape != (len(model.component_ids),)
        or not np.all(np.isfinite(c))
        or np.any(c < 0)
        or c.sum() <= 0
        or not isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("molecular interface needs positive T and nonnegative nonempty densities")
    return c


def _free_energy(model: ResidualHelmholtzModel, temperature: float, c: FloatArray) -> float:
    rho = float(c.sum())
    ideal = fsum(float(v) * (log(float(v)) - 1) for v in c if v > 0)
    residual = rho * float(model.alphar(temperature, rho, tuple(float(v / rho) for v in c)))
    return model.gas_constant_J_mol_K * temperature * fsum((ideal, residual))


def molecular_interface_bulk(
    model: ResidualHelmholtzModel, temperature_K: float, component_densities_mol_m3: Sequence[float]
) -> MolecularInterfaceBulk:
    """Free energy in the c_ref=1 mol/m3 ideal gauge; absent ideal mu is undefined.

    Species-linear reference terms cancel in the coexistence grand potential.
    No absolute caloric reference or local mechanical stability is required.
    """
    c = _density(model, temperature_K, component_densities_mol_m3)
    rho, rt = float(c.sum()), model.gas_constant_J_mol_K * temperature_K
    residual_mu = molecular_residual_chemical_potentials(
        model, temperature_K, rho, tuple(float(v / rho) for v in c)
    )
    mu = tuple(None if v == 0 else rt * log(float(v)) + residual_mu[i] for i, v in enumerate(c))
    f = _free_energy(model, temperature_K, c)
    p = fsum(float(v) * float(m) for v, m in zip(c, mu, strict=True) if m is not None) - f
    return MolecularInterfaceBulk(f, mu, p)


def _coexistence(
    model: ResidualHelmholtzModel,
    left: MolecularEOSState,
    right: MolecularEOSState,
    influence: MolecularInfluence,
) -> dict[str, Any]:
    if influence.component_ids != model.component_ids:
        raise ValueError("influence and EOS component identities must agree")
    t = left.temperature_K
    if (
        abs(t - right.temperature_K) > 1e-9
        or not influence.temperature_range_K[0] <= t <= influence.temperature_range_K[1]
    ):
        raise ValueError("interface endpoints require one temperature inside the influence domain")
    c1 = np.array(left.mole_fractions) * left.molar_density_mol_m3
    c2 = np.array(right.mole_fractions) * right.molar_density_mol_m3
    if np.any((c1 > 0) != (c2 > 0)):
        raise ValueError("interface endpoints require the same exact component support")
    active = np.flatnonzero(c1 > 0)
    if not len(active):
        raise ValueError("interface endpoints require nonempty active component support")
    if np.max(np.abs(c1[active] - c2[active]) / np.maximum(c1[active], c2[active])) < 1e-7:
        raise ValueError("interface endpoints must be distinct phases")
    a, b = (molecular_interface_bulk(model, t, tuple(c)) for c in (c1, c2))
    rt = model.gas_constant_J_mol_K * t
    mu = np.array([float(a.chemical_potentials_J_mol[i]) for i in active])
    other_mu = np.array([float(b.chemical_potentials_J_mol[i]) for i in active])
    mu_error = float(np.max(np.abs(mu - other_mu)) / rt)
    pressure_error = abs(a.pressure_Pa - b.pressure_Pa) / max(abs(a.pressure_Pa), 1.0)
    if (
        not left.mechanically_stable
        or not right.mechanically_stable
        or a.pressure_Pa <= 0
        or b.pressure_Pa <= 0
        or mu_error > 1e-6
        or pressure_error > 1e-7
        or any(
            abs(state.pressure_Pa - bulk.pressure_Pa) > max(1e-4, 1e-7 * abs(bulk.pressure_Pa))
            for state, bulk in ((left, a), (right, b))
        )
    ):
        raise ValueError("interface endpoints do not coexist on the declared free energy")
    full_matrix = np.asarray(influence.matrix_at(t), dtype=float)
    matrix = full_matrix[np.ix_(active, active)]
    declared = influence.declared_rank
    full_rank = declared is None or declared == len(model.component_ids)
    nullspace: FloatArray
    if full_rank:
        rank = len(active)
        nullspace = np.empty((len(active), 0))
    else:
        if declared is None:
            raise AssertionError("invariant violated: declared is not None")
        if isinstance(influence, SingularSquareGradientInfluence):
            factors = np.asarray(influence.gradient_factors_sqrt_J_m5_mol2)
        else:
            if not isinstance(influence, PolynomialSquareGradientInfluence):
                raise AssertionError(
                    "invariant violated: influence must be PolynomialSquareGradientInfluence"
                )
            factors = influence.gradient_factors_at(t)
        restricted = factors[active]
        rank = (
            int(declared)
            if len(active) == len(model.component_ids)
            else int(np.linalg.matrix_rank(restricted))
        )
        if len(active) == len(model.component_ids) and np.linalg.matrix_rank(restricted) != rank:
            raise ValueError("declared gradient rank is numerically unresolved")
        vectors, singular_values, _ = np.linalg.svd(restricted, full_matrices=True)
        if (
            rank
            and singular_values[rank - 1]
            <= 64 * np.finfo(float).eps * len(active) * singular_values[0]
        ):
            raise ValueError("declared active gradient rank is numerically unresolved")
        nullspace = vectors[:, rank:]
    if rank < 1:
        raise ValueError("active interface support has no gradient energy")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if nullspace.shape[1]:
        endpoint_curvature = min(
            float(np.linalg.eigvalsh(nullspace.T @ _hessian(model, t, c, active) @ nullspace)[0])
            for c in (c1, c2)
        )
        if endpoint_curvature <= 0:
            raise ValueError("singular influence endpoints lack a stable algebraic nullspace")
    separation = float(np.max(np.abs(c1[active] - c2[active]) / np.maximum(c1[active], c2[active])))
    return {
        "temperature": t,
        "left": c1,
        "right": c2,
        "active": active,
        "mu": mu,
        "reference_f": a.helmholtz_density_J_m3,
        "RT": rt,
        "C": matrix,
        "rank": rank,
        "nullspace": nullspace,
        "stable_grand_response": separation < 0.05,
        "diagnostics": {
            "endpoint_mu_error_RT": mu_error,
            "endpoint_pressure_relative_error": pressure_error,
            "coexistence_pressure_Pa": a.pressure_Pa,
            "active_components": len(active),
            "influence_rank": rank,
            "influence_eigenvalues_J_m5_mol2": tuple(map(float, eigenvalues)),
            "influence_temperature_K": t,
            "gradient_theory": "positive_definite"
            if rank == len(active)
            else "semidefinite_differential_algebraic",
            "relative_endpoint_separation": separation,
            "grand_potential_evaluation": "Hessian_path_integral"
            if separation < 0.05
            else "same_free_energy_reference_difference",
        },
    }


def _omega(model: ResidualHelmholtzModel, pair: dict[str, Any], c: FloatArray) -> float:
    if pair["stable_grand_response"]:
        return _grand_response(model, pair, c)[0]
    return fsum(
        (
            _free_energy(model, pair["temperature"], c),
            -pair["reference_f"],
            -float(pair["mu"] @ (c - pair["left"])[pair["active"]]),
        )
    )


_GRAND_NODES, _GRAND_WEIGHTS = np.polynomial.legendre.leggauss(12)
_GRAND_CHECK_NODES, _GRAND_CHECK_WEIGHTS = np.polynomial.legendre.leggauss(24)


def _grand_response(
    model: ResidualHelmholtzModel, pair: dict[str, Any], c: FloatArray, *, check_order: bool = False
) -> tuple[float, FloatArray]:
    """Reference-subtracted free energy without subtracting large close values.

    Delta mu=int H(c_left+s*Delta c) Delta c ds and
    omega=int (1-s) Delta c.T H Delta c ds. This uses the same free energy,
    including ideal mixing, and introduces neither a Landau replacement nor an
    endpoint fit. It is selected for close coexistence endpoints.
    """
    delta = c - pair["left"]
    active_delta = delta[pair["active"]]
    chemical = np.zeros(len(active_delta))
    energy = 0.0
    nodes, weights = (
        (_GRAND_CHECK_NODES, _GRAND_CHECK_WEIGHTS)
        if check_order
        else (_GRAND_NODES, _GRAND_WEIGHTS)
    )
    for point, weight in zip(nodes, weights, strict=True):
        s = (float(point) + 1) / 2
        h = _hessian(model, pair["temperature"], pair["left"] + s * delta, pair["active"])
        change = h @ active_delta
        chemical += float(weight) / 2 * change
        energy += float(weight) / 2 * (1 - s) * float(active_delta @ change)
    return float(energy), chemical


def _grand_quadrature_check(
    model: ResidualHelmholtzModel, pair: dict[str, Any], profile: Sequence[Sequence[float]]
) -> float:
    """Numerical consistency of 12 versus 24 nodes; not an independent oracle."""
    errors, scale = [], 1e-30
    for row in profile:
        c = np.asarray(row, dtype=float)
        first = _grand_response(model, pair, c)[0]
        second = _grand_response(model, pair, c, check_order=True)[0]
        errors.append(abs(first - second))
        scale = max(scale, abs(first), abs(second))
    return max(errors, default=0.0) / scale


def _quadrature(
    function: Any, lo: float, hi: float, tolerance: float, absolute: float
) -> tuple[float, float, int]:
    value, error, info, *warning = quad(
        function, lo, hi, epsrel=tolerance, epsabs=absolute, limit=150, full_output=True
    )
    if warning or not isfinite(value) or error > max(absolute, tolerance * abs(value)):
        raise ValueError("interface quadrature failed: " + str(warning or error))
    return float(value), float(error), int(info["neval"])


def pure_square_gradient_interface(
    model: ResidualHelmholtzModel,
    left_state: MolecularEOSState,
    right_state: MolecularEOSState,
    influence: MolecularInfluence,
    *,
    interior_density_grid: Sequence[float],
    quadrature_tolerance: float = 1e-9,
) -> MolecularInterfaceResult:
    """Exact scalar first integral, full tension and explicitly truncated profile."""
    pair = _coexistence(model, left_state, right_state, influence)
    if len(pair["active"]) != 1:
        raise ValueError("pure interface integral requires exactly one active component")
    if not isfinite(quadrature_tolerance) or quadrature_tolerance <= 0:
        raise ValueError("interface quadrature tolerance must be positive")
    index = pair["active"][0]
    lo, hi = sorted((pair["left"][index], pair["right"][index]))
    requested = tuple(float(v) for v in interior_density_grid)
    if not requested or any(not isfinite(v) or not lo < v < hi for v in requested):
        raise ValueError("pure profile densities must be strictly inside coexistence endpoints")
    coefficient = float(pair["C"][0, 0])

    def omega(rho: float) -> float:
        c = np.zeros(len(model.component_ids))
        c[index] = rho
        value = _omega(model, pair, c)
        if value <= 0:
            raise ValueError("nonpositive interior grand potential; no single scalar interface")
        return value

    try:
        if pair["stable_grand_response"]:
            check = _grand_quadrature_check(
                model,
                pair,
                [
                    pair["left"] + s * (pair["right"] - pair["left"])
                    for s in (0.1, 0.25, 0.5, 0.75, 0.9)
                ],
            )
            pair["diagnostics"].update(
                grand_path_quadrature_order=12,
                grand_path_check_order=24,
                grand_path_quadrature_relative_difference=check,
            )
            if check > quadrature_tolerance:
                raise ValueError("near-coexistence grand-potential quadrature is unresolved")
        tension, error, evaluations = _quadrature(
            lambda rho: sqrt(2 * coefficient * omega(rho)), lo, hi, quadrature_tolerance, 1e-14
        )
        positions, errors = [], []
        for rho in requested:
            position, bound, calls = _quadrature(
                lambda r: sqrt(coefficient / (2 * omega(r))),
                (lo + hi) / 2,
                rho,
                quadrature_tolerance,
                1e-16,
            )
            positions.append(position)
            errors.append(bound)
            evaluations += calls
        density_rows = []
        for rho in requested:
            row = np.zeros(len(model.component_ids))
            row[index] = rho
            density_rows.append(tuple(float(v) for v in row))
        return MolecularInterfaceResult(
            True,
            tension,
            tuple(positions),
            tuple(density_rows),
            diagnostics={
                **pair["diagnostics"],
                "method": "scalar_first_integral_quadrature",
                "tension_error_estimate_N_m": error,
                "position_error_estimates_m": errors,
                "evaluations": evaluations,
                "profile_tails": "infinite_asymptotes_excluded_from_requested_grid",
            },
        )
    except (ValueError, ArithmeticError) as exc:
        return MolecularInterfaceResult(
            False, failure_reason=str(exc), diagnostics=pair["diagnostics"]
        )


def _hessian(
    model: ResidualHelmholtzModel, temperature: float, c: FloatArray, active: Any
) -> FloatArray:
    values = {str(i): float(c[i]) for i in active}

    def extensive(v: dict[str, Any]) -> Any:
        concentrations = tuple(v.get(str(i), 0.0) for i in range(len(c)))
        rho = sum(concentrations)
        return rho * model.alphar(temperature, rho, tuple(x / rho for x in concentrations))

    rt = model.gas_constant_J_mol_K * temperature
    return np.array(
        [
            [
                rt
                * (
                    partial_derivative(extensive, values, (str(i), str(j)))
                    + (1 / c[i] if i == j else 0)
                )
                for j in active
            ]
            for i in active
        ]
    )


def molecular_planar_interface(
    model: ResidualHelmholtzModel,
    left_state: MolecularEOSState,
    right_state: MolecularEOSState,
    influence: MolecularInfluence,
    *,
    mesh: DiffusionMesh,
    component_density_bounds: Sequence[tuple[float, float]],
    initial_profile: Sequence[Sequence[float]] | None = None,
    max_iterations: int = 500,
    residual_tolerance: float = 1e-7,
    first_integral_tolerance: float = 0.05,
) -> MolecularInterfaceResult:
    """Free component densities minimize the finite-domain planar functional.

    A line is only the default initial guess. Each interior component is an
    independent variable; adsorption/overshoot are allowed within explicit
    bounds. The result is a finite-domain local stationary minimum, not a global
    profile or bulk-stability proof. Mesh/length convergence is the caller's task.
    """
    pair = _coexistence(model, left_state, right_state, influence)
    if mesh.geometry != "planar" or mesh.cells < 4:
        raise ValueError("interface requires a planar mesh with at least four intervals")
    if (
        type(max_iterations) is not int
        or max_iterations < 1
        or any(not isfinite(v) or v <= 0 for v in (residual_tolerance, first_integral_tolerance))
    ):
        raise ValueError("interface solver requires positive iteration budget and tolerance")
    active, cmat = pair["active"], pair["C"]
    n, count = len(active), mesh.cells - 1
    nodes = np.asarray(mesh.faces_m)
    dz = np.diff(nodes)
    weights = (dz[:-1] + dz[1:]) / 2
    scales = np.maximum(pair["left"][active], pair["right"][active])
    bounds = np.asarray(component_density_bounds, dtype=float)
    if (
        bounds.shape != (len(model.component_ids), 2)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[active, 0] <= 0)
        or np.any(bounds[active, 1] <= bounds[active, 0])
        or np.any(pair["left"] < bounds[:, 0])
        or np.any(pair["left"] > bounds[:, 1])
        or np.any(pair["right"] < bounds[:, 0])
        or np.any(pair["right"] > bounds[:, 1])
    ):
        raise ValueError("declared density bounds must contain both endpoints and positive support")
    fraction = nodes / nodes[-1]
    # Break exact grid-translation symmetry in the initial guess, which can
    # otherwise pin a coarse-grid saddle at a node. No path is constrained.
    fraction += 0.01 * np.sin(np.pi * fraction)
    initial = (
        pair["left"] + (pair["right"] - pair["left"]) * fraction[:, None]
        if initial_profile is None
        else np.asarray(initial_profile, dtype=float)
    )
    if (
        initial.shape != (len(nodes), len(model.component_ids))
        or not np.all(np.isfinite(initial))
        or np.any(initial < bounds[:, 0])
        or np.any(initial > bounds[:, 1])
        or not np.allclose(initial[0], pair["left"], rtol=1e-12, atol=0)
        or not np.allclose(initial[-1], pair["right"], rtol=1e-12, atol=0)
    ):
        raise ValueError("interface initial profile must match endpoints and declared bounds")
    inactive = [i for i in range(len(model.component_ids)) if i not in active]
    if inactive and np.any(initial[:, inactive] != 0):
        raise ValueError("interface cannot invent absent components")
    energy_scale = max(
        pair["RT"] * float(scales.sum()) * nodes[-1] * 0.01,
        float(scales @ cmat @ scales) / nodes[-1],
    )
    lower = np.tile(bounds[active, 0] / scales, count)
    upper = np.tile(bounds[active, 1] / scales, count)

    def unpack(q: FloatArray) -> FloatArray:
        profile = np.zeros_like(initial)
        profile[0], profile[-1] = pair["left"], pair["right"]
        profile[1:-1, active] = q.reshape(count, n) * scales
        return profile

    def evaluate(q: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        profile = unpack(q)
        omegas, mus = [], []
        for c in profile[1:-1]:
            try:
                if pair["stable_grand_response"]:
                    omega, delta_mu = _grand_response(model, pair, c)
                    omegas.append(omega)
                    mus.append(delta_mu)
                else:
                    bulk = molecular_interface_bulk(model, pair["temperature"], tuple(c))
                    omegas.append(_omega(model, pair, c))
                    mus.append(
                        np.array([float(bulk.chemical_potentials_J_mol[i]) for i in active])
                        - pair["mu"]
                    )
            except (ValueError, ArithmeticError) as exc:
                raise ResidualDomainError(str(exc)) from exc
        differences = np.diff(profile[:, active], axis=0)
        gradient_energy = 0.5 * float(np.sum((differences @ cmat) * differences / dz[:, None]))
        force = (differences[:-1] / dz[:-1, None] - differences[1:] / dz[1:, None]) @ cmat
        gradient = weights[:, None] * np.asarray(mus) + force
        energy = float(weights @ omegas) + gradient_energy
        return energy, gradient, np.array(omegas)

    domain_rejections: dict[str, Any] = {
        "count": 0,
        "reasons": {},
        "policy": "finite_infeasible_trial_penalty; never a physical energy",
    }
    largest_objective_magnitude = 1.0

    def objective(q: FloatArray) -> tuple[float, FloatArray]:
        nonlocal largest_objective_magnitude
        try:
            energy, gradient, _ = evaluate(q)
        except ResidualDomainError as exc:
            domain_rejections["count"] += 1
            reason = str(exc)
            domain_rejections["reasons"][reason] = domain_rejections["reasons"].get(reason, 0) + 1
            # The rectangular component bounds can include EOS-invalid mixtures.
            # Reject only this optimizer trial. Infinity makes L-BFGS-B's line
            # interpolation collapse to a zero step; a finite value above every
            # evaluated physical objective lets it backtrack. This value and its
            # zero gradient are optimizer machinery, never thermodynamic data.
            return 1e3 * largest_objective_magnitude, np.zeros_like(q)
        value = energy / energy_scale
        largest_objective_magnitude = max(largest_objective_magnitude, abs(value))
        return value, (gradient * scales / energy_scale).ravel()

    def residual(q: FloatArray) -> FloatArray:
        return (evaluate(q)[1] / weights[:, None] / pair["RT"]).ravel()

    def jacobian(q: FloatArray) -> Any:
        profile = unpack(q)
        matrix = lil_matrix((count * n, count * n))
        for k, c in enumerate(profile[1:-1]):
            try:
                diagonal = _hessian(model, pair["temperature"], c, active)
            except (ValueError, ArithmeticError) as exc:
                raise ResidualDomainError(str(exc)) from exc
            diagonal += cmat * (1 / dz[k] + 1 / dz[k + 1]) / weights[k]
            row = slice(k * n, (k + 1) * n)
            matrix[row, row] = diagonal * scales / pair["RT"]
            if k > 0:
                matrix[row, slice((k - 1) * n, k * n)] = (
                    -cmat * scales / (dz[k] * weights[k] * pair["RT"])
                )
            if k < count - 1:
                matrix[row, slice((k + 1) * n, (k + 2) * n)] = (
                    -cmat * scales / (dz[k + 1] * weights[k] * pair["RT"])
                )
        return matrix.tocsr()

    def minimum_curvature(q: FloatArray) -> float:
        hessian = diags(
            np.repeat(weights, n) * np.tile(scales, count) * pair["RT"] / energy_scale
        ) @ jacobian(q)
        hessian = (hessian + hessian.T) * 0.5
        bandwidth = 2 * n - 1
        banded = np.zeros((bandwidth + 1, count * n))
        for offset in range(bandwidth + 1):
            banded[bandwidth - offset, offset:] = hessian.diagonal(offset)
        curvature = float(
            eig_banded(banded, lower=False, eigvals_only=True, select="i", select_range=(0, 0))[0]
        )
        return curvature

    def accepted_profile(q: FloatArray) -> MolecularInterfaceResult:
        profile = unpack(q)
        energy, _, omegas = evaluate(q)
        if pair["stable_grand_response"]:
            check = _grand_quadrature_check(
                model, pair, profile[np.linspace(0, len(profile) - 1, 9, dtype=int)]
            )
            diagnostics.update(
                grand_path_quadrature_order=12,
                grand_path_check_order=24,
                grand_path_quadrature_relative_difference=check,
            )
            if check > residual_tolerance:
                return MolecularInterfaceResult(
                    False,
                    failure_reason="near_coexistence_grand_potential_unresolved",
                    diagnostics=diagnostics,
                )
        curvature = minimum_curvature(q)
        negative_tolerance = 1e-8 * pair["RT"] * float(scales.sum())
        diagnostics.update(
            scaled_minimum_curvature=curvature,
            minimum_grand_potential_J_m3=float(omegas.min()),
            negative_grand_potential_tolerance_J_m3=negative_tolerance,
        )
        if pair["nullspace"].shape[1]:
            nullspace = pair["nullspace"]
            null_curvatures, algebraic_errors = [], []
            for c in profile:
                h = _hessian(model, pair["temperature"], c, active)
                null_curvatures.append(float(np.linalg.eigvalsh(nullspace.T @ h @ nullspace)[0]))
                if pair["stable_grand_response"]:
                    delta_mu = _grand_response(model, pair, c)[1]
                else:
                    bulk = molecular_interface_bulk(model, pair["temperature"], tuple(c))
                    delta_mu = (
                        np.array([float(bulk.chemical_potentials_J_mol[i]) for i in active])
                        - pair["mu"]
                    )
                algebraic_errors.append(float(np.max(np.abs(nullspace.T @ delta_mu))) / pair["RT"])
            diagnostics.update(
                nullspace_dimension=nullspace.shape[1],
                minimum_nullspace_bulk_curvature_J_m3_mol2=min(null_curvatures),
                algebraic_nullspace_residual_RT=max(algebraic_errors),
                algebraic_residual_bound_RT=sqrt(n) * residual_tolerance,
                singular_regularization="none; original C retained",
            )
            if min(null_curvatures) <= 0 or max(algebraic_errors) > sqrt(n) * residual_tolerance:
                return MolecularInterfaceResult(
                    False,
                    failure_reason="singular_algebraic_branch_not_resolved_stable",
                    diagnostics=diagnostics,
                )
        if curvature < -1e-7 or energy <= 0 or float(omegas.min()) < -negative_tolerance:
            return MolecularInterfaceResult(
                False, failure_reason="nonminimum_or_unstable_interface", diagnostics=diagnostics
            )
        slopes = np.diff(profile[:, active], axis=0) / dz[:, None]
        gradient_density = 0.5 * np.sum((slopes @ cmat) * slopes, axis=1)
        middle = 0.5 * (profile[1:] + profile[:-1])
        middle_omega = np.array([_omega(model, pair, c) for c in middle])
        integral = gradient_density - middle_omega
        barrier = max(float(middle_omega.max()), 1e-30)
        diagnostics.update(
            first_integral_variation_relative=float(np.ptp(integral)) / barrier,
            first_integral_boundary_J_m3=(float(integral[0]), float(integral[-1])),
            gradient_tension_N_m=float(2 * gradient_density @ dz),
            maximum_grand_potential_J_m3=barrier,
            stationarity_residual_RT=float(np.max(np.abs(residual(q)))),
        )
        resolved = diagnostics["first_integral_variation_relative"] <= first_integral_tolerance
        diagnostics["first_integral_tolerance"] = first_integral_tolerance
        return MolecularInterfaceResult(
            resolved,
            energy,
            tuple(float(v) for v in nodes),
            tuple(tuple(float(v) for v in row) for row in profile),
            "" if resolved else "profile_underresolved_first_integral",
            diagnostics=diagnostics,
        )

    used = 0

    def initialize_translation(root_initial: FloatArray, limit: int) -> FloatArray:
        nonlocal used
        # The position of a well separated interface is almost a null mode.
        # Temporarily pin its translation to find a good numerical seed. The
        # final solve below removes both the pin and its multiplier and checks
        # every original Euler-Lagrange equation within the same root budget.
        before_pin = used
        tangent = np.gradient(unpack(root_initial)[:, active] / scales, nodes / nodes[-1], axis=0)[
            1:-1
        ].ravel()
        tangent_norm = np.linalg.norm(tangent)
        tangent /= tangent_norm
        if limit - used > 1:
            anchor = root_initial.copy()

            def pinned_jacobian(v: FloatArray) -> Any:
                return bmat(
                    [[jacobian(v[:-1]), tangent[:, None]], [tangent[None, :], None]], format="csr"
                )

            pin_attempts: list[dict[str, Any]] = []
            pin_cache: dict[float, Any] = {}

            def pinned_at(offset: float) -> Any:
                nonlocal used
                if offset in pin_cache:
                    return pin_cache[offset]
                if used >= limit - 1:
                    raise ValueError("translation initialization exhausted root budget")

                def pinned_residual(v: FloatArray) -> FloatArray:
                    return np.r_[
                        residual(v[:-1]) + v[-1] * tangent, tangent @ (v[:-1] - anchor) - offset
                    ]

                pinned = solve_root(
                    pinned_residual,
                    np.r_[np.clip(anchor + offset * tangent, lower, upper), 0.0],
                    jacobian=pinned_jacobian,
                    variables=(*variables, ResidualVariable("translation_multiplier", "1")),
                    residual_scales=np.ones(count * n + 1),
                    policy=RootPolicy(
                        tolerance=residual_tolerance * 0.01,
                        maximum_iterations=min(25, limit - 1 - used),
                    ),
                )
                used += pinned.diagnostics["iterations"]
                pin_attempts.append(
                    {
                        "offset": float(offset),
                        "success": pinned.success,
                        "failure_reason": pinned.failure_reason,
                        **pinned.diagnostics,
                    }
                )
                pin_cache[offset] = pinned
                if not pinned.success:
                    raise ValueError("translation initialization root failed")
                pin_attempts[-1]["multiplier"] = float(pinned.value[-1])
                return pinned

            candidates = []
            try:
                pinned = pinned_at(0.0)
                root_initial = pinned.value[:-1]
                # If full stationarity already holds, the usual curvature check
                # still decides acceptance. Otherwise resolve the weak mesh
                # translation force rather than following a huge linear step.
                if (
                    np.max(np.abs(residual(root_initial))) > residual_tolerance
                    or minimum_curvature(root_initial) < -1e-7
                ):
                    width = float(tangent_norm * np.min(dz) / nodes[-1])
                    offsets = np.linspace(-0.5 * width, 0.5 * width, 5)
                    forces = [float(pinned_at(float(s)).value[-1]) for s in offsets]
                    for left, right, fleft, fright in zip(
                        offsets[:-1], offsets[1:], forces[:-1], forces[1:], strict=True
                    ):
                        if fleft * fright <= 0:
                            zero = brentq(
                                lambda s: float(pinned_at(s).value[-1]),
                                float(left),
                                float(right),
                                xtol=max(1e-14, width * 1e-8),
                                maxiter=20,
                            )
                            candidates.append(pinned_at(zero).value[:-1])
                    if candidates:
                        root_initial = min(candidates, key=lambda q: evaluate(q)[0])
            except (ValueError, RuntimeError) as exc:
                diagnostics["translation_initialization_failure"] = str(exc)
            diagnostics["translation_initialization"] = {
                "attempts": pin_attempts,
                "stationary_candidates": len(candidates),
                "iterations": used - before_pin,
                "accepted_profile_constraint": "none",
            }
        return root_initial

    diagnostics = {
        **pair["diagnostics"],
        "method": "free_density_variational_and_shared_root",
        "nodes": len(nodes),
        "length_m": float(nodes[-1]),
        "profile_scope": "local_finite_domain; grid_and_length_convergence_external",
        "path_constraint": "none; all active component densities independently free",
        "minimization_domain_rejections": domain_rejections,
    }
    try:
        initial_q = (initial[1:-1, active] / scales).ravel()
        # Neither an invalid initial profile nor an invalid optimizer output can
        # acquire validity from the nonphysical line-search rejection value.
        evaluate(initial_q)
        variables = tuple(
            ResidualVariable(f"density_{i}", "scaled_mol_m3", lower=float(lo), upper=float(hi))
            for i, (lo, hi) in enumerate(zip(lower, upper, strict=True))
        )
        # A supplied profile is only a seed. Spend part of the original root
        # budget on its stationary continuation before paying for minimization.
        # Every successful fast path passes the same complete physical checks.
        if initial_profile is not None and max_iterations > 1:
            before_warm = dict(diagnostics)
            warm_limit = min(100, max_iterations - 1)
            warm_initial = initialize_translation(initial_q, warm_limit)
            warm = solve_root(
                residual,
                warm_initial,
                jacobian=jacobian,
                variables=variables,
                residual_scales=np.ones(count * n),
                policy=RootPolicy(
                    tolerance=residual_tolerance, maximum_iterations=warm_limit - used
                ),
            )
            used += warm.diagnostics["iterations"]
            warm_record = {
                "root_success": warm.success,
                "root_failure": warm.failure_reason,
                **warm.diagnostics,
                "root_iterations_total": used,
                "translation_initialization": diagnostics.get("translation_initialization"),
                "translation_initialization_failure": diagnostics.get(
                    "translation_initialization_failure", ""
                ),
            }
            if warm.success:
                diagnostics["root"] = {"success": True, "failure_reason": "", **warm.diagnostics}
                diagnostics["root_iterations_total"] = used
                try:
                    candidate = accepted_profile(warm.value)
                except (ValueError, ArithmeticError) as exc:
                    candidate = MolecularInterfaceResult(
                        False, failure_reason=str(exc), diagnostics=diagnostics
                    )
                warm_record["physical_success"] = candidate.success
                warm_record["physical_failure"] = candidate.failure_reason
                if candidate.success:
                    diagnostics["warm_start"] = warm_record
                    diagnostics["minimization"] = {
                        "success": True,
                        "iterations": 0,
                        "evaluations": 0,
                        "message": "skipped_after_full_warm_root_acceptance",
                    }
                    return candidate
                warm_record["failed_acceptance"] = candidate.to_dict()
            diagnostics.clear()
            diagnostics.update(before_warm)
            diagnostics["warm_start"] = warm_record
        optimum = minimize(
            objective,
            initial_q,
            jac=True,
            method="L-BFGS-B",
            bounds=list(zip(lower, upper, strict=True)),
            options={"maxiter": max_iterations, "gtol": 1e-10, "ftol": 1e-15, "maxcor": 30},
        )
        diagnostics["minimization"] = {
            "success": bool(optimum.success),
            "iterations": int(optimum.nit),
            "evaluations": int(optimum.nfev),
            "message": str(optimum.message),
        }
        evaluate(optimum.x)
        root_initial = initialize_translation(optimum.x, max_iterations)
        root = solve_root(
            residual,
            root_initial,
            jacobian=jacobian,
            variables=variables,
            residual_scales=np.ones(count * n),
            policy=RootPolicy(
                tolerance=residual_tolerance, maximum_iterations=max_iterations - used
            ),
        )
        diagnostics["root"] = {
            "success": root.success,
            "failure_reason": root.failure_reason,
            **root.diagnostics,
        }
        diagnostics["root_iterations_total"] = used + root.diagnostics["iterations"]
        if not root.success:
            return MolecularInterfaceResult(
                False, failure_reason="interface_root_failed", diagnostics=diagnostics
            )
        return accepted_profile(root.value)
    except (ValueError, ArithmeticError, ResidualDomainError) as exc:
        return MolecularInterfaceResult(False, failure_reason=str(exc), diagnostics=diagnostics)
