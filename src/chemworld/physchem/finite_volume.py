"""One-dimensional conservative Fick diffusion, returning mol/s, never time steps.

Inventories belong to the caller. Coordinates use m, concentrations mol/m³,
diffusivity m²/s and prescribed outward boundary flux mol/(m² s). The adapter
below contributes to the existing coupled solver; it owns no persistent state.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from math import isfinite, pi
from numbers import Integral
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix, sparray, spmatrix

from chemworld.physchem.coupled_system import (
    CoupledEvent,
    CoupledProcess,
    CoupledStateView,
    InternalTransfer,
    MaterialPort,
    ProcessContribution,
)

if TYPE_CHECKING:
    from chemworld.physchem.residual_system import ResidualSystem, ResidualVariable


_VectorCallback = Callable[[float, NDArray[np.float64], NDArray[np.float64]], ArrayLike]
_MatrixCallback = Callable[
    [float, NDArray[np.float64], NDArray[np.float64]], ArrayLike | spmatrix | sparray,
]


class _DerivativeOptions(TypedDict, total=False):
    rhs_jacobian: _MatrixCallback
    rhs_time_derivative: _VectorCallback
    rhs_parameter_jacobian: _MatrixCallback


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=float).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DiffusionMesh:
    geometry: str
    faces_m: tuple[float, ...]
    area_m2: float = 1.0
    particle_count: float = 1.0
    centers_m: np.ndarray = field(init=False, repr=False, compare=False)
    cell_volumes_m3: np.ndarray = field(init=False, repr=False, compare=False)
    face_areas_m2: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        faces = np.asarray(self.faces_m, dtype=float)
        if (self.geometry not in {"planar", "spherical"} or faces.ndim != 1
                or faces.size < 2 or not np.all(np.isfinite(faces))
                or faces[0] != 0 or np.any(np.diff(faces) <= 0)):
            raise ValueError("mesh requires a planar/radial increasing grid starting at zero")
        if any(not isfinite(v) or v <= 0 for v in (self.area_m2, self.particle_count)):
            raise ValueError("mesh area and particle count must be finite and positive")
        if self.geometry == "planar":
            if self.particle_count != 1:
                raise ValueError("particle count applies only to spherical geometry")
            volumes = self.area_m2 * np.diff(faces)
            areas: NDArray[np.float64] = np.full(faces.size, self.area_m2)
        else:
            if self.area_m2 != 1:
                raise ValueError("planar area does not apply to spherical geometry")
            volumes = 4 * pi / 3 * self.particle_count * np.diff(faces ** 3)
            areas = 4 * pi * self.particle_count * faces ** 2
        if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
            raise ValueError("mesh volumes must be finite and positive")
        if not np.all(np.isfinite(areas)):
            raise ValueError("mesh face areas must be finite")
        centers = faces[:-1] + np.diff(faces) / 2
        if (np.any(centers <= faces[:-1]) or np.any(centers >= faces[1:])
                or np.any(np.diff(centers) <= 0)):
            raise ValueError("mesh cell centers must be representable inside their faces")
        object.__setattr__(self, "faces_m", tuple(float(v) for v in faces))
        object.__setattr__(self, "centers_m", _readonly(centers))
        object.__setattr__(self, "cell_volumes_m3", _readonly(volumes))
        object.__setattr__(self, "face_areas_m2", _readonly(areas))

    @classmethod
    def planar(
        cls, length_m: float, cells: int | np.integer, *, area_m2: float = 1.0,
    ) -> DiffusionMesh:
        return cls("planar", cls._uniform_faces(length_m, cells), area_m2=area_m2)

    @classmethod
    def spherical(
        cls, radius_m: float, cells: int | np.integer, *, particle_count: float = 1.0,
    ) -> DiffusionMesh:
        return cls("spherical", cls._uniform_faces(radius_m, cells),
                   particle_count=particle_count)

    @staticmethod
    def _uniform_faces(extent_m: float, cells: int | np.integer) -> tuple[float, ...]:
        if (not isfinite(extent_m) or extent_m <= 0 or not isinstance(cast(object, cells), Integral)
                or isinstance(cells, bool) or cells < 1):
            raise ValueError("mesh extent and integer cell count must be positive")
        return tuple(np.linspace(0, extent_m, cells + 1))

    @property
    def cells(self) -> int:
        return len(self.faces_m) - 1

    @property
    def shell_mean_square_radius_m2(self) -> NDArray[np.float64]:
        """Volume average of r^2; stable polynomial quotient for thin shells."""
        if self.geometry != "spherical":
            raise ValueError("shell moments require spherical geometry")
        a, b = np.asarray(self.faces_m[:-1]), np.asarray(self.faces_m[1:])
        return 3 / 5 * (b**4 + b**3 * a + b**2 * a**2 + b * a**3 + a**4) / (
            b**2 + b * a + a**2)

    @cached_property
    def spherical_surface_distance_m(self) -> float:
        """Surface derivative to outer shell-average difference for a+b*r^2."""
        radius = self.faces_m[-1]
        return (radius**2 - self.shell_mean_square_radius_m2[-1]) / (2 * radius)


@dataclass(frozen=True)
class DiffusionBoundary:
    kind: str = "symmetry"
    value: float = 0.0

    def __post_init__(self) -> None:
        if (self.kind not in {"symmetry", "flux", "concentration"}
                or not isfinite(self.value)
                or (self.kind == "concentration" and self.value < 0)
                or (self.kind == "symmetry" and self.value != 0)):
            raise ValueError("invalid diffusion boundary or physical value")


@dataclass(frozen=True)
class DiffusionRates:
    amount_rates_mol_s: np.ndarray
    # Positive along increasing coordinate; each internal face is computed once.
    face_rates_mol_s: np.ndarray

    def __post_init__(self) -> None:
        for name in ("amount_rates_mol_s", "face_rates_mol_s"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def outward_boundary_rates_mol_s(self) -> tuple[float, float]:
        return (-float(self.face_rates_mol_s[0]), float(self.face_rates_mol_s[-1]))


@dataclass(frozen=True)
class DiffusionCoupling:
    process: CoupledProcess
    ports: tuple[MaterialPort, ...]


@dataclass(frozen=True)
class ConservativeDiffusion:
    mesh: DiffusionMesh
    diffusivity_m2_s: float | tuple[float, ...]
    left: DiffusionBoundary = DiffusionBoundary()
    right: DiffusionBoundary = DiffusionBoundary()
    storage_fractions: tuple[float, ...] | None = field(default=None, kw_only=True)
    _jacobian: csr_matrix = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        coefficient = np.asarray(self.diffusivity_m2_s, dtype=float)
        if (coefficient.shape not in {(), (self.mesh.cells,)}
                or not np.all(np.isfinite(coefficient)) or np.any(coefficient < 0)):
            raise ValueError("diffusivity must be finite and nonnegative")
        if coefficient.ndim:
            object.__setattr__(self, "diffusivity_m2_s", tuple(coefficient))
        if self.storage_fractions is not None:
            fractions = np.asarray(self.storage_fractions, dtype=float)
            if (fractions.shape != (self.mesh.cells,) or not np.all(np.isfinite(fractions))
                    or np.any(fractions <= 0) or np.any(fractions > 1)):
                raise ValueError("storage fractions require one finite value in (0, 1] per cell")
            object.__setattr__(self, "storage_fractions", tuple(fractions))
        if self.mesh.geometry == "spherical" and (
            self.left.kind == "concentration" or self.left.value != 0
        ):
            raise ValueError("the sphere center requires symmetry/zero flux")
        rows, columns, values = [], [], []

        def add(row: int, column: int, value: float) -> None:
            rows.append(row)
            columns.append(column)
            values.append(value)

        volumes = self.storage_volumes_m3
        for face, conductance in enumerate(self._internal_conductances(), 1):
            left, right = face - 1, face
            add(left, left, -conductance / volumes[left])
            add(left, right, conductance / volumes[right])
            add(right, left, conductance / volumes[left])
            add(right, right, -conductance / volumes[right])
        for side, index, boundary in (("left", 0, self.left), ("right", -1, self.right)):
            if boundary.kind == "concentration":
                cell = index % self.mesh.cells
                add(cell, cell, -self._boundary_conductance(side) / volumes[index])
        matrix = coo_matrix((values, (rows, columns)),
                            shape=(self.mesh.cells, self.mesh.cells)).tocsr()
        matrix.eliminate_zeros()
        object.__setattr__(self, "_jacobian", matrix)

    def _internal_conductances(self) -> NDArray[np.float64]:
        faces = np.asarray(self.mesh.faces_m[1:-1])
        left_distance = faces - self.mesh.centers_m[:-1]
        right_distance = self.mesh.centers_m[1:] - faces
        if self.mesh.geometry == "spherical":
            # Inventories encode shell averages, not point values at arithmetic
            # midpoints. In r^2 coordinates these distances reproduce c=a+b*r^2
            # exactly, including on graded meshes, while retaining one positive
            # conservative two-point conductance at each face.
            means = self.mesh.shell_mean_square_radius_m2
            left_distance = (faces**2 - means[:-1]) / (2 * faces)
            right_distance = (means[1:] - faces**2) / (2 * faces)
        if isinstance(self.diffusivity_m2_s, tuple):
            # Resistances of the two half-cells are in series, including at
            # material interfaces. One face flux serves both adjacent cells.
            coefficients = np.asarray(self.diffusivity_m2_s)
            valid = (coefficients[:-1] > 0) & (coefficients[1:] > 0)
            result = np.zeros(self.mesh.cells - 1)
            result[valid] = self.mesh.face_areas_m2[1:-1][valid] / (
                left_distance[valid] / coefficients[:-1][valid]
                + right_distance[valid] / coefficients[1:][valid]
            )
            return result
        return (self.diffusivity_m2_s * self.mesh.face_areas_m2[1:-1]
                / (left_distance + right_distance))

    def _boundary_conductance(self, side: str) -> float:
        index = 0 if side == "left" else -1
        distance = abs(self.mesh.faces_m[index] - self.mesh.centers_m[index])
        if self.mesh.geometry == "spherical" and side == "right":
            distance = self.mesh.spherical_surface_distance_m
        coefficient = (self.diffusivity_m2_s[index]
                       if isinstance(self.diffusivity_m2_s, tuple) else self.diffusivity_m2_s)
        return coefficient * self.mesh.face_areas_m2[index] / distance

    @property
    def storage_volumes_m3(self) -> NDArray[np.float64]:
        """Physical pore volume, not an extra copy of the caller's inventory."""
        if self.storage_fractions is None:
            return self.mesh.cell_volumes_m3
        return self.mesh.cell_volumes_m3 * np.asarray(self.storage_fractions)

    @property
    def inventory_jacobian_s_inv(self) -> csr_matrix:
        """dn_dot/dn, a tridiagonal sparse matrix; callers receive an independent copy."""
        return self._jacobian.copy()

    def evaluate(self, amounts_mol: ArrayLike) -> DiffusionRates:
        amounts = np.asarray(amounts_mol, dtype=float)
        if (amounts.shape != (self.mesh.cells,) or not np.all(np.isfinite(amounts))
                or np.any(amounts < 0)):
            raise ValueError("diffusion inventory must be finite, nonnegative and match cells")
        concentration = amounts / self.storage_volumes_m3
        faces = np.zeros(self.mesh.cells + 1)
        faces[1:-1] = self._internal_conductances() * (
            concentration[:-1] - concentration[1:])
        for side, index, boundary in (("left", 0, self.left), ("right", -1, self.right)):
            outward = 0.0
            if boundary.kind == "flux":
                outward = boundary.value * self.mesh.face_areas_m2[index]
            elif boundary.kind == "concentration":
                outward = self._boundary_conductance(side) * (
                    concentration[index] - boundary.value)
            faces[index] = -outward if side == "left" else outward
        derivatives = faces[:-1] - faces[1:]
        if not np.all(np.isfinite(faces)) or not np.all(np.isfinite(derivatives)):
            raise ValueError("diffusion face rates exceed the finite model domain")
        return DiffusionRates(derivatives, faces)

    def residual_system(
        self, inventory_keys: Sequence[str], *, system_id: str, basis: str = "amount",
        concentration_scale_mol_m3: float = 1.0, time_scale_s: float = 1.0,
        reaction_rates: _VectorCallback | None = None,
        reaction_jacobian: _MatrixCallback | None = None,
        reaction_parameter_jacobian: _MatrixCallback | None = None,
        reaction_time_derivative: _VectorCallback | None = None,
        parameters: Sequence[ResidualVariable] = (), parameter_values: Sequence[float] = (),
        derivatives: str = "analytic",
    ) -> ResidualSystem:
        """Reuse the affine face operator in an amount or concentration residual.

        Reaction callbacks consume amounts (mol) and return mol/s, with their
        Jacobian with respect to those amounts. Concentration coordinates use
        the actual cell-volume mass matrix. The affine diffusion extension to
        solver trials does not clip inventories; accepted states remain bounded.
        """
        from scipy.sparse import diags

        from chemworld.physchem.residual_solver import mass_matrix_system
        from chemworld.physchem.residual_system import (
            ResidualEquation,
            ResidualVariable,
            matrix,
            vector,
        )

        keys = tuple(inventory_keys)
        n = self.mesh.cells
        if len(keys) != n or len(set(keys)) != n or basis not in {"amount", "concentration"}:
            raise ValueError("diffusion needs one unique key per cell and a declared basis")
        if not isfinite(concentration_scale_mol_m3) or concentration_scale_mol_m3 <= 0:
            raise ValueError("concentration scale must be finite and positive")
        if not isfinite(time_scale_s) or time_scale_s <= 0:
            raise ValueError("diffusion time scale must be finite and positive")
        if reaction_rates is not None and derivatives == "analytic" and reaction_jacobian is None:
            raise ValueError("analytic reaction diffusion requires the reaction Jacobian")
        mass = np.ones(n) if basis == "amount" else self.storage_volumes_m3
        amounts_scale = concentration_scale_mol_m3 * self.storage_volumes_m3
        variables = tuple(ResidualVariable(
            key, "mol" if basis == "amount" else "mol/m^3", amounts_scale[i] / mass[i], lower=0,
        ) for i, key in enumerate(keys))
        equations = tuple(ResidualEquation(key, "mol/s", amounts_scale[i] / time_scale_s)
                          for i, key in enumerate(keys))
        operator = self.inventory_jacobian_s_inv
        boundary = self.evaluate(np.zeros(n)).amount_rates_mol_s

        def rhs(
            t: float, y: NDArray[np.float64], p: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            amounts = y * mass
            rates = operator @ amounts + boundary
            if reaction_rates is not None:
                rates += vector(reaction_rates(t, _readonly(amounts), p), n, "reaction rates")
            return rates

        def jacobian(t: float, y: NDArray[np.float64], p: NDArray[np.float64]) -> csr_matrix:
            total = operator
            if reaction_rates is not None:
                total = total + matrix(cast(_MatrixCallback, reaction_jacobian)(
                    t, _readonly(y * mass), p),
                                       (n, n), "reaction Jacobian")
            return csr_matrix(total) @ diags(mass)

        options: _DerivativeOptions = {}
        if derivatives == "analytic":
            options["rhs_jacobian"] = jacobian
            if reaction_rates is None:
                options["rhs_time_derivative"] = lambda t, y, p: np.zeros(n)
                options["rhs_parameter_jacobian"] = lambda t, y, p: np.zeros((n, len(parameters)))
            else:
                if reaction_time_derivative is not None:
                    options["rhs_time_derivative"] = lambda t, y, p: reaction_time_derivative(
                        t, _readonly(y * mass), p
                    )
                if reaction_parameter_jacobian is not None:
                    options["rhs_parameter_jacobian"] = lambda t, y, p: reaction_parameter_jacobian(
                        t, _readonly(y * mass), p
                    )
        elif any(fn is not None for fn in (reaction_jacobian, reaction_parameter_jacobian,
                                          reaction_time_derivative)):
            raise ValueError("numerical reaction derivatives cannot include analytic callbacks")
        return mass_matrix_system(system_id, variables, equations, diags(mass), rhs,
                                  derivatives=derivatives, parameters=parameters,
                                  parameter_values=parameter_values, time_scale_s=time_scale_s,
                                  **options)

    def couple(
        self, inventory_keys: Sequence[str], *, process_id: str, owner_id: str,
        boundary_enthalpies_J_mol: Mapping[str, float] | None = None,
    ) -> DiffusionCoupling:
        """Bind existing inventory keys and explicit boundary enthalpies, without a clock.

        An imposed outward flux stops at depletion of its boundary cell. Domains
        needing other limits (e.g. saturation) contribute their own shared events.
        """
        keys = tuple(inventory_keys)
        if (len(keys) != self.mesh.cells or len(set(keys)) != len(keys)
                or any(not isinstance(key, str) or not key for key in keys)
                or not process_id or not owner_id):
            raise ValueError("diffusion coupling needs unique cell keys and explicit owners")
        sides = [(side, index, boundary)
                 for side, index, boundary in (("left", 0, self.left), ("right", -1, self.right))
                 if self.mesh.face_areas_m2[index] > 0 and boundary.kind != "symmetry"
                 and not (boundary.kind == "flux" and boundary.value == 0)]
        enthalpies = dict(boundary_enthalpies_J_mol or {})
        if (set(enthalpies) != {side for side, _, _ in sides}
                or any(not isfinite(h) for h in enthalpies.values())):
            raise ValueError("each open diffusion boundary needs its declared enthalpy")
        ports = tuple(
            MaterialPort(
                f"{process_id}:{side}:{direction}", keys[index],
                f"boundary:{process_id}:{side}" if direction == "in" else owner_id,
                owner_id if direction == "in" else f"boundary:{process_id}:{side}", direction,
            )
            for side, index, _ in sides for direction in ("in", "out")
        )

        def evaluate(state: CoupledStateView) -> ProcessContribution:
            rates = self.evaluate([state.amounts_mol[key] for key in keys])
            transfers = tuple(
                InternalTransfer(f"{process_id}:face:{face}",
                                 keys[face - 1] if rate >= 0 else keys[face],
                                 keys[face] if rate >= 0 else keys[face - 1], abs(float(rate)))
                for face, rate in enumerate(rates.face_rates_mol_s[1:-1], 1) if rate
            )
            port_rates, port_enthalpies = {}, {}
            outward = dict(zip(("left", "right"), rates.outward_boundary_rates_mol_s,
                               strict=True))
            for port in ports:
                side = port.port_id.split(":")[-2]
                port_rates[port.port_id] = max(
                    outward[side] if port.direction == "out" else -outward[side], 0.0)
                port_enthalpies[port.port_id] = enthalpies[side]
            return ProcessContribution(internal_transfers=transfers,
                                       port_rates_mol_s=port_rates,
                                       port_enthalpies_J_mol=port_enthalpies)

        def depletion_condition(key: str) -> Callable[[CoupledStateView], float]:
            def remaining_amount(state: CoupledStateView) -> float:
                return state.amounts_mol[key]

            return remaining_amount

        events = tuple(
            CoupledEvent(f"{process_id}:depleted:{side}",
                         depletion_condition(keys[index]))
            for side, index, boundary in sides if boundary.kind == "flux" and boundary.value > 0
        )
        dependencies = {key: keys[max(0, i - 1):min(len(keys), i + 2)]
                        for i, key in enumerate(keys)}

        def validate(state: CoupledStateView) -> None:
            self.evaluate([state.amounts_mol[key] for key in keys])

        return DiffusionCoupling(CoupledProcess(
            process_id, evaluate, events, validate,
            inventory_dependencies=dependencies,
        ), ports)
