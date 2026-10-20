"""One-dimensional conservative Fick diffusion, returning mol/s, never time steps.

Inventories belong to the caller. Coordinates use m, concentrations mol/m³,
diffusivity m²/s and prescribed outward boundary flux mol/(m² s). The adapter
below contributes to the existing coupled solver; it owns no persistent state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite, pi
from numbers import Integral

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from chemworld.physchem.coupled_system import (
    CoupledEvent,
    CoupledProcess,
    InternalTransfer,
    MaterialPort,
    ProcessContribution,
)


def _readonly(values):
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

    def __post_init__(self):
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
            areas = np.full(faces.size, self.area_m2)
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
    def planar(cls, length_m, cells, *, area_m2=1.0):
        return cls("planar", cls._uniform_faces(length_m, cells), area_m2=area_m2)

    @classmethod
    def spherical(cls, radius_m, cells, *, particle_count=1.0):
        return cls("spherical", cls._uniform_faces(radius_m, cells),
                   particle_count=particle_count)

    @staticmethod
    def _uniform_faces(extent_m, cells):
        if (not isfinite(extent_m) or extent_m <= 0 or not isinstance(cells, Integral)
                or isinstance(cells, bool) or cells < 1):
            raise ValueError("mesh extent and integer cell count must be positive")
        return tuple(np.linspace(0, extent_m, cells + 1))

    @property
    def cells(self):
        return len(self.faces_m) - 1


@dataclass(frozen=True)
class DiffusionBoundary:
    kind: str = "symmetry"
    value: float = 0.0

    def __post_init__(self):
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

    def __post_init__(self):
        for name in ("amount_rates_mol_s", "face_rates_mol_s"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def outward_boundary_rates_mol_s(self):
        return (-float(self.face_rates_mol_s[0]), float(self.face_rates_mol_s[-1]))


@dataclass(frozen=True)
class DiffusionCoupling:
    process: CoupledProcess
    ports: tuple[MaterialPort, ...]


@dataclass(frozen=True)
class ConservativeDiffusion:
    mesh: DiffusionMesh
    diffusivity_m2_s: float
    left: DiffusionBoundary = DiffusionBoundary()
    right: DiffusionBoundary = DiffusionBoundary()
    _jacobian: csr_matrix = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isfinite(self.diffusivity_m2_s) or self.diffusivity_m2_s < 0:
            raise ValueError("diffusivity must be finite and nonnegative")
        if self.mesh.geometry == "spherical" and (
            self.left.kind == "concentration" or self.left.value != 0
        ):
            raise ValueError("the sphere center requires symmetry/zero flux")
        rows, columns, values = [], [], []

        def add(row, column, value):
            rows.append(row)
            columns.append(column)
            values.append(value)

        volumes = self.mesh.cell_volumes_m3
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

    def _internal_conductances(self):
        return (self.diffusivity_m2_s * self.mesh.face_areas_m2[1:-1]
                / np.diff(self.mesh.centers_m))

    def _boundary_conductance(self, side):
        index = 0 if side == "left" else -1
        distance = abs(self.mesh.faces_m[index] - self.mesh.centers_m[index])
        return self.diffusivity_m2_s * self.mesh.face_areas_m2[index] / distance

    @property
    def inventory_jacobian_s_inv(self):
        """dn_dot/dn, a tridiagonal sparse matrix; callers receive an independent copy."""
        return self._jacobian.copy()

    def evaluate(self, amounts_mol: Sequence[float]) -> DiffusionRates:
        amounts = np.asarray(amounts_mol, dtype=float)
        if (amounts.shape != (self.mesh.cells,) or not np.all(np.isfinite(amounts))
                or np.any(amounts < 0)):
            raise ValueError("diffusion inventory must be finite, nonnegative and match cells")
        concentration = amounts / self.mesh.cell_volumes_m3
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

        def evaluate(state):
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

        events = tuple(
            CoupledEvent(f"{process_id}:depleted:{side}",
                         lambda state, key=keys[index]: state.amounts_mol[key])
            for side, index, boundary in sides if boundary.kind == "flux" and boundary.value > 0
        )
        dependencies = {key: keys[max(0, i - 1):min(len(keys), i + 2)]
                        for i, key in enumerate(keys)}

        def validate(state):
            self.evaluate([state.amounts_mol[key] for key in keys])

        return DiffusionCoupling(CoupledProcess(
            process_id, evaluate, events, validate,
            inventory_dependencies=dependencies,
        ), ports)
