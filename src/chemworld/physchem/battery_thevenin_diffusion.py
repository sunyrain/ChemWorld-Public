"""Optional normalized planar SOC diffusion, on the shared conservative FV operator.

The equations are PyBaMM's equivalent-circuit diffusion element with the voltage
and heat signs corrected in official PR 5869. x is dimensionless on [0, 1]:
z_t = z_xx/tau, z_x(0)=0, z_x(1)=-tau*I/Q (positive I discharges).
The shared FV geometry is evaluated on a unit-length numerical domain only to
obtain its conservative linear operator; no molar inventory or material port is
invented for dimensionless SOC. There is no integrator or persistent state here.

Average SOC is owned by the circuit. Only N-1 deviations are additional state;
the last deviation is eliminated by the volume-weighted zero-mean constraint.
Surface SOC uses linear extrapolation of the last two cell centers, matching
the explicitly selected upstream ``surf`` convention (use_bcs=False).
"""

from dataclasses import dataclass, field
from math import isfinite

import numpy as np
from scipy import sparse

from chemworld.data.scientific_inputs import ScientificTable, _keys, _number, _sequence, _text
from chemworld.foundation.units import convert_value
from chemworld.physchem.coupled_system import CoupledCoordinate
from chemworld.physchem.finite_volume import ConservativeDiffusion, DiffusionMesh

DIFFUSION_CONVENTION = "surface-ocv-passive-pr5869"
TAU_COLUMN = "diffusion_time_constant_s"


@dataclass(frozen=True)
class TheveninDiffusionSpec:
    parameters: ScientificTable
    parameter_row_id: str
    faces: tuple[float, ...]
    convention: str = DIFFUSION_CONVENTION
    tau_s: float = field(init=False)
    weights: np.ndarray = field(init=False, repr=False, compare=False)
    centers: np.ndarray = field(init=False, repr=False, compare=False)
    _operator: sparse.csr_matrix = field(init=False, repr=False, compare=False)
    _surface_weights: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.parameters, ScientificTable) or self.parameters.kind != "parameters":
            raise ValueError("Thevenin diffusion requires a sourced parameter table")
        _text(self.parameter_row_id, "parameter_row_id")
        rows = {r.row_id: r for r in self.parameters.rows}
        if self.parameter_row_id not in rows:
            raise ValueError("Thevenin diffusion parameter row does not exist")
        row = rows[self.parameter_row_id]
        if set(row.values) != {TAU_COLUMN}:
            raise ValueError("diffusion parameter row requires only diffusion_time_constant_s")
        tau = convert_value(row.values[TAU_COLUMN].value, row.values[TAU_COLUMN].unit, "s")
        if not isfinite(tau) or tau <= 0:
            raise ValueError("diffusion time constant must be finite and positive")
        if self.convention != DIFFUSION_CONVENTION:
            raise ValueError("unsupported diffusion voltage/heat/surface convention")
        faces = tuple(_number(v, "normalized face") for v in _sequence(self.faces, "faces"))
        if len(faces) < 3 or faces[-1] != 1.:
            raise ValueError("normalized diffusion needs at least two cells ending at one")
        mesh = DiffusionMesh("planar", faces)
        weights = mesh.cell_volumes_m3
        diffusion = ConservativeDiffusion(mesh, 1./tau)
        operator = (sparse.diags(1./weights) @ diffusion.inventory_jacobian_s_inv
                    @ sparse.diags(weights)).tocsr()
        # Use deviations in the matrix product: the constant mean's derivative
        # is exactly zero, without amplifying floating cancellation on fine grids.
        surface_weights = np.zeros(mesh.cells)
        ratio = (1.-mesh.centers_m[-1])/(mesh.centers_m[-1]-mesh.centers_m[-2])
        surface_weights[-2:] = (-ratio, 1.+ratio)
        surface_weights.setflags(write=False)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "tau_s", tau)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "centers", mesh.centers_m)
        object.__setattr__(self, "_operator", operator)
        object.__setattr__(self, "_surface_weights", surface_weights)

    @property
    def coordinate_keys(self):
        return tuple(f"thevenin:diffusion_deviation_{i}" for i in range(len(self.weights)-1))

    @property
    def coordinate_declarations(self):
        return tuple(CoupledCoordinate(key, "dimensionless", 1.) for key in self.coordinate_keys)

    def deviations(self, coordinates):
        values = np.array([coordinates[key] for key in self.coordinate_keys], dtype=float)
        return np.r_[values, -np.dot(self.weights[:-1], values)/self.weights[-1]]

    def profile(self, mean_soc, coordinates):
        return float(mean_soc)+self.deviations(coordinates)

    def surface_soc(self, mean_soc, coordinates):
        return float(mean_soc)+float(self._surface_weights @ self.deviations(coordinates))

    @property
    def surface_deviation_gradient(self):
        """Surface derivative w.r.t. independent deviations, for local analysis."""
        return (self._surface_weights[:-1]
                - self._surface_weights[-1]*self.weights[:-1]/self.weights[-1])

    def deviation_rates(self, coordinates, current_A, capacity_C):
        if not isfinite(current_A) or not isfinite(capacity_C) or capacity_C <= 0:
            raise ValueError("diffusion current/capacity must be finite and capacity positive")
        rates = self._operator @ self.deviations(coordinates)
        # The boundary sink is in the eliminated last cell. Adding I/Q to the
        # retained rates removes mean_SOC'=-I/Q, preserving the sole charge state.
        return dict(zip(self.coordinate_keys, rates[:-1]+current_A/capacity_C, strict=True))

    def validate_profile(self, mean_soc, coordinates, lower, upper):
        profile = self.profile(mean_soc, coordinates)
        surface = self.surface_soc(mean_soc, coordinates)
        guard = 32*np.spacing(1.)
        if (not np.all(np.isfinite(profile)) or not isfinite(surface)
                or min(float(profile.min()), surface) < lower-guard
                or max(float(profile.max()), surface) > upper+guard):
            raise ValueError("Thevenin internal/surface SOC is outside its physical domain")

    def to_dict(self):
        return {"parameters": self.parameters.to_dict(), "parameter_row_id": self.parameter_row_id,
                "faces": list(self.faces), "convention": self.convention}

    @classmethod
    def from_dict(cls, payload):
        _keys(payload, ("parameters", "parameter_row_id", "faces", "convention"))
        return cls(ScientificTable.from_dict(payload["parameters"]), payload["parameter_row_id"],
                   payload["faces"], payload["convention"])


__all__ = ["DIFFUSION_CONVENTION", "TheveninDiffusionSpec"]
