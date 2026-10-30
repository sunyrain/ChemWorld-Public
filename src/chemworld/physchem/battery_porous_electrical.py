"""Shared finite-volume porous-electrode charge integration, with no state/solver.

The DFN and Newman-Tobias models use the identical geometry map. Reaction
currents are cell-integrated amperes; solid conductivities are effective values.
"""

from dataclasses import asdict, dataclass
from functools import cached_property
from math import isfinite
from numbers import Integral

import numpy as np

from chemworld.physchem.finite_volume import DiffusionMesh

PHI_E = "battery:electrolyte-offset_V"
ELECTRICAL_KEYS = ("battery:current_A", "battery:voltage_V")


@dataclass(frozen=True)
class PorousElectrodeGeometrySpec:
    area_m2: float = 0.02
    lengths_m: tuple[float, float, float] = (1e-4, 2.5e-5, 1e-4)
    porosities: tuple[float, float, float] = (0.4, 0.5, 0.4)
    bruggeman: tuple[float, float, float] = (1.5, 1.5, 1.5)
    cells: tuple[int, int, int] = (8, 8, 8)
    conductivity_S_m: float = 1.0
    solid_conductivity_S_m: tuple[float, float] = (100.0, 100.0)

    def __post_init__(self) -> None:
        for name, size in (
            ("lengths_m", 3),
            ("porosities", 3),
            ("bruggeman", 3),
            ("cells", 3),
            ("solid_conductivity_S_m", 2),
        ):
            values = tuple(getattr(self, name))
            if len(values) != size or any(
                isinstance(v, bool) or not isfinite(v) or v <= 0 for v in values
            ):
                raise ValueError(f"{name} needs {size} positive finite entries")
            object.__setattr__(self, name, values)
        if any(not isinstance(n, Integral) or n < 2 for n in self.cells):
            raise ValueError("porous regions need at least two integer cells")
        if any(p >= 1 for p in self.porosities):
            raise ValueError("porosity must be inside (0, 1)")
        for name in ("area_m2", "conductivity_S_m"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be positive and finite")
        effective = self.conductivity_S_m * np.power(self.porosities, self.bruggeman)
        if np.any(~np.isfinite(effective)) or np.any(effective <= 0):
            raise ValueError("effective ionic conductivities must be representable")

    @cached_property
    def mesh(self):
        faces = [0.0]
        for length, count in zip(self.lengths_m, self.cells, strict=True):
            faces.extend(faces[-1] + np.linspace(0, length, count + 1)[1:])
        return DiffusionMesh("planar", tuple(faces), area_m2=self.area_m2)

    def to_dict(self):
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


def reaction_keys(geometry):
    return tuple(
        f"battery:{name}:reaction:{i}_A"
        for name, region in (("negative", 0), ("positive", 2))
        for i in range(geometry.cells[region] - 1)
    )


def reaction_currents(geometry, keys, values):
    current = values[ELECTRICAL_KEYS[0]]
    selected = np.array([values[k] for k in keys])
    split = geometry.cells[0] - 1
    negative, positive = selected[:split], selected[split:]
    return np.r_[negative, current - sum(negative), positive, -current - sum(positive)]


def integral_centres(face_values, widths, coefficients):
    """Exact integral of a linear face field from the left boundary to centres."""
    full = (face_values[:-1] + face_values[1:]) * widths / (2 * coefficients)
    half = (3 * face_values[:-1] + face_values[1:]) * widths / (8 * coefficients)
    return np.r_[0.0, np.cumsum(full)[:-1]] + half, float(sum(full))


def linear_electrical(geometry, keys, series_resistance_ohm, values):
    current, voltage = (values[key] for key in ELECTRICAL_KEYS)
    q = reaction_currents(geometry, keys, values)
    return electrical_from_cell_currents(
        geometry, q, current, voltage, values[PHI_E], series_resistance_ohm
    )


def electrical_from_cell_currents(
    geometry, q, current, voltage, electrolyte_offset_V, series_resistance_ohm
):
    """Integrate the same charge equations from complete electrode cell currents.

    ``q`` contains negative then positive cell-integrated reaction currents (A).
    Callers enforce electrode totals; multiple materials are summed before this
    geometry-only map. The legacy wrapper retains its original reconstruction.
    """
    nn, ns, _ = geometry.cells
    source = np.r_[q[:nn], np.zeros(ns), q[nn:]]
    ionic_faces = np.r_[0.0, np.cumsum(source)] / geometry.area_m2
    widths = np.diff(geometry.mesh.faces_m)
    conductivity = np.repeat(
        geometry.conductivity_S_m * np.power(geometry.porosities, geometry.bruggeman),
        geometry.cells,
    )
    resistance, _ = integral_centres(ionic_faces, widths, conductivity)
    phi_e = electrolyte_offset_V - resistance
    solid, solid_faces = [], []
    for name, region, start in (("negative", 0, 0), ("positive", 2, nn + ns)):
        size = geometry.cells[region]
        faces = current / geometry.area_m2 - ionic_faces[start : start + size + 1]
        integral, full = integral_centres(
            faces, widths[start : start + size], geometry.solid_conductivity_S_m[region // 2]
        )
        solid.append(
            -integral
            if name == "negative"
            else voltage + current * series_resistance_ohm + full - integral
        )
        solid_faces.append(faces)
    return q, ionic_faces, phi_e, np.concatenate(solid), solid_faces


def electrical_maps(geometry, keys, series_resistance_ohm, algebraic_names):
    """Exact linear maps from unit columns, never finite-difference nonlinear BV."""
    columns, reactions = [], []
    nn, _, np_ = geometry.cells
    for column in np.eye(len(algebraic_names)).T:
        values = dict(zip(algebraic_names, column, strict=True))
        q, _, phi_e, phi_s, _ = linear_electrical(geometry, keys, series_resistance_ohm, values)
        columns.append(phi_s - np.r_[phi_e[:nn], phi_e[-np_:]])
        reactions.append(q)
    return np.array(reactions).T, np.array(columns).T
