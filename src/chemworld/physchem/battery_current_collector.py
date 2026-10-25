"""One-dimensional current-collector algebra for collector SPM models.

The collector equations are linear once the cell reaction currents are known.
This module deliberately contains no time integration or electrochemical state;
it is the geometry/port primitive consumed by :mod:`battery_collector_spm`.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class CollectorGeometrySpec:
    """Finite-volume current-collector geometry and tab convention.

    ``z`` nodes are cell centres.  ``negative_tab`` chooses the negative
    collector gauge (the positive collector is always tabbed on the right).
    ``q`` uses the electrochemical convention: negative-electrode q is positive
    on discharge and positive-electrode q is negative on discharge.
    """

    height_m: float = 1.0e-3
    width_m: float = 1.0e-2
    negative_thickness_m: float = 1.0e-5
    positive_thickness_m: float = 1.0e-5
    negative_conductivity_S_m: float = 5.96e7
    positive_conductivity_S_m: float = 3.5e7
    cells: int = 8
    negative_tab: str = "left"

    def __post_init__(self):
        if any(
            not isfinite(value) or value <= 0.0
            for value in (
                self.height_m,
                self.width_m,
                self.negative_thickness_m,
                self.positive_thickness_m,
                self.negative_conductivity_S_m,
                self.positive_conductivity_S_m,
            )
        ):
            raise ValueError("collector dimensions and conductivities must be positive")
        if type(self.cells) is not int or self.cells < 2:
            raise ValueError("collector cells must be an integer >= 2")
        if self.negative_tab not in {"left", "right"}:
            raise ValueError("negative_tab must be left or right")

    @property
    def dz_m(self):
        return self.height_m / self.cells

    @property
    def z_centres_m(self):
        return (np.arange(self.cells, dtype=float) + 0.5) * self.dz_m

    @property
    def negative_sheet_conductance_s(self):
        return self.negative_conductivity_S_m * self.negative_thickness_m * self.width_m

    @property
    def positive_sheet_conductance_s(self):
        return self.positive_conductivity_S_m * self.positive_thickness_m * self.width_m

    @property
    def negative_green(self):
        z = self.z_centres_m
        distance = z if self.negative_tab == "left" else self.height_m - z
        return np.minimum(distance[:, None], distance[None, :]) / self.negative_sheet_conductance_s

    @property
    def positive_green(self):
        z = self.z_centres_m
        distance = self.height_m - z
        return np.minimum(distance[:, None], distance[None, :]) / self.positive_sheet_conductance_s

    def _validate_currents(self, q, name):
        result = np.asarray(q, dtype=float)
        if result.shape != (self.cells,) or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be a finite vector of length cells")
        return result

    def potentials(self, q_negative, q_positive, tab_voltage_V):
        """Return local collector potentials, with the negative tab as gauge.

        ``tab_voltage_V`` is the positive physical tab minus the negative tab.
        The electrochemical q convention is retained; no current is silently
        repartitioned or normalized.
        """
        qn = self._validate_currents(q_negative, "q_negative")
        qp = self._validate_currents(q_positive, "q_positive")
        if not isfinite(tab_voltage_V):
            raise ValueError("tab_voltage_V must be finite")
        negative = -self.negative_green @ qn
        positive = float(tab_voltage_V) - self.positive_green @ qp
        return negative, positive

    def face_currents(self, q_negative, q_positive):
        """Current through each collector face, oriented left-to-right.

        The returned arrays have ``cells + 1`` entries and include the tab
        boundaries.  They are diagnostics, while ``potentials`` is the map
        used by the coupled residual.
        """
        qn = self._validate_currents(q_negative, "q_negative")
        qp = self._validate_currents(q_positive, "q_positive")
        # Charge leaves either collector at the signed interfacial rate q.
        # Therefore i_face[j+1] - i_face[j] = -q[j]. The insulating
        # endpoint fixes the integration constant, never the cell ordering.
        neg = np.r_[0.0, -np.cumsum(qn)]
        if self.negative_tab == "left":
            neg += np.sum(qn)
            neg[-1] = 0.0
        pos = np.r_[0.0, -np.cumsum(qp)]
        return neg, pos

    def boundary_voltage_v(self, positive_potential, *, source_linear=True):
        """Source-style positive boundary voltage diagnostic.

        Finite-volume source codes commonly extrapolate the right tab from the
        last cell centre.  The native physical tab value remains the algebraic
        ``tab_voltage``; this diagnostic keeps the two outputs separate.
        """
        values = np.asarray(positive_potential, dtype=float)
        if values.shape != (self.cells,):
            raise ValueError("positive_potential has the wrong collector shape")
        if not source_linear:
            return float(values[-1])
        if self.cells == 1:
            return float(values[-1])
        slope = (values[-1] - values[-2]) / self.dz_m
        return float(values[-1] + 0.5 * self.dz_m * slope)


def collector_potentials(geometry, q_negative, q_positive, tab_voltage_V):
    """Functional wrapper used by tests and downstream readout code."""
    if not isinstance(geometry, CollectorGeometrySpec):
        raise TypeError("geometry must be CollectorGeometrySpec")
    return geometry.potentials(q_negative, q_positive, tab_voltage_V)


__all__ = ["CollectorGeometrySpec", "collector_potentials"]
