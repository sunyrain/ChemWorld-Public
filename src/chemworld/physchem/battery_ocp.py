"""Sourced directional OCP and persistent Axen electrode states.

Equations follow the actual PyBaMM 25.4.2 current-sigmoid and Axen
submodels. These operational potentials do not replace equilibrium OCV or
declare a hysteresis chemical-energy law. The caller owns the one shared
electrode state and integrates it through the common coupled solver.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from math import isfinite, tanh
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Self, cast

from chemworld.data.scientific_inputs import ScientificTable, _keys, _text
from chemworld.foundation.units import convert_value
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.coupled_system import CoupledCoordinate, CoupledStateView
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL as F

if TYPE_CHECKING:
    from chemworld.physchem.battery_composite import BatteryCompositeSPMSpec
    from chemworld.physchem.battery_models import BatterySPMSpec
    from chemworld.physchem.battery_plating import _PlatingSPMAdapter

OCP_CONVENTION = "pybamm-25.4.2-ocp"
OCP_UNITS = {
    "current_sigmoid": {"collector_area_m2": "m^2", "sigmoid_slope_m2_A": "m^2/A"},
    "axen": dict.fromkeys(("initial_h", "lithiation_decay", "delithiation_decay"), "dimensionless"),
}


def hysteresis_key(electrode: str) -> str:
    if electrode not in {"negative", "positive"}:
        raise ValueError("unknown OCP electrode")
    return f"battery:{electrode}:hysteresis"


@dataclass(frozen=True)
class ElectrodeOCPSpec:
    mode: str = "single"
    lithiation: ElectrodeOCV | None = None
    delithiation: ElectrodeOCV | None = None
    parameters: ScientificTable | None = None
    parameter_row_id: str = ""
    convention: str = OCP_CONVENTION
    _native: Mapping[str, float] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.mode not in {"single", *OCP_UNITS} or self.convention != OCP_CONVENTION:
            raise ValueError("unsupported operational OCP mode or convention")
        if self.mode == "single":
            if (
                self.lithiation is not None
                or self.delithiation is not None
                or self.parameters is not None
                or self.parameter_row_id
            ):
                raise ValueError("single OCP uses the unchanged equilibrium curve only")
            object.__setattr__(self, "_native", MappingProxyType({}))
            return
        if (not isinstance(self.lithiation, ElectrodeOCV)
                or not isinstance(self.delithiation, ElectrodeOCV)):
            raise ValueError("directional OCP requires explicit sourced electrode branches")
        if self.lithiation.material_id != self.delithiation.material_id:
            raise ValueError("OCP branches must describe the same declared material")
        if not isinstance(self.parameters, ScientificTable) or self.parameters.kind != "parameters":
            raise ValueError("directional OCP requires a sourced parameter table")
        _text(self.parameter_row_id, "parameter_row_id")
        rows = {r.row_id: r for r in self.parameters.rows}
        if self.parameter_row_id not in rows:
            raise ValueError("OCP parameter row does not exist")
        cells = rows[self.parameter_row_id].values
        if cells.keys() != OCP_UNITS[self.mode].keys():
            raise ValueError("OCP parameters must match the selected equation")
        if any(v.quality != "valid" or v.value is None for v in cells.values()):
            raise ValueError("OCP requires valid finite parameter values")
        values = {
            k: convert_value(cast(float, cells[k].value), cells[k].unit, unit)
            for k, unit in OCP_UNITS[self.mode].items()
        }
        if any(not isfinite(v) or (k != "initial_h" and v <= 0) for k, v in values.items()):
            raise ValueError("OCP scales and decay parameters must be positive and finite")
        if self.mode == "axen" and not -1 <= values["initial_h"] <= 1:
            raise ValueError("initial hysteresis state must be in [-1, 1]")
        if self.mode == "current_sigmoid":
            gain = values["sigmoid_slope_m2_A"] / values["collector_area_m2"]
            if not isfinite(gain) or gain <= 0:
                raise ValueError("OCP current-density gain is outside the finite domain")
        object.__setattr__(self, "_native", MappingProxyType(values))

    @property
    def native_parameters(self) -> Mapping[str, float]:
        return self._native

    def derivatives(
        self, electrode: str, theta: float, current_A: float, h: float | None = None,
    ) -> tuple[float, float, float, float]:
        """Return (U, dU/dtheta, dU/dI, dU/dh); theta permits solver trials."""
        hysteresis_key(electrode)
        if self.mode == "single":
            raise ValueError("single OCP is evaluated using its owning spec.ocv")
        if not isfinite(current_A):
            raise ValueError("OCP current must be finite")
        # Non-single modes require both source curves at construction.
        lith, dlith = cast(ElectrodeOCV, self.lithiation).derivatives(theta, trial=True)
        de, dde = cast(ElectrodeOCV, self.delithiation).derivatives(theta, trial=True)
        if self.mode == "current_sigmoid":
            gain = self._native["sigmoid_slope_m2_A"] / self._native["collector_area_m2"]
            gain *= -1 if electrode == "negative" else 1
            t = tanh(gain * current_A)
            weight = (1 + t) / 2
            di, dh = 0.5 * gain * (1 - t * t) * (lith - de), 0.0
        else:
            # Newton trials may extend h infinitesimally. Accepted states are
            # checked separately; no trial or accepted state is clipped.
            if h is None or not isfinite(h):
                raise ValueError("Axen OCP requires its persistent finite state")
            weight = (1 - h) / 2
            di, dh = 0.0, (de - lith) / 2
        return (weight * lith + (1 - weight) * de, weight * dlith + (1 - weight) * dde, di, dh)

    def rate_derivatives(
        self, electrode: str, current_A: float, h: float, capacity_mol: float,
    ) -> tuple[float, float, float]:
        """Axen h rate, current partial and state partial in physical SI units."""
        if self.mode != "axen":
            return 0.0, 0.0, 0.0
        if not isfinite(capacity_mol) or capacity_mol <= 0:
            raise ValueError("Axen requires actual positive total electrode capacity")
        sign = 1 if electrode == "negative" else -1
        q = sign * current_A
        if q == 0:
            # No unique two-sided current derivative is claimed at rest.
            return 0.0, 0.0, 0.0
        direction = 1 if q > 0 else -1
        decay = self._native["delithiation_decay" if q > 0 else "lithiation_decay"]
        gain = decay / F / capacity_mol
        return (
            gain * abs(q) * (direction - h),
            gain * direction * sign * (direction - h),
            -gain * abs(q),
        )

    def reaction_rate_derivatives(
        self, delithiation_current_A: float, h: float, capacity_mol: float,
    ) -> tuple[float, float, float]:
        """Axen rate and partials in the actual main-reaction current.

        A side reaction changes the negative particle current without changing
        the terminal current. The negative-electrode convention of the existing
        equation is already delithiation-positive, so reuse it directly. This
        does not change current-sigmoid's collector-current convention.
        """
        return self.rate_derivatives("negative", delithiation_current_A, h, capacity_mol)

    @cached_property
    def cv_monotone_certified(self) -> bool:
        """Conservative source-domain certificate, never a sampled root guess.

        Nonpositive branch slopes and U_lith <= U_delith imply each passive
        BV terminal map is strictly decreasing. Interval enclosures certify
        these conditions; declarations not proven here remain usable under CC.
        """
        if self.mode == "single":
            return True
        lithiation = cast(ElectrodeOCV, self.lithiation)
        delithiation = cast(ElectrodeOCV, self.delithiation)
        branches = (lithiation, delithiation)
        lower = max(c.stoichiometry_domain[0] for c in branches)
        upper = min(c.stoichiometry_domain[1] for c in branches)
        if lower >= upper:
            return False
        identical = all(
            getattr(self.lithiation, k) == getattr(self.delithiation, k)
            for k in ("polynomial_V", "exponential_terms", "tanh_terms", "endpoint_policy")
        )

        def constant(curve: ElectrodeOCV) -> bool:
            return (
                curve.endpoint_policy is None
                and not any(curve.polynomial_V[1:])
                and all(a * k == 0 for a, k, _ in curve.exponential_terms)
                and all(a * k == 0 for a, k, _ in curve.tanh_terms)
            )

        for i in range(64):
            a, b = lower + (upper - lower) * i / 64, lower + (upper - lower) * (i + 1) / 64
            if any(
                not constant(c) and c.enclosure(a, b, derivative=True).upper > 0
                for c in branches
            ):
                return False
            if (
                self.mode == "current_sigmoid"
                and not identical
                and (lithiation.enclosure(a, b) - delithiation.enclosure(a, b)).upper > 0
            ):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "lithiation": None if self.lithiation is None else self.lithiation.to_dict(),
            "delithiation": None if self.delithiation is None else self.delithiation.to_dict(),
            "parameters": None if self.parameters is None else self.parameters.to_dict(),
            "parameter_row_id": self.parameter_row_id,
            "convention": self.convention,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        _keys(
            value,
            ("mode", "lithiation", "delithiation", "parameters", "parameter_row_id", "convention"),
        )
        return cls(
            value["mode"],
            None if value["lithiation"] is None else ElectrodeOCV.from_dict(value["lithiation"]),
            None
            if value["delithiation"] is None
            else ElectrodeOCV.from_dict(value["delithiation"]),
            None if value["parameters"] is None else ScientificTable.from_dict(value["parameters"]),
            value["parameter_row_id"],
            value["convention"],
        )


def hysteresis_coordinates(
    spec: BatterySPMSpec | BatteryCompositeSPMSpec | _PlatingSPMAdapter,
) -> tuple[CoupledCoordinate, ...]:
    # Do not declare endpoint root events: h=+/-1 is an admissible invariant
    # endpoint, not a termination. Strict accepted-state validation is separate.
    return tuple(CoupledCoordinate(key, "dimensionless", 1) for key in spec.hysteresis_keys)


def validate_hysteresis(
    spec: BatterySPMSpec | BatteryCompositeSPMSpec | _PlatingSPMAdapter, state: CoupledStateView,
) -> None:
    for key in spec.hysteresis_keys:
        if key not in state.coordinates or not isfinite(state.coordinates[key]):
            raise ValueError("missing or nonfinite persistent electrode hysteresis state")
        if not -1 <= state.coordinates[key] <= 1:
            raise ValueError("electrode hysteresis state is outside [-1, 1]")


def hysteresis_rates(
    spec: BatterySPMSpec | BatteryCompositeSPMSpec | _PlatingSPMAdapter,
    state: CoupledStateView, current_A: float,
    *, reaction_currents_A: Mapping[str, float] | None = None,
) -> dict[str, float]:
    if reaction_currents_A is not None:
        return {
            hysteresis_key(name): ocp.reaction_rate_derivatives(
                reaction_currents_A[name], state.coordinates[hysteresis_key(name)],
                getattr(spec, name).capacity_mol,
            )[0]
            for name in ("negative", "positive")
            if (ocp := getattr(spec, name + "_ocp")) is not None and ocp.mode == "axen"
        }
    return {
        hysteresis_key(name): ocp.rate_derivatives(
            name,
            current_A,
            state.coordinates[hysteresis_key(name)],
            getattr(spec, name).capacity_mol,
        )[0]
        for name in ("negative", "positive")
        if (ocp := getattr(spec, name + "_ocp")) is not None and ocp.mode == "axen"
    }


def validate_ocp_cv(spec: BatterySPMSpec | BatteryCompositeSPMSpec | _PlatingSPMAdapter) -> None:
    if any(
        c is not None and not c.cv_monotone_certified
        for c in (spec.negative_ocp, spec.positive_ocp)
    ):
        raise ValueError(
            "operational OCP CV monotonicity is uncertified; use explicit current control"
        )
