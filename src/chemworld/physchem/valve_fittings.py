"""Engineering valve resistances and signed tee coefficients on explicit bases.

Crane numerical constants define the correlations, not installation certification.
Named Darby/Hooper observations are supplied as common ScientificTables.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, radians, sin, sqrt
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.straight_friction import crane_fully_turbulent_friction

if TYPE_CHECKING:
    from chemworld.data.scientific_inputs import ScientificTable


CRANE_VALVE_TYPES = (
    "gate",
    "globe",
    "angle 0",
    "angle 1",
    "angle 2",
    "swing check straight",
    "swing check angled",
    "lift check straight",
    "lift check angled",
    "tilting check 5",
    "tilting check 15",
    "stop check globe 1",
    "stop check globe 2",
    "stop check globe 3",
    "stop check angle 1",
    "stop check angle 2",
    "stop check angle 3",
    "ball",
    "diaphragm weir",
    "diaphragm straight",
    "foot poppet",
    "foot hinged",
    "butterfly centric",
    "butterfly double offset",
    "butterfly triple offset",
    "plug straight",
    "plug three way straight",
    "plug three way right angle",
)

VALVE_LIFT_TYPES = (
    "swing check angled",
    "swing check straight",
    "swing check UL",
    "lift check straight",
    "lift check angled",
    "tilting check 5",
    "tilting check 15",
    "stop check globe 1",
    "stop check angle 1",
    "stop check globe 2",
    "stop check angle 2",
    "stop check globe 3",
    "stop check angle 3",
    "foot poppet",
    "foot hinged",
)


def _geometry(pipe: float, throat: float | None, angle: float) -> tuple[float, float]:
    _positive(pipe, "pipe diameter")
    bore = pipe if throat is None else throat
    _positive(bore, "throat diameter")
    if bore > pipe or not isfinite(angle) or not 0 <= angle <= 180:
        raise ValueError("throat must fit pipe and angle must lie in [0,180] degrees")
    return bore / pipe, radians(angle)


def crane_valve_coefficient(
    valve_type: str,
    *,
    pipe_diameter_m: float,
    throat_diameter_m: float | None = None,
    angle_degrees: float = 0,
    friction_factor: float | None = None,
) -> float:
    """K on the connected pipe velocity basis for 28 fully open valve variants."""
    if valve_type not in CRANE_VALVE_TYPES:
        raise ValueError("unknown Crane valve type")
    beta, angle = _geometry(pipe_diameter_m, throat_diameter_m, angle_degrees)
    friction = (
        crane_fully_turbulent_friction(pipe_diameter_m)
        if friction_factor is None
        else friction_factor
    )
    _positive(friction, "Darcy friction factor")
    deficits = 1 - beta**2
    size_index = 0 if pipe_diameter_m <= 0.2286 else 1 if pipe_diameter_m <= 0.381 else 2
    fixed = {
        "swing check straight": 50,
        "swing check angled": 100,
        "diaphragm weir": 149,
        "diaphragm straight": 39,
        "foot poppet": 420,
        "foot hinged": 75,
    }
    if valve_type in fixed:
        return fixed[valve_type] * friction
    if valve_type.startswith("butterfly"):
        choices = {
            "butterfly centric": (45, 35, 25),
            "butterfly double offset": (74, 52, 43),
            "butterfly triple offset": (218, 96, 55),
        }
        return choices[valve_type][size_index] * friction
    if valve_type.startswith("tilting"):
        if valve_type == "tilting check 5":
            return (40, 30, 20)[size_index] * friction
        # The 15-degree correlation uses strict thresholds, unlike the 5-degree fit.
        strict_index = 0 if pipe_diameter_m < 0.2286 else 1 if pipe_diameter_m < 0.381 else 2
        return (120, 90, 60)[strict_index] * friction
    factors = {
        "gate": 8,
        "ball": 3,
        "globe": 340,
        "angle 0": 55,
        "angle 1": 150,
        "angle 2": 55,
        "lift check straight": 600,
        "lift check angled": 55,
        "stop check globe 1": 400,
        "stop check globe 2": 300,
        "stop check globe 3": 55,
        "stop check angle 1": 200,
        "stop check angle 2": 350,
        "stop check angle 3": 55,
        "plug straight": 18,
        "plug three way straight": 30,
        "plug three way right angle": 90,
    }
    base = factors[valve_type] * friction
    if beta == 1 or (valve_type == "gate" and angle_degrees == 0):
        return base
    if valve_type in {"gate", "ball"}:
        extra = (
            sin(angle / 2) * (0.8 * deficits + 2.6 * deficits**2)
            if angle_degrees <= 45
            else 0.5 * sqrt(sin(angle / 2)) * deficits + deficits**2
        )
    elif valve_type.startswith("plug"):
        extra = 0.5 * sqrt(sin(angle / 2)) * deficits + deficits**2
    elif valve_type == "globe" or valve_type.startswith("angle"):
        extra = beta * (0.5 * (1 - beta) ** 2 + deficits**2)
    else:
        extra = beta * (0.5 * deficits + deficits**2)
    return (base + extra) / beta**4


def minimum_valve_lift_velocity(
    valve_type: str,
    *,
    density_kg_m3: float,
    throat_to_pipe_ratio: float = 1,
) -> float:
    """Minimum velocity [m/s] for the stated check-valve lift convention."""
    _positive(density_kg_m3, "density")
    if not 0 < throat_to_pipe_ratio <= 1 or valve_type not in VALVE_LIFT_TYPES:
        raise ValueError("known lift style and throat/pipe ratio in (0,1] required")
    factors = (45, 75, 120, 50, 170, 100, 40, 70, 95, 75, 75, 170, 170, 20, 45)
    factor = factors[VALVE_LIFT_TYPES.index(valve_type)]
    geometry = throat_to_pipe_ratio**2 if valve_type.startswith(("lift", "stop")) else 1
    return factor * geometry / sqrt(density_kg_m3)


@dataclass(frozen=True)
class JunctionLossCoefficients:
    branch: float
    run: float
    combined_flow_m3_s: float
    pipe_diameter_m: float
    basis: str = "combined_stream_velocity"

    def pressure_changes(self, density_kg_m3: float) -> tuple[float, float]:
        """Signed branch/run pressure differences; negative means recovery."""
        from math import pi

        _positive(density_kg_m3, "density")
        dynamic = (
            0.5
            * density_kg_m3
            * (4 * self.combined_flow_m3_s / (pi * self.pipe_diameter_m**2)) ** 2
        )
        return self.branch * dynamic, self.run * dynamic


def junction_loss_coefficients(
    mode: str,
    *,
    pipe_diameter_m: float,
    branch_diameter_m: float,
    run_flow_m3_s: float,
    branch_flow_m3_s: float,
    angle_degrees: float = 90,
) -> JunctionLossCoefficients:
    """Crane tee/wye recovery on combined-stream velocity, not branch velocity."""
    beta, angle = _geometry(pipe_diameter_m, branch_diameter_m, angle_degrees)
    if not 30 <= angle_degrees <= 90 or mode not in {"converging", "diverging"}:
        raise ValueError("declared converging/diverging junction and angle in [30,90] required")
    if any(not isfinite(v) or v < 0 for v in (run_flow_m3_s, branch_flow_m3_s)):
        raise ValueError("nonnegative finite junction flows required")
    total = run_flow_m3_s + branch_flow_m3_s
    _positive(total, "combined flow")
    q, area = branch_flow_m3_s / total, beta**2
    if mode == "converging":
        factor = 1 if area <= 0.35 else 0.9 * (1 - q) if q <= 0.4 else 0.55
        # Piecewise-linear angle coefficients defining the Crane correlation.
        f_branch = float(np.interp(angle_degrees, (30, 45, 60, 90), (1.74, 1.41, 1, 0)))
        f_run = float(np.interp(angle_degrees, (30, 45, 60), (1.74, 1.41, 1)))
        branch = factor * (1 + (q / area) ** 2 - 2 * (1 - q) ** 2 - f_branch * q * q / area)
        run = 1.55 * q - q * q if angle_degrees >= 75 else 1 - (1 - q) ** 2 - f_run * q * q / area
    else:
        h, j = (1, 2) if angle_degrees < 60 or beta <= 2 / 3 else (0.3, 0)
        if angle_degrees < 75:
            factor = (
                (1.1 - 0.7 * q if q <= 0.4 else 0.85)
                if area <= 0.35
                else (1 - 0.6 * q if q <= 0.6 else 0.6)
            )
        else:
            factor = 1 if area <= 2 / 3 else 1 + 0.3 * q * q
        branch = factor * (1 + h * (q / area) ** 2 - j * q / area * cos(angle))
        multiplier = 0.4 if area <= 0.4 else 2 * (2 * q - 1) if q <= 0.5 else 0.3 * (2 * q - 1)
        run = multiplier * q * q
    return JunctionLossCoefficients(branch, run, total, pipe_diameter_m)


def resistance_from_valve_coefficient(
    value: float, *, diameter_m: float, scale: str = "Kv"
) -> float:
    from chemworld.physchem.valve_characteristics import convert_valve_coefficient

    _positive(diameter_m, "diameter")
    _positive(value, "valve coefficient")
    kv = convert_valve_coefficient(value, from_scale=scale, to_scale="Kv")
    return 1.6e9 * diameter_m**4 / kv**2


def valve_coefficient_from_resistance(
    value: float, *, diameter_m: float, scale: str = "Kv"
) -> float:
    from chemworld.physchem.valve_characteristics import convert_valve_coefficient

    _positive(diameter_m, "diameter")
    _positive(value, "loss coefficient")
    kv = diameter_m**2 * sqrt(1.6e9 / value)
    return convert_valve_coefficient(kv, from_scale="Kv", to_scale=scale)


def change_loss_velocity_basis(
    value: float, *, from_diameter_m: float, to_diameter_m: float
) -> float:
    _positive(from_diameter_m, "original diameter")
    _positive(to_diameter_m, "target diameter")
    if not isfinite(value):
        raise ValueError("finite loss/recovery coefficient required")
    return value * (to_diameter_m / from_diameter_m) ** 4


def darby_resistance(
    reynolds: float, nominal_size: float, *, k1: float, ki: float, kd: float
) -> float:
    for value in (reynolds, nominal_size):
        _positive(value, "Reynolds/nominal size")
    if any(not isfinite(v) or v < 0 for v in (k1, ki, kd)):
        raise ValueError("finite nonnegative fitting constants required")
    return k1 / reynolds + ki * (1 + kd * nominal_size**-0.3)


def hooper_resistance(reynolds: float, diameter_m: float, *, k1: float, k_infinity: float) -> float:
    _positive(reynolds, "Reynolds number")
    _positive(diameter_m, "internal diameter")
    if any(not isfinite(v) or v < 0 for v in (k1, k_infinity)):
        raise ValueError("finite nonnegative fitting constants required")
    # Hooper's dimensional fit uses inches; the public native interface is SI.
    return k1 / reynolds + k_infinity * (1 + 0.0254 / diameter_m)


class FittingCoefficientCatalog:
    """Caller rows grouped as Darby/Hooper; sample_id is the fitting name.

    Darby columns k1/ki/kd; Hooper columns k1/k_infinity. Different parameter
    sets use separate common tables; source identities survive JSON roundtrip.
    """

    def __init__(self, table: ScientificTable, *, model: str) -> None:
        expected = {"Darby": ("k1", "ki", "kd"), "Hooper": ("k1", "k_infinity")}
        if model not in expected or table.kind != "parameters":
            raise ValueError("Darby/Hooper parameter table required")
        self.table, self.model = table, model
        self._parameters: dict[str, dict[str, float]] = {}
        for row in table.rows:
            if row.sample_id in self._parameters:
                raise ValueError("duplicate named fitting; select one source explicitly")
            cells = {name: row.values[name].to("dimensionless") for name in expected[model]}
            if any(c.quality != "valid" or cast(float, c.value) < 0 for c in cells.values()):
                raise ValueError("valid nonnegative fitting parameter cells required")
            self._parameters[row.sample_id] = {
                name: float(cast(float, c.value)) for name, c in cells.items()
            }
        if not self._parameters:
            raise ValueError("nonempty fitting parameter table required")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._parameters)

    def parameters(self, name: str) -> dict[str, float]:
        if name not in self._parameters:
            raise ValueError("fitting absent from supplied catalog")
        return dict(self._parameters[name])

    def resistance(self, name: str, *, reynolds: float, size: float) -> float:
        """size is dimensionless nominal for Darby, internal diameter [m] for Hooper."""
        values = self.parameters(name)
        if self.model == "Darby":
            return darby_resistance(reynolds, size, **values)
        return hooper_resistance(reynolds, size, **values)

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table.to_dict(), "model": self.model}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FittingCoefficientCatalog:
        from chemworld.data.scientific_inputs import ScientificTable

        return cls(ScientificTable.from_dict(value["table"]), model=value["model"])


def nominal_size_from_bore(
    table: ScientificTable, diameter_m: float, *, extrapolate: bool = False
) -> float:
    """Linear source mapping; different from selecting a manufactured pipe size."""
    _positive(diameter_m, "internal diameter")
    pairs = []
    for row in table.rows:
        diameter = row.values["inner_diameter"].to("m").value
        nominal = row.values["nominal_size"].to("dimensionless").value
        if diameter is None or nominal is None:
            raise ValueError("nominal mapping requires numeric bore and size values")
        pairs.append((float(diameter), float(nominal)))
    pairs.sort()
    if len(pairs) < 2 or len({p[0] for p in pairs}) != len(pairs):
        raise ValueError("at least two unique bore nodes required")
    x, y = zip(*pairs, strict=True)
    if not all(isfinite(v) and v > 0 for v in (*x, *y)):
        raise ValueError("finite positive nominal mapping required")
    if x[0] <= diameter_m <= x[-1]:
        return float(np.interp(diameter_m, x, y))
    if not extrapolate:
        raise ValueError("bore outside supplied nominal mapping")
    left, right = pairs[:2] if diameter_m < x[0] else pairs[-2:]
    return left[1] + (diameter_m - left[0]) * (right[1] - left[1]) / (right[0] - left[0])
