"""SI equipment geometry, independent from heat-transfer or material models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import acos, asin, atan, cos, degrees, hypot, isclose, isfinite, pi, radians, sin
from typing import TYPE_CHECKING, Any

from scipy.special import ellipe

from chemworld.physchem.flow_meters import _positive

if TYPE_CHECKING:
    from chemworld.physchem.transport import PipeSpec


@dataclass(frozen=True)
class EquipmentGeometry:
    model: str
    inputs: dict[str, Any]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "inputs": dict(self.inputs), "metrics": dict(self.metrics)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EquipmentGeometry:
        factories: dict[str, Callable[..., EquipmentGeometry]] = {
            "helical_coil": helical_coil_geometry,
            "plate_exchanger": plate_exchanger_geometry,
            "fin_exchanger": fin_exchanger_geometry,
            "air_cooled": air_cooled_geometry,
        }
        if value["model"] not in factories:
            raise ValueError("unknown equipment geometry")
        return factories[value["model"]](**value["inputs"])

    def tube_pipe(self, *, roughness_m: float = 0) -> PipeSpec:
        """Consume a coil's actual internal bore and developed tube length."""
        from chemworld.physchem.transport import PipeSpec

        if self.model != "helical_coil" or "Di" not in self.metrics:
            raise ValueError("coil with declared internal bore required")
        return PipeSpec(self.metrics["tube_length"], self.metrics["Di"], roughness_m)


def solve_tube_pitch(
    *,
    angle_degrees: float | None = None,
    pitch_m: float | None = None,
    parallel_m: float | None = None,
    normal_m: float | None = None,
) -> tuple[float, float, float, float]:
    """Exactly two geometric inputs give angle/pitch/parallel/normal in SI."""
    if sum(v is not None for v in (angle_degrees, pitch_m, parallel_m, normal_m)) != 2:
        raise ValueError("exactly two tube-pitch quantities required")
    for value in (pitch_m, parallel_m, normal_m):
        if value is not None and (not isfinite(value) or value < 0):
            raise ValueError("nonnegative finite pitch dimensions required")
    if angle_degrees is not None:
        if not isfinite(angle_degrees) or not 0 <= angle_degrees <= 90:
            raise ValueError("pitch angle in [0,90] required")
        angle = radians(angle_degrees)
        if pitch_m is None:
            divisor = sin(angle) if normal_m is not None else cos(angle)
            if abs(divisor) < 1e-15:
                raise ValueError("pitch is indeterminate at this angle")
            component = normal_m if normal_m is not None else parallel_m
            assert component is not None
            pitch_m = component / divisor
        parallel_m, normal_m = pitch_m * cos(angle), pitch_m * sin(angle)
    elif pitch_m is not None:
        _positive(pitch_m, "pitch")
        coordinate = normal_m if normal_m is not None else parallel_m
        assert coordinate is not None
        if coordinate > pitch_m:
            raise ValueError("pitch component exceeds pitch")
        angle = asin(coordinate / pitch_m) if normal_m is not None else acos(coordinate / pitch_m)
        angle_degrees = degrees(angle)
        parallel_m, normal_m = pitch_m * cos(angle), pitch_m * sin(angle)
    else:
        assert parallel_m is not None and normal_m is not None
        pitch_m = hypot(parallel_m, normal_m)
        _positive(pitch_m, "pitch")
        angle_degrees = degrees(asin(normal_m / pitch_m))
    assert angle_degrees is not None and parallel_m is not None and normal_m is not None
    _positive(pitch_m, "pitch")
    return angle_degrees, pitch_m, parallel_m, normal_m


def corrugated_plate_enlargement(amplitude_m: float, wavelength_m: float) -> float:
    _positive(amplitude_m, "corrugation amplitude")
    _positive(wavelength_m, "corrugation wavelength")
    return 2 * float(ellipe(-((2 * pi * amplitude_m / wavelength_m) ** 2))) / pi


def helical_coil_geometry(
    *,
    tube_diameter_m: float,
    coil_diameter_m: float | None = None,
    total_diameter_m: float | None = None,
    pitch_m: float | None = None,
    height_m: float | None = None,
    turns: float | None = None,
    total_height_m: float | None = None,
    inner_diameter_m: float | None = None,
) -> EquipmentGeometry:
    inputs = locals().copy()
    dt = tube_diameter_m
    _positive(dt, "tube diameter")
    if total_diameter_m is not None:
        if coil_diameter_m is not None and not isclose(total_diameter_m, coil_diameter_m + dt):
            raise ValueError("conflicting coil diameters")
        coil_diameter_m = total_diameter_m - dt
    if total_height_m is not None:
        if height_m is not None and not isclose(total_height_m, height_m + dt):
            raise ValueError("conflicting coil heights")
        height_m = total_height_m - dt
    if coil_diameter_m is None:
        raise ValueError("coil diameter required")
    _positive(coil_diameter_m, "coil diameter")
    if sum(v is not None for v in (pitch_m, height_m, turns)) != 2:
        raise ValueError("exactly two of pitch, height and turns required")
    if turns is None:
        assert height_m is not None and pitch_m is not None
        _positive(pitch_m, "pitch")
        turns = height_m / pitch_m
    elif pitch_m is None:
        assert height_m is not None
        _positive(turns, "turns")
        pitch_m = height_m / turns
    else:
        height_m = turns * pitch_m
    assert height_m is not None
    for value in (pitch_m, turns, height_m):
        _positive(value, "coil dimension")
    if (pitch_m < dt and not isclose(pitch_m, dt, rel_tol=1e-12)) or coil_diameter_m < dt:
        raise ValueError("coil tubes collide")
    if pitch_m < dt:
        pitch_m = dt
        height_m = turns * pitch_m
    circumference = pi * coil_diameter_m
    length = hypot(circumference * turns, height_m)
    angle = atan(pitch_m / circumference)
    area = pi * dt**2 / 4
    metrics: dict[str, Any] = {
        "Dt": dt,
        "Do": coil_diameter_m,
        "Do_total": coil_diameter_m + dt,
        "N": turns,
        "pitch": pitch_m,
        "H": height_m,
        "H_total": height_m + dt,
        "tube_circumference": circumference,
        "tube_length": length,
        "surface_area": pi * dt * length,
        "helix_angle": angle,
        "curvature": dt / coil_diameter_m / (1 + 4 * pitch_m**2 / coil_diameter_m**2),
        "total_inlet_area": area,
        "total_volume": area * length,
    }
    if inner_diameter_m is not None:
        _positive(inner_diameter_m, "internal tube diameter")
        if inner_diameter_m > dt:
            raise ValueError("internal bore exceeds tube diameter")
        inner = pi * inner_diameter_m**2 / 4
        metrics.update(
            Di=inner_diameter_m,
            inner_surface_area=pi * inner_diameter_m * length,
            inlet_area=inner,
            inner_volume=inner * length,
            annulus_area=area - inner,
            annulus_volume=(area - inner) * length,
        )
    return EquipmentGeometry("helical_coil", inputs, metrics)


def plate_exchanger_geometry(
    *,
    amplitude_m: float,
    wavelength_m: float,
    chevron_angles: tuple[float, float] = (45, 45),
    width_m: float | None = None,
    length_m: float | None = None,
    thickness_m: float | None = None,
    port_diameter_m: float | None = None,
    plates: int | None = None,
) -> EquipmentGeometry:
    inputs = locals().copy()
    enlargement = corrugated_plate_enlargement(amplitude_m, wavelength_m)
    if len(chevron_angles) != 2 or any(not 0 <= a <= 90 for a in chevron_angles):
        raise ValueError("two physical chevron angles required")
    angle = sum(chevron_angles) / 2
    metrics: dict[str, Any] = {
        "amplitude": amplitude_m,
        "wavelength": wavelength_m,
        "b": 2 * amplitude_m,
        "chevron_angles": tuple(chevron_angles),
        "chevron_angle": angle,
        "inclination_angle": 90 - angle,
        "plate_corrugation_aspect_ratio": 4 * amplitude_m / wavelength_m,
        "plate_enlargement_factor": enlargement,
        "D_eq": 4 * amplitude_m,
        "D_hydraulic": 4 * amplitude_m / enlargement,
    }
    for key, value in (
        ("width", width_m),
        ("length", length_m),
        ("thickness", thickness_m),
        ("d_port", port_diameter_m),
    ):
        if value is not None:
            _positive(value, key)
            metrics[key] = value
    if port_diameter_m is not None and length_m is not None:
        metrics["length_port"] = length_m + port_diameter_m
    if width_m is not None:
        metrics["A_channel_flow"] = width_m * 2 * amplitude_m
    if width_m is not None and length_m is not None:
        metrics["A_plate_surface"] = width_m * length_m * enlargement
    if plates is not None:
        if isinstance(plates, bool) or int(plates) != plates or plates < 3:
            raise ValueError("at least three whole plates required")
        metrics.update(plates=plates, channels=plates - 1, channels_per_fluid=(plates - 1) / 2)
        if "A_plate_surface" in metrics:
            metrics["A_heat_transfer"] = (plates - 2) * metrics["A_plate_surface"]
    return EquipmentGeometry("plate_exchanger", inputs, metrics)


def fin_exchanger_geometry(
    *,
    fin_height_m: float,
    fin_thickness_m: float,
    fin_spacing_m: float,
    offset_length_m: float | None = None,
    length_m: float | None = None,
    width_m: float | None = None,
    layers: int | None = None,
    plate_thickness_m: float | None = None,
    flow: str = "crossflow",
) -> EquipmentGeometry:
    inputs = locals().copy()
    h, t, s = fin_height_m, fin_thickness_m, fin_spacing_m
    for value in (h, t, s):
        _positive(value, "fin dimension")
    if t >= min(h, s):
        raise ValueError("fin thickness blocks channel")
    area = (s - t) * (h - t)
    perimeter = 2 * (s + h - 2 * t)
    metrics: dict[str, Any] = {
        "fin_height": h,
        "fin_thickness": t,
        "fin_spacing": s,
        "channel_height": h - t,
        "channel_width": s - t,
        "fin_count": 1 / s,
        "A_channel": area,
        "P_channel": perimeter,
        "Dh": 4 * area / perimeter,
        "blockage_ratio": area / (s * h),
        "flow": flow,
        "length": length_m,
        "width": width_m,
        "layers": layers,
        "plate_thickness": plate_thickness_m,
    }
    if offset_length_m is not None:
        _positive(offset_length_m, "strip length")
        if s <= 2 * t:
            raise ValueError("offset strip requires positive staggered opening")
        length = offset_length_m
        surface = 2 * (length * (h - t) + length * (s - t) + t * (h - t)) + t * (s - 2 * t)
        metrics.update(
            fin_length=length,
            blockage_ratio=2 * t / s * (1 - t / h) + t / h * (1 - 2 * t / s),
            blockage_ratio_Kim=t / h + t / s - t * t / (h * s),
            alpha=s / h,
            delta=t / length,
            gamma=t / s,
            A=surface,
            Dh=4 * length * area / surface,
            Dh_Kays_London=4 * area / perimeter,
            Dh_Joshi_Webb=2
            * length
            * (h - t)
            * (s - 2 * t)
            / (length * (h - t) + length * (s - t) + t * (h - t)),
        )
    for dimension in (length_m, width_m, plate_thickness_m):
        if dimension is not None:
            _positive(dimension, "overall exchanger dimension")
    if layers is not None and (isinstance(layers, bool) or int(layers) != layers or layers <= 0):
        raise ValueError("positive whole layer count required")
    if length_m is not None and width_m is not None:
        count = round(width_m / s, 0)
        layer_area = perimeter * length_m * count
        metrics.update(layer_fin_count=count, A_HX_layer=layer_area)
        if layers is not None:
            metrics["A_HX"] = layers * layer_area
            if plate_thickness_m is not None:
                # Channel fin height plus one separator per layer and two outer faces.
                height = (h + plate_thickness_m) * layers + plate_thickness_m
                metrics.update(
                    layer_thickness=h + plate_thickness_m,
                    height=height,
                    volume=length_m * width_m * height,
                    A_specific_HX=layers * layer_area / (length_m * width_m * height),
                )
    return EquipmentGeometry("fin_exchanger", inputs, metrics)


def air_cooled_geometry(
    *,
    tube_rows: int,
    tube_passes: int,
    tubes_per_row: int,
    tube_length_m: float,
    tube_diameter_m: float,
    fin_thickness_m: float,
    angle_degrees: float | None = None,
    pitch_m: float | None = None,
    parallel_pitch_m: float | None = None,
    normal_pitch_m: float | None = None,
    pitch_ratio: float | None = None,
    fin_diameter_m: float | None = None,
    fin_height_m: float | None = None,
    fin_density_per_m: float | None = None,
    fin_interval_m: float | None = None,
    parallel_bays: int = 1,
    bundles_per_bay: int = 1,
    fans_per_bay: int = 1,
    corbels: bool = False,
    tube_thickness_m: float | None = None,
    fan_diameter_m: float | None = None,
) -> EquipmentGeometry:
    inputs = locals().copy()
    diameter, length, thickness = tube_diameter_m, tube_length_m, fin_thickness_m
    for value in (diameter, length, thickness):
        _positive(value, "tube/fin dimension")
    for value in (
        tube_rows,
        tube_passes,
        tubes_per_row,
        parallel_bays,
        bundles_per_bay,
        fans_per_bay,
    ):
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError("positive whole equipment count required")
    if pitch_ratio is not None:
        if pitch_m is not None:
            raise ValueError("supply pitch or pitch ratio, not both")
        _positive(pitch_ratio, "pitch ratio")
        pitch_m = diameter * pitch_ratio
    angle, pitch, parallel, normal = solve_tube_pitch(
        angle_degrees=angle_degrees,
        pitch_m=pitch_m,
        parallel_m=parallel_pitch_m,
        normal_m=normal_pitch_m,
    )
    if (fin_diameter_m is None) == (fin_height_m is None):
        raise ValueError("one fin diameter/height required")
    if fin_diameter_m is not None:
        fin_height_m = (fin_diameter_m - diameter) / 2
    else:
        assert fin_height_m is not None
        fin_diameter_m = diameter + 2 * fin_height_m
    assert fin_height_m is not None
    _positive(fin_height_m, "fin height")
    if (fin_density_per_m is None) == (fin_interval_m is None):
        raise ValueError("one fin density/interval required")
    if fin_interval_m is not None:
        _positive(fin_interval_m, "fin interval")
        fin_density_per_m = 1 / fin_interval_m
    else:
        assert fin_density_per_m is not None
        _positive(fin_density_per_m, "fin density")
        fin_interval_m = 1 / fin_density_per_m
    assert fin_density_per_m is not None
    if fin_interval_m <= thickness:
        raise ValueError("fin thickness blocks air passage")
    diagonal = hypot(parallel, 0.5 * normal)
    counts = {
        "per_tube": 1,
        "per_row": tubes_per_row,
        "per_bundle": tubes_per_row * tube_rows,
        "per_bay": tubes_per_row * tube_rows * bundles_per_bay,
        "": tubes_per_row * tube_rows * bundles_per_bay * parallel_bays,
    }
    bare = pi * diameter * length
    showing = bare * (1 - thickness / fin_interval_m)
    per_fin = pi / 2 * (fin_diameter_m**2 - diameter**2) + pi * fin_diameter_m * thickness
    fin_area = fin_density_per_m * length * per_fin
    metrics: dict[str, Any] = {
        "angle": angle,
        "pitch": pitch,
        "pitch_ratio": pitch / diameter,
        "pitch_parallel": parallel,
        "pitch_normal": normal,
        "pitch_diagonal": diagonal,
        "fin_height": fin_height_m,
        "fin_diameter": fin_diameter_m,
        "fin_interval": fin_interval_m,
        "fin_density": fin_density_per_m,
        "bare_length": fin_interval_m - thickness,
        "tubes_per_bundle": counts["per_bundle"],
        "tubes_per_bay": counts["per_bay"],
        "tubes": counts[""],
        "A_per_fin": per_fin,
        "A_increase": (showing + fin_area) / bare,
    }
    for category, unit in (
        ("A_bare_tube", bare),
        ("A_tube_showing", showing),
        ("A_fin", fin_area),
        ("A", showing + fin_area),
    ):
        for suffix, count in counts.items():
            metrics[category + ("_" + suffix if suffix else "")] = unit * count
    blocked = diameter + 2 * fin_density_per_m * fin_height_m * thickness
    diagonal_area = 2 * tubes_per_row * length * (diagonal - blocked)
    normal_area = tubes_per_row * length * (normal - blocked)
    minimum = min(diagonal_area, normal_area)
    if minimum <= 0:
        raise ValueError("tube/fin bundle blocks air flow")
    face = normal * length * (tubes_per_row + (0.5 if corbels else 1))
    metrics.update(
        A_diagonal_per_bundle=diagonal_area,
        A_normal_per_bundle=normal_area,
        A_min_per_bundle=minimum,
        A_min_per_bay=minimum * bundles_per_bay,
        A_min=minimum * bundles_per_bay * parallel_bays,
        A_face_per_bundle=face,
        A_face_per_bay=face * bundles_per_bay,
        A_face=face * bundles_per_bay * parallel_bays,
        flow_area_contraction_ratio=minimum / face,
        channels=counts["per_bundle"] / tube_passes
        if tube_rows % tube_passes == 0
        else tubes_per_row,
    )
    pitch_names: dict[float, tuple[str, str]] = {
        30: ("triangular", "staggered"),
        60: ("rotated triangular", "staggered"),
        45: ("rotated square", "in-line"),
        90: ("square", "in-line"),
    }
    metrics["pitch_str"], metrics["pitch_class"] = pitch_names.get(angle, ("custom", "custom"))
    if tube_thickness_m is not None:
        _positive(tube_thickness_m, "tube thickness")
        inner = diameter - 2 * tube_thickness_m
        _positive(inner, "tube internal bore")
        flow_area = pi * inner**2 / 4
        metrics["Di"] = inner
        for suffix, count in counts.items():
            metrics[
                "A_tube_flow"
                + ("_total" if suffix == "" else "" if suffix == "per_tube" else "_" + suffix)
            ] = flow_area * count
            metrics["tube_volume" + ("_" + suffix if suffix else "")] = flow_area * length * count
    return EquipmentGeometry("air_cooled", inputs, metrics)
