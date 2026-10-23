"""Fixed SI equipment-geometry and primitive reference comparisons."""

import argparse
import json
import time
from itertools import combinations, product
from math import cos, isclose, pi, radians, sin, sqrt
from pathlib import Path

import fluids
from fluids import geometry as ref

from chemworld.physchem.equipment_geometry import (
    air_cooled_geometry,
    fin_exchanger_geometry,
    helical_coil_geometry,
    plate_exchanger_geometry,
    solve_tube_pitch,
)
from chemworld.physchem.geometric_primitives import (
    CoolingTowerGeometry,
    circular_segment_area,
    circular_segment_height,
    cylinder_geometry,
    partial_sphere_geometry,
    shape_factor,
)


def inputs():
    for dt, dc, p, n, route in product((0.01, 0.03), (0.2, 1), (1, 2), (2, 10), range(3)):
        yield {"family": "coil", "Dt": dt, "Dc": dc, "pitch": p * dt, "turns": n, "route": route}
    for a, w, angles, scope in product(
        (0.001, 0.003), (0.005, 0.015), ((30, 30), (30, 60)), range(3)
    ):
        yield {"family": "plate", "a": a, "w": w, "angles": angles, "scope": scope}
    for h, s, t, offset, plate in product(
        (0.005, 0.01), (0.003, 0.006), (0.0001, 0.0003), (None, 0.005), (False, True)
    ):
        yield {"family": "fin_plate" if plate else "fin", "h": h, "s": s, "t": t, "offset": offset}
    for rows, passes, angle, fin_route, pitch_route in product(
        (2, 3), (1, 2), (30, 45, 60, 90), range(2), range(2)
    ):
        yield {
            "family": "air",
            "rows": rows,
            "passes": passes,
            "angle": angle,
            "fin_route": fin_route,
            "pitch_route": pitch_route,
        }
    for hi, ho, di, do, throat, route in product(
        (10, 20), (100, 150), (80, 100), (65, 75), (None, 0.6), range(2)
    ):
        yield {
            "family": "tower",
            "Hi": hi,
            "Ho": ho,
            "Di": di,
            "Do": do,
            "throat": throat,
            "route": route,
        }
    for diameter, length, holes in product((0.01, 1), (0.1, 1, 10), range(3)):
        yield {"family": "cylinder", "D": diameter, "L": length, "holes": holes}
    for diameter, fraction in product((0.01, 1), (-0.1, 0, 0.001, 0.1, 0.5, 0.9, 1, 1.1)):
        yield {"family": "partial", "D": diameter, "fraction": fraction}
    for diameter, length in product((0.01, 1), (0.1, 1, 10)):
        yield {"family": "shape", "D": diameter, "L": length}
    for route, angle, pitch in product(list(combinations(range(4), 2)), (30, 45, 60), (0.01, 0.1)):
        yield {"family": "pitch", "route": route, "angle": angle, "pitch": pitch}


def compare(quantity, native, expected):
    categorical = isinstance(expected, str)
    return {
        "quantity": quantity,
        "native": native,
        "reference": expected,
        "rtol": 0 if categorical else 3e-8,
        "atol": 0 if categorical else 1e-10,
        "passed": native == expected
        if categorical
        else isclose(native, expected, rel_tol=3e-8, abs_tol=1e-10),
    }


def compare_metrics(native, reference, names):
    return [compare(name, native.metrics[name], getattr(reference, name)) for name in names]


def constructor_slot(function, kwargs):
    try:
        function(**kwargs)
        return {
            "quantity": "source_constructor",
            "native": "accepted",
            "reference": "accepted",
            "passed": True,
        }
    except Exception as error:
        return {
            "quantity": "source_constructor",
            "native": "accepted",
            "reference_exception": f"{type(error).__name__}: {error}",
            "passed": False,
        }


def evaluate(c):
    family = c["family"]
    if family == "coil":
        dt, dc, p, n = c["Dt"], c["Dc"], c["pitch"], c["turns"]
        native = {"tube_diameter_m": dt, "inner_diameter_m": 0.8 * dt}
        source = {"Dt": dt, "Di": 0.8 * dt}
        if c["route"] == 0:
            native.update(coil_diameter_m=dc, pitch_m=p, turns=n)
            source.update(Do=dc, pitch=p, N=n)
        elif c["route"] == 1:
            native.update(total_diameter_m=dc + dt, total_height_m=n * p + dt, turns=n)
            source.update(Do_total=dc + dt, H_total=n * p + dt, N=n)
        else:
            native.update(coil_diameter_m=dc, height_m=n * p, pitch_m=p)
            source.update(Do=dc, H=n * p, pitch=p)
        result = helical_coil_geometry(**native)
        assert len(result.metrics) == 20
        try:
            reference = ref.HelicalCoil(**source)
        except Exception as error:
            return [
                {
                    "quantity": name,
                    "native": value,
                    "reference_exception": f"{type(error).__name__}: {error}",
                    "passed": False,
                }
                for name, value in result.metrics.items()
            ]
        return compare_metrics(result, reference, result.metrics)
    if family == "plate":
        native = {"amplitude_m": c["a"], "wavelength_m": c["w"], "chevron_angles": c["angles"]}
        source = {"amplitude": c["a"], "wavelength": c["w"], "chevron_angles": c["angles"]}
        names = [
            "b",
            "chevron_angle",
            "inclination_angle",
            "plate_corrugation_aspect_ratio",
            "plate_enlargement_factor",
            "D_eq",
            "D_hydraulic",
        ]
        if c["scope"]:
            native.update(width_m=0.3, length_m=0.5)
            source.update(width=0.3, length=0.5)
            names.extend(("A_channel_flow", "A_plate_surface"))
        if c["scope"] == 2:
            native.update(thickness_m=0.0005, port_diameter_m=0.04, plates=21)
            source.update(thickness=0.0005, d_port=0.04, plates=21)
            names.extend(("length_port", "A_heat_transfer", "channels", "channels_per_fluid"))
        return compare_metrics(
            plate_exchanger_geometry(**native), ref.PlateExchanger(**source), names
        )
    if family in {"fin", "fin_plate"}:
        native = {
            "fin_height_m": c["h"],
            "fin_thickness_m": c["t"],
            "fin_spacing_m": c["s"],
            "offset_length_m": c["offset"],
            "length_m": 0.4,
            "width_m": 0.3,
            "layers": 10,
        }
        source = {
            "fin_height": c["h"],
            "fin_thickness": c["t"],
            "fin_spacing": c["s"],
            "length": 0.4,
            "width": 0.3,
            "layers": 10,
        }
        cls = ref.RectangularFinExchanger
        names = [
            "A_channel",
            "P_channel",
            "Dh",
            "fin_count",
            "blockage_ratio",
            "layer_fin_count",
            "A_HX_layer",
            "A_HX",
        ]
        if c["offset"] is not None:
            cls = ref.RectangularOffsetStripFinExchanger
            source["fin_length"] = c["offset"]
            names.extend(
                (
                    "blockage_ratio_Kim",
                    "alpha",
                    "delta",
                    "gamma",
                    "A",
                    "Dh_Kays_London",
                    "Dh_Joshi_Webb",
                )
            )
        if family == "fin":
            return compare_metrics(fin_exchanger_geometry(**native), cls(**source), names)
        native["plate_thickness_m"] = 0.0005
        source["plate_thickness"] = 0.0005
        result = fin_exchanger_geometry(**native)
        height = (c["h"] + 0.0005) * 10 + 0.0005
        volume = 0.4 * 0.3 * height
        area = 2 * (c["s"] + c["h"] - 2 * c["t"]) * 0.4 * round(0.3 / c["s"], 0) * 10
        return [
            compare("height", result.metrics["height"], height),
            compare("volume", result.metrics["volume"], volume),
            compare("specific_area", result.metrics["A_specific_HX"], area / volume),
            constructor_slot(cls, source),
        ]
    if family == "air":
        native = {
            "tube_rows": c["rows"],
            "tube_passes": c["passes"],
            "tubes_per_row": 10,
            "tube_length_m": 2,
            "tube_diameter_m": 0.025,
            "fin_thickness_m": 0.0003,
            "angle_degrees": c["angle"],
            "parallel_bays": 2,
            "bundles_per_bay": 3,
            "fans_per_bay": 2,
            "tube_thickness_m": 0.002,
        }
        source = {
            "tube_rows": c["rows"],
            "tube_passes": c["passes"],
            "tubes_per_row": 10,
            "tube_length": 2,
            "tube_diameter": 0.025,
            "fin_thickness": 0.0003,
            "angle": c["angle"],
            "pitch": 0.08,
            "parallel_bays": 2,
            "bundles_per_bay": 3,
            "fans_per_bay": 2,
            "tube_thickness": 0.002,
        }
        if c["fin_route"] == 0:
            native.update(fin_diameter_m=0.05, fin_density_per_m=200)
            source.update(fin_diameter=0.05, fin_density=200)
        else:
            native.update(fin_height_m=0.0125, fin_interval_m=0.005)
            source.update(fin_height=0.0125, fin_interval=0.005)
        if c["pitch_route"]:
            native["pitch_ratio"] = 0.08 / 0.025
        else:
            native["pitch_m"] = 0.08
        result = air_cooled_geometry(**native)
        assert len(result.metrics) == 59, len(result.metrics)
        comparisons = compare_metrics(result, ref.AirCooledExchanger(**source), result.metrics)
        if c["pitch_route"]:
            source.pop("pitch")
            source["pitch_ratio"] = 0.08 / 0.025
            comparisons.append(constructor_slot(ref.AirCooledExchanger, source))
        return comparisons
    if family == "tower":
        ht = 2 * c["Ho"] / 3
        dt = (0.63 if c["throat"] is None else c["throat"]) * c["Di"]
        base = dt * sqrt(1 + (ht / (dt * (ht - c["Hi"]) / sqrt(c["Di"] ** 2 - dt**2))) ** 2)
        native = {
            "inlet_height_m": c["Hi"],
            "outlet_height_m": c["Ho"],
            "outlet_diameter_m": c["Do"],
            "throat_diameter_m": dt if c["throat"] is not None or c["route"] else None,
        }
        source = {
            "H_inlet": c["Hi"],
            "H_outlet": c["Ho"],
            "D_outlet": c["Do"],
            "D_throat": native["throat_diameter_m"],
        }
        if c["route"]:
            native["base_diameter_m"] = base
            source["D_base"] = base
        else:
            native["inlet_diameter_m"] = c["Di"]
            source["D_inlet"] = c["Di"]
        result, reference = CoolingTowerGeometry(**native), ref.HyperbolicCoolingTower(**source)
        pairs = (
            ("inlet_diameter_m", "D_inlet"),
            ("base_diameter_m", "D_base"),
            ("throat_diameter_m", "D_throat"),
            ("throat_height_m", "H_throat"),
            ("_lower", "b_lower"),
            ("_upper", "b_upper"),
        )
        comparisons = [compare(a, getattr(result, a), getattr(reference, b)) for a, b in pairs]
        return comparisons + [
            compare("diameter", result.diameter(h), reference.diameter(h))
            for h in (0, c["Hi"], ht, (ht + c["Ho"]) / 2, c["Ho"])
        ]
    if family == "cylinder":
        diameter, length = c["D"], c["L"]
        holes = (
            ()
            if not c["holes"]
            else ((0.4 * diameter, 1),)
            if c["holes"] == 1
            else ((0.1 * diameter, 3), (0.2 * diameter, 2))
        )
        volume, area = cylinder_geometry(diameter, length, holes=holes)
        if not holes:
            expected_v, expected_a = (
                ref.V_cylinder(diameter, length),
                ref.A_cylinder(diameter, length),
            )
        elif c["holes"] == 1:
            expected_v, expected_a = (
                ref.V_hollow_cylinder(0.4 * diameter, diameter, length),
                ref.A_hollow_cylinder(0.4 * diameter, diameter, length),
            )
        else:
            expected_v, expected_a = (
                ref.V_multiple_hole_cylinder(diameter, length, list(holes)),
                ref.A_multiple_hole_cylinder(diameter, length, list(holes)),
            )
        return [compare("volume", volume, expected_v), compare("area", area, expected_a)]
    if family == "partial":
        diameter, height = c["D"], c["D"] * c["fraction"]
        area = circular_segment_area(diameter, height)
        volume, surface = partial_sphere_geometry(diameter, height)
        return [
            compare("circle_area", area, ref.A_partial_circle(diameter, height)),
            compare("sphere_volume", volume, ref.V_partial_sphere(diameter, height)),
            compare("sphere_surface", surface, ref.SA_partial_sphere(diameter, height)),
            compare(
                "circle_inverse",
                circular_segment_height(diameter, area),
                ref.circle_segment_h_from_A(area, diameter),
            ),
        ]
    if family == "shape":
        volume, area = cylinder_geometry(c["D"], c["L"])
        return [
            compare(
                "sphericity",
                shape_factor("sphericity", area_m2=area, volume_m3=volume),
                ref.sphericity(area, volume),
            ),
            compare(
                "aspect",
                shape_factor(
                    "aspect_ratio", minimum_diameter_m=0.5 * c["D"], maximum_diameter_m=c["D"]
                ),
                ref.aspect_ratio(0.5 * c["D"], c["D"]),
            ),
            compare(
                "circularity",
                shape_factor("circularity", area_m2=pi * c["D"] ** 2 / 4, perimeter_m=pi * c["D"]),
                ref.circularity(pi * c["D"] ** 2 / 4, pi * c["D"]),
            ),
        ]
    values = (
        c["angle"],
        c["pitch"],
        c["pitch"] * cos(radians(c["angle"])),
        c["pitch"] * sin(radians(c["angle"])),
    )
    names = ("angle_degrees", "pitch_m", "parallel_m", "normal_m")
    source_names = ("angle", "pitch", "pitch_parallel", "pitch_normal")
    native = solve_tube_pitch(**{names[i]: values[i] for i in c["route"]})
    expected = ref.pitch_angle_solver(**{source_names[i]: values[i] for i in c["route"]})
    return [compare(name, n, e) for name, n, e in zip(names, native, expected, strict=True)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve attempts")
    cases = list(inputs())
    assert len(cases) == 308, len(cases)
    started = time.perf_counter()
    records = []
    for index, c in enumerate(cases, 1):
        try:
            records.append({"input": c, "comparisons": evaluate(c)})
        except Exception as error:
            records.append({"input": c, "exception": f"{type(error).__name__}: {error}"})
        if index % 50 == 0 or index == len(cases):
            elapsed = time.perf_counter() - started
            print(
                f"equipment geometry {index}/{len(cases)} {index / elapsed:.1f} inputs/s "
                f"ETA {(len(cases) - index) * elapsed / index:.1f}s",
                flush=True,
            )
    comparisons = [c for r in records for c in r.get("comparisons", [])]
    summary = {
        "reference_version": fluids.__version__,
        "inputs": len(cases),
        "expected_comparisons": 6214,
        "completed_comparisons": len(comparisons),
        "passed": sum(c["passed"] for c in comparisons),
        "failed": sum(not c["passed"] for c in comparisons),
        "source_exceptions": sum("reference_exception" in c for c in comparisons),
        "exceptions": sum("exception" in r for r in records),
        "failures": [
            r
            for r in records
            if "exception" in r or any(not c["passed"] for c in r.get("comparisons", []))
        ],
        "records": records,
        "elapsed_s": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "failures"}}))


if __name__ == "__main__":
    main()
