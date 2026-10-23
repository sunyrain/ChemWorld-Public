"""Fixed input block for remaining fitting families and caller curve leaves."""

import argparse
import json
import time
from itertools import product
from math import isclose
from pathlib import Path

import fluids
from fitting_source_curves import source_curves, specifications
from fluids import fittings as ref

from chemworld.physchem.bend_fittings import bend_loss_coefficient, coil_loss_coefficient
from chemworld.physchem.fitting_curves import FittingCurves
from chemworld.physchem.local_fittings import (
    entrance_loss_coefficient,
    staged_diffuser_loss,
    transition_loss_coefficient,
)


def inputs():
    for method in ("Rennels", "Miller", "Swamee", "Blevins", "Crane", "Idelchik"):
        yield {"family": "entrance", "shape": "sharp", "method": method}
    for method, t, length in product(
        ("Rennels", "Miller", "Idelchik", "Harris", "Crane"), (0, 0.02, 0.1, 0.4), (0.2, 2, 12)
    ):
        yield {
            "family": "entrance",
            "shape": "distance",
            "method": method,
            "thickness": t,
            "projection": length,
        }
    for t in (0, 0.02, 0.1, 0.4):
        yield {"family": "entrance", "shape": "distance45", "method": "Miller", "thickness": t}
    for angle in (0, 30, 60, 90):
        yield {"family": "entrance", "shape": "angled", "method": "Idelchik", "angle": angle}
    for method, length, angle in product(("Rennels", "Idelchik"), (0, 0.05, 0.3), (15, 45, 75)):
        yield {
            "family": "entrance",
            "shape": "beveled",
            "method": method,
            "length": length,
            "angle": angle,
        }
    for beta, length, angle in product((0.4, 0.7, 1), (0, 0.05, 0.3), (15, 45, 75)):
        yield {
            "family": "entrance",
            "shape": "beveled_orifice",
            "method": "Rennels",
            "beta": beta,
            "length": length,
            "angle": angle,
        }
    for method, radius in product(
        ("Rennels", "Swamee", "Crane", "Miller", "Harris", "Idelchik"),
        (0, 0.02, 0.1, 0.16, 0.3, 1.1),
    ):
        yield {"family": "entrance", "shape": "rounded", "method": method, "radius": radius}
    yield {"family": "entrance", "shape": "exit", "method": "Rennels"}
    designs = {
        "contraction sharp": (
            ("Rennels", "Hooper", "Crane"),
            (180,),
            (1000, 2500, 1e5),
            (0,),
            (0,),
            (None,),
        ),
        "contraction rounded": (
            ("Rennels", "Miller", "Idelchik"),
            (180,),
            (1e5,),
            (0, 0.02, 0.1, 0.3),
            (0,),
            (None,),
        ),
        "contraction conical": (
            ("Rennels", "Crane", "Swamee", "Idelchik", "Blevins", "Miller", "Hooper"),
            (5, 20, 45, 60, 120, 180),
            (1000, 1e5),
            (0,),
            (0,),
            (None,),
        ),
        "contraction beveled": (("Rennels",), (20, 60, 120), (1e5,), (0,), (0, 0.05, 0.2), (None,)),
        "diffuser sharp": (
            ("Rennels", "Hooper"),
            (180,),
            (1000, 3999, 4000, 1e5),
            (0,),
            (0,),
            (None,),
        ),
        "diffuser conical": (
            ("Rennels", "Crane", "Miller", "Idelchik", "Swamee", "Hooper"),
            (5, 20, 40, 60, 90, 180),
            (1000, 1e5),
            (0,),
            (0,),
            (None,),
        ),
        "diffuser curved": (("Rennels",), (45,), (1e5,), (0,), (0.1, 1, 3), (None,)),
        "diffuser pipe reducer": (("Rennels",), (45,), (1e5,), (0,), (0.1, 1, 3), (None, 0.03)),
    }
    for shape, design in designs.items():
        for beta, method, angle, re, radius, length, fd2 in product((0.3, 0.6, 0.9), *design):
            yield {
                "family": "transition",
                "shape": shape,
                "beta": beta,
                "method": method,
                "angle": angle,
                "Re": re,
                "radius": radius,
                "length": length,
                "fd2": fd2,
            }
    for scale in (1, 2, 4):
        yield {"family": "staged", "scale": scale}
    for method, radius, angle, re in product(
        ("Rennels", "Miller", "Crane", "Crane standard", "Ito", "Swamee"),
        (0.6, 1, 1.5, 2, 5, 10, 60),
        (10, 45, 90, 180),
        (100, 2e4, 1e6, 1e8),
    ):
        yield {
            "family": "bend",
            "shape": "rounded",
            "method": method,
            "radius": radius,
            "angle": angle,
            "Re": re,
            "length": 20,
        }
    for method, angle, re, length in product(
        ("Rennels", "Crane", "Miller", "Blevins"),
        (5, 45, 90, 120),
        (100, 2e4, 1e6, 1e8),
        (0.005, 5, 40),
    ):
        yield {
            "family": "bend",
            "shape": "miter",
            "method": method,
            "radius": 5,
            "angle": angle,
            "Re": re,
            "length": length,
        }
    for radius, re, angle, length in product(
        (0.6, 0.8, 1, 1.5, 2, 5), (1e4, 2e5, 1e6, 1e8), (45, 90), (0, 0.01, 5, 40)
    ):
        yield {
            "family": "bend",
            "shape": "rounded",
            "method": "Miller",
            "radius": radius,
            "angle": angle,
            "Re": re,
            "length": length,
        }
    for radius, pitch, turns in product((2, 10), (0.5, 2), (1, 5)):
        yield {"family": "coil", "shape": "helix", "radius": radius, "pitch": pitch, "turns": turns}
    for radius, pitch, outer in product((2, 10), (0.5, 2), (2, 5)):
        yield {
            "family": "coil",
            "shape": "spiral",
            "radius": radius,
            "pitch": pitch,
            "outer": outer,
        }
    for name, kind, arrays, _, _ in specifications():
        for group, values in arrays.items():
            for i, value in enumerate(values):
                yield {
                    "family": "cell",
                    "name": name,
                    "kind": kind,
                    "group": group,
                    "index": i,
                    "value": float(value),
                }


def compare(name, native, source, exact=False):
    rtol, atol = (0, 0) if exact else (3e-8, 1e-10)
    return {
        "quantity": name,
        "native": native,
        "reference": source,
        "rtol": rtol,
        "atol": atol,
        "passed": isclose(native, source, rel_tol=rtol, abs_tol=atol),
    }


def evaluate(c, curves):
    family = c["family"]
    if family == "cell":
        curve = curves.curves[c["name"]]
        values = (
            (curve.x if c["group"] == "x" else curve.y)
            if c["kind"] == "linear"
            else curve._arrays[c["group"]]
        )
        return [compare("source_cell", float(values[c["index"]]), c["value"], True)]
    if family == "entrance":
        shape, method = c["shape"], c["method"]
        native = entrance_loss_coefficient(
            shape,
            diameter_m=0.1,
            method=method,
            thickness_m=0.1 * c.get("thickness", 0),
            projection_m=0.1 * c.get("projection", 1),
            rounding_m=0.1 * c.get("radius", 0),
            bevel_length_m=0.1 * c.get("length", 0),
            angle_degrees=c.get("angle", 45),
            orifice_diameter_m=0.1 * c.get("beta", 1),
            curves=curves,
        )
        if shape == "exit":
            source = ref.exit_normal()
        elif shape == "sharp":
            source = ref.entrance_sharp(method)
        elif shape == "distance":
            source = ref.entrance_distance(
                0.1, t=0.1 * c["thickness"], l=0.1 * c["projection"], method=method
            )
        elif shape == "distance45":
            source = ref.entrance_distance_45_Miller(0.1, 0.1 + 2 * 0.1 * c["thickness"])
        elif shape == "angled":
            source = ref.entrance_angled(c["angle"], method)
        elif shape == "beveled":
            source = ref.entrance_beveled(0.1, 0.1 * c["length"], c["angle"], method)
        elif shape == "beveled_orifice":
            source = ref.entrance_beveled_orifice(
                0.1, 0.1 * c["beta"], 0.1 * c["length"], c["angle"]
            )
        else:
            source = ref.entrance_rounded(0.1, 0.1 * c["radius"], method)
    elif family == "transition":
        shape = c["shape"]
        d1, d2 = (
            (0.1, 0.1 * c["beta"]) if shape.startswith("contraction") else (0.1, 0.1 / c["beta"])
        )
        small = min(d1, d2)
        lengths = {"contraction beveled", "diffuser curved", "diffuser pipe reducer"}
        native = transition_loss_coefficient(
            shape,
            inlet_diameter_m=d1,
            outlet_diameter_m=d2,
            method=c["method"],
            length_m=small * c["length"] if shape in lengths else None,
            angle_degrees=c["angle"]
            if shape not in {"diffuser curved", "diffuser pipe reducer"}
            else None,
            rounding_m=small * c["radius"],
            reynolds=c["Re"],
            friction_factor=0.02,
            outlet_friction_factor=c["fd2"],
            curves=curves,
        )
        common = {"Di1": d1, "Di2": d2}
        if shape in {"contraction sharp", "diffuser sharp"}:
            common.update(fd=0.02, Re=c["Re"], method=c["method"])
        elif shape == "contraction rounded":
            common.update(rc=small * c["radius"], method=c["method"])
        elif shape in {"contraction conical", "diffuser conical"}:
            common.update(angle=c["angle"], fd=0.02, Re=c["Re"], method=c["method"])
        elif shape == "contraction beveled":
            common.update(l=small * c["length"], angle=c["angle"])
        elif shape == "diffuser curved":
            common.update(l=small * c["length"])
        else:
            common.update(l=small * c["length"], fd1=0.02, fd2=c["fd2"])
        name = shape.replace("rounded", "round").replace(" ", "_")
        source = getattr(ref, name)(**common)
    elif family == "staged":
        d = tuple(v * c["scale"] for v in (0.05, 0.07, 0.1))
        ls = tuple(v * c["scale"] for v in (0.1, 0.2))
        raw = staged_diffuser_loss(d, ls, friction_factor=0.02, common_velocity_basis=False)
        source = ref.diffuser_conical_staged(d[0], d[-1], [d[1]], list(ls), fd=0.02)
        physical = staged_diffuser_loss(d, ls, friction_factor=0.02)
        expected = sum(
            ref.diffuser_conical(d[i], d[i + 1], l=ls[i], fd=0.02) * (d[0] / d[i]) ** 4
            for i in (0, 1)
        )
        return [
            compare("raw_K_convention", raw, source),
            compare("pressure_weighted", physical, expected),
        ]
    elif family == "bend":
        native = bend_loss_coefficient(
            c["shape"],
            diameter_m=0.1,
            angle_degrees=c["angle"],
            method=c["method"],
            radius_m=0.1 * c["radius"],
            reynolds=c["Re"],
            roughness_m=1e-5,
            friction_factor=0.02,
            unimpeded_length_m=0.1 * c["length"],
            curves=curves,
        )
        common = {
            "Di": 0.1,
            "angle": c["angle"],
            "Re": c["Re"],
            "roughness": 1e-5,
            "L_unimpeded": 0.1 * c["length"],
            "method": c["method"],
        }
        if c["shape"] == "rounded":
            common.update(rc=0.1 * c["radius"], bend_diameters=c["radius"], fd=0.02)
        source = getattr(ref, "bend_" + c["shape"])(**common)
    else:
        native = coil_loss_coefficient(
            c["shape"],
            diameter_m=0.1,
            friction_factor=0.02,
            pitch_m=0.1 * c["pitch"],
            radius_m=0.1 * c["radius"],
            turns=c.get("turns", 1),
            outer_radius_m=0.1 * c["radius"] * c.get("outer", 2),
        )
        source = (
            ref.helix(0.1, 0.1 * c["radius"], 0.1 * c["pitch"], c["turns"], 0.02)
            if c["shape"] == "helix"
            else ref.spiral(
                0.1, 0.1 * c["radius"] * c["outer"], 0.1 * c["radius"], 0.1 * c["pitch"], 0.02
            )
        )
    return [compare("loss_coefficient", native, source)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve all attempts")
    cases = list(inputs())
    assert len(cases) == 3513, len(cases)
    curves = FittingCurves.from_dict(json.loads(json.dumps(source_curves().to_dict())))
    records = []
    started = time.perf_counter()
    for index, c in enumerate(cases, 1):
        try:
            records.append({"input": c, "comparisons": evaluate(c, curves)})
        except Exception as error:
            records.append({"input": c, "exception": f"{type(error).__name__}: {error}"})
        if index % 500 == 0 or index == len(cases):
            elapsed = time.perf_counter() - started
            print(
                f"local fittings {index}/{len(cases)} {index / elapsed:.1f} inputs/s "
                f"ETA {(len(cases) - index) * elapsed / index:.1f}s",
                flush=True,
            )
    comparisons = [c for r in records for c in r.get("comparisons", [])]
    summary = {
        "reference_version": fluids.__version__,
        "inputs": len(cases),
        "expected_comparisons": 3516,
        "completed_comparisons": len(comparisons),
        "passed": sum(c["passed"] for c in comparisons),
        "failed": sum(not c["passed"] for c in comparisons),
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
