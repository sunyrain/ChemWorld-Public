"""Fixed head/tank geometry grid; retain every input and comparison."""

import argparse
import json
import time
from dataclasses import replace
from functools import lru_cache
from itertools import combinations, product
from math import isclose
from pathlib import Path

import fluids
from fluids import geometry as ref

from chemworld.physchem.tank_geometry import TankGeometry
from chemworld.physchem.tank_heads import TankHead

HEADS = [TankHead()]
for kind in ("conical", "ellipsoidal", "spherical"):
    HEADS.extend(
        TankHead(kind, ratio) for ratio in (0.1, 0.25, 0.5, 0.75 if kind == "spherical" else 1)
    )
HEADS.extend(
    TankHead("torispherical", crown_ratio=f, knuckle_ratio=k)
    for f, k in ((1, 0.06), (0.8, 0.1), (0.5, 0.1))
)
HEADS.extend(TankHead("guppy", ratio) for ratio in (0.1, 0.25, 0.5, 1))
PAIRS = ((0, 0), (2, 6), (10, 13), (13, 13), (8, 4), (17, 18), (12, 0))
UTILITIES = ((True, 0, 0), (True, 2, 6), (True, 13, 13), (False, 8, 4), (False, 10, 10))
HORIZONTAL_FRACTIONS = (-0.1, 0, 1e-5, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99999, 1, 1.1)
VERTICAL_FRACTIONS = (-0.1, 0, 1e-5, 0.1, 0.5, 0.9, 1, 1.1, 2)
LEVELS = (0, 0.1, 0.3, 0.5, 0.7, 0.9, 1)
SPECS = ("volume", "area", "partial_volume", "wetted_area", "cross_section")


def inputs():
    for i, d in product(range(20), (0.02, 1, 4)):
        yield {"family": "head_full", "head": i, "D": d, "slots": 3}
        for fraction in HORIZONTAL_FRACTIONS:
            yield {"family": "head_horizontal", "head": i, "D": d, "fraction": fraction, "slots": 2}
        if HEADS[i].kind in ("flat", "guppy"):
            continue
        for fraction in VERTICAL_FRACTIONS:
            yield {"family": "head_vertical", "head": i, "D": d, "fraction": fraction, "slots": 2}
            if i != 12:
                yield {
                    "family": "head_concave",
                    "head": i,
                    "D": d,
                    "fraction": fraction,
                    "slots": 1,
                }
    for horizontal, pair, d, ratio in product((True, False), range(7), (0.1, 2), (0, 1, 5)):
        if (not horizontal and pair == 5) or (pair == 0 and ratio == 0):
            continue
        base = {"horizontal": horizontal, "pair": pair, "D": d, "ratio": ratio}
        yield {**base, "family": "tank_full", "slots": 9}
        for fraction in LEVELS:
            yield {**base, "family": "tank_level", "fraction": fraction, "slots": 3}
    for case in range(5):
        for route in range(6):
            yield {"family": "sizing", "case": case, "route": route, "slots": 2}
        for custom, thickness in product((False, True), (-0.01, 0.01)):
            yield {
                "family": "thickness",
                "case": case,
                "custom": custom,
                "thickness": thickness,
                "slots": 4,
            }
        for fraction in LEVELS:
            yield {"family": "inverse", "case": case, "fraction": fraction, "slots": 1}
        for fraction in LEVELS[1:-1]:
            for family in ("table", "chebyshev"):
                yield {"family": family, "case": case, "fraction": fraction, "slots": 2}
    for horizontal in (True, False):
        for first, second in combinations(SPECS, 2):
            if (
                not horizontal
                and first != "volume"
                and (first, second) != ("area", "cross_section")
            ):
                continue
            yield {
                "family": "two_specs",
                "horizontal": horizontal,
                "names": (first, second),
                "slots": 2,
            }


def source_kwargs(head, d):
    if head.kind == "torispherical":
        return {"D": d, "f": head.crown_ratio, "k": head.knuckle_ratio}
    return {"D": d, "a": head.depth * d}


def tank_kwargs(tank):
    kwargs = {"D": tank.diameter_m, "L": tank.length_m, "horizontal": tank.horizontal}
    for label, head in (("sideA", tank.head_a), ("sideB", tank.head_b)):
        kwargs[label] = None if head.kind == "flat" else head.kind
        kwargs[label + "_a"] = head.depth * tank.diameter_m
        if head.kind == "torispherical":
            kwargs[label + "_f"] = head.crown_ratio
            kwargs[label + "_k"] = head.knuckle_ratio
    return kwargs


@lru_cache(maxsize=8)
def utility(case):
    horizontal, a, b = UTILITIES[case]
    native = TankGeometry(1.2, 3.6, horizontal, HEADS[a], HEADS[b])
    return native, ref.TANK(**tank_kwargs(native))


@lru_cache(maxsize=8)
def tables(case):
    native, source = utility(case)
    source.set_table(n=41)
    return native.table(41), source


@lru_cache(maxsize=8)
def polynomials(case):
    return utility(case)[0].chebyshev(40, 100)


def compare(name, native, source, *, scale=1.0, rtol=2e-7, atol=2e-10):
    value = {"quantity": name, "native": float(native), "rtol": rtol, "atol": atol * scale}
    try:
        expected = float(source() if callable(source) else source)
        value.update(
            reference=expected, passed=isclose(native, expected, rel_tol=rtol, abs_tol=atol * scale)
        )
    except Exception as error:
        value.update(reference_exception=f"{type(error).__name__}: {error}", passed=False)
    return value


def evaluate(c):
    family = c["family"]
    if family.startswith("head"):
        d = c["D"]
        head = HEADS[c["head"]]
        kw = source_kwargs(head, d)
        if family == "head_full":
            tank = TankGeometry(d, d, True, head)
            source = tank_kwargs(tank)
            area_source = {k: v for k, v in source.items() if k != "horizontal"}
            return [
                compare(
                    "volume", head.total_volume(d), lambda: ref.V_tank(**source)[1], scale=d**3
                ),
                compare(
                    "area", head.total_area(d), lambda: ref.SA_tank(**area_source)[1], scale=d**2
                ),
                compare(
                    "depth",
                    head.depth * d,
                    lambda: (
                        ref.a_torispherical(d, head.crown_ratio, head.knuckle_ratio)
                        if head.kind == "torispherical"
                        else head.depth * d
                    ),
                    scale=d,
                ),
            ]
        if family == "head_horizontal":
            h = c["fraction"] * d
            if head.kind == "flat":

                def volume():
                    return 0.0

                def area():
                    return ref.A_partial_circle(d, h)
            else:

                def volume():
                    return getattr(ref, "V_horiz_" + head.kind)(**kw, L=0, h=h, headonly=True)

                def area():
                    return getattr(ref, "SA_partial_horiz_" + head.kind + "_head")(**kw, h=h)

            return [
                compare("volume", head.horizontal_measure(d, h), volume, scale=d**3),
                compare("area", head.horizontal_measure(d, h, "area"), area, scale=d**2),
            ]
        h = c["fraction"] * head.depth * d
        if family == "head_concave":
            head = replace(head, concave=True)
            if "a" in kw:
                kw["a"] *= -1
            return [
                compare(
                    "volume",
                    head.vertical_capacity(d, h),
                    lambda: getattr(ref, "V_vertical_" + head.kind + "_concave")(**kw, h=h),
                    scale=d**3,
                )
            ]
        return [
            compare(
                "volume",
                head.vertical_capacity(d, h),
                lambda: getattr(ref, "V_vertical_" + head.kind)(**kw, h=h),
                scale=d**3,
            ),
            compare(
                "area",
                head.vertical_measure(d, h, "area"),
                lambda: getattr(ref, "SA_partial_vertical_" + head.kind + "_head")(**kw, h=h),
                scale=d**2,
            ),
        ]
    if family.startswith("tank"):
        a, b = PAIRS[c["pair"]]
        d = c["D"]
        tank = TankGeometry(d, d * c["ratio"], c["horizontal"], HEADS[a], HEADS[b])
        source = ref.TANK(**tank_kwargs(tank))
        if family == "tank_full":
            return [
                *(
                    compare("V_" + name, value, expected, scale=d**3)
                    for name, value, expected in zip(
                        ("total", "a", "b", "body"),
                        tank.volume_parts_m3,
                        (source.V_total, source.V_sideA, source.V_sideB, source.V_lateral),
                        strict=True,
                    )
                ),
                *(
                    compare("A_" + name, value, expected, scale=d**2)
                    for name, value, expected in zip(
                        ("total", "a", "b", "body"),
                        tank.area_parts_m2,
                        (source.A, source.A_sideA, source.A_sideB, source.A_lateral),
                        strict=True,
                    )
                ),
                compare("height", tank.maximum_height_m, source.h_max, scale=d),
            ]
        # Compare at the same physical datum. Deep spherical envelopes deliberately
        # expose the source's cylinder-height clipping, rather than aligning it away.
        h = c["fraction"] * tank.maximum_height_m
        return [
            compare("volume", tank.volume_at_height(h), lambda: source.V_from_h(h), scale=d**3),
            compare("area", tank.wetted_area_at_height(h), lambda: source.SA_from_h(h), scale=d**2),
            compare(
                "cross",
                tank.cross_section_at_height(h),
                lambda: source.A_cross_sectional(h),
                scale=d**2,
                rtol=5e-6,
                atol=2e-8,
            ),
        ]
    if family == "two_specs":
        tank = TankGeometry(1.2, 3.6, c["horizontal"], HEADS[2], HEADS[6])
        h = 0.48 if tank.horizontal else 1.5
        goals = {name: tank.specification(name, h) for name in c["names"]}
        solved = TankGeometry.from_two_specs(
            goals,
            height_m=h,
            horizontal=tank.horizontal,
            head_a=tank.head_a,
            head_b=tank.head_b,
            initial_diameter_m=1.1,
            initial_length_m=3.3,
        )
        return [
            compare(name, solved.specification(name, h), target) for name, target in goals.items()
        ]
    tank, source = utility(c["case"])
    if family == "sizing":
        route = c["route"]
        dimension_options = (
            {"diameter_m": tank.diameter_m, "length_m": tank.length_m},
            {"diameter_m": tank.diameter_m, "length_over_diameter": 3},
            {"length_m": tank.length_m, "length_over_diameter": 3},
            {"volume_m3": source.V_total, "diameter_m": tank.diameter_m},
            {"volume_m3": source.V_total, "length_m": tank.length_m},
            {"volume_m3": source.V_total, "length_over_diameter": 3},
        )
        solved = TankGeometry.from_dimensions(
            **dimension_options[route],
            horizontal=tank.horizontal,
            head_a=tank.head_a,
            head_b=tank.head_b,
        )
        return [compare("D", solved.diameter_m, source.D), compare("L", solved.length_m, source.L)]
    if family == "thickness":
        t = c["thickness"]
        native_kw = (
            {"head_a_thickness_m": t / 2, "head_b_thickness_m": 1.5 * t} if c["custom"] else {}
        )
        source_kw = {"sideA_thickness": t / 2, "sideB_thickness": 1.5 * t} if c["custom"] else {}
        expanded = tank.with_wall_thickness(t, **native_kw)
        reference = source.add_thickness(t, **source_kw)
        return [
            compare("D", expanded.diameter_m, reference.D),
            compare("L", expanded.length_m, reference.L),
            compare("volume", expanded.total_volume_m3, reference.V_total),
            compare("area", expanded.total_area_m2, reference.A),
        ]
    fraction = c["fraction"]
    h = fraction * tank.maximum_height_m
    volume = tank.volume_at_height(h)
    if family == "inverse":
        return [compare("height", tank.height_for_volume(volume), h)]
    if family == "table":
        table, source = tables(c["case"])
        return [
            compare("height", table.height_for_volume(volume), lambda: source.h_from_V(volume)),
            compare("volume", table.volume_at_height(h), lambda: source.V_from_h(h)),
        ]
    polynomial = polynomials(c["case"])
    return [
        compare(
            "height",
            polynomial.height_for_volume(volume),
            h,
            rtol=0,
            atol=5e-4,
            scale=tank.maximum_height_m,
        ),
        compare(
            "volume",
            polynomial.volume_at_height(h),
            volume,
            rtol=0,
            atol=5e-4,
            scale=tank.total_volume_m3,
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count-only", action="store_true")
    args = parser.parse_args()
    cases = list(inputs())
    expected = sum(c["slots"] for c in cases)
    print(json.dumps({"inputs": len(cases), "comparisons": expected}), flush=True)
    assert (len(cases), expected) == (2245, 5213)
    if args.count_only:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    failures = []
    passed = completed = exceptions = source_exceptions = 0
    started = time.monotonic()
    with (args.output / "cases.jsonl").open("w", encoding="utf-8") as stream:
        for i, c in enumerate(cases, 1):
            try:
                comparisons = evaluate(c)
                assert len(comparisons) == c["slots"]
                completed += len(comparisons)
                passed += sum(item["passed"] for item in comparisons)
                source_exceptions += sum("reference_exception" in item for item in comparisons)
                row = {"input": c, "comparisons": comparisons}
                if not all(item["passed"] for item in comparisons):
                    failures.append(row)
            except Exception as error:
                exceptions += 1
                row = {"input": c, "exception": f"{type(error).__name__}: {error}"}
                failures.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if i % 25 == 0 or i == len(cases):
                elapsed = time.monotonic() - started
                print(
                    f"tank {i}/{len(cases)} {i / max(elapsed, 0.001):.1f} inputs/s "
                    f"ETA={(len(cases) - i) * elapsed / i:.1f}s pass={passed} "
                    f"completed={completed} exceptions={exceptions}",
                    flush=True,
                )
    result = {
        "reference_version": fluids.__version__,
        "inputs": len(cases),
        "expected_comparisons": expected,
        "completed_comparisons": completed,
        "passed": passed,
        "failed": completed - passed,
        "source_exceptions": source_exceptions,
        "exceptions": exceptions,
        "elapsed_seconds": time.monotonic() - started,
        "failures": failures,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    main()
