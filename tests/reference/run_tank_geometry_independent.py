"""Independent Cartesian checks of all fixed-block source disagreements."""

import argparse
import json
import time
from functools import lru_cache
from itertools import pairwise
from math import acos, asin, cos, hypot, isclose, pi, sin, sqrt
from pathlib import Path

from run_tank_geometry_block import HEADS, PAIRS
from scipy.integrate import quad


def integral(function, low, high):
    return (
        0.0 if high <= low else quad(function, low, high, epsabs=2e-12, epsrel=2e-11, limit=200)[0]
    )


def clipped(x, low, high):
    return min(high, max(low, x))


def disk(radius, level):
    if level <= -radius or radius == 0:
        return 0.0
    if level >= radius:
        return pi * radius**2
    return radius**2 * acos(-level / radius) + level * sqrt(max(0, radius**2 - level**2))


def depth(index):
    head = HEADS[index]
    if head.kind == "flat":
        return 0.0
    if head.kind == "torispherical":
        f, k = head.crown_ratio, head.knuckle_ratio
        return f - sqrt(f * f - 2 * f * k + k - 0.25)
    return head.depth_ratio


def extent(index):
    a = depth(index)
    return (a * a + 0.25) / (2 * a) if HEADS[index].kind == "spherical" and a > 0.5 else 0.5


@lru_cache(maxsize=512)
def guppy_area(a, h):
    """Graph x=a/(2R)*(R+y-z^2/(R-y)); integrate sqrt(1+x_y²+x_z²).

    y=R*cos(theta), z=R*sin(theta)*u regularizes the circular projection.
    This is independent of production's oblique-cone generatrix integration.
    """
    if h <= 0:
        return 0.0
    lower = acos(2 * min(1, h) - 1)

    def outer(theta):
        c, s = cos(theta), sin(theta)
        ratio = s / (1 - c)

        def inner(u):
            dy = a * (1 - u * u * ratio * ratio)
            dz = 2 * a * u * ratio
            return sqrt(1 + dy * dy + dz * dz)

        return 0.5 * s * s * integral(inner, 0, 1)

    return integral(outer, lower, pi)


@lru_cache(maxsize=2048)
def head_value(index, h, measure):
    """Dimensionless Cartesian axial x integration, h from cylinder bottom."""
    kind = HEADS[index].kind
    a = depth(index)
    y = h - 0.5
    if kind == "flat":
        return disk(0.5, y) if measure == "area" else 0.0
    if kind == "guppy":
        if measure == "area":
            return guppy_area(a, h)
        if measure == "volume":
            return integral(
                lambda z: 4 * a / 3 * z * sqrt(max(0, z * (1 - z))), 0, clipped(h, 0, 1)
            )
        return 4 * a / 3 * h * sqrt(max(0, h * (1 - h))) if 0 < h < 1 else 0.0
    cuts = [0.0, a]
    r_target = abs(y)
    if kind == "conical":
        cuts.append(2 * a * r_target)

        def profile(x):
            r = x / (2 * a)
            return r, 2 * pi * r * hypot(1, 0.5 / a)
    elif kind == "ellipsoidal":
        if r_target < 0.5:
            cuts.append(a * (1 - sqrt(1 - 4 * r_target * r_target)))

        def profile(x):
            r = 0.5 * sqrt(max(0, 2 * x / a - (x / a) ** 2))
            return r, 2 * pi * sqrt(r * r + (0.25 / a * (1 - x / a)) ** 2)
    elif kind == "spherical":
        radius = (a * a + 0.25) / (2 * a)
        if r_target < radius:
            offset = sqrt(radius * radius - r_target * r_target)
            cuts.extend((radius - offset, radius + offset))

        def profile(x):
            return sqrt(max(0, 2 * radius * x - x * x)), 2 * pi * radius
    else:
        f, k = HEADS[index].crown_ratio, HEADS[index].knuckle_ratio
        angle = asin((0.5 - k) / (f - k))
        join = f * (1 - cos(angle))
        cuts.append(join)
        if r_target < f:
            cuts.append(f - sqrt(f * f - r_target * r_target))
        if 0 <= r_target - 0.5 + k <= k:
            cuts.append(a - sqrt(max(0, k * k - (r_target - 0.5 + k) ** 2)))

        def profile(x):
            if x <= join:
                return sqrt(max(0, 2 * f * x - x * x)), 2 * pi * f
            inner = sqrt(max(0, k * k - (a - x) ** 2))
            r = 0.5 - k + inner
            return r, 2 * pi * r * k / inner

    cuts = sorted({0.0, a, *(x for x in cuts if 0 < x < a)})

    def value(x):
        radius, surface = profile(x)
        if measure == "volume":
            return disk(radius, y)
        if measure == "cross":
            return 2 * sqrt(max(0, radius * radius - y * y))
        fraction = acos(clipped(-y / radius, -1, 1)) / pi if radius else 0
        return surface * fraction

    return sum(integral(value, low, high) for low, high in pairwise(cuts))


def full(index, measure):
    return head_value(index, 0.5 + extent(index), measure)


def expected(case, quantity):
    family = case["family"]
    if family == "head_full":
        return full(case["head"], quantity) * case["D"] ** (3 if quantity == "volume" else 2)
    if family == "head_horizontal":
        return head_value(case["head"], case["fraction"], quantity) * case["D"] ** (
            3 if quantity == "volume" else 2
        )
    d = case["D"]
    a, b = PAIRS[case["pair"]]
    length = case["ratio"]
    radius = max(extent(a), extent(b))
    if family == "tank_full":
        if quantity == "height":
            return 2 * radius * d
        measure = "volume" if quantity.startswith("V_") else "area"
        va, vb = full(a, measure), full(b, measure)
        body = pi * length / (4 if measure == "volume" else 1)
        unit = {"total": va + vb + body, "a": va, "b": vb, "body": body}[quantity.split("_")[1]]
        return unit * d ** (3 if measure == "volume" else 2)
    if not case["horizontal"]:
        assert quantity == "cross" and case["fraction"] == 1
        return 0.0
    h = 2 * radius * case["fraction"] - (radius - 0.5)
    if quantity == "cross" and case["fraction"] in (0, 1):
        return 0.0
    if quantity == "volume":
        body = disk(0.5, h - 0.5) * length
    elif quantity == "area":
        body = length * acos(clipped(1 - 2 * h, -1, 1))
    else:
        body = 2 * length * sqrt(max(0, h * (1 - h)))
    return (body + head_value(a, h, quantity) + head_value(b, h, quantity)) * d ** (
        3 if quantity == "volume" else 2
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.source.read_text())
    units = [
        (row["input"], v)
        for row in summary["failures"]
        for v in row["comparisons"]
        if not v["passed"]
    ]
    assert len(units) == 333
    args.output.mkdir(exist_ok=False, parents=True)
    results = []
    start = time.monotonic()
    for i, (case, value) in enumerate(units, 1):
        quantity = value["quantity"]
        power = (
            3
            if quantity.startswith("V_") or quantity == "volume"
            else 1
            if quantity == "height"
            else 2
        )
        tolerance = 2e-10 * case["D"] ** power
        try:
            oracle = expected(case, quantity)
            row = {
                "input": case,
                "quantity": quantity,
                "native": value["native"],
                "independent": oracle,
                "passed": isclose(value["native"], oracle, rel_tol=3e-8, abs_tol=tolerance),
            }
        except Exception as error:
            row = {
                "input": case,
                "quantity": quantity,
                "exception": f"{type(error).__name__}: {error}",
                "passed": False,
            }
        results.append(row)
        if i % 25 == 0 or i == len(units):
            elapsed = time.monotonic() - start
            print(
                f"independent tank {i}/{len(units)} rate={i / max(elapsed, 0.001):.1f}/s "
                f"ETA={(len(units) - i) * elapsed / i:.1f}s",
                flush=True,
            )
    output = {
        "inputs": len(units),
        "comparisons": len(units),
        "passed": sum(r["passed"] for r in results),
        "failed": sum(not r["passed"] for r in results),
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "results"}))


if __name__ == "__main__":
    main()
