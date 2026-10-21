"""Independent series-primitive diagnosis for the fixed Zabransky entropy cases."""

import argparse
import json
import math
import time
import traceback
from itertools import pairwise
from pathlib import Path

import chemicals
from quasi_entropy_reference import quasi_entropy_primitive
from run_heat_capacity_foundation_block import (
    GAUSS,
    make_card,
    near,
    parameters,
    ref_value,
    reference_integral,
)

from chemworld.physchem.property_equations import integrate_correlation


def check(c, path):
    curve = make_card("zabransky_quasi_cp", c)
    actuals, quadratures, series, source, tails, convergence = [], [], [], [], [], []
    for low, high in pairwise(path):
        native = integrate_correlation(curve, initial_temperature_K=low, final_temperature_K=high,
            over_temperature=True, epsabs=1e-9, epsrel=1e-11).value
        answers = []
        for order in (64, 128):
            nodes, weights = GAUSS[order]
            midpoint, half = (low+high)/2, (high-low)/2
            answer = half * math.fsum(float(w) * ref_value("zabransky_quasi_cp", c, T) / T
                for x, w in zip(nodes, weights, strict=True) for T in (midpoint+half*float(x),))
            answers.append(answer)
        assert abs(answers[0]-answers[1]) <= max(2e-8, 2e-11*abs(answers[1]))
        high_series, high_tail = quasi_entropy_primitive(high, c)
        low_series, low_tail = quasi_entropy_primitive(low, c)
        actuals.append(native)
        quadratures.append(answers[1])
        series.append(high_series-low_series)
        tails.append(low_tail+high_tail)
        source.append(reference_integral("zabransky_quasi_cp", c, low, high, True)[0])
        convergence.append(abs(answers[0]-answers[1]))
    actual, quad, analytic, original = map(math.fsum, (actuals, quadratures, series, source))
    return {"path_K": path, "native": actual, "gauss_128": quad, "series_primitive": analytic,
        "source_primitive": original, "native_series_error": near(actual, analytic, 2e-10, 2e-8),
        "gauss_series_error": near(quad, analytic, 2e-10, 2e-8),
        "source_series_difference": original-analytic, "series_tail_bound": math.fsum(tails),
        "max_gauss_order_difference": max(convergence)}


def run(output):
    if output.exists():
        raise FileExistsError("preserve previous diagnostic blocks")
    assert chemicals.__version__ == "1.5.2"
    started = time.perf_counter()
    results = []
    for case in (0, 1, "a1-zero"):
        c = parameters("zabransky_quasi_cp", case if isinstance(case, int) else 0)
        if case == "a1-zero":
            c["a1"] = 0
        for path in ((280.5, 379.5), (379.5, 280.5), (280.5, 330, 379.5)):
            result = {"case": case, "path": path}
            try:
                result.update(status="passed", metrics=check(c, path))
            except Exception:
                result.update(status="failed", failure=traceback.format_exc())
            results.append(result)
            print(f"quasi entropy reference {len(results)}/9", flush=True)
    failures = [r for r in results if r["status"] == "failed"]
    summary = {"block": "22C1-quasi-entropy-reference", "mode": "development", "total": 9,
        "passed": 9-len(failures), "failed": len(failures),
        "results": results, "failures": failures,
        "elapsed_seconds": time.perf_counter()-started}
    output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    fields = ("total", "passed", "failed", "elapsed_seconds")
    print(json.dumps({k: summary[k] for k in fields}))
    return int(bool(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.output))
