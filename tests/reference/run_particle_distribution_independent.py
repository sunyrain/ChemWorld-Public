"""Fixed 32-input/44-comparison analytic checks for preserved source defects."""

import argparse
import json
from itertools import product
from math import isclose
from pathlib import Path

import mpmath as mp

from chemworld.physchem.particle_distributions import (
    AnalyticDiameterDistribution,
    CustomDiameterDistribution,
)


def compare(quantity, value, expected, *, rtol=2e-8, atol=1e-10):
    expected = float(expected)
    return {
        "quantity": quantity,
        "native": value,
        "reference": expected,
        "passed": isclose(value, expected, rel_tol=rtol, abs_tol=atol),
        "rtol": rtol,
        "atol": atol,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve attempts")
    records = []
    with mp.workdps(80):
        scale = mp.mpf("1e-4")
        for shape, order, ratio in product((4, 6), (0, 2), (0.2, 0.8, 1.5)):
            model = AnalyticDiameterDistribution("GGS", 1e-4, shape)
            exponent, x = mp.mpf(shape + order - 3), mp.mpf(str(ratio))
            density = exponent / scale * x ** (exponent - 1) if x <= 1 else 0
            cdf = min(x, mp.mpf(1)) ** exponent
            records.append(
                {
                    "family": "GGS_density",
                    "shape": shape,
                    "order": order,
                    "ratio": ratio,
                    "comparisons": [
                        compare("pdf", model.pdf(ratio * 1e-4, order=order), density, atol=1e-7),
                        compare("cdf", model.cdf(ratio * 1e-4, order=order), cdf),
                    ],
                }
            )
        for shape, order, fraction in product((4, 6), (0, 2), (0.1, 0.5, 0.9)):
            model = AnalyticDiameterDistribution("GGS", 1e-4, shape)
            quantile = scale * mp.mpf(str(fraction)) ** (mp.mpf(1) / (shape + order - 3))
            records.append(
                {
                    "family": "GGS_quantile",
                    "shape": shape,
                    "order": order,
                    "fraction": fraction,
                    "comparisons": [
                        compare(
                            "quantile", model.quantile(fraction, order=order), quantile, atol=1e-12
                        )
                    ],
                }
            )
        for shape, (p, q) in product((4, 6), ((3, 2), (1, 0), (3, 3))):
            model = AnalyticDiameterDistribution("GGS", 1e-4, shape)
            mean = (
                scale * (mp.mpf(shape + q - 3) / (shape + p - 3)) ** (mp.mpf(1) / (p - q))
                if p != q
                else scale * mp.exp(-mp.mpf(1) / (shape + p - 3))
            )
            records.append(
                {
                    "family": "GGS_mean",
                    "shape": shape,
                    "p": p,
                    "q": q,
                    "comparisons": [compare("mean", model.mean_size(p, q), mean)],
                }
            )
        for order in (0, 2):
            model = CustomDiameterDistribution(
                lambda d: 1 / 0.8e-4, scale_m=1e-4, lower_m=0.2e-4, upper_m=1e-4
            )
            a, x, b = mp.mpf(".2"), mp.mpf(".3"), mp.mpf(1)
            power = order - 2
            cdf = (
                (x**power - a**power) / (b**power - a**power)
                if power
                else mp.log(x / a) / mp.log(b / a)
            )
            records.append(
                {
                    "family": "custom_uniform",
                    "order": order,
                    "comparisons": [compare("cdf", model.cdf(0.3e-4, order=order), cdf)],
                }
            )
    comparisons = [item for record in records for item in record["comparisons"]]
    assert len(records) == 32 and len(comparisons) == 44
    report = {
        "inputs": 32,
        "comparisons": 44,
        "passed": sum(c["passed"] for c in comparisons),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}))


if __name__ == "__main__":
    main()
