"""Prospective 427-input/818-comparison diameter distribution block."""

import argparse
import json
import time
from itertools import product
from math import isclose
from pathlib import Path

import fluids
import fluids.particle_size_distribution as reference
from scipy.integrate import quad
from scipy.stats import uniform

from chemworld.data.correlation_calibration import CorrelationFitParameter
from chemworld.data.particle_distribution_calibration import DiameterDistributionCalibration
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.particle_distributions import (
    AnalyticDiameterDistribution,
    BinnedDiameterDistribution,
    CustomDiameterDistribution,
    InterpolatedDiameterDistribution,
    diameter_spacing,
)
from chemworld.physchem.sieve_catalog import SieveCatalog

MODELS = (("lognormal", 0.3), ("lognormal", 0.8), ("GGS", 4), ("GGS", 6), ("RR", 4), ("RR", 6))
MEANS = ((3, 2), (4, 3), (1, 0), (3, 3))
SIEVE_FIELDS = tuple(
    n for n in reference.Sieve.__slots__ if n not in ("designation", "old_designation")
)


class SourceFailure:
    def __init__(self, error):
        self.message = f"{type(error).__name__}: {error}"


class SafeReference:
    """Preserve individual source exceptions without hiding native comparisons."""

    def __init__(self, factory):
        self.factory = factory

    def __getattr__(self, name):
        def evaluate(*args, **kwargs):
            try:
                return getattr(self.factory(), name)(*args, **kwargs)
            except Exception as error:
                return SourceFailure(error)

        return evaluate


def source(kind="literature"):
    return InputSource(
        DataSourceProvenance(
            "psd-reference",
            0,
            "Local fluids1.3.1 public equations and sieve records",
            "1.3.1",
            "Source standards data accessed locally; not redistributed as production data",
        ),
        kind,
    )


def models(family, shape, *, truncated=False):
    cls = {
        "lognormal": reference.PSDLognormal,
        "GGS": reference.PSDGatesGaudinSchuhman,
        "RR": reference.PSDRosinRammler,
    }[family]
    kwargs = (
        {"d_min": 0.25e-4, "d_max": min(1e-4, 1.5e-4) if family == "GGS" else 1.5e-4}
        if truncated
        else {}
    )
    native = AnalyticDiameterDistribution(
        family,
        1e-4,
        shape,
        lower_m=kwargs.get("d_min", 0),
        upper_m=kwargs.get("d_max", float("inf")),
    )
    old = SafeReference(lambda: cls(1e-4 if family != "RR" else 1e-4 ** (-shape), shape, **kwargs))
    return native, old


def layout(name):
    return ([1e-4, 2e-4, 4e-4] if name == "nodes" else [1e-4, 2e-4, 4e-4, 8e-4]), [0.2, 0.3, 0.5]


def load_sieves():
    lengths = {
        "opening",
        "X_variation_max",
        "Y_variation_avg",
        "d_wire",
        "d_wire_min",
        "d_wire_max",
        "max_opening",
        "opening_inch",
    }
    columns = tuple(
        QuantityColumn(
            field,
            "m" if field in lengths else "dimensionless",
            "observation",
            (0, 1e6),
            "Source-declared field; no compliance certification",
        )
        for field in SIEVE_FIELDS
    )
    rows, aliases = [], {}
    for standard, catalog in (
        ("ASTM", reference.ASTM_E11_sieves),
        ("ISO", reference.ISO_3310_1_sieves),
    ):
        for designation, item in catalog.items():
            identity = standard + ":" + designation
            values = {}
            for column in columns:
                field = column.quantity_id
                value = getattr(item, field)
                if value is not None and field == "opening_inch":
                    value *= 0.0254
                values[field] = InputValue(
                    value,
                    column.unit,
                    None,
                    "Original source uncertainty unspecified; inch field converted explicitly",
                    "missing" if value is None else "valid",
                )
            rows.append(
                ScientificRow(identity, "psd-reference", standard, identity, "test", values)
            )
            if item.old_designation is not None:
                aliases[identity] = (item.old_designation,)
    series = {
        name: tuple("ISO:" + item.designation for item in getattr(reference, name))
        for name in ("ISO_3310_1_R10", "ISO_3310_1_R20", "ISO_3310_1_R20_3", "ISO_3310_1_R40_3")
    }
    data = SieveCatalog(
        ScientificTable("sieves", "measurements", columns, (source(),), tuple(rows)),
        series=series,
        aliases=aliases,
    )
    return SieveCatalog.from_dict(json.loads(json.dumps(data.to_dict())))


def inputs():
    for model, order, ratio in product(MODELS, (0, 2, 3), (0.2, 0.8, 1.5)):
        yield {"family": "density", "model": model, "order": order, "ratio": ratio}
    for model, order, fraction in product(MODELS, (0, 2, 3), (0.1, 0.5, 0.9)):
        yield {"family": "quantile", "model": model, "order": order, "fraction": fraction}
    for model, mean in product(MODELS, MEANS):
        yield {"family": "mean", "model": model, "mean": mean}
    for model, order, ratio in product(MODELS, (0, 2, 3), (0.4, 0.8, 1.2)):
        yield {"family": "truncated", "model": model, "order": order, "ratio": ratio}
    for model, power, interval in product(MODELS, (-2, 0, 2), ((0.2, 0.8), (0.5, 1))):
        yield {"family": "integral", "model": model, "power": power, "interval": interval}
    for arrangement, order in product(("nodes", "bins"), (0, 2, 3)):
        yield {"family": "binned", "layout": arrangement, "order": order}
    for arrangement, diameter in product(("nodes", "bins"), (1e-5, 1.5e-4, 3e-4, 5e-4, 1e-3)):
        yield {"family": "interpolated", "layout": arrangement, "diameter": diameter}
    for method, lower, points in product(("linear", "logarithmic"), (1e-6, 1e-4), (5, 10)):
        yield {
            "family": "spacing",
            "method": method,
            "lower": lower,
            "upper": lower * 100,
            "points": points,
        }
    for series, endpoint in product((5, 10, 20), ("lower", "upper")):
        yield {"family": "spacing", "method": f"R{series}", endpoint: 1e-4, "points": 5}
    for standard, catalog in (
        ("ASTM", reference.ASTM_E11_sieves),
        ("ISO", reference.ISO_3310_1_sieves),
    ):
        for designation in catalog:
            yield {"family": "sieve", "standard": standard, "designation": designation}
    for series, interval in product(
        ("ISO_3310_1_R10", "ISO_3310_1_R20", "ISO_3310_1_R20_3", "ISO_3310_1_R40_3"),
        ((1e-5, 1e-3), (1e-3, 0.1)),
    ):
        yield {"family": "sieve_series", "series": series, "interval": interval}
    for order, ratio in product((0, 2, 3), (0.3, 0.7, 0.9)):
        yield {"family": "custom", "order": order, "ratio": ratio}
    for model in (("lognormal", 0.5), ("GGS", 4), ("RR", 4)):
        yield {"family": "fit", "model": model}


def comparison(quantity, native, expected, *, rtol=2e-8, atol=1e-10):
    if isinstance(expected, SourceFailure):
        return {
            "quantity": quantity,
            "native": native,
            "reference_exception": expected.message,
            "passed": False,
            "rtol": rtol,
            "atol": atol,
        }
    return {
        "quantity": quantity,
        "native": native,
        "reference": expected,
        "rtol": rtol,
        "atol": atol,
        "passed": isclose(native, expected, rel_tol=rtol, abs_tol=atol),
    }


def array_comparison(quantity, native, expected):
    native, expected = list(native), list(expected)
    return {
        "quantity": quantity,
        "native": native,
        "reference": expected,
        "passed": len(native) == len(expected)
        and all(
            isclose(a, b, rel_tol=2e-12, abs_tol=1e-14)
            for a, b in zip(native, expected, strict=True)
        ),
    }


def fit_case(family, shape):
    _, expected = models(family, shape)
    columns = (
        QuantityColumn("diameter", "m", "condition", (0, 1), "Particle diameter"),
        QuantityColumn("cdf", "dimensionless", "observation", (0, 1), "Volume-basis CDF"),
    )
    rows = []
    for split, ratios in (
        ("train", (0.3, 0.4, 0.6, 0.8, 1, 1.2)),
        ("validation", (0.5, 0.7, 0.9, 1.1)),
    ):
        for ratio in ratios:
            identity = split + str(ratio)
            rows.append(
                ScientificRow(
                    identity,
                    "psd-reference",
                    split,
                    identity,
                    split,
                    {
                        "diameter": InputValue(
                            ratio * 1e-4, "m", None, "Exact synthetic condition"
                        ),
                        "cdf": InputValue(
                            expected.cdf(ratio * 1e-4),
                            "dimensionless",
                            0.01,
                            "Declared synthetic fit weight",
                        ),
                    },
                )
            )
    table = ScientificTable("cdf-fit", "measurements", columns, (source("synthetic"),), tuple(rows))
    initial = AnalyticDiameterDistribution(family, 1.2e-4, shape * 1.2)
    parameters = (
        CorrelationFitParameter("scale_m", (0.5e-4, 2e-4), 1.2e-4, "log"),
        CorrelationFitParameter("shape", (0.2, 8), shape * 1.2, "log"),
    )
    report = DiameterDistributionCalibration(table, initial, parameters).fit(max_nfev=80)
    if report["status"] != "fitted":
        raise ValueError(json.dumps({key: value for key, value in report.items() if key != "runs"}))
    return [
        comparison("fit_scale", report["estimate"]["scale_m"], 1e-4, rtol=1e-5),
        comparison("fit_shape", report["estimate"]["shape"], shape, rtol=1e-5),
        comparison(
            "heldout_max_cdf_error",
            max(abs(row["error"]) for row in report["validation"]["measurements"]),
            0,
            rtol=0,
            atol=1e-6,
        ),
    ]


def evaluate(case, sieves):
    family = case["family"]
    if "model" in case:
        name, shape = case["model"]
        native, expected = models(name, shape, truncated=family == "truncated")
    if family in {"density", "truncated"}:
        diameter, order = case["ratio"] * 1e-4, case["order"]
        result = [
            comparison(
                "pdf", native.pdf(diameter, order=order), expected.pdf(diameter, n=order), atol=1e-7
            ),
            comparison("cdf", native.cdf(diameter, order=order), expected.cdf(diameter, n=order)),
        ]
        if family == "truncated":
            _, untruncated = models(name, shape)
            low, high, ratio = native.lower_m / 1e-4, native.upper_m / 1e-4, case["ratio"]

            def density(x):
                return untruncated.pdf(x * 1e-4) * 1e-4 * x ** (order - 3)

            normalization = quad(density, low, high, epsabs=1e-12, epsrel=1e-12)[0]
            independent_pdf = density(ratio) / (1e-4 * normalization) if low <= ratio <= high else 0
            independent_cdf = (
                quad(density, low, min(high, ratio), epsabs=1e-12, epsrel=1e-12)[0] / normalization
            )
            result += [
                comparison(
                    "independent_pdf", native.pdf(diameter, order=order), independent_pdf, atol=1e-7
                ),
                comparison("independent_cdf", native.cdf(diameter, order=order), independent_cdf),
            ]
        return result
    if family == "quantile":
        value = native.quantile(case["fraction"], order=case["order"])
        return [
            comparison(
                "quantile", value, expected.dn(case["fraction"], n=case["order"]), atol=1e-12
            ),
            comparison("cdf_roundtrip", native.cdf(value, order=case["order"]), case["fraction"]),
        ]
    if family == "mean":
        p, q = case["mean"]
        return [
            comparison(
                "mean",
                native.mean_size(p, q),
                expected.mean_size(p, q),
                rtol=2e-5 if p == q else 2e-8,
            )
        ]
    if family == "integral":
        low, high = (value * 1e-4 for value in case["interval"])
        function = {
            "lognormal": reference.pdf_lognormal_basis_integral,
            "GGS": reference.pdf_Gates_Gaudin_Schuhman_basis_integral,
            "RR": reference.pdf_Rosin_Rammler_basis_integral,
        }[name]
        scale = 1e-4 if name != "RR" else 1e-4 ** (-shape)
        exact = function(high, scale, shape, case["power"]) - function(
            low, scale, shape, case["power"]
        )
        return [
            comparison(
                "basis_integral", native.basis_integral(case["power"], low, high), exact, atol=1e-20
            )
        ]
    if family == "binned":
        ds, fs = layout(case["layout"])
        native = BinnedDiameterDistribution(ds, fs, order=case["order"])
        expected = reference.ParticleSizeDistribution(ds, fs, order=case["order"])
        records = []
        for order, attribute in enumerate(
            ("number_fractions", "length_fractions", "area_fractions", "fractions")
        ):
            records.extend(
                comparison(f"fraction_{order}_{i}", a, b)
                for i, (a, b) in enumerate(
                    zip(native.fractions(order), getattr(expected, attribute), strict=True)
                )
            )
        records.extend(
            comparison(f"mean_{p}_{q}", native.mean_size(p, q), expected.mean_size(p, q))
            for p, q in MEANS
        )
        records.append(comparison("vssa", native.volume_specific_surface_area_m_inv, expected.vssa))
        return records
    if family == "interpolated":
        ds, fs = layout(case["layout"])
        native, expected = (
            InterpolatedDiameterDistribution(ds, fs),
            reference.PSDInterpolated(ds, fs),
        )
        d = case["diameter"]
        return [
            comparison("pdf", native.pdf(d), expected.pdf(d), atol=1e-7),
            comparison("cdf", native.cdf(d), expected.cdf(d)),
        ]
    if family == "spacing":
        native = diameter_spacing(
            lower_m=case.get("lower"),
            upper_m=case.get("upper"),
            points=case["points"],
            method=case["method"],
        )
        expected = reference.psd_spacing(
            d_min=case.get("lower"),
            d_max=case.get("upper"),
            pts=case["points"],
            method=case["method"],
        )
        return [array_comparison("spacing", native, expected)]
    if family == "sieve":
        original = (
            reference.ASTM_E11_sieves if case["standard"] == "ASTM" else reference.ISO_3310_1_sieves
        )[case["designation"]]
        native = sieves.lookup(case["standard"], case["designation"])
        observed = {name: native.values[name].value for name in SIEVE_FIELDS}
        expected = {
            name: getattr(original, name) * (0.0254 if name == "opening_inch" else 1)
            if getattr(original, name) is not None
            else None
            for name in SIEVE_FIELDS
        }
        observed["aliases"] = list(native.aliases)
        expected["aliases"] = [] if original.old_designation is None else [original.old_designation]
        return [
            {
                "quantity": "sieve_record",
                "native": observed,
                "reference": expected,
                "passed": observed == expected,
            }
        ]
    if family == "sieve_series":
        low, high = case["interval"]
        expected = sorted(
            s.opening for s in getattr(reference, case["series"]) if low <= s.opening <= high
        )
        return [
            array_comparison(
                "series", sieves.spacing(case["series"], lower_m=low, upper_m=high), expected
            )
        ]
    if family == "custom":
        frozen = uniform(loc=0.2e-4, scale=0.8e-4)
        native = CustomDiameterDistribution(frozen.pdf, scale_m=1e-4, lower_m=0.2e-4, upper_m=1e-4)
        expected = reference.PSDCustom(frozen, d_excessive=1e-4, d_min=0.2e-4, d_max=1e-4)
        d, order = case["ratio"] * 1e-4, case["order"]
        return [
            comparison("pdf", native.pdf(d, order=order), expected.pdf(d, n=order), atol=1e-7),
            comparison("cdf", native.cdf(d, order=order), expected.cdf(d, n=order)),
        ]
    return fit_case(*case["model"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve every attempt")
    cases, records = list(inputs()), []
    assert len(cases) == 427
    sieves = load_sieves()
    counts = {
        "density": 2,
        "quantile": 2,
        "mean": 1,
        "truncated": 4,
        "integral": 1,
        "binned": 17,
        "interpolated": 2,
        "spacing": 1,
        "sieve": 1,
        "sieve_series": 1,
        "custom": 2,
        "fit": 3,
    }
    assert sum(counts[case["family"]] for case in cases) == 818
    started = time.monotonic()
    for index, case in enumerate(cases, 1):
        try:
            comparisons = evaluate(case, sieves)
            assert len(comparisons) == counts[case["family"]]
            records.append({"input": case, "comparisons": comparisons})
        except Exception as error:
            records.append(
                {
                    "input": case,
                    "exception": f"{type(error).__name__}: {error}",
                    "expected_comparisons": counts[case["family"]],
                }
            )
        if index % 25 == 0 or index == len(cases):
            elapsed = max(1e-6, time.monotonic() - started)
            print(
                f"PSD {index}/{len(cases)} {index / elapsed:.1f} inputs/s "
                f"ETA {(len(cases) - index) * elapsed / index:.1f}s",
                flush=True,
            )
    comparisons = [item for record in records for item in record.get("comparisons", [])]
    summary = {
        "reference_version": fluids.__version__,
        "inputs": 427,
        "expected_comparisons": 818,
        "completed_comparisons": len(comparisons),
        "passed": sum(item["passed"] for item in comparisons),
        "failed": sum(not item["passed"] for item in comparisons),
        "exceptions": sum("exception" in record for record in records),
        "source_exceptions": sum("reference_exception" in item for item in comparisons),
        "failures": [
            record
            for record in records
            if "exception" in record
            or any(not item["passed"] for item in record.get("comparisons", []))
        ],
        "records": records,
        "elapsed_s": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in ("records", "failures")}
        )
    )


if __name__ == "__main__":
    main()
