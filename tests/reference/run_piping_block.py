"""All source pipe/wire rows and fixed selection probes; reference only here."""

import argparse
import json
import time
from itertools import pairwise, product
from math import isclose
from pathlib import Path

import fluids
import fluids.piping as reference

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.piping_catalog import (
    PipeScheduleCatalog,
    WireGaugeCatalog,
    piping_erosional_velocity,
)
from chemworld.physchem.transport import FluidState


def catalogs():
    source = InputSource(
        DataSourceProvenance(
            "local-piping",
            0,
            "Installed fluids 1.3.1 piping catalogs",
            "1.3.1",
            "Local source tables, not bundled",
        ),
        "literature",
    )
    pipe_rows = []
    for schedule, columns in reference.schedule_lookup.items():
        for index, (nominal, inner, outer, wall) in enumerate(zip(*columns, strict=True)):
            pipe_rows.append(
                ScientificRow(
                    f"{schedule}:{index}",
                    "local-piping",
                    schedule,
                    f"{schedule}:{index}",
                    "parameters",
                    {
                        "nominal_size": InputValue(
                            nominal, "dimensionless", None, "Source identifier"
                        ),
                        "inner_diameter": InputValue(inner, "mm", None, "Source dimension"),
                        "outer_diameter": InputValue(outer, "mm", None, "Source dimension"),
                        "wall_thickness": InputValue(wall, "mm", None, "Source dimension"),
                    },
                )
            )
    assert len(pipe_rows) == 1340
    pipe_columns = (
        QuantityColumn(
            "nominal_size", "dimensionless", "parameter", (0, 10000), "Nominal identifier"
        ),
        *(
            QuantityColumn(k, "mm", "parameter", (0, 10000), "Source geometry")
            for k in ("inner_diameter", "outer_diameter", "wall_thickness")
        ),
    )
    pipes = PipeScheduleCatalog(
        ScientificTable("pipes", "parameters", pipe_columns, (source,), tuple(pipe_rows))
    )
    wire_rows = []
    for schedule, (gauges, inches, published_si, _) in reference.wire_schedules.items():
        for index, (gauge, nominal, thickness) in enumerate(
            zip(gauges, inches, published_si, strict=True)
        ):
            wire_rows.append(
                ScientificRow(
                    f"{schedule}:{index}",
                    "local-piping",
                    schedule,
                    f"{schedule}:{index}",
                    "parameters",
                    {
                        "gauge": InputValue(gauge, "dimensionless", None, "Source identifier"),
                        "thickness": InputValue(thickness, "m", None, "Published SI column"),
                        "nominal_thickness": InputValue(
                            nominal, "inch", None, "Source inch column"
                        ),
                    },
                )
            )
    assert len(wire_rows) == 342
    wire_columns = (
        QuantityColumn("gauge", "dimensionless", "parameter", (0, 100), "Gauge identifier"),
        QuantityColumn("thickness", "m", "parameter", (0, 1), "Published SI size"),
        QuantityColumn("nominal_thickness", "inch", "parameter", (0, 1), "Nominal inch size"),
    )
    wires = WireGaugeCatalog(
        ScientificTable("wires", "parameters", wire_columns, (source,), tuple(wire_rows))
    )
    return pipes, wires


def inputs():
    for schedule, (nominals, inners, outers, _) in reference.schedule_lookup.items():
        for index, nominal in enumerate(nominals):
            yield {"family": "pipe", "schedule": schedule, "selector": "nominal", "value": nominal}
            for name, values in (("inner", inners), ("outer", outers)):
                value = (values[index] + (values[index - 1] if index else 0)) * 0.0005
                yield {"family": "pipe", "schedule": schedule, "selector": name, "value": value}
    for schedule, (gauges, inches, _, _) in reference.wire_schedules.items():
        for gauge, inch in zip(gauges, inches, strict=True):
            yield {"family": "gauge", "schedule": schedule, "value": gauge}
            yield {"family": "wire_inverse", "schedule": schedule, "value": inch * 0.0254}
        sizes = sorted(inches)
        for low, high in pairwise(sizes):
            for fraction in (0.05, 0.2, 0.8):
                yield {
                    "family": "wire_inverse",
                    "schedule": schedule,
                    "value": (low + fraction * (high - low)) * 0.0254,
                }
        for bound, value in (("below", sizes[0] * 0.9), ("above", sizes[-1] * 1.1)):
            yield {
                "family": "wire_domain",
                "schedule": schedule,
                "bound": bound,
                "value": value * 0.0254,
            }
    for density, factor in product((0.5, 1.2, 100, 750, 1000), (100, 125, 150, 200, 250)):
        yield {"family": "erosion", "density": density, "factor": factor}


def evaluate(c, pipes, wires):
    family = c["family"]
    if family == "pipe":
        kwargs = {
            {
                "nominal": "nominal_size",
                "inner": "minimum_inner_diameter_m",
                "outer": "minimum_outer_diameter_m",
            }[c["selector"]]: c["value"]
        }
        actual = pipes.select(c["schedule"], **kwargs)
        ref_kwargs = {{"nominal": "NPS", "inner": "Di", "outer": "Do"}[c["selector"]]: c["value"]}
        expected = reference.nearest_pipe(schedule=c["schedule"], **ref_kwargs)
        values = (
            actual.nominal_size,
            actual.inner_diameter_m,
            actual.outer_diameter_m,
            actual.wall_thickness_m,
        )
        yield from zip(("NPS", "inner", "outer", "wall"), values, expected, strict=True)
    elif family == "gauge":
        actual = wires.from_gauge(c["schedule"], c["value"])
        yield (
            "published_SI",
            actual.thickness_m,
            reference.t_from_gauge(c["value"], True, c["schedule"]),
        )
        yield (
            "nominal_m",
            actual.nominal_thickness_m,
            reference.t_from_gauge(c["value"], False, c["schedule"]) * 0.0254,
        )
    elif family == "wire_inverse":
        actual = wires.from_thickness(c["schedule"], c["value"])
        yield "gauge", actual.gauge, reference.gauge_from_t(c["value"], True, c["schedule"])
    elif family == "wire_domain":
        rejected = False
        try:
            wires.from_thickness(c["schedule"], c["value"])
        except ValueError:
            rejected = True
        yield "native_domain_rejection", rejected, True
    else:
        yield (
            "erosional_velocity",
            piping_erosional_velocity(
                FluidState(c["density"], 0.001), empirical_factor_imperial=c["factor"]
            ),
            reference.erosional_velocity(c["density"], c["factor"]),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = list(inputs())
    assert len(cases) == 5749, len(cases)
    start = time.perf_counter()
    pipes, wires = catalogs()
    rows, failures, count, passed, source_domain = [], [], 0, 0, []
    for index, case in enumerate(cases, 1):
        row = {"input": case, "comparisons": []}
        try:
            for label, native, expected in evaluate(case, pipes, wires):
                ok = isclose(native, expected, rel_tol=2e-11, abs_tol=1e-12)
                comparison = {
                    "quantity": label,
                    "native": native,
                    "reference": expected,
                    "passed": ok,
                }
                row["comparisons"].append(comparison)
                count += 1
                passed += ok
                if not ok:
                    failures.append({"input": case, **comparison})
        except Exception as error:
            row["exception"] = {"type": type(error).__name__, "message": str(error)}
            failures.append({"input": case, **row["exception"]})
        if case["family"] == "wire_domain":
            diagnostic = {"input": case}
            try:
                diagnostic["source_value"] = reference.gauge_from_t(
                    case["value"], True, case["schedule"]
                )
                diagnostic["meaning"] = "source returns a nominal length instead of a gauge"
            except ValueError as error:
                diagnostic["source_rejection"] = str(error)
            source_domain.append(diagnostic)
        rows.append(row)
        if index % 500 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"piping {index}/5749, {index / elapsed:.1f} inputs/s, "
                f"ETA {(5749 - index) * elapsed / index:.2f}s",
                flush=True,
            )
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 5749,
        "completed_inputs": len(rows),
        "expected_comparisons": 18151,
        "completed_comparisons": count,
        "passed_comparisons": passed,
        "failures": failures,
        "source_domain_diagnostics": source_domain,
        "elapsed_s": time.perf_counter() - start,
    }
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"failures", "source_domain_diagnostics"}}
        )
    )
    print(f"failures={len(failures)}", flush=True)
    return bool(failures or count != 18151)


if __name__ == "__main__":
    raise SystemExit(main())
