"""Fixed development reference block; external source tables stay outside Git."""

from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path
from typing import Any

import fluids
import fluids.open_flow as reference

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.open_channel import (
    chezy_to_manning,
    chezy_velocity,
    manning_roughness_table,
    manning_to_chezy,
    manning_velocity,
    select_manning_roughness,
    uniform_channel_flow,
    weir_flow,
)


def run(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []

    def evaluate(kind: str, inputs: dict[str, Any], function: Any) -> None:
        row: dict[str, Any] = {"kind": kind, "inputs": inputs, "comparisons": []}
        try:
            for label, actual, expected in function():
                row["comparisons"].append({"quantity": label, "native": actual,
                    "reference": expected,
                    "passed": abs(actual-expected) <= 1e-12*(1+abs(expected))})
            row["passed"] = all(item["passed"] for item in row["comparisons"])
        except Exception as exc:
            row["passed"] = False
            row["exception"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    for radius, slope, n in product((.1, 1., 5.), (0., .001, .03), (.01, .03, .1)):
        def channel(radius=radius, slope=slope, n=n):
            c = manning_to_chezy(n, radius)
            expected_v = reference.V_Manning(radius, slope, n)
            return [("ManningV", manning_velocity(radius, slope, n), expected_v),
                ("C", c, reference.n_Manning_to_C_Chezy(n, radius)),
                ("ChezyV", chezy_velocity(radius, slope, c), reference.V_Chezy(radius, slope, c)),
                ("n", chezy_to_manning(c, radius), reference.C_Chezy_to_n_Manning(c, radius)),
                ("Q", uniform_channel_flow(2, 2/radius, slope, roughness=n), 2*expected_v)]
        evaluate("channel", {"Rh": radius, "slope": slope, "n": n}, channel)
    methods = {
        "kindsvater_carter": reference.Q_weir_rectangular_Kindsvater_Carter,
        "sia_contracted_reference": reference.Q_weir_rectangular_SIA,
        "ackers_full": reference.Q_weir_rectangular_full_Ackers,
        "kindsvater_carter_full": reference.Q_weir_rectangular_full_Kindsvater_Carter,
        "rehbock_full": reference.Q_weir_rectangular_full_Rehbock,
        "sia_full_reference": reference.Q_weir_rectangular_full_SIA,
    }
    for method, ref in methods.items():
        for h, p, b in product((.05, .15, .3), (.6, 1.), (.6, 1.2)):
            def rectangle(method=method, ref=ref, h=h, p=p, b=b):
                kwargs = ({"channel_width_m": 3*b}
                          if method in {"kindsvater_carter", "sia_contracted_reference"} else {})
                actual = weir_flow(method, head_m=h, crest_height_m=p, width_m=b,
                    reference_only="reference" in method, **kwargs).flow_m3_s
                expected = (ref(h, p, b, 3*b) if method == "sia_contracted_reference"
                            else ref(h, p, b))
                return [("Q", actual, expected)]
            evaluate(method, {"h": h, "p": p, "b": b}, rectangle)
    for angle, h in product((20, 40, 60, 80, 100), (.05, .1, .15)):
        evaluate("shen_v", {"angle": angle, "h": h}, lambda angle=angle, h=h: [("Q",
            weir_flow("shen_v", head_m=h, crest_height_m=.8, width_m=2,
                angle_degrees=angle).flow_m3_s, reference.Q_weir_V_Shen(h, angle))])
    source_counts = {}
    for name in ("n_closed_conduit", "n_excavated_dredged", "n_lined_built", "n_natural"):
        source = InputSource(DataSourceProvenance(name, 0,
            "fluids 1.3.1 Chow/Blevins roughness source; local reference consumption",
            version="1.3.1", license="original table redistribution not established"), "literature")
        original = getattr(reference, name)
        table = manning_roughness_table(name, original, source=source)
        source_counts[name] = len(table.rows)
        for row in table.rows:
            path = tuple(json.loads(row.row_id))
            expected = original
            for key in path:
                expected = expected[key]

            def source_values(table=table, path=path, expected=expected):
                comparisons = []
                for stat, n in zip(("minimum", "normal", "maximum"), expected, strict=True):
                    selected = select_manning_roughness(table, path, statistic=stat).value
                    comparisons.append((stat, selected, n))
                    comparisons.append((stat+"V", manning_velocity(.5, .001, selected),
                                        reference.V_Manning(.5, .001, n)))
                return comparisons
            evaluate(name, {"path": path}, source_values)
    comparisons = [item for row in rows for item in row["comparisons"]]
    summary = {"task": "NATIVE-CONV-30", "mode": "development",
        "block": "open_flow", "reference_version": fluids.__version__,
        "expected_units": 223, "expected_comparisons": 876,
        "units_completed": len(rows), "units_passed": sum(row["passed"] for row in rows),
        "comparisons_executed": len(comparisons),
        "comparisons_passed": sum(item["passed"] for item in comparisons),
        "source_rows": source_counts, "elapsed_s": time.perf_counter()-started,
        "failures": [row for row in rows if not row["passed"]],
        "boundaries": ["SIA reference-only comparisons do not establish physical validity",
                       "Source statistics are not measurement uncertainties",
                       "Steady engineering, no dynamic open-channel solver"]}
    (output/"rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
