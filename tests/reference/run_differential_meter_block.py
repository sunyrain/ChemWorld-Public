"""Fixed 970-input/1705-comparison meter block; source data loaded locally."""

import argparse
import json
import time
from itertools import product
from math import isclose, pi, sqrt
from pathlib import Path

import fluids
import fluids.flow_meter as reference
import mpmath as mp

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.differential_meters import (
    DifferentialMeterSpec,
    meter_discharge,
    meter_flow_coefficient,
    solve_differential_meter,
)
from chemworld.physchem.meter_correlations import (
    METER_TYPES,
    meter_beta,
    meter_coefficient_from_loss,
    meter_discharge_coefficient,
    meter_expansibility,
    meter_loss_coefficient,
    miller_orifice_coefficient,
    wet_venturi_correction,
)
from chemworld.physchem.meter_tables import MeterTableSurface
from chemworld.physchem.transport import FluidState

FORMULA = tuple(name for name in METER_TYPES if not name.startswith("Hollingshead"))
GRID_NAMES = {
    "orifice": "Hollingshead orifice",
    "cone": "Hollingshead v cone",
    "wedge": "Hollingshead wedge",
    "smooth": "Hollingshead venturi smooth",
    "sharp": "Hollingshead venturi sharp",
}


def grids():
    result = {}
    for key, prefix in (("orifice", "orifice_std"), ("cone", "cone"), ("wedge", "wedge")):
        result[key] = (
            getattr(reference, prefix + "_betas_Hollingshead"),
            getattr(reference, prefix + "_Res_Hollingshead"),
            getattr(reference, prefix + "_Hollingshead_Cs"),
        )
    for key in ("smooth", "sharp"):
        result[key] = (
            [0.5],
            reference.venturi_Res_Hollingshead,
            [getattr(reference, "venturi_" + key + "_Cs_Hollingshead")],
        )
    return result


def table(name, xs, ys, zs, *, recovery=False):
    source = InputSource(
        DataSourceProvenance(
            "meter-reference",
            0,
            "Local fluids1.3.1 source calibration observations",
            "1.3.1",
            "Source observations not production distributed",
        ),
        "literature",
    )
    yname, unit = ("diameter", "m") if recovery else ("reynolds", "dimensionless")
    columns = (
        QuantityColumn(
            "beta", "dimensionless", "parameter", (0, 1), "Effective area ratio square root"
        ),
        QuantityColumn(yname, unit, "parameter", (0, 1e10), "Original grid coordinate"),
        QuantityColumn(
            "coefficient", "dimensionless", "parameter", (0, 10), "Original source value"
        ),
    )
    rows = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            rows.append(
                ScientificRow(
                    str(len(rows)),
                    "meter-reference",
                    name,
                    f"{name}:{i}:{j}",
                    "parameters",
                    {
                        "beta": InputValue(
                            x, "dimensionless", None, "Source uncertainty not supplied"
                        ),
                        yname: InputValue(y, unit, None, "Source uncertainty not supplied"),
                        "coefficient": InputValue(
                            zs[i][j], "dimensionless", None, "Source uncertainty not supplied"
                        ),
                    },
                )
            )
    native = MeterTableSurface(
        ScientificTable(name, "parameters", columns, (source,), tuple(rows)),
        y_column=yname,
        y_unit=unit,
        log_y=not recovery,
        degree_x=1 if recovery else 3,
        degree_y=1 if recovery else 3,
        outside="clamp" if recovery else "raise",
    )
    return MeterTableSurface.from_dict(json.loads(json.dumps(native.to_dict())))


def load_tables():
    result = {name: table(name, *grid) for name, grid in grids().items()}
    zs = list(zip(reference.venturi_tube_dP_high, reference.venturi_tube_dP_low, strict=True))
    result["recovery"] = table(
        "recovery", reference.venturi_tube_betas, reference.D_bound_venturi_tube, zs, recovery=True
    )
    return result


def inputs():
    for name, d, ratio, re in product(FORMULA, (0.05, 0.15), (0.4, 0.65), (2e4, 2e5)):
        yield {"family": "formula", "meter": name, "D": d, "ratio": ratio, "Re": re}
    for taps, d, beta, re in product(
        ("corner", "flange", "D and D/2", "pipe"), (0.05, 0.0584, 0.15), (0.3, 0.7), (200, 1e5)
    ):
        yield {
            "family": "Miller",
            "subtype": "orifice",
            "taps": taps,
            "D": d,
            "beta": beta,
            "Re": re,
        }
    for taps, position, d, beta, re in product(
        ("flange", "vena contracta"),
        ("180 degree", "90 degree"),
        (0.05, 0.15),
        (0.3, 0.7),
        (1e4, 1e5),
    ):
        yield {
            "family": "Miller",
            "subtype": "eccentric orifice",
            "taps": taps,
            "position": position,
            "D": d,
            "beta": beta,
            "Re": re,
        }
    for taps, d, beta in product(("flange", "vena contracta"), (0.05, 0.15), (0.3, 0.7)):
        yield {
            "family": "Miller",
            "subtype": "segmental orifice",
            "taps": taps,
            "D": d,
            "beta": beta,
            "Re": 1e5,
        }
    for beta, factor in product((0.3, 0.7), (249, 250, 500, 501)):
        yield {
            "family": "Miller",
            "subtype": "conical orifice",
            "taps": "corner",
            "D": 0.1,
            "beta": beta,
            "Re": beta * factor,
        }
    for d, beta in product((0.05, 0.15), (0.3, 0.7)):
        yield {
            "family": "Miller",
            "subtype": "quarter circle orifice",
            "taps": "corner",
            "D": d,
            "beta": beta,
            "Re": 1e5,
        }
    for d, ratio in product((0.01, 0.01778, 0.02, 0.03556, 0.05), (0.2, 0.5, 0.8)):
        yield {"family": "Miller_wedge", "D": d, "ratio": ratio}
    for method, beta, ratio, k in product(
        ("orifice", "orifice_1989", "nozzle", "cone", "conical_orifice"),
        (0.3, 0.7),
        (0.75, 0.9, 1),
        (1, 1 + 1e-10, 1.4),
    ):
        yield {"family": "expansion", "method": method, "beta": beta, "ratio": ratio, "k": k}
    for ratio, pressure_ratio in product((0.2, 0.5, 0.8), (0.75, 0.9)):
        yield {"family": "effective_beta", "ratio": ratio, "pressure_ratio": pressure_ratio}
    for beta, coefficient in product((0.2, 0.5, 0.8), (0.5, 0.75, 0.99)):
        yield {"family": "conversion", "beta": beta, "C": coefficient}
    for name, d, ratio, dp in product(
        (n for n in FORMULA if n not in {"unspecified meter", "venturi nozzle"}),
        (0.05, 0.15),
        (0.4, 0.65),
        (1e3, 1e4),
    ):
        yield {"family": "loss", "meter": name, "D": d, "ratio": ratio, "dP": dp}
    for name, k in product(FORMULA, (None, 1.4)):
        yield {"family": "inverse", "meter": name, "k": k}
    for name, (betas, res, coefficients) in grids().items():
        for i, beta in enumerate(betas):
            for j, re in enumerate(res):
                yield {
                    "family": "grid",
                    "grid": name,
                    "beta": beta,
                    "Re": re,
                    "coefficient": coefficients[i][j],
                }
    for name, betas in (
        ("orifice", (0.55, 0.625, 0.675)),
        ("cone", (0.68, 0.75)),
        ("wedge", (0.55,)),
        ("smooth", (0.5,)),
        ("sharp", (0.5,)),
    ):
        for beta, re in product(betas, (3, 75, 500, 5e4)):
            yield {"family": "interpolation", "grid": name, "beta": beta, "Re": re}
    for i, beta in enumerate(reference.venturi_tube_betas):
        for j, d in enumerate(reference.D_bound_venturi_tube):
            yield {
                "family": "recovery",
                "beta": beta,
                "D": d,
                "value": (reference.venturi_tube_dP_high, reference.venturi_tube_dP_low)[j][i],
            }
    for mg, ml, rhog, beta, h in product((0.2, 2, 20), (0, 0.1, 1), (5, 50), (0.4, 0.7), (0.5, 1)):
        yield {"family": "wet", "mg": mg, "ml": ml, "rhog": rhog, "beta": beta, "H": h}


def compare(quantity, native, expected, *, rtol=2e-9, atol=1e-10):
    try:
        target = expected() if callable(expected) else expected
    except Exception as error:
        return {
            "quantity": quantity,
            "native": native,
            "reference_exception": f"{type(error).__name__}: {error}",
            "passed": False,
        }
    return {
        "quantity": quantity,
        "native": native,
        "reference": target,
        "rtol": rtol,
        "atol": atol,
        "passed": isclose(native, target, rel_tol=rtol, abs_tol=atol),
    }


def reference_expansion(method, beta, ratio, k):
    d, do, p1, p2 = 0.1, 0.1 * beta, 2e5, 2e5 * ratio
    if method == "cone":
        return reference.cone_meter_expansibility_Stewart(d, d * sqrt(1 - beta * beta), p1, p2, k)
    if method == "conical_orifice":
        return 0.5 * (
            reference.orifice_expansibility(d, do, p1, p2, k)
            + reference.nozzle_expansibility(d, do, p1, p2, k)
        )
    function = {
        "orifice": reference.orifice_expansibility,
        "orifice_1989": reference.orifice_expansibility_1989,
        "nozzle": reference.nozzle_expansibility,
    }[method]
    return function(d, do, p1, p2, k)


def independent_expansion(method, beta, ratio, k):
    with mp.workdps(80):
        b, r, k = mp.mpf(str(beta)), mp.mpf(str(ratio)), mp.mpf(str(k))
        if r == 1:
            return 1.0
        orifice = 1 - (mp.mpf(".351") + mp.mpf(".256") * b**4 + mp.mpf(".93") * b**8) * (
            1 - r ** (1 / k)
        )
        if method == "orifice":
            return float(orifice)
        if method == "orifice_1989":
            return float(1 - (mp.mpf(".41") + mp.mpf(".35") * b**4) * (1 - r) / k)
        if method == "cone":
            return float(1 - (mp.mpf(".649") + mp.mpf(".696") * b**4) * (1 - r) / k)
        term = -mp.log(r) if k == 1 else k / (k - 1) * (1 - r ** ((k - 1) / k))
        nozzle = mp.sqrt(r ** (2 / k) * (1 - b**4) / (1 - b**4 * r ** (2 / k)) * term / (1 - r))
        return float((nozzle + orifice) / 2 if method == "conical_orifice" else nozzle)


def spec_for(name, d, ratio, tables):
    return DifferentialMeterSpec(
        d,
        d * ratio,
        meter_type=name,
        taps="flange"
        if name in {"Miller eccentric orifice", "Miller segmental orifice"}
        else "corner",
        specified_coefficient=0.7 if name == "unspecified meter" else None,
        recovery_table=tables["recovery"]
        if "venturi" in name and name != "venturi nozzle"
        else None,
    )


def evaluate(case, tables):
    family = case["family"]
    if family in {"formula", "loss"}:
        spec = spec_for(case["meter"], case["D"], case["ratio"], tables)
        mass = case.get("Re", 1e5) * pi * spec.pipe_diameter_m * 0.001 / 4
        common = {
            "D": spec.pipe_diameter_m,
            "D2": spec.secondary_dimension_m,
            "m": mass,
            "P1": 2e5,
            "P2": 1.8e5,
            "rho": 1000,
            "mu": 0.001,
            "k": 1.3,
            "meter_type": spec.meter_type,
            "taps": spec.taps,
            "tap_position": spec.tap_position,
            "C_specified": spec.specified_coefficient,
        }
        c, epsilon = reference.differential_pressure_meter_C_epsilon(**common)
        native_c = spec.coefficient(case.get("Re", 1e5))
        if family == "loss":
            return [
                compare(
                    "permanent_loss",
                    spec.permanent_loss(case["dP"], native_c),
                    reference.differential_pressure_meter_dP(
                        spec.pipe_diameter_m,
                        spec.secondary_dimension_m,
                        2e5,
                        2e5 - case["dP"],
                        C=c,
                        meter_type=spec.meter_type,
                    ),
                )
            ]
        native_e = spec.expansibility(0.9, 1.3)
        return [
            compare(
                "beta",
                spec.beta,
                reference.differential_pressure_meter_beta(
                    spec.pipe_diameter_m, spec.secondary_dimension_m, spec.meter_type
                ),
            ),
            compare("C", native_c, c),
            compare("epsilon", native_e, epsilon),
            compare(
                "direct_mass",
                meter_discharge(
                    pipe_diameter_m=spec.pipe_diameter_m,
                    beta=spec.beta,
                    pressure_difference_Pa=2e4,
                    density_kg_m3=1000,
                    coefficient=native_c,
                    expansibility=native_e,
                ),
                reference.flow_meter_discharge(
                    spec.pipe_diameter_m,
                    spec.secondary_dimension_m,
                    2e5,
                    1.8e5,
                    1000,
                    c,
                    epsilon,
                    spec.meter_type,
                ),
            ),
        ]
    if family == "Miller":
        d, b, re = case["D"], case["beta"], case["Re"]
        m = re * pi * d * 0.001 / 4
        native = miller_orifice_coefficient(
            b,
            d,
            re,
            subtype=case["subtype"],
            taps=case["taps"],
            tap_position=case.get("position", "180 degree"),
        )
        return [
            compare(
                "C_Miller",
                native,
                reference.C_Miller_1996(
                    d,
                    d * b,
                    1000,
                    0.001,
                    m,
                    subtype=case["subtype"],
                    taps=case["taps"],
                    tap_position=case.get("position", "180 degree"),
                ),
            )
        ]
    if family == "Miller_wedge":
        d, h = case["D"], case["D"] * case["ratio"]
        native = meter_discharge_coefficient(
            meter_beta(d, h, "wedge meter"), d, 1e5, meter_type="Miller wedge meter"
        )
        return [compare("C_Miller_wedge", native, reference.C_wedge_meter_Miller(d, h))]
    if family == "expansion":
        method, b, r, k = case["method"], case["beta"], case["ratio"], case["k"]
        native = meter_expansibility(b, r, k, method=method)
        return [
            compare("epsilon_source", native, lambda: reference_expansion(method, b, r, k)),
            compare("epsilon_independent80", native, independent_expansion(method, b, r, k)),
        ]
    if family == "effective_beta":
        d, h, r = 0.1, 0.1 * case["ratio"], case["pressure_ratio"]
        b = meter_beta(d, h, "wedge meter")
        native = meter_expansibility(b, r, 1, method="nozzle")
        return [
            compare(
                "override_source",
                native,
                lambda: reference.nozzle_expansibility(d, h, 2e5, 2e5 * r, 1, beta=b),
            ),
            compare("override_independent80", native, independent_expansion("nozzle", b, r, 1)),
        ]
    if family == "conversion":
        b, c = case["beta"], case["C"]
        k = meter_loss_coefficient(b, c)
        return [
            compare("K", k, reference.discharge_coefficient_to_K(0.1, 0.1 * b, c)),
            compare(
                "C",
                meter_coefficient_from_loss(b, k),
                reference.K_to_discharge_coefficient(0.1, 0.1 * b, k),
            ),
            compare(
                "flow_factor",
                meter_flow_coefficient(b, c),
                reference.flow_coefficient(0.1, 0.1 * b, c),
            ),
            compare(
                "approach_factor",
                meter_flow_coefficient(b),
                reference.velocity_of_approach_factor(0.1, 0.1 * b),
            ),
        ]
    if family == "inverse":
        spec = spec_for(case["meter"], 0.1, 0.5, tables)
        fluid = FluidState(1000, 0.001)
        solved = solve_differential_meter(
            spec,
            fluid,
            upstream_pressure_Pa=2e5,
            downstream_pressure_Pa=1.8e5,
            heat_capacity_ratio=case["k"],
        )
        mass = solved.mass_flow_kg_s
        expected = reference.differential_pressure_meter_solver(
            0.1,
            1000,
            0.001,
            k=case["k"],
            D2=0.05,
            P1=2e5,
            P2=1.8e5,
            meter_type=spec.meter_type,
            taps=spec.taps,
            tap_position=spec.tap_position,
            C_specified=spec.specified_coefficient,
            epsilon_specified=1 if case["k"] is None else None,
        )
        common = {"mass_flow_kg_s": mass, "heat_capacity_ratio": case["k"]}
        lower = solve_differential_meter(spec, fluid, upstream_pressure_Pa=2e5, **common)
        upper = solve_differential_meter(spec, fluid, downstream_pressure_Pa=1.8e5, **common)
        geometry = solve_differential_meter(
            spec,
            fluid,
            upstream_pressure_Pa=2e5,
            downstream_pressure_Pa=1.8e5,
            solve_secondary_dimension=True,
            **common,
        )
        return [
            compare("mass", mass, expected, rtol=2e-8),
            compare("P1", upper.upstream_pressure_Pa, 2e5, rtol=2e-8),
            compare("P2", lower.downstream_pressure_Pa, 1.8e5, rtol=2e-8),
            compare("D2", geometry.spec.secondary_dimension_m, 0.05, rtol=2e-8),
        ]
    if family in {"grid", "interpolation"}:
        b, re, name = case["beta"], case["Re"], case["grid"]
        native = tables[name].evaluate(b, re)
        if family == "grid":
            return [compare("source_node", native, case["coefficient"])]
        if name in {"smooth", "sharp", "orifice"}:
            d2 = 0.1 * b
        elif name == "cone":
            d2 = 0.1 * sqrt(1 - b * b)
        else:
            # Independent source geometry inverse only for fixture coordinates.
            from scipy.optimize import brentq

            d2 = brentq(
                lambda h: reference.diameter_ratio_wedge_meter(0.1, h) - b, 1e-12, 0.1 - 1e-12
            )
        c, _ = reference.differential_pressure_meter_C_epsilon(
            0.1, d2, re * pi * 0.1 * 0.001 / 4, 2e5, 1.8e5, 1000, 0.001, 1.3, GRID_NAMES[name]
        )
        return [compare("interpolated_C", native, c)]
    if family == "recovery":
        return [
            compare(
                "recovery_node", tables["recovery"].evaluate(case["beta"], case["D"]), case["value"]
            )
        ]
    common = {
        "pipe_diameter_m": 0.1,
        "throat_diameter_m": 0.1 * case["beta"],
        "gas_mass_flow_kg_s": case["mg"],
        "liquid_mass_flow_kg_s": case["ml"],
        "gas_density_kg_m3": case["rhog"],
        "liquid_density_kg_m3": 1000,
        "pressure_difference_Pa": 1e4,
        "liquid_factor": case["H"],
    }
    c, loss = wet_venturi_correction(**common)
    expected_c = reference.C_Reader_Harris_Gallagher_wet_venturi_tube(
        case["mg"], case["ml"], case["rhog"], 1000, 0.1, 0.1 * case["beta"], case["H"]
    )
    expected_loss = reference.dP_Reader_Harris_Gallagher_wet_venturi_tube(
        0.1, 0.1 * case["beta"], 2e5, 1.9e5, case["ml"], case["mg"], 1000, case["rhog"], case["H"]
    )
    return [compare("wet_C", c, expected_c), compare("wet_loss", loss, expected_loss)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve attempts")
    cases, records = list(inputs()), []
    counts = {
        "formula": 4,
        "Miller": 1,
        "Miller_wedge": 1,
        "expansion": 2,
        "effective_beta": 2,
        "conversion": 4,
        "loss": 1,
        "inverse": 4,
        "grid": 1,
        "interpolation": 1,
        "recovery": 1,
        "wet": 2,
    }
    assert len(cases) == 970, len(cases)
    assert sum(counts[c["family"]] for c in cases) == 1705
    tables = load_tables()
    started = time.monotonic()
    for index, case in enumerate(cases, 1):
        try:
            comparisons = evaluate(case, tables)
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
        if index % 50 == 0 or index == len(cases):
            elapsed = max(1e-6, time.monotonic() - started)
            print(
                f"meters {index}/{len(cases)} {index / elapsed:.1f} inputs/s "
                f"ETA {(len(cases) - index) * elapsed / index:.1f}s",
                flush=True,
            )
    comparisons = [item for record in records for item in record.get("comparisons", [])]
    summary = {
        "reference_version": fluids.__version__,
        "inputs": 970,
        "expected_comparisons": 1705,
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
        "elapsed_s": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"records", "failures"}}))


if __name__ == "__main__":
    main()
