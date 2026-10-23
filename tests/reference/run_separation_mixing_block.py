"""Prospective 400-input separation/mixing development comparison."""

import argparse
import json
import time
from itertools import product
from math import isclose, pi, sqrt
from pathlib import Path

import fluids
import fluids.mixing as mixing
import fluids.separator as separator
import mpmath as mp
from fluids.friction import friction_factor
from scipy.interpolate import splev

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.mixing_engineering import (
    agitator_mixing_time,
    ribbon_mixing_time,
    ribbon_power_reynolds,
    static_mixer_cov,
    static_mixer_loss_coefficient,
    static_mixer_performance,
    tee_mixing_cov,
)
from chemworld.physchem.separator_capacity import (
    PSI_PA,
    SeparatorCapacityCurve,
    separator_demister_capacity,
    separator_gas_capacity,
    separator_theoretical_capacity,
    separator_watkins_capacity,
)
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.transport import FluidState


def chart():
    tck = separator.tck_Watkins
    knots = sorted(set(map(float, tck[0])))
    coordinates = [
        knots[0],
        (knots[0] + knots[1]) / 2,
        *knots[1:-1],
        (knots[-2] + knots[-1]) / 2,
        knots[-1],
    ]
    assert len(coordinates) == 9
    source = InputSource(
        DataSourceProvenance(
            "local-watkins",
            0,
            "Installed fluids 1.3.1 Watkins chart",
            "1.3.1",
            "Local reference data, not bundled",
        ),
        "literature",
    )
    rows = tuple(
        ScientificRow(
            str(i),
            "local-watkins",
            "Watkins",
            str(i),
            "parameters",
            {
                "ln_factor": InputValue(x, "dimensionless", None, "Source spline abscissa"),
                "ln_K": InputValue(
                    float(splev(x, tck)),
                    "dimensionless",
                    None,
                    "Source ln(K/(1 ft/s)); not experimental uncertainty",
                ),
            },
        )
        for i, x in enumerate(coordinates)
    )
    columns = tuple(
        QuantityColumn(
            k, "dimensionless", "parameter", (-100, 100), "Explicit logged chart coordinate"
        )
        for k in ("ln_factor", "ln_K")
    )
    return SeparatorCapacityCurve(
        TabulatedScalar(
            ScientificTable("watkins", "parameters", columns, (source,), rows),
            (("ln_factor", "ln_factor", "dimensionless"),),
            "ln_K",
            "dimensionless",
        )
    )


def inputs():
    for method, factor, rhol, rhog, horizontal in product(
        ("spline", "blackwell", "branan"),
        (0.0061, 0.01, 0.1, 1, 5.3),
        (750, 1000),
        (1.2, 10),
        (False, True),
    ):
        yield {
            "family": "watkins",
            "method": method,
            "factor": factor,
            "rhol": rhol,
            "rhog": rhog,
            "horizontal": horizontal,
        }
    for pressure, horizontal in product(
        (0.5, 1, 2, 14.99, 15, 15.01, 39.99, 40, 40.01, 100, 975, 5500, 6000),
        (False, True),
    ):
        yield {"family": "york", "pressure": pressure, "horizontal": horizontal}
    for diameter, drag, gravity in product((0.0001, 0.0005, 0.001), (0.1, 1, 4), (9.80665, 1.62)):
        yield {"family": "theory", "diameter": diameter, "drag": drag, "gravity": gravity}
    for rhol, rhog, factor in product((500, 1000), (1.2, 10), (0.05, 0.1, 0.2)):
        yield {"family": "capacity", "rhol": rhol, "rhog": rhog, "factor": factor}
    for coefficient, ratio, length in product((0.5, 0.8, 0.95), (0.01, 0.2, 1), (0, 1, 5, 20)):
        yield {"family": "static_cov", "coefficient": coefficient, "ratio": ratio, "length": length}
    for factor, length, diameter, friction in product(
        (10, 100, 250), (0, 0.5, 2), (0.01, 0.1), (0.01, 0.04)
    ):
        yield {
            "family": "static_K",
            "factor": factor,
            "length": length,
            "diameter": diameter,
            "friction": friction,
        }
    for diameter, height, blades, pitch, width in product(
        (0.5, 1.9), (0.5, 1), (1, 2), (0.5, 1), (0.05, 0.1)
    ):
        yield {
            "family": "ribbon",
            "diameter": diameter,
            "height": height * diameter,
            "blades": blades,
            "pitch": pitch * diameter,
            "width": width * diameter,
            "tank": diameter / 0.95,
        }
    for speed, power, viscosity, homogeneity in product(
        (0.1, 1, 5), (10, 1000), (0.001, 0.5), (0.9, 0.95, 0.995)
    ):
        yield {
            "family": "agitator",
            "speed": speed,
            "power": power,
            "viscosity": viscosity,
            "homogeneity": homogeneity,
        }
    for criterion, viscosity in product((6360, 6380), (0.001, 0.5)):
        yield {
            "family": "agitator",
            "speed": 1,
            "power": 1000 * (criterion * viscosity / 1000) ** 3,
            "viscosity": viscosity,
            "homogeneity": 0.95,
        }
    for jets, parameter, length in product((1, 2, 3, 4), (0.2, 0.5, 1, 3), (1, 3, 5)):
        yield {"family": "tee", "jets": jets, "parameter": parameter, "length": length}
    for jets, ratio in product((1, 2, 3, 4), (0.9999, 1.0001)):
        yield {
            "family": "tee",
            "jets": jets,
            "parameter": (0.7, 0.8, 0.8, 2)[jets - 1] * ratio,
            "length": 5,
        }
    for density, viscosity, flow, ratio in product(
        (1.2, 750, 1000), (1.8e-5, 0.001), (0.001, 0.01), (0.1, 0.5)
    ):
        yield {
            "family": "consumer",
            "density": density,
            "viscosity": viscosity,
            "flow": flow,
            "ratio": ratio,
        }


def evaluate(c, source_chart):
    family = c["family"]
    if family == "watkins":
        quality = sqrt(c["rhog"] / c["rhol"]) / (c["factor"] + sqrt(c["rhog"] / c["rhol"]))
        actual = separator_watkins_capacity(
            FluidState(c["rhol"], 0.001),
            FluidState(c["rhog"], 1.8e-5),
            vapor_mass_fraction=quality,
            method=c["method"],
            horizontal=c["horizontal"],
            chart=source_chart if c["method"] == "spline" else None,
        )
        yield (
            "K",
            actual,
            separator.K_separator_Watkins(
                quality, c["rhol"], c["rhog"], c["horizontal"], c["method"]
            ),
            False,
        )
    elif family == "york":
        pressure = c["pressure"] * PSI_PA
        yield (
            "K",
            separator_demister_capacity(
                pressure, horizontal=c["horizontal"], endpoint_policy="clip"
            ),
            separator.K_separator_demister_York(pressure, c["horizontal"]),
            False,
        )
    elif family == "theory":
        native = separator_theoretical_capacity(c["diameter"], c["drag"], gravity_m_s2=c["gravity"])
        expected = separator.K_Souders_Brown_theoretical(c["diameter"], c["drag"], c["gravity"])
        yield "K", native, expected, False
        yield (
            "velocity",
            separator_gas_capacity(
                FluidState(1000, 0.001), FluidState(1.2, 1e-5), capacity_factor_m_s=native
            ).maximum_gas_velocity_m_s,
            separator.v_Souders_Brown(expected, 1000, 1.2),
            False,
        )
    elif family == "capacity":
        native = separator_gas_capacity(
            FluidState(c["rhol"], 0.001),
            FluidState(c["rhog"], 1e-5),
            capacity_factor_m_s=c["factor"],
            gas_mass_flow_kg_s=1,
            design_fraction=0.8,
        )
        expected = separator.v_Souders_Brown(c["factor"], c["rhol"], c["rhog"])
        yield "velocity", native.maximum_gas_velocity_m_s, expected, False
        yield "gas_flow_area", native.required_gas_area_m2, 1 / (c["rhog"] * 0.8 * expected), False
    elif family == "static_cov":
        yield (
            "CoV",
            static_mixer_cov(c["coefficient"], 1, c["ratio"], c["length"]),
            mixing.COV_motionless_mixer(c["coefficient"], 1, c["ratio"], c["length"]),
            False,
        )
    elif family == "static_K":
        args = [c[k] for k in ("factor", "length", "diameter", "friction")]
        yield "K", static_mixer_loss_coefficient(*args), mixing.K_motionless_mixer(*args), False
    elif family == "ribbon":
        actual = ribbon_power_reynolds(
            impeller_diameter_m=c["diameter"],
            ribbon_height_m=c["height"],
            blades=c["blades"],
            pitch_m=c["pitch"],
            blade_width_m=c["width"],
            tank_diameter_m=c["tank"],
        )
        expected = mixing.Kp_helical_ribbon_Rieger(
            c["diameter"], c["height"], c["blades"], c["pitch"], c["width"], c["tank"]
        )
        yield "NpRe", actual, expected, False
        yield (
            "time",
            ribbon_mixing_time(actual, 1.5),
            mixing.time_helical_ribbon_Grenville(expected, 1.5),
            False,
        )
    elif family == "agitator":
        kwargs = {
            "rotation_frequency_Hz": c["speed"],
            "fluid_power_W": c["power"],
            "tank_diameter_m": 2,
            "liquid_height_m": 1.5,
            "impeller_diameter_m": 1,
            "homogeneity": c["homogeneity"],
        }
        fluid = FluidState(1000, c["viscosity"])
        si = agitator_mixing_time(fluid, **kwargs)
        compatible = agitator_mixing_time(fluid, **kwargs, power_convention="reference_g")
        expected = mixing.agitator_time_homogeneous(
            c["speed"], c["power"], 2, 1.5, c["viscosity"], 1000, 1, c["homogeneity"]
        )
        yield "reference_g_time", compatible.time_s, expected, False
        yield "SI_vs_unmodified_source", si.time_s, expected, True
        with mp.workdps(80):
            speed, power, viscosity, fraction = (
                mp.mpf(c[k]) for k in ("speed", "power", "viscosity", "homogeneity")
            )
            npower = power / (1000 * speed**3)
            criterion = mp.root(npower, 3) * 1000 * speed / viscosity
            fourier = mp.mpf("5.2") / criterion if criterion >= 6370 else (183 / criterion) ** 2
            time_s = (
                1000
                * mp.sqrt(12)
                / viscosity
                * fourier
                * mp.log(1 - fraction)
                / mp.log(mp.mpf(".05"))
            )
            yield "SI_independent_80_digit_time", si.time_s, float(time_s), False
            yield "SI_power_number", si.power_number, float(npower), False
    elif family == "tee":
        diameter = c["jets"] * 0.1 * 0.5 / sqrt(c["parameter"])
        actual = tee_mixing_cov(
            main_flow_m3_s=1,
            injected_flow_m3_s=0.1,
            main_diameter_m=0.5,
            jet_diameter_m=diameter,
            jets=c["jets"],
            length_diameters=c["length"],
        )
        yield (
            "CoV",
            actual.concentration_cov,
            mixing.size_tee(1, 0.1, 0.5, diameter, c["jets"], c["length"]),
            False,
        )
    else:
        main_flow, injection = c["flow"], c["flow"] * c["ratio"]
        actual = static_mixer_performance(
            FluidState(c["density"], c["viscosity"]),
            main_flow_m3_s=main_flow,
            injected_flow_m3_s=injection,
            diameter_m=0.1,
            length_m=1,
            mixing_coefficient=0.8,
            manufacturer_loss_factor=100,
        )
        velocity = (main_flow + injection) / (pi * 0.1**2 / 4)
        reynolds = c["density"] * velocity * 0.1 / c["viscosity"]
        f = friction_factor(reynolds, Method="Colebrook")
        k = mixing.K_motionless_mixer(100, 1, 0.1, f)
        pressure = k * c["density"] * velocity**2 / 2
        yield (
            "CoV",
            actual.concentration_cov,
            mixing.COV_motionless_mixer(0.8, main_flow, injection, 10),
            False,
        )
        yield "K", actual.loss_coefficient, k, False
        yield "pressure", actual.pressure_drop_Pa, pressure, False
        yield "hydraulic_power", actual.hydraulic_power_W, pressure * (main_flow + injection), False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = list(inputs())
    assert len(cases) == 400
    start = time.perf_counter()
    rows, failures, comparisons, passed, unexpected = [], [], 0, 0, 0
    source_chart = chart()
    for index, case in enumerate(cases, 1):
        row = {"input": case, "comparisons": []}
        try:
            for label, native, expected, known in evaluate(case, source_chart):
                ok = isclose(native, expected, rel_tol=2e-9, abs_tol=1e-10)
                item = {
                    "quantity": label,
                    "native": native,
                    "reference": expected,
                    "passed": ok,
                    "prospective_source_difference": known,
                }
                comparisons += 1
                passed += ok
                row["comparisons"].append(item)
                if not ok:
                    failures.append({"input": case, **item})
                    unexpected += not known
        except Exception as error:
            row["exception"] = {"type": type(error).__name__, "message": str(error)}
            failures.append({"input": case, **row["exception"]})
            unexpected += 1
        rows.append(row)
        if index % 100 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"separation/mixing {index}/400, {index / elapsed:.1f} inputs/s, "
                f"ETA {(400 - index) * elapsed / index:.2f}s",
                flush=True,
            )
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 400,
        "completed_inputs": len(rows),
        "expected_comparisons": 654,
        "completed_comparisons": comparisons,
        "passed_comparisons": passed,
        "prospective_source_differences": 40,
        "unexpected_failures": unexpected,
        "failures": failures,
        "elapsed_s": time.perf_counter() - start,
    }
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}))
    return bool(unexpected or comparisons != 654)


if __name__ == "__main__":
    raise SystemExit(main())
