"""Fixed575-input/634-comparison block; source data loaded locally, not bundled."""

import argparse
import json
import time
from itertools import product
from math import isclose, pi, sqrt
from pathlib import Path

import fluids
import fluids.friction as reference

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.pipe_roughness import (
    PipeRoughnessCatalog,
    farshad_pipe_roughness,
    smooth_pipe_observation_pairs,
)
from chemworld.physchem.single_phase_pressure import (
    single_phase_acceleration_gradient,
    single_phase_acceleration_pressure_drop,
    single_phase_friction_pressure_drop,
    single_phase_gravity_pressure_drop,
)
from chemworld.physchem.transport import FluidState


def source():
    return InputSource(
        DataSourceProvenance(
            "local-friction-data",
            0,
            "Local fluids1.3.1 HHR/Farshad/Oregon published source data",
            "fluids1.3.1",
            "Original data accessed locally only; not production-distributed",
        ),
        "literature",
    )


def cell(value, unit="dimensionless"):
    return InputValue(float(value), unit, None, "Source supplies no numerical uncertainty")


def load_data():
    rows = []
    for material, statistics in reference.HHR_roughness.items():
        for statistic, value in zip(("min", "max", "average"), statistics, strict=True):
            if value is not None:
                rows.append(
                    ScientificRow(
                        str(len(rows)),
                        "local-friction-data",
                        material,
                        material + ":" + statistic,
                        "parameters",
                        {"roughness": cell(value, "m")},
                    )
                )
    for material, value in reference._roughness.items():
        rows.append(
            ScientificRow(
                str(len(rows)),
                "local-friction-data",
                material,
                material + ":average",
                "parameters",
                {"roughness": cell(value, "m")},
            )
        )
    roughness = ScientificTable(
        "roughness",
        "parameters",
        (QuantityColumn("roughness", "m", "parameter", (0, 10), "Absolute source roughness"),),
        (source(),),
        tuple(rows),
    )
    columns = (
        QuantityColumn("average_roughness", "m", "parameter", (0, 1), "Source mean"),
        QuantityColumn("coefficient", "dimensionless", "parameter", (0, 1), "Inch-normalized A"),
        QuantityColumn("exponent", "dimensionless", "parameter", (-5, 5), "Farshad B"),
    )
    rows = tuple(
        ScientificRow(
            str(i),
            "local-friction-data",
            material,
            material + ":fit",
            "parameters",
            {
                "average_roughness": cell(values[0], "m"),
                "coefficient": cell(values[1]),
                "exponent": cell(values[2]),
            },
        )
        for i, (material, values) in enumerate(reference._Farshad_roughness.items())
    )
    farshad = ScientificTable("farshad", "parameters", columns, (source(),), rows)
    catalog = PipeRoughnessCatalog.from_dict(
        json.loads(json.dumps(PipeRoughnessCatalog(roughness, farshad).to_dict()))
    )
    columns = (
        QuantityColumn("reynolds", "dimensionless", "condition", (0, 1e8), "Oregon measured Re"),
        QuantityColumn(
            "darcy_factor", "dimensionless", "observation", (0, 10), "Oregon measured f_D"
        ),
    )
    rows = tuple(
        ScientificRow(
            str(i),
            "local-friction-data",
            "Oregon",
            str(i),
            "test",
            {"reynolds": cell(reynolds), "darcy_factor": cell(factor)},
        )
        for i, (reynolds, factor) in enumerate(zip(*reference.oregon_smooth_data, strict=True))
    )
    observations = ScientificTable("oregon", "measurements", columns, (source(),), rows)
    observations = ScientificTable.from_dict(json.loads(json.dumps(observations.to_dict())))
    return catalog, smooth_pipe_observation_pairs(observations)


def inputs():
    for material, optimism in product(
        tuple(reference.HHR_roughness) + tuple(reference._roughness), (None, True, False)
    ):
        yield {"family": "catalog", "material": material, "optimism": optimism}
    for material, diameter in product(reference._Farshad_roughness, (None, 0.005, 0.05, 0.5)):
        yield {"family": "Farshad", "material": material, "D": diameter}
    for coefficient, exponent, diameter in product(
        (0.0002, 0.0014, 0.0021), (-1.01, -1, -0.9), (0.005, 0.05, 0.5)
    ):
        yield {"family": "custom", "A": coefficient, "B": exponent, "D": diameter}
    for query in (
        "concrete",
        "condensate pipes",
        "Carbon steel bare",
        "Cr13 bare",
        "galvanized iron",
    ):
        yield {"family": "nearest", "query": query}
    for index in range(59):
        yield {"family": "observation", "index": index}
    for method, mass, diameter in product(
        ("Colebrook", "Haaland", "Blasius"), (0.01, 0.1, 1), (0.01, 0.1)
    ):
        yield {"family": "friction", "method": method, "m": mass, "D": diameter}
    for rho_i, rho_o, di, do in product((100, 1000), (100, 1000), (0.01, 0.1), (0.01, 0.1)):
        yield {"family": "discrete", "rho_i": rho_i, "rho_o": rho_o, "Di": di, "Do": do}
    for density, flux, derivative in product((10, 1000), (50, 500), (-0.001, 0, 0.001)):
        yield {"family": "gradient", "rho": density, "G": flux, "dA": derivative}
    for density, flux, derivative in product((10, 1000), (50, 500), (-0.001, 0.001)):
        yield {"family": "independent", "rho": density, "G": flux, "dA": derivative}
    for density, angle, length in product((10, 1000), (-90, 0, 90), (0, 2)):
        yield {"family": "gravity", "rho": density, "angle": angle, "L": length}


def compare(quantity, native, expected):
    return {
        "quantity": quantity,
        "native": native,
        "reference": expected,
        "relative_tolerance": 2e-9,
        "absolute_tolerance": 1e-10,
        "passed": native == expected
        if isinstance(native, str)
        else isclose(native, expected, rel_tol=2e-9, abs_tol=1e-10),
    }


def evaluate(case, catalog, observations):
    family = case["family"]
    if family in {"catalog", "Farshad"}:
        name, optimism, diameter = case["material"], case.get("optimism"), case.get("D")
        value = catalog.evaluate(name, diameter_m=diameter, optimism=optimism)
        return [
            compare(
                "roughness_m",
                value.roughness_m,
                reference.material_roughness(name, D=diameter, optimism=optimism),
            )
        ]
    if family == "custom":
        a, b, d = case["A"], case["B"], case["D"]
        return [
            compare(
                "roughness_m",
                farshad_pipe_roughness(d, a, b),
                reference.roughness_Farshad(D=d, coeffs=(a, b)),
            )
        ]
    if family == "nearest":
        return [
            compare(
                "suggested_material",
                catalog.nearest_materials(case["query"], count=1)[0],
                reference.nearest_material_roughness(case["query"]),
            )
        ]
    if family == "observation":
        index = case["index"]
        return [
            compare(name, observations[index][axis], reference.oregon_smooth_data[axis][index])
            for axis, name in enumerate(("measured_Re", "measured_f_D"))
        ]
    if family == "friction":
        method, mass, d = case["method"], case["m"], case["D"]
        roughness = 0 if method == "Blasius" else 1e-5
        native = single_phase_friction_pressure_drop(
            FluidState(1000, 0.001),
            mass_flow_kg_s=mass,
            diameter_m=d,
            roughness_m=roughness,
            method=method,
            check_recommended_ranges=False,
        )
        if method == "Blasius":
            # Public Blasius exists but upstream's friction dispatcher omits it.
            flux = mass / (pi * d * d / 4)
            reynolds = flux * d / 0.001
            factor = 64 / reynolds if reynolds < 2040 else reference.Blasius(reynolds)
            expected = factor / d * flux**2 / (2 * 1000)
        else:
            expected = reference.one_phase_dP(
                mass, 1000, 0.001, d, roughness=roughness, Method=method
            )
        return [compare("pressure_Pa", native, expected)]
    if family == "discrete":
        native = single_phase_acceleration_pressure_drop(
            mass_flow_kg_s=0.1,
            inlet_density_kg_m3=case["rho_i"],
            outlet_density_kg_m3=case["rho_o"],
            inlet_diameter_m=case["Di"],
            outlet_diameter_m=case["Do"],
        )
        expected = reference.one_phase_dP_acceleration(
            0.1, case["Do"], case["rho_o"], case["rho_i"], case["Di"]
        )
        return [compare("pressure_Pa", native, expected)]
    if family in {"gradient", "independent"}:
        density, flux, derivative = case["rho"], case["G"], case["dA"]
        volume_derivative = 0 if family == "independent" else -4e-7
        native = single_phase_acceleration_gradient(
            mass_flux_kg_m2_s=flux,
            density_kg_m3=density,
            volume_pressure_derivative_m3_kg_Pa=volume_derivative,
            pressure_drop_gradient_Pa_m=120,
            area_m2=0.01,
            area_gradient_m=derivative,
        )
        if family == "gradient":
            expected = reference.one_phase_dP_dz_acceleration(
                flux * 0.01, sqrt(0.04 / pi), density, volume_derivative, 120, derivative
            )
        else:

            def kinetic(z):
                return density / 2 * (flux * 0.01 / (density * (0.01 + z * derivative))) ** 2

            expected = (kinetic(1e-6) - kinetic(-1e-6)) / 2e-6
        return [compare("gradient_Pa_m", native, expected)]
    native = single_phase_gravity_pressure_drop(
        case["rho"], length_m=case["L"], inclination_degrees=case["angle"]
    )
    return [
        compare(
            "pressure_Pa",
            native,
            reference.one_phase_dP_gravitational(case["angle"], case["rho"], case["L"]),
        )
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite an attempt")
    cases = list(inputs())
    assert len(cases) == 575
    records, failures = [], []
    start = time.perf_counter()
    catalog, observations = load_data()
    for index, case in enumerate(cases, 1):
        record = {"unit": index, "input": case}
        try:
            record["measurements"] = evaluate(case, catalog, observations)
            record["passed"] = all(m["passed"] for m in record["measurements"])
        except Exception as exc:
            record.update(measurements=[], passed=False, error=f"{type(exc).__name__}: {exc}")
        records.append(record)
        if not record["passed"]:
            failures.append(record)
        if index % 50 == 0 or index == 575:
            elapsed = time.perf_counter() - start
            print(
                f"roughness {index}/575 rate={index / max(elapsed, 1e-9):.1f}/s "
                f"ETA={elapsed / index * (575 - index):.2f}s failures={len(failures)}",
                flush=True,
            )
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 575,
        "expected_comparisons": 634,
        "completed_inputs": len(records),
        "passed_inputs": sum(r["passed"] for r in records),
        "completed_comparisons": sum(len(r["measurements"]) for r in records),
        "passed_comparisons": sum(m["passed"] for r in records for m in r["measurements"]),
        "elapsed_s": time.perf_counter() - start,
        "failures": failures,
        "records": records,
    }
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in {"failures", "records"}}))


if __name__ == "__main__":
    main()
