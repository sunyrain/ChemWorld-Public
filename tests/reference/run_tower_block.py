"""Fixed 1084 input packed-tower and demister development block."""

import argparse
import json
import time
from itertools import product
from math import isclose, pi, sqrt
from pathlib import Path

import fluids
import fluids.packed_tower as reference

from chemworld.physchem.packed_towers import (
    TowerPackingSpec,
    demister_dry_pressure_drop,
    demister_removal_efficiency,
    demister_wet_pressure_drop,
    mesh_specific_area,
    mesh_voidage_from_mass,
    robbins_tower_pressure_drop,
    stichlmair_flooding_velocity,
    stichlmair_tower_pressure,
)
from chemworld.physchem.transport import FluidState

COEFFICIENTS = ((32, 7, 1), (120, 5, 0.5))


def inputs():
    for velocity, rho, mu, voidage, area, height, coefficients in product(
        (0.1, 0.4, 1),
        (1.2, 5),
        (1.8e-5, 5e-5),
        (0.68, 0.9),
        (100, 260),
        (0.5, 2),
        COEFFICIENTS,
    ):
        yield {
            "family": "dry_tower",
            "velocity": velocity,
            "rho": rho,
            "mu": mu,
            "voidage": voidage,
            "area": area,
            "height": height,
            "coefficients": coefficients,
        }
    for velocity, liquid_velocity, rho, voidage, height in product(
        (0.05, 0.1, 0.2),
        (0.001, 0.003, 0.005),
        (1.2, 5),
        (0.68, 0.9),
        (0.5, 2),
    ):
        yield {
            "family": "wet_tower",
            "velocity": velocity,
            "liquid_velocity": liquid_velocity,
            "rho": rho,
            "mu": 5e-5,
            "voidage": voidage,
            "height": height,
            "area": 260,
            "coefficients": COEFFICIENTS[0],
        }
    for liquid_velocity, rho, voidage, coefficients in product(
        (0.001, 0.003, 0.005),
        (1.2, 5),
        (0.68, 0.9),
        COEFFICIENTS,
    ):
        yield {
            "family": "flood",
            "liquid_velocity": liquid_velocity,
            "rho": rho,
            "mu": 5e-5,
            "voidage": voidage,
            "height": 1,
            "area": 260,
            "coefficients": coefficients,
        }
    for liquid_flux, gas_flux, rhol, rhog, mul, height, factor in product(
        (2, 12.2, 20),
        (0.5, 2.03, 4),
        (750, 1000),
        (1.1853, 5),
        (0.001, 0.01),
        (0.5, 2),
        (24, 60),
    ):
        yield {
            "family": "robbins",
            "liquid_flux": liquid_flux,
            "gas_flux": gas_flux,
            "rhol": rhol,
            "rhog": rhog,
            "mul": mul,
            "height": height,
            "factor": factor,
        }
    for literature, area, voidage, velocity, rho, mu, depth in product(
        (False, True),
        (250, 500, 1000),
        (0.974, 0.983),
        (0.98, 1.2, 7.5),
        (1.2, 10),
        (1.8e-5, 3e-5),
        (0.1, 0.2, 1),
    ):
        yield {
            "family": "dry_demister",
            "literature": literature,
            "area": area,
            "voidage": voidage,
            "velocity": velocity,
            "rho": rho,
            "mu": mu,
            "depth": depth,
        }
    for velocity, voidage, wire, depth in product(
        (0.98, 3, 7.5), (0.974, 0.983), (0.0002, 0.00032), (0.1, 0.2)
    ):
        yield {
            "family": "wet_demister",
            "velocity": velocity,
            "voidage": voidage,
            "wire": wire,
            "depth": depth,
        }
    for velocity, voidage, wire, drop in product(
        (0.98, 3, 7.5), (0.974, 0.983), (0.0002, 0.00032), (0.001, 0.003, 0.005)
    ):
        yield {
            "family": "efficiency",
            "velocity": velocity,
            "voidage": voidage,
            "wire": wire,
            "drop": drop,
        }
    for diameter, height, voidage, density in product(
        (0.1, 1), (0.1, 1), (0.5, 0.98), (2500, 8000)
    ):
        yield {
            "family": "geometry",
            "diameter": diameter,
            "height": height,
            "voidage": voidage,
            "density": density,
        }


def wet_residuals(case, result):
    voidage = case["voidage"]
    diameter = 6 * (1 - voidage) / case["area"]
    reynolds = result.gas_superficial_velocity_m_s * case["rho"] * diameter / case["mu"]
    c1, c2, c3 = case["coefficients"]
    factor = c1 / reynolds + c2 / sqrt(reynolds) + c3
    exponent = (2 + (-c1 / reynolds - c2 / (2 * sqrt(reynolds))) / factor) / 3
    h0 = 0.555 * (case["liquid_velocity"] ** 2 * case["area"] / (9.80665 * voidage**4.65)) ** (
        1 / 3
    )
    y = result.pressure_drop_Pa / (case["height"] * 1200 * 9.80665)
    holdup = h0 * (1 + 20 * y * y)
    dry = 0.75 * factor * (1 - voidage) * case["rho"] * result.gas_superficial_velocity_m_s**2
    dry /= diameter * voidage**4.65 * 1200 * 9.80665
    multiplier = ((1 - voidage + holdup) / (1 - voidage)) ** exponent * (
        voidage / (voidage - holdup)
    ) ** 4.65
    balance = y / (dry * multiplier) - 1
    tangent = 1 - 40 * h0 * y * y * (exponent / (1 - voidage + holdup) + 4.65 / (voidage - holdup))
    return balance, tangent


def evaluate(c):
    family = c["family"]
    if family in {"dry_tower", "wet_tower", "flood"}:
        packing = TowerPackingSpec(
            c["voidage"], c["area"], c["coefficients"], "Fixed authored qualification input"
        )
        gas, liquid = FluidState(c["rho"], c["mu"]), FluidState(1200, 0.001)
        if family == "flood":
            actual = stichlmair_flooding_velocity(
                packing, gas, liquid, liquid_velocity_m_s=c["liquid_velocity"], height_m=c["height"]
            )
            # Compute all independent criteria even if the reference solver fails.
            balance, tangent = wet_residuals(c, actual)
            yield "balance", balance, 0.0, 2e-9
            yield "fold_tangent", tangent, 0.0, 2e-9
            expected = reference.Stichlmair_flood(
                c["liquid_velocity"],
                c["rho"],
                1200,
                c["mu"],
                c["voidage"],
                c["area"],
                *c["coefficients"],
                c["height"],
            )
            yield "gas_velocity", actual.gas_superficial_velocity_m_s, expected, 1e-9
        elif family == "wet_tower":
            actual = stichlmair_tower_pressure(
                packing,
                gas,
                gas_velocity_m_s=c["velocity"],
                liquid=liquid,
                liquid_velocity_m_s=c["liquid_velocity"],
                height_m=c["height"],
            )
            yield "balance", wet_residuals(c, actual)[0], 0.0, 2e-9
            expected = reference.Stichlmair_wet(
                c["velocity"],
                c["liquid_velocity"],
                c["rho"],
                1200,
                c["mu"],
                c["voidage"],
                c["area"],
                *c["coefficients"],
                c["height"],
            )
            yield "pressure", actual.pressure_drop_Pa, expected, 1e-9
        else:
            actual = stichlmair_tower_pressure(
                packing, gas, gas_velocity_m_s=c["velocity"], height_m=c["height"]
            )
            expected = reference.Stichlmair_dry(
                c["velocity"],
                c["rho"],
                c["mu"],
                c["voidage"],
                c["area"],
                *c["coefficients"],
                c["height"],
            )
            yield "pressure", actual.pressure_drop_Pa, expected, 1e-9
    elif family == "robbins":
        actual = robbins_tower_pressure_drop(
            FluidState(c["rhol"], c["mul"]),
            FluidState(c["rhog"], 1.8e-5),
            liquid_mass_flux_kg_m2_s=c["liquid_flux"],
            gas_mass_flux_kg_m2_s=c["gas_flux"],
            packing_factor_per_ft=c["factor"],
            height_m=c["height"],
        )
        expected = reference.Robbins(
            c["liquid_flux"],
            c["gas_flux"],
            c["rhol"],
            c["rhog"],
            c["mul"],
            c["height"],
            c["factor"],
        )
        yield "pressure", actual, expected, 1e-9
    elif family == "dry_demister":
        actual = demister_dry_pressure_drop(
            FluidState(c["rho"], c["mu"]),
            superficial_velocity_m_s=c["velocity"],
            voidage=c["voidage"],
            specific_area_m_inv=c["area"],
            depth_m=c["depth"],
            include_literature_fit=c["literature"],
        )
        method = (
            reference.dP_demister_dry_Setekleiv_Svendsen_lit
            if c["literature"]
            else reference.dP_demister_dry_Setekleiv_Svendsen
        )
        yield (
            "pressure",
            actual,
            method(c["area"], c["voidage"], c["velocity"], c["rho"], c["mu"], c["depth"]),
            1e-9,
        )
    elif family == "wet_demister":
        actual = demister_wet_pressure_drop(
            superficial_velocity_m_s=c["velocity"],
            voidage=c["voidage"],
            wire_diameter_m=c["wire"],
            depth_m=c["depth"],
        )
        expected = reference.dP_demister_wet_ElDessouky(
            c["velocity"], c["voidage"], c["wire"], c["depth"]
        )
        yield "pressure", actual, expected, 1e-9
    elif family == "efficiency":
        actual = demister_removal_efficiency(
            superficial_velocity_m_s=c["velocity"],
            voidage=c["voidage"],
            wire_diameter_m=c["wire"],
            droplet_diameter_m=c["drop"],
            endpoint_policy="clip",
        )
        expected = reference.separation_demister_ElDessouky(
            c["velocity"], c["voidage"], c["wire"], c["drop"]
        )
        yield "efficiency", actual.removal_fraction, expected, 1e-9
    else:
        mass = (1 - c["voidage"]) * c["density"] * pi * c["diameter"] ** 2 * c["height"] / 4
        actual = mesh_voidage_from_mass(mass, c["density"], c["diameter"], c["height"])
        expected = reference.voidage_experimental(mass, c["density"], c["diameter"], c["height"])
        yield "voidage", actual, expected, 1e-9
        yield (
            "specific_area",
            mesh_specific_area(actual, 0.0003),
            reference.specific_area_mesh(expected, 0.0003),
            1e-9,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = list(inputs())
    assert len(cases) == 1084, len(cases)
    start = time.perf_counter()
    rows, failures, count, passed = [], [], 0, 0
    for index, case in enumerate(cases, 1):
        row = {"input": case, "comparisons": []}
        try:
            for label, actual, expected, atol in evaluate(case):
                ok = isclose(actual, expected, rel_tol=2e-8, abs_tol=atol)
                comparison = {
                    "quantity": label,
                    "native": actual,
                    "reference": expected,
                    "rtol": 2e-8,
                    "atol": atol,
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
        rows.append(row)
        if index % 200 == 0:
            elapsed = time.perf_counter() - start
            print(
                f"tower {index}/1084, {index / elapsed:.1f} inputs/s, "
                f"ETA {(1084 - index) * elapsed / index:.2f}s",
                flush=True,
            )
    summary = {
        "mode": "development",
        "reference_version": fluids.__version__,
        "expected_inputs": 1084,
        "completed_inputs": len(rows),
        "expected_comparisons": 1220,
        "completed_comparisons": count,
        "passed_comparisons": passed,
        "failures": failures,
        "elapsed_s": time.perf_counter() - start,
    }
    (args.output / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}))
    print(f"failures={len(failures)}", flush=True)
    return bool(failures or count != 1220)


if __name__ == "__main__":
    raise SystemExit(main())
