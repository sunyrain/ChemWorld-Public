"""Fixed MSIS scientific matrix with every exception and difference retained."""

import argparse
import json
import time
from itertools import product
from math import isclose
from pathlib import Path

from fluids.nrlmsise00 import ap_array, gtd7, nrlmsise_flags, nrlmsise_input, nrlmsise_output
from msis_source_parameters import source_parameters

from chemworld.physchem.msis_atmosphere import MSISAtmosphere, MSISParameters


def run():
    started = time.perf_counter()
    parameters = source_parameters()
    restored = MSISParameters.from_dict(json.loads(json.dumps(parameters.to_dict())))
    model = MSISAtmosphere(restored)
    failures = []
    checks = passed = inputs = 0
    exceptions = []

    def compare(case, name, native, reference, exact=False):
        nonlocal checks, passed
        checks += 1
        success = (
            native == reference
            if exact
            else isclose(native, reference, rel_tol=1e-8, abs_tol=1e-35)
        )
        if success:
            passed += 1
        else:
            failures.append(
                {"input": case, "property": name, "native": native, "reference": reference}
            )

    for row, old in zip(restored.table.rows, parameters.table.rows, strict=True):
        inputs += 1
        compare(
            [row.group_id, row.row_id],
            "coefficient",
            row.values["value"].value,
            old.values["value"].value,
            True,
        )
    states = product(
        (
            0,
            10,
            20,
            32.5,
            45,
            55,
            62.5,
            70,
            72.5,
            80,
            90,
            100,
            110,
            120,
            150,
            200,
            300,
            450,
            700,
            1000,
        ),
        ((-80, -150), (-30, 0), (35, 110), (85, 179)),
        ((0, 0), (80, 21600), (172, 43200), (355, 64800)),
        ((150, 150, 0, None), (70, 80, 4, None), (210, 180, 20, (20, 30, 25, 18, 12, 9, 6))),
    )
    for altitude, (latitude, longitude), (day, seconds), (flux, mean, ap, history) in states:
        inputs += 1
        case = [altitude, latitude, longitude, day, seconds, flux, mean, ap, history]
        try:
            native = model.evaluate(
                altitude * 1000,
                latitude_degrees=latitude,
                longitude_degrees=longitude,
                day_of_year=day,
                seconds_utc=seconds,
                solar_flux=flux,
                mean_solar_flux=mean,
                daily_ap=ap,
                ap_history=history,
            )
            flags = nrlmsise_flags()
            flags.switches = [0] + [1] * 23
            reference_input = nrlmsise_input(
                doy=day,
                sec=seconds,
                alt=altitude,
                g_lat=latitude,
                g_long=longitude,
                lst=seconds / 3600 + longitude / 15,
                f107A=mean,
                f107=flux,
                ap=ap,
            )
            if history is not None:
                flags.switches[9] = -1
                reference_input.ap_a = ap_array()
                reference_input.ap_a.a = list(history)
            output = nrlmsise_output()
            gtd7(reference_input, flags, output)
            numbers = [output.d[i] * 1e6 for i in (0, 1, 2, 3, 4, 6, 7, 8)]
            total = sum(numbers[:7])
            rho = sum(
                n * m
                for n, m in zip(
                    numbers,
                    (4.002602, 15.9994, 28.0134, 31.9988, 39.948, 1.00794, 14.0067),
                    strict=False,
                )
            ) / (1000 * 6.02214076e23)
            pairs = [
                ("T", native.temperature_K, output.t[1]),
                ("Texosphere", native.exospheric_temperature_K, output.t[0]),
                ("rho", native.density_kg_m3, output.d[5] * 1000),
                ("rho_composition", native.composition_density_kg_m3, rho),
                ("P", native.pressure_Pa, total * 1.380649e-23 * output.t[1]),
            ]
            pairs += [
                (f"number{i}", a, b)
                for i, (a, b) in enumerate(zip(native.number_densities_m3, numbers, strict=True))
            ]
            pairs += [
                (f"fraction{i}", a, b / total)
                for i, (a, b) in enumerate(zip(native.mole_fractions, numbers[:7], strict=True))
            ]
            pairs += [
                ("fraction_sum", sum(native.mole_fractions), 1),
                (
                    "ideal_pressure",
                    native.pressure_Pa,
                    sum(native.number_densities_m3[:7]) * 1.380649e-23 * native.temperature_K,
                ),
                (
                    "integer_mass",
                    native.density_kg_m3,
                    1.66e-27
                    * sum(
                        n * m
                        for n, m in zip(
                            native.number_densities_m3, (4, 16, 28, 32, 40, 1, 14), strict=False
                        )
                    ),
                ),
            ]
            assert len(pairs) == 23
            for name, actual, expected in pairs:
                compare(case, name, actual, expected)
        except Exception as error:
            exceptions.append({"input": case, "error": repr(error)})
        if (inputs - 3200) % 120 == 0:
            print(
                f"MSIS {inputs - 3200}/960states elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return {
        "mode": "development",
        "inputs": inputs,
        "planned_comparisons": 25280,
        "comparisons": checks,
        "passed": passed,
        "failures": failures,
        "exceptions": exceptions,
        "elapsed_s": time.perf_counter() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print({k: v for k, v in result.items() if k not in ("failures", "exceptions")})
    print("differences", len(result["failures"]), "exceptions", len(result["exceptions"]))
