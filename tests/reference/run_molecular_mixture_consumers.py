# ruff: noqa: B023 -- each fixed-unit callback is evaluated before advancing the loop.
"""Fixed source comparison of absolute mixture and finite-inventory consumers."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np
from run_pcsaft_core import models
from run_saftvrmie import teqp_targets

from chemworld.physchem.coupled_system import (
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    HeatFlux,
    MaterialPort,
)
from chemworld.physchem.eos import R_J_PER_MOL_K as R
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_mixture import (
    MolecularCaloricModel,
    MolecularIsochoricThermodynamics,
    MolecularStream,
)
from chemworld.physchem.solver_backend import ODESolverPolicy

NAMES = (
    "pressure_Pa",
    "compressibility_factor",
    "helmholtz_energy_J_mol",
    "internal_energy_J_mol",
    "enthalpy_J_mol",
    "entropy_J_mol_K",
    "cv_J_mol_K",
    "cp_J_mol_K",
    "dP_dT_rho_Pa_K",
    "dP_drho_T_Pa_m3_mol",
    "speed_of_sound_m_s",
    "gibbs_energy_J_mol",
)


def bind(residual):
    return MolecularCaloricModel(
        residual,
        tuple(
            HelmholtzFluid(
                key,
                "",
                0.04 + 0.02 * i,
                R,
                300,
                1000,
                100,
                1000,
                1e9,
                (
                    HelmholtzTerm("IdealGasHelmholtzLead", {"a1": 2 + i, "a2": 3 + i}),
                    HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5 + i}),
                ),
                (),
                "explicit synthetic monatomic caloric reference",
            )
            for i, key in enumerate(residual.component_ids)
        ),
    )


def target(ref, t, rho, x):
    departures = teqp_targets(ref, t, rho, x)
    u0 = np.array([R * ((1.5 + i) * t + (3 + i) * 300) for i in range(len(x))])
    cv0 = R * sum(xi * (1.5 + i) for i, xi in enumerate(x))
    s0 = R * sum(
        xi * ((1.5 + i) - (2 + i) - math.log(rho / 1000) - (1.5 + i) * math.log(300 / t))
        for i, xi in enumerate(x)
    )
    s0 -= R * sum(xi * math.log(xi) for xi in x if xi > 0)
    u = sum(x * u0) + departures["residual_internal_energy_J_mol"]
    s = s0 + departures["residual_entropy_J_mol_K"]
    p = departures["pressure_Pa"]
    h = u + p / rho
    cv = cv0 + departures["residual_cv_J_mol_K"]
    cp = cv0 + R + departures["residual_cp_J_mol_K"]
    mass = sum(xi * (0.04 + 0.02 * i) for i, xi in enumerate(x))
    speed = math.sqrt(
        (
            departures["dP_drho_T_Pa_m3_mol"]
            + t * departures["dP_dT_rho_Pa_K"] ** 2 / (rho * rho * cv)
        )
        / mass
    )
    result = dict(
        zip(
            NAMES,
            (
                p,
                departures["compressibility_factor"],
                u - t * s,
                u,
                h,
                s,
                cv,
                cp,
                departures["dP_dT_rho_Pa_K"],
                departures["dP_drho_T_Pa_m3_mol"],
                speed,
                h - t * s,
            ),
            strict=True,
        )
    )
    partials = (
        u0
        + ref.build_Psir_gradient_autodiff(t, rho * x)
        - t * ref.build_d2PsirdTdrhoi_autodiff(t, rho * x)
    )
    result.update({f"partial_U_{i}": v for i, v in enumerate(partials)})
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    cache = {case: models(case, "Gross-Sadowski-2001") for case in (3, 4, 5, 7, 8)}
    native = {case: bind(values[0]) for case, values in cache.items()}
    started = time.perf_counter()
    count, passed, units, failures = 0, 0, 0, []

    def compare(label, names, actual, expected):
        nonlocal count, passed, units
        values = {}
        for side, callback in (("native", actual), ("reference", expected)):
            try:
                values[side] = callback()
            except Exception as exc:
                values[side] = f"{type(exc).__name__}: {exc}"
        rows = []
        for name in names:
            atol = (
                1e-4
                if name == "pressure_Pa"
                else 2e-5
                if (name.endswith("J_mol") or name.startswith("partial_U") or name == "energy_J")
                else 2e-7
            )
            row = {"unit": label, "quantity": name, "atol": atol, "rtol": 2e-7}
            for side, value in values.items():
                try:
                    if isinstance(value, str):
                        raise ValueError(value)
                    number = float(value[name])
                    if not math.isfinite(number):
                        raise ValueError("nonfinite response")
                    row[side] = number
                except Exception as exc:
                    row[side + "_exception"] = str(exc)
            row["passed"] = (
                "native" in row
                and "reference" in row
                and math.isclose(row["native"], row["reference"], rel_tol=2e-7, abs_tol=atol)
            )
            count += 1
            passed += row["passed"]
            if not row["passed"]:
                failures.append(row)
            rows.append(row)
        with (output / "records.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"unit": label, "comparisons": rows}) + "\n")
        units += 1
        elapsed = time.perf_counter() - started
        print(
            f"Mixture consumers {units}/54; {units / elapsed:.2f}units/s; "
            f"ETA {(54 - units) * elapsed / units:.1f}s; failures {len(failures)}",
            flush=True,
        )

    for case, t, rho in product(cache, (400.0, 600.0, 800.0), (1.0, 500.0, 2000.0)):
        _, ref, x = cache[case]
        model = native[case]

        def actual():
            result = dict(model.state(t, rho, x).__dict__)
            partials = model.partial_inventory_energies(
                t, 1 / rho, dict(zip(model.component_ids, x, strict=True))
            )
            result.update({f"partial_U_{i}": v for i, v in enumerate(partials)})
            return result

        compare(
            f"state-{case}-{t}-{rho}",
            (*NAMES, *(f"partial_U_{i}" for i in range(len(x)))),
            actual,
            lambda: target(ref, t, rho, x),
        )
    for case, (_, ref, x) in cache.items():
        compare(
            f"inverse-{case}",
            ("density",),
            lambda: {
                "density": native[case]
                .state_tp(
                    600,
                    target(ref, 600, 500, x)["pressure_Pa"],
                    x,
                    initial_density_mol_m3=550,
                    density_bounds_mol_m3=(300, 700),
                )
                .molar_density_mol_m3
            },
            lambda: {"density": 500.0},
        )
    model = native[3]
    ids = model.component_ids
    _, ref, feed_x = cache[5]
    for form, mode in product(("ode", "residual"), ("heat", "feed")):

        def advance():
            thermal = MolecularIsochoricThermodynamics(model, 4.0, 5.0)
            start = CoupledState({ids[0]: 2.0, ids[1]: 0.0}, 400.0, 4.0)
            options = {
                "numerical_form": form,
                "policy": ODESolverPolicy(
                    "mixture-consumers", method="BDF", rtol=1e-11, atol=1e-13
                ),
            }
            if mode == "heat":
                options["heat_flux"] = lambda _: HeatFlux(20.0)
            else:
                feed = MolecularStream(model.state(500, 500, feed_x), 0.01)
                ports = {key: f"feed-{i}" for i, key in enumerate(ids)}
                options["ports"] = tuple(
                    MaterialPort(p, key, "source", "vessel", "in") for key, p in ports.items()
                )
                options["processes"] = (CoupledProcess("feed", lambda _: feed.contribution(ports)),)
            final = CoupledSystem(thermal, **options).advance(start, 2.0).final_state
            return {
                "n0": final.amounts_mol[ids[0]],
                "n1": final.amounts_mol[ids[1]],
                "time": final.time_s,
                "energy_J": thermal.internal_energy_j(final) - thermal.internal_energy_j(start),
            }

        compare(
            f"inventory-{form}-{mode}",
            ("n0", "n1", "time", "energy_J"),
            advance,
            lambda: {
                "n0": 2.0 if mode == "heat" else 2.006,
                "n1": 0.0 if mode == "heat" else 0.014,
                "time": 2.0,
                "energy_J": 40.0
                if mode == "heat"
                else 0.02 * target(ref, 500, 500, feed_x)["enthalpy_J_mol"],
            },
        )
    assert units == 54 and count == 669
    summary = {
        "mode": "development",
        "units": units,
        "comparisons": count,
        "passed": passed,
        "failed": len(failures),
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "models": {str(case): model.to_dict() for case, model in native.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
