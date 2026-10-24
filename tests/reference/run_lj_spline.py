# ruff: noqa: B023 -- fixed-unit callbacks run before advancing the loop.
"""Fixed public-control LJ spline comparison against ThermoPack 2.2.3."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

from molecular_reference import thermopack_residual_targets
from thermopack.ljs_bh import ljs_bh
from thermopack.ljs_wca import ljs_uv, ljs_wca

from chemworld.physchem.lj_spline import (
    SPLINE_CUTOFF,
    SPLINE_START,
    WCA_MINIMUM,
    LJSplineBHModel,
    LJSplineUVModel,
    LJSplineWCAModel,
)
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL

BASE = ("synthetic", 3.42e-10, 124.0, (50.0, 2000.0), "Fixed LJ spline comparison")
SCALE = AVOGADRO_PER_MOL * BASE[1] ** 3
NAMES = (
    "pressure_Pa",
    "compressibility_factor",
    "residual_helmholtz_J_mol",
    "residual_internal_energy_J_mol",
    "residual_enthalpy_J_mol",
    "residual_entropy_J_mol_K",
    "residual_cv_J_mol_K",
    "residual_cp_J_mol_K",
    "dP_dT_rho_Pa_K",
    "dP_drho_T_Pa_m3_mol",
    "mu_J_mol",
)


def declarations():
    result = []
    for cls, fields in (
        (
            LJSplineBHModel,
            (
                "use_lafitte_a3",
                "enable_chi_correction",
                "enable_hs",
                "enable_a1",
                "enable_a2",
                "enable_a3",
            ),
        ),
        (
            LJSplineWCAModel,
            ("enable_cavity", "enable_hs", "enable_a1", "enable_a2", "enable_a3", "enable_a4"),
        ),
    ):
        for values in product((False, True), repeat=6):
            result.append(cls(*BASE, **dict(zip(fields, values, strict=True))))
    result.extend(
        LJSplineUVModel(*BASE, use_temperature_dependent_u_fraction=value)
        for value in (False, True)
    )
    assert len(result) == 130
    return result


def reference(native):
    if isinstance(native, LJSplineBHModel):
        ref = ljs_bh()
        ref.model_control(
            use_Lafitte_a3=native.use_lafitte_a3,
            enable_chi_correction=native.enable_chi_correction,
            enable_hs=native.enable_hs,
            enable_a1=native.enable_a1,
            enable_a2=native.enable_a2,
            enable_a3=native.enable_a3,
        )
    elif isinstance(native, LJSplineWCAModel):
        ref = ljs_wca()
        ref.model_control(
            **{
                key: getattr(native, key)
                for key in (
                    "enable_cavity",
                    "enable_hs",
                    "enable_a1",
                    "enable_a2",
                    "enable_a3",
                    "enable_a4",
                )
            }
        )
    else:
        ref = ljs_uv()
        ref.model_control(
            use_temperature_dependent_u_fraction=native.use_temperature_dependent_u_fraction
        )
    ref.set_sigma_eps(native.diameter_m, native.well_depth_K)
    return ref


def second_response(ref, t, rho, order):
    if order == (2, 0):
        _, cv = ref.internal_energy_tv(t, 1 / rho, [1.0], dedt=True, property_flag="R")
        return -cv / ref.Rgas
    p, pv = ref.pressure_tv(t, 1 / rho, [1.0], dpdv=True)
    z = p / (rho * ref.Rgas * t)
    return -pv / (rho**2 * ref.Rgas * t) - 2 * z + 1


def third_reference(ref, t, rho, i, j):
    order = (2, 0) if i >= 2 else (0, 2)
    axis = 0 if (i, j) in ((3, 0), (1, 2)) else 1
    coordinate = t if axis == 0 else rho
    step = coordinate * 2e-4

    def f(offset):
        return second_response(
            ref,
            t + offset * step if axis == 0 else t,
            rho + offset * step if axis == 1 else rho,
            order,
        )

    derivative = (f(-2) - 8 * f(-1) + 8 * f(1) - f(2)) / (12 * step)
    value = (-t if axis == 0 else rho) * derivative
    if (i, j) in ((3, 0), (0, 3)):
        value -= 2 * second_response(ref, t, rho, order)
    return value


def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    cases = declarations()
    full_bh = [
        k
        for k, c in enumerate(cases[:64])
        if c.enable_hs and c.enable_a1 and c.enable_a2 and c.enable_a3
    ]
    defaults = [*full_bh, 127, 128, 129]
    assert len(defaults) == 7
    states = list(product((0.8, 1.5, 4.0), (0.0001, 0.05, 0.45, 1.1)))
    grid = [("state", k, t, r) for k in range(130) for t, r in states]
    grid += [("inverse", k, 4.0, 0.03) for k in range(130)]
    grid += [("third", k, t, r) for k in defaults for t, r in states]
    grid += [("diameter", 31, t, 0.0) for t in (0.8, 1.5, 4.0)]
    grid += [
        ("potential", 31, 2.0, r)
        for r in (
            0.8,
            1.0,
            WCA_MINIMUM,
            1.24,
            SPLINE_START,
            1.5,
            SPLINE_CUTOFF - 1e-6,
            SPLINE_CUTOFF,
            2.0,
        )
    ]
    assert len(grid) == 1786
    count = passed = 0
    cache, failures = {}, []
    started = time.perf_counter()

    def compare(name, actual, target, third=False):
        nonlocal count, passed
        atol = (
            2e-9
            if name in ("compressibility_factor", "diameter")
            else 2e-5
            if name.endswith("_J_mol")
            else 1e-4
            if name == "pressure_Pa"
            else 2e-7
        )
        rtol = 2e-5 if third else 2e-7
        row = {"quantity": name, "rtol": rtol, "atol": atol}
        for role, fn in (("native", actual), ("reference", target)):
            try:
                value = float(fn())
                if not math.isfinite(value):
                    raise ValueError("nonfinite result")
                row[role] = value
            except Exception as exc:
                row[role + "_exception"] = f"{type(exc).__name__}: {exc}"
        row["passed"] = (
            "native" in row
            and "reference" in row
            and math.isclose(row["native"], row["reference"], rel_tol=rtol, abs_tol=atol)
        )
        count += 1
        passed += row["passed"]
        return row

    with (directory / "records.jsonl").open("w", encoding="utf8") as stream:
        for unit, (kind, case, ts, rs) in enumerate(grid, 1):
            native = cases[case]
            if case not in cache:
                cache[case] = reference(native)
            ref = cache[case]
            t, rho = ts * 124, rs / SCALE
            records = []
            if kind == "state":
                actual, expected, errors = {}, {}, {}
                try:
                    actual = molecular_eos_state(
                        native, t, rho, [1.0], require_stable=False
                    ).to_dict()
                    actual["mu_J_mol"] = molecular_residual_chemical_potentials(
                        native, t, rho, [1.0]
                    )[0]
                except Exception as exc:
                    errors["native"] = exc
                try:
                    expected, mu = thermopack_residual_targets(ref, t, rho, [1.0])
                    expected["mu_J_mol"] = mu[0]
                except Exception as exc:
                    errors["reference"] = exc

                def lookup(role, name):
                    if role in errors:
                        raise errors[role]
                    return (actual if role == "native" else expected)[name]

                records = [
                    compare(name, lambda: lookup("native", name), lambda: lookup("reference", name))
                    for name in NAMES
                ]
            elif kind == "inverse":

                def inverse():
                    pressure = ref.pressure_tv(t, 1 / rho, [1.0])[0]
                    return molecular_density_from_pressure(
                        native,
                        t,
                        pressure,
                        [1.0],
                        initial_density_mol_m3=1.1 * rho,
                        density_bounds_mol_m3=(0.8 * rho, 1.2 * rho),
                    ).molar_density_mol_m3

                records = [compare("density_mol_m3", inverse, lambda: rho)]
            elif kind == "third":
                for i, j in ((3, 0), (2, 1), (1, 2), (0, 3)):
                    records.append(
                        compare(
                            f"Ar{i}{j}",
                            lambda: molecular_scaled_derivative(
                                native, t, rho, [1.0], inverse_temperature_order=i, density_order=j
                            ),
                            lambda: third_reference(ref, t, rho, i, j),
                            True,
                        )
                    )
            elif kind == "diameter":
                ref.activate()
                records = [
                    compare(
                        "diameter",
                        lambda: native.diameter_ratio(t),
                        lambda: ref.get_bh_diameter_div_sigma(ts),
                    )
                ]
            else:
                records = [
                    compare(
                        "potential_K",
                        lambda: native.potential_energy_over_k_K(rs * BASE[1]),
                        lambda: ref.potential(1, 1, [rs * BASE[1]], t)[0],
                    )
                ]
            for record in records:
                row = {
                    "unit": unit,
                    "case": case,
                    "block": kind,
                    "Tstar": ts,
                    "rhostar_or_radius": rs,
                    **record,
                }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if not row["passed"]:
                    failures.append(row)
            stream.flush()
            if unit % 5 == 0 or unit == len(grid):
                elapsed = time.perf_counter() - started
                print(
                    f"LJ spline {unit}/{len(grid)}; {unit / elapsed:.2f} units/s; "
                    f"ETA {(len(grid) - unit) * elapsed / unit:.1f}s; {passed}/{count} comparisons",
                    flush=True,
                )
    assert count == 17638
    summary = {
        "mode": "development",
        "backend": "ThermoPack 2.2.3",
        "units": len(grid),
        "comparisons": count,
        "passed": passed,
        "failed": len(failures),
        "elapsed_s": time.perf_counter() - started,
        "models": [c.to_dict() for c in cases],
        "failures": failures,
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k not in ("models", "failures")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
