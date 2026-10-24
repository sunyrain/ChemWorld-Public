# ruff: noqa: B023 -- each callback executes in the current fixed unit.
"""Prospectively fixed CSP comparison blocks; source imports only here."""

import argparse
import ctypes as ct
import json
import math
import re
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import teqp
from molecular_reference import thermopack_residual_targets
from thermopack.cubic import cubic
from thermopack.extended_csp import ext_csp
from thermopack.lee_kesler import lee_kesler
from thermopack.thermo import c_len_type

from chemworld.data.helmholtz import read_helmholtz_fluid
from chemworld.physchem.corresponding_states import (
    CorrespondingStatesComponent,
    CubicShapeComponent,
    ExtendedCorrespondingStatesModel,
    HuberElyModel,
    LeeKeslerPlockerModel,
    MBWRReference,
)
from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)

IDS = ("C1", "C3", "NC4")
CASES = (
    [((i,), (1.0,)) for i in range(3)]
    + [((0, 1), x) for x in ((1.0, 0.0), (0.0, 1.0), (0.3, 0.7), (0.7, 0.3))]
    + [((0, 1, 2), x) for x in ((0.2, 0.3, 0.5), (0.0, 0.4, 0.6))]
)
ORDERS = [(i, j) for i in range(4) for j in range(4 - i)]
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
)
LK_NAMES = tuple(
    n for n in NAMES if n not in ("residual_internal_energy_J_mol", "residual_cv_J_mol_K")
)
SOURCE = "Explicit parameters for CSP development comparison"


class UnavailableCubicReference:
    """Keep all planned units after source constructors terminated the process."""

    def __getattr__(self, name):
        raise RuntimeError(
            "ThermoPack 2.2.3 ext_csp cubic reference constructor terminates: "
            "Wrong input to extcsp_eos_allocate_and_init; source refEosType unset"
        )


def ext_csp_corrected_abi(components, shape, reference):
    """2.2.3 Python swaps sh_alpha/sh_mixing versus the Fortran declaration.

    Keep the installed library untouched; this reference-only adapter follows
    eoslibinit.f90::init_extcsp's actual argument order and explicit lengths.
    """
    model = ext_csp()
    model.activate()
    values = (components, shape, "vdW", "Classic", reference, "C3", "Classic", "Default")
    strings = [ct.c_char_p(v.encode("ascii")) for v in values]
    lengths = [c_len_type(len(v)) for v in values]
    fn = model.eoslibinit_init_extcsp
    fn.argtypes = [ct.c_char_p] * 8 + [c_len_type] * 8
    fn.restype = None
    fn(*strings, *lengths)
    model.nc = len(components.split(","))
    return model


def basic_components(ref, selected, estimated_volume=False):
    result = []
    for i, idx in enumerate(selected, 1):
        t, v, p = ref.get_critical_parameters(i)
        w = ref.acentric_factor(i)
        if estimated_volume:
            v = (0.2905 - 0.085 * w) * ref.Rgas * t / p
        result.append(CorrespondingStatesComponent(IDS[idx], t, v, w, SOURCE))
    return tuple(result)


def _set_pair(ref, i, j, value):
    ref.activate()
    fn = getattr(ref.tp, ref.get_export_name("", "thermopack_setkijandji"))
    fn.argtypes = [ct.POINTER(ct.c_int), ct.POINTER(ct.c_int), ct.POINTER(ct.c_double)]
    fn.restype = None
    fn(ct.byref(ct.c_int(i)), ct.byref(ct.c_int(j)), ct.byref(ct.c_double(value)))


def lkp_models(case, backend):
    selected, x = CASES[case]
    ref = lee_kesler(",".join(IDS[i] for i in selected))
    components = basic_components(ref, selected, backend == "teqp")
    kmat = tuple(tuple(1.0 for _ in x) for _ in x)
    if backend == "thermopack":
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                _set_pair(ref, i + 1, j + 1, 1.0)
    native = LeeKeslerPlockerModel(
        components,
        kmat,
        (100.0, 1000.0),
        SOURCE,
        interpolation="common_density" if backend == "teqp" else "common_pressure",
        coefficient_convention="teqp-0.23.2" if backend == "teqp" else "Lee-Kesler-1975",
    )
    if backend == "teqp":
        ref = teqp.make_model(
            {
                "kind": "LKP",
                "model": {
                    "Tcrit / K": [c.critical_temperature_K for c in components],
                    "pcrit / Pa": [
                        (0.2905 - 0.085 * c.acentric_factor)
                        * native.gas_constant_J_mol_K
                        * c.critical_temperature_K
                        / c.critical_molar_volume_m3_mol
                        for c in components
                    ],
                    "acentric": [c.acentric_factor for c in components],
                    "R / J/mol/K": native.gas_constant_J_mol_K,
                    "kmat": kmat,
                },
            }
        )
    return native, ref, np.asarray(x)


def huber_models(case):
    path = Path(teqp.get_datapath()) / "dev/fluids/R134a.json"
    fluid = read_helmholtz_fluid(path, source="teqp 0.23.2 public R134a JSON", eos_index=0)
    refdata = {
        "name": str(path),
        "acentric": 0.326680,
        "Z_crit": 4.056e6 / (5030.8 * 8.314471 * 374.179),
        "T_crit / K": 374.179,
        "rhomolar_crit / mol/m^3": 5030.8,
    }
    data = (
        {
            "name": "R143a",
            "acentric": 0.25540,
            "T_crit / K": 346.3,
            "rhomolar_crit / mol/m^3": 1000 / 0.194,
            "Z_crit": 3.76e6 / (346.3 * 8.314471 * 1000 / 0.194),
            "f_T_coeffs": [-0.022807, -0.64746],
            "h_T_coeffs": [0.36563, -0.026004],
        }
        if case == 0
        else {**refdata, "f_T_coeffs": [0.2, -0.1], "h_T_coeffs": [0.3, 0.1]}
        if case == 1
        else {
            "name": "synthetic",
            "acentric": 0.6,
            "T_crit / K": 500.0,
            "rhomolar_crit / mol/m^3": 3000.0,
            "Z_crit": 0.25,
            "f_T_coeffs": [0.2, -0.1],
            "h_T_coeffs": [0.3, 0.1],
        }
    )

    def component(v):
        return CorrespondingStatesComponent(
            v["name"], v["T_crit / K"], 1 / v["rhomolar_crit / mol/m^3"], v["acentric"], SOURCE
        )

    native = HuberElyModel(
        component(data),
        data["Z_crit"],
        component(refdata),
        refdata["Z_crit"],
        tuple(data["f_T_coeffs"]),
        tuple(data["h_T_coeffs"]),
        fluid,
        (150.0, 450.0),
        SOURCE,
    )
    ref = teqp.make_model(
        {
            "kind": "multifluid-ECS-HuberEly1994",
            "model": {"reference_fluid": refdata, "fluid": data},
        },
        validate=False,
    )
    return native, ref, np.asarray([1.0])


def cubic_components(selected, model):
    ref = cubic(",".join(IDS[i] for i in selected), model)
    cs = basic_components(ref, selected)
    aa, bb = ref.get_energy_constants() * 1e-6, ref.get_covolumes() * 1e-3
    result = []
    for c, a, b in zip(cs, aa, bb, strict=True):
        w = c.acentric_factor
        m = (
            0.48 + 1.574 * w - 0.176 * w * w
            if model == "SRK"
            else 0.37464 + 1.54226 * w - 0.26992 * w * w
        )
        result.append(
            CubicShapeComponent(
                c.component_id, a, b, c.critical_temperature_K, "soave", (m,), SOURCE
            )
        )
    return tuple(result)


def mbwr_parameters(source_dir, variant):
    source = (source_dir / "mbwrdata.f90").read_text()
    pattern = (
        rf"TYPE\(mbwr{variant}Data\).*?=\s*mbwr{variant}Data\(\s*&?\s*"
        r'\d+,\s*"C3".*?\(/(.*?)/\)'
    )
    match = re.search(pattern, source, re.S | re.I)
    # Prevent a permissive match from crossing the preceding declaration.
    blocks = re.findall(
        rf'mbwr{variant}Data\(\s*&?\s*\d+,\s*"([^"]+)"(.*?)\)\s*&?\s*\)', source, re.S | re.I
    )
    for name, body in blocks:
        if name == "C3":
            values = re.search(r"\(/(.*?)/", body, re.S).group(1)
            coefficients = [
                float(v) for v in re.findall(r"[-+]?\d*\.\d+(?:[EeDd][-+]?\d+)?", values)
            ]
            assert len(coefficients) == variant + 1
            return coefficients
    raise ValueError(f"MBWR{variant} propane parameters not found; first match={bool(match)}")


def ecs_models(case, shape_model, reference_model, source_dir):
    selected, x = CASES[case]
    components = cubic_components(selected, shape_model)
    shape = cubic_components((1,), shape_model)[0]
    if reference_model in ("SRK", "PR"):
        c = cubic_components((1,), reference_model)[0]
        base = CPAModel(
            (
                CPAComponent(
                    c.component_id,
                    c.attraction_Pa_m6_mol2,
                    c.covolume_m3_mol,
                    c.critical_temperature_K,
                    c.alpha_coefficients[0],
                    c.source,
                ),
            ),
            ((0.0,),),
            (10.0, 3000.0),
            SOURCE,
            cubic_model="srk" if reference_model == "SRK" else "peng_robinson",
        )
    else:
        if reference_model == "NIST_MEOS":
            base = read_helmholtz_fluid(
                Path(teqp.get_datapath()) / "dev/fluids/n-Propane.json",
                source="teqp public propane coefficients",
                eos_index=0,
            )
            tc, pc = 369.89, 4251200.0
            base = replace(base, minimum_temperature_K=10.0, maximum_temperature_K=3000.0)
        else:
            variant = int(reference_model[-2:])
            coeffs = mbwr_parameters(source_dir, variant)
            base = MBWRReference(
                "C3",
                variant,
                tuple(coeffs[:-1]),
                coeffs[-1],
                (10.0, 3000.0),
                "ThermoPack public explicit MBWR coefficients",
            )
            tc, pc = (
                (370.16219999990909, 4297516.9991680998)
                if variant == 19
                else (369.85079999998726, 4247648.2466478515)
            )
        data = cubic("C3", shape_model)
        old_tc, _, old_pc = data.get_critical_parameters(1)
        shape = replace(
            shape,
            critical_temperature_K=tc,
            attraction_Pa_m6_mol2=shape.attraction_Pa_m6_mol2 * (tc / old_tc) ** 2 * old_pc / pc,
            covolume_m3_mol=shape.covolume_m3_mol * tc / old_tc * old_pc / pc,
        )
    shape_parameters = cubic(",".join(IDS[i] for i in selected), shape_model)
    interaction = tuple(
        tuple(0.0 if i == j else shape_parameters.get_kij(i + 1, j + 1) for j in range(len(x)))
        for i in range(len(x))
    )
    native = ExtendedCorrespondingStatesModel(
        components,
        interaction,
        shape,
        base,
        (100.0, 1000.0),
        SOURCE,
        cubic_model="srk" if shape_model == "SRK" else "peng_robinson",
    )
    ref = (
        UnavailableCubicReference()
        if reference_model in ("SRK", "PR")
        else ext_csp_corrected_abi(",".join(IDS[i] for i in selected), shape_model, reference_model)
    )
    return native, ref, np.asarray(x)


def teqp_targets(ref, t, rho, x):
    a = {
        (i, j): ref.get_Arxy(i, j, t, rho, x)
        for i, j in ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0))
    }
    R, z = ref.get_R(x), 1 + a[0, 1]
    stiffness, thermal = 1 + 2 * a[0, 1] + a[0, 2], z - a[1, 1]
    return dict(
        zip(
            NAMES,
            (
                rho * R * t * z,
                z,
                R * t * a[0, 0],
                R * t * a[1, 0],
                R * t * (a[1, 0] + z - 1),
                R * (a[1, 0] - a[0, 0]),
                -R * a[2, 0],
                R * (-a[2, 0] + thermal**2 / stiffness - 1),
                rho * R * thermal,
                R * t * stiffness,
            ),
            strict=True,
        )
    )


def lkp_targets(ref, t, pressure, x):
    v, vt, vp = ref.specific_volume(t, pressure, x, ref.VAPPH, dvdt=True, dvdp=True)
    z = pressure * v / (ref.Rgas * t)
    h, cp = ref.enthalpy(t, pressure, x, ref.VAPPH, dhdt=True, residual=True)
    (entropy,) = ref.entropy(t, pressure, x, ref.VAPPH, residual=True)
    (phi,) = ref.thermo(t, pressure, x, ref.VAPPH)
    values = {
        "pressure_Pa": pressure,
        "compressibility_factor": z,
        "residual_helmholtz_J_mol": ref.Rgas * t * (sum(x * phi) - z + 1 + math.log(z)),
        "residual_enthalpy_J_mol": h,
        "residual_entropy_J_mol_K": entropy - ref.Rgas * math.log(z),
        "residual_cp_J_mol_K": cp,
        "dP_dT_rho_Pa_K": -vt / vp,
        "dP_drho_T_Pa_m3_mol": -v * v / vp,
    }
    values.update({f"lnphi_{i}": value for i, value in enumerate(phi)})
    return 1 / v, values


def run(output, block, source_dir):
    output.mkdir(parents=True, exist_ok=False)
    is_teqp = block in ("teqp-lkp", "huber-ely")
    if is_teqp:
        cases = range(3) if block == "huber-ely" else range(9)
        temps = (200.0, 300.0, 400.0) if block == "huber-ely" else (250.0, 400.0, 600.0)
        grid = [
            ("derivative", c, "", "", t, r)
            for c, t, r in product(
                cases,
                temps,
                (0.0, 1e-6, 1.0, 1000.0, 5000.0, 12000.0 if block == "teqp-lkp" else 15000.0),
            )
        ]
        grid += [
            ("state", c, "", "", temps[-1], r) for c, r in product(cases, (1.0, 1000.0, 5000.0))
        ]
    elif block == "thermopack-lkp":
        grid = [
            ("state", c, "", "", t, p)
            for c, t, p in product(range(9), (250.0, 400.0, 600.0), (1e4, 1e5, 1e6, 1e7))
        ]
        grid += [("inverse", c, "", "", 600.0, 1e5) for c in range(9)]
    else:
        models = list(product(("SRK", "PR"), ("SRK", "PR", "NIST_MEOS", "MBWR19", "MBWR32")))
        grid = [
            ("state", c, s, b, t, r)
            for (s, b), c, t, r in product(
                models, range(9), (200.0, 300.0, 400.0), (1.0, 1000.0, 5000.0, 12000.0)
            )
        ]
        grid += [("inverse", c, s, b, 400.0, 1000.0) for (s, b), c in product(models, range(9))]
    expected = {
        "teqp-lkp": (189, 1968),
        "huber-ely": (63, 648),
        "thermopack-lkp": (117, 1077),
        "thermopack-ecs": (1170, 12930),
    }[block]
    assert len(grid) == expected[0]
    cache, failures, metadata, count, passed = {}, [], [], 0, 0
    started = time.perf_counter()

    def compare(name, actual, reference):
        nonlocal count, passed
        atol = (
            1e-4
            if name == "pressure_Pa"
            else 2e-5
            if name.endswith("_J_mol") or name.startswith("mu_")
            else 2e-9
            if name.startswith(("Ar", "lnphi")) or name == "compressibility_factor"
            else 2e-7
        )
        record = {"quantity": name, "rtol": 2e-7, "atol": atol}
        for role, fun in (("native", actual), ("reference", reference)):
            try:
                value = float(fun())
                if not math.isfinite(value):
                    raise ValueError("nonfinite result")
                record[role] = value
            except Exception as exc:
                record[role + "_exception"] = f"{type(exc).__name__}: {exc}"
        record["passed"] = (
            "native" in record
            and "reference" in record
            and math.isclose(record["native"], record["reference"], rel_tol=2e-7, abs_tol=atol)
        )
        count += 1
        passed += record["passed"]
        return record

    with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
        for unit, (kind, case, shape, base, t, value) in enumerate(grid, 1):
            key = case, shape, base
            if key not in cache:
                cache[key] = (
                    huber_models(case)
                    if block == "huber-ely"
                    else lkp_models(case, "teqp" if is_teqp else "thermopack")
                    if block.endswith("lkp")
                    else ecs_models(case, shape, base, source_dir)
                )
                metadata.append(
                    {"case": case, "shape": shape, "base": base, "model": cache[key][0].to_dict()}
                )
            native, ref, x = cache[key]
            rho, records = value, []
            if kind == "derivative":
                for i, j in ORDERS:
                    records.append(
                        compare(
                            f"Ar{i}{j}",
                            lambda: molecular_scaled_derivative(
                                native, t, rho, x, inverse_temperature_order=i, density_order=j
                            ),
                            lambda: ref.get_Arxy(i, j, t, rho, x),
                        )
                    )
            else:
                errors, targets, actual = {}, {}, {}
                try:
                    if block == "thermopack-lkp":
                        rho, targets = lkp_targets(ref, t, value, x)
                    elif is_teqp:
                        targets = teqp_targets(ref, t, rho, x)
                        targets.update(
                            {
                                f"lnphi_{i}": math.log(v)
                                for i, v in enumerate(ref.get_fugacity_coefficients(t, rho * x))
                            }
                        )
                    else:
                        targets, mus = thermopack_residual_targets(ref, t, rho, x)
                        targets.update({f"mu_{i}": v for i, v in enumerate(mus)})
                except Exception as exc:
                    errors["reference"] = exc
                try:
                    state = molecular_eos_state(native, t, rho, x, require_stable=False)
                    actual = dict(state.__dict__)
                    if block == "thermopack-ecs":
                        actual.update(
                            {
                                f"mu_{i}": v
                                for i, v in enumerate(
                                    molecular_residual_chemical_potentials(native, t, rho, x)
                                )
                            }
                        )
                    elif state.log_fugacity_coefficients is not None:
                        actual.update(
                            {f"lnphi_{i}": v for i, v in enumerate(state.log_fugacity_coefficients)}
                        )
                except Exception as exc:
                    errors["native"] = exc

                def get(name, ref=False):
                    if ("reference" if ref else "native") in errors:
                        raise errors["reference" if ref else "native"]
                    return (targets if ref else actual)[name]

                if kind == "state":
                    names = list(LK_NAMES if block == "thermopack-lkp" else NAMES)
                    names += [
                        f"{'mu' if block == 'thermopack-ecs' else 'lnphi'}_{i}"
                        for i in range(len(x))
                    ]
                    records += [compare(n, lambda: get(n), lambda: get(n, True)) for n in names]
                if kind == "inverse" or is_teqp:
                    records.append(
                        compare(
                            "pressure_density_inverse",
                            lambda: (
                                molecular_density_from_pressure(
                                    native,
                                    t,
                                    get("pressure_Pa", True),
                                    x,
                                    initial_density_mol_m3=1.1 * rho,
                                    density_bounds_mol_m3=(0.5 * rho, 1.5 * rho),
                                ).molar_density_mol_m3
                            ),
                            lambda: rho,
                        )
                    )
            row = {
                "unit": unit,
                "kind": kind,
                "case": case,
                "shape": shape,
                "base": base,
                "T_K": t,
                "input_density_or_pressure": value,
                "comparisons": records,
            }
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            failures.extend(
                {**{k: v for k, v in row.items() if k != "comparisons"}, **r}
                for r in records
                if not r["passed"]
            )
            if unit % 6 == 0 or unit == len(grid):
                elapsed = time.perf_counter() - started
                print(
                    f"CSP {block} {unit}/{len(grid)}; {unit / elapsed:.2f} units/s; "
                    f"ETA {(len(grid) - unit) * elapsed / unit:.1f}s; failures {len(failures)}",
                    flush=True,
                )
    summary = {
        "mode": "development",
        "block": block,
        "units": len(grid),
        "comparisons": count,
        "expected_comparisons": expected[1],
        "passed": passed,
        "failed": len(failures),
        "failures": failures,
        "model_inputs": metadata,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("failures", "model_inputs")}))
    assert count == expected[1]
    return not failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--block",
        required=True,
        choices=("teqp-lkp", "huber-ely", "thermopack-lkp", "thermopack-ecs"),
    )
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.block, args.source_dir) else 1)
