"""NATIVE-OPT-01 fixed development block; no reference imports in production.

Run with uv run --no-sync python scripts/benchmark_helmholtz.py --output <external-dir>.
Each measurement uses a fresh process, with disjoint native/reference imports.
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

FLUIDS = ("Water", "CarbonDioxide", "Methane", "Nitrogen", "Ammonia", "R134a")
BACKENDS = ("legacy_native", "fused_native", "coolprop_heos")
FIELDS = (
    "pressure_Pa",
    "enthalpy_J_mol",
    "internal_energy_J_mol",
    "entropy_J_mol_K",
    "cp_J_mol_K",
    "cv_J_mol_K",
    "speed_of_sound_m_s",
    "compressibility_factor",
    "dP_dT_rho_Pa_K",
    "dP_drho_T_Pa_m3_mol",
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def memory():
    import psutil

    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):
        peak, source = info.peak_wset, "psutil.peak_wset_bytes"
    else:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak *= 1 if sys.platform == "darwin" else 1024
        source = "resource.ru_maxrss_converted_bytes"
    return {"rss_bytes": info.rss, "peak_rss_bytes": peak, "peak_source": source}


def native_reader(sources, legacy, snapshot=None):
    from chemworld.data.helmholtz import load_helmholtz_fluid
    from chemworld.physchem.helmholtz import HelmholtzFluid

    class LegacyFluid(HelmholtzFluid):
        # f0f6f573 state(): exact old six-evaluation strategy, no old runtime.
        def _state_derivatives(self, delta, tau):
            return tuple(
                self.alpha_derivative(kind, delta=delta, tau=tau, delta_order=d, tau_order=t)
                for kind, d, t in (
                    ("total", 0, 0),
                    ("total", 0, 1),
                    ("total", 0, 2),
                    ("residual", 1, 0),
                    ("residual", 2, 0),
                    ("residual", 1, 1),
                )
            )

    if legacy and snapshot is not None:
        import importlib

        sys.path.insert(0, str(snapshot))
        load_helmholtz_fluid = importlib.import_module(
            "chemworld_opt01_baseline.data"
        ).load_helmholtz_fluid
    models = {
        name: load_helmholtz_fluid(raw, source="CoolProp 8.0.0/" + name, eos_index=0)
        for name, raw in sources.items()
    }
    if legacy and snapshot is None:
        models = {name: LegacyFluid.from_dict(model.to_dict()) for name, model in models.items()}

    def read(name, temperature, density):
        state = models[name].state(temperature_K=temperature, molar_density_mol_m3=density)
        return tuple(getattr(state, key) for key in FIELDS)

    return read


def reference_reader(sources, enforce_domain=False):
    import CoolProp
    import CoolProp.CoolProp as CP  # noqa: N817

    if CoolProp.__version__ != "8.0.0":
        raise ValueError("fixed block requires CoolProp 8.0.0")
    models = {name: CP.AbstractState("HEOS", name) for name in sources}
    for model in models.values():
        model.specify_phase(CP.iphase_gas)

    def read(name, temperature, density):
        eos = sources[name]["EOS"][0]
        minimum = eos["STATES"]["sat_min_liquid"]["T"]
        if enforce_domain and not minimum <= temperature <= eos["T_max"]:
            raise ValueError("temperature outside Helmholtz source validity")
        model = models[name]
        model.update(CP.DmolarT_INPUTS, density, temperature)
        return (
            model.p(),
            model.hmolar(),
            model.umolar(),
            model.smolar(),
            model.cpmolar(),
            model.cvmolar(),
            model.speed_sound(),
            model.compressibility_factor(),
            model.first_partial_deriv(CP.iP, CP.iT, CP.iDmolar),
            model.first_partial_deriv(CP.iP, CP.iDmolar, CP.iT),
        )

    return read


def worker(backend, output, inputs, design, legacy_kernel):
    started = time.perf_counter()
    sources = json.loads(inputs.read_text(encoding="utf-8"))
    read = (
        reference_reader(sources, design == "declared-domain")
        if backend == "coolprop_heos"
        else native_reader(
            sources,
            backend == "legacy_native",
            inputs.parent / "baseline" if legacy_kernel == "snapshot" else None,
        )
    )
    setup_s = time.perf_counter() - started
    setup_memory = memory()
    points = []
    for name, raw in sources.items():
        reducing = raw["EOS"][0]["STATES"]["reducing"]
        temperatures = (
            [tf * reducing["T"] for tf in (1.01, 1.4)]
            if design == "original"
            else [reducing["T"] + f * (raw["EOS"][0]["T_max"] - reducing["T"]) for f in (0.02, 0.4)]
        )
        points.extend(
            (name, temperature, df * reducing["rhomolar"])
            for temperature in temperatures
            for df in (0.02, 0.4, 1.2)
        )
    first = time.perf_counter()
    results, failures = [], []
    for name, temperature, density in points:
        row = {"fluid": name, "temperature_K": temperature, "density_mol_m3": density}
        try:
            values = read(name, temperature, density)
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite response")
            if values[0] <= 0 or values[5] <= 0 or values[9] <= 0:
                raise ValueError("not locally stable")
            if values[0] > sources[name]["EOS"][0]["p_max"]:
                raise ValueError("pressure outside source domain")
            row.update(values=dict(zip(FIELDS, values, strict=True)), passed=True)
        except Exception:
            row.update(passed=False, error=traceback.format_exc())
            failures.append(row)
        results.append(row)
    first_pass_s = time.perf_counter() - first
    timings = []
    for name in sources:
        group = [point for point in points if point[0] == name]
        try:
            if any(row["fluid"] == name for row in failures):
                raise ValueError("fixed fluid contains a failed state; no selective timing")
            for _ in range(2):
                for point in group:
                    read(*point)
            start = time.perf_counter()
            checksum, sweeps = 0.0, 0
            while sweeps < 10000:
                for point in group:
                    checksum += sum(read(*point))
                sweeps += 1
                elapsed = time.perf_counter() - start
                if sweeps >= 10 and elapsed >= 0.25:
                    break
            timings.append(
                {
                    "fluid": name,
                    "passed": True,
                    "calls": sweeps * len(group),
                    "elapsed_s": elapsed,
                    "seconds_per_call": elapsed / (sweeps * len(group)),
                    "checksum": checksum,
                }
            )
        except Exception:
            timings.append({"fluid": name, "passed": False, "error": traceback.format_exc()})
        print(
            f"{backend}: {len(timings)}/{len(sources)} fluids; "
            f"elapsed {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    domain_checks = []
    if design == "declared-domain":
        reducing = sources["R134a"]["EOS"][0]["STATES"]["reducing"]
        for fraction in (0.02, 0.4, 1.2):
            try:
                read("R134a", 1.4 * reducing["T"], fraction * reducing["rhomolar"])
                domain_checks.append(
                    {
                        "density_fraction": fraction,
                        "passed": False,
                        "error": "out-of-domain temperature accepted",
                    }
                )
            except ValueError as exc:
                domain_checks.append(
                    {
                        "density_fraction": fraction,
                        "passed": "temperature outside" in str(exc),
                        "diagnostic": str(exc),
                    }
                )
    write(
        output,
        {
            "backend": backend,
            "design": design,
            "legacy_kernel": legacy_kernel,
            "setup_s": setup_s,
            "setup_memory": setup_memory,
            "first_pass_s": first_pass_s,
            "memory": memory(),
            "points": results,
            "timings": timings,
            "domain_checks": domain_checks,
            "passed": not failures and all(r["passed"] for r in timings + domain_checks),
            "reference_imported": "CoolProp" in sys.modules,
            "native_imported": "chemworld" in sys.modules,
        },
    )


def main(output, design, legacy_kernel):
    import CoolProp
    import CoolProp.CoolProp as CP  # noqa: N817
    import psutil

    if CoolProp.__version__ != "8.0.0":
        raise ValueError("fixed block requires CoolProp 8.0.0")
    output.mkdir(parents=True, exist_ok=False)
    if legacy_kernel == "snapshot":
        target = output / "baseline" / "chemworld_opt01_baseline"
        target.mkdir(parents=True)
        (target / "__init__.py").write_text("", encoding="utf-8")
        for name, source in (
            ("caloric_math", "physchem/caloric_math"),
            ("helmholtz_terms", "physchem/helmholtz_terms"),
            ("helmholtz", "physchem/helmholtz"),
            ("data", "data/helmholtz"),
        ):
            original = subprocess.check_output(
                ["git", "show", f"f0f6f573:src/chemworld/{source}.py"], text=True, encoding="utf-8"
            )
            (target / (name + ".py")).write_text(
                original.replace("from chemworld.physchem.", "from ."), encoding="utf-8"
            )
    inputs = output / "sources.json"
    write(inputs, {name: json.loads(CP.get_fluid_param_string(name, "JSON"))[0] for name in FLUIDS})
    env = {
        **os.environ,
        **dict.fromkeys(
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"),
            "1",
        ),
    }
    jobs = []
    started = time.monotonic()
    for repeat in range(5):
        order = BACKENDS[repeat % 3 :] + BACKENDS[: repeat % 3]
        for backend in order:
            path = output / f"r{repeat + 1}-{backend}.json"
            before = time.monotonic()
            with path.with_suffix(".log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        __file__,
                        "--worker",
                        backend,
                        "--design",
                        design,
                        "--legacy-kernel",
                        legacy_kernel,
                        "--inputs",
                        str(inputs),
                        "--output",
                        str(path),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                timeout = False
                while process.poll() is None:
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        live = time.monotonic() - before
                        print(f"OPT01 {len(jobs)}/15; {backend} alive {live:.0f}s", flush=True)
                        if live >= 120:
                            process.kill()
                            process.wait()
                            timeout = True
            row = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.exists()
                else {"backend": backend, "passed": False, "error": "worker produced no result"}
            )
            row.update(
                repeat=repeat + 1,
                process_wall_s=time.monotonic() - before,
                returncode=process.returncode,
                timeout=timeout,
            )
            row["passed"] &= process.returncode == 0 and not timeout
            jobs.append(row)
            elapsed = time.monotonic() - started
            print(
                f"OPT01 {len(jobs)}/15; {len(jobs) / elapsed:.2f} processes/s; "
                f"ETA {elapsed / len(jobs) * (15 - len(jobs)):.1f}s; passed={row['passed']}",
                flush=True,
            )
            write(output / "partial.json", jobs)
    comparisons = []
    for repeat in range(1, 6):
        by = {r["backend"]: r for r in jobs if r["repeat"] == repeat}
        for index in range(36):
            row = {"repeat": repeat, "point_index": index, "metrics": {}}
            try:
                new = by["fused_native"]["points"][index]
                row.update(
                    fluid=new["fluid"],
                    temperature_K=new["temperature_K"],
                    density_mol_m3=new["density_mol_m3"],
                )
                for backend, atol, rtol in (
                    ("legacy_native", 2e-10, 2e-11),
                    ("coolprop_heos", 2e-7, 2e-8),
                ):
                    for field in FIELDS:
                        value = new["values"][field]
                        reference = by[backend]["points"][index]["values"][field]
                        error, limit = abs(value - reference), atol + rtol * abs(reference)
                        row["metrics"][backend + ":" + field] = {
                            "absolute_error": error,
                            "allowed_error": limit,
                            "passed": error <= limit,
                        }
                row["passed"] = all(m["passed"] for m in row["metrics"].values())
            except Exception:
                row.update(passed=False, error=traceback.format_exc())
            comparisons.append(row)
    costs = []
    for fluid in FLUIDS:
        row = {"fluid": fluid, "backends": {}}
        for backend in BACKENDS:
            values = [
                t["seconds_per_call"]
                for j in jobs
                if j["backend"] == backend
                for t in j.get("timings", [])
                if t["fluid"] == fluid and t["passed"]
            ]
            row["backends"][backend] = {
                "completed": len(values),
                "samples_s": values,
                "median_s": statistics.median(values) if values else None,
                "min_s": min(values) if values else None,
                "max_s": max(values) if values else None,
            }
        if all(row["backends"][b]["completed"] == 5 for b in BACKENDS):
            old, new, ref = (row["backends"][b]["median_s"] for b in BACKENDS)
            row.update(native_speedup=old / new, native_over_heos_time=new / ref)
        costs.append(row)
    complete = all("native_speedup" in c for c in costs)
    speedup = statistics.geometric_mean(c["native_speedup"] for c in costs) if complete else None
    ratio = (
        statistics.geometric_mean(c["native_over_heos_time"] for c in costs) if complete else None
    )
    summary = {
        "development_only": True,
        "design": design,
        "legacy_kernel": legacy_kernel,
        "baseline_source": "f0f6f573",
        "reference": "CoolProp 8.0.0 HEOS",
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
            "psutil": psutil.__version__,
            "logical_cpus": os.cpu_count(),
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "development_worktree": True,
            "load_after_percent": psutil.cpu_percent(interval=1),
        },
        "expected_processes": 15,
        "processes_passed": sum(j["passed"] for j in jobs),
        "expected_states_per_process": 36,
        "comparison_rows": len(comparisons),
        "comparison_rows_passed": sum(r["passed"] for r in comparisons),
        "expected_scalar_comparisons": 3600,
        "domain_checks": sum(len(j.get("domain_checks", [])) for j in jobs),
        "domain_checks_passed": sum(r["passed"] for j in jobs for r in j.get("domain_checks", [])),
        "costs": costs,
        "geomean_native_speedup": speedup,
        "geomean_native_over_heos_time": ratio,
        "performance_adoption_passed": complete
        and speedup >= 1.1
        and all(c["native_speedup"] >= 0.95 for c in costs),
        "numerical_passed": all(j["passed"] for j in jobs)
        and all(r["passed"] for r in comparisons),
        "elapsed_s": time.monotonic() - started,
        "jobs": jobs,
        "comparisons": comparisons,
    }
    write(output / "summary.json", summary)
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"jobs", "comparisons", "costs"}})
    )
    return 0 if summary["numerical_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker", choices=BACKENDS)
    parser.add_argument(
        "--legacy-kernel", choices=("reconstructed", "snapshot"), default="reconstructed"
    )
    parser.add_argument("--design", choices=("original", "declared-domain"), default="original")
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.output, args.inputs, args.design, args.legacy_kernel)
    else:
        sys.exit(main(args.output, args.design, args.legacy_kernel))
