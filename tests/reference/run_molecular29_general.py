"""Frozen parent29 source and native qualification with retained failed attempts."""

from __future__ import annotations

import argparse
import json
import traceback
import warnings
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import numpy as np
from scipy.interpolate import PchipInterpolator

from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.molecular_interface import (
    molecular_planar_interface,
    pure_square_gradient_interface,
)
from reference.molecular29_cases import fixed_cases, initial_tanh, native_pair
from reference.molecular29_source_reference import (
    pure_interior_positions,
    source_interface,
    source_model,
)

PURE_CHECKS = (
    "source_endpoint",
    "source_validity",
    "source_convergence",
    "native_success",
    "tension_reference",
    "positions_reference",
)
PROFILE_CHECKS = (
    "source_endpoint",
    "source_validity",
    "source_convergence",
    "source_profile_convergence",
    "all_six_profiles",
    "tension_reference",
    "grid_convergence_L",
    "grid_convergence_2L",
    "domain_convergence",
    "density_path_reference",
    "width_reference",
    "spatial_shape_reference",
    "stationarity",
    "first_integral",
)
FRACTIONS = np.linspace(0.1, 0.9, 17)
PURE_FRACTIONS = (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_new(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False, default=json_default)
        handle.write("\n")


class Progress:
    def __init__(self, total, stage):
        self.total, self.completed, self.stage = total, 0, stage
        self.started, self.stop = monotonic(), Event()
        self.thread = Thread(target=self.heartbeat, daemon=True)
        self.thread.start()

    def heartbeat(self):
        while not self.stop.wait(30):
            self.report()

    def report(self):
        elapsed = monotonic() - self.started
        rate = self.completed / max(elapsed, 0.001)
        eta = f"{(self.total - self.completed) / rate:.1f}s" if rate else "pending"
        print(
            f"{self.stage}; {self.completed}/{self.total}; elapsed={elapsed:.1f}s; "
            f"rate={rate:.4f}/s; ETA={eta}",
            flush=True,
        )

    def close(self):
        self.stop.set()
        self.thread.join()


def profile_coordinates(z, rho, left, right):
    """Interior first crossing; no sorting a multi-interface path into a valid one."""
    z, rho, left, right = map(np.asarray, (z, rho, left, right))
    fraction = (rho[:, 0] - left[0]) / (right[0] - left[0])
    interior = np.flatnonzero((fraction >= 0.05) & (fraction <= 0.95))
    if len(interior) < 4 or np.any(np.diff(interior) != 1):
        raise ValueError("unresolved or multiple interior anchor crossings")
    indices = np.arange(max(0, interior[0] - 1), min(len(z), interior[-1] + 2))
    q = fraction[indices]
    if np.any(np.diff(q) <= 0) or q[0] > 0.1 or q[-1] < 0.9:
        raise ValueError("interior anchor profile must have one monotone crossing")
    positions = PchipInterpolator(q, z[indices])(FRACTIONS)
    densities = PchipInterpolator(q, rho[indices], axis=0)(FRACTIONS)
    centered = positions - positions[len(FRACTIONS) // 2]
    width = positions[-1] - positions[0]
    if not np.isfinite(width) or width <= 0:
        raise ValueError("nonpositive profile width")
    return {"positions": centered, "densities": densities, "width": float(width)}


def compare_profiles(actual, expected):
    left, right = expected["left"], expected["right"]
    a = profile_coordinates(actual["z"], actual["rho"], left, right)
    b = profile_coordinates(expected["z"], expected["rho"], left, right)
    span = np.abs(np.asarray(right) - left)
    if np.any(span <= 0):
        raise ValueError("fixed reference profile requires distinct component endpoints")
    return {
        "density_path_relative": float(np.max(abs(a["densities"] - b["densities"]) / span)),
        "width_relative": abs(a["width"] / b["width"] - 1),
        "spatial_shape_relative": float(np.max(abs(a["positions"] - b["positions"])) / b["width"]),
        "actual_width_m": a["width"],
        "source_width_m": b["width"],
    }


def source_diagnostics(result, case):
    raw = np.asarray(result["raw_dimensionless_grand_potential"])
    finite = bool(
        np.all(np.isfinite(raw))
        and np.all(np.isfinite(result["rho"]))
        and np.all(np.isfinite(result["z"]))
        and result["gamma"] > 0
    )
    maximum = float(np.max(raw))
    diagnostics = {
        "finite": finite,
        "raw_omega_min": float(np.min(raw)),
        "raw_omega_max": maximum,
        "unclipped_omega_valid": maximum > 0 and np.min(raw) >= -1e-6 * maximum,
    }
    if case["family"] != "pure" and case["beta"] != 0:
        summary = result["source_summary"]
        diagnostics["root_valid"] = bool(
            summary.get("success", False) and summary.get("fun_norm", 1.0) <= 1e-7
        )
    if case["beta"] == 0:
        model = source_model(case["family"], 0)
        factors = model.sgt_adim(case["T"])
        auxiliary = model.temperature_aux(case["T"])
        left = np.asarray(result["left"]) * 1e-6 * factors[2]
        mu = model.muad_aux(left, auxiliary)
        _, _, vectors = np.linalg.svd(np.sqrt(np.diag(model.ci(case["T"])))[None, :])
        null = vectors[1:].T
        residual, curvature = 0.0, float("inf")
        for c in np.asarray(result["rho"])[1:-1] * 1e-6 * factors[2]:
            residual = max(
                residual, float(np.max(abs(null.T @ (model.muad_aux(c, auxiliary) - mu))))
            )
            step = min(1e-5, float(np.min(c)) * 0.1)
            hessian = np.column_stack(
                [
                    (
                        model.muad_aux(c + np.eye(len(c))[j] * step, auxiliary)
                        - model.muad_aux(c - np.eye(len(c))[j] * step, auxiliary)
                    )
                    / (2 * step)
                    for j in range(len(c))
                ]
            )
            curvature = min(curvature, float(np.min(np.linalg.eigvalsh(null.T @ hessian @ null))))
        diagnostics.update(
            null_mu_residual=residual,
            null_min_curvature=curvature,
            null_valid=residual <= 1e-7 and curvature > 0,
        )
    diagnostics["valid"] = bool(
        finite
        and diagnostics["unclipped_omega_valid"]
        and diagnostics.get("root_valid", True)
        and diagnostics.get("null_valid", True)
    )
    return diagnostics


def build_sources(directory):
    directory.mkdir(parents=True, exist_ok=False)
    progress, groups = Progress(36, "source starting"), []
    try:
        for case in fixed_cases():
            group = {"case": case, "runs": []}
            lengths = case.get("lengths", [None])
            if not (case["family"] == "symmetric_lle" and case["beta"] == 0.5):
                lengths = [None]
            for length in lengths:
                initial = None
                for nodes in (40, 80, 120):
                    progress.stage = f"source {case['name']} n={nodes} L={length}"
                    run = {"nodes": nodes, "length": length}
                    started = monotonic()
                    try:
                        method = "auto" if case["beta"] != 0 and length is None else "reference"
                        with warnings.catch_warnings(record=True) as caught:
                            result = source_interface(
                                case["family"],
                                case["T"],
                                case["P"],
                                beta=case["beta"],
                                length_m=length or 3e-9,
                                nodes=nodes,
                                initial=initial,
                                method=method,
                            )
                        initial = result.pop("source_result")
                        run.update(result=result, warnings=[str(w.message) for w in caught])
                        run["diagnostics"] = source_diagnostics(result, case)
                        if case["family"] == "pure":
                            run["positions"] = pure_interior_positions(case["T"], PURE_FRACTIONS)
                    except Exception:
                        run["exception"] = traceback.format_exc()
                    run["elapsed_s"] = monotonic() - started
                    group["runs"].append(run)
                    write_new(directory / f"source-{progress.completed + 1:02d}.json", run)
                    progress.completed += 1
                    progress.report()
            groups.append(group)
        summary = {
            "mode": "development",
            "expected_runs": 36,
            "completed_runs": progress.completed,
            "groups": groups,
            "elapsed_s": monotonic() - progress.started,
        }
        source_checks = []
        for group in groups:
            checks, _ = source_group_checks(group)
            if group["case"]["family"] == "pure":
                checks.pop("source_profile_convergence")
            source_checks.append({"group": group["case"]["name"], "checks": checks})
        summary["source_checks"] = source_checks
        summary["success"] = progress.completed == 36 and all(
            all(row["checks"].values()) for row in source_checks
        )
        write_new(directory / "SOURCE.json", summary)
        return summary
    finally:
        progress.close()


def source_group_checks(group):
    runs = group["runs"]
    final = [r for r in runs if r["nodes"] == 120]
    expected = (
        2 if (group["case"]["family"] == "symmetric_lle" and group["case"]["beta"] == 0.5) else 1
    )
    complete = len(runs) == 3 * expected and len(final) == expected
    checks = {
        "source_endpoint": complete
        and all("result" in r and r["result"]["source_endpoint_mu_error"] <= 1e-7 for r in final),
        "source_validity": complete
        and all(r.get("diagnostics", {}).get("valid", False) for r in final),
    }
    differences = []
    profiles = []
    for last in final:
        previous = next(r for r in runs if r["length"] == last["length"] and r["nodes"] == 80)
        if "result" not in last or "result" not in previous:
            differences.append(None)
            continue
        differences.append(abs(last["result"]["gamma"] / previous["result"]["gamma"] - 1))
        if group["case"]["family"] != "pure":
            try:
                profiles.append(compare_profiles(previous["result"], last["result"]))
            except ValueError:
                profiles.append(None)
    checks["source_convergence"] = complete and all(
        v is not None and v <= 1e-4 for v in differences
    )
    checks["source_profile_convergence"] = bool(
        profiles
        and all(
            p is not None
            and max(p["density_path_relative"], p["width_relative"], p["spatial_shape_relative"])
            <= 0.05
            for p in profiles
        )
    )
    return checks, {
        "source_tension_convergence": differences,
        "source_profile_convergence": profiles,
    }


def source_pressure_integrity(sources):
    rows = []
    for group in sources["groups"]:
        case = group["case"]
        pair = group["runs"][-1]["result"]
        model = source_model(case["family"], case["beta"])
        for side in ("left", "right"):
            density = np.asarray(pair[side])
            volume = 1e6 / sum(density)
            pressure = (
                model.pressure(case["T"], volume)
                if case["family"] == "pure"
                else model.pressure(density / sum(density), volume, case["T"])
            )
            measured = float(np.asarray(pressure).ravel()[0]) * 1e5
            error = abs(measured / pair["P"] - 1)
            rows.append(
                {
                    "group": case["name"],
                    "side": side,
                    "source_pressure_Pa": pair["P"],
                    "recomputed_pressure_Pa": measured,
                    "relative_error": error,
                    "pass": error <= 1e-7,
                }
            )
    return {
        "expected": 20,
        "evaluated": len(rows),
        "passed": sum(row["pass"] for row in rows),
        "rows": rows,
    }


def run_native(sources, directory):
    directory.mkdir(parents=True, exist_ok=False)
    progress, groups = Progress(45, "native starting"), []
    try:
        for source in sources["groups"]:
            case = source["case"]
            pure = case["family"] == "pure"
            names = PURE_CHECKS if pure else PROFILE_CHECKS
            group = {"case": case, "runs": [], "checks": dict.fromkeys(names, False)}
            try:
                source_checks, convergence = source_group_checks(source)
                group["checks"].update({k: v for k, v in source_checks.items() if k in names})
                group.update(convergence)
                model, states, influence, pair = native_pair(
                    case["family"],
                    case["T"],
                    case["P"],
                    beta=case["beta"],
                    endpoints=source["runs"][-1]["result"],
                )
                if pure:
                    grid = [
                        pair["left"][0] + x * (pair["right"][0] - pair["left"][0])
                        for x in PURE_FRACTIONS
                    ]
                    progress.stage = f"native {case['name']} pure"
                    result = pure_square_gradient_interface(
                        model, *states, influence, interior_density_grid=grid
                    )
                    group["runs"].append(result.to_dict())
                    progress.completed += 1
                    reference = source["runs"][-1]
                    group["checks"]["native_success"] = result.success
                    if result.success and "result" in reference:
                        tension_error = abs(
                            result.surface_tension_N_m / reference["result"]["gamma"] - 1
                        )
                        position_error = float(
                            max(abs(np.asarray(result.positions_m) - reference["positions"]))
                        )
                        width = reference["positions"][-2] - reference["positions"][1]
                        group.update(tension_error=tension_error, position_error_m=position_error)
                        group["checks"].update(
                            tension_reference=tension_error <= 2e-5,
                            positions_reference=position_error <= 2e-5 * width,
                        )
                    progress.report()
                else:
                    for length in case["lengths"]:
                        previous = None
                        for nodes in (257, 513, 1025):
                            progress.stage = f"native {case['name']} n={nodes} L={length:g}"
                            coordinate = np.linspace(0, length, nodes)
                            initial = initial_tanh(pair["left"], pair["right"], nodes)
                            if previous is not None:
                                initial = np.column_stack(
                                    [
                                        np.interp(
                                            coordinate,
                                            previous.positions_m,
                                            np.asarray(previous.component_densities_mol_m3)[:, i],
                                        )
                                        for i in range(len(model.component_ids))
                                    ]
                                )
                            run = {"nodes": nodes, "length": length}
                            started = monotonic()
                            try:
                                result = molecular_planar_interface(
                                    model,
                                    *states,
                                    influence,
                                    mesh=DiffusionMesh.planar(length, nodes - 1),
                                    component_density_bounds=((0.001, 14000.0),)
                                    * len(model.component_ids),
                                    initial_profile=initial,
                                    max_iterations=500,
                                    residual_tolerance=1e-7,
                                    first_integral_tolerance=0.05,
                                )
                                run["result"] = result.to_dict()
                                if len(result.positions_m) == nodes and np.all(
                                    np.isfinite(result.component_densities_mol_m3)
                                ):
                                    previous = result
                            except Exception:
                                run["exception"] = traceback.format_exc()
                                previous = None
                            run["elapsed_s"] = monotonic() - started
                            group["runs"].append(run)
                            write_new(directory / f"native-{progress.completed + 1:02d}.json", run)
                            progress.completed += 1
                            progress.report()
                    evaluate_native_profiles(group, source)
            except Exception:
                group["exception"] = traceback.format_exc()
            groups.append(group)
            write_new(directory / f"group-{len(groups):02d}.json", group)
            print(
                f"group {case['name']}: {sum(group['checks'].values())}/{len(names)} checks",
                flush=True,
            )
        summary = {
            "mode": "development",
            "expected_groups": 10,
            "completed_groups": len(groups),
            "expected_runs": 45,
            "completed_runs": progress.completed,
            "expected_checks": 116,
            "evaluated_checks": sum(len(g["checks"]) for g in groups),
            "passed_checks": sum(sum(g["checks"].values()) for g in groups),
            "groups": groups,
            "elapsed_s": monotonic() - progress.started,
        }
        summary["success"] = summary["completed_runs"] == 45 and summary["passed_checks"] == 116
        write_new(directory / "SUMMARY.json", summary)
        lines = [
            "# Parent29 G development qualification",
            "",
            f"Groups {len(groups)}/10; native runs {progress.completed}/45; "
            f"checks {summary['passed_checks']}/116; elapsed {summary['elapsed_s']:.3f}s.",
            "",
        ]
        for group in groups:
            failed = [name for name, passed in group["checks"].items() if not passed]
            lines.append(
                f"- {group['case']['name']}: "
                f"{sum(group['checks'].values())}/{len(group['checks'])}; "
                f"failed: {', '.join(failed) if failed else 'none'}"
            )
        with (directory / "SUMMARY.md").open("x", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        return summary
    finally:
        progress.close()


def evaluate_native_profiles(group, source):
    runs, checks = group["runs"], group["checks"]
    results = [r.get("result", {}) for r in runs]
    checks["all_six_profiles"] = len(results) == 6 and all(r.get("success", False) for r in results)
    checks["stationarity"] = all(
        r.get("diagnostics", {}).get("stationarity_residual_RT", 1) <= 1e-7 for r in results
    )
    checks["first_integral"] = all(
        r.get("diagnostics", {}).get("first_integral_variation_relative", 1) <= 0.05
        for r in results
    )

    def delta(a, b):
        first, second = results[a].get("surface_tension_N_m"), results[b].get("surface_tension_N_m")
        return (
            abs(first / second - 1)
            if first is not None and second is not None and second > 0
            else None
        )

    errors = [delta(2, 1), delta(5, 4), delta(5, 2)]
    group["native_tension_convergence"] = errors
    for name, error in zip(
        ("grid_convergence_L", "grid_convergence_2L", "domain_convergence"), errors, strict=True
    ):
        checks[name] = error is not None and error <= 0.02
    metrics, tensions = [], []
    for index in (2, 5):
        candidates = [
            r
            for r in source["runs"]
            if r["nodes"] == 120 and r["length"] in (None, runs[index]["length"])
        ]
        reference = candidates[0].get("result") if candidates else None
        actual = results[index]
        if reference is None or actual.get("surface_tension_N_m") is None:
            metrics.append(None)
            tensions.append(None)
            continue
        tensions.append(abs(actual["surface_tension_N_m"] / reference["gamma"] - 1))
        try:
            metrics.append(
                compare_profiles(
                    {"z": actual["positions_m"], "rho": actual["component_densities_mol_m3"]},
                    reference,
                )
            )
        except (ValueError, KeyError):
            metrics.append(None)
    group.update(reference_tension_errors=tensions, reference_profile_errors=metrics)
    checks["tension_reference"] = all(error is not None and error <= 0.03 for error in tensions)
    for name, metric, tolerance in (
        ("density_path_reference", "density_path_relative", 0.03),
        ("width_reference", "width_relative", 0.05),
        ("spatial_shape_reference", "spatial_shape_relative", 0.05),
    ):
        checks[name] = all(value is not None and value[metric] <= tolerance for value in metrics)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--source-pressure", action="store_true")
    args = parser.parse_args()
    if args.source_only:
        return int(not build_sources(args.output)["success"])
    if args.sources is None:
        parser.error("native qualification requires the fixed independent SOURCE.json")
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    if args.source_pressure:
        report = source_pressure_integrity(sources)
        write_new(args.output, report)
        return int(report["passed"] != report["expected"])
    return int(not run_native(sources, args.output)["success"])


if __name__ == "__main__":
    raise SystemExit(main())
