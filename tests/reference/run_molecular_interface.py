"""Five fixed interface units; analytic and optional PhasePy oracles only here."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from math import isfinite, log, sqrt
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import numpy as np

from chemworld.physchem.caloric_math import logarithm, scalar_value
from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.molecular_flash import MolecularDensityBasin, molecular_tp_flash
from chemworld.physchem.molecular_interface import (
    SquareGradientInfluence,
    molecular_planar_interface,
    pure_square_gradient_interface,
)


@dataclass(frozen=True)
class QuarticModel:
    left: tuple[float, ...] = (1000.,)
    right: tuple[float, ...] = (5000.,)
    amplitudes: tuple[float, ...] = (1e-8,)
    gas_constant_J_mol_K: float = 8.31446261815324

    @property
    def component_ids(self):
        return tuple(f"component_{i}" for i in range(len(self.left)))

    def alphar(self, temperature, density, fractions):
        c = tuple(density*x for x in fractions)
        f = sum(a*(x-lo)**2*(hi-x)**2 for x, lo, hi, a in
                zip(c, self.left, self.right, self.amplitudes, strict=True))-1e5
        ideal = sum(x*(logarithm(x)-1) for x in c if scalar_value(x) > 0)
        return (f/(self.gas_constant_J_mol_K*temperature)-ideal)/density


def analytic_pair(binary=False):
    model = (QuarticModel((1000., 2000.), (5000., 6000.), (1e-8, 2e-8))
             if binary else QuarticModel())
    states = tuple(molecular_eos_state(model, 300, sum(c), tuple(v/sum(c) for v in c))
                   for c in (model.left, model.right))
    n = len(model.component_ids)
    influence = SquareGradientInfluence(model.component_ids,
        tuple(tuple(1e-18 if i == j else 0 for j in range(n)) for i in range(n)),
        (200, 400), "explicit analytic interface fixture")
    gamma = sum(sqrt(2*a*1e-18)*(hi-lo)**3/6 for a, lo, hi in
                zip(model.amplitudes, model.left, model.right, strict=True))
    return model, states, influence, gamma


def source_model(binary=False):
    import phasepy

    data = (("propane", 369.83, 42.48, .152, 1e-20),
            ("butane", 425.12, 37.96, .2, 1.5e-20))[:2 if binary else 1]
    components = [phasepy.component(name=name, Tc=tc, Pc=pc, w=w, cii=[c])
                  for name, tc, pc, w, c in data]
    source = phasepy.preos(components[0]+components[1] if binary else components[0])
    if binary:
        source.beta_sgt(np.array([[0., .5], [.5, 0.]]))
    native = CPAModel(tuple(CPAComponent(name, source.oma*8.314**2*tc**2/(pc*1e5),
        source.omb*8.314*tc/(pc*1e5), tc, float(np.asarray(source.k).ravel()[i]),
        "PhasePy public PR constants, explicit reference fixture")
        for i, (name, tc, pc, _, _) in enumerate(data)),
        tuple(tuple(0. for _ in data) for _ in data), (200, 500), "same PR fixture",
        cubic_model="peng_robinson", gas_constant_J_mol_K=8.314)
    coefficients = ((1e-20, .5*sqrt(1e-20*1.5e-20)),
                    (.5*sqrt(1e-20*1.5e-20), 1.5e-20)) if binary else ((1e-20,),)
    influence = SquareGradientInfluence(native.component_ids, coefficients, (200, 500),
                                         "explicit source cii, beta=.5 for binary")
    return native, source, influence


def source_pair(temperature, binary=False):
    model, source, influence = source_model(binary)
    if binary:
        result = molecular_tp_flash(model, temperature, 6e5, {"propane": .5, "butane": .5},
            basins=(MolecularDensityBasin("vapor", (1, 2000), 250),
                    MolecularDensityBasin("liquid", (7500, 13500), 10000)))
        if not result.success:
            raise ValueError(f"source reference endpoint flash failed: {result.to_dict()}")
        states = tuple(p.state for p in sorted(result.phases,
                       key=lambda p: p.state.molar_density_mol_m3))
        pressure = 6e5
    else:
        psat, vl, vv = (float(np.asarray(v).ravel()[0]) for v in source.psat(temperature))
        pressure = psat*1e5
        states = tuple(molecular_eos_state(model, temperature, 1e6/v, (1,)) for v in (vv, vl))
    return model, source, influence, states, pressure


def main():
    from phasepy.sgt import sgt_mix, sgt_pure

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinement-seed", type=Path)
    args = parser.parse_args()
    if args.refinement_seed is not None:
        return refinement(args.refinement_seed, args.output)
    started = monotonic()
    units = []
    for name in ("analytic_pure", "analytic_binary", "PR_propane_280", "PR_propane_300",
                 "PR_binary_300"):
        unit = {"name": name, "comparisons": [], "runs": []}
        try:
            binary = "binary" in name
            if name.startswith("analytic"):
                model, states, influence, expected = analytic_pair(binary)
                length, bounds = 40e-9, ((100., 8000.),)*len(model.component_ids)
            else:
                temperature = 280 if name.endswith("280") else 300
                model, source, influence, states, pressure = source_pair(temperature, binary)
                if binary:
                    densities = [np.asarray(s.mole_fractions)*s.molar_density_mol_m3/1e6
                                 for s in states]
                    reference = sgt_mix(*densities, temperature, pressure/1e5, source,
                        rho0="hyperbolic", n=40, itmax=10, full_output=True)
                    expected = float(reference.tension)*1e-3
                    unit["source_summary"] = {k: v.tolist() if isinstance(v, np.ndarray) else v
                        for k, v in dict(reference).items() if k not in ("rho", "z", "GPT")}
                else:
                    with warnings.catch_warnings(record=True) as caught:
                        reference = sgt_pure(states[0].molar_density_mol_m3/1e6,
                            states[1].molar_density_mol_m3/1e6, temperature, pressure/1e5,
                            source, n=100)
                    unit["source_warnings"] = [str(w.message) for w in caught]
                    expected = float(np.asarray(reference).ravel()[0])*1e-3
                if not isfinite(expected):
                    raise ValueError("nonfinite PhasePy tension")
                length, bounds = 12e-9, ((.01, 7000.),)*len(model.component_ids)
            unit["reference_tension_N_m"] = expected
            if binary:
                for extent in (length, 2*length):
                    for nodes in (33, 65, 129):
                        result = molecular_planar_interface(model, *states, influence,
                            mesh=DiffusionMesh.planar(extent, nodes-1),
                            component_density_bounds=bounds, max_iterations=500)
                        unit["runs"].append(result.to_dict())
                        print(f"Interface unit {len(units)+1}/5 {name}; grid run "
                              f"{len(unit['runs'])}/6 nodes={nodes} L={extent:g}; "
                              f"success={result.success}; elapsed={monotonic()-started:.1f}s",
                              flush=True)
                last = unit["runs"][-1]
                values = [r["surface_tension_N_m"] for r in unit["runs"]]
                checks = [("all_six_profiles", all(r["success"] for r in unit["runs"])),
                    ("tension_reference", values[-1] is not None and
                     abs(values[-1]/expected-1) <= (.02 if name.startswith("analytic") else .03)),
                    ("length_convergence", values[2] is not None and values[5] is not None
                     and abs(values[5]/values[2]-1) <= .02),
                    ("grid_convergence_L", values[1] is not None and values[2] is not None
                     and abs(values[2]/values[1]-1) <= .02),
                    ("grid_convergence_2L", values[4] is not None and values[5] is not None
                     and abs(values[5]/values[4]-1) <= .02),
                    ("stationarity", last["diagnostics"].get("stationarity_residual_RT", 1) <=1e-5),
                    ("first_integral", last["diagnostics"].get(
                        "first_integral_variation_relative", 1) <=.05)]
            else:
                lo, hi = sorted(s.molar_density_mol_m3 for s in states)
                grid = [lo+(hi-lo)*p for p in (.01, .1, .25, .5, .75, .9, .99)]
                result = pure_square_gradient_interface(model, *states, influence,
                                                          interior_density_grid=grid)
                unit["runs"].append(result.to_dict())
                checks = [("success", result.success), ("tension_reference",
                    result.surface_tension_N_m is not None and
                    abs(result.surface_tension_N_m/expected-1) <= (
                        2e-8 if name.startswith("analytic") else 2e-5))]
                if name.startswith("analytic") and result.success:
                    positions = [log((v-lo)/(hi-v))/((hi-lo)*sqrt(2*1e-8/1e-18)) for v in grid]
                    checks.append(("analytic_positions", max(abs(a-b) for a, b in zip(
                        positions, result.positions_m, strict=True)) <=2e-5*length))
            unit["comparisons"] = [{"name": k, "pass": bool(v)} for k, v in checks]
        except Exception as exc:
            unit["exception"] = f"{type(exc).__name__}: {exc}"
            unit["comparisons"] = [{"name": "unit_execution", "pass": False}]
        units.append(unit)
        print(f"Interface unit {len(units)}/5 {name}; "
              f"passed={sum(c['pass'] for c in unit['comparisons'])}/"
              f"{len(unit['comparisons'])}; elapsed={monotonic()-started:.1f}s", flush=True)
        report = {"physical_units": len(units), "fixed_expected_units": 5,
                  "grid_runs": sum(len(u['runs']) for u in units), "units": units}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False,
            default=lambda value: value.item() if isinstance(value, np.generic)
            else value.tolist())+"\n", encoding="utf-8")
    return int(any(not c["pass"] for u in units for c in u["comparisons"]))


def refinement(seed_path, output_path):
    original = json.loads(seed_path.read_text(encoding="utf-8"))
    unit = next(u for u in original["units"] if u["name"] == "PR_binary_300")
    model, _source, influence, states, _pressure = source_pair(300, True)
    runs, started = [], monotonic()
    finished = Event()

    def heartbeat():
        while not finished.wait(30):
            elapsed = monotonic()-started
            print(f"Interface refinement alive; completed={len(runs)}/6; "
                  f"elapsed={elapsed:.1f}s; rate={len(runs)/max(elapsed, 1e-6):.4f}/s", flush=True)

    Thread(target=heartbeat, daemon=True).start()
    report = {"scope": "separately_declared_fixed_resolution_supplement",
              "expected_runs": 6, "runs": runs,
              "reference_tension_N_m": unit["reference_tension_N_m"]}
    for length, seed_index in ((12e-9, 2), (24e-9, 5)):
        previous = unit["runs"][seed_index]
        for nodes in (257, 513, 1025):
            mesh = DiffusionMesh.planar(length, nodes-1)
            old = np.asarray(previous["component_densities_mol_m3"])
            seed = np.column_stack([np.interp(mesh.faces_m, previous["positions_m"], old[:, i])
                                    for i in range(2)])
            result = molecular_planar_interface(model, *states, influence, mesh=mesh,
                component_density_bounds=((.01, 7000.),)*2, initial_profile=seed,
                max_iterations=500)
            value = result.to_dict()
            runs.append(value)
            if value["positions_m"]:
                previous = value
            elapsed = monotonic()-started
            print(f"Interface refinement {len(runs)}/6 nodes={nodes} L={length:g}; "
                  f"success={result.success}; elapsed={elapsed:.1f}s; "
                  f"rate={len(runs)/max(elapsed, 1e-6):.3f}/s; "
                  f"ETA={(6-len(runs))*elapsed/len(runs):.1f}s", flush=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                                   encoding="utf-8")
    values = [v["surface_tension_N_m"] for v in runs]
    expected = unit["reference_tension_N_m"]
    report["comparisons"] = [
        {"name": "all_six_resolved", "pass": all(r["success"] for r in runs)},
        {"name": "finest_source_L", "pass": values[2] is not None
         and abs(values[2]/expected-1) <=.03},
        {"name": "finest_source_2L", "pass": values[5] is not None
         and abs(values[5]/expected-1) <=.03},
        {"name": "last_grid_L", "pass": values[1] is not None and values[2] is not None
         and abs(values[2]/values[1]-1) <=.02},
        {"name": "last_grid_2L", "pass": values[4] is not None and values[5] is not None
         and abs(values[5]/values[4]-1) <=.02},
        {"name": "finest_lengths", "pass": values[2] is not None and values[5] is not None
         and abs(values[5]/values[2]-1) <=.02}]
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    finished.set()
    return int(any(not c["pass"] for c in report["comparisons"]))


if __name__ == "__main__":
    raise SystemExit(main())
