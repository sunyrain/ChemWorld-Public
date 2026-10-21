"""Fixed full-source 21-phase MQMQA response qualification; no native reference imports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import time
import traceback
from collections import Counter
from copy import deepcopy
from itertools import combinations_with_replacement, product
from pathlib import Path


def reference(directory, output):
    import numpy as np
    import pycalphad
    import pycalphad.variables as v
    import symengine as se
    from pycalphad.io.cs_dat import parse_cs_dat, rename_element_charge
    from pycalphad.models.model_mqmqa import ModelMQMQA, _get_species

    if pycalphad.__version__ != "0.11.2":
        raise ValueError("reference must be pycalphad 0.11.2")
    report = {"expected_phases": driver.PHASE_COUNT, "expected_points": driver.POINT_COUNT,
              "units": [],
              "dependencies": dict(sorted((d.metadata["Name"], d.version)
                                          for d in importlib.metadata.distributions()))}
    started = time.perf_counter()
    for name in driver.FILES:
        path = directory/name
        header, solutions, pure = parse_cs_dat(path.read_text(encoding="utf-8-sig").upper())
        database = pycalphad.Database(str(path))
        originals = (*solutions, *pure)
        counts, seen = Counter(p.phase_name for p in originals), Counter()
        for raw in originals:
            seen[raw.phase_name] += 1
            if raw.phase_type not in {"SUBG", "SUBQ"}:
                continue
            phase = (raw.phase_name if counts[raw.phase_name] == 1 else
                     f"{raw.phase_name}#OCCURRENCE{seen[raw.phase_name]}")
            unit = {"database": name, "phase": phase, "source_label": raw.phase_name,
                    "points": [], "failures": [],
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            report["units"].append(unit)
            try:
                selected = database
                if phase != raw.phase_name:
                    selected = deepcopy(database)
                    occurrence = deepcopy(raw)
                    occurrence.phase_name = phase
                    occurrence.insert(selected, header.pure_elements,
                                      header.gibbs_coefficient_idxs,
                                      header.excess_coefficient_idxs)
                    selected.process_parameter_queue()
                components = sorted(selected.elements)
                if any("VA" in s.constituents for s in selected.species) and "VA" not in components:
                    components.append("VA")
                model = ModelMQMQA(selected, components, phase)
                species = {s.name: s for s in (*model.cations, *model.anions)}
                names = [rename_element_charge(s, q) for s, q in zip(
                    raw.subl_1_const, raw.subl_1_charges, strict=True)]
                names += [rename_element_charge(s, -q) for s, q in zip(
                    raw.subl_2_const, raw.subl_2_charges, strict=True)]
                ions = [species[n] for n in names]
                nc = len(raw.subl_1_const)
                quads = [(*c, *a) for c, a in product(
                    combinations_with_replacement(range(nc), 2),
                    combinations_with_replacement(range(nc, len(ions)), 2))]
                labels = ["Q"+"_".join(str(i+1) for i in quad) for quad in quads]
                variables = [v.Y(phase, 0, _get_species(*(ions[i] for i in quad)))
                             for quad in quads]
                unit["constituents"] = [sorted(labels)]
                unit["quadruplet_source_indices"] = quads
                elements = sorted({e for ion in ions for e in ion.constituents
                                   if e in header.pure_elements and e not in {"VA", "/-"}})
                charge = sum(ion.charge*model._n_i(selected, ion) for ion in ions)
                rows = {"G": model.GM, **{f"element:{e}": model.moles(e) for e in elements},
                        "site_sum:0": model.get_internal_constraints()[0], "charge": charge}
                axes = [("temperature", v.T), ("pressure", v.P),
                        *(("site:0:"+label, variable)
                          for label, variable in zip(labels, variables, strict=True))]
                expressions = {"H": model.HM, "S": model.SM, "Cp_fixed": model.CPM}
                for label, expression in rows.items():
                    prefix = "response:"+label
                    expressions[prefix] = expression
                    for axis, variable in axes:
                        first = expression.diff(variable)
                        expressions[prefix+"|"+axis] = first
                        for other_axis, other in axes:
                            expressions[prefix+"|"+axis+"|"+other_axis] = first.diff(other)
                for quad, label in zip(quads, labels, strict=True):
                    for position, ion in enumerate(quad):
                        expressions[f"coordination:{label}:{position}"] = se.Float(
                            model.Z(selected, ions[ion], *(ions[i] for i in quad)))
                for i, ion in enumerate(ions):
                    expressions[f"ion_n:{i+1}"] = model._n_i(selected, ion)
                for label, key in (("reference", "ref"), ("ideal", "idmix"), ("excess", "xsmix")):
                    expressions["contribution:"+label] = model.models[key]/model.normalization
                evaluator = se.Lambdify([x for _, x in axes], list(expressions.values()),
                                        cse=True, backend="llvm")
                upper_limits = []
                for member in raw.endmembers:
                    previous = .01
                    for interval in member.intervals:
                        upper = 10000. if previous == interval.T_max else interval.T_max
                        previous = interval.T_max
                    upper_limits.append(upper)
                low, high = 300., min(2000., *upper_limits)
                if low >= high:
                    raise ValueError("no required common temperature interval")
                for tilted in (False, True):
                    weights = (np.arange(1, len(labels)+1, dtype=float) if tilted else
                               np.ones(len(labels)))
                    fractions = weights/weights.sum()
                    sites = [dict(zip(labels, map(float, fractions), strict=True))]
                    for fraction in (.2, .5, .8):
                        t = low+(high-low)*fraction
                        values = np.asarray(evaluator([t, driver.PRESSURE, *fractions]),
                                            dtype=float).ravel()
                        if len(values) != len(expressions) or not np.all(np.isfinite(values)):
                            raise ValueError("reference produced incomplete/nonfinite responses")
                        unit["points"].append({"T": t, "profile": "tilted" if tilted else "uniform",
                            "sites": sites, "values": dict(zip(expressions, map(float, values),
                                                               strict=True)), "failures": []})
                        driver.write(output, report)
            except Exception:
                unit["failures"].append(traceback.format_exc())
            driver.write(output, report)
            elapsed = max(1e-9, time.perf_counter()-started)
            completed = len(report["units"])
            print(f"reference MQMQA {completed}/{driver.PHASE_COUNT} "
                  f"rate={completed/elapsed:.2f}/s "
                  f"eta={(driver.PHASE_COUNT-completed)*elapsed/completed:.1f}s {phase}",
                  flush=True)


def native_model(database, phase, components):
    from chemworld.physchem.mqmqa import MQMQAPhaseModel

    return MQMQAPhaseModel(database, phase, components=components, activate_suppressed=True)


def native_values(model, sites, temperature):
    values, diagnostics = ordinary_values(model, sites, temperature)
    topology = model._topology
    for quad, label in zip(topology.quadruplets, topology.labels, strict=True):
        for position, ion in enumerate(quad):
            values[f"coordination:{label}:{position}"] = topology.z(ion, quad)
    for i, n in topology.ion_amounts(sites[0]).items():
        values[f"ion_n:{i+1}"] = float(n)
    p = model.evaluate(sites, temperature, driver.PRESSURE)
    values.update({"contribution:"+key: value for key, value in p.contributions_J_mol.items()})
    return values, diagnostics


def tolerance(key, expected):
    if key.startswith(("coordination:", "contribution:")):
        return 1e-8+1e-8*abs(expected)
    return ordinary_tolerance(key, expected)


def configure_driver(entrypoint):
    global driver, ordinary_values, ordinary_tolerance

    spec = importlib.util.spec_from_file_location("common_dat_response_driver",
        Path(__file__).with_name("run_chem_sage_cef_block.py"))
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    ordinary_values, ordinary_tolerance = driver.native_values, driver.tolerance
    driver.__file__ = entrypoint
    driver.PHASE_COUNT, driver.POINT_COUNT = 21, 126
    driver.reference, driver.native_model = reference, native_model
    driver.native_values, driver.tolerance = native_values, tolerance
    return driver


if __name__ == "__main__":
    raise SystemExit(configure_driver(__file__).main())
