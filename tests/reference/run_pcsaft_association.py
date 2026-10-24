# ruff: noqa: B023 -- comparison callbacks execute synchronously in each unit.
"""Fixed 384-unit association/simplified PC-SAFT source comparison."""

import argparse
import json
import math
import time
from importlib.metadata import version
from itertools import product
from pathlib import Path

from molecular_reference import thermopack_residual_targets
from thermopack.pcsaft import pcsaft

from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
)
from chemworld.physchem.pcsaft import PCSAFTModel, SAFTSegment
from chemworld.physchem.saft_association import AssociationSite, SAFTAssociation

CASES = [((c,), (1,), ref) for c, ref in (
    ("MEOH", "Default"), ("H2O", "Default"), ("ETOH", "Default"),
    ("H2O", "Rehner2020_3B_polar"))]
CASES += [(('MEOH', 'H2O'), x, 'Default') for x in ((1, 0), (0, 1), (.3, .7), (.7, .3))]
CASES += [(('ETOH', 'C1'), x, 'Default') for x in ((1, 0), (.2, .8))]
CASES += [(('MEOH', 'H2O', 'C1'), x, 'Default') for x in ((.2, .3, .5), (0, .4, .6))]
CASES += [(('ETOH',), (1,), 'Gross_Sadowski2002'),
          (('H2O',), (1,), 'Kontogeorgis_Folas2010')]


def models(case, simplified):
    ids, xs, ref = CASES[case]
    upstream = pcsaft(','.join(ids), parameter_reference=ref, simplified=simplified, polar=False)
    params = [upstream.get_pure_fluid_param(i+1) for i in range(len(ids))]
    interactions = tuple(tuple(.02 if i != j else 0 for j in ids) for i in ids)
    for i in range(len(ids)):
        for j in range(i+1, len(ids)):
            upstream.set_kij(i+1, j+1, .02)
            upstream.set_kij(j+1, i+1, .02)
    # Explicit labelled sites and polarity for each selected source topology.
    polarities = {'MEOH': (0, -1), 'H2O': (-1, -1, 1, 1), 'ETOH': (0, -1), 'C1': ()}
    if ref == 'Rehner2020_3B_polar':
        polarities['H2O'] = (1, 1, -1)
    if ref in ('Gross_Sadowski2002', 'Kontogeorgis_Folas2010'):
        polarities[ids[0]] = (-1, 1)
    sites = [(i, k, p) for i, c in enumerate(ids) for k, p in enumerate(polarities[c])]
    assert upstream.get_n_assoc_sites() == len(sites)
    energy, volume = [], []
    for i, _, p in sites:
        erow, vrow = [], []
        for j, _, q in sites:
            enabled = p*q <= 0  # bipolar site associates with either polarity, including itself
            erow.append((params[i][3]+params[j][3])/(2*upstream.Rgas) if enabled else 0)
            vrow.append(math.sqrt(params[i][4]*params[j][4])*(params[i][1]*params[j][1])**1.5
                        if enabled else 0)
        energy.append(tuple(erow))
        volume.append(tuple(vrow))
    spec = SAFTAssociation(tuple(AssociationSite(ids[i], str(k)) for i, k, _ in sites),
                           tuple(energy), tuple(volume), 'Explicit fixed ThermoPack pair inputs')
    native = PCSAFTModel(tuple(SAFTSegment(c, *par[:3], 'ThermoPack '+ref)
                               for c, par in zip(ids, params, strict=True)),
                        interactions, (200, 1000), 'Fixed association comparison',
                        gas_constant_J_mol_K=upstream.Rgas, association=spec, simplified=simplified)
    metadata = {"components": ids, "fractions": xs, "parameter_reference": ref,
                    "parameters": params,
                    "sites": sites, "gas_constant": upstream.Rgas, "simplified": simplified}
    return native, upstream, xs, metadata


def scalar(value):
    return float(value[0] if isinstance(value, tuple) else value)


def run(output, two_site_block=False):
    output.mkdir(parents=True, exist_ok=False)
    cases = (12, 13) if two_site_block else range(12)
    grid = [('state', c, s, t, r) for c, s, t, r in product(
        cases, (False, True), (300, 500, 800), (1e-6, 1, 1000, 10000, 30000))]
    grid += [('inverse', c, s, 800, 1000) for c, s in product(cases, (False, True))]
    expected = sum(10+len(CASES[c][0]) if k == 'state' else 1 for k, c, *_ in grid)
    assert (len(grid), expected) == ((64, 664) if two_site_block else (384, 4284))
    cache, failures, metadata = {}, [], []
    comparisons, passed, started = 0, 0, time.perf_counter()

    def compare(name, native, reference, rtol=2e-7, atol=2e-7):
        nonlocal comparisons, passed
        record = {"quantity": name, "rtol": rtol, "atol": atol}
        for role, function in (('native', native), ('reference', reference)):
            try:
                value = float(function())
                if not math.isfinite(value):
                    raise ValueError('nonfinite output')
                record[role] = value
            except Exception as exc:
                record[role+'_exception'] = f'{type(exc).__name__}: {exc}'
        record['passed'] = ('native' in record and 'reference' in record and math.isclose(
            record['native'], record['reference'], rel_tol=rtol, abs_tol=atol))
        comparisons += 1
        passed += record['passed']
        return record

    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for index, (kind, case, simplified, t, rho) in enumerate(grid, 1):
            key = case, simplified
            if key not in cache:
                cache[key] = models(*key)
                metadata.append(cache[key][3])
            model, ref, xs, _ = cache[key]
            state_error = reference_error = None
            try:
                state = molecular_eos_state(model, t, rho, xs, require_stable=False)
            except Exception as exc:
                state_error = exc
            try:
                targets, an = thermopack_residual_targets(ref, t, rho, xs)
            except Exception as exc:
                reference_error = exc

            def native_value(name):
                if state_error:
                    raise state_error
                return getattr(state, name)

            def reference_value(name):
                if reference_error:
                    raise reference_error
                return targets[name]

            records = []
            if kind == 'inverse':
                records.append(compare('pressure_density_inverse', lambda:
                    molecular_density_from_pressure(model, t, reference_value('pressure_Pa'), xs,
                        initial_density_mol_m3=rho*1.1,
                        density_bounds_mol_m3=(rho*.5, rho*1.5)).molar_density_mol_m3,
                    lambda: rho, rtol=2e-9, atol=2e-9))
            else:
                names = ('pressure_Pa', 'compressibility_factor', 'residual_helmholtz_J_mol',
                    'residual_internal_energy_J_mol', 'residual_enthalpy_J_mol',
                    'residual_entropy_J_mol_K', 'residual_cv_J_mol_K', 'residual_cp_J_mol_K',
                    'dP_dT_rho_Pa_K', 'dP_drho_T_Pa_m3_mol')
                for name in names:
                    atol = (1e-4 if name == 'pressure_Pa' else 2e-9 if name ==
                            'compressibility_factor' else 2e-5 if name.endswith('_J_mol') else 2e-7)
                    records.append(compare(name, lambda: native_value(name),
                                           lambda: reference_value(name), atol=atol))

                def native_mu(i):
                    return molecular_residual_chemical_potentials(model, t, rho, xs)[i]

                def reference_mu(i):
                    if reference_error:
                        raise reference_error
                    return an[i]

                for i in range(len(xs)):
                    records.append(compare(f'mu_residual_{i}', lambda: native_mu(i),
                                           lambda: reference_mu(i), atol=2e-5))
            unit = {"unit": index, "kind": kind, "case": case, "simplified": simplified,
                        "T_K": t, "rho_mol_m3": rho, "comparisons": records}
            stream.write(json.dumps(unit)+'\n')
            failures.extend({**{k: v for k, v in unit.items() if k != 'comparisons'}, **r}
                            for r in records if not r['passed'])
            if index % 12 == 0 or index == len(grid):
                elapsed = time.perf_counter()-started
                print(f'Association {index}/{len(grid)}, {index/elapsed:.2f} units/s, '
                      f'ETA {(len(grid)-index)*elapsed/index:.1f}s; '
                      f'failures {len(failures)}', flush=True)
    summary = {"mode": 'development', "reference": 'ThermoPack '+version('thermopack'),
        "units": len(grid), "comparisons": comparisons,
        "expected_comparisons": expected, "passed": passed,
        "failed": len(failures), "failures": failures, "model_inputs": metadata,
        "elapsed_seconds": time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('failures', 'model_inputs')}))
    assert comparisons == expected
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--two-site-block', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.two_site_block) else 1)
