# ruff: noqa: B023 -- comparison callbacks execute immediately within each fixed unit.
"""Fixed 444-unit /4572-comparison CPA block."""

import argparse
import json
import math
import time
from ctypes import POINTER, byref, c_int
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import teqp
from molecular_reference import thermopack_residual_targets
from thermopack.cpa import cpa

from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL
from chemworld.physchem.saft_association import (
    AssociationSite,
    GeometricAssociationPair,
    SAFTAssociation,
)

PARAMETERS = (('H2O', .12277, 1.4515e-5, 647.3, .67359, 16655, .0692, ('H','H','O','O')),
              ('MEOH', .40531, 3.0978e-5, 512.6, .43102, 24591, .0161, ('H','O')))
CASES = [((0,), (1,)), ((1,), (1,))]+[((0, 1), x)
    for x in ((1, 0), (0, 1), (.3, .7), (.7, .3))]
ORDERS = [(i, j) for i in range(4) for j in range(4-i)]


def elliott_association(spec):
    energies = [list(row) for row in spec.energy_K]
    volumes = [list(row) for row in spec.bonding_volume_m3]
    own = {s.component_id: next((i, j) for i, left in enumerate(spec.sites)
        for j, right in enumerate(spec.sites) if left.component_id == s.component_id
        and right.component_id == s.component_id and energies[i][j] > 0)
        for s in spec.sites}
    pairs = []
    for i, left in enumerate(spec.sites):
        for j, right in enumerate(spec.sites[i+1:], i+1):
            if left.component_id != right.component_id and energies[i][j] > 0:
                pairs.append(GeometricAssociationPair((i, j),
                    (own[left.component_id], own[right.component_id])))
                energies[i][j] = energies[j][i] = volumes[i][j] = volumes[j][i] = 0
    return replace(spec, energy_K=energies, bonding_volume_m3=volumes, geometric_pairs=pairs)


def models(case, shape, contact, backend, elliott=False):
    indices, xs = CASES[case]
    params = [PARAMETERS[i] for i in indices]
    components = tuple(CPAComponent(*p[:5], 'Explicit published CPA input') for p in params)
    interactions = tuple(tuple(-.09 if i != j else 0 for j in indices) for i in indices)
    sites = [(i, k, kind) for i, p in enumerate(params) for k, kind in enumerate(p[7])]
    R = 8.31446261815324
    energies = tuple(tuple((params[i][5]+params[j][5])/(2*R) if a != b else 0
                           for j, _, b in sites) for i, _, a in sites)
    volumes = tuple(tuple((params[i][2]+params[j][2])/2*math.sqrt(params[i][6]*params[j][6])
                         /AVOGADRO_PER_MOL if a != b else 0 for j, _, b in sites)
                    for i, _, a in sites)
    association = SAFTAssociation(tuple(AssociationSite(params[i][0], str(k)) for i, k, _ in sites),
                                  energies, volumes, 'Explicit CPA CR1 pairs')
    if elliott:
        association = elliott_association(association)
    native = CPAModel(components, interactions, (200, 1000), 'Fixed CPA comparison',
                      association, shape, contact, R)
    if backend == 'native':
        return native, None, xs
    if backend == 'thermopack':
        reference = cpa(','.join(p[0] for p in params), eos='SRK')
        reference.activate()
        # The 2.2.3 Python wrapper declares c_bool pointers but passes c_int.
        # Both Fortran logical inputs use its documented default integer ABI.
        function = reference.s_set_cpa_formulation
        function.argtypes = [POINTER(c_int), POINTER(c_int)]
        function.restype = None
        function(byref(c_int(reference._true_int_value if contact == 'Kontogeorgis' else 0)),
                 byref(c_int(reference._true_int_value if elliott else 0)))
    else:
        if elliott:
            raise ValueError('teqp CPA factory does not declare Elliott strength mixing')
        spec = {'cubic': 'SRK' if shape == 'srk' else 'PR',
            'radial_dist': 'KG' if contact == 'Kontogeorgis' else 'CS', 'R_gas / J/mol/K': R,
            'kmat': [list(r) for r in interactions], 'pures': [
                {'a0i / Pa m^6/mol^2': p[1], 'bi / m^3/mol': p[2], 'Tc / K': p[3],
                 'c1': p[4], 'epsABi / J/mol': p[5], 'betaABi': p[6], 'sites': list(p[7])}
                for p in params]}
        # The shipped schema predates the factory's explicit sites/kmat fields.
        # All fields above were verified against the pinned CPA factory source.
        reference = teqp.make_model({'kind': 'CPA', 'model': spec}, validate=False)
    return native, reference, xs


def run(output, elliott=False):
    output.mkdir(parents=True, exist_ok=False)
    contacts = ('Kontogeorgis', 'Carnahan-Starling')
    grid = [('state', c, 'srk', g, t, r) for c, g, t, r in product(
        range(6), contacts, (300, 500, 800), (1, 100, 1000, 15000))]
    grid += [('inverse', c, 'srk', g, 800, 1000) for c, g in product(range(6), contacts)]
    if not elliott:
        grid += [('teqp', c, s, g, t, r) for c, s, g, t, r in product(
            range(6), ('srk', 'peng_robinson'), contacts, (300, 500, 800), (1, 100, 1000, 15000))]
    expected = sum(10+len(CASES[c][0]) if k == 'state' else 1 if k == 'inverse' else 10
                   for k, c, *_ in grid)
    assert (len(grid), expected) == ((156, 1692) if elliott else (444, 4572))
    failures, cache, metadata = [], {}, []
    count, passed, started = 0, 0, time.perf_counter()

    def compare(name, native, reference, rtol=2e-7, atol=2e-9):
        nonlocal count, passed
        record = {'quantity': name, 'rtol': rtol, 'atol': atol}
        for role, f in (('native', native), ('reference', reference)):
            try:
                value = float(f())
                if not math.isfinite(value):
                    raise ValueError('nonfinite comparison')
                record[role] = value
            except Exception as exc:
                record[role+'_exception'] = f'{type(exc).__name__}: {exc}'
        record['passed'] = ('native' in record and 'reference' in record and math.isclose(
            record['native'], record['reference'], rel_tol=rtol, abs_tol=atol))
        count += 1
        passed += record['passed']
        return record

    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for index, (kind, case, shape, contact, t, rho) in enumerate(grid, 1):
            key = case, shape, contact, 'teqp' if kind == 'teqp' else 'thermopack'
            if key not in cache:
                cache[key] = models(*key, elliott=elliott)
                metadata.append({'case': case, 'backend': key[-1],
                                 'model': cache[key][0].to_dict()})
            native, ref, xs = cache[key]
            records = []
            if kind == 'teqp':
                for i, j in ORDERS:
                    records.append(compare(f'Ar{i}{j}', lambda: molecular_scaled_derivative(
                        native, t, rho, xs, inverse_temperature_order=i, density_order=j),
                        lambda: ref.get_Arxy(i, j, t, rho, np.asarray(xs, dtype=float))))
            else:
                actual_error = target_error = None
                try:
                    state = molecular_eos_state(native, t, rho, xs, require_stable=False)
                    actual = {**state.__dict__, **{f'mu_{i}': mu for i, mu in enumerate(
                        molecular_residual_chemical_potentials(native, t, rho, xs))}}
                except Exception as exc:
                    actual_error = exc
                try:
                    targets, mus = thermopack_residual_targets(ref, t, rho, xs)
                    targets.update({f'mu_{i}': mu for i, mu in enumerate(mus)})
                except Exception as exc:
                    target_error = exc

                def value(name, upstream=False):
                    error = target_error if upstream else actual_error
                    if error:
                        raise error
                    return (targets if upstream else actual)[name]

                if kind == 'inverse':
                    records.append(compare('pressure_density_inverse', lambda:
                        molecular_density_from_pressure(native, t, value('pressure_Pa', True), xs,
                            initial_density_mol_m3=1.1*rho,
                            density_bounds_mol_m3=(.5*rho, 1.5*rho)).molar_density_mol_m3,
                        lambda: rho, rtol=2e-9))
                else:
                    names = ['pressure_Pa', 'compressibility_factor', 'residual_helmholtz_J_mol',
                        'residual_internal_energy_J_mol', 'residual_enthalpy_J_mol',
                        'residual_entropy_J_mol_K', 'residual_cv_J_mol_K', 'residual_cp_J_mol_K',
                        'dP_dT_rho_Pa_K', 'dP_drho_T_Pa_m3_mol']+[f'mu_{i}' for i in range(len(xs))]
                    for name in names:
                        atol = (1e-4 if name == 'pressure_Pa' else 2e-9 if name ==
                            'compressibility_factor' else 2e-5 if name.endswith('_J_mol')
                            or name.startswith('mu_') else 2e-7)
                        records.append(compare(name, lambda: value(name),
                                               lambda: value(name, True), atol=atol))
            unit = {'unit': index, 'kind': kind, 'case': case, 'shape': shape, 'contact': contact,
                    'T_K': t, 'rho_mol_m3': rho, 'comparisons': records}
            stream.write(json.dumps(unit)+'\n')
            failures.extend({**{k: v for k, v in unit.items() if k != 'comparisons'}, **r}
                            for r in records if not r['passed'])
            if index % 12 == 0:
                elapsed = time.perf_counter()-started
                print(f'CPA {index}/{len(grid)}, {index/elapsed:.2f} units/s, '
                      f'ETA {(len(grid)-index)*elapsed/index:.1f}s; failures {len(failures)}',
                      flush=True)
    summary = {'mode': 'development', 'elliott': elliott, 'units': len(grid), 'comparisons': count,
        'expected_comparisons': expected, 'passed': passed, 'failed': len(failures),
        'failures': failures, 'model_inputs': metadata,
        'elapsed_seconds': time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('failures', 'model_inputs')}))
    assert count == expected
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--elliott', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.elliott) else 1)
