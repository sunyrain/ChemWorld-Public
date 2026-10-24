# ruff: noqa: B023 -- callbacks execute before advancing each fixed unit.
"""Fixed Mie core, ThermoPack and full teqp-polar comparison blocks."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
import teqp
from molecular_reference import thermopack_residual_targets
from thermopack.saftvrmie import saftvrmie

from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.saft_polar import SAFTMultipole, SAFTPolar
from chemworld.physchem.saftvrmie import MieSegment, SAFTVRMieModel

PARAMETERS = (('C1', 1, 3.7412e-10, 153.36, 12.650),
              ('C2', 1.4373, 3.7257e-10, 206.12, 12.400),
              ('C3', 1.6845, 3.9056e-10, 239.89, 13.006),
              ('C4', 3.1, 4e-10, 320, 20))
CASES = [((i,), (1,)) for i in range(4)]+[
    ((0, 3), x) for x in ((1, 0), (0, 1), (.3, .7), (.7, .3))]+[
    ((0, 1, 3), x) for x in ((.2, .3, .5), (0, .4, .6))]
ORDERS = [(i, j) for i in range(4) for j in range(4-i)]
NAMES = ('pressure_Pa', 'compressibility_factor', 'residual_helmholtz_J_mol',
         'residual_internal_energy_J_mol', 'residual_enthalpy_J_mol',
         'residual_entropy_J_mol_K', 'residual_cv_J_mol_K', 'residual_cp_J_mol_K',
         'dP_dT_rho_Pa_K', 'dP_drho_T_Pa_m3_mol')


def models(case, mixing, backend, polar):
    ids, xs = CASES[case]
    cs = tuple(MieSegment(*PARAMETERS[i][:4], 'Fixed Mie input', PARAMETERS[i][4]) for i in ids)
    interaction = tuple(tuple(.02 if i != j else 0 for j in ids) for i in ids)
    spec = None
    if polar:
        moments = {0: ((1.1, .8),), 6: ((1.1, 0), (0, .8)),
                   8: ((1.1, 0), (0, .8), (.3, .5))}[case]
        spec = SAFTPolar(tuple(SAFTMultipole(c.component_id, *p, 'Fixed synthetic moments')
            for c, p in zip(cs, moments, strict=True)), 'Fixed teqp convention', 'teqp-0.23.2')
    native = SAFTVRMieModel(cs, interaction, (100, 1000), 'Fixed Mie comparison',
                            energy_mixing=mixing, polar=spec)
    if backend == 'thermopack':
        ref = saftvrmie(','.join('NC4' if c.component_id == 'C4' else c.component_id for c in cs))
        for i, c in enumerate(cs, 1):
            ref.set_pure_fluid_param(i, c.segment_count, c.diameter_m, c.well_depth_K,
                                    c.attractive_exponent, c.repulsive_exponent)
        for i in range(len(cs)):
            for j in range(i+1, len(cs)):
                ref.set_eps_kij(i+1, j+1, .02)
                ref.set_sigma_lij(i+1, j+1, 0.)
                ref.set_lr_gammaij(i+1, j+1, 0.)
        ref.set_hard_sphere_reference('LAFITTE')
    else:
        coeffs = []
        for i, c in enumerate(cs):
            p = spec.moments[i] if spec else None
            coeffs.append({'name': c.component_id, 'm': c.segment_count,
                'sigma_m': c.diameter_m, 'epsilon_over_k': c.well_depth_K,
                'lambda_a': c.attractive_exponent, 'lambda_r': c.repulsive_exponent,
                'BibTeXKey': 'fixed-input', '(mu^*)^2': p.reduced_dipole_squared if p else 0,
                '(Q^*)^2': p.reduced_quadrupole_squared if p else 0,
                'nmu': 1 if p else 0, 'nQ': 1 if p else 0})
        # The factory processes energy flags via its polar-capable path; explicit
        # zero moments/counts activate that path without adding polar energy.
        # The shipped schema excludes these documented factory fields. Source
        # v0.23.2 verifies every key; disable only the reference JSON schema.
        ref = teqp.make_model({'kind': 'SAFT-VR-Mie', 'model': {'coeffs': coeffs,
            'kmat': [list(r) for r in interaction], 'polar_model': 'GrossVrabec',
            'SAFTVRMie_flags': {'epsilon_ij': mixing}}}, validate=False)
    return native, ref, np.asarray(xs, dtype=float)


def teqp_targets(ref, t, rho, x):
    a = {(i, j): ref.get_Arxy(i, j, t, rho, x) for i, j in
         ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0))}
    R, z = ref.get_R(x), 1+a[0, 1]
    stiffness, thermal = 1+2*a[0, 1]+a[0, 2], z-a[1, 1]
    return dict(zip(NAMES, (rho*R*t*z, z, R*t*a[0, 0], R*t*a[1, 0],
        R*t*(a[1, 0]+z-1), R*(a[1, 0]-a[0, 0]), -R*a[2, 0],
        R*(-a[2, 0]+thermal**2/stiffness-1), rho*R*thermal, R*t*stiffness), strict=True))


def run(output, block):
    output.mkdir(parents=True, exist_ok=False)
    backend = 'thermopack' if block == 'thermopack' else 'teqp'
    polar = block == 'polar'
    cases = (0, 6, 8) if polar else range(10)
    mixtures = ('Lafitte',) if backend == 'thermopack' else ('Lafitte', 'Lorentz-Berthelot')
    if backend == 'thermopack':
        grid = [('state', c, m, t, r) for c, m, t, r in product(
            cases, mixtures, (180, 300, 600), (1, 1000, 8000, 16000))]
        grid += [('inverse', c, 'Lafitte', 600, 1000) for c in cases]
    else:
        densities = (1e-6, 1, 1000, 8000, 16000) if polar else (0, 1e-6, 1, 100, 1000, 8000, 16000)
        grid = [('derivative', c, m, t, r) for c, m, t, r in product(
            cases, mixtures, (180, 300, 600), densities)]
        grid += [('state', c, m, 600, r) for c, m, r in product(cases, mixtures, (1, 1000, 8000))]
    expected = sum(10 if kind == 'derivative' else 1 if kind == 'inverse' else
        (10 if backend == 'thermopack' else 11)+len(CASES[c][1]) for kind, c, *_ in grid)
    assert (len(grid), expected) == {'core': (480, 4968), 'polar': (108, 1134),
                                   'thermopack': (130, 1426)}[block]
    cache, failures, metadata, count, passed = {}, [], [], 0, 0
    started = time.perf_counter()

    def compare(name, actual, reference):
        nonlocal count, passed
        atol = (1e-4 if name == 'pressure_Pa' else 2e-5 if name.endswith('_J_mol')
                or name.startswith('mu_') else 2e-9 if name.startswith(('Ar', 'lnphi'))
                or name == 'compressibility_factor' else 2e-7)
        record = {'quantity': name, 'rtol': 2e-7, 'atol': atol}
        for role, f in (('native', actual), ('reference', reference)):
            try:
                value = float(f())
                if not math.isfinite(value):
                    raise ValueError('nonfinite result')
                record[role] = value
            except Exception as exc:
                record[role+'_exception'] = f'{type(exc).__name__}: {exc}'
        record['passed'] = ('native' in record and 'reference' in record and math.isclose(
            record['native'], record['reference'], rel_tol=2e-7, abs_tol=atol))
        count += 1
        passed += record['passed']
        return record

    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for unit, (kind, case, mixing, t, rho) in enumerate(grid, 1):
            key = case, mixing
            if key not in cache:
                cache[key] = models(case, mixing, backend, polar)
                metadata.append({'case': case, 'mixing': mixing, 'model': cache[key][0].to_dict()})
            native, ref, xs = cache[key]
            records = []
            if kind == 'derivative':
                for i, j in ORDERS:
                    records.append(compare(f'Ar{i}{j}', lambda: molecular_scaled_derivative(
                        native, t, rho, xs, inverse_temperature_order=i, density_order=j),
                        lambda: ref.get_Arxy(i, j, t, rho, xs)))
            else:
                actual, targets, errors = {}, {}, {}
                try:
                    state = molecular_eos_state(native, t, rho, xs, require_stable=False)
                    actual = dict(state.__dict__)
                    if backend == 'thermopack':
                        actual.update({f'mu_{i}': v for i, v in enumerate(
                            molecular_residual_chemical_potentials(native, t, rho, xs))})
                    else:
                        actual.update({f'lnphi_{i}': v for i, v in enumerate(
                            state.log_fugacity_coefficients)})
                except Exception as exc:
                    errors['native'] = exc
                try:
                    if backend == 'thermopack':
                        targets, mus = thermopack_residual_targets(ref, t, rho, xs)
                        targets.update({f'mu_{i}': v for i, v in enumerate(mus)})
                    else:
                        targets = teqp_targets(ref, t, rho, xs)
                        targets.update({f'lnphi_{i}': math.log(v) for i, v in enumerate(
                            ref.get_fugacity_coefficients(t, rho*xs))})
                except Exception as exc:
                    errors['reference'] = exc
                def value(name, reference=False):
                    role = 'reference' if reference else 'native'
                    if role in errors:
                        raise errors[role]
                    return (targets if reference else actual)[name]
                if kind == 'state':
                    names = list(NAMES)+[f'{"mu" if backend == "thermopack" else "lnphi"}_{i}'
                                        for i in range(len(xs))]
                    records.extend(compare(name, lambda: value(name), lambda: value(name, True))
                                   for name in names)
                if kind == 'inverse' or backend == 'teqp':
                    records.append(compare('pressure_density_inverse', lambda:
                        molecular_density_from_pressure(native, t, value('pressure_Pa', True), xs,
                            initial_density_mol_m3=1.1*rho,
                            density_bounds_mol_m3=(.5*rho, 1.5*rho)).molar_density_mol_m3,
                        lambda: rho))
            row = {'unit': unit, 'kind': kind, 'case': case, 'mixing': mixing,
                   'T_K': t, 'rho_mol_m3': rho, 'comparisons': records}
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            failures.extend({**{k: v for k, v in row.items() if k != 'comparisons'}, **r}
                            for r in records if not r['passed'])
            if unit % 12 == 0 or unit == len(grid):
                elapsed = time.perf_counter()-started
                print(f'Mie {block} {unit}/{len(grid)}; {unit/elapsed:.2f} units/s; '
                      f'ETA {(len(grid)-unit)*elapsed/unit:.1f}s; failures {len(failures)}',
                      flush=True)
    summary = {'mode': 'development', 'block': block, 'units': len(grid), 'comparisons': count,
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
    parser.add_argument('--block', choices=('core', 'polar', 'thermopack'), default='core')
    parser.add_argument('--source-quadrature-diagnostic', action='store_true')
    args = parser.parse_args()
    if args.source_quadrature_diagnostic:
        from mie_quadrature_diagnostic import source_diameter
        with patch('chemworld.physchem.saftvrmie.mie_effective_diameter', source_diameter):
            passed = run(args.output, args.block)
        result = json.loads((args.output/'summary.json').read_text(encoding='utf-8'))
        result['qualification_scope'] = 'reference_quadrature_isolation_only_not_native_default'
        (args.output/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    else:
        passed = run(args.output, args.block)
    raise SystemExit(0 if passed else 1)
