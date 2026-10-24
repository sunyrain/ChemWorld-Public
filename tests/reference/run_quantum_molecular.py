"""Prospective fixed quantum EOS blocks against ThermoPack2.2.3."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np
from molecular_reference import thermopack_residual_targets
from thermopack.quantum_cubic import qcubic
from thermopack.saftvrqmie import saftvrqmie

from chemworld.physchem.molecular_eos import (
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.quantum_molecular import QuantumMieSegment, SAFTVRQMieModel
from chemworld.physchem.quantum_molecular_cubic import QuantumCubicComponent, QuantumCubicModel

FILES = {'HE': 'Helium', 'NE': 'Neon', 'H2': 'Hydrogen', 'D2': 'Deuterium',
         'P-H2': 'ParaHydrogen', 'O-H2': 'OrthoHydrogen'}
BETA = {'HE': (1.4912, 3.2634), 'NE': (.4673, 2.4634), 'H2': (3.0696, 12.682),
        'D2': (1.6501, 7.309), 'P-H2': (3.0710, 12.688), 'O-H2': (3.0702, 12.684)}
BINARY = ((1., 0.), (0., 1.), (.2, .8), (.5, .5), (.8, .2))
TERNARY = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (.2, .3, .5))
CUBIC_CASES = [((name,), (1.,)) for name in FILES]+[
    (pair, x) for pair in (('HE', 'NE'), ('H2', 'NE')) for x in BINARY]+[
    (('HE', 'H2', 'D2'), x) for x in TERNARY]
MIE_CASES = [((name,), (1.,)) for name in ('HE', 'NE', 'H2')]+[
    (('HE', 'NE'), x) for x in BINARY]+[(('HE', 'NE', 'H2'), x) for x in TERNARY]
NAMES = ('pressure_Pa', 'compressibility_factor', 'residual_helmholtz_J_mol',
         'residual_internal_energy_J_mol', 'residual_enthalpy_J_mol',
         'residual_entropy_J_mol_K', 'residual_cv_J_mol_K', 'residual_cp_J_mol_K',
         'dP_dT_rho_Pa_K', 'dP_drho_T_Pa_m3_mol')
ORDERS = [(i, j) for i in range(4) for j in range(4-i)]


def source_data(root, name):
    return json.loads((root/(FILES[name]+'.json')).read_text(encoding='utf-8'))


def cubic_models(names, sources, alpha_policy='wheel-default'):
    ref = qcubic(','.join(names))
    attractions, covolumes = ref.get_energy_constants()*1e-6, ref.get_covolumes()*1e-3
    components = []
    for i, name in enumerate(names):
        fits = source_data(sources, name)['PR']
        alpha = next(v['correlation'] for k, v in fits.items()
                     if k.startswith('TWU') and v['ref'] == 'QuantumCubic')
        # Wheel2.2.3 lacks the source's P/O-H2 Twu fits and silently falls back.
        # Its public setter also retains the active classic shape for these species.
        # Keep that failed attempt independently replayable, without hiding the default.
        if alpha_policy == 'set-source-coefficients':
            ref.set_alpha_corr(i+1, 'TWU', alpha)
        shift = ref.get_ci(i+1)
        assert shift[-1] == 1 and all(v == 0 for v in shift[1:-1])
        components.append(QuantumCubicComponent(name, float(attractions[i]), float(covolumes[i]),
            ref.get_critical_parameters(i+1)[0], tuple(alpha),
            *BETA[name], shift[0], 'Aasen2020/source-explicit'))
    interaction = tuple(tuple(0. if i == j else ref.get_kij(i+1, j+1)
                        for j in range(len(names))) for i in range(len(names)))
    covolume_interaction = tuple(tuple(0. if i == j else ref.get_lij(i+1, j+1)
                        for j in range(len(names))) for i in range(len(names)))
    native = QuantumCubicModel(tuple(components), interaction, (2., 300.),
                              'Aasen2020 explicit input', ref.Rgas, covolume_interaction)
    return native, ref


def mie_models(names, sources, order, hs):
    ref = saftvrqmie(','.join(names), feynman_hibbs_order=order,
        additive_hard_sphere_reference=hs == 'additive-exact', parameter_reference='AASEN2019-FH1')
    components = []
    for i, name in enumerate(names, 1):
        fits = source_data(sources, name)
        parameters = next(v for k, v in fits.items() if k.startswith('SAFTVRMIE')
                          and v['ref'] == 'AASEN2019-FH1')
        ref.set_mass(i, parameters['mass'])
        ref.set_pure_fluid_param(i, parameters['m'], parameters['sigma'],
            parameters['eps_depth_divk'], parameters['lambda_a'], parameters['lambda_r'])
        components.append(QuantumMieSegment(name, parameters['m'], parameters['sigma'],
            parameters['eps_depth_divk'], 'Aasen2019-FH1', parameters['lambda_r'],
            parameters['lambda_a'], parameters['mass']))
    n = len(names)
    matrices = [tuple(tuple(0. if i == j else method(i+1, j+1)
                   for j in range(n)) for i in range(n)) for method in (
                       ref.get_eps_kij, ref.get_sigma_lij, ref.get_lr_gammaij)]
    native = SAFTVRQMieModel(tuple(components), matrices[0], (2., 300.),
        'FH1 parameters held fixed across all orders', order, hs,
        diameter_interaction=matrices[1], repulsion_interaction=matrices[2],
        gas_constant_J_mol_K=ref.Rgas, planck_J_s=6.626069311e-34)
    return native, ref


def reference_scaled_second(ref, t, rho, x):
    targets, _ = thermopack_residual_targets(ref, t, rho, x)
    r, z = ref.Rgas, targets['compressibility_factor']
    return {(0, 0): targets['residual_helmholtz_J_mol']/(r*t),
        (1, 0): targets['residual_internal_energy_J_mol']/(r*t),
        (2, 0): -targets['residual_cv_J_mol_K']/r,
        (0, 1): z-1, (1, 1): z-targets['dP_dT_rho_Pa_K']/(rho*r),
        (0, 2): targets['dP_drho_T_Pa_m3_mol']/(r*t)-1-2*(z-1)}


def reference_scaled(ref, t, rho, x):
    values = reference_scaled_second(ref, t, rho, x)

    def derivative(variable, quantity):
        base = t if variable == 'T' else rho
        h = .001*base
        results = []
        for shift in (-2, -1, 1, 2):
            tt, rr = (t+shift*h, rho) if variable == 'T' else (t, rho+shift*h)
            results.append(reference_scaled_second(ref, tt, rr, x)[quantity])
        return (results[0]-8*results[1]+8*results[2]-results[3])/(12*h)

    values[3, 0] = -t*derivative('T', (2, 0))-2*values[2, 0]
    values[2, 1] = rho*derivative('rho', (2, 0))
    values[1, 2] = -t*derivative('T', (0, 2))
    values[0, 3] = rho*derivative('rho', (0, 2))-2*values[0, 2]
    return values


def run(output, block, sources, alpha_policy):
    output.mkdir(parents=True, exist_ok=False)
    cases = CUBIC_CASES if block == 'cubic' else MIE_CASES
    settings = [(0, '')] if block == 'cubic' else list(product(
        (0, 1, 2), ('additive-exact', 'nonadditive-Santos')))
    temperatures = (20., 50., 100.) if block == 'cubic' else (30., 80.)
    densities = (1., 500., 5000.) if block == 'cubic' else (1., 1000., 10000.)
    grid = list(product(range(len(cases)), settings, temperatures, densities))
    expected = sum(20+len(cases[c][1]) for c, *_ in grid)
    assert (len(grid), expected) == ((180, 3942) if block == 'cubic' else (432, 9540))
    cache, metadata, failures, count, passed = {}, [], [], 0, 0
    started = time.perf_counter()
    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for unit, (case, (order, hs), t, rho) in enumerate(grid, 1):
            names, fractions = cases[case]
            key = names, order, hs
            if key not in cache:
                cache[key] = (cubic_models(names, sources, alpha_policy) if block == 'cubic' else
                              mie_models(names, sources, order, hs))
                metadata.append(cache[key][0].to_dict())
            native, ref = cache[key]
            x = np.asarray(fractions)
            actual, target, errors = {}, {}, {}
            for role in ('native', 'reference'):
                try:
                    if role == 'native':
                        state = molecular_eos_state(native, t, rho, x, require_stable=False)
                        actual = dict(state.__dict__)
                        actual.update({f'mu_{i}': v for i, v in enumerate(
                            molecular_residual_chemical_potentials(native, t, rho, x))})
                        actual.update({f'Ar{i}{j}': molecular_scaled_derivative(native, t, rho, x,
                            inverse_temperature_order=i, density_order=j) for i, j in ORDERS})
                    else:
                        target, mus = thermopack_residual_targets(ref, t, rho, x)
                        target.update({f'mu_{i}': v for i, v in enumerate(mus)})
                        target.update({f'Ar{i}{j}': v for (i, j), v in reference_scaled(
                            ref, t, rho, x).items()})
                except Exception as exc:
                    errors[role] = f'{type(exc).__name__}: {exc}'
            record = {'unit': unit, 'case': case, 'components': names, 'fractions': fractions,
                'FH': order, 'HS': hs, 'T_K': t, 'rho_mol_m3': rho}
            comparisons = []
            for name in (*NAMES, *(f'mu_{i}' for i in range(len(x))),
                         *(f'Ar{i}{j}' for i, j in ORDERS)):
                third = name.startswith('Ar') and int(name[-1])+int(name[-2]) == 3
                atol, rtol = (2e-6, 2e-5) if third else (2e-7, 3e-6)
                row = {'quantity': name, 'atol': atol, 'rtol': rtol}
                for role, values in (('native', actual), ('reference', target)):
                    if role in errors:
                        row[role+'_exception'] = errors[role]
                    elif name in values and math.isfinite(float(values[name])):
                        row[role] = float(values[name])
                    else:
                        row[role+'_exception'] = 'nonfinite or missing response'
                row['passed'] = ('native' in row and 'reference' in row and abs(
                    row['native']-row['reference']) <= atol+rtol*max(
                        abs(row['native']), abs(row['reference'])))
                count += 1
                passed += row['passed']
                if not row['passed']:
                    failures.append({**record, **row})
                comparisons.append(row)
            stream.write(json.dumps({**record, 'comparisons': comparisons})+'\n')
            stream.flush()
            if unit % 6 == 0 or unit == len(grid):
                elapsed = time.perf_counter()-started
                print(f'Quantum {block}: {unit}/{len(grid)}, {unit/elapsed:.2f} states/s, '
                      f'ETA {(len(grid)-unit)*elapsed/unit:.1f}s, failures={len(failures)}',
                      flush=True)
    summary = {'mode': 'development', 'block': block, 'reference': 'ThermoPack2.2.3',
        'units': len(grid), 'comparisons': count, 'expected_comparisons': expected,
        'passed': passed, 'failed': len(failures), 'seconds': time.perf_counter()-started,
        'all_failures': failures, 'inputs': metadata,
        'cubic_reference_alpha_policy': alpha_policy if block == 'cubic' else None,
        'reference_third_derivative': 'five-point derivative of analytic second; rel step .001'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    assert count == expected
    print(json.dumps({k: v for k, v in summary.items() if k not in ('all_failures', 'inputs')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--block', choices=('cubic', 'mie'), required=True)
    parser.add_argument('--cubic-alpha-policy',
        choices=('wheel-default', 'set-source-coefficients'), default='wheel-default')
    args = parser.parse_args()
    run(args.output, args.block, args.sources, args.cubic_alpha_policy)
