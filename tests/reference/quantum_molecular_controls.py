"""Fixed prospective quantum chain/HS/control/truncation qualification."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
from molecular_reference import thermopack_residual_targets
from quantum_molecular_discriminators import pressure_consistency
from run_dufal import CASES as ASSOCIATION_CASES
from run_dufal import PARAMS as ASSOCIATION_PARAMS
from run_dufal import models as association_models
from run_quantum_molecular import NAMES, ORDERS, mie_models, reference_scaled
from thermopack.saftvrqmie import saftvrqmie

from chemworld.physchem.molecular_eos import (
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.quantum_molecular import QuantumMieSegment, SAFTVRQMieModel

HS = tuple(product(('Lafitte', 'additive', 'nonadditive-Santos'), (False, True), (False, True)))
FLAGS = ('enable_hard_sphere', 'enable_a1', 'enable_a2', 'enable_a3', 'enable_chain')
CHAIN_CASES = ((('HE',), (1.,)), (('HE', 'NE'), (1., 0.)),
               (('HE', 'NE'), (0., 1.)), (('HE', 'NE'), (.4, .6)))


def grid(block):
    if block == 'association':
        return [{'names': tuple(ASSOCIATION_PARAMS[i][0] for i in indices), 'x': x,
                 'm': tuple(ASSOCIATION_PARAMS[i][1] for i in indices), 'order': order,
                 'hs': hs, 'exact': hs != 'Lafitte', 'extra': hs == 'additive-exact',
                 't': t, 'rho': rho, 'association_case': index}
                for (index, (indices, x)), order, hs, t, rho in product(
                    enumerate(ASSOCIATION_CASES), range(3),
                    ('Lafitte', 'additive-exact', 'nonadditive-Santos'),
                    (300., 600.), (100., 5000.))]
    if block == 'chain':
        return [{'names': names, 'x': x, 'order': order, 'hs': hs, 'exact': exact, 'extra': extra,
                 'm': (1.5,) if len(names) == 1 else (1.5, 2.3), 't': t, 'rho': rho}
                for (names, x), order, (hs, exact, extra), t, rho in product(
                    CHAIN_CASES, range(3), HS, (30., 80.), (100., 5000.))]
    if block == 'control':
        return [{'names': names, 'x': x, 'order': order, 'hs': hs, 'exact': exact, 'extra': extra,
                 'm': (1.5,) if len(names) == 1 else (1.5, 2.3), 't': 80., 'rho': 1000.,
                 'flags': flags}
                for (names, x), order, (hs, exact, extra), flags in product(
                    (CHAIN_CASES[0], CHAIN_CASES[-1]), range(3), HS,
                    tuple(product((False, True), repeat=5)))]
    return [{'names': (name,), 'x': (1.,), 'order': order, 'hs': 'nonadditive-Santos',
             'exact': True, 'extra': False, 'm': (m,), 't': t, 'rho': rho,
             'truncation': cutoff is not None, 'shift': shift, 'cutoff': cutoff or 3.5}
            for name, order, m, t, rho, (cutoff, shift) in product(
                ('HE', 'NE', 'H2'), range(3), (1., 1.7), (30., 80.), (100., 5000.),
                ((None, False), (2.5, False), (2.5, True), (4., False), (4., True)))]


def models(case, sources):
    if 'association_case' in case:
        association, _, _ = association_models(case['association_case'], 'thermopack',
                                                source_topology=True)
        masses = {'H2O': 2.9915076e-26, 'MEOH': 5.321e-26, 'NH3': 2.827e-26}
        parameters = {p[0]: p for p in ASSOCIATION_PARAMS}
        cs = []
        ref = saftvrqmie(','.join(case['names']), feynman_hibbs_order=case['order'],
                        parameter_reference='Dufal2015')
        for i, name in enumerate(case['names'], 1):
            p = parameters[name]
            cs.append(QuantumMieSegment(name, p[1], p[2], p[3], 'Dufal2015',
                                       p[4], 6., masses[name]))
            ref.set_mass(i, masses[name])
            for j in range(i+1, len(case['names'])+1):
                ref.set_eps_kij(i, j, 0.)
                ref.set_sigma_lij(i, j, 0.)
                ref.set_lr_gammaij(i, j, 0.)
        native = SAFTVRQMieModel(tuple(cs), ((0.,)*len(cs),)*len(cs), (200., 800.),
            'Dufal2015 explicit quantum composition', case['order'],
            gas_constant_J_mol_K=ref.Rgas, planck_J_s=6.626069311e-34,
            association=association.association)
    else:
        native, ref = mie_models(case['names'], sources, case['order'], 'nonadditive-Santos')
    native = replace(native, components=tuple(replace(c, segment_count=m)
        for c, m in zip(native.components, case['m'], strict=True)),
        hard_sphere_reference=case['hs'], exact_binary_dhs=case['exact'],
        enable_hs_extra=case['extra'])
    for i, c in enumerate(native.components, 1):
        ref.set_pure_fluid_param(i, c.segment_count, c.diameter_m, c.well_depth_K,
                                c.attractive_exponent, c.repulsive_exponent)
    ref.set_hard_sphere_reference({'Lafitte': 'LAFITTE', 'additive': 'ADDITIVE',
        'additive-exact': 'ADDITIVE', 'nonadditive-Santos': 'NON-ADDITIVE'}[case['hs']],
        case['exact'], case['extra'])
    if 'truncation' in case:
        native = replace(native, enable_truncation_correction=case['truncation'],
            enable_shift_correction=case['shift'], reduced_radius_cut=case['cutoff'])
        ref.truncation_correction(case['truncation'], case['shift'], case['cutoff'])
    return native, ref


def run(block, sources, output, control_api='scalar'):
    output.mkdir(parents=True, exist_ok=False)
    cases = grid('chain' if block == 'chain-pressure' else block)
    if block == 'chain-pressure':
        cases = [c for c in cases if c['x'] == (.4, .6) and c['extra']]
    expected_units, expected = {'chain': (576, 12528), 'control': (2304, 4608),
        'truncation': (360, 7560), 'chain-pressure': (72, 288), 'association': (324, 7092)}[block]
    assert len(cases) == expected_units
    failures, metadata, cache = [], [], {}
    count = passed = 0
    started = time.perf_counter()
    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for unit, case in enumerate(cases, 1):
            key = json.dumps({k: v for k, v in case.items() if k not in ('x', 't', 'rho', 'flags')})
            if key not in cache:
                cache[key] = models(case, sources)
                metadata.append(cache[key][0].to_dict())
            native, ref = cache[key]
            if 'flags' in case:
                native = replace(native, **dict(zip(FLAGS, case['flags'], strict=True)))
                for flag, setter in zip(case['flags'], (ref.model_control_hard_sphere,
                    ref.model_control_a1, ref.model_control_a2, ref.model_control_a3,
                    ref.model_control_chain), strict=True):
                    setter(flag)
            t, rho, x = case['t'], case['rho'], np.asarray(case['x'])
            quantities = (('dP_drho_T_Pa_m3_mol', 'Ar02', 'residual_cp_J_mol_K', 'Ar03')
                if block == 'chain-pressure' else
                ('residual_helmholtz_J_mol', 'pressure_Pa') if block == 'control'
                else (*NAMES, *(f'mu_{i}' for i in range(len(x))),
                      *(f'Ar{i}{j}' for i, j in ORDERS)))
            actual, target, errors = {}, {}, {}
            for role in ('native', 'reference'):
                try:
                    if role == 'native':
                        if block == 'control':
                            actual = {'residual_helmholtz_J_mol': native.gas_constant_J_mol_K*t
                                *native.alphar(t, rho, x), 'pressure_Pa': rho
                                *native.gas_constant_J_mol_K*t*(1+molecular_scaled_derivative(
                                    native, t, rho, x, density_order=1))}
                        else:
                            actual = dict(molecular_eos_state(native, t, rho, x,
                                require_stable=False).__dict__)
                            actual.update({f'mu_{i}': v for i, v in enumerate(
                                molecular_residual_chemical_potentials(native, t, rho, x))})
                            actual.update({f'Ar{i}{j}': molecular_scaled_derivative(native,
                                t, rho, x, inverse_temperature_order=i, density_order=j)
                                for i, j in ORDERS})
                    elif block == 'chain-pressure':
                        target = pressure_consistency(ref, t, rho, x)
                    elif block == 'control':
                        if control_api == 'full-derivative':
                            u = ref.internal_energy_tv(t, 1/rho, x, dedt=True, property_flag='R')[0]
                            s = ref.entropy_tv(t, 1/rho, x, dsdt=True, property_flag='R')[0]
                            target = {'residual_helmholtz_J_mol': u-t*s,
                                'pressure_Pa': ref.pressure_tv(t, 1/rho, x, dpdv=True)[0]}
                        else:
                            target = {'residual_helmholtz_J_mol': ref.helmholtz_tv(t, 1/rho, x,
                                property_flag='R')[0],
                                'pressure_Pa': ref.pressure_tv(t, 1/rho, x)[0]}
                    else:
                        target, mu = thermopack_residual_targets(ref, t, rho, x)
                        target.update({f'mu_{i}': v for i, v in enumerate(mu)})
                        target.update({f'Ar{i}{j}': v for (i, j), v in
                                       reference_scaled(ref, t, rho, x).items()})
                except Exception as exc:
                    errors[role] = f'{type(exc).__name__}: {exc}'
            comparisons = []
            for quantity in quantities:
                third = quantity.startswith('Ar') and int(quantity[-1])+int(quantity[-2]) == 3
                atol, rtol = (2e-6, 2e-5) if third else (2e-7, 3e-6)
                row = {'quantity': quantity, 'atol': atol, 'rtol': rtol}
                for role, values in (('native', actual), ('reference', target)):
                    if role in errors:
                        row[role+'_exception'] = errors[role]
                    elif quantity in values and math.isfinite(float(values[quantity])):
                        row[role] = float(values[quantity])
                    else:
                        row[role+'_exception'] = 'nonfinite or missing response'
                row['passed'] = ('native' in row and 'reference' in row and abs(
                    row['native']-row['reference']) <= atol+rtol*max(
                        abs(row['native']), abs(row['reference'])))
                count += 1
                passed += row['passed']
                if not row['passed']:
                    failures.append({'unit': unit, **case, **row})
                comparisons.append(row)
            stream.write(json.dumps({'unit': unit, **case, 'comparisons': comparisons})+'\n')
            stream.flush()
            if unit % (96 if block == 'control' else 6) == 0 or unit == expected_units:
                elapsed = time.perf_counter()-started
                print(f'Quantum {block}: {unit}/{expected_units}, {unit/elapsed:.2f} states/s, '
                    f'ETA {(expected_units-unit)*elapsed/unit:.1f}s, failed={len(failures)}',
                    flush=True)
    summary = {'mode': 'development', 'block': block, 'reference': 'ThermoPack2.2.3',
        'units': len(cases), 'comparisons': count, 'expected_comparisons': expected,
        'passed': passed, 'failed': len(failures), 'seconds': time.perf_counter()-started,
        'all_failures': failures, 'inputs': metadata, 'control_reference_api': control_api}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    assert count == expected
    print(json.dumps({k: v for k, v in summary.items() if k not in ('all_failures', 'inputs')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', choices=(
        'chain', 'control', 'truncation', 'chain-pressure', 'association'), required=True)
    parser.add_argument('--control-api', choices=('scalar', 'full-derivative'), default='scalar')
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.block, args.sources, args.output, args.control_api)
