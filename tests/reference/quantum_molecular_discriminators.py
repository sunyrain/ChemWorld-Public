"""Independent quantum cubic equations and source additive-HS consistency."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

import mpmath as mp
import numpy as np
from molecular_reference import thermopack_residual_targets
from run_quantum_molecular import (
    CUBIC_CASES,
    MIE_CASES,
    NAMES,
    ORDERS,
    cubic_models,
    mie_models,
)

from chemworld.physchem.molecular_eos import (
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)


def independent_cubic(model, temperature, density, fractions):
    cs = model.components
    r = mp.mpf(str(model.gas_constant_J_mol_K))
    xs = tuple(mp.mpf(str(x)) for x in fractions)
    t, rho = mp.mpf(str(temperature)), mp.mpf(str(density))
    kij = [[mp.mpf(str(v)) for v in row] for row in model.interaction]
    lij = [[mp.mpf(str(v)) for v in row] for row in model.covolume_interaction]

    def alpha(tt, rr, x):
        attractions, covolumes, shifts = [], [], []
        for c in cs:
            tc = mp.mpf(str(c.critical_temperature_K))
            ell, m, n = [mp.mpf(str(v)) for v in c.twu_coefficients]
            tr = tt/tc
            attractions.append(mp.mpf(str(c.attraction_Pa_m6_mol2))*tr**(n*(m-1))
                               *mp.exp(ell*(1-tr**(m*n))))
            xi, adj = mp.mpf(str(c.beta_xi_K)), mp.mpf(str(c.beta_temperature_shift_K))
            covolumes.append(mp.mpf(str(c.covolume_m3_mol))
                             *((1+xi/(tt+adj))/(1+xi/(tc+adj)))**3)
            shifts.append(mp.mpf(str(c.volume_translation_m3_mol)))
        n = len(cs)
        a = mp.fsum(x[i]*x[j]*mp.sqrt(attractions[i]*attractions[j])*(1-kij[i][j])
                    for i in range(n) for j in range(n))
        b = mp.fsum(x[i]*x[j]*(covolumes[i]+covolumes[j])/2*(1-lij[i][j])
                    for i in range(n) for j in range(n))
        v = 1/rr
        ve = v+mp.fsum(x[i]*shifts[i] for i in range(n))
        root2 = mp.sqrt(2)
        return -mp.log((ve-b)/v)-a/(2*root2*b*r*tt)*mp.log(
            (ve+(1+root2)*b)/(ve+(1-root2)*b))

    scaled = {(i, j): (1/t)**i*rho**j*mp.diff(
        lambda beta, rr: alpha(1/beta, rr, xs), (1/t, rho), (i, j)) for i, j in ORDERS}
    z = 1+scaled[0, 1]
    stiffness = 1+2*scaled[0, 1]+scaled[0, 2]
    thermal = z-scaled[1, 1]
    a, u, cv = r*t*scaled[0, 0], r*t*scaled[1, 0], -r*scaled[2, 0]
    values = (rho*r*t*z, z, a, u, u+r*t*(z-1), (u-a)/t, cv,
              cv+r*(thermal**2/stiffness-1), rho*r*thermal, r*t*stiffness)
    targets = dict(zip(NAMES, values, strict=True))
    targets.update({f'Ar{i}{j}': v for (i, j), v in scaled.items()})
    for index in range(len(xs)):
        def extensive(amount, index=index):
            amounts = list(xs)
            amounts[index] = amount
            total = mp.fsum(amounts)
            return total*alpha(t, rho*total, tuple(v/total for v in amounts))
        targets[f'mu_{index}'] = r*t*mp.diff(extensive, xs[index])
    return {k: float(v) for k, v in targets.items()}


def pressure_consistency(ref, t, rho, x):
    state, _ = thermopack_residual_targets(ref, t, rho, x)
    h = .001*rho
    ps = [float(ref.pressure_tv(t, 1/(rho+k*h), x)[0]) for k in (-2, -1, 0, 1, 2)]
    slope = (ps[0]-8*ps[1]+8*ps[3]-ps[4])/(12*h)
    second = (-ps[0]+16*ps[1]-30*ps[2]+16*ps[3]-ps[4])/(12*h*h)
    z, r = state['compressibility_factor'], ref.Rgas
    ar02 = slope/(r*t)-1-2*(z-1)
    return {'dP_drho_T_Pa_m3_mol': slope, 'Ar02': ar02,
        'residual_cp_J_mol_K': state['residual_cv_J_mol_K']
        +t*state['dP_dT_rho_Pa_K']**2/(rho*rho*slope)-r,
        'Ar03': rho*second/(r*t)-2*(z-1)-4*ar02}


def run(output, sources, block):
    mp.mp.dps = 60
    output.mkdir(parents=True, exist_ok=False)
    cases = CUBIC_CASES if block == 'cubic' else [MIE_CASES[c] for c in (5, 6, 7, 11)]
    grid = list(product(range(len(cases)), (0,) if block == 'cubic' else (0, 1, 2),
        (20., 50., 100.) if block == 'cubic' else (30., 80.),
        (1., 500., 5000.) if block == 'cubic' else (1., 1000., 10000.)))
    cache, failures, count, passed, metadata = {}, [], 0, 0, []
    started = time.perf_counter()
    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for unit, (case, order, t, rho) in enumerate(grid, 1):
            names, fractions = cases[case]
            key = names, order
            if key not in cache:
                cache[key] = (cubic_models(names, sources) if block == 'cubic' else
                              mie_models(names, sources, order, 'additive-exact'))
                metadata.append(cache[key][0].to_dict())
            native, ref = cache[key]
            x = np.asarray(fractions)
            actual = dict(molecular_eos_state(native, t, rho, x, require_stable=False).__dict__)
            orders = ORDERS if block == 'cubic' else ((0, 2), (0, 3))
            actual.update({f'Ar{i}{j}': molecular_scaled_derivative(native, t, rho, x,
                inverse_temperature_order=i, density_order=j) for i, j in orders})
            if block == 'cubic':
                actual.update({f'mu_{i}': v for i, v in enumerate(
                    molecular_residual_chemical_potentials(native, t, rho, x))})
            expected = (independent_cubic(native, t, rho, fractions) if block == 'cubic' else
                        pressure_consistency(ref, t, rho, x))
            records = []
            for name, value in expected.items():
                atol, rtol = ((2e-8, 2e-8) if block == 'cubic' else
                              (2e-6, 2e-5) if name == 'Ar03' else (2e-7, 3e-6))
                row = {'quantity': name, 'native': float(actual[name]), 'reference': value,
                       'atol': atol, 'rtol': rtol}
                row['passed'] = (math.isfinite(row['native']) and math.isfinite(value)
                    and abs(row['native']-value) <= atol+rtol*max(abs(row['native']), abs(value)))
                records.append(row)
                count += 1
                passed += row['passed']
                if not row['passed']:
                    failures.append({'unit': unit, 'case': case, 'FH': order,
                                     'T_K': t, 'rho_mol_m3': rho, **row})
            stream.write(json.dumps({'unit': unit, 'case': case, 'FH': order,
                'T_K': t, 'rho_mol_m3': rho, 'comparisons': records})+'\n')
            stream.flush()
            if unit % 6 == 0 or unit == len(grid):
                elapsed = time.perf_counter()-started
                print(f'Quantum discriminator {block}: {unit}/{len(grid)}, '
                    f'{unit/elapsed:.2f} states/s; ETA {(len(grid)-unit)*elapsed/unit:.1f}s; '
                    f'failures {len(failures)}', flush=True)
    summary = {'mode': 'development', 'block': block, 'units': len(grid), 'comparisons': count,
        'expected_comparisons': 3942 if block == 'cubic' else 288,
        'passed': passed, 'failed': len(failures), 'all_failures': failures, 'inputs': metadata,
        'seconds': time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    assert count == summary['expected_comparisons']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--block', choices=('cubic', 'mie-pressure'), required=True)
    args = parser.parse_args()
    run(args.output, args.sources, args.block)
