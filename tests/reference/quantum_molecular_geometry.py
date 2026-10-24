"""Independent high precision quantum pair potential and geometry equations."""

import argparse
import json
import time
from itertools import combinations_with_replacement, product
from pathlib import Path

import mpmath as mp
from run_quantum_molecular import mie_models

from chemworld.physchem.caloric_math import derivative_values


def equations(pair, temperature):
    sigma, eps, lr, la, mass, h, kb = (mp.mpf(str(v)) for v in (
        pair.diameter_m, pair.well_depth_K, pair.repulsive_exponent,
        pair.attractive_exponent, pair.reduced_mass_kg, pair.planck_J_s, pair.boltzmann_J_K))
    d = h*h/(96*mp.pi**2*mass*kb*sigma**2*temperature)
    c = lr/(lr-la)*(lr/la)**(la/(lr-la))

    def potential(radius):
        result = radius**(-lr)-radius**(-la)
        if pair.feynman_hibbs_order >= 1:
            result += d*(lr*(lr-1)*radius**(-lr-2)-la*(la-1)*radius**(-la-2))
        if pair.feynman_hibbs_order >= 2:
            result += d*d/2*(lr*(lr-1)*(lr+1)*(lr+2)*radius**(-lr-4)
                           -la*(la-1)*(la+1)*(la+2)*radius**(-la-4))
        return c*eps*result

    return potential


def roots(pair, temperature):
    potential = equations(pair, temperature)
    zero = mp.findroot(potential, (mp.mpf('.5'), mp.mpf('8')), solver='bisect',
                       maxsteps=mp.mp.prec+30)
    minimum = mp.findroot(lambda r: mp.diff(potential, r), (zero, zero*4), solver='bisect',
                          maxsteps=mp.mp.prec+30)
    depth = -potential(minimum)
    return zero, depth


def diameter_derivatives(pair, temperature, zero_derivatives):
    """Leibniz rule; f(s(T),T)=0, retaining moving-boundary derivatives."""
    zero, ds, dds = zero_derivatives[:3]

    def integrand(radius, order):
        if radius == 0:
            return mp.mpf(1) if order == 0 else mp.mpf(0)
        v = equations(pair, temperature)(radius)/temperature
        if v > 10000:  # below 10^-4342; independent precision is60digits
            return mp.mpf(1) if order == 0 else mp.mpf(0)
        derivatives = [mp.diff(lambda t: equations(pair, t)(radius)/t, temperature, k)
                       for k in range(1, order+1)]
        e = mp.exp(-v)
        if order == 0:
            return 1-e
        if order == 1:
            return e*derivatives[0]
        if order == 2:
            return e*(derivatives[1]-derivatives[0]**2)
        return e*(derivatives[2]-3*derivatives[0]*derivatives[1]+derivatives[0]**3)

    values = [mp.quad(lambda r, k=k: integrand(r, k), [0, zero/2, zero*mp.mpf('.8'), zero])
              for k in range(4)]
    values[2] += integrand(zero, 1)*ds
    values[3] += 2*integrand(zero, 2)*ds+mp.diff(lambda r: integrand(r, 1), zero)*ds**2
    values[3] += integrand(zero, 1)*dds
    return values


def run(output, sources):
    mp.mp.dps = 60
    output.mkdir(parents=True, exist_ok=False)
    grid = list(product(combinations_with_replacement(range(3), 2), (0, 1, 2), (30., 80.)))
    cache = {order: mie_models(('HE', 'NE', 'H2'), sources, order, 'nonadditive-Santos')[0]
             for order in (0, 1, 2)}
    started, failures, count, passed = time.perf_counter(), [], 0, 0
    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for unit, ((i, j), order, temperature) in enumerate(grid, 1):
            pair = cache[order].pair(i, j)
            tt = mp.mpf(str(temperature))
            target_roots = [[mp.diff(lambda t, index=index, pair=pair: roots(pair, t)[index], tt, k)
                             for k in range(4)] for index in range(2)]
            targets = [*target_roots, diameter_derivatives(pair, tt, target_roots[0])]
            rows = []
            for radius in (.9, 1.1, 1.5, 2.):
                rows.append((f'potential_r{radius}', pair.potential_K(temperature, radius),
                             float(equations(pair, tt)(mp.mpf(str(radius))))))
            for index, name in enumerate(('sigma_eff', 'epsilon_eff', 'BH')):
                actual = derivative_values(
                    lambda t, index=index, pair=pair: pair.geometry(t)[index], temperature, order=3)
                for derivative in range(4):
                    value = targets[index][derivative]
                    # Scale the T derivative to preserve a meaningful fixed tolerance.
                    rows.append((name+f'_d{derivative}', actual[derivative]*temperature**derivative,
                                 float(value)*temperature**derivative))
            records = []
            for name, actual, reference in rows:
                row = {'quantity': name, 'native': actual, 'reference': reference,
                    'atol': 2e-10, 'rtol': 2e-8,
                    'passed': abs(actual-reference) <= 2e-10+2e-8*max(abs(actual), abs(reference))}
                records.append(row)
                count += 1
                passed += row['passed']
                if not row['passed']:
                    failures.append({'unit': unit, 'pair': (i, j), 'FH': order,
                                     'T_K': temperature, **row})
            stream.write(json.dumps({'unit': unit, 'pair': (i, j), 'FH': order,
                                     'T_K': temperature, 'comparisons': records})+'\n')
            stream.flush()
            elapsed = time.perf_counter()-started
            print(f'Quantum geometry {unit}/{len(grid)}, {unit/elapsed:.2f} units/s; '
                  f'ETA {(len(grid)-unit)*elapsed/unit:.1f}s; failed={len(failures)}', flush=True)
    summary = {'mode': 'development',
        'reference': 'independent mpmath60 equations/adaptive integral',
        'units': len(grid), 'comparisons': count, 'expected_comparisons': 576,
        'passed': passed, 'failed': len(failures), 'all_failures': failures,
        'seconds': time.perf_counter()-started,
        'inputs': [model.to_dict() for model in cache.values()]}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    assert count == 576


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.sources)
