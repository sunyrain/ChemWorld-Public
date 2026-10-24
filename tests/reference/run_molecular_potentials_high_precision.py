# ruff: noqa: B023 -- differentiation executes within each fixed unit.
"""Independent mpmath potential derivatives, from pinned source numeric tables."""

import argparse
import json
import math
import re
import time
from itertools import product
from pathlib import Path

import mpmath as mp
from run_molecular_potentials import RHO_SCALE, models
from run_saftvrmie import ORDERS

from chemworld.physchem.molecular_eos import molecular_scaled_derivative


def oracle(source, family):
    if family == 'johnson':
        text = (source/'teqp-lj-johnson.hpp').read_text()
        table = re.search(r'const std::vector<double> x\s*\{(.*?)\};', text, re.S)[1]
        x = [mp.mpf(v.strip()) for v in re.sub(r'//[^\n]*', '', table).split(',') if v.strip()]
        def alpha(t, r):
            a = [x[1]*t+x[2]*mp.sqrt(t)+x[3]+x[4]/t+x[5]/t**2,
                 x[6]*t+x[7]+x[8]/t+x[9]/t**2, x[10]*t+x[11]+x[12]/t, x[13],
                 x[14]/t+x[15]/t**2, x[16]/t, x[17]/t+x[18]/t**2, x[19]/t**2]
            b = [x[20]/t**2+x[21]/t**3, x[22]/t**2+x[23]/t**4,
                 x[24]/t**2+x[25]/t**3, x[26]/t**2+x[27]/t**4,
                 x[28]/t**2+x[29]/t**3, x[30]/t**2+x[31]/t**3+x[32]/t**4]
            # Incomplete gamma is independent of native series/source recurrence.
            integrals = [mp.gammainc(i, 0, 3*r*r)/(2*3**i) for i in range(1, 7)]
            return (mp.fsum(v*r**i/i for i, v in enumerate(a, 1))
                    +mp.fsum(v*g for v, g in zip(b, integrals, strict=True)))/t
        return alpha
    if family == 'pets':
        text = (source/'pets.f90').read_text()
        vectors = []
        for key in ('a_vec', 'b_vec'):
            block = re.search(key+r' = \(/(.*?)/\)', text, re.S)[1].replace('&', '')
            vectors.append([mp.mpf(v.strip()) for v in block.split(',')])
        def alpha(t, r):
            d = 1-mp.mpf('.127112544')*mp.exp(-mp.mpf('3.052785558')/t)
            e = mp.pi*r*d**3/6
            i1, i2 = [mp.polyval(row[::-1], e) for row in vectors]
            c = (1-e)**4/((1-e)**4+8*e-2*e*e)
            return (4*e-3*e*e)/(1-e)**2-2*mp.pi*r*i1/t-mp.pi*r*c*i2/t**2
        return alpha
    text = (source/'teqp-squarewell.hpp').read_text()
    tables = {}
    for key in ('phivals', 'thetavals', 'gammanvals'):
        block = re.search(key+r' = \{(.*?)\n    \};', text, re.S)[1]
        tables[key] = [[mp.mpf(v.strip()) for v in row.split(',')] for row in
                       re.findall(r'\{\s*([-+\d.eE,\s]+)\}', block)]
    w, pi = mp.mpf('2.5'), mp.pi
    s = [-2*pi*(w**3-1)/(3*mp.factorial(i)) for i in range(1, 5)]
    p4, p5 = 32*w**3-18*w*w-48, 5*w**6-32*w**3+18*w*w+26
    b3 = [-(pi/6)**2*17, (pi/6)**2*(-mp.mpf(17)/2+p4),
          (pi/6)**2*(-mp.mpf(17)/6+p4-p5),
          (pi/6)**2*(-mp.mpf(17)/24+7*p4/12-3*p5/2)]
    phi = [mp.polyval(row[::-1], w) for row in tables['phivals']]
    theta = [(mp.fsum(row[k]*w**k for k in range(1, 5)),
              mp.fsum(row[k]*w**(k-4) for k in range(5, 8))) for row in tables['thetavals']]
    z = w**3-1
    gamma = [row[1]*w+row[2]*w*w+(row[3]+mp.fsum(row[k]*z**(k-2) for k in range(4, 9)))
             /(row[9]+mp.fsum(row[k]*z**(k-7) for k in range(10, 15)))
             for row in tables['gammanvals']]
    def alpha(t, r):
        e = pi*r/6
        a = [s[0]*r+b3[0]*r*r+mp.fsum(v*r**i for i, v in enumerate(gamma, 3)),
             s[1]*r*(1-r*r/mp.mpf('1.5129'))*mp.exp(b3[1]/s[1]*r+phi[0]*r**3+phi[1]*r**4)]
        a += [s[i]*r*mp.exp(b3[i]/s[i]*r+theta[i-2][0]*r*r/(1+theta[i-2][1]*r))
              for i in (2, 3)]
        return (4*e-3*e*e)/(1-e)**2+mp.fsum(v/t**i for i, v in enumerate(a, 1))
    return alpha


def run(output, source):
    mp.mp.dps = 80
    output.mkdir(parents=True, exist_ok=False)
    grid = list(product(('johnson', 'pets', 'squarewell'), (.8, 1.3, 3),
                        (0, 1e-8, .01, .3, .8, 1.1)))
    cache, count, passed, failures = {}, 0, 0, []
    start = time.perf_counter()
    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for unit, (family, t, r) in enumerate(grid, 1):
            if family not in cache:
                model, _, _ = models(9 if family == 'squarewell' else 0,
                                      'pets' if family == 'pets' else 'pure')
                cache[family] = model, oracle(source, family)
            model, expression = cache[family]
            tm, rm = mp.mpf(str(t)), mp.mpf(str(r))
            records = []
            for i, j in ORDERS:
                rec = {'quantity': f'Ar{i}{j}', 'rtol': 2e-7, 'atol': 2e-9}
                try:
                    rec['native'] = molecular_scaled_derivative(model, t*120, r*RHO_SCALE, (1,),
                        inverse_temperature_order=i, density_order=j)
                    rec['reference'] = float(mp.diff(lambda a, b: expression(tm/(1+a), rm*(1+b)),
                                                     (mp.mpf(0), mp.mpf(0)), (i, j)))
                except Exception as exc:
                    rec['exception'] = f'{type(exc).__name__}: {exc}'
                rec['passed'] = ('native' in rec and 'reference' in rec and math.isfinite(
                    rec['native']) and math.isfinite(rec['reference']) and math.isclose(
                    rec['native'], rec['reference'], rel_tol=2e-7, abs_tol=2e-9))
                count += 1
                passed += rec['passed']
                records.append(rec)
            info = {'unit': unit, 'family': family, 'Tstar': t, 'rhostar': r}
            stream.write(json.dumps({**info, 'comparisons': records})+'\n')
            stream.flush()
            failures += [{**info, **rec} for rec in records if not rec['passed']]
            if unit % 12 == 0 or unit == len(grid):
                elapsed = time.perf_counter()-start
                print(f'Potential mp80 {unit}/{len(grid)}; {unit/elapsed:.2f}units/s; '
                      f'ETA {(len(grid)-unit)*elapsed/unit:.1f}s; failures {len(failures)}',
                      flush=True)
    summary = {'mode': 'development', 'units': len(grid), 'comparisons': count,
        'expected_comparisons': 540, 'passed': passed, 'failed': len(failures),
        'failures': failures, 'elapsed_seconds': time.perf_counter()-start, 'decimal_precision': 80}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'failures'}))
    assert count == 540
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-directory', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.source_directory) else 1)
