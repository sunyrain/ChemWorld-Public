# ruff: noqa: B023 -- evaluate callbacks within current unit.
"""Independent high precision polynomial and site-root differentiation."""

import argparse
import json
import math
import re
import time
from itertools import product
from pathlib import Path

import mpmath as mp
from run_dufal import CASES, PARAMS, models
from run_saftvrmie import ORDERS

from chemworld.physchem.molecular_eos import (
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)


class Oracle:
    def __init__(self, source, case, basis, source_topology=False):
        # Parse independent Fortran table, never import native coefficients.
        pattern = r'b\((\d+),(\d+),(\d+)\)\s*=\s*([-+\d.Ee]+)'
        table = re.findall(pattern, source.read_text())
        assert len(table) == 462
        self.coeff = [(int(i), int(j), int(k), mp.mpf(c)) for i, j, k, c in table]
        self.ps = [PARAMS[i] for i in CASES[case][0]]
        self.basis = basis
        self.source_topology = source_topology
        self.m = [mp.mpf(str(p[1])) for p in self.ps]
        self.sigma = [mp.mpf(str(p[2])) for p in self.ps]
        self.na = mp.mpf('6.02214076e23')
        self.R = mp.mpf('8.31446261815324')
        self.site_owners = [i for i in range(len(self.ps)) for _ in (0, 1)]
        self.counts = [p[7].count(label) for p in self.ps for label in ('e', 'H')]
        self.pair = {}
        for i, a in enumerate(self.ps):
            for j, b in enumerate(self.ps):
                eps = mp.sqrt(mp.mpf(str(a[3]))*mp.mpf(str(b[3])))
                eps *= (self.sigma[i]*self.sigma[j])**mp.mpf('1.5')
                eps /= ((self.sigma[i]+self.sigma[j])/2)**3
                lam = 3+mp.sqrt(mp.mpf(str(a[4]-3))*mp.mpf(str(b[4]-3)))
                coefficients = {(u, v): mp.fsum(c*lam**k for ii, jj, k, c in self.coeff
                    if (ii, jj) == (u, v)) for u in range(11) for v in range(11-u)}
                hb = mp.sqrt(mp.mpf(a[5])*mp.mpf(b[5]))
                vol = ((mp.root(mp.mpf(str(a[6])), 3)+mp.root(mp.mpf(str(b[6])), 3))/2)**3
                self.pair[i, j] = (eps, hb, vol, coefficients)

    def alpha(self, t, rho, x):
        mbar = mp.fsum(v*m for v, m in zip(x, self.m, strict=True))
        w = [v*m/mbar for v, m in zip(x, self.m, strict=True)] if self.basis == 'segment' else x
        rs = rho*self.na*(mbar if self.basis == 'segment' else 1)*mp.fsum(
            w[i]*w[j]*((self.sigma[i]+self.sigma[j])/2)**3
            for i in range(len(x)) for j in range(len(x)))
        strengths = {}
        for key, (eps, hb, vol, coeff) in self.pair.items():
            integral = mp.fsum(c*rs**i*(t/eps)**j for (i, j), c in coeff.items())
            strengths[key] = rho*self.na*vol*mp.expm1(hb/t)*integral
        matrix = mp.matrix([[x[j]*self.counts[b]*strengths[i, j] if (
            a % 2 != b % 2 or (self.source_topology and i != j
                               and (self.ps[i][0] == 'NH3' or self.ps[j][0] == 'NH3'))
        ) else 0
            for b, j in enumerate(self.site_owners)] for a, i in enumerate(self.site_owners)])
        n = len(self.counts)
        def residual(*xs):
            return tuple(xs[i]*(1+mp.fsum(matrix[i, j]*xs[j] for j in range(n)))-1
                         for i in range(n))
        initial = tuple(2/(1+mp.sqrt(1+4*mp.fsum(matrix[i, j] for j in range(n))))
                        for i in range(n))
        # The oracle differentiator raises precision beyond80digits for tiny
        # perturbations. Scale root tolerance to *current* precision, not80.
        root = mp.findroot(residual, initial, solver='mdnewton', tol=mp.eps*100,
                           maxsteps=80)
        return mp.fsum(x[i]*self.counts[a]*(mp.log(root[a])-root[a]/2+mp.mpf('.5'))
                       for a, i in enumerate(self.site_owners))


def run(output, source):
    mp.mp.dps = 80
    output.mkdir(parents=True, exist_ok=False)
    grid = [('derivative', b, c, t, r) for b, c, t, r in product(
        ('molecular', 'segment'), range(9), (300, 450, 600), (0, 1, 1000, 15000))]
    grid += [('chemical_potential', b, c, t, r) for b, c, t, r in product(
        ('molecular', 'segment'), range(9), (300, 450, 600), (1, 1000, 15000))]
    cache, failures, passed, count = {}, [], 0, 0
    started = last_progress = time.perf_counter()
    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for unit, (kind, basis, case, t, rho) in enumerate(grid, 1):
            key = basis, case
            if key not in cache:
                native, _, _ = models(case, 'teqp' if basis == 'molecular' else 'thermopack')
                cache[key] = native, Oracle(source, case, basis)
            native, oracle = cache[key]
            x = tuple(mp.mpf(str(v)) for v in CASES[case][1])
            records = []
            if kind == 'derivative':
                for i, j in ORDERS:
                    rec = {'quantity': f'Ar{i}{j}', 'rtol': 2e-7, 'atol': 2e-9}
                    try:
                        rec['native'] = molecular_scaled_derivative(native, t, rho,
                            CASES[case][1], inverse_temperature_order=i, density_order=j)
                        # Relative coordinate derivatives avoid enormous/small
                        # beta,rho scales, while retaining vacuum continuation.
                        rec['reference'] = float(mp.diff(
                            lambda a, b: oracle.alpha(mp.mpf(t)/(1+a), mp.mpf(rho)*(1+b), x),
                            (mp.mpf(0), mp.mpf(0)), (i, j)))
                    except Exception as exc:
                        rec['exception'] = f'{type(exc).__name__}: {exc}'
                    records.append(rec)
            else:
                actual = molecular_residual_chemical_potentials(native, t, rho, CASES[case][1])
                for i in range(len(x)):
                    rec = {'quantity': f'mu_{i}', 'rtol': 2e-7, 'atol': 2e-5,
                           'native': actual[i]}
                    def extensive(z):
                        amounts = list(x)
                        amounts[i] += z
                        total = mp.fsum(amounts)
                        return total*oracle.alpha(mp.mpf(t), rho*total,
                                                  [v/total for v in amounts])
                    try:
                        rec['reference'] = float(oracle.R*t*mp.diff(extensive, mp.mpf(0)))
                    except Exception as exc:
                        rec['exception'] = f'{type(exc).__name__}: {exc}'
                    records.append(rec)
            for rec in records:
                rec['passed'] = ('native' in rec and 'reference' in rec and math.isfinite(
                    rec['native']) and math.isfinite(rec['reference']) and math.isclose(
                    rec['native'], rec['reference'], rel_tol=rec['rtol'], abs_tol=rec['atol']))
                count += 1
                passed += rec['passed']
            info = {'unit': unit, 'kind': kind, 'basis': basis, 'case': case,
                    'T_K': t, 'rho_mol_m3': rho}
            stream.write(json.dumps({**info, 'comparisons': records})+'\n')
            stream.flush()
            failures += [{**info, **rec} for rec in records if not rec['passed']]
            now = time.perf_counter()
            if now-last_progress > 20 or unit == len(grid):
                elapsed = now-started
                print(f'Dufal mp80 {unit}/{len(grid)}; {unit/elapsed:.2f}units/s; '
                      f'ETA {(len(grid)-unit)*elapsed/unit:.1f}s; failures {len(failures)}',
                      flush=True)
                last_progress = now
    summary = {'mode': 'development', 'units': len(grid), 'comparisons': count,
        'expected_comparisons': 2466, 'passed': passed, 'failed': len(failures),
        'failures': failures, 'elapsed_seconds': time.perf_counter()-started,
        'source_table': 'ThermoPack d68c794c saftvrmie_association.f90', 'decimal_precision': 80}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'failures'}))
    assert count == 2466
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--coefficient-source', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.coefficient_source) else 1)
