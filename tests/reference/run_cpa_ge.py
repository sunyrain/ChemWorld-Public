# ruff: noqa: B023 -- callbacks execute within each fixed measurement unit.
"""Fixed 320-unit CPA/GE combination; independent 80-digit equations."""

import argparse
import json
import math
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import mpmath as mp
from run_cpa import ORDERS, PARAMETERS, models

from chemworld.physchem.cubic_mixing import GibbsExcessMixingSpec
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.molecular_eos import (
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)

RULES = ('huron_vidal', 'mhv', 'mhv1', 'mhv2', 'wong_sandler')


def declaration(rule, shape, contact, elliott):
    native = models(4, shape, contact, 'native', elliott=elliott)[0]
    activity = ActivityModelSpec('explicit-cpa-ge', native.component_ids, 'nrtl', {
        'tau_b:H2O|MEOH': 100., 'tau_b:MEOH|H2O': 50.,
        'alpha:H2O|MEOH': .3, 'alpha:MEOH|H2O': .3})
    q = ((-.53 if shape == 'peng_robinson' else -.593,) if rule == 'mhv1' else
         ((-.4347, -.003654) if shape == 'peng_robinson' else (-.478, -.0047))
         if rule == 'mhv2' else ())
    ge = GibbsExcessMixingSpec(rule, activity, q, {'H2O|MEOH': .04}
                              if rule == 'wong_sandler' else {})
    return replace(native, interaction=((0., 0.), (0., 0.)), ge_mixing=ge)


def reference_alpha(rule, shape, contact, elliott, T, rho, x):
    """Public equations, independent roots and symmetry-reduced H/O balances.

    Neither the native site system nor native mixing/derivatives are used.
    The pure parameter values are the fixed experiment inputs.
    """
    R = mp.mpf('8.31446261815324')
    p = [[mp.mpf(str(v)) for v in row[1:7]] for row in PARAMETERS]
    ai = [a*(1+c*(1-mp.sqrt(T/tc)))**2 for a, b, tc, c, e, beta in p]
    bi = [row[1] for row in p]
    alpha = [a/(b*R*T) for a, b in zip(ai, bi, strict=True)]
    b = sum(xi*bj for xi, bj in zip(x, bi, strict=True))
    mean = sum(xi*aj for xi, aj in zip(x, alpha, strict=True))
    # Binary NRTL, tau_ij=b_ij/T and G_ij=exp(-.3 tau_ij).
    tau12, tau21 = 100/T, 50/T
    g12, g21 = mp.exp(-mp.mpf('.3')*tau12), mp.exp(-mp.mpf('.3')*tau21)
    excess = x[0]*x[1]*(tau21*g21/(x[0]+x[1]*g21)+tau12*g12/(x[1]+x[0]*g12))
    volume = sum(xi*mp.log(b/bj) for xi, bj in zip(x, bi, strict=True))
    s1, s2 = (1+mp.sqrt(2), 1-mp.sqrt(2)) if shape == 'peng_robinson' else (mp.mpf(1), mp.mpf(0))
    C = -mp.log((1+s1)/(1+s2))/(s1-s2)
    if rule == 'mhv1':
        q = mp.mpf('-.53' if shape == 'peng_robinson' else '-.593')
        am = mean+(excess+volume)/q
    elif rule == 'mhv2':
        q1, q2 = (mp.mpf(v) for v in (('-.4347', '-.003654') if shape == 'peng_robinson'
                                     else ('-.478', '-.0047')))
        target = excess+volume+q1*mean+q2*sum(xi*aj**2 for xi, aj in zip(x, alpha, strict=True))
        am = (-q1-mp.sqrt(q1*q1+4*q2*target))/(2*q2)
    elif rule == 'mhv':
        def energy(a):
            u = (a-s1-s2-mp.sqrt((a-s1-s2)**2-4*(a+s1*s2)))/2
            return -1-mp.log(u-1)-a/(s1-s2)*mp.log((u+s1)/(u+s2))
        target = excess+volume+sum(xi*energy(aj) for xi, aj in zip(x, alpha, strict=True))
        am = mp.findroot(lambda a: energy(a)-target, (mean, mean*mp.mpf('1.001')))
    else:
        am = mean+excess/C
        if rule == 'wong_sandler':
            q = sum(x[i]*x[j]*((bi[i]-ai[i]/(R*T))+(bi[j]-ai[j]/(R*T)))/2
                    *(1-(mp.mpf('.04') if i != j else 0)) for i in range(2) for j in range(2))
            b = q/(1-am)
    cubic = -mp.log(1-b*rho)-am/(s1-s2)*mp.log((1+s1*b*rho)/(1+s2*b*rho))
    eta = rho*b/4
    contact_value = 1/(1-mp.mpf('1.9')*eta) if contact == 'Kontogeorgis' else (1-eta/2)/(1-eta)**3
    strengths = [[contact_value*(bi[i]+bi[j])/2*mp.sqrt(p[i][5]*p[j][5])
                  *mp.expm1((p[i][4]+p[j][4])/(2*R*T)) for j in range(2)] for i in range(2)]
    if elliott:
        strengths[0][1] = strengths[1][0] = mp.sqrt(strengths[0][0]*strengths[1][1])
    populations = (2*x[0], x[1])  # each water has2H/2O; methanol1H/1O
    def balances(y0, y1):
        ys = (y0, y1)
        return tuple(ys[i]*(1+rho*sum(populations[j]*strengths[i][j]*ys[j]
                                     for j in range(2)))-1 for i in range(2))
    initial = tuple(2/(1+mp.sqrt(1+4*rho*sum(populations[j]*strengths[i][j]
                                           for j in range(2)))) for i in range(2))
    ys = mp.findroot(balances, initial, maxsteps=100)
    if any(y <= 0 or y > 1 for y in ys):
        raise ValueError('independent association solve left physical site domain')
    return cubic+sum(2*populations[i]*(mp.log(ys[i])-ys[i]/2+mp.mpf('.5')) for i in range(2))


def reference_state(rule, shape, contact, elliott, t, r, fractions):
    T, rho = mp.mpf(str(t)), mp.mpf(str(r))
    xs = tuple(mp.mpf(str(x)) for x in fractions)
    R, beta = mp.mpf('8.31446261815324'), 1/T
    def ar(t, r, x=xs):
        return reference_alpha(rule, shape, contact, elliott, t, r, x)
    ds = {(i, j): beta**i*rho**j*mp.diff(lambda b, d: ar(1/b, d), (beta, rho), (i, j))
          for i, j in ORDERS}
    z = 1+ds[0, 1]
    stiffness, thermal = 1+2*ds[0, 1]+ds[0, 2], z-ds[1, 1]
    u, cv = R*T*ds[1, 0], -R*ds[2, 0]
    result = {'pressure_Pa': rho*R*T*z, 'compressibility_factor': z,
        'residual_helmholtz_J_mol': R*T*ds[0, 0], 'residual_internal_energy_J_mol': u,
        'residual_enthalpy_J_mol': u+R*T*(z-1), 'residual_entropy_J_mol_K': R*(ds[1, 0]-ds[0, 0]),
        'residual_cv_J_mol_K': cv, 'residual_cp_J_mol_K': cv+R*(thermal**2/stiffness-1),
        'dP_dT_rho_Pa_K': rho*R*thermal, 'dP_drho_T_Pa_m3_mol': R*T*stiffness,
        **{f'Ar{i}{j}': value for (i, j), value in ds.items()}}
    for k in range(2):
        def extensive(nk):
            amounts = list(xs)
            amounts[k] = nk
            total = sum(amounts)
            return total*R*T*ar(T, rho*total, tuple(n/total for n in amounts))
        result[f'mu_{k}'] = mp.diff(extensive, xs[k])
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    grid = list(product(RULES, ('srk', 'peng_robinson'), ('Kontogeorgis', 'Carnahan-Starling'),
                        (False, True), ((.3, .7), (.7, .3)), (200, 210), (100, 10000)))
    assert len(grid) == 320
    failures, cache, count = [], {}, 0
    started = time.perf_counter()
    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for unit, (rule, shape, contact, elliott, xs, t, rho) in enumerate(grid, 1):
            key = rule, shape, contact, elliott
            if key not in cache:
                cache[key] = declaration(*key)
            native = cache[key]
            observed, targets, errors = {}, {}, {}
            try:
                observed = molecular_eos_state(native, t, rho, xs, require_stable=False).__dict__
                observed = {**observed, **{f'mu_{i}': value for i, value in enumerate(
                    molecular_residual_chemical_potentials(native, t, rho, xs))},
                    **{f'Ar{i}{j}': molecular_scaled_derivative(native, t, rho, xs,
                        inverse_temperature_order=i, density_order=j) for i, j in ORDERS}}
            except Exception as exc:
                errors['native_exception'] = f'{type(exc).__name__}: {exc}'
            try:
                with mp.workdps(80):
                    targets = {k: float(v) for k, v in reference_state(*key, t, rho, xs).items()}
            except Exception as exc:
                errors['reference_exception'] = f'{type(exc).__name__}: {exc}'
            names = ['pressure_Pa', 'compressibility_factor', 'residual_helmholtz_J_mol',
                'residual_internal_energy_J_mol', 'residual_enthalpy_J_mol',
                'residual_entropy_J_mol_K',
                'residual_cv_J_mol_K', 'residual_cp_J_mol_K', 'dP_dT_rho_Pa_K',
                'dP_drho_T_Pa_m3_mol', 'mu_0', 'mu_1']+[f'Ar{i}{j}' for i, j in ORDERS]
            for name in names:
                atol = (1e-4 if name == 'pressure_Pa' else 2e-9 if name == 'compressibility_factor'
                        or name.startswith('Ar') else 2e-5 if name.endswith('_J_mol')
                        or name.startswith('mu_') else 2e-7)
                actual, target = observed.get(name), targets.get(name)
                passed = (actual is not None and target is not None and math.isfinite(actual)
                    and math.isfinite(target)
                    and math.isclose(actual, target, rel_tol=2e-7, abs_tol=atol))
                record = {'unit': unit, 'rule': rule, 'shape': shape, 'contact': contact,
                    'elliott': elliott, 'x': xs, 'T_K': t, 'rho_mol_m3': rho, 'quantity': name,
                    'native': actual, 'reference': target, 'rtol': 2e-7, 'atol': atol,
                    'passed': passed, **errors}
                stream.write(json.dumps(record)+'\n')
                count += 1
                if not passed:
                    failures.append(record)
            stream.flush()
            if unit % 4 == 0:
                elapsed = time.perf_counter()-started
                print(f'CPA GE {unit}/320; {unit/elapsed:.2f} units/s; '
                      f'ETA {(320-unit)*elapsed/unit:.1f}s; failures {len(failures)}', flush=True)
    summary = {'mode': 'development', 'units': len(grid), 'comparisons': count,
        'expected_comparisons': 7040, 'passed': count-len(failures), 'failed': len(failures),
        'failures': failures, 'elapsed_seconds': time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'failures'}))
    assert count == 7040
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
