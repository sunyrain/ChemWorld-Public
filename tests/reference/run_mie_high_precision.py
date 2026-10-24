"""Independent pure Mie arithmetic for normal and extreme dense states."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

import mpmath as mp
from run_saftvrmie import NAMES, ORDERS, PARAMETERS, models

from chemworld.physchem.molecular_eos import (
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)

PHI_ROWS = (
    '7.5365557 -359.44 1550.9 -1.19932 -1911.28 9236.9',
    '-37.60463 1825.6 -5070.1 9.063632 21390.175 -129430',
    '71.745953 -3168.0 6534.6 -17.9482 -51320.7 357230',
    '-46.83552 1884.2 -3288.7 11.34027 37064.54 -315530',
    '-2.467982 -.82376 -2.7171 20.52142 1103.742 1390.2',
    '-.50272 -3.1935 2.0883 -56.6377 -3264.61 -4518.2',
    '8.0956883 3.7090 0 40.53683 2556.181 4241.6')
EFFECTIVE_ROWS = ('0.81096 1.7888 -37.578 92.284', '1.0205 -19.341 151.26 -463.50',
                  '-1.9057 22.845 -228.14 973.92', '1.0885 -6.1962 106.98 -677.64')


def pure_expression(case, temperature):
    _, m0, sigma0, e0, lr0 = PARAMETERS[case]
    m, sigma, eps, lr = (mp.mpf(str(v)) for v in (m0, sigma0*1e10, e0, lr0))
    la, T0 = mp.mpf(6), mp.mpf(str(temperature))
    C = lr/(lr-la)*(lr/la)**(la/(lr-la))
    alpha = C*(1/(la-3)-1/(lr-3))
    phi = [[mp.mpf(v) for v in row.split()] for row in PHI_ROWS]
    ceff = [[mp.mpf(v) for v in row.split()] for row in EFFECTIVE_ROWS]
    f = [sum(phi[k][q]*alpha**k for k in range(4))
         /(1+sum(phi[k][q]*alpha**(k-3) for k in range(4, 7))) for q in range(6)]
    diameters = []
    for order in range(4):
        def integrand(r, order=order):
            if r == 0:
                return mp.mpf(1) if order == 0 else mp.mpf(0)
            v = C*eps/T0*(r**(-lr)-r**(-la))
            if order == 0:
                return -mp.expm1(-v)
            polynomial = -v if order == 1 else 2*v-v*v if order == 2 else -6*v+6*v*v-v**3
            return mp.exp(-v)*polynomial/T0**order
        diameters.append(sigma*mp.quad(integrand, [0, .5, .8, .95, 1]))

    def ar(T, rho):
        d = sum(v*(T-T0)**k/mp.factorial(k) for k, v in enumerate(diameters))
        rhos = rho*mp.mpf('6.02214076e-7')*m
        z = mp.pi/6*rhos*d**3
        zb = mp.pi/6*rhos*sigma**3
        x = sigma/d
        def term(lam, z):
            ze = sum(sum(ceff[k][q]/lam**q for q in range(4))*z**(k+1) for k in range(4))
            integral_i = (1-x**(3-lam))/(lam-3)
            integral_j = (1-(lam-3)*x**(4-lam)+(lam-4)*x**(3-lam))/((lam-3)*(lam-4))
            return x**lam*((1-z/2)*integral_i/(1-z)**3
                          -mp.mpf('4.5')*z*(1+z)*integral_j/(1-z)**3
                          -(1-ze/2)/((lam-3)*(1-ze)**3))
        fa, fr = term(la, z), term(lr, z)
        faa, far, frr = term(2*la, z), term(la+lr, z), term(2*lr, z)
        def compressibility(z):
            return (1-z)**4/(1+4*z+4*z*z-4*z**3+z**4)
        chi = f[0]*zb+f[1]*zb**5+f[2]*zb**8
        mono = m*((4*z-3*z*z)/(1-z)**2+2*mp.pi*rhos*d**3*eps*C*(fa-fr)/T
            +mp.pi*rhos*d**3*eps**2*C*C*compressibility(z)*(1+chi)*(faa-2*far+frr)/T**2
            -eps**3*f[3]*zb*mp.exp(f[4]*zb+f[5]*zb*zb)/T**3)
        k0 = -mp.log(1-z)+(42*z-39*z*z+9*z**3-2*z**4)/(6*(1-z)**3)
        k1, k2 = (z**4+6*z*z-12*z)/(2*(1-z)**3), -3*z*z/(8*(1-z)**2)
        k3 = (-z**4+3*z*z+3*z)/(6*(1-z)**3)
        loggd = k0+k1*x+k2*x*x+k3*x**3
        g1 = C*(3*mp.diff(lambda zz: zz*(term(la, zz)-term(lr, zz)), z)-la*fa+lr*fr)
        g2 = C*C*(mp.mpf('1.5')*mp.diff(lambda zz: zz*compressibility(zz)*(
            term(2*la, zz)-2*term(la+lr, zz)+term(2*lr, zz)), z)
            -compressibility(z)*(la*faa-(la+lr)*far+lr*frr))
        gamma = 10*(1-mp.tanh(10*(mp.mpf('.57')-alpha)))*zb*mp.expm1(eps/T)*mp.exp(
            -mp.mpf('6.7')*zb-8*zb*zb)
        return mono-(m-1)*(loggd+(eps/T*g1+(eps/T)**2*(1+gamma)*g2)*mp.exp(-loggd))
    return ar


def targets(expression, t, rho):
    R, t, rho = mp.mpf('8.31446261815324'), mp.mpf(t), mp.mpf(rho)
    b = 1/t
    ds = {(i, j): b**i*rho**j*mp.diff(lambda beta, r: expression(1/beta, r),
                                      (b, rho), (i, j)) for i, j in ORDERS}
    z = 1+ds[0, 1]
    stiffness, thermal = 1+2*ds[0, 1]+ds[0, 2], z-ds[1, 1]
    values = (rho*R*t*z, z, R*t*ds[0, 0], R*t*ds[1, 0], R*t*(ds[1, 0]+z-1),
        R*(ds[1, 0]-ds[0, 0]), -R*ds[2, 0], R*(-ds[2, 0]+thermal**2/stiffness-1),
        rho*R*thermal, R*t*stiffness)
    return {**dict(zip(NAMES, values, strict=True)), **{f'Ar{i}{j}': v for (i, j), v in ds.items()},
            'mu_0': R*t*(ds[0, 0]+ds[0, 1])}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    failures, count, cache = [], 0, {}
    start = time.perf_counter()
    with (output/'records.jsonl').open('w', encoding='utf-8') as stream, mp.workdps(80):
        for unit, (case, t, rho) in enumerate(product(range(4), (180, 300, 600), (1000, 16000)), 1):
            key = case, t
            if key not in cache:
                cache[key] = pure_expression(case, t)
            native = models(case, 'Lafitte', 'thermopack', False)[0]
            errors, actual, expected = {}, {}, {}
            try:
                actual = dict(molecular_eos_state(
                    native, t, rho, (1,), require_stable=False).__dict__)
                actual.update({f'Ar{i}{j}': molecular_scaled_derivative(native, t, rho, (1,),
                    inverse_temperature_order=i, density_order=j) for i, j in ORDERS})
                actual['mu_0'] = molecular_residual_chemical_potentials(native, t, rho, (1,))[0]
            except Exception as exc:
                errors['native_exception'] = f'{type(exc).__name__}: {exc}'
            try:
                expected = {k: float(v) for k, v in targets(cache[key], t, rho).items()}
            except Exception as exc:
                errors['reference_exception'] = f'{type(exc).__name__}: {exc}'
            for name in (*NAMES, *(f'Ar{i}{j}' for i, j in ORDERS), 'mu_0'):
                a, b = actual.get(name), expected.get(name)
                atol = (1e-4 if name == 'pressure_Pa' else 2e-5 if name.endswith('_J_mol')
                    or name.startswith('mu_') else 2e-9 if name.startswith('Ar')
                    or name == 'compressibility_factor' else 2e-7)
                passed = (a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
                          and math.isclose(a, b, rel_tol=2e-7, abs_tol=atol))
                row = {'unit': unit, 'case': case, 'T_K': t, 'rho_mol_m3': rho,
                    'quantity': name, 'native': a, 'reference': b, 'rtol': 2e-7, 'atol': atol,
                    'passed': passed, **errors}
                stream.write(json.dumps(row)+'\n')
                count += 1
                if not passed:
                    failures.append(row)
            stream.flush()
            elapsed = time.perf_counter()-start
            print(f'Mie high precision {unit}/24; {unit/elapsed:.2f}units/s; '
                  f'ETA {(24-unit)*elapsed/unit:.1f}s; failures {len(failures)}', flush=True)
    summary = {'mode': 'development', 'units': 24, 'comparisons': count,
        'expected_comparisons': 504,
        'passed': count-len(failures), 'failed': len(failures), 'failures': failures,
        'elapsed_seconds': time.perf_counter()-start}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    assert count == 504
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
