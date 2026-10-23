"""Complete PPR78/EPPR78 source interactions and original EOS consumer block."""

import argparse
import json
import math
import time
from itertools import pairwise
from pathlib import Path

import mpmath as mp
from run_activity_thermodynamics_block import new_record
from thermo.group_contribution import ppr78 as reference

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.eos import (
    CubicEOSSpec,
    EOSComponentSpec,
    cubic_mixture_parameters,
    evaluate_cubic_eos,
)
from chemworld.physchem.ppr78 import PPR78Spec, ppr78_interaction_response


def compare(record, name, value, target, *, atol, rtol):
    error = abs(value-target)
    tolerance = atol+rtol*abs(target)
    record['comparisons'] += 1
    record['max_absolute_error'] = max(record['max_absolute_error'], error)
    if not math.isfinite(value) or error > tolerance:
        record['failures'].append({'quantity':name, 'actual':value, 'expected':target,
                                   'absolute_error':error, 'tolerance':tolerance})


def independent(T, A, B):
    R = mp.mpf('8.31446261815324')
    wa, wb = mp.mpf('0.4572355289213821893834601962251837888504'), mp.mpf(
        '0.0777960739038884559718447100373331839711')
    pure = []
    for tc, pc, omega in ((500, 4000000, mp.mpf('.2')), (650, 3000000, mp.mpf('.6'))):
        kappa = (mp.mpf('.37464')+mp.mpf('1.54226')*omega-mp.mpf('.26992')*omega**2
                 if omega <= mp.mpf('.491') else mp.mpf('.379642')+mp.mpf('1.48503')*omega
                 -mp.mpf('.164423')*omega**2+mp.mpf('.016666')*omega**3)
        pure.append((wa*R*R*tc*tc/pc*(1+kappa*(1-mp.sqrt(T/tc)))**2, wb*R*tc/pc))
    (ai, bi), (aj, bj) = pure
    group = 0 if A == 0 else mp.mpf('1e6')*A*(mp.mpf('298.15')/T)**(B/A-1)
    kij = (group-(mp.sqrt(ai)/bi-mp.sqrt(aj)/bj)**2)/(2*mp.sqrt(ai*aj)/(bi*bj))
    return kij, ai, aj, bi, bj


def high_state(A, B):
    T, P, R, x = mp.mpf(700), mp.mpf(100000), mp.mpf('8.31446261815324'), mp.mpf('.35')

    def mixing(temp):
        kij, ai, aj, bi, bj = independent(temp, A, B)
        cross = mp.sqrt(ai*aj)*(1-kij)
        return x*x*ai+2*x*(1-x)*cross+(1-x)**2*aj, x*bi+(1-x)*bj, cross, ai, aj, bi, bj

    a, b, cross, ai, aj, bi, bj = mixing(T)
    da = mp.diff(lambda temp: mixing(temp)[0], T)
    shape = mp.sqrt(2)
    reduced_a, reduced_b = a*P/(R*T)**2, b*P/(R*T)
    c2, c1, c0 = (reduced_b-1, reduced_a-3*reduced_b**2-2*reduced_b,
                  -reduced_a*reduced_b+reduced_b**2+reduced_b**3)
    def polynomial(z):
        return z**3+c2*z*z+c1*z+c0
    discriminant = c2*c2-3*c1
    stationary = ([] if discriminant < 0 else
                  [(-c2-mp.sqrt(discriminant))/3, (-c2+mp.sqrt(discriminant))/3])
    points = [reduced_b]+[v for v in stationary if v>reduced_b]+[
        max(2, 2*(1+abs(c2)+abs(c1)+abs(c0)))]
    roots = []
    for low, high in pairwise(points):
        if polynomial(low)*polynomial(high) > 0:
            continue
        for _ in range(300):
            midpoint = (low+high)/2
            if polynomial(low)*polynomial(midpoint) <= 0:
                high = midpoint
            else:
                low = midpoint
        roots.append((low+high)/2)
    Z = max(roots)
    volume = Z*R*T/P
    logratio = mp.log((volume+(1+shape)*b)/(volume+(1-shape)*b))
    phi = [mp.exp(bb/b*(Z-1)-mp.log(Z-P*b/(R*T))
                  -a/(2*shape*b*R*T)*(2*attraction/a-bb/b)*logratio)
           for bb, attraction in ((bi, x*ai+(1-x)*cross), (bj, x*cross+(1-x)*aj))]
    H = R*T*(Z-1)+(T*da-a)/(2*shape*b)*logratio
    S = R*mp.log(Z-P*b/(R*T))+da/(2*shape*b)*logratio
    return (a, da, b, Z, *phi, H, S)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--domain-aware', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    units, start = [], time.monotonic()
    components = (EOSComponentSpec('a', 500, 4e6, .2), EOSComponentSpec('b', 650, 3e6, .6))
    sources = {}
    for variant, rows in (('original', reference.PPR78_INTERACTIONS_BY_ID),
                          ('extended', reference.EPPR78_INTERACTIONS_BY_ID)):
        converted = {f'{i}|{j}': pair for (i,j), pair in rows.items()}
        sources[variant] = len(rows)
        first = None
        for index, ((i, j), (A, B)) in enumerate(rows.items(), 1):
            groups = {'a': {str(i): 1}, 'b': {str(j): 1}}
            model = PPR78Spec(variant, groups, converted,
                              DataSourceProvenance(f'PPR78-{variant}', 0,
                                                   'Published PPR78 interaction table',
                                                   'thermo0.6'))
            first = model if first is None else first
            with mp.workdps(80):
                # Identical single groups have no group difference even on a self row.
                a, b = (mp.mpf(str(v)) for v in (A, B))
                for T in (275., 400., 650.):
                    result = ppr78_interaction_response(model, *components, temperature_K=T)
                    ref = reference.PPR78_kij(T, {i:1}, {j:1}, 500, 4e6, .2,
                                              650, 3e6, .6, version=variant, string=False)
                    record = new_record('interaction', variant, 2, row=index, temperature=T)
                    compare(record, 'k_ij', result['k_ij'], ref, atol=2e-8, rtol=2e-8)
                    for order, key in enumerate(('dk_dT','d2k_dT2','d3k_dT3'), 1):
                        truth = mp.diff(lambda t, a=a, b=b: independent(t,a,b)[0], mp.mpf(T), order)
                        compare(record, key, result[key], float(truth), atol=2e-8, rtol=2e-8)
                    units.append(record)
                eos = CubicEOSSpec('predictive', 'peng_robinson', components,
                                   pure_parameterization='pr78', predictive_interaction=model)
                record = new_record('EOS', variant, 2, row=index)
                try:
                    vals = independent(mp.mpf(700), a, b)
                    k, ai, aj = vals[:3]
                    reference_a = mp.mpf('.35')**2*ai+2*mp.mpf('.35')*mp.mpf('.65')*mp.sqrt(
                        ai*aj)*(1-k)+mp.mpf('.65')**2*aj
                    if args.domain_aware and reference_a < 0:
                        record['kind'] = 'EOS_domain_rejection'
                        record['reference_a_mix'] = float(reference_a)
                        try:
                            cubic_mixture_parameters(eos, {'a':.35,'b':.65}, temperature_K=700)
                        except ValueError:
                            compare(record, 'negative_attraction_rejected', 1, 1, atol=0, rtol=0)
                        else:
                            compare(record, 'negative_attraction_rejected', 0, 1, atol=0, rtol=0)
                        units.append(record)
                        continue
                    mix = cubic_mixture_parameters(eos, {'a':.35,'b':.65}, temperature_K=700)
                    state = evaluate_cubic_eos(eos, {'a':.35,'b':.65},
                                                temperature_K=700, pressure_Pa=1e5)
                    observed = (mix.a_mix, mix.da_mix_dT, mix.b_mix, state.compressibility_factor,
                                state.fugacity_coefficients['a'], state.fugacity_coefficients['b'],
                                state.molar_residual_enthalpy_J_mol,
                                state.molar_residual_entropy_J_mol_K)
                    for key, value, truth in zip(('a','da','b','Z','phi_a','phi_b','H','S'),
                                                  observed, high_state(a,b), strict=True):
                        compare(record, key, value, float(truth), atol=2e-8, rtol=2e-8)
                except Exception as error:
                    record['failures'].append({'exception':type(error).__name__,
                                               'message':str(error)})
                units.append(record)
            if index % 100 == 0 or index == len(rows):
                elapsed = max(time.monotonic()-start, 1e-9)
                complete = (225 if variant == 'extended' else 0)+index
                print(f'PPR source rows {complete}/1825; {complete/elapsed:.1f}/s; '
                      f'ETA {(1825-complete)*elapsed/complete:.1f}s', flush=True)
        record = new_record('roundtrip', variant, 2)
        compare(record, 'JSON', int(PPR78Spec.from_dict(json.loads(json.dumps(
            first.to_dict()))).to_dict() == first.to_dict()), 1, atol=0, rtol=0)
        units.append(record)
    failures = [{'unit':{k:v for k,v in u.items() if k != 'failures'}, 'failure':f}
                for u in units for f in u['failures']]
    summary = {'block':'23F PPR78 all source rows and actual EOS', 'mode':'development',
               'source_rows':sources, 'expected_units':7302,
               'expected_comparisons':35942 if args.domain_aware else 36502,
               'units':units, 'comparisons':sum(u['comparisons'] for u in units),
               'failures':failures, 'elapsed_seconds':time.monotonic()-start}
    summary['domain_rejection_count'] = sum(u['kind']=='EOS_domain_rejection' for u in units)
    summary['domain_aware'] = args.domain_aware
    summary['passed'] = (not failures and len(units)==7302
                         and summary['comparisons']==summary['expected_comparisons'])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print({k:v for k,v in summary.items() if k not in {'units','failures'}}, flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
