"""Prospective 23G: five GE rules, analytic mixing, actual original EOS states."""

import argparse
import json
import time
from itertools import pairwise
from pathlib import Path

import mpmath as mp
from activity_reference_cases import model_case, mp_ge
from run_activity_thermodynamics_block import new_record
from run_ppr78_block import compare

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.cubic_mixing import (
    GibbsExcessMixingSpec,
    cubic_mixing_expression,
    cubic_residual_helmholtz,
)
from chemworld.physchem.eos import (
    CubicEOSSpec,
    EOSComponentSpec,
    cubic_state_derivatives,
    evaluate_cubic_eos,
)

RULES = ('huron_vidal','mhv','mhv1','mhv2','wong_sandler')
R = mp.mpf('8.31446261815324')


def declaration(rule, family, n):
    activity = model_case('nrtl', n)
    coefficients = ((-.53 if family=='peng_robinson' else -.593,) if rule=='mhv1' else
                    ((-.4347,-.003654) if family=='peng_robinson' else (-.478,-.0047))
                    if rule=='mhv2' else ())
    ids = activity.component_ids
    mixing = GibbsExcessMixingSpec(rule, activity, coefficients,
        {f'{a}|{b}':.04 for i,a in enumerate(ids) for b in ids[i+1:]}
        if rule=='wong_sandler' else {})
    components = tuple(EOSComponentSpec(key, tc, pc, omega) for key,tc,pc,omega in
        zip(ids, (500,580,650), (4e6,3.5e6,3e6), (.2,.4,.6), strict=False))
    return CubicEOSSpec('fixed23G',family,components,ge_mixing=mixing)


def reference_mix(spec,T,x):
    R = mp.mpf('8.31446261815324')
    pr = spec.model == 'peng_robinson'
    wa,wb = (mp.mpf('.45724'),mp.mpf('.07780')) if pr else (
        mp.mpf('.42747'),mp.mpf('.08664'))
    s1,s2 = (1+mp.sqrt(2),1-mp.sqrt(2)) if pr else (mp.mpf(1),mp.mpf(0))
    pure = []
    for component in spec.components:
        Tc,Pc,w = (mp.mpf(str(v)) for v in (component.critical_temperature_K,
                    component.critical_pressure_Pa,component.acentric_factor))
        kappa = (mp.mpf('.37464')+mp.mpf('1.54226')*w-mp.mpf('.26992')*w*w if pr else
                 mp.mpf('.480')+mp.mpf('1.574')*w-mp.mpf('.176')*w*w)
        pure.append((wa*R*R*Tc*Tc/Pc*(1+kappa*(1-mp.sqrt(T/Tc)))**2,wb*R*Tc/Pc))
    alpha = [a/(b*R*T) for a,b in pure]
    b = sum(z*pair[1] for z,pair in zip(x,pure,strict=True))
    mean = sum(z*a for z,a in zip(x,alpha,strict=True))
    excess = mp_ge(spec.ge_mixing.activity,T,x)/(R*T)
    volume = sum(z*mp.log(b/pair[1]) for z,pair in zip(x,pure,strict=True))
    C = -mp.log((1+s1)/(1+s2))/(s1-s2)
    rule = spec.ge_mixing.rule
    if rule == 'mhv1':
        am = mean+(excess+volume)/mp.mpf(str(spec.ge_mixing.coefficients[0]))
    elif rule == 'mhv2':
        q1,q2 = (mp.mpf(str(v)) for v in spec.ge_mixing.coefficients)
        target = excess+volume+q1*mean+q2*sum(z*a*a for z,a in zip(x,alpha,strict=True))
        am = (-q1-mp.sqrt(q1*q1+4*q2*target))/(2*q2)
    elif rule == 'mhv':
        def energy(value):
            u = (value-s1-s2-mp.sqrt((value-s1-s2)**2-4*(value+s1*s2)))/2
            return -1-mp.log(u-1)-value/(s1-s2)*mp.log((u+s1)/(u+s2))
        target = excess+volume+sum(z*energy(a) for z,a in zip(x,alpha,strict=True))
        am = mp.findroot(lambda value: energy(value)-target, (mean, mean*mp.mpf('1.001')))
    else:
        am = mean+excess/C
        if rule == 'wong_sandler':
            Q = sum(x[i]*x[j]*((bi-ai/(R*T))+(bj-aj/(R*T)))/2
                    *(1-(mp.mpf('.04') if i != j else 0))
                    for i,(ai,bi) in enumerate(pure) for j,(aj,bj) in enumerate(pure))
            b = Q/(1-am)
    return am*b*R*T,b


def reference_helmholtz(spec,T,V,x):
    a,b = reference_mix(spec,T,x)
    s1,s2 = (1+mp.sqrt(2),1-mp.sqrt(2)) if spec.model=='peng_robinson' else (1,0)
    return -mp.mpf('8.31446261815324')*T*mp.log(1-b/V)-a/(b*(s1-s2))*mp.log(
        (V+s1*b)/(V+s2*b))


def reference_roots(spec,T,P,x):
    a,b = reference_mix(spec,T,x)
    R = mp.mpf('8.31446261815324')
    A,B = a*P/(R*T)**2,b*P/(R*T)
    c2,c1,c0 = ((B-1,A-3*B*B-2*B,-A*B+B*B+B**3) if spec.model=='peng_robinson'
                else (-mp.mpf(1),A-B-B*B,-A*B))
    def polynomial(z):
        return z**3+c2*z*z+c1*z+c0
    disc = c2*c2-3*c1
    stationary = [] if disc<0 else [(-c2-mp.sqrt(disc))/3,(-c2+mp.sqrt(disc))/3]
    bounds = [B]+[s for s in stationary if s>B]+[max(2,2*(1+abs(c2)+abs(c1)+abs(c0)))]
    roots = []
    for low,high in pairwise(bounds):
        if polynomial(low)*polynomial(high)>0:
            continue
        for _ in range(300):
            mid = (low+high)/2
            if polynomial(low)*polynomial(mid)<=0:
                high = mid
            else:
                low = mid
        roots.append((low+high)/2)
    return roots


def reference_state(spec,T,P,x,phase):
    R = mp.mpf('8.31446261815324')
    roots = reference_roots(spec,T,P,x)
    Z = min(roots) if phase=='liquid' else max(roots)
    V = Z*R*T/P
    A = reference_helmholtz(spec,T,V,x)
    AT = mp.diff(lambda t: reference_helmholtz(spec,t,V,x),T)
    ATT = mp.diff(lambda t: reference_helmholtz(spec,t,V,x),T,2)
    ATV = mp.diff(lambda t,v: reference_helmholtz(spec,t,v,x),(T,V),(1,1))
    AVV = mp.diff(lambda v: reference_helmholtz(spec,T,v,x),V,2)
    PT,PV = R/V-ATV,-R*T/V**2-AVV
    H,S = A-T*AT+R*T*(Z-1),-AT+R*mp.log(Z)
    values = {'Z':Z,'V':V,'A':A,'H':H,'S':S,'G':H-T*S,
              'Cp':-T*ATT-T*PT*PT/PV-R,'dVdT':-PT/PV,'dVdP':1/PV}
    for index,key in enumerate(spec.component_ids):
        def extensive(ni, index=index):
            amounts = list(x)
            amounts[index] = ni
            total = sum(amounts)
            return total*reference_helmholtz(spec,T,V/total,[v/total for v in amounts])
        mu = mp.diff(extensive,x[index])
        values['phi_'+key] = mp.exp(mu/(R*T))/Z
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    units,start = [],time.monotonic()
    for rule in RULES:
        for family in ('peng_robinson','srk'):
            for n in (2,3):
                spec = declaration(rule,family,n)
                for T in (280.,340.,400.):
                    for xs in (((.25,.75),(.7,.3)) if n==2 else ((.2,.3,.5),(.65,.25,.1))):
                        record = new_record('mixing',rule,n,family=family,T=T,x=xs)
                        with mp.workdps(80):
                            temp = mp.mpf(str(T))
                            x = [mp.mpf(str(v)) for v in xs]
                            try:
                                for index,name in enumerate(('a','b')):
                                    def expression(v, index=index, spec=spec, xs=xs):
                                        return cubic_mixing_expression(spec,xs,v['T'])[index]
                                    for order in range(4):
                                        observed = (expression({'T':T}) if order==0 else
                                            partial_derivative(expression,{'T':T},('T',)*order))
                                        truth = mp.diff(lambda t,index=index,spec=spec,x=x:
                                            reference_mix(spec,t,x)[index],temp,order)
                                        compare(record,f'{name}_d{order}',observed,float(truth),
                                                atol=2e-8,rtol=2e-8)
                            except Exception as error:
                                record['failures'].append({'exception':type(error).__name__,
                                                          'message':str(error)})
                            units.append(record)
                            for P in (1e5,2e6):
                                for phase in ('liquid','vapor'):
                                    record = new_record('EOS',rule,n,family=family,T=T,x=xs,
                                                        P=P,phase=phase)
                                    try:
                                        composition = dict(zip(spec.component_ids,xs,strict=True))
                                        state = evaluate_cubic_eos(spec,composition,temperature_K=T,
                                            pressure_Pa=P,phase=phase)
                                        derivatives = cubic_state_derivatives(spec,composition,
                                            temperature_K=T,pressure_Pa=P,phase=phase)
                                        observed = {'Z':state.compressibility_factor,
                                            'V':state.molar_volume_m3_mol,
                                            'A':cubic_residual_helmholtz(spec,xs,T,
                                                state.molar_volume_m3_mol),
                                            'H':state.molar_residual_enthalpy_J_mol,
                                            'S':state.molar_residual_entropy_J_mol_K,
                                            'G':state.molar_residual_gibbs_J_mol,
                                            'Cp':derivatives['cp_departure_J_mol_K'],
                                            'dVdT':derivatives['d_molar_volume_dT_P'],
                                            'dVdP':derivatives['d_molar_volume_dP_T'],
                                            **{'phi_'+k:v for k,v in
                                               state.fugacity_coefficients.items()}}
                                        truth = reference_state(spec,temp,mp.mpf(str(P)),x,phase)
                                        for key,value in observed.items():
                                            compare(record,key,value,float(truth[key]),
                                                atol=2e-6 if key=='Cp' else 2e-8,
                                                rtol=2e-7 if key=='Cp' else 2e-8)
                                    except Exception as error:
                                        record['failures'].append({'exception':type(error).__name__,
                                                                  'message':str(error)})
                                    units.append(record)
                        elapsed = max(time.monotonic()-start,1e-9)
                        print(f'GE mixing states {len(units)}/600; {len(units)/elapsed:.2f}/s; '
                              f'ETA {(600-len(units))*elapsed/len(units):.1f}s',flush=True)
    failures = [{'unit':{k:v for k,v in u.items() if k!='failures'},'failure':f}
                for u in units for f in u['failures']]
    summary = {'block':'23G five GE cubic mixing rules','mode':'development',
               'expected_units':600,'expected_comparisons':6480,'units':units,
               'comparisons':sum(u['comparisons'] for u in units),'failures':failures,
               'elapsed_seconds':time.monotonic()-start}
    summary['passed'] = not failures and len(units)==600 and summary['comparisons']==6480
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print({k:v for k,v in summary.items() if k not in {'units','failures'}},flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
