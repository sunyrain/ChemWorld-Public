# ruff: noqa: B023 -- fixed-unit callbacks are evaluated immediately.
"""Pinned pure potential, PeTS and soft-SAFT comparisons in explicit SI scales."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

import numpy as np
import teqp
from molecular_reference import thermopack_residual_targets
from run_saftvrmie import NAMES, ORDERS, teqp_targets
from thermopack.pets import pets

from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.molecular_potentials import PurePotentialModel, SoftSAFTModel
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL, SAFTSegment

PURE_CASES = [('LJ-Johnson-1993', None, 'LJ126_Johnson1993'),
              ('LJ-Kolafa-1994', None, 'LJ126_KolafaNezbeda1994'),
              ('LJ-Thol-2016', None, 'LJ126_TholJPCRD2016')]+[
    ('exp6-Kataoka-1992', a, 'EXP6_Kataoka1992') for a in (12, 14, 18)]+[
    ('square-well-2009', w, 'SW_EspindolaHeredia2009') for w in (1.25, 1.5, 2, 2.5)]
SOFT_PARAMETERS = (('A', 1, 3.4e-10, 120), ('B', 1.8, 3.8e-10, 180), ('C', 3, 4e-10, 220))
SOFT_CASES = [((i,), (1,)) for i in range(3)]+[
    ((0, 2), x) for x in ((1, 0), (0, 1), (.3, .7), (.7, .3))]+[
    ((0, 1, 2), x) for x in ((.2, .3, .5), (0, .4, .6))]
RHO_SCALE = 1/(AVOGADRO_PER_MOL*(3.4e-10)**3)


class ScaledReference:
    def __init__(self, model):
        self.model = model

    def get_Arxy(self, i, j, t, rho, x):  # noqa: N802 - reference API
        return self.model.get_Arxy(i, j, t/120, rho/RHO_SCALE, x)

    def get_R(self, x):  # noqa: N802 - reference API
        return R_J_PER_MOL_K

    def chemical_potentials(self, t, rho, x):
        return R_J_PER_MOL_K*120*self.model.build_Psir_gradient_autodiff(t/120, rho/RHO_SCALE*x)


def models(case, block):
    if block == 'softsaft':
        ids, fractions = SOFT_CASES[case]
        components = tuple(SAFTSegment(*SOFT_PARAMETERS[i], 'fixed synthetic scales') for i in ids)
        native = SoftSAFTModel(components, (60, 1200), 'fixed Johnson soft-SAFT')
        ref = teqp.make_model({'kind': 'SoftSAFT', 'model': {
            'm': [c.segment_count for c in components],
            'sigma / m': [c.diameter_m for c in components],
            'epsilon/kB / K': [c.well_depth_K for c in components]}})
        return native, ref, np.asarray(fractions, dtype=float)
    component = SAFTSegment('A', 1, 3.4e-10, 120, 'fixed synthetic scales')
    if block == 'pets':
        native = PurePotentialModel(component, 'PeTS-2018', (60, 1200), 'Heier2018')
        ref = pets()
        ref.s_set_pure_params = getattr(ref.tp, ref.get_export_name(
            'saft_interface', 'pets_set_pure_params'))
        ref.set_pure_params(1, component.diameter_m, component.well_depth_K)
        assert ref.get_pure_params() == (component.diameter_m, component.well_depth_K)
    else:
        family, shape, source_kind = PURE_CASES[case]
        native = PurePotentialModel(component, family, (60, 1200), 'fixed published model', shape)
        parameters = ({'alpha': shape} if family.startswith('exp6') else
                      {'lambda': shape} if family.startswith('square') else {})
        ref = ScaledReference(teqp.make_model({'kind': source_kind, 'model': parameters}))
    return native, ref, np.array([1.])


def run(output, block):
    output.mkdir(parents=True, exist_ok=False)
    if block == 'pure':
        grid = [('derivative', c, t*120, r*RHO_SCALE) for c, t, r in product(
            range(10), (.8, 1.3, 3), (0, 1e-8, .01, .3, .8, 1.1))]
        grid += [('state', c, 360, r*RHO_SCALE) for c, r in product(range(10), (.01, .3, .8))]
    elif block == 'softsaft':
        grid = [('derivative', c, t, r) for c, t, r in product(
            range(9), (150, 300, 600), (0, 1e-6, 100, 1000, 5000))]
        grid += [('state', c, 600, r) for c, r in product(range(9), (100, 1000, 5000))]
    else:
        grid = [('state', 0, t*120, r*RHO_SCALE) for t, r in product(
            (.8, 1.3, 3), (1e-8, .01, .3, .8, 1.1))]
        grid += [('inverse', 0, 360, r*RHO_SCALE) for r in (.01, .3, .8)]
    expected = {'pure': 2160, 'pets': 168, 'softsaft': 1698}[block]
    count, passed, failures, cache, metadata = 0, 0, [], {}, []
    started = time.perf_counter()

    def compare(name, actual, reference):
        nonlocal count, passed
        atol = (2e-9 if name.startswith('Ar') or name == 'compressibility_factor' else
                1e-4 if name == 'pressure_Pa' else 2e-5 if name.endswith('_J_mol')
                or name.startswith('mu_') else 2e-7)
        rec = {'quantity': name, 'rtol': 2e-7, 'atol': atol}
        for role, callback in (('native', actual), ('reference', reference)):
            try:
                v = float(callback())
                if not math.isfinite(v):
                    raise ValueError('nonfinite result')
                rec[role] = v
            except Exception as exc:
                rec[role+'_exception'] = f'{type(exc).__name__}: {exc}'
        rec['passed'] = ('native' in rec and 'reference' in rec and math.isclose(
            rec['native'], rec['reference'], rel_tol=2e-7, abs_tol=atol))
        count += 1
        passed += rec['passed']
        return rec

    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for unit, (kind, case, t, rho) in enumerate(grid, 1):
            if case not in cache:
                cache[case] = models(case, block)
                metadata.append({'case': case, 'model': cache[case][0].to_dict()})
            native, ref, x = cache[case]
            records = []
            if kind == 'derivative':
                records = [compare(f'Ar{i}{j}', lambda: molecular_scaled_derivative(native,
                    t, rho, x, inverse_temperature_order=i, density_order=j),
                    lambda: ref.get_Arxy(i, j, t, rho, x)) for i, j in ORDERS]
            else:
                actual, targets, errors = {}, {}, {}
                try:
                    actual = dict(molecular_eos_state(native, t, rho, x,
                                                      require_stable=False).__dict__)
                    actual.update({f'mu_{i}': v for i, v in enumerate(
                        molecular_residual_chemical_potentials(native, t, rho, x))})
                except Exception as exc:
                    errors['native'] = exc
                try:
                    if block == 'pets':
                        targets, mus = thermopack_residual_targets(ref, t, rho, x)
                    else:
                        targets = teqp_targets(ref, t, rho, x)
                        mus = (ref.chemical_potentials(t, rho, x) if block == 'pure' else
                               ref.build_Psir_gradient_autodiff(t, rho*x))
                    targets.update({f'mu_{i}': v for i, v in enumerate(mus)})
                except Exception as exc:
                    errors['reference'] = exc
                def value(name, source=False):
                    role = 'reference' if source else 'native'
                    if role in errors:
                        raise errors[role]
                    return (targets if source else actual)[name]
                if kind == 'state':
                    records += [compare(name, lambda: value(name), lambda: value(name, True))
                                for name in (*NAMES, *(f'mu_{i}' for i in range(len(x))))]
                if block != 'pets' or kind == 'inverse':
                    records.append(compare('pressure_density_inverse', lambda:
                        molecular_density_from_pressure(native, t, value('pressure_Pa', True), x,
                            initial_density_mol_m3=1.1*rho,
                            density_bounds_mol_m3=(.5*rho, 1.5*rho)).molar_density_mol_m3,
                        lambda: rho))
            row = {'unit': unit, 'kind': kind, 'case': case, 'T_K': t, 'rho_mol_m3': rho,
                   'comparisons': records}
            stream.write(json.dumps(row)+'\n')
            stream.flush()
            failures += [{**{k: v for k, v in row.items() if k != 'comparisons'}, **r}
                         for r in records if not r['passed']]
            if unit % 12 == 0 or unit == len(grid):
                elapsed = time.perf_counter()-started
                print(f'Potentials {block} {unit}/{len(grid)}; {unit/elapsed:.2f}units/s; '
                      f'ETA {(len(grid)-unit)*elapsed/unit:.1f}s; failures {len(failures)}',
                      flush=True)
    summary = {'mode': 'development', 'block': block, 'units': len(grid),
        'comparisons': count, 'expected_comparisons': expected, 'passed': passed,
        'failed': len(failures), 'failures': failures, 'model_inputs': metadata,
        'elapsed_seconds': time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('failures', 'model_inputs')}))
    assert count == expected
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--block', choices=('pure', 'pets', 'softsaft'), required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.block) else 1)
