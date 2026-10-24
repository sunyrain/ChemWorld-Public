# ruff: noqa: B023 -- callbacks evaluated within each fixed unit.
"""Fixed Dufal blocks, keeping molecular and segment conventions distinct."""

import argparse
import json
import math
import time
from ctypes import POINTER, c_double, c_int
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import teqp
from molecular_reference import thermopack_residual_targets
from run_saftvrmie import NAMES, ORDERS, teqp_targets
from thermopack.saftvrmie import saftvrmie

from chemworld.physchem.dufal_association import DufalAssociation
from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.saft_association import AssociationSite, SAFTAssociation

PARAMS = (('H2O', 1, 3.0555e-10, 418, 35.823, 1600, 496.66e-30, ('e', 'e', 'H', 'H')),
          ('MEOH', 1.7989, 3.1425e-10, 276.92, 16.968, 2156, 222.18e-30, ('e', 'H', 'H')),
          ('NH3', 1, 3.3309e-10, 323.70, 36.832, 1105, 560.73e-30, ('e', 'H', 'H', 'H')))
CASES = [((i,), (1,)) for i in range(3)]+[
    ((0, 1), x) for x in ((1, 0), (0, 1), (.3, .7), (.7, .3))]+[
    ((0, 1, 2), x) for x in ((.2, .3, .5), (0, .4, .6))]


@dataclass
class IsolatedAssociation:
    association: DufalAssociation
    gas_constant_J_mol_K: float = R_J_PER_MOL_K

    @property
    def component_ids(self):
        return self.association.component_ids

    def alphar(self, t, rho, x):
        return self.association.alphar(t, rho, x)


@dataclass
class ReferenceDifference:
    full: object
    nonpolar: object

    def get_Arxy(self, i, j, t, rho, x):  # noqa: N802 - pinned reference API
        return self.full.get_Arxy(i, j, t, rho, x)-self.nonpolar.get_Arxy(i, j, t, rho, x)

    def get_R(self, x):  # noqa: N802 - pinned reference API
        return self.full.get_R(x)

    def chemical_potentials(self, t, rho, x):
        return (self.full.build_Psir_gradient_autodiff(t, rho*x)
                -self.nonpolar.build_Psir_gradient_autodiff(t, rho*x))


def models(case, backend, source_topology=False):
    indices, x = CASES[case]
    ps = [PARAMS[i] for i in indices]
    sites = tuple(AssociationSite(p[0], label, p[7].count(label)) for p in ps
                  for label in ('e', 'H'))
    owners = [ps[i//2] for i in range(len(sites))]
    def bonded(i, j):
        return i % 2 != j % 2 or (source_topology and i//2 != j//2
                                 and (owners[i][0] == 'NH3' or owners[j][0] == 'NH3'))
    energy = tuple(tuple(math.sqrt(a[5]*b[5]) if bonded(i, j) else 0
                         for j, b in enumerate(owners)) for i, a in enumerate(owners))
    volume = tuple(tuple(((a[6]**(1/3)+b[6]**(1/3))/2)**3 if bonded(i, j) else 0
                         for j, b in enumerate(owners)) for i, a in enumerate(owners))
    pair_epsilon = tuple(tuple(math.sqrt(a[3]*b[3])*(a[2]*b[2])**1.5/((a[2]+b[2])/2)**3
                               for b in ps) for a in ps)
    pair_lambda = tuple(tuple(3+math.sqrt((a[4]-3)*(b[4]-3)) for b in ps) for a in ps)
    assoc = DufalAssociation(tuple(p[0] for p in ps), tuple(p[1] for p in ps),
        tuple(p[2] for p in ps), pair_epsilon, pair_lambda,
        SAFTAssociation(sites, energy, volume, 'Fixed Dufal HB, explicit cross-volume erratum'),
        'Dufal2015 fixed parameters', 'molecular' if backend == 'teqp' else 'segment')
    interaction = [[0.]*len(ps) for _ in ps]
    if backend == 'teqp':
        coeffs = [{'name': p[0], 'm': p[1], 'sigma_m': p[2], 'epsilon_over_k': p[3],
                   'lambda_r': p[4], 'lambda_a': 6, 'BibTeXKey': 'Dufal2015'} for p in ps]
        nonpolar = {'kind': 'SAFT-VR-Mie', 'model': {'coeffs': coeffs, 'kmat': interaction}}
        spec = {'sigma / m': [p[2] for p in ps], 'epsilon/kB / K': [p[3] for p in ps],
            'lambda_r': [p[4] for p in ps], 'kmat': interaction,
            'epsilon_HB / J/mol': [p[5]*R_J_PER_MOL_K for p in ps],
            'K_HB / m^3': [p[6] for p in ps], 'molecule_sites': [list(p[7]) for p in ps],
            'Delta_rule': 'Dufal',
            'options': {'alpha': .5, 'rtol': 1e-13, 'atol': 1e-13, 'max_iters': 1000}}
        full = teqp.make_model({'kind': 'genericSAFT', 'model': {'nonpolar': nonpolar,
            'association': {'kind': 'Dufal', 'model': spec}}}, validate=False)
        reference = ReferenceDifference(full, teqp.make_model(nonpolar, validate=False))
    else:
        reference = saftvrmie(','.join(p[0] for p in ps), parameter_reference='Dufal2015')
        # Correct only the reference wrapper's getter ABI; its setter argtypes
        # are erroneously assigned by the shipped getter. Values remain unchanged.
        reference.s_get_active_assoc_params.argtypes = [POINTER(c_int), POINTER(c_double),
                                                       POINTER(c_double)]
        for i, p in enumerate(ps, 1):
            original = reference.get_pure_fluid_param(i)
            hb, vol = reference.get_pure_assoc_param(i)
            assert np.allclose(original, (p[1], p[2], p[3], 6, p[4]), rtol=2e-12, atol=0)
            assert math.isclose(hb, p[5]*R_J_PER_MOL_K, rel_tol=2e-12)
            assert math.isclose(vol, p[6], rel_tol=2e-12)
        for i in range(len(ps)):
            for j in range(i+1, len(ps)):
                reference.set_eps_kij(i+1, j+1, 0.)
                reference.set_sigma_lij(i+1, j+1, 0.)
                reference.set_lr_gammaij(i+1, j+1, 0.)
        for control in (reference.model_control_hard_sphere, reference.model_control_a1,
                        reference.model_control_a2, reference.model_control_a3,
                        reference.model_control_chain):
            control(False)
    return IsolatedAssociation(assoc), reference, np.array(x, dtype=float)


def run(output, backend, source_topology=False):
    output.mkdir(parents=True, exist_ok=False)
    if backend == 'teqp':
        grid = [('derivative', c, t, r) for c, t, r in product(
            range(9), (300, 450, 600), (0, 1, 1000, 15000))]
        grid += [('state', c, 600, r) for c, r in product(range(9), (1, 1000))]
    else:
        grid = [('state', c, t, r) for c, t, r in product(
            range(9), (300, 450, 600), (1, 1000, 15000))]
        grid += [('inverse', c, 600, 1000) for c in range(9)]
    expected = 1312 if backend == 'teqp' else 972
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
                cache[case] = models(case, backend, source_topology)
                metadata.append({'case': case, 'model': cache[case][0].association.to_dict()})
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
                    if backend == 'teqp':
                        targets = teqp_targets(ref, t, rho, x)
                        mus = ref.chemical_potentials(t, rho, x)
                    else:
                        targets, mus = thermopack_residual_targets(ref, t, rho, x)
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
                if backend == 'teqp' or kind == 'inverse':
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
                print(f'Dufal {backend} {unit}/{len(grid)}; {unit/elapsed:.2f}units/s; '
                      f'ETA {(len(grid)-unit)*elapsed/unit:.1f}s; failures {len(failures)}',
                      flush=True)
    summary = {'mode': 'development', 'block': backend, 'units': len(grid),
        'comparisons': count, 'expected_comparisons': expected, 'passed': passed,
        'failed': len(failures), 'failures': failures, 'model_inputs': metadata,
        'scope': 'ideal_gas_plus_isolated_association',
        'source_topology_diagnostic': source_topology,
        'elapsed_seconds': time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('failures', 'model_inputs')}))
    assert count == expected
    return not failures


def delta_diagnostic(output):
    """Public source matrix against explicit symmetric Dufal pair strengths."""
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in range(9):
        native, ref, x = models(case, 'teqp')
        public = ref.full.get_assoc_calcs(300, 1000, x)
        ps = [PARAMS[i] for i in CASES[case][0]]
        contacts = native.association.contacts(300, 1000, x)
        owners = [item[1][0] for item in public['to_CompSite']]
        expected = [[math.expm1(math.sqrt(ps[i][5]*ps[j][5])/300)
            *((ps[i][6]**(1/3)+ps[j][6]**(1/3))/2)**3*contacts[i][j]
            for j in owners] for i in owners]
        delta = np.asarray(public['Delta'])
        rows.append({'case': case, 'T_K': 300, 'rho_mol_m3': 1000,
            'public_source': public, 'declared_symmetric_pair_Delta_m3': expected,
            'max_source_asymmetry_m3': float(np.max(np.abs(delta-delta.T))),
            'max_pair_discrepancy_m3': float(np.max(np.abs(delta-expected)))})
    (output/'summary.json').write_text(json.dumps({'scope': 'diagnostic_only',
        'units': 9, 'records': rows}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--backend', choices=('teqp', 'thermopack'), required=True)
    parser.add_argument('--thermopack-cross-site-diagnostic', action='store_true')
    parser.add_argument('--delta-diagnostic', action='store_true')
    args = parser.parse_args()
    if args.delta_diagnostic:
        delta_diagnostic(args.output)
        raise SystemExit(0)
    if args.thermopack_cross_site_diagnostic and args.backend != 'thermopack':
        parser.error('cross-site diagnostic applies only to ThermoPack')
    raise SystemExit(0 if run(args.output, args.backend,
                             args.thermopack_cross_site_diagnostic) else 1)
