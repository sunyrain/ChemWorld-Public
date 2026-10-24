# ruff: noqa: B023 -- all source/native callbacks execute within the current unit.
"""Fixed dual-convention 457-unit /6022-comparison polar block."""

import argparse
import json
import math
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import teqp
from molecular_reference import thermopack_residual_targets
from thermopack.pcsaft import pcsaft

from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_residual_chemical_potentials,
    molecular_scaled_derivative,
)
from chemworld.physchem.pcsaft import PCSAFTModel, SAFTSegment
from chemworld.physchem.saft_association import AssociationSite, SAFTAssociation
from chemworld.physchem.saft_polar import SAFTMultipole, SAFTPolar

CASES = [((c,), (1,)) for c in ('DME', 'ACETONE', 'CO2')]
CASES += [(('DME', 'CO2'), x) for x in ((1, 0), (0, 1), (.3, .7), (.7, .3))]
CASES += [(('DME', 'ACETONE'), (.4, .6))]
CASES += [(('DME', 'CO2', 'C1'), x) for x in ((.2, .3, .5), (0, .4, .6), (0, 0, 1))]
CASES += [(('H2O',), (1,)), (('H2O',), (1,))]
MOMENTS = {'DME': (1.3, 0), 'ACETONE': (2.88, 0), 'CO2': (0, 3.9546), 'C1': (0, 0)}
ORDERS = [(i, j) for i in range(4) for j in range(4-i)]


def models(case, simplified):
    ids, xs = CASES[case]
    parameter_reference = ('Rehner2020_3B_polar' if case == 11 else 'Rehner2020_4C_polar'
        if case == 12 else 'Gross2005ADJQ' if 'CO2' in ids else 'Gross2006')
    reference = pcsaft(','.join(ids), parameter_reference=parameter_reference,
                       simplified=simplified, polar=True)
    params = [reference.get_pure_fluid_param(i+1) for i in range(len(ids))]
    association = None
    if case in (11, 12):
        polarities = (1, 1, -1) if case == 11 else (-1, -1, 1, 1)
        assert reference.get_n_assoc_sites() == len(polarities)
        energy, volume = params[0][3]/reference.Rgas, params[0][4]*params[0][1]**3
        association = SAFTAssociation(
            tuple(AssociationSite('H2O', str(k)) for k in range(len(polarities))),
            tuple(tuple(energy if p*q < 0 else 0 for q in polarities) for p in polarities),
            tuple(tuple(volume if p*q < 0 else 0 for q in polarities) for p in polarities),
            'Explicit '+parameter_reference)
    else:
        assert reference.get_n_assoc_sites() == 0
    interactions = tuple(tuple(reference.get_kij(i+1, j+1) for j in range(len(ids)))
                         for i in range(len(ids)))
    components = tuple(SAFTSegment(c, *par[:3], 'Explicit ThermoPack input')
                       for c, par in zip(ids, params, strict=True))
    moments = []
    for c, (m, sigma, epsilon, _, _) in zip(ids, params, strict=True):
        mu, quad = ((1.9374 if case == 11 else 1.505, 0) if c == 'H2O' else MOMENTS[c])
        moments.append(SAFTMultipole(c, 1e-49*mu**2/(m*1.380649e-23*epsilon*sigma**3),
            1e-69*quad**2/(m*1.380649e-23*epsilon*sigma**5), 'Explicit published moment'))
    polar = SAFTPolar(tuple(moments), 'Fixed source comparison')
    model = PCSAFTModel(components, interactions, (200, 1000), 'Fixed polar comparison',
        gas_constant_J_mol_K=reference.Rgas, simplified=simplified, polar=polar,
        association=association)
    return model, reference, xs


def teqp_model(native):
    spec = {'coeffs': [dict(name=c.component_id, m=c.segment_count,
        sigma_Angstrom=c.diameter_m*1e10, epsilon_over_k=c.well_depth_K, BibTeXKey='explicit',
        **{'(mu^*)^2': p.reduced_dipole_squared, '(Q^*)^2': p.reduced_quadrupole_squared,
           'nmu': p.dipole_count, 'nQ': p.quadrupole_count})
        for c, p in zip(native.components, native.polar.moments, strict=True)],
        'kmat': [list(row) for row in native.interaction]}
    return teqp.make_model({'kind': 'PCSAFT', 'model': spec})


def run(output, association_block=False):
    output.mkdir(parents=True, exist_ok=False)
    cases = (11, 12) if association_block else range(11)
    densities = (1, 1000, 10000, 30000) if association_block else (1e-6, 1, 1000, 8000, 16000)
    grid = [('state', c, s, t, r) for c, s, t, r in product(
        cases, (False, True), (300, 500, 800), densities)]
    grid += [('inverse', c, s, 800, 1000) for c, s in product(cases, (False, True))]
    if not association_block:
        grid += [('teqp', c, False, t, r) for c, t, r in product(
            (0, 1, 2, 5, 6, 7, 8), (300, 500, 800), (1e-6, 1, 1000, 8000, 16000))]
    expected = sum(13+len(CASES[c][0]) if k == 'state' else 1 if k == 'inverse' else 10
                   for k, c, *_ in grid)
    assert (len(grid), expected) == ((52, 676) if association_block else (457, 6022))
    cache, metadata, failures = {}, [], []
    comparisons, passed, started = 0, 0, time.perf_counter()

    def compare(name, native, upstream, rtol=2e-7, atol=2e-9):
        nonlocal comparisons, passed
        record = {'quantity': name, 'rtol': rtol, 'atol': atol}
        for role, function in (('native', native), ('reference', upstream)):
            try:
                value = float(function())
                if not math.isfinite(value):
                    raise ValueError('nonfinite comparison')
                record[role] = value
            except Exception as exc:
                record[role+'_exception'] = f'{type(exc).__name__}: {exc}'
        record['passed'] = ('native' in record and 'reference' in record and math.isclose(
            record['native'], record['reference'], rel_tol=rtol, abs_tol=atol))
        comparisons += 1
        passed += record['passed']
        return record

    with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
        for index, (kind, case, simplified, t, rho) in enumerate(grid, 1):
            key = case, simplified, kind == 'teqp'
            if key not in cache:
                model, ref, xs = models(case, simplified)
                if kind == 'teqp':
                    model = replace(model, polar=replace(model.polar,
                                    convention='teqp-0.23.2', contributions=('DD', 'QQ')))
                    ref = teqp_model(model)
                cache[key] = model, ref, xs
                metadata.append({'case': case, 'model': model.to_dict()})
            model, ref, xs = cache[key]
            records = []
            if kind == 'teqp':
                for i, j in ORDERS:
                    records.append(compare(f'Ar{i}{j}', lambda: molecular_scaled_derivative(
                        model, t, rho, xs, inverse_temperature_order=i, density_order=j),
                        lambda: ref.get_Arxy(i, j, t, rho, np.asarray(xs, dtype=float))))
            else:
                ref.polar_model_control(qq=True, dd=True, dq=True)
                native_error = reference_error = None
                try:
                    state = molecular_eos_state(model, t, rho, xs, require_stable=False)
                    chemical = molecular_residual_chemical_potentials(model, t, rho, xs)
                except Exception as exc:
                    native_error = exc
                try:
                    targets, mus = thermopack_residual_targets(ref, t, rho, xs)
                except Exception as exc:
                    reference_error = exc

                def native_value(name):
                    if native_error:
                        raise native_error
                    return getattr(state, name)

                def reference_value(name):
                    if reference_error:
                        raise reference_error
                    return targets[name]

                def chemical_value(i, native=True):
                    error = native_error if native else reference_error
                    if error:
                        raise error
                    return chemical[i] if native else mus[i]

                if kind == 'inverse':
                    records.append(compare('pressure_density_inverse', lambda:
                        molecular_density_from_pressure(
                            model, t, reference_value('pressure_Pa'), xs,
                            initial_density_mol_m3=1.1*rho,
                            density_bounds_mol_m3=(.5*rho, 1.5*rho)).molar_density_mol_m3,
                        lambda: rho, rtol=2e-9))
                else:
                    for name in ('pressure_Pa', 'compressibility_factor',
                        'residual_helmholtz_J_mol', 'residual_internal_energy_J_mol',
                        'residual_enthalpy_J_mol', 'residual_entropy_J_mol_K',
                        'residual_cv_J_mol_K', 'residual_cp_J_mol_K',
                        'dP_dT_rho_Pa_K', 'dP_drho_T_Pa_m3_mol'):
                        atol = (1e-4 if name == 'pressure_Pa' else 2e-9 if name ==
                            'compressibility_factor' else 2e-5 if name.endswith('_J_mol') else 2e-7)
                        records.append(compare(name, lambda: native_value(name),
                                               lambda: reference_value(name), atol=atol))
                    for i in range(len(xs)):
                        records.append(compare(f'mu_residual_{i}', lambda: chemical_value(i),
                                               lambda: chemical_value(i, False), atol=2e-5))
                    for term in ('DD', 'QQ', 'DQ'):
                        records.append(compare(term, lambda: model.contributions(t, rho, xs)[term],
                            lambda: ref.fres_polar(t, 1/rho, xs,
                                dd=term == 'DD', qq=term == 'QQ', dq=term == 'DQ')))
            unit = {'unit': index, 'kind': kind, 'case': case, 'simplified': simplified,
                    'T_K': t, 'rho_mol_m3': rho, 'comparisons': records}
            stream.write(json.dumps(unit)+'\n')
            failures.extend({**{k: v for k, v in unit.items() if k != 'comparisons'}, **r}
                            for r in records if not r['passed'])
            if index % 12 == 0 or index == len(grid):
                elapsed = time.perf_counter()-started
                print(f'Polar {index}/{len(grid)}, {index/elapsed:.2f} units/s, '
                      f'ETA {(len(grid)-index)*elapsed/index:.1f}s; '
                      f'failures {len(failures)}', flush=True)
    summary = {'mode': 'development', 'references': ['teqp 0.23.2', 'ThermoPack 2.2.3'],
        'units': len(grid), 'comparisons': comparisons, 'expected_comparisons': expected,
        'passed': passed, 'failed': len(failures), 'failures': failures, 'model_inputs': metadata,
        'elapsed_seconds': time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('failures', 'model_inputs')}))
    assert comparisons == expected
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--association-block', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.association_block) else 1)
