# ruff: noqa: B023 -- all derivative callbacks finish within the current unit.
"""Fixed independent discriminator for the 72 exact-zero source derivatives."""

import argparse
import json
import math
import time
from itertools import product
from pathlib import Path

from run_pcsaft_association import models as association_models
from run_pcsaft_association import scalar

from chemworld.physchem.molecular_eos import molecular_residual_chemical_potentials


def run(output, polar=False, cpa=False, elliott=False):
    output.mkdir(parents=True, exist_ok=False)
    if cpa:
        from run_cpa import models as cpa_models

        def models(case, simplified):
            return cpa_models(case, 'srk',
                'Kontogeorgis' if simplified else 'Carnahan-Starling', 'thermopack',
                elliott=elliott)
    elif polar:
        from run_pcsaft_polar import models
    else:
        models = association_models
    cases = (2, 3) if cpa else (3, 4, 9) if polar else (4, 5, 11)
    densities = ((1, 100, 1000, 15000) if cpa else (1, 1000, 8000, 16000)
                 if polar else (1, 1000, 10000, 30000))
    grid = list(product(cases, (False, True), (300, 500, 800), densities))
    records, failures, cache = [], [], {}
    started = time.perf_counter()
    for unit, (case, simplified, t, rho) in enumerate(grid, 1):
        key = case, simplified
        if key not in cache:
            cache[key] = models(*key)
        model, ref, xs = cache[key][:3]
        tracer = xs.index(0)
        record = {'unit': unit, 'case': case, 'simplified': simplified,
                  'T_K': t, 'rho_mol_m3': rho}
        try:
            native = molecular_residual_chemical_potentials(model, t, rho, xs)[tracer]

            def amount(h):
                values = list(xs)
                values[tracer] += h
                return values

            def energy(h):
                return scalar(ref.helmholtz_tv(t, 1/rho, amount(h), property_flag='R'))

            def stencil(h):
                return math.fsum(c*energy(i*h) for i, c in enumerate((-25, 48, -36, 16, -3)))/(12*h)

            coarse, fine = stencil(1e-4), stencil(5e-5)
            extrapolated = (16*fine-coarse)/15
            tiny = ref.helmholtz_tv(t, 1/rho, amount(1e-8), dadn=True, property_flag='R')[1][tracer]
            zero = ref.helmholtz_tv(t, 1/rho, xs, dadn=True, property_flag='R')[1][tracer]
            record.update(native=native, forward_coarse=coarse, forward_fine=fine,
                          extrapolated=extrapolated, upstream_tiny_amount=float(tiny),
                          upstream_exact_zero=float(zero))
            for name, value in (('forward_difference', extrapolated), ('tiny_amount_limit', tiny)):
                passed = math.isfinite(value) and math.isclose(
                    native, value, rel_tol=2e-6, abs_tol=2e-3)
                record[name+'_passed'] = passed
                if not passed:
                    failures.append({**record, 'quantity': name})
        except Exception as exc:
            record['exception'] = f'{type(exc).__name__}: {exc}'
            failures.extend({**record, 'quantity': name}
                            for name in ('forward_difference', 'tiny_amount_limit'))
        records.append(record)
        if unit % 12 == 0:
            elapsed = time.perf_counter()-started
            print(f'Zero-component {unit}/{len(grid)}, {unit/elapsed:.2f} units/s, '
                  f'ETA {(len(grid)-unit)*elapsed/unit:.1f}s; failures {len(failures)}', flush=True)
    (output/'records.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    summary = {'mode': 'development', 'units': len(grid), 'comparisons': 2*len(grid),
        'passed': 2*len(grid)-len(failures), 'failed': len(failures), 'failures': failures,
        'rtol': 2e-6, 'atol_J_mol': 2e-3, 'elapsed_seconds': time.perf_counter()-started}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))
    return not failures


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--polar', action='store_true')
    mode.add_argument('--cpa', action='store_true')
    mode.add_argument('--cpa-elliott', action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if run(args.output, args.polar, args.cpa or args.cpa_elliott,
                             args.cpa_elliott) else 1)
