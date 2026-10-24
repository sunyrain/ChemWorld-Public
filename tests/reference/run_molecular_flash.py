"""Fixed analytic binodal/lever-rule comparison; no external runtime EOS import."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import log
from pathlib import Path

from scipy.optimize import brentq

from chemworld.physchem.molecular_flash import MolecularDensityBasin, molecular_tp_flash


@dataclass(frozen=True)
class SymmetricExcess:
    amplitude: float
    component_ids: tuple[str, ...] = ("first", "second")
    gas_constant_J_mol_K: float = 8.31446261815324

    def alphar(self, temperature, density, fractions):
        return self.amplitude*fractions[0]*fractions[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    basin = MolecularDensityBasin("homogeneous", (1, 100), 40)
    units = []
    for amplitude in (0., 1., 3.):
        endpoint = (brentq(lambda x, a=amplitude: log(x/(1-x))+a*(1-2*x), 1e-10, .49,
                          xtol=1e-14) if amplitude > 2 else None)
        for fraction in (.1, .3, .5, .7, .9):
            unit = {"amplitude": amplitude, "feed_first_fraction": fraction, "comparisons": []}
            try:
                result = molecular_tp_flash(SymmetricExcess(amplitude), 300, 1e5,
                    {"first": 2*fraction, "second": 2*(1-fraction)}, basins=(basin,),
                    composition_starts=((fraction, 1-fraction), (.5, .5), (.05, .95), (.95, .05)))
                unit["result"] = result.to_dict()
                expected_count = 1 if endpoint is None else 2
                expected_x = fraction if endpoint is None else endpoint
                expected_amount = 2 if endpoint is None else 2*(1-endpoint-fraction)/(1-2*endpoint)
                observations = (
                    ("phase_count", len(result.phases) if result.success else -1,
                     expected_count, 0),
                    ("first_endpoint", result.phases[0].state.mole_fractions[0]
                     if result.phases else None, expected_x, 2e-7),
                    ("first_amount_mol", result.phases[0].amount_mol if result.phases else None,
                     expected_amount, 2e-6),
                    ("material_error_mol", result.material_balance_error_mol, 0, 1e-12),
                    ("chemical_potential_error_RT", result.chemical_potential_error_RT, 0, 2e-7),
                    ("pressure_relative_error", result.pressure_relative_error, 0, 2e-10),
                    ("negative_post_tpd_RT", min(0., *(s.minimum_tpd_RT
                      for s in result.post_split_stability)) if result.post_split_stability else
                     min(0., result.stability.minimum_tpd_RT), 0., 1e-8))
                for name, value, expected, tolerance in observations:
                    unit["comparisons"].append({"name": name, "actual": value,
                        "expected": expected, "absolute_tolerance": tolerance,
                        "pass": value is not None and abs(value-expected) <= tolerance})
            except Exception as exc:
                unit["exception"] = f"{type(exc).__name__}: {exc}"
                unit["comparisons"] = [{"name": name, "pass": False, "exception": unit["exception"]}
                    for name in ("phase_count", "first_endpoint", "first_amount_mol",
                                 "material_error_mol", "chemical_potential_error_RT",
                                 "pressure_relative_error", "negative_post_tpd_RT")]
            units.append(unit)
            print(f"Molecular TP reference {len(units)}/15; "
                  f"A={amplitude:g}, z={fraction:g}; "
                  f"passed={sum(c['pass'] for c in unit['comparisons'])}/7", flush=True)
    records = [c for u in units for c in u["comparisons"]]
    report = {"scope": "development; fixed analytic symmetric-excess block",
        "unit_count": len(units), "comparison_count": len(records),
        "passed": sum(c["pass"] for c in records), "failed": sum(not c["pass"] for c in records),
        "units": units}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "units"}), flush=True)
    return int(bool(report["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
