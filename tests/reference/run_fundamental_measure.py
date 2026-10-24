"""Fixed public ThermoPack planar FMT and contact comparison."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from thermopack.saftvrmie import saftvrmie

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.fundamental_measure import FundamentalMeasureFunctional, fmt_contact_value


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    ref = saftvrmie("C1")
    rows = np.array([[0.7, 0.4, 0.9, eta, 0.03, -0.07] for eta in (0.05, 0.1, 0.2, 0.4, 0.7)])
    records = []

    def compare(unit, name, actual, target):
        record = {"unit": unit, "quantity": name, "rtol": 3e-7, "atol": 2e-9}
        if isinstance(actual, Exception) or isinstance(target, Exception):
            record.update(passed=False, error=f"native={actual!r}; source={target!r}")
        else:
            a, b = float(actual), float(target)
            record.update(
                native=a,
                reference=b,
                passed=math.isfinite(a)
                and math.isfinite(b)
                and math.isclose(a, b, rel_tol=3e-7, abs_tol=2e-9),
            )
        records.append(record)

    for family in ("RF", "WB", "WBII"):
        native = FundamentalMeasureFunctional(family).evaluate_many(
            rows, gradient=True, hessian=True
        )
        source = ref.fmt_energy_density(rows, phi_n=True, phi_nn=True, fmt_model=family)
        for i in range(len(rows)):
            unit = f"{family}-{i}"
            compare(unit, "phi", native[0][i], source[0][i])
            for j in range(6):
                compare(unit, f"grad{j}", native[1][i, j], source[1][i, j])
                for k in range(6):
                    compare(unit, f"hess{j},{k}", native[2][i, j, k], source[2][i, j, k])
    for i, row in enumerate(rows):
        source = ref.calc_bmcsl_gij_fmt(row, 0.2, calc_g_ij_n=True, mu_ij_T=0.017)
        values = {f"n{j}": v for j, v in enumerate(row)}
        compare(f"contact-{i}", "g", fmt_contact_value(row, 0.2), source[0])
        for j in range(6):
            actual = partial_derivative(
                lambda v: fmt_contact_value(tuple(v.values()), 0.2), values, (f"n{j}",)
            )
            compare(f"contact-{i}", f"grad{j}", actual, source[1][j])
        actual = (
            partial_derivative(
                lambda v, row=row: fmt_contact_value(row, v["mu"]), {"mu": 0.2}, ("mu",)
            )
            * 0.017
        )
        compare(f"contact-{i}", "g_T", actual, source[2])
    assert len(records) == 685
    failures = [r for r in records if not r["passed"]]
    summary = {
        "mode": "development",
        "units": 20,
        "comparisons": 685,
        "passed": 685 - len(failures),
        "failed": len(failures),
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "source": "ThermoPack 2.2.3",
        "weighted_densities_SI": rows.tolist(),
        "pair_length_m": 0.2,
        "pair_length_temperature_derivative_m_K": 0.017,
    }
    (output / "records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
