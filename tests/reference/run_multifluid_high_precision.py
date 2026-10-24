"""Independent 60-digit synthetic multifluid derivative/caloric qualification."""

import argparse
import itertools
import json
import time
from pathlib import Path

import mpmath as mp
from tests.test_multifluid import synthetic_model

from chemworld.physchem.molecular_eos import molecular_scaled_derivative


def m(value):
    return mp.mpf(str(value))


def residual(model, temperature, density, fractions):
    # Standalone literal equations. Do not call native reducing/term evaluators.
    tr = sum(
        x * x * m(f.reducing_temperature_K) for x, f in zip(fractions, model.fluids, strict=True)
    )
    vr = sum(
        x * x / m(f.reducing_molar_density_mol_m3)
        for x, f in zip(fractions, model.fluids, strict=True)
    )
    for pair in model.binary_pairs:
        i, j = [model.component_ids.index(k) for k in pair.component_ids]
        x, y = fractions[i], fractions[j]
        fi, fj = model.fluids[i], model.fluids[j]
        tr += (
            2
            * m(pair.beta_T)
            * m(pair.gamma_T)
            * mp.sqrt(m(fi.reducing_temperature_K) * m(fj.reducing_temperature_K))
            * x
            * y
            * (x + y)
            / (m(pair.beta_T) ** 2 * x + y)
        )
        v = (
            m(fi.reducing_molar_density_mol_m3) ** (-mp.mpf(1) / 3)
            + m(fj.reducing_molar_density_mol_m3) ** (-mp.mpf(1) / 3)
        ) ** 3 / 8
        vr += (
            2
            * m(pair.beta_v)
            * m(pair.gamma_v)
            * v
            * x
            * y
            * (x + y)
            / (m(pair.beta_v) ** 2 * x + y)
        )
    delta, tau = density * vr, tr / temperature
    result = mp.mpf(0)
    for x, fluid in zip(fractions, model.fluids, strict=True):
        for term in fluid.residual_terms:
            c = term.coefficients
            for n, d, t, ell in zip(c["n"], c["d"], c["t"], c["l"], strict=True):
                result += (
                    x
                    * m(n)
                    * delta ** m(d)
                    * tau ** m(t)
                    * mp.exp(-(delta ** m(ell)) if ell else 0)
                )
    for pair in model.binary_pairs:
        i, j = [model.component_ids.index(k) for k in pair.component_ids]
        for term in pair.departure_terms:
            exponent = (
                (-(delta ** m(term.l)) if term.l else 0)
                if term.kind == "power"
                else (
                    -m(term.eta) * (delta - m(term.epsilon)) ** 2
                    - m(term.beta) * (delta - m(term.gamma))
                )
            )
            result += (
                fractions[i]
                * fractions[j]
                * m(pair.F)
                * m(term.n)
                * delta ** m(term.d)
                * tau ** m(term.t)
                * mp.exp(exponent)
            )
    return result


def total_a(model, temperature, density, fractions):
    ideal = mp.mpf(0)
    for x, f in zip(fractions, model.fluids, strict=True):
        lead, heat = f.ideal_terms
        tc = m(f.reducing_temperature_K)
        ai = mp.log(density / m(f.reducing_molar_density_mol_m3)) + m(lead.coefficients["a1"])
        ai += m(lead.coefficients["a2"]) * tc / temperature + m(heat.coefficients["a"]) * mp.log(
            tc / temperature
        )
        ideal += x * (ai + mp.log(x))
    return (
        m(model.gas_constant_J_mol_K)
        * temperature
        * (ideal + residual(model, temperature, density, fractions))
    )


def compare_state(model, xs, t, rho, compare, case):
    mx = tuple(map(m, xs))
    beta, r = 1 / m(t), m(rho)
    for i in range(4):
        for j in range(4 - i):
            expected = (
                beta**i
                * r**j
                * mp.diff(
                    lambda b, density: residual(model, 1 / b, density, mx),
                    (beta, r),
                    (i, j),
                )
            )
            compare(
                case,
                f"Ar{i}{j}",
                molecular_scaled_derivative(
                    model, t, rho, xs, inverse_temperature_order=i, density_order=j
                ),
                expected,
            )
    for i, component in enumerate(model.component_ids[:-1]):

        def fn(x, i=i):
            values = list(mx)
            values[-1] += values[i] - x
            values[i] = x
            return residual(model, m(t), r, values)

        for order in (1, 2, 3):
            compare(
                case,
                f"simplex_{component}_{order}",
                model.composition_derivative(t, rho, xs, (component,) * order),
                mp.diff(fn, mx[i], order),
            )
    a = total_a(model, m(t), r, mx)
    at = mp.diff(lambda temp: total_a(model, temp, r, mx), m(t))
    att = mp.diff(lambda temp: total_a(model, temp, r, mx), m(t), 2)
    p = r * r * mp.diff(lambda density: total_a(model, m(t), density, mx), r)
    pt = r * r * mp.diff(lambda temp, density: total_a(model, temp, density, mx), (m(t), r), (1, 1))
    pr = mp.diff(
        lambda density: density * density * mp.diff(lambda d: total_a(model, m(t), d, mx), density),
        r,
    )
    targets = {
        "helmholtz_energy_J_mol": a,
        "entropy_J_mol_K": -at,
        "internal_energy_J_mol": a - m(t) * at,
        "enthalpy_J_mol": a - m(t) * at + p / r,
        "cv_J_mol_K": -m(t) * att,
        "cp_J_mol_K": -m(t) * att + m(t) * pt * pt / (r * r * pr),
        "gibbs_energy_J_mol": a + p / r,
    }
    actual = model.caloric_state(t, rho, xs)
    for name, target in targets.items():
        compare(case, name, getattr(actual, name), target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite previous attempt")
    mp.mp.dps = 60
    rows, errors = [], []
    start = time.monotonic()

    def compare(case, name, actual, expected):
        a, b = float(actual), float(expected)
        rows.append(
            {
                "case": case,
                "quantity": name,
                "native": a,
                "reference": b,
                "passed": abs(a - b) <= 2e-10 + 2e-10 * abs(b),
            }
        )

    for n in (2, 3):
        model = synthetic_model(n)
        xs = (0.25, 0.75) if n == 2 else (0.2, 0.3, 0.5)
        for t, rho in itertools.product((280.0, 400.0), (200.0, 1000.0)):
            case = [n, t, rho]
            try:
                compare_state(model, xs, t, rho, compare, case)
            except Exception as exc:
                errors.append({"case": case, "error": repr(exc)})
            print(
                f"mpmath multifluid {case}: {len(rows)}/172 comparisons; "
                f"{time.monotonic() - start:.1f}s",
                flush=True,
            )
    result = {
        "mode": "development",
        "reference": "mpmath " + mp.__version__,
        "digits": 60,
        "expected": 172,
        "executed": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "errors": errors,
        "elapsed_s": time.monotonic() - start,
        "comparisons": rows,
    }
    result["complete_and_passed"] = (
        len(rows) == 172 and not errors and all(r["passed"] for r in rows)
    )
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}, indent=2))


if __name__ == "__main__":
    main()
