"""Fixed 22G2 DIPPR mathematical block; all attempts and comparisons are retained."""

import argparse
import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import chemicals
import mpmath as mp
from chemicals import dippr
from test_native_dippr_reference import CASES, correlation

from chemworld.physchem.dippr import (
    dippr106_coefficients_from_point,
    dippr_mixed_derivative,
    dippr_parameter_jacobian,
    dippr_temperature_response,
)

TEMPERATURES = (240., 320., 400.)
JACOBIANS = {"101": "ABCDE", "102": "ABCD", "105": "ABCD", "106": "ABCDE", "107": "ABCDE"}
EXPECTED = {"high_precision": 1704, "public_derivatives": 234,
            "public_jacobians": 138, "local_recovery": 60}


def mp_expression(number, T, parameters):
    """Independent 80-digit transcription of the public mathematical definitions.

    Hyperbolic forms here are direct arbitrary-precision sinh/cosh, not the
    native stabilized exponentials or derivative arithmetic. No native kernel
    or native derivative is used to construct reference values.
    """
    c = parameters
    A, B, C, D, E, F, G = (c.get(key, mp.mpf(0)) for key in "ABCDEFG")
    base = number.removesuffix("_reciprocal")
    if base == "100":
        value = sum(c.get(key, 0)*T**i for i, key in enumerate("ABCDEFG"))
    elif base == "101":
        value = mp.exp(A+B/T+C*mp.log(T)+D*T**E)
    elif base == "102":
        value = A*T**B/(1+C/T+D/T**2)
    elif base == "104":
        value = A+B/T+C/T**3+D/T**8+E/T**9
    elif base == "105":
        value = A*B**(-(1+(1-T/C)**D))
    elif base == "106":
        r = T/c["Tc"]
        value = A*(1-r)**(B+C*r+D*r*r+E*r**3)
    elif base == "107":
        value = A+B*(C/T/mp.sinh(C/T))**2+D*(E/T/mp.cosh(E/T))**2
    elif base == "114":
        tau = 1-T/c["Tc"]
        value = (A**2/tau+B-2*A*C*tau-A*D*tau**2-C**2*tau**3/3
                 -C*D*tau**4/2-D**2*tau**5/5)
    elif base == "115":
        value = mp.exp(A+B/T+C*mp.log(T)+D*T*T+E/T**2)
    elif base == "116":
        tau = 1-T/c["Tc"]
        value = A+B*tau**mp.mpf(".35")+C*tau**(mp.mpf(2)/3)+D*tau+E*tau**(mp.mpf(4)/3)
    elif base == "127":
        value = A+sum(a*(theta/T)**2*mp.exp(theta/T)/mp.expm1(theta/T)**2
                      for a, theta in ((B, C), (D, E), (F, G)))
    else:
        raise ValueError(f"unmapped reference {number}")
    return 1/value if number.endswith("_reciprocal") else value


def run(output):
    if output.exists():
        raise ValueError("choose a new attempt directory; previous results are not overwritten")
    output.mkdir(parents=True)
    started = time.perf_counter()
    mp.mp.dps = 80
    rows = []
    inputs = []
    assert chemicals.__version__ == "1.5.2"
    methods = ({"EQ"+number for number in CASES}
               | {"EQ"+number+"_fitting_jacobian" for number in JACOBIANS}
               | {"EQ106_AB", "EQ106_ABC"})
    assert methods == set(dippr.__all__)

    def check(group, label, actual, expected, *, atol=1e-11):
        row = {"group": group, "id": label, "absolute_tolerance": atol,
               "relative_tolerance": 2e-9}
        try:
            a, e = float(actual()), float(expected())
            error = abs(a-e)
            row.update(actual=a if mp.isfinite(a) else str(a),
                       expected=e if mp.isfinite(e) else str(e),
                       absolute_error=error if mp.isfinite(error) else str(error),
                       passed=bool(mp.isfinite(a) and mp.isfinite(e)
                                   and error <= atol+2e-9*abs(e)))
        except Exception as exc:  # One failed comparison must not hide the remaining fixed block.
            row.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)

    state_count = 0
    for number in CASES:
        for parameter_set in (0, 1):
            card = correlation(number, parameter_set)
            coeffs = {name: mp.mpf(str(value)) for name, value in card.coefficients.items()}
            for temperature in TEMPERATURES:
                T = mp.mpf(temperature)
                label = f"{number}/{parameter_set}/{temperature:g}"
                inputs.append({"id": label, "temperature_K": temperature,
                               "correlation": card.to_dict()})
                # Bind values explicitly so each callable describes its own fixed state.
                def reference(t, number=number, coeffs=coeffs):
                    return mp_expression(number, t, coeffs)

                def response(card=card, temperature=temperature):
                    return dippr_temperature_response(card, temperature_K=temperature).values

                public = getattr(dippr, "EQ"+number)
                for order in range(4):
                    check("high_precision", f"{label}/dT{order}",
                          lambda order=order: response()[order],
                          lambda order=order, T=T: mp.diff(reference, T, order))
                orders = [0, 1]
                if number in {"101", "105", "106", "115"}:
                    orders += [2, 3]
                elif number.endswith("_reciprocal"):
                    orders += [2]
                for order in orders:
                    check("public_derivatives", f"{label}/dT{order}",
                          lambda order=order: response()[order],
                          lambda order=order, public=public, card=card, temperature=temperature:
                              public(temperature, **card.coefficients, order=order))
                for name in card.coefficients:
                    def reference_parameter(t, value, name=name, number=number, coeffs=coeffs):
                        return mp_expression(number, t, {**coeffs, name: value})
                    for temperature_order in range(3):
                        coords = ("temperature",)*temperature_order+(name,)
                        check("high_precision", f"{label}/{'/'.join(coords)}",
                            lambda card=card, temperature=temperature, coords=coords:
                                dippr_mixed_derivative(card, temperature_K=temperature,
                                                       coordinates=coords).value,
                            lambda o=temperature_order, name=name, T=T, coeffs=coeffs:
                                mp.diff(reference_parameter, (T, coeffs[name]), (o, 1)))
                state_count += 1
        elapsed = time.perf_counter()-started
        print(f"DIPPR states {state_count}/84, comparisons {len(rows)}, "
              f"{state_count/max(elapsed, 1e-6):.1f} states/s", flush=True)
    for number, names in JACOBIANS.items():
        for parameter_set in (0, 1):
            card = correlation(number, parameter_set)
            for i, temperature in enumerate(TEMPERATURES):
                for j, name in enumerate(names):
                    check("public_jacobians", f"{number}/{parameter_set}/{temperature:g}/{name}",
                        lambda card=card, names=names, i=i, j=j: dippr_parameter_jacobian(
                            card, TEMPERATURES, parameters=names).values[i][j],
                        lambda number=number, card=card, i=i, j=j: getattr(
                            dippr, "EQ"+number+"_fitting_jacobian")(
                                list(TEMPERATURES), **card.coefficients)[i][j])
    for form in ("AB", "ABC"):
        for parameter_set in (0, 1):
            original = correlation("106", parameter_set)
            coeffs = {**original.coefficients, "C": 0. if form == "AB" else
                      original.coefficients["C"], "D": 0., "E": 0.}
            card = replace(original, coefficients=coeffs)
            for temperature in TEMPERATURES:
                values = tuple(dippr.EQ106(temperature, **coeffs, order=i) for i in range(3))
                def recover(temperature=temperature, coeffs=coeffs, values=values, form=form):
                    return dippr106_coefficients_from_point(temperature_K=temperature,
                        critical_temperature_K=coeffs["Tc"], value=values[0],
                        first_derivative=values[1],
                        second_derivative=None if form == "AB" else values[2])

                reference_coefficients = getattr(dippr, "EQ106_"+form)(
                    temperature, coeffs["Tc"], *values[:len(form)])
                label = f"{form}/{parameter_set}/{temperature:g}"
                for i, name in enumerate(form):
                    check("local_recovery", f"{label}/{name}",
                          lambda name=name: recover()[name],
                          lambda i=i, ref=reference_coefficients: ref[i], atol=1e-8)
                for order in range(len(form)):
                    check("local_recovery", f"{label}/reconstructed-dT{order}",
                        lambda card=card, temperature=temperature, order=order:
                            dippr_temperature_response(replace(card, coefficients=recover()),
                                                        temperature_K=temperature).values[order],
                        lambda order=order, values=values: values[order])
    counts = dict(Counter(row["group"] for row in rows))
    failures = [row for row in rows if not row["passed"]]
    report = {"schema": "chemworld-conv22g2-dippr-1", "mode": "development",
        "reference_versions": {"chemicals": chemicals.__version__, "mpmath": mp.__version__},
        "public_methods": sorted(methods), "expected_comparisons": EXPECTED,
        "completed_comparisons": counts, "states": state_count, "inputs": inputs,
        "comparisons": rows, "failures": failures,
        "passed": not failures and counts == EXPECTED and state_count == 84,
        "seconds": time.perf_counter()-started,
        "boundary": "synthetic mathematical qualification, not a data fit or material validation"}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                                         encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("passed", "states", "completed_comparisons", "seconds")}), flush=True)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
