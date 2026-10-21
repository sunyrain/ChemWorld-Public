"""Fixed 160-unit D2B block; public API and independent high-precision equations."""
# ruff: noqa: C408 -- explicit scientific parameters

import argparse
import json
import math
import time
import traceback
from itertools import product
from pathlib import Path

import chemicals
import mpmath as mp
from chemicals import volume as reference

from chemworld.physchem import (
    mixture_volume_composition_jacobian,
    mixture_volume_derivatives,
    mixture_volume_value,
    volume_derivatives,
    volume_parameter_jacobian,
    volume_pressure_derivatives,
    volume_value,
)

TCS = (500, 600, 700)
PCS = (4e6, 5e6, 6e6)
VCS = (2e-4, 3e-4, 4e-4)
OMEGAS = (0.1, 0.2, 0.4)
MWS = (50, 80, 100)
ZRS = (0.25, 0.27, 0.29)
XS = ((1, 0, 0), (0, 1, 0), (0.2, 0.3, 0.5), (0.7, 0, 0.3))
TS = (250, 300, 350)
PS = (1e7, 5e7, 1e8)


def cases():
    rows = []

    def add(method, t, p, xs=None, example=False):
        rows.append(dict(id=f"V{len(rows)+1:03d}", method=method, T=t,
                         parameters=p, xs=xs, example=example))

    for method, xs, t in product(("Rackett_mixture", "COSTALD_mixture"), XS, TS):
        p = (dict(Tcs=TCS, Pcs=PCS, MWs=MWS, Zrs=ZRS) if method == "Rackett_mixture"
             else dict(Tcs=TCS, Vcs=VCS, omegas=OMEGAS))
        add(method, t, p, xs)
    for xs, k in product(XS, (1, 2, 3)):
        add("Amgat", 300, dict(Vms=tuple(i * 1e-4 for i in (k, k+1, k+2))), xs)
    for xs, t, estimate, pressure in product(XS, TS, (False, True), PS):
        p = dict(Tcs=TCS, Vcs=VCS, omegas=OMEGAS, P=pressure)
        p.update(dict(saturation_pressure_model="generalized_riedel") if estimate
                 else dict(Psat=1e5))
        add("COSTALD_mixture_compressed", t, p, xs)
    for i, t, pressure in product(range(3), TS, PS):
        add("COSTALD_compressed", t, dict(Tc=TCS[i], Pc=PCS[i], omega=OMEGAS[i],
                                          P=pressure, Psat=1e5, Vs=(1, 1.5, 2)[i]*1e-4))
    for method, bc, pressure in product(("Tait", "Tait_molar"),
                                      ((1e7, .05), (1e8, .1), (2e8, .15)), PS):
        add(method, 300, dict(P=pressure, P_ref=1e5, B=bc[0], C=bc[1],
                            **(dict(rho_ref=800) if method == "Tait" else dict(V_ref=1e-4))))
    add("Amgat", 300, dict(Vms=(4.057e-5, 5.861e-5)), (.5, .5), True)
    add("Rackett_mixture", 298, dict(MWs=(32.04, 18.01), Tcs=(512.58, 647.29),
        Pcs=(8.096e6, 2.209e7), Zrs=(.2332, .2374)), (.4576, .5424), True)
    add("COSTALD_mixture", 298, dict(Tcs=(512.58, 647.29), Vcs=(.000117, 5.6e-5),
        omegas=(.559, .344)), (.4576, .5424), True)
    add("COSTALD_compressed", 303, dict(P=9.8e7, Psat=85857.9, Tc=466.7, Pc=3640000.,
        omega=.281, Vs=.000105047), example=True)
    add("COSTALD_mixture_compressed", 344.26111, dict(Tcs=(305.3278, 617.594),
        Vcs=(.00014576928794529767, .0006192229409547785), omegas=(.0983, .4916),
        P=20684271.8795, saturation_pressure_model="generalized_riedel"), (.2, .8), True)
    add("Tait", 300, dict(P=1e8, P_ref=101325, rho_ref=784.85,
                          B=79337060., C=.099102), example=True)
    add("Tait_molar", 300, dict(P=1e8, P_ref=101325, V_ref=4.0825e-5,
                                B=79337060., C=.099102), example=True)
    assert len(rows) == 160
    return rows


def convert(value):
    if isinstance(value, dict):
        return {k: convert(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return tuple(convert(v) for v in value)
    return value if isinstance(value, str) else mp.mpf(str(value))


def mathematical(method, t, p, xs=None):
    """Independent literal transcription, using 60-digit mpmath arithmetic."""
    r = mp.mpf("8.31446261815324")

    def sat(tc, vc, w):
        tr = t/tc
        tau = 1-tr
        v0 = 1 - mp.mpf("1.52816")*tau**(mp.mpf(1)/3)
        v0 += mp.mpf("1.43907")*tau**(mp.mpf(2)/3) - mp.mpf(".81446")*tau
        v0 += mp.mpf(".190454")*tau**(mp.mpf(4)/3)
        vd = (mp.mpf("-.296123") + mp.mpf(".386914")*tr - mp.mpf(".0427258")*tr**2
              - mp.mpf(".0480645")*tr**3)/(tr-mp.mpf("1.00001"))
        return vc*v0*(1-w*vd)

    def dense(tc, pc, w, psat, vs):
        tau = 1-t/tc
        e = mp.exp(mp.mpf("4.79594") + mp.mpf(".250047")*w + mp.mpf("1.14188")*w**2)
        b = pc*(-1 - mp.mpf("9.070217")*tau**(mp.mpf(1)/3)
                + mp.mpf("62.45326")*tau**(mp.mpf(2)/3)
                - mp.mpf("135.1102")*tau + e*tau**(mp.mpf(4)/3))
        c = mp.mpf(".0861488") + mp.mpf(".0344483")*w
        return vs*(1-c*mp.log((b+p["P"])/(b+psat)))

    if method == "Amgat":
        return sum(x*v for x, v in zip(xs, p["Vms"], strict=True))
    if method.startswith("Tait"):
        factor = 1-p["C"]*mp.log((p["B"]+p["P"])/(p["B"]+p["P_ref"]))
        return p["rho_ref"]/factor if method == "Tait" else p["V_ref"]*factor
    if method == "COSTALD_compressed":
        return dense(p["Tc"], p["Pc"], p["omega"], p["Psat"], p["Vs"])
    if method == "Rackett_mixture":
        tc = sum(x*v for x, v in zip(xs, p["Tcs"], strict=True))
        zr = sum(x*v for x, v in zip(xs, p["Zrs"], strict=True))
        mw = sum(x*v for x, v in zip(xs, p["MWs"], strict=True))
        return r*mw*zr**(1+(1-t/tc)**(mp.mpf(2)/7))*sum(
            x*tc/(mw*pc) for x, tc, mw, pc in zip(xs, p["Tcs"], p["MWs"], p["Pcs"], strict=True))
    vc = (sum(x*v for x, v in zip(xs, p["Vcs"], strict=True)) + 3*sum(
        x*v**(mp.mpf(2)/3) for x, v in zip(xs, p["Vcs"], strict=True))*sum(
        x*v**(mp.mpf(1)/3) for x, v in zip(xs, p["Vcs"], strict=True)))/4
    # Double sum deliberately retains the published pair expression.
    tc = sum(xs[i]*xs[j]*mp.sqrt(p["Vcs"][i]*p["Tcs"][i]*p["Vcs"][j]*p["Tcs"][j])
             for i in range(len(xs)) for j in range(len(xs)))/vc
    w = sum(x*v for x, v in zip(xs, p["omegas"], strict=True))
    vs = sat(tc, vc, w)
    if method == "COSTALD_mixture":
        return vs
    pc = (mp.mpf(".291")-mp.mpf(".080")*w)*r*tc/vc
    psat = p.get("Psat")
    if psat is None:
        tr = t/tc
        alpha = 35-36/tr-mp.mpf("96.736")*mp.log10(tr)+tr**6
        beta = mp.log10(tr)+mp.mpf(".03721754")*alpha
        psat = pc*10**(mp.mpf("5.8031817")*mp.log10(tr)+mp.mpf(".07608141")*alpha
                       + w*mp.mpf("4.86601")*beta)
    return dense(tc, pc, w, psat, vs)


def upstream(method, t, p, xs=None):
    p = {k: v for k, v in p.items() if k != "saturation_pressure_model"}
    f = getattr(reference, method)
    if xs is not None:
        return f(xs=xs, **p, **({} if method == "Amgat" else {"T": t}))
    return f(**p, **({} if method.startswith("Tait") else {"T": t}))


def difference(f, x, h):
    return (f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)


def run_unit(case):
    method, t, p, xs = (case[k] for k in ("method", "T", "parameters", "xs"))
    comparisons = []

    def check(label, actual, expected, *, derivative=False, fd=False):
        actual, expected = float(actual), float(expected)
        atol = (max(1e-30, abs(expected)*1e-12) if expected else 1e-25) if derivative else 1e-13
        rtol = 2e-8 if derivative else 3e-10
        if fd:
            rtol, atol = 2e-6, 1e-12
        passed = (math.isfinite(actual) and math.isfinite(expected)
                  and abs(actual-expected) <= atol+rtol*abs(expected))
        comparisons.append(dict(label=label, actual=actual, reference=expected,
            absolute_error=abs(actual-expected), rtol=rtol, atol=atol, passed=passed))

    try:
        value = (mixture_volume_value(method, t, xs, **p) if xs is not None
                 else volume_value(method, t, **p))
        check("public_API_value", value, upstream(method, t, p, xs))
        if not case["example"]:
            mt, pm = convert(t), convert(p)
            xm = None if xs is None else convert(xs)
            check("mathematical_value", value, mathematical(method, mt, pm, xm))
            ds = (mixture_volume_derivatives(method, t, xs, p, order=3) if xs is not None
                  else volume_derivatives(method, t, p, order=3))
            for n in range(1, 4):
                check(f"temperature_order_{n}", ds[n],
                      mp.diff(lambda z: mathematical(method, z, pm, xm), mt, n), derivative=True)
            if "P" in p:
                ds = (mixture_volume_derivatives(method, t, xs, p, order=3, variable="pressure")
                      if xs is not None else volume_pressure_derivatives(method, t, p, order=3))
                for n in range(1, 4):
                    check(f"pressure_order_{n}", ds[n], mp.diff(
                        lambda z: mathematical(method, mt, {**pm, "P": z}, xm), pm["P"], n),
                        derivative=True)
                if xs is None:
                    check("parameter_pressure_Jacobian", volume_parameter_jacobian(
                        method, t, p, names=("P",))["P"], ds[1], derivative=True)
            if xs is not None:
                slopes = mixture_volume_composition_jacobian(method, t, xs, p)
                for i, slope in enumerate(slopes):
                    def vary(delta, fractions, coordinate=i):
                        return [v+(delta if j == coordinate else -delta if j == len(xs)-1 else 0)
                                for j, v in enumerate(fractions)]
                    check(f"composition_{i}", slope, mp.diff(
                        lambda d: mathematical(method, mt, pm, vary(d, xm)), mp.mpf(0)),
                        derivative=True)
                    for h in (1e-4, 5e-5):
                        check(f"composition_{i}_reference_step_{h}", slope, difference(
                            lambda d: upstream(method, t, p, vary(d, xs)), 0., h), fd=True)
        return dict(**case, status="passed" if all(c["passed"] for c in comparisons) else "failed",
                    comparisons=comparisons)
    except Exception as exc:
        return dict(**case, status="failed", comparisons=comparisons,
                    error=repr(exc), traceback=traceback.format_exc())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    mp.mp.dps = 60
    start = time.perf_counter()
    rows = []
    with (args.output / "units.jsonl").open("x", encoding="utf-8") as handle:
        for case in cases():
            row = run_unit(case)
            rows.append(row)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            elapsed = time.perf_counter()-start
            print(f"{row['id']} {row['status']} {len(rows)}/160 "
                  f"{len(rows)/elapsed:.1f} units/s ETA {(160-len(rows))*elapsed/len(rows):.1f}s",
                  flush=True)
    summary = dict(mode="development", reference_version=chemicals.__version__, units=160,
        passed=sum(r["status"] == "passed" for r in rows),
        failed=sum(r["status"] == "failed" for r in rows),
        comparisons=sum(len(r["comparisons"]) for r in rows),
        failed_comparisons=sum(not c["passed"] for r in rows for c in r["comparisons"]),
        elapsed_s=time.perf_counter()-start, results=rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n",
                                             encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)
    return bool(summary["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
