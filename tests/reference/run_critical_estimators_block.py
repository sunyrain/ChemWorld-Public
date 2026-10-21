"""Fixed 119-unit development block; see NATIVE_CONV22D1_EXPERIMENT.md."""

import argparse
import json
import math
import time
import traceback
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path

import chemicals
from chemicals import acentric, critical

from chemworld.physchem import critical_kernels as native

FITS = ("Perry8E", "Chemicals2021FitInorganic", "Chemicals2021FitOrganic",
        "Chemicals2021FitElements", "Chemicals2021FitBinary", "Chemicals2021FitTernary",
        "Chemicals2021FitCl", "Chemicals2021FitN", "Chemicals2021FitC", "Chemicals2021FitF",
        "Chemicals2021FitO", "Chemicals2021FitI", "Chemicals2021FitBr", "Chemicals2021FitSi")
MIXTURES = ("Li", "Chueh_Prausnitz_Tc", "Chueh_Prausnitz_Vc", "Grieves_Thodos",
            "modified_Wilson_Tc", "modified_Wilson_Vc")


def cases():
    for method in ("Ihmels", "Meissner", "Grigoras", "Meissner_MPa"):
        for case, (Tc, Vc) in enumerate(((600., .0003), (851., .000308))):
            Pc = (2.08 * Tc / (Vc - 8e-6) if method == "Meissner_MPa"
                  else getattr(critical, method)(Tc=Tc, Vc=Vc))
            for index, field in enumerate(("Tc", "Pc", "Vc")):
                params = {"Tc": Tc, "Pc": Pc, "Vc": Vc}
                expected = params.pop(field)
                yield f"surface/{method}/{case}/{field}", params, (
                    lambda method=method, params=params, index=index:
                    native.critical_surface(method=method, **params)[index]), (
                    lambda expected=expected: expected)
    for fit in FITS:
        for Tb in (300., 427.15):
            Tc = critical.Tb_Tc_relationship(Tb=Tb, fit=fit)
            for field, value, index in (("Tb", Tb, 1), ("Tc", Tc, 0)):
                params = {field: value, "fit": fit}
                yield f"boiling/{fit}/{Tb}/{field}", params, (
                    lambda params=params, index=index: native.boiling_critical_temperature(
                        **params)[index]), (
                    lambda params=params: critical.Tb_Tc_relationship(**params))
    molecules = ({"C": 10, "H": 22}, {"H": 2, "O": 1}, {"C": 2, "H": 6, "O": 1},
                 {"C": 1, "Cl": 4}, {"Si": 1, "H": 4}, {"C": 2, "H": 3, "Br": 1})
    for index, atoms in enumerate(molecules):
        for regressed in (False, True):
            native_params = ({"coefficient": 4.261206523632586, "power": .5597281770786228,
                              "radii": native.MERSMANN_KIND_REGRESSED_RADII_ANGSTROM}
                             if regressed else {})
            ref_params = ({"coeff": 4.261206523632586, "power": .5597281770786228,
                           "covalent_radii": critical.rcovs_regressed} if regressed else {})
            yield f"molecular/{index}/{regressed}", {"atoms": atoms, "regressed": regressed}, (
                lambda atoms=atoms, p=native_params: native.mersmann_kind_volume(atoms, **p)), (
                lambda atoms=atoms, p=ref_params: critical.Mersmann_Kind_predictor(atoms, **p))
    for method in MIXTURES:
        for index, xs in enumerate(((.3, .7), (.2, .3, .5), (0., .4, .6))):
            n = len(xs)
            Tcs, Vcs = (425.12, 469.7, 507.6)[:n], (.000255, .000313, .000371)[:n]
            cp = method.startswith("Chueh")
            matrix = [[0 if i == j else 2 + abs(i-j) if cp else 1+.05*(i+1)-.03*(j+1)
                       for j in range(n)] for i in range(n)]
            params = {"Tcs": Tcs, "Vcs": Vcs}
            if method != "Li":
                params["interactions"] = matrix
            if method == "Li":
                ref_args = (xs, Tcs, Vcs)
            elif method == "Chueh_Prausnitz_Tc":
                ref_args = (xs, Tcs, Vcs, matrix)
            else:
                ref_args = (xs, Vcs if method.endswith("_Vc") else Tcs, matrix)
            yield f"mixture/{method}/{index}", {"fractions": xs, **params}, (
                lambda method=method, xs=xs, params=params:
                native.mixture_critical_property(method, xs, **params)), (
                lambda method=method, args=ref_args: getattr(critical, method)(*args))
    for index, (Pc, Tc, ratio, boiling, omega) in enumerate(zip(
        (3e6, 5e6, 8e6), (500., 650., 800.), (.01, .008, .005),
        (.55, .65, .75), (.1, .3, .6), strict=True,
    )):
        for method, params, function, reference in (
            ("omega_definition", {"Psat": Pc*ratio, "Pc": Pc},
             lambda Psat, Pc: native.acentric_factor_from_pressure(
                 Psat, Pc, reduced_temperature=.7), acentric.omega_definition),
            ("LK_omega", {"Tb": Tc*boiling, "Tc": Tc, "Pc": Pc},
             native.lee_kesler_acentric_factor, acentric.LK_omega),
            ("Stiel_polar_factor", {"Psat": Pc*ratio, "Pc": Pc, "omega": omega},
             lambda Psat, Pc, omega: native.stiel_polar_factor(
                 Psat, Pc, omega, reduced_temperature=.6), acentric.Stiel_polar_factor),
        ):
            yield f"acentric/{method}/{index}", params, (
                lambda function=function, params=params: function(**params)), (
                lambda function=reference, params=params: function(**params))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-mode", choices=("upstream", "documented-zero-limit"),
                        default="upstream")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain prior attempts; select a new output")
    assert chemicals.__version__ == "1.5.2"
    units = list(cases())
    assert len(units) == 119 and len({c[0] for c in units}) == 119
    start, rows = time.perf_counter(), []
    for identity, inputs, function, reference in units:
        row = {"unit": identity, "inputs": inputs}
        try:
            actual = function()
            row["actual"] = actual
            if (identity == "mixture/Grieves_Thodos/2"
                    and args.reference_mode == "documented-zero-limit"):
                with localcontext() as ctx:
                    ctx.prec = 50
                    d = Decimal
                    expected = float(d('469.7')/(1+d('1.5')*d('1.01'))
                                     + d('507.6')/(1+d(2)/d(3)*d('1.09')))
                row["reference_kind"] = "published equation zero-fraction limit; Decimal(50)"
            else:
                expected = reference()
                row["reference_kind"] = (
                    "Martinez-Vitela and Gracia-Fadrique 2019 eq. 3; MPa convention"
                    if identity.startswith("surface/Meissner_MPa/") else "chemicals-1.5.2"
                )
            row.update(actual=actual, expected=expected, absolute_error=abs(actual-expected))
            assert math.isclose(actual, expected, rel_tol=3e-12, abs_tol=1e-12), row
            row["status"] = "passed"
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
    result = {"mode": "development", "reference_version": chemicals.__version__,
              "reference_mode": args.reference_mode,
              "total": len(rows), "passed": sum(r["status"] == "passed" for r in rows),
              "failed": sum(r["status"] == "failed" for r in rows),
              "families": dict(Counter(r["unit"].split('/')[0] for r in rows)),
              "seconds": time.perf_counter()-start, "results": rows}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    for row in rows:
        if row["status"] == "failed":
            print(row["unit"], row["failure"])
    return bool(result["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
