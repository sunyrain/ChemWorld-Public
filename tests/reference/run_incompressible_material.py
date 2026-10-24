# ruff: noqa: B023 -- every callback is consumed synchronously in its creating iteration
"""Fixed 54-state / 1188-check original material-property acceptance block."""

import argparse
import json
from collections import Counter
from pathlib import Path

from tests.reference.incompressible_fixtures import amounts, material
from tests.reference.run_incompressible import compare

QUANTITIES = {
    "density": ("rhomass", 1e-7, False),
    "source_cp": ("cpmass", 1e-7, True),
    "enthalpy": ("hmass", 1e-5, True),
    "internal_energy": ("umass", 1e-5, True),
    "entropy": ("smass", 1e-7, True),
    "molar_volume": (None, 1e-12, False),
    "viscosity": ("viscosity", 1e-12, False),
    "thermal_conductivity": ("conductivity", 1e-10, False),
}
PARTIALS = (
    ("enthalpy", "partial_enthalpy", 1e-4, 1e-7),
    ("internal_energy", "partial_internal_energy", 1e-4, 1e-7),
    ("entropy", "partial_entropy", 1e-6, 1e-9),
    ("molar_volume", "partial_volume", 1e-11, 1e-13),
)


def run():
    import CoolProp
    import CoolProp.CoolProp as CP  # noqa: N817

    if CoolProp.__version__ != "8.0.0":
        raise ValueError("fixed material acceptance requires CoolProp 8.0.0")
    rows, cases = [], []
    for name in ("MEG", "MPG", "MCA"):
        model, _, phase = material(name)
        for T in (290.0, 300.0, 310.0):
            for x in (0.05, 0.15, 0.25):
                for p in (101325.0, 2e6):
                    n = amounts(model, x)
                    total = sum(n.values())
                    mass_per_mole = 1 / total
                    inputs = {"fluid": name, "T_K": T, "p_Pa": p, "mass_fraction": x, "n_mol": n}

                    def evaluate(ns=n, temperature=T, requested=None):
                        return phase.evaluate(
                            ns,
                            temperature_K=temperature,
                            pressure_Pa=p,
                            quantities=requested or (*QUANTITIES, "cp", *(q[1] for q in PARTIALS)),
                        )

                    def reference_value(method, temperature=T):
                        ref = CP.AbstractState("INCOMP", name)
                        ref.set_mass_fractions([x])
                        ref.update(CP.PT_INPUTS, p, temperature)
                        return getattr(ref, method)()

                    for key, (method, atol, molar) in QUANTITIES.items():

                        def reference(method=method, molar=molar):
                            return (
                                mass_per_mole / reference_value("rhomass")
                                if method is None
                                else reference_value(method) * (mass_per_mole if molar else 1)
                            )

                        rows.append(
                            compare(
                                inputs, key, lambda key=key: evaluate().value(key), reference, atol
                            )
                        )
                    for source in ("native", "reference"):

                        def difference(source=source):
                            values = [
                                (
                                    evaluate(temperature=T + d).value("enthalpy")
                                    if source == "native"
                                    else reference_value("hmass", T + d) * mass_per_mole
                                )
                                for d in (0.001, -0.001)
                            ]
                            return (values[0] - values[1]) / 0.002

                        rows.append(
                            compare(
                                inputs,
                                "cp_vs_" + source + "_h_T",
                                lambda: evaluate().value("cp"),
                                difference,
                                1e-5,
                                2e-5,
                            )
                        )
                    for quantity, partial, atol, euler_atol in PARTIALS:
                        for key in n:

                            def difference(key=key, quantity=quantity):
                                step = 1e-6 * max(1.0, n[key])
                                values = []
                                for sign in (1, -1):
                                    ns = {**n, key: n[key] + sign * step}
                                    values.append(sum(ns.values()) * evaluate(ns).value(quantity))
                                return (values[0] - values[1]) / (2 * step)

                            rows.append(
                                compare(
                                    inputs,
                                    partial + ":" + key,
                                    lambda partial=partial, key=key: evaluate().value(
                                        partial, component=key
                                    ),
                                    difference,
                                    atol,
                                    2e-5,
                                )
                            )
                        rows.append(
                            compare(
                                inputs,
                                "Euler:" + quantity,
                                lambda partial=partial: sum(
                                    n[k] * evaluate().value(partial, component=k) for k in n
                                ),
                                lambda quantity=quantity: total * evaluate().value(quantity),
                                euler_atol,
                                2e-10,
                            )
                        )
                    try:
                        result = evaluate().to_dict()
                    except Exception as exc:
                        result = f"{type(exc).__name__}: {exc}"
                    cases.append({"inputs": inputs, "report": result})
                    print(
                        f"incompressible material {len(cases)}/54; {len(rows)}/1188 checks",
                        flush=True,
                    )
    assert len(cases) == 54 and len(rows) == 1188
    return {
        "evidence_mode": "development",
        "reference": "CoolProp 8.0.0",
        "cases": cases,
        "comparisons": rows,
        "counts": dict(Counter(r["status"] for r in rows)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("never overwrite an earlier material result")
    result = run()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(result["counts"])


if __name__ == "__main__":
    main()
