"""Fixed 23-unit ChemSep16 and formation/stoichiometry development reference."""

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import chemicals.reaction as reference
from run_caloric_data_block import integrals, near

from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.stoichiometry import (
    balance_stoichiometry,
    stoichiometric_matrix,
    stoichiometry_mass_to_molar,
    stoichiometry_molar_to_mass,
)
from chemworld.physchem.thermochemistry import formation_entropy, formation_gibbs_energy

CASES = (
    (("NH3", "O2", "NO", "H2O"), (True, True, False, False), None, (4, 5, 4, 6)),
    (("C2H6", "O2", "CO2", "H2O"), (True, True, False, False), None, (2, 7, 4, 6)),
    (
        ("Fe", "MnO4", "H", "Fe", "Mn", "H2O"),
        (True, True, True, False, False, False),
        (2, -1, 1, 3, 2, 0),
        (5, 1, 8, 5, 1, 4),
    ),
    (("H2", "D2", "HD"), (True, True, False), None, (1, 1, 2)),
    (("C0.5H1.5", "O2", "CO2", "H2O"), (True, True, False, False), None, (8, 7, 4, 6)),
    (("Fe", "Fe", ""), (True, False, False), (2, 3, -1), (1, 1, 1)),
)


def curve_case(variant):
    values, unit = (
        ((20000, 100, 10, 0.001, -1e-6), "J/(kmol*K)"),
        ((15, -200, 3, 0.0001, 1e-7), "J/(mol*K)"),
    )[variant]
    curve = PropertyCorrelation(
        f"fixed-{variant}",
        "chemsep16",
        dict(zip("ABCDE", values, strict=True)),
        {"temperature": "K"},
        unit,
        "liquid_heat_capacity",
        {"temperature": (250, 550)},
    )
    a, b, c, d, e = values
    scale = 0.001 if variant == 0 else 1

    def function(T):
        return scale * (a + math.exp(b / T + c + d * T + e * T * T))

    return curve, function, scale


def point(variant, T):
    curve, function, scale = curve_case(variant)
    error = near(evaluate_correlation(curve, temperature_K=T).to("J/(mol*K)").value, function(T))
    h = T * 1e-4
    slope = (
        function(T - 2 * h) - 8 * function(T - h) + 8 * function(T + h) - function(T + 2 * h)
    ) / (12 * h)
    actual = correlation_temperature_derivative(curve, temperature_K=T).value * scale
    return {"value_error": error, "slope_error": near(actual, slope, 3e-7, 2e-9)}


def integral(variant, reverse):
    from collections import Counter

    curve, function, _ = curve_case(variant)
    metrics = Counter()
    low, high = (550, 250) if reverse else (250, 550)
    integrals(curve, function, low, high, metrics)
    return dict(metrics)


def balance(index):
    formulas, sides, charges, expected = CASES[index]
    atoms = [parse_formula(s) if s else {} for s in formulas]
    matrix = stoichiometric_matrix(atoms, sides, charges=charges)
    actual = balance_stoichiometry(matrix)
    assert actual == expected
    assert all(sum(c * v for c, v in zip(row, actual, strict=True)) == 0 for row in matrix)
    if index < 2:
        assert (
            tuple(
                reference.balance_stoichiometry(reference.stoichiometric_matrix(atoms, list(sides)))
            )
            == actual
        )
    return {"coefficients": list(map(str, actual))}


def masses(index):
    formulas, _, _, coefficients = CASES[index]
    weights = [molecular_weight(parse_formula(s)) for s in formulas]
    mass = stoichiometry_molar_to_mass(coefficients, weights)
    expected = reference.stoichiometry_molar_to_mass(list(coefficients), weights)
    for a, b in zip(mass, expected, strict=True):
        near(a, b, 2e-12, 1e-10)
    inverse = stoichiometry_mass_to_molar(mass, weights)
    for a, b in zip(inverse, reference.stoichiometry_mass_to_molar(expected, weights), strict=True):
        near(a, b, 2e-12, 1e-10)
    return {"mass_coefficients": mass}


def formation():
    def datum(H, S):
        return CaloricReference(
            phase="gas", temperature_K=300, pressure_Pa=100000, energy_J_mol=H, entropy_J_mol_K=S
        )

    value = formation_gibbs_energy(datum(-1000, 100), (datum(0, 20), datum(0, 30)), (1, 2))
    assert value == reference.Gibbs_formation(-1000, 100, [0, 0], [20, 30], [1, 2], T_ref=300)
    assert value == -7000
    assert (
        formation_entropy(
            formation_enthalpy_J_mol=-1000, formation_gibbs_J_mol=value, temperature_K=300
        )
        == 20
    )
    return {"Gf_J_mol": value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started, results = time.perf_counter(), []
    cases = [
        (f"point-{v}-{T}", lambda v=v, T=T: point(v, T), 1) for v in (0, 1) for T in (300, 400, 500)
    ]
    # Each paired H/S execution contains two fixed units.
    cases += [
        (f"integral-{v}-{rev}", lambda v=v, rev=rev: integral(v, rev), 2)
        for v in (0, 1)
        for rev in (False, True)
    ]
    cases += [(f"balance-{i}", lambda i=i: balance(i), 1) for i in range(6)]
    cases += [(f"mass-{i}", lambda i=i: masses(i), 1) for i in range(2)]
    cases += [("formation", formation, 1)]
    for key, function, count in cases:
        row = {"id": key, "units": count}
        try:
            row.update(passed=True, metrics=function())
        except Exception:
            row.update(passed=False, error=traceback.format_exc())
        results.append(row)
    summary = {
        "units": sum(r["units"] for r in results),
        "passed": sum(r["units"] for r in results if r["passed"]),
        "failed": sum(r["units"] for r in results if not r["passed"]),
        "elapsed_s": time.perf_counter() - started,
        "results": results,
    }
    assert summary["units"] == 23
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    raise SystemExit(bool(summary["failed"]))


if __name__ == "__main__":
    main()
