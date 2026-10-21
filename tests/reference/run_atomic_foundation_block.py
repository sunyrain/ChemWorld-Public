"""Fixed 22B1 development block: 3057 atomic records and 24 formula cases.

The oracle reads the original public-domain data literals independently. It
does not import the native data generator or reference parsing algorithms.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
import traceback
import zipfile
from pathlib import Path

import chemicals
from chemicals.elements import periodic_table

from chemworld.physchem import (
    atom_fractions,
    element_data,
    formula_charge,
    isotope_data,
    mass_fractions_from_formula,
    molecular_weight,
    parse_formula,
    serialize_formula,
)
from chemworld.physchem.elements import ELEMENTS

# Explicit pre-extension baseline, not read from today's native implementation.
COMPATIBILITY = {
    "H": 1.008,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Fe": 55.845,
    "Cu": 63.546,
    "Zn": 65.38,
    "Br": 79.904,
    "I": 126.90447,
    "Pt": 195.084,
}
FORMULAS = (
    ("H2O", {"H": 2, "O": 1}, 0),
    ("C2H6O", {"C": 2, "H": 6, "O": 1}, 0),
    ("Ca(OH)2", {"Ca": 1, "O": 2, "H": 2}, 0),
    ("K4[Fe(CN)6]", {"K": 4, "Fe": 1, "C": 6, "N": 6}, 0),
    ("Al2{SO4}3", {"Al": 2, "S": 3, "O": 12}, 0),
    ("Pd(NH3)4.0001+2", {"Pd": 1, "N": 4.0001, "H": 12.0003}, 2),
    ("Fe0.5Mg0.5O", {"Fe": 0.5, "Mg": 0.5, "O": 1}, 0),
    ("CuSO4·5H2O", {"Cu": 1, "S": 1, "O": 9, "H": 10}, 0),
    ("Al2(SO4)3·18H2O", {"Al": 2, "S": 3, "O": 30, "H": 36}, 0),
    ("[13C]2H6O", {"[13C]": 2, "H": 6, "O": 1}, 0),
    ("D2O", {"[2H]": 2, "O": 1}, 0),
    ("T2", {"[3H]": 2}, 0),
    ("[235U]O2", {"[235U]": 1, "O": 2}, 0),
    ("NH4+", {"N": 1, "H": 4}, 1),
    ("Br3-", {"Br": 3}, -1),
    ("Fe+3", {"Fe": 1}, 3),
    ("Fe^3+", {"Fe": 1}, 3),
    ("SO4-2", {"S": 1, "O": 4}, -2),
    ("SO4(--)", {"S": 1, "O": 4}, -2),
    ("Fe+++", {"Fe": 1}, 3),
    ("NH4(+)", {"N": 1, "H": 4}, 1),
    ("[Fe(CN)6]-3", {"Fe": 1, "C": 6, "N": 6}, -3),
    ("C60", {"C": 60}, 0),
    ("XeF6", {"Xe": 1, "F": 6}, 0),
)


def source_estimate(text):
    """Independent token arithmetic for last-place uncertainties and intervals."""
    text = text.strip().replace("#", "")
    nominal, bracket, bounds = text.partition("[")
    interval = None
    if bracket:
        interval = tuple(float(part) for part in bounds.split("]")[0].split(","))
    nominal = nominal.strip()
    if not nominal:
        return None, None, interval
    value_text, parenthesis, uncertainty_text = nominal.partition("(")
    value = float(value_text)
    uncertainty = None
    if parenthesis:
        places = len(value_text.split(".")[1]) if "." in value_text else 0
        uncertainty_text = uncertainty_text.rstrip(")")
        uncertainty = (
            float(uncertainty_text)
            if "." in uncertainty_text
            else int(uncertainty_text) / 10**places
        )
    return value, uncertainty, interval


def references(wheel):
    if chemicals.__version__ != "1.5.2":
        raise ValueError("fixed reference requires chemicals 1.5.2")
    with zipfile.ZipFile(wheel) as archive:
        if "periodictable-2.1.0.dist-info/METADATA" not in archive.namelist():
            raise ValueError("fixed reference requires periodictable 2.1.0 wheel")
        source = archive.read("periodictable/mass.py").decode("iso-8859-15")
    tables = {}
    for assignment in ast.parse(source).body:
        if not isinstance(assignment, ast.Assign):
            continue
        for target in assignment.targets:
            if isinstance(target, ast.Name) and target.id in {
                "element_mass",
                "isotope_mass",
                "isotope_abundance",
            }:
                tables[target.id] = ast.literal_eval(assignment.value)
    weights = {}
    for line in tables["element_mass"].splitlines():
        columns = line.split("\t")
        assert columns[1] not in weights
        weights[columns[1]] = source_estimate(columns[3])
    abundances = {}
    symbol = None
    for line in tables["isotope_abundance"].splitlines():
        if not line.strip():
            continue
        columns = line.strip().split("\t")
        if line[0].isspace():
            assert symbol is not None
            label = f"[{columns[0]}{symbol}]"
            assert label not in abundances
            abundances[label] = source_estimate(columns[1])
        else:
            symbol = columns[1]
    isotopes = {}
    for line in tables["isotope_mass"].splitlines():
        identity, mass_text, *_ = line.split(",")
        number, symbol, mass_number = identity.split("-")
        label = f"[{mass_number}{symbol}]"
        assert label not in isotopes
        isotopes[label] = {
            "number": int(number),
            "symbol": symbol,
            "mass_number": int(mass_number),
            "mass": source_estimate(mass_text),
            "estimated": "#" in mass_text,
            "abundance": abundances.get(label, (None, None, None)),
        }
    assert len(list(periodic_table)) == 118
    assert len(weights) == 84 and len(abundances) == 289 and len(isotopes) == 2939
    assert set(abundances).issubset(isotopes)
    return weights, isotopes, abundances


def check_number(actual, expected, errors, *, atol=1e-15, rtol=2e-14):
    if expected is None:
        assert actual is None, f"unknown reference was replaced by {actual!r}"
        return
    assert actual is not None, f"missing native value, reference={expected!r}"
    error = abs(actual - expected)
    errors.append(error)
    assert math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), (
        f"native={actual!r}, reference={expected!r}, absolute_error={error!r}"
    )


def check_estimate(actual, expected, errors):
    check_number(actual.value, expected[0], errors)
    check_number(actual.standard_uncertainty, expected[1], errors)
    if expected[2] is None:
        assert actual.interval is None
    else:
        assert actual.interval is not None
        for native, reference in zip(actual.interval, expected[2], strict=True):
            check_number(native, reference, errors)


def check_element(reference, weights):
    errors = []
    symbol = reference.symbol
    actual = element_data()[symbol]
    assert (actual.atomic_number, actual.symbol, actual.name) == (
        reference.number,
        symbol,
        reference.name,
    )
    expected = weights.get(symbol, (None, None, None))
    check_estimate(actual.standard_weight, expected, errors)
    native = ELEMENTS[symbol]
    assert (native.atomic_number, native.name) == (reference.number, reference.name)
    if symbol in COMPATIBILITY:
        assert native.atomic_weight_g_mol == COMPATIBILITY[symbol]
        assert native.weight_source == "chemworld-original-26"
    else:
        check_number(native.atomic_weight_g_mol, expected[0], errors)
        assert native.weight_source == "CIAAW2021"
    assert set(element_data()) == {item.symbol for item in periodic_table}
    assert len({item.atomic_number for item in element_data().values()}) == 118
    return {"max_field_absolute_error": max(errors, default=0)}


def check_isotope(label, reference, reference_labels):
    errors = []
    actual = isotope_data()[label]
    assert (actual.atomic_number, actual.symbol, actual.mass_number) == (
        reference["number"],
        reference["symbol"],
        reference["mass_number"],
    )
    assert actual.estimated_mass == reference["estimated"]
    check_estimate(actual.mass, reference["mass"], errors)
    check_estimate(actual.abundance, reference["abundance"], errors)
    assert set(isotope_data()) == reference_labels
    return {"max_field_absolute_error": max(errors, default=0)}


def check_formula(case, weights, isotopes):
    text, expected, charge = case
    actual = parse_formula(text)
    canonical = serialize_formula(text)
    roundtrip = parse_formula(canonical)
    errors = []
    assert set(actual) == set(roundtrip) == set(expected)
    for atom, count in expected.items():
        check_number(actual[atom], count, errors, atol=1e-12, rtol=0)
        check_number(roundtrip[atom], count, errors, atol=1e-12, rtol=0)
    assert formula_charge(text) == formula_charge(canonical) == charge
    atom_weights = {
        atom: (
            isotopes[atom]["mass"][0]
            if atom.startswith("[")
            else COMPATIBILITY[atom]
            if atom in COMPATIBILITY
            else weights[atom][0]
        )
        for atom in expected
    }
    expected_mass = sum(atom_weights[atom] * count for atom, count in expected.items())
    native_mass = molecular_weight(actual)
    mass_errors = []
    check_number(native_mass, expected_mass, mass_errors, atol=1e-12)
    number_fractions = atom_fractions(actual)
    weight_fractions = mass_fractions_from_formula(actual)
    for atom, count in expected.items():
        check_number(number_fractions[atom], count / sum(expected.values()), [], atol=1e-14)
        check_number(
            weight_fractions[atom], atom_weights[atom] * count / expected_mass, [], atol=1e-14
        )
    fraction_errors = []
    for fractions in (number_fractions, weight_fractions):
        check_number(sum(fractions.values()), 1, fraction_errors, atol=1e-14, rtol=0)
    return {
        "canonical": canonical,
        "expected_charge": charge,
        "expected_composition": expected,
        "native_composition": actual,
        "max_count_absolute_error": max(errors),
        "native_mass_g_mol": native_mass,
        "reference_mass_g_mol": expected_mass,
        "mass_absolute_error": max(mass_errors),
        "max_fraction_sum_error": max(fraction_errors),
    }


def run(wheel, output):
    if output.exists():
        raise FileExistsError("preserve previous block results: select a new output")
    weights, isotopes, abundances = references(wheel)
    isotope_labels = set(isotopes)
    units = []
    for element in periodic_table:
        units.append(("element", element.symbol, lambda e=element: check_element(e, weights)))
    for label, ref in isotopes.items():
        units.append(("isotope", label, lambda k=label, r=ref: check_isotope(k, r, isotope_labels)))
    for case in FORMULAS:
        units.append(("formula", case[0], lambda c=case: check_formula(c, weights, isotopes)))
    assert len(units) == 3081
    started = time.monotonic()
    results = []
    for category, identity, evaluate in units:
        unit = {"category": category, "identity": identity}
        try:
            unit.update(status="passed", metrics=evaluate())
        except Exception:
            unit.update(status="failed", failure=traceback.format_exc())
        results.append(unit)
        count = len(results)
        if count % 250 == 0 or count == len(units):
            elapsed = time.monotonic() - started
            rate = count / max(elapsed, 1e-9)
            print(
                f"atomic block {count}/{len(units)} rate={rate:.1f}/s "
                f"ETA={(len(units) - count) / rate:.1f}s "
                f"failed={sum(r['status'] == 'failed' for r in results)}",
                flush=True,
            )
    failures = [unit for unit in results if unit["status"] == "failed"]
    summary = {
        "mode": "development",
        "block": "NATIVE-CONV-22B1",
        "references": {"periodictable": "2.1.0", "chemicals": chemicals.__version__},
        "denominators": {"elements": 118, "isotopes": 2939, "formulas": 24, "total": 3081},
        "field_coverage": {
            "standard_weights": len(weights),
            "abundances": len(abundances),
            "compatibility_weights": len(COMPATIBILITY),
        },
        "passed": len(results) - len(failures),
        "failed": len(failures),
        "skipped": 0,
        "elapsed_seconds": time.monotonic() - started,
        "units": results,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in {"units", "failures"}},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return bool(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.reference_wheel, args.output))
