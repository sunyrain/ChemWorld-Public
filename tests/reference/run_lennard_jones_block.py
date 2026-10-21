"""Fixed 220-method / 397-source LJ blocks; upstream execution stays test-only."""

import argparse
import csv
import json
import math
import time
import traceback
from decimal import Decimal, localcontext
from pathlib import Path

import chemicals
from chemicals import lennard_jones as reference
from pressure_reference import finite_difference

from chemworld.data.lennard_jones import LennardJonesDatabase
from chemworld.physchem import lennard_jones as native
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation

PAIRS = tuple([(1, s) for s in range(1, 8)] + [(2, s) for s in range(2, 7)]
              + [(3, s) for s in range(3, 6)] + [(4, 4)])
METHODS = ("Neufeld_Janzen_Aziz", "Kim_Monroe")
ENERGY_INPUTS = {
    "BSL_boiling": ("Tb",), "BSL_critical": ("Tc",), "BSL_melting": ("Tm",),
    "Flynn": ("Tc",), "Stiel_Thodos": ("Tc", "Zc"),
    "Tee_Gotoh_Steward_1": ("Tc",), "Tee_Gotoh_Steward_2": ("Tc", "omega"),
}
DIAMETER_INPUTS = {
    "BSL_boiling": ("Vb",), "BSL_critical_1": ("Vc",), "BSL_critical_2": ("Tc", "Pc"),
    "BSL_melting": ("Vm",), "Flynn": ("Vc",), "Silva_Liu_Macedo": ("Tc", "Pc"),
    "Stiel_Thodos": ("Vc", "Zc"), "Tee_Gotoh_Steward_1": ("Tc", "Pc"),
    "Tee_Gotoh_Steward_2": ("Tc", "Pc", "omega"),
}


def compare(actual, expected, *, rtol=3e-12, atol=1e-12):
    assert math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), (actual, expected)
    return {"actual": actual, "reference": expected, "error": abs(actual-expected)}


def collision(method, temperature, pair):
    l_order, s_order = pair
    function = getattr(reference, "collision_integral_" + method)
    actual, derivative = native.collision_value_slope(temperature, method=method,
                                                     l=l_order, s=s_order)
    expected = function(temperature, l=l_order, s=s_order)
    slopes = finite_difference(lambda t: function(t, l=l_order, s=s_order), temperature, 1)
    compare(*slopes, rtol=5e-7, atol=3e-8)
    return {"value": compare(actual, expected), "derivatives": [
        compare(derivative, d, rtol=5e-7, atol=3e-8) for d in slopes]}


def transport(method, temperature):
    nitrogen = ComponentSpec("nitrogen", "N2", 28.014, default_phase="gas")
    oxygen = ComponentSpec("oxygen", "O2", 31.9988, default_phase="gas")
    a = native.LennardJonesSpec("nitrogen", 100, 3.7e-10, "fixed synthetic LJ parameters")
    b = native.LennardJonesSpec("oxygen", 120, 3.5e-10, "fixed synthetic LJ parameters")
    mu = native.lennard_jones_viscosity_report(nitrogen, a, temperature_K=temperature,
                                               collision_method=method).evaluation.value
    diffusivity = native.lennard_jones_diffusivity_report(
        nitrogen, a, oxygen, b, temperature_K=temperature, pressure_Pa=101325,
        collision_method=method, combining_rule="Lorentz_Berthelot",
    ).evaluation.value
    omega = getattr(reference, "collision_integral_" + method)
    mu_ref = 2.6693e-6 * math.sqrt(28.014*temperature) / 3.7**2 / omega(
        temperature/100, l=2, s=2)
    d_ref = .001858 * temperature**1.5 * math.sqrt(1/28.014+1/31.9988) / 3.6**2 / omega(
        temperature/math.sqrt(100*120), l=1, s=1) * 1e-4
    return {"viscosity": compare(mu, mu_ref, rtol=5e-4, atol=1e-15),
            "diffusivity": compare(diffusivity, d_ref, rtol=5e-4, atol=1e-15)}


def method_units():
    parameters = {"Tc": (400, 600, 800), "Tb": (250, 400, 500), "Tm": (150, 250, 350),
                  "Zc": (.25, .27, .29), "omega": (.1, .3, .5), "Pc": (3e6, 5e6, 8e6),
                  "Vc": (.0002, .0003, .0004), "Vb": (.0001, .00015, .0002),
                  "Vm": (.00009, .00012, .00016)}
    for family, declarations, function in (
        ("epsilon", ENERGY_INPUTS, native.lennard_jones_energy),
        ("sigma", DIAMETER_INPUTS, native.lennard_jones_diameter),
    ):
        for method, keys in declarations.items():
            ref = getattr(reference, family + "_" + method.replace("BSL", "Bird_Stewart_Lightfoot"))
            for index in range(3):
                params = {k: parameters[k][index] for k in keys}
                yield f"{family}/{method}/{index}", params, (
                    lambda function=function, method=method, params=params, ref=ref, family=family:
                    compare(function(method, **params), ref(**params)*(1e-10 if family == "sigma"
                                                                       else 1),
                            atol=1e-22 if family == "sigma" else 1e-12))
    for T, epsilon in zip((300, 400, 500), (100, 200, 300), strict=True):
        for kind in ("epsilon_k", "epsilon_J"):
            params = {kind: epsilon if kind == "epsilon_k" else epsilon*1.380649e-23}
            yield f"reduced/{T}/{kind}", params, (
                lambda T=T, params=params, epsilon=epsilon:
                compare(native.lennard_jones_reduced_temperature(T, **params), T/epsilon))
    for method in METHODS:
        for pair in PAIRS:
            for T in (.4, 1, 3, 20, 50):
                yield f"collision/{method}/{pair}/{T}", {"T_star": T, "pair": pair}, (
                    lambda method=method, T=T, pair=pair: collision(method, T, pair))
        for T in (200, 350, 600):
            yield f"transport/{method}/{T}", {"T": T, "P": 101325}, (
                lambda method=method, T=T: transport(method, T))


def source_unit(database, source, index, raw):
    records = database.records(source)
    record = records[index]
    values = {}
    for key, scale, unit in (("Stockmayer", 1, "K"), ("molecular_diameter", 1e-10, "m")):
        literal = raw[key]
        expected = float(literal) if literal.strip() else None
        value = record.value(key)
        assert record.raw(key) == literal
        if expected is None or not math.isfinite(expected):
            assert value.quality != "valid"
        else:
            compare(value.value, expected)
            compare(value.quantity(unit).value, expected*scale, atol=1e-22)
        values[key] = value
    assert record.source_key == raw["CAS"]
    assert record in database.records_for(record.source_key, method=source)
    valid = all(v.quality == "valid" and v.value > 0 for v in values.values())
    if not valid:
        return {"computable": False, "quality": {k: v.quality for k, v in values.items()}}
    energy, diameter = values["Stockmayer"].quantity("K").value, values[
        "molecular_diameter"].quantity("m").value
    comparisons = []
    for method, equation in (("Neufeld_Janzen_Aziz", "lj_viscosity_neufeld"),
                              ("Kim_Monroe", "lj_viscosity_kim_monroe")):
        curve = PropertyCorrelation(
            f"{source}/{index}/{method}", equation,
            {"MW": 40, "sigma": diameter, "epsilon_k": energy}, {"temperature": "K"},
            "Pa*s", "gas_viscosity", {"temperature": (.5*energy, 20*energy)},
            "Fixed synthetic MW=40 g/mol; source sigma and epsilon/k",
        )
        omega = getattr(reference, "collision_integral_" + method)
        for reduced in (1, 3, 10):
            temperature = energy*reduced
            function = lambda t, omega=omega: 2.6693e-6*math.sqrt(40*t)/(diameter*1e10)**2/omega(  # noqa: E731
                t/energy, l=2, s=2)
            comparisons.append(compare(evaluate_correlation(curve, temperature_K=temperature).value,
                                       function(temperature), rtol=5e-4, atol=1e-15))
            actual = correlation_temperature_derivative(curve, temperature_K=temperature).value
            for expected in finite_difference(function, temperature, 1):
                compare(actual, expected, rtol=5e-4, atol=1e-15)
    assert source in database.methods(record.source_key, "Stockmayer", sources=(source,))
    return {"computable": True, "value_comparisons": comparisons, "derivative_comparisons": 12}


def source_units(root):
    database = LennardJonesDatabase(root)
    for source, filename, count in (("MAGALHAES_LJ", "MagalhaesLJ.tsv", 322),
                                     ("POLING_LJ", "PolingLJ.tsv", 75)):
        with (root / "Viscosity" / filename).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) == len(database.records(source)) == count
        for index, raw in enumerate(rows):
            yield f"{source}/{index}", raw, (
                lambda source=source, index=index, raw=raw:
                source_unit(database, source, index, raw))


def published_flynn_units():
    for volume in (.0002, .0003, .0004):
        def run(volume=volume):
            with localcontext() as ctx:
                ctx.prec = 50
                d = Decimal
                expected = float(d('.561') * (d(str(volume))*d('1e6'))**(d(5)/d(12))*d('1e-10'))
            return compare(native.lennard_jones_diameter("Flynn_published", Vc=volume), expected,
                           atol=1e-22)
        yield f"Flynn_published/{volume}", {"Vc": volume}, run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", choices=("methods", "sources", "flynn-published"), required=True)
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior attempts; select a new path")
    assert chemicals.__version__ == "1.5.2"
    start = time.perf_counter()
    units = list(method_units() if args.block == "methods" else published_flynn_units()
                 if args.block == "flynn-published" else source_units(args.source_directory))
    assert len(units) == {"methods": 220, "sources": 397, "flynn-published": 3}[args.block]
    rows = []
    for identity, inputs, function in units:
        row = {"unit": identity, "inputs": inputs}
        try:
            row.update(metrics=function(), status="passed")
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
    result = {"mode": "development", "block": args.block,
              "reference_version": chemicals.__version__,
              "total": len(rows), "passed": sum(r["status"] == "passed" for r in rows),
              "failed": sum(r["status"] == "failed" for r in rows),
              "seconds": time.perf_counter()-start, "results": rows}
    if args.block == "sources":
        result["computable"] = sum(r.get("metrics", {}).get("computable", False) for r in rows)
        result["noncomputable"] = sum(r.get("metrics", {}).get("computable") is False for r in rows)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    failures = [r for r in rows if r["status"] == "failed"]
    for row in failures[:5]:
        print(row["unit"], row["failure"])
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
