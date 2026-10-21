"""Fixed independent F4 development blocks; no upstream implementation inspection."""
# ruff: noqa: C408 -- scientific input dictionaries

import argparse
import csv
import itertools
import json
import time
import traceback
from fractions import Fraction
from pathlib import Path

import chemicals
from chemicals import combustion as reference
from run_solubility_block import compare

from chemworld.data.assessments import AssessmentValue
from chemworld.data.caloric import CaloricDatabase
from chemworld.data.fuels import (
    FUEL_METHODS,
    FuelDatabase,
    fuel_quality_comparison,
    source_ignition_quality,
    source_octane_index,
)
from chemworld.foundation.units import Quantity
from chemworld.physchem import combustion as native
from chemworld.physchem.combustion_streams import air_fuel_ratio_solver, mixture_combustion_spec
from chemworld.physchem.elements import molecular_weight, parse_formula
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec, MixtureSpec

FORMULAS = ("N2", "O2", "H2", "CH4", "C2H6", "C3H8", "H2O", "CO2")
IDENTITIES = (
    "7727-37-9",
    "7782-44-7",
    "1333-74-0",
    "74-82-8",
    "74-84-0",
    "74-98-6",
    "7732-18-5",
    "124-38-9",
)
COMPOSITIONS = (
    (
        (0.79, 0.205, 0, 0, 0, 0, 0.0045, 0.0005),
        (0.025, 0.025, 0, 0.85, 0.07, 0.029, 0.0005, 0.0005),
        (0.1, 0.005, 0, 0.5, 0.39, 0, 0.005, 0),
    ),
    (
        (0.78, 0.21, 0, 0, 0, 0, 0.005, 0.005),
        (0.05, 0.02, 0.2, 0.55, 0.15, 0.01, 0.01, 0.01),
        (0.1, 0, 0.1, 0.6, 0.1, 0.05, 0.025, 0.025),
    ),
)


def reference_arguments(compositions, n_third):
    weights = [molecular_weight(parse_formula(f)) for f in FORMULAS]
    a, f, t = compositions
    arguments = dict(
        zs_air=list(a),
        zs_fuel=list(f),
        CASs=list(IDENTITIES),
        atomss=[parse_formula(formula) for formula in FORMULAS],
        MW_air=sum(z * w for z, w in zip(a, weights, strict=True)),
        MW_fuel=sum(z * w for z, w in zip(f, weights, strict=True)),
        Vm_air=0.02493,
        Vm_fuel=0.02488,
    )
    if n_third:
        arguments.update(
            zs_third=list(t),
            n_third=n_third,
            Vm_third=0.024,
            MW_third=sum(z * w for z, w in zip(t, weights, strict=True)),
        )
    return arguments


def stream_unit(compositions, n_third, selections, equivalence=False):
    call = reference.combustion_spec_solver if n_third else reference.fuel_air_spec_solver
    arguments = reference_arguments(compositions, n_third)
    # This reference-defined equivalence transformation is fixed before execution.
    baseline = call(**arguments, n_fuel=5, O2_excess=0.3)
    specs = {}
    for key in selections:
        if key.endswith("_ratio"):
            specs.update(ratio=baseline[key], ratio_basis=key.removesuffix("_ratio"))
        else:
            specs[key] = baseline[key]
    expected = baseline if equivalence else call(**arguments, **specs)
    components = [
        ComponentSpec(f, f, default_phase="gas", cas_number=cas)
        for f, cas in zip(FORMULAS, IDENTITIES, strict=True)
    ]
    mixes = [
        MixtureSpec.from_mole_fractions(
            components, zs, phase_label="gas", temperature_K=300, pressure_Pa=101325
        )
        for zs in compositions
    ]
    original = [m.to_dict() for m in mixes]
    mixes = [MixtureSpec.from_dict(json.loads(json.dumps(m.to_dict()))) for m in mixes]
    options = dict(specs, Vm_air=0.02493, Vm_fuel=0.02488)
    if n_third:
        options.update(n_third=n_third, Vm_third=0.024)
    actual = mixture_combustion_spec(
        mixes[0], mixes[1], components, third=mixes[2] if n_third else None, **options
    )
    assert original == [m.to_dict() for m in mixes]
    checks = {
        k: compare(actual[k], expected[k])
        for k in (
            "n_air",
            "n_fuel",
            "n_out",
            "O2_excess",
            "frac_out_O2",
            "frac_out_O2_dry",
            "mole_ratio",
            "mass_ratio",
            "volume_ratio",
        )
    }
    for key in ("ns_out", "zs_out"):
        checks[key] = [compare(a, b) for a, b in zip(actual[key], expected[key], strict=True)]
    feed = [
        actual["n_air"] * a + actual["n_fuel"] * f + n_third * t
        for a, f, t in zip(*compositions, strict=True)
    ]
    atoms = [c.composition for c in components]
    for element in ("C", "H", "O", "N"):
        difference = sum(
            (n - o) * a.get(element, 0)
            for n, o, a in zip(feed, actual["ns_out"], atoms, strict=True)
        )
        checks["element:" + element] = compare(difference, 0, atol=1e-9, rtol=0)
    difference = sum(
        (n - o) * c.molecular_weight_g_mol
        for n, o, c in zip(feed, actual["ns_out"], components, strict=True)
    )
    checks["mass"] = compare(difference, 0, atol=1e-9, rtol=0)
    assert json.loads(json.dumps(actual)) == actual
    return dict(
        specifications=specs, baseline=baseline, actual=actual, reference=expected, checks=checks
    )


def stream_units(equivalence=False):
    extensive = ("n_air", "n_fuel", "n_out")
    pairs = list(itertools.combinations(extensive, 2))
    pairs += list(
        itertools.product(
            extensive,
            (
                "O2_excess",
                "frac_out_O2",
                "frac_out_O2_dry",
                "mole_ratio",
                "mass_ratio",
                "volume_ratio",
            ),
        )
    )
    for i, compositions in enumerate(COMPOSITIONS):
        for third in (0, 1):
            for selected in pairs:
                yield (
                    f"streams/{i}/{third}/{selected}",
                    dict(compositions=compositions, n_third=third, selected=selected),
                    lambda cs=compositions, q=third, sel=selected: stream_unit(
                        cs, q, sel, equivalence
                    ),
                )


def mapping_checks(actual, expected):
    return {
        k: compare(actual.get(k, 0), expected.get(k, 0)) for k in actual.keys() | expected.keys()
    }


def stoichiometry_unit(atoms, handling):
    actual = native.combustion_stoichiometry(atoms, missing_handling=handling)
    real = {k: v for k, v in atoms.items() if k != "Ash"}
    mw = molecular_weight(real) + atoms.get("Ash", 0)
    expected = reference.combustion_stoichiometry(atoms, MW=mw, missing_handling=handling)
    checks = mapping_checks(actual, expected)
    return dict(actual=actual, reference=expected, MW=mw, checks=checks)


def heat_unit(formula, hf, selection):
    atoms = parse_formula(formula)
    mw = molecular_weight(atoms)
    expected = reference.combustion_data(atoms, Hf=hf, MW=mw, method="Stoichiometry")
    kwargs = {selection: getattr(expected, selection)}
    result = native.combustion_data(
        formula, MW=mw, method="Stoichiometry" if selection == "Hf" else "Specification", **kwargs
    )
    restored = native.CombustionData.from_dict(json.loads(json.dumps(result.to_dict())))
    assert restored.to_dict() == result.to_dict()
    checks = {
        k: compare(getattr(restored, k), getattr(expected, k), atol=1e-7)
        for k in ("Hf", "HHV", "LHV", "MW")
    }
    for target in ("HHV", "LHV"):
        checks[target + "_heat"] = compare(
            restored.heat(Quantity(2, "mol"), basis=target).value,
            2 * getattr(expected, target),
            atol=1e-7,
        )
    return dict(result=result.to_dict(), checks=checks)


def quality_unit(delay, quantity):
    function = (
        reference.IDT_to_DCN
        if quantity == "DCN"
        else getattr(reference, "Perez_Boehman_" + quantity + "_from_ignition_delay")
    )
    return compare(native.ignition_delay_quality(delay, quantity=quantity), function(delay))


def ratio_unit(basis, known):
    kwargs = dict(
        Vm_air=0.024936627188566596,
        Vm_fuel=0.024880983160354486,
        MW_air=28.850334,
        MW_fuel=17.86651,
        basis=basis,
    )
    kwargs["ratio"] = 5 * (
        1
        if basis == "mole"
        else kwargs["MW_air"] / kwargs["MW_fuel"]
        if basis == "mass"
        else kwargs["Vm_air"] / kwargs["Vm_fuel"]
    )
    kwargs.update({k: v for k, v in (("n_air", 25), ("n_fuel", 5)) if k in known})
    actual = air_fuel_ratio_solver(**kwargs)
    expected = reference.air_fuel_ratio_solver(**kwargs)
    return dict(
        actual=actual,
        reference=expected,
        checks=[compare(a, b) for a, b in zip(actual, expected, strict=True)],
    )


def product_unit(scale, reactivities):
    atoms = [parse_formula(f) for f in ("CH4", "C2H6", "Ar", "C15H32")]
    amounts = [scale * v for v in (0.9, 0.05, 0.04, 0.01)]
    identities = ["74-82-8", "74-84-0", "7440-37-1", "629-62-9"]
    actual = native.combustion_products_mixture(
        atoms, amounts, identities=identities, reactivities=reactivities
    )
    expected = reference.combustion_products_mixture(
        atoms, amounts, CASs=identities, reactivities=reactivities
    )
    return dict(actual=actual, reference=expected, checks=mapping_checks(actual, expected))


def mathematics_units():
    formulas = (
        "CH4",
        "C2H6",
        "C2H6O",
        "H2",
        "CO",
        "CO2",
        "H2O",
        "N2",
        "H2S",
        "CH3Cl",
        "CH3F",
        "CBr4",
        "CI4",
        "H3PO4",
        "NaOH",
        "C2H5Na",
    )
    compositions = [parse_formula(f) for f in formulas]
    compositions.append(dict(C=0.05961, H=0.053571, S=0.000499, N=0.001142, O=0.005813, Ash=0.105))
    for i, atoms in enumerate(compositions):
        for handling in ("elemental", "ash"):
            yield (
                f"stoichiometry/{i}/{handling}",
                dict(atoms=atoms, handling=handling),
                lambda a=atoms, h=handling: stoichiometry_unit(a, h),
            )
    for formula, hf in zip(
        ("CH4", "C2H6", "C2H6O", "CH4O", "C3H8", "CH3Cl", "CH3F", "H2S"),
        (-74520, -84000, -277000, -239100, -104700, -82000, -234000, -20000),
        strict=True,
    ):
        for selection in ("Hf", "HHV", "LHV"):
            yield (
                f"heat/{formula}/{selection}",
                dict(formula=formula, Hf=hf, selection=selection),
                lambda f=formula, h=hf, s=selection: heat_unit(f, h, s),
            )
    for name, masses, methods in (
        ("coal", dict(C=0.716, H=0.054, S=0.016, N=0.016, O=0.093, Ash=0.105), ("Boie", "Dulong")),
        ("tire", dict(C=0.8033, H=0.0766, S=0.0087, N=0.0035, O=0.1079), ("Boie",)),
        ("synthetic", dict(C=0.75, H=0.25), ("Boie", "Dulong")),
    ):
        for method in methods:
            function = reference.HHV_Boie if method == "Boie" else reference.HHV_modified_Dulong
            yield (
                f"mass/{name}/{method}",
                dict(mass_fractions=masses, method=method),
                lambda ms=masses, m=method, fn=function: compare(
                    native.heating_value_mass(ms, method=m), fn(ms), atol=1e-7
                ),
            )
    for delay in (0.002, 0.0031, 0.0036, 0.004, 0.0065, 0.0067, 0.02, 0.067):
        for quantity in ("RON", "MON", "DCN"):
            yield (
                f"quality/{delay}/{quantity}",
                dict(delay=delay, quantity=quantity),
                lambda d=delay, q=quantity: quality_unit(d, q),
            )
    for ron, mon in ((0, 0), (108.6, 89.7), (120, -5), (80, 85)):
        for quantity in ("AKI", "sensitivity"):
            function = reference.AKI if quantity == "AKI" else reference.octane_sensitivity
            yield (
                f"octane/{ron}/{mon}/{quantity}",
                dict(RON=ron, MON=mon, quantity=quantity),
                lambda r=ron, m=mon, q=quantity, fn=function: compare(
                    native.octane_index(r, m, quantity=q), fn(r, m)
                ),
            )
    for basis in ("mole", "mass", "volume"):
        for known in (("n_air",), ("n_fuel",), ("n_air", "n_fuel")):
            yield (
                f"ratio/{basis}/{known}",
                dict(basis=basis, known=known),
                lambda b=basis, k=known: ratio_unit(b, k),
            )
    for scale in (1, 10):
        for active in (
            (True, True, True, True),
            (True, True, True, False),
            (False, True, True, False),
        ):
            yield (
                f"products/{scale}/{active}",
                dict(scale=scale, reactivities=active),
                lambda s=scale, r=active: product_unit(s, r),
            )


SOURCE_COUNTS = dict(
    DAHMEN_MARQUARDT=58,
    FLORIAN_CETANE=85,
    FLORIAN_MON=255,
    FLORIAN_RON=257,
    FLORIAN_LIMING_ANN=532,
    COMBUSTDB_CETANE=408,
    COMBUSTDB_MON=278,
    COMBUSTDB_PREDICTIONS=1066,
    COMBUSTDB_RON=278,
)


def source_unit(db, source, index, header, row):
    record = db.records(source.source_id)[index]
    assert list(record.headers) == header and list(record.raw_fields) == row
    values = []
    for prop, methods in FUEL_METHODS.items():
        for method, source_id in methods.items():
            if source_id != source.source_id:
                continue
            value = db.fuel_value(record.record_id, prop, method=method)
            restored = AssessmentValue.from_dict(json.loads(json.dumps(value.to_dict())))
            assert restored.to_dict() == value.to_dict()
            field = "IGNITION_DELAY" if prop == "ignition_delay" else prop
            raw = row[header.index(field)]
            expected = None if raw == "" else float(raw)
            if expected is None:
                assert value.value is None and value.quality == "missing"
            else:
                compare(value.value, expected)
            if prop in ("RON", "MON", "ignition_delay"):
                actual_methods = set(db.fuel_methods(record.cas_number or record.source_key, prop))
                expected_methods = set(
                    getattr(reference, prop + "_methods")(record.cas_number or record.source_key)
                )
                assert actual_methods == expected_methods, (actual_methods, expected_methods)
                refvalue = getattr(reference, prop)(
                    record.cas_number or record.source_key, method=method
                )
                if refvalue is None:
                    assert value.value is None
                else:
                    compare(value.value, refvalue)
            consumption = {}
            if value.quality == "valid":
                if prop == "ignition_delay":
                    for quantity in ("RON", "MON", "DCN"):
                        report = source_ignition_quality(restored, quantity=quantity)
                        fn = (
                            reference.IDT_to_DCN
                            if quantity == "DCN"
                            else getattr(
                                reference, "Perez_Boehman_" + quantity + "_from_ignition_delay"
                            )
                        )
                        consumption[quantity] = compare(report.value, fn(expected))
                else:
                    temperature = prop in ("POUR_POINT", "CLOUD_POINT")
                    criterion = (
                        Quantity(273.15, "K") if temperature else Quantity(50, "dimensionless")
                    )
                    report = fuel_quality_comparison(
                        restored, criterion, relation="maximum" if temperature else "minimum"
                    )
                    assert report["meets_criterion"] == (
                        expected <= 0 if temperature else expected >= 50
                    )
                    consumption["criterion"] = report
                if prop == "RON":
                    mon = db.fuel_value(
                        record.cas_number or record.source_key, "MON", method=method
                    )
                    if mon.quality == "valid":
                        for quantity, fn in (
                            ("AKI", reference.AKI),
                            ("sensitivity", reference.octane_sensitivity),
                        ):
                            report = source_octane_index(restored, mon, quantity=quantity)
                            consumption[quantity] = compare(report.value, fn(expected, mon.value))
                    else:
                        consumption["missing_partner"] = mon.to_dict()
            values.append(dict(value=value.to_dict(), consumer=consumption))
    return dict(values=values, raw_fields=len(row))


def source_units(directory):
    db = FuelDatabase(directory)
    for source in db.catalog:
        with (directory / source.relative_path).open(encoding="utf8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            rows = list(reader)
        assert len(rows) == SOURCE_COUNTS[source.source_id]
        for index, row in enumerate(rows):
            yield (
                f"{source.source_id}/{index}",
                dict(header=header, row=row),
                lambda s=source, i=index, h=header, r=row: source_unit(db, s, i, h, r),
            )


def basis_unit(atoms, handling):
    real = {k: v for k, v in atoms.items() if k != "Ash"}
    mw = molecular_weight(real) + atoms.get("Ash", 0)
    actual = native.combustion_stoichiometry(atoms, missing_handling=handling)
    expected = reference.combustion_stoichiometry(atoms, MW=mw, missing_handling=handling)
    expanded = {}
    for product, n in actual.items():
        if product == "Ash":
            continue
        for symbol, count in parse_formula(product).items():
            expanded[symbol] = expanded.get(symbol, Fraction(0)) + Fraction(str(n)) * Fraction(
                str(count)
            )
    supported = {"C", "H", "N", "O", "S", "Br", "I", "Cl", "F", "P"}
    if handling == "ash":
        for symbol, count in real.items():
            if symbol not in supported:
                expanded[symbol] = expanded.get(symbol, Fraction(0)) + Fraction(str(count))
    residuals = {k: float(expanded.get(k, 0) - Fraction(str(v))) for k, v in real.items()}
    for difference in residuals.values():
        compare(difference, 0, atol=1e-12, rtol=0)
    physical_mass = sum(
        n * (1 if k == "Ash" else molecular_weight(parse_formula(k))) for k, n in actual.items()
    )
    compare(physical_mass, mw, atol=1e-9, rtol=0)
    difference = 0
    adjusted = dict(expected)
    if handling == "ash" and "Ash" in actual:
        from chemicals.elements import molecular_weight as reference_weight

        supported_atoms = {k: v for k, v in real.items() if k in supported}
        difference = molecular_weight(supported_atoms) - reference_weight(supported_atoms)
        adjusted["Ash"] = expected.get("Ash", 0) - difference
    return dict(
        actual=actual,
        original_reference=expected,
        adjusted_reference=adjusted,
        supported_weight_difference=difference,
        element_residuals=residuals,
        checks=mapping_checks(actual, adjusted),
        mass_residual_g=physical_mass - mw,
    )


def basis_units():
    for identity, inputs, _ in mathematics_units():
        if identity.startswith("stoichiometry/"):
            yield (identity, inputs, lambda i=inputs: basis_unit(i["atoms"], i["handling"]))


def heat_reference_unit(directory, cas, formula, phase, water_phase, amount):
    db = CaloricDatabase(directory)
    ids = {cas: phase, "124-38-9": "gas", "7782-44-7": "gas", "7732-18-5": water_phase}
    refs = {
        k: db.formation_reference(
            k, phase=p, enthalpy_method="CRC_THERMODYNAMIC", pressure_Pa=101325
        )
        for k, p in ids.items()
    }
    refs = {k: CaloricReference(**json.loads(json.dumps(r.to_dict()))) for k, r in refs.items()}
    c = ComponentSpec(cas, formula, default_phase=phase, cas_number=cas)
    c = ComponentSpec.from_dict(json.loads(json.dumps(c.to_dict())))
    products = {"CO2": refs["124-38-9"], "O2": refs["7782-44-7"], "H2O": refs["7732-18-5"]}
    actual = native.combustion_reference_heat(c, refs[cas], products, Quantity(amount, "mol"))
    path = directory / db.source("CRC_THERMODYNAMIC").relative_path
    with path.open(encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    independent = {}
    for key, p in ids.items():
        selected = [row for row in rows if row.get("CAS") == key]
        assert len(selected) == 1, (key, len(selected))
        independent[key] = float(selected[0]["Hfg" if p == "gas" else "Hfl"])
    stoich = reference.combustion_stoichiometry(parse_formula(formula))
    expected = (
        reference.HHV_stoichiometry(
            stoich,
            independent[cas],
            Hf_chemicals={
                "CO2": independent["124-38-9"],
                "O2": independent["7782-44-7"],
                "H2O": independent["7732-18-5"],
            },
        )
        * amount
    )
    return dict(
        check=compare(actual.value, expected, atol=1e-7),
        actual=actual.to_dict(),
        references={k: r.to_dict() for k, r in refs.items()},
        independent_Hf=independent,
    )


def heat_reference_units(directory):
    for cas, formula, phase in (("74-82-8", "CH4", "gas"), ("64-17-5", "C2H6O", "liquid")):
        for water in ("liquid", "gas"):
            for amount in (0.1, 2):
                yield (
                    f"reference-heat/{cas}/{water}/{amount}",
                    dict(cas=cas, formula=formula, phase=phase, water_phase=water, amount=amount),
                    lambda c=cas, f=formula, p=phase, w=water, n=amount: heat_reference_unit(
                        directory, c, f, p, w, n
                    ),
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        choices=(
            "streams",
            "streams-equivalence",
            "mathematics",
            "sources",
            "stoichiometry-basis",
            "reference-heat",
        ),
        required=True,
    )
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert chemicals.__version__ == "1.5.2"
    units = list(
        source_units(args.source_directory)
        if args.block == "sources"
        else mathematics_units()
        if args.block == "mathematics"
        else basis_units()
        if args.block == "stoichiometry-basis"
        else heat_reference_units(args.source_directory)
        if args.block == "reference-heat"
        else stream_units(args.block == "streams-equivalence")
    )
    assert len(units) == {
        "sources": 3217,
        "mathematics": 110,
        "stoichiometry-basis": 34,
        "reference-heat": 8,
    }.get(args.block, 84)
    results = []
    started = time.perf_counter()
    print(f"{args.block}: fixed denominator {len(units)}", flush=True)
    for identity, inputs, run in units:
        row = dict(unit=identity, inputs=inputs)
        try:
            row.update(status="passed", metrics=run())
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        results.append(row)
        if len(results) % 200 == 0 or len(results) == len(units):
            dt = time.perf_counter() - started
            print(
                f"{len(results)}/{len(units)}; {len(results) / dt:.1f}/s; "
                f"ETA {(len(units) - len(results)) * dt / len(results):.1f}s",
                flush=True,
            )
    result = dict(
        mode="development",
        block=args.block,
        reference_version=chemicals.__version__,
        total=len(results),
        passed=sum(r["status"] == "passed" for r in results),
        failed=sum(r["status"] == "failed" for r in results),
        seconds=time.perf_counter() - started,
        results=results,
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}))
    for r in [r for r in results if r["status"] == "failed"][:3]:
        print(r["unit"], r["failure"])
    return bool(result["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
