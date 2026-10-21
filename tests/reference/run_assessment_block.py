"""Fixed complete F3 source and mathematical development blocks."""
# ruff: noqa: C408 -- scientific input dictionaries

import argparse
import csv
import gzip
import json
import time
import traceback
from pathlib import Path

import chemicals
from chemicals import environment, safety
from run_solubility_block import compare

from chemworld.data.assessment_catalog import ASSESSMENT_METHODS
from chemworld.data.assessments import AssessmentDatabase, AssessmentValue
from chemworld.foundation.units import Quantity
from chemworld.physchem.safety_environment import (
    crowl_louvar_flammability_limit,
    emission_impact,
    fire_mixture_limit,
    gas_exposure_comparison,
    iso_10156_lower_limit,
    mass_concentration_to_ppmv,
    nfpa_30_classification,
    octanol_water_partition,
    ppmv_to_mass_concentration,
    source_flammable_mixture,
    suzuki_flammability_limit,
)
from chemworld.physchem.specs import ComponentSpec, MixtureSpec

COUNTS = dict(
    IPCC_2007=62,
    IPCC_2014=200,
    IPCC_2021=248,
    ODP=71,
    CRC_LOGP=449,
    SYRRES=11569,
    IEC_2010=327,
    NFPA_2008=231,
    SERAT=870,
    IARC=863,
    NTP=234,
    ONTARIO=765,
    ONTARIO_RAW=714,
    WIKIDATA=10122,
)


def reference_module(prop):
    return environment if prop in {"GWP", "GTP", "ODP", "logP"} else safety


def consume(value, domain_checked=False):
    prop, number = value.property_id, value.value
    if value.quality != "valid":
        return dict(unavailable=value.quality)
    if prop in {"Carcinogen", "Skin"} or value.unit is None:
        return dict(semantic=value.to_dict())
    assert value.evaluate().value == number
    if prop in {"GWP", "GTP", "ODP"}:
        actual = emission_impact(
            {value.component_id: Quantity(1, "kg")}, {value.component_id: value}
        )
        return compare(actual.value, number)
    if prop == "logP":
        result = octanol_water_partition(
            {value.component_id: 1},
            {value.component_id: value},
            water_volume_L=1,
            octanol_volume_L=0.25,
            temperature_K=300,
        )
        coefficient = 10**number
        expected = coefficient * 0.25 / (1 + coefficient * 0.25)
        return dict(
            organic=compare(
                result.organic_amounts_mol[value.component_id], expected, atol=1e-10, rtol=0
            ),
            aqueous=compare(
                result.aqueous_amounts_mol[value.component_id], 1 - expected, atol=1e-10, rtol=0
            ),
            conservation=compare(result.material_balance_error_mol, 0, atol=1e-10, rtol=0),
        )
    if prop in {"TWA", "STEL", "Ceiling"}:
        if value.unit == "ppmv":
            converted = ppmv_to_mass_concentration(
                number, 40, temperature_K=300, pressure_Pa=101325
            )
            expected = safety.ppmv_to_mgm3(number, 40, T=300, P=101325)
            inverse = mass_concentration_to_ppmv(
                converted, 40, temperature_K=300, pressure_Pa=101325
            )
        else:
            converted = mass_concentration_to_ppmv(
                number, 40, temperature_K=300, pressure_Pa=101325
            )
            expected = safety.mgm3_to_ppmv(number, 40, T=300, P=101325)
            inverse = ppmv_to_mass_concentration(
                converted, 40, temperature_K=300, pressure_Pa=101325
            )
        return dict(
            conversion=compare(converted, expected), inverse=compare(inverse, number, atol=1e-10)
        )
    if prop in {"LFL", "UFL"}:
        if domain_checked and not 0 < number <= 1:
            try:
                fire_mixture_limit([1], [number])
            except ValueError as exc:
                return dict(reported_value=number, expected_rejection=str(exc))
            raise AssertionError("nonphysical source limit accepted by mixture consumer")
        return compare(fire_mixture_limit([1], [number]), safety.fire_mixing([1], [number]))
    if prop == "T_flash":
        actual = nfpa_30_classification(number, boiling_temperature_K=350)
        assert actual == safety.NFPA_30_classification(number, Tb=350)
        return dict(classification=actual)
    return compare(value.input_value().to("degC").to("K").value, number)


def source_unit(database, source, index, header, raw, key, domain_checked=False):
    record = database.records(source.source_id)[index]
    assert record.source_key == key and list(record.headers) == header
    assert list(record.raw_fields) == raw
    assert json.loads(
        json.dumps(dict(zip(record.field_names, record.raw_fields, strict=True)))
    ) == dict(zip(record.field_names, record.raw_fields, strict=True))
    results = []
    for spec in (s for s in ASSESSMENT_METHODS if s.source_id == source.source_id):
        value = database.assessment(record.record_id, spec.property_id, method=spec.method)
        restored = AssessmentValue.from_dict(json.loads(json.dumps(value.to_dict())))
        assert restored.to_dict() == value.to_dict()
        module = reference_module(spec.property_id)
        reference_identity = key if source.source_id == "ONTARIO" else record.cas_number or key
        expected = getattr(module, spec.property_id)(CASRN=reference_identity, method=spec.method)
        if expected is None:
            assert value.value is None and value.quality in {"missing", "not_listed"}, value
        elif isinstance(expected, tuple):
            assert value.unit == ("ppmv" if expected[1] == "ppm" else expected[1])
            compare(value.value, expected[0])
        elif isinstance(expected, str | bool):
            assert value.value == expected, (value, expected)
        else:
            compare(value.value, float(expected))
        actual_methods = set(
            database.assessment_methods(record.cas_number or key, spec.property_id)
        )
        expected_methods = set(
            getattr(module, spec.property_id + "_methods")(CASRN=reference_identity)
        )
        assert actual_methods == expected_methods, (actual_methods, expected_methods)
        result = dict(
            method=spec.method,
            assessment=value.to_dict(),
            consumer=consume(restored, domain_checked),
        )
        if spec.property_id in {"TWA", "STEL", "Ceiling"}:
            result["both_source_units"] = []
            for unit, column in (("ppmv", "ppm"), ("mg/m^3", "mg/m^3")):
                cell = database.assessment(
                    record.record_id, spec.property_id, method=spec.method, unit=unit
                )
                assert cell.value == record.raw(f"{spec.property_id} ({column})")
                result["both_source_units"].append(
                    dict(value=cell.to_dict(), consumer=consume(cell))
                )
        results.append(result)
    return dict(assessments=results, raw_fields=len(raw))


def source_units(directory, domain_checked=False):
    database = AssessmentDatabase(directory)
    for source in database.catalog:
        path = directory / source.relative_path
        if source.kind == "mapping-json":
            payload = json.loads(path.read_text(encoding="utf8"))
            raw_rows = [(list(row), list(row.values()), key) for key, row in payload.items()]
        else:
            opener = gzip.open if source.kind == "tsv-gzip" else open
            with opener(path, "rt", encoding="utf8", newline="") as stream:
                reader = csv.reader(stream, delimiter="\t")
                header = next(reader)
                key_column = header.index("CAS") if "CAS" in header else 0
                raw_rows = [(header, row, row[key_column]) for row in reader]
        assert len(raw_rows) == COUNTS[source.source_id] == len(database.records(source.source_id))
        for index, (header, raw, key) in enumerate(raw_rows):
            yield (
                f"{source.source_id}/{index}",
                dict(source=source.source_id, row=index, header=header, raw=raw, key=key),
                lambda s=source, i=index, h=header, r=raw, k=key: source_unit(
                    database, s, i, h, r, k, domain_checked
                ),
            )


def concentration_unit(c, mw, t, p):
    actual = ppmv_to_mass_concentration(c, mw, temperature_K=t, pressure_Pa=p)
    expected = safety.ppmv_to_mgm3(c, mw, T=t, P=p)
    inverse = mass_concentration_to_ppmv(actual, mw, temperature_K=t, pressure_Pa=p)
    return dict(
        forward=compare(actual, expected),
        inverse=compare(inverse, safety.mgm3_to_ppmv(expected, mw, T=t, P=p), atol=1e-10),
        roundtrip=compare(inverse, c, atol=1e-10),
    )


def classification_unit(flash, **kwargs):
    native = {"Tb": "boiling_temperature_K", "Psat_100F": "vapor_pressure_at_100F_Pa"}
    value = nfpa_30_classification(flash, **{native[k]: v for k, v in kwargs.items()})
    expected = safety.NFPA_30_classification(flash, **kwargs)
    assert value == expected, (value, expected)
    return dict(value=value)


def mathematics_units():
    for c in (0, 1, 100):
        for mw in (16, 40):
            for t in (273.15, 300, 400):
                for p in (50000, 101325, 500000):
                    yield (
                        f"concentration/{c}/{mw}/{t}/{p}",
                        dict(c=c, MW=mw, T=t, P=p),
                        lambda c=c, mw=mw, t=t, p=p: concentration_unit(c, mw, t, p),
                    )
    for heat in (-0.9e6, -1.5e6, -3.5366e6, -5e6, -6e6):
        for limit in ("LFL", "UFL"):
            yield (
                f"Suzuki/{heat}/{limit}",
                dict(Hc=heat, limit=limit),
                lambda h=heat, lim=limit: compare(
                    suzuki_flammability_limit(h, limit=lim), getattr(safety, "Suzuki_" + lim)(h)
                ),
            )
    for atoms in (
        dict(C=1, H=4),
        dict(C=2, H=6),
        dict(C=2, H=6, O=1),
        dict(C=6, H=14),
        dict(C=3, H=6, O=1),
    ):
        for limit in ("LFL", "UFL"):
            yield (
                f"Crowl/{atoms}/{limit}",
                dict(atoms=atoms, limit=limit),
                lambda a=atoms, lim=limit: compare(
                    crowl_louvar_flammability_limit(a, limit=lim),
                    getattr(safety, "Crowl_Louvar_" + lim)(a),
                ),
            )
    for fractions in ((1, 0), (0.2, 0.8), (0, 0.4, 0.6), (0.2, 0.3, 0.5)):
        for limits in ((0.02, 0.08, 0.15), (0.05, 0.2, 0.4)):
            limits = limits[: len(fractions)]
            yield (
                f"mix/{fractions}/{limits}",
                dict(fractions=fractions, limits=limits),
                lambda y=fractions, lim=limits: compare(
                    fire_mixture_limit(y, lim), safety.fire_mixing(y, lim)
                ),
            )
    for flash in (250, 295, 300, 320, 350, 380):
        for boiling in (290, 350):
            yield (
                f"NFPA/{flash}/{boiling}",
                dict(flash=flash, boiling=boiling),
                lambda f=flash, b=boiling: classification_unit(f, Tb=b),
            )
    for fahrenheit in (73, 100, 140, 200):
        flash = (fahrenheit - 32) * 5 / 9 + 273.15
        yield (
            f"NFPA-boundary/{fahrenheit}",
            dict(flash=flash, Tb=350),
            lambda f=flash: classification_unit(f, Tb=350),
        )
    for pressure in (90000, 120000):
        yield (
            f"NFPA-P/{pressure}",
            dict(flash=250, Psat_100F=pressure),
            lambda p=pressure: classification_unit(250, Psat_100F=p),
        )
    yield (
        "NFPA-Tb-boundary",
        dict(flash=250, Tb=(100 - 32) * 5 / 9 + 273.15),
        lambda: classification_unit(250, Tb=(100 - 32) * 5 / 9 + 273.15),
    )


def restored_assessment(database, identity, prop, method, **kwargs):
    value = database.assessment(identity, prop, method=method, **kwargs)
    return AssessmentValue.from_dict(json.loads(json.dumps(value.to_dict())))


def impact_unit(database, prop, method, masses):
    identities = ("75-69-4", "56-23-5") if prop == "ODP" else ("74-82-8", "10024-97-2")
    units = ("kg", "g")
    source = {k: restored_assessment(database, k, prop, method) for k in identities}
    result = emission_impact(
        {k: Quantity(m, u) for k, m, u in zip(identities, masses, units, strict=True)}, source
    )
    expected = sum(
        (m if u == "kg" else m / 1000) * getattr(environment, prop)(k, method=method)
        for k, m, u in zip(identities, masses, units, strict=True)
    )
    return dict(
        impact=compare(result.value, expected), source=[v.to_dict() for v in source.values()]
    )


def mixture(identities, formulas, fractions, t, p):
    components = [
        ComponentSpec(k, f, default_phase="gas") for k, f in zip(identities, formulas, strict=True)
    ]
    value = MixtureSpec.from_mole_fractions(
        components, fractions, phase_label="gas", temperature_K=t, pressure_Pa=p
    )
    return MixtureSpec.from_dict(json.loads(json.dumps(value.to_dict())))


def exposure_unit(database, x, t, p, prop, unit):
    gas = mixture(("acetone", "nitrogen"), ("C3H6O", "N2"), (x, 1 - x), t, p)
    source = restored_assessment(database, "67-64-1", prop, "Ontario Limits", unit=unit)
    result = gas_exposure_comparison(
        gas, "acetone", source, source_identity="67-64-1", averaging_window=prop
    )
    concentration = (
        x * 1e6
        if unit == "ppmv"
        else safety.ppmv_to_mgm3(x * 1e6, gas.molecular_weights_g_mol[0], T=t, P=p)
    )
    field = f"{prop} ({'ppm' if unit == 'ppmv' else unit})"
    # Read the independent original JSON view, including its derived-unit provenance.
    raw = json.loads(
        (database.source_directory / "Safety/Ontario Exposure Limits.json").read_text(
            encoding="utf8"
        )
    )["67-64-1"][field]
    return dict(
        ratio=compare(result.value, concentration / raw),
        source=source.to_dict(),
        mixture=gas.to_dict(),
    )


def combustible_unit(database, x, prop, method):
    gas = mixture(("methane", "ethane"), ("CH4", "C2H6"), (x, 1 - x), 300, 101325)
    identities = dict(methane="74-82-8", ethane="74-84-0")
    source = {k: restored_assessment(database, cas, prop, method) for k, cas in identities.items()}
    result = source_flammable_mixture(gas, source, source_identities=identities, limit=prop)
    expected = safety.fire_mixing(
        [x, 1 - x], [getattr(safety, prop)(CASRN=cas, method=method) for cas in identities.values()]
    )
    return dict(
        limit=compare(result.value, expected),
        mixture=gas.to_dict(),
        sources=[v.to_dict() for v in source.values()],
    )


def partition_unit(database, amounts, volumes, method):
    identities = ("64-17-5", "71-36-3")
    source = {k: restored_assessment(database, k, "logP", method) for k in identities}
    result = octanol_water_partition(
        dict(zip(identities, amounts, strict=True)),
        source,
        water_volume_L=volumes[0],
        octanol_volume_L=volumes[1],
        temperature_K=300,
    )
    checks = []
    for k, amount in zip(identities, amounts, strict=True):
        ratio = 10 ** environment.logP(k, method=method) * volumes[1] / volumes[0]
        organic = amount * ratio / (1 + ratio)
        aqueous = amount - organic
        checks.extend(
            (
                compare(result.organic_amounts_mol[k], organic, atol=1e-10, rtol=0),
                compare(result.aqueous_amounts_mol[k], aqueous, atol=1e-10, rtol=0),
                compare(
                    result.organic_amounts_mol[k] + result.aqueous_amounts_mol[k],
                    amount,
                    atol=1e-10,
                    rtol=0,
                ),
            )
        )
    return dict(checks=checks, partition=result.to_dict())


def consumer_units(directory):
    database = AssessmentDatabase(directory)
    for prop, method in (
        ("GWP", "IPCC (2021) 20yr"),
        ("GWP", "IPCC (2021) 100yr"),
        ("GTP", "IPCC (2021) 50yr"),
        ("GTP", "IPCC (2021) 100yr"),
        ("ODP", "ODP1 Max"),
        ("ODP", "ODP2 Max"),
    ):
        for masses in ((1, 500), (2, 3000)):
            yield (
                f"impact/{prop}/{method}/{masses}",
                dict(prop=prop, method=method, masses=masses),
                lambda p=prop, m=method, ms=masses: impact_unit(database, p, m, ms),
            )
    for x in (0.0001, 0.0005):
        for t, p in ((300, 101325), (350, 200000)):
            for prop in ("TWA", "STEL"):
                for unit in ("ppmv", "mg/m^3"):
                    yield (
                        f"exposure/{x}/{t}/{p}/{prop}/{unit}",
                        dict(x=x, T=t, P=p, prop=prop, unit=unit),
                        lambda x=x, t=t, p=p, prop=prop, u=unit: exposure_unit(
                            database, x, t, p, prop, u
                        ),
                    )
    for x in (0.1, 0.5, 0.9):
        for prop in ("LFL", "UFL"):
            for method in ("IEC 60079-20-1 (2010)", "NFPA 497 (2008)"):
                yield (
                    f"combustible/{x}/{prop}/{method}",
                    dict(x=x, prop=prop, method=method),
                    lambda x=x, p=prop, m=method: combustible_unit(database, x, p, m),
                )
    for amounts in ((0.1, 0.2), (2, 3)):
        for volumes in ((0.2, 0.5), (1, 0.25)):
            for method in ("CRC", "SYRRES"):
                yield (
                    f"partition/{amounts}/{volumes}/{method}",
                    dict(amounts=amounts, volumes=volumes, method=method),
                    lambda a=amounts, v=volumes, m=method: partition_unit(database, a, v, m),
                )


ISO_DILUENTS = (
    ("7782-44-7", "O2", 1),
    ("7727-37-9", "N2", 1),
    ("124-38-9", "CO2", 1.5),
    ("7440-59-7", "He", 0.9),
    ("7440-37-1", "Ar", 0.55),
    ("7440-01-9", "Ne", 0.7),
    ("7439-90-9", "Kr", 0.5),
    ("7440-63-3", "Xe", 0.5),
    ("7446-09-5", "SO2", 1.5),
    ("2551-62-4", "SF6", 4),
    ("75-73-0", "CF4", 2),
    ("76-19-7", "C3F8", 1.5),
    ("354-33-6", "C2HF5", 3.5),
)


def iso_unit(database, diluents, fractions):
    from chemworld.physchem import ISO_10156_DILUENT_COEFFICIENTS

    identities = ("74-82-8", "74-84-0", *(d[0] for d in diluents))
    formulas = ("CH4", "C2H6", *(d[1] for d in diluents))
    for cas, _, coefficient in diluents:
        assert ISO_10156_DILUENT_COEFFICIENTS[cas] == coefficient
        assert coefficient == safety.ISO_10156_2017_Kks[cas]
    gas = mixture(identities, formulas, fractions, 300, 101325)
    source = {
        cas: restored_assessment(database, cas, "LFL", "IEC 60079-20-1 (2010)")
        if index < 2
        else None
        for index, cas in enumerate(identities)
    }
    limits = [s.value if s is not None else None for s in source.values()]
    report = source_flammable_mixture(
        gas,
        source,
        source_identities={k: k for k in identities},
        limit="LFL",
        method="ISO_10156_2017",
    )
    serialized = json.loads(json.dumps(report.to_dict()))
    assert serialized == report.to_dict()
    expected = safety.LFL_ISO_10156_2017(list(fractions), limits, list(identities))
    checks = [
        compare(serialized["value"], expected),
        compare(iso_10156_lower_limit(fractions, limits, identities), expected),
    ]
    if not diluents:
        checks.append(compare(serialized["value"], safety.fire_mixing(list(fractions), limits)))
    return dict(
        checks=checks,
        mixture=gas.to_dict(),
        report=report.to_dict(),
        sources=[s.to_dict() if s is not None else None for s in source.values()],
    )


def iso_published_unit():
    fractions = [0.15, 0.15, 0.3, 0.3895, 0.0105]
    limits = [0.04, 0.044, None, None, None]
    identities = ["1333-74-0", "74-82-8", "124-38-9", "7727-37-9", "7782-44-7"]
    result = iso_10156_lower_limit(fractions, limits, identities)
    return dict(
        printed=compare(result, 0.1427372274, atol=1e-10, rtol=0),
        reference=compare(result, safety.LFL_ISO_10156_2017(fractions, limits, identities)),
        fractions=fractions,
        limits=limits,
        identities=identities,
    )


def selection_unit(database, identity, hc, atoms, limit, method):
    actual_methods = set(
        database.flammability_methods(
            identity, limit=limit, heat_of_combustion_J_mol=hc, atoms=atoms
        )
    )
    expected_methods = set(getattr(safety, limit + "_methods")(CASRN=identity, Hc=hc, atoms=atoms))
    assert actual_methods == expected_methods, (actual_methods, expected_methods)
    actual = database.flammability_value(
        identity, limit=limit, method=method, heat_of_combustion_J_mol=hc, atoms=atoms
    )
    serialized = json.loads(json.dumps(actual.to_dict()))
    assert serialized == actual.to_dict()
    expected = (
        getattr(safety, limit)(CASRN=identity, method=method)
        if method == "IEC 60079-20-1 (2010)"
        else getattr(safety, limit + "_estimate")(Hc=hc, atoms=atoms, method=method)
    )
    return dict(
        check=compare(actual.value, expected),
        report=actual.to_dict(),
        available_methods=sorted(actual_methods),
    )


def iso_selection_units(directory):
    database = AssessmentDatabase(directory)
    for diluent in ISO_DILUENTS:
        for a in (0.3, 0.6, 0.9):
            zs = (0.4 * a, 0.6 * a, 1 - a)
            yield (
                f"iso/{diluent[0]}/{a}",
                dict(diluent=diluent, fractions=zs),
                lambda d=diluent, z=zs: iso_unit(database, (d,), z),
            )
    for x in (0.1, 0.5, 0.9):
        zs = (x, 1 - x)
        yield (f"iso/pure/{x}", dict(fractions=zs), lambda z=zs: iso_unit(database, (), z))
    for a in (0.3, 0.6, 0.9):
        zs = (0.4 * a, 0.6 * a, *((1 - a) / 3 for _ in range(3)))
        ds = (ISO_DILUENTS[2], ISO_DILUENTS[1], ISO_DILUENTS[0])
        yield (
            f"iso/multi/{a}",
            dict(diluents=ds, fractions=zs),
            lambda d=ds, z=zs: iso_unit(database, d, z),
        )
    yield ("iso/published", dict(source="public documented sample"), iso_published_unit)
    for cas, hc, atoms in (
        ("74-82-8", -900000, {"C": 1, "H": 4}),
        ("74-84-0", -1500000, {"C": 2, "H": 6}),
    ):
        for limit in ("LFL", "UFL"):
            for method in ("IEC 60079-20-1 (2010)", "Suzuki (1994)", "Crowl and Louvar (2001)"):
                yield (
                    f"selection/{cas}/{limit}/{method}",
                    dict(identity=cas, Hc=hc, atoms=atoms, limit=limit, method=method),
                    lambda c=cas, h=hc, a=atoms, prop=limit, m=method: selection_unit(
                        database, c, h, a, prop, m
                    ),
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        choices=("sources", "sources-domain", "mathematics", "consumers", "iso-selection"),
        required=True,
    )
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert chemicals.__version__ == "1.5.2"
    units = list(
        mathematics_units()
        if args.block == "mathematics"
        else consumer_units(args.source_directory)
        if args.block == "consumers"
        else iso_selection_units(args.source_directory)
        if args.block == "iso-selection"
        else source_units(args.source_directory, domain_checked=args.block == "sources-domain")
    )
    assert len(units) == {"mathematics": 101, "consumers": 48, "iso-selection": 58}.get(
        args.block, 26725
    )
    start = time.perf_counter()
    print(f"{args.block}: fixed denominator {len(units)}", flush=True)
    results = []
    for identity, inputs, run in units:
        row = dict(unit=identity, inputs=inputs)
        try:
            row.update(metrics=run(), status="passed")
        except Exception:
            row.update(failure=traceback.format_exc(), status="failed")
        results.append(row)
        if len(results) % 200 == 0 or len(results) == len(units):
            elapsed = time.perf_counter() - start
            print(
                f"{len(results)}/{len(units)}; {len(results) / elapsed:.1f}/s; "
                f"ETA {(len(units) - len(results)) * elapsed / len(results):.1f}s",
                flush=True,
            )
    summary = dict(
        mode="development",
        block=args.block,
        reference_version=chemicals.__version__,
        total=len(results),
        passed=sum(r["status"] == "passed" for r in results),
        failed=sum(r["status"] == "failed" for r in results),
        seconds=time.perf_counter() - start,
        results=results,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    for row in [r for r in results if r["status"] == "failed"][:3]:
        print(row["unit"], row["failure"])
    return bool(summary["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
