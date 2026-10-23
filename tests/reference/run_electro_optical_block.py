"""Complete F2 development blocks. Reference implementations stay in tests."""
# ruff: noqa: C408 -- compact scientific parameters

import argparse
import csv
import json
import math
import time
import traceback
from functools import lru_cache
from pathlib import Path

import chemicals
import mpmath as mp
from chemicals import dipole, iapws, permittivity, refractivity, viscosity
from run_solubility_block import compare, differences

from chemworld.data.electro_optical import (
    DIPOLE_SOURCES,
    REFRACTIVE_INDEX_SOURCES,
    ElectroOpticalDatabase,
    RefractiveIndexObservation,
)
from chemworld.data.identifiers import IdentifierDatabase, UnknownIdentifierError
from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.electro_optics import (
    brix_reading,
    brix_to_refractive_index,
    molar_refractivity,
    parallel_plate_capacitance,
    polarizability_volume,
    refractive_index_from_molar_refractivity,
    refractive_index_tde,
    refractive_index_to_brix,
    relative_permittivity_crc,
    water_permittivity_iapws,
    water_refractive_index_iapws,
)
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.viscosity_kernels import viscosity_value

R8_POINTS = (
    (240, 54.33701, 104.34982),
    (300, 55.31735, 77.74735),
    (300, 55.56148, 78.11269),
    (300, 68.69265, 103.69632),
    (650, 2.24692, 1.26715),
    (650, 40.31090, 17.71733),
    (650, 52.58636, 26.62132),
    (870, 1.45275, 1.12721),
    (870, 20.98927, 4.98281),
    (870, 45.01376, 15.09746),
)
R9_POINTS = (
    (
        0.22650,
        (
            ("1.394527", "1.394711", "1.396526", "1.412733"),
            ("1.0002168", "1.375622", "1.377286", "1.391983"),
            ("1.0001683", "1.0017754", "1.338299", "1.359330"),
            ("1.0001008", "1.0010155", "1.0109906", "1.198312"),
        ),
    ),
    (
        0.58900,
        (
            ("1.334344", "1.334494", "1.335969", "1.349101"),
            ("1.0001876", "1.318725", "1.320084", "1.332057"),
            ("1.0001456", "1.0015359", "1.287891", "1.305191"),
            ("1.0000871", "1.0008773", "1.0094939", "1.170231"),
        ),
    ),
    (
        1.01398,
        (
            ("1.326135", "1.326279", "1.327710", "1.340435"),
            ("1.0001837", "1.311257", "1.312577", "1.324202"),
            ("1.0001427", "1.0015052", "1.281529", "1.298369"),
            ("1.0000856", "1.0008619", "1.0093267", "1.167119"),
        ),
    ),
)
METHOD_MAP = {
    "CCCBDB": "CCCBDB",
    "MULLER": "MULLER",
    "POLING": "POLING",
    "PSI4_DIPOLE": "PSI4_2022A",
    "CRC_RI": "CRC",
    "WIKIDATA": "WIKIDATA",
}
COUNTS = dict(
    CRC_PERMITTIVITY=1303,
    CCCBDB=424,
    MULLER=254,
    POLING=264,
    PSI4_DIPOLE=411,
    CRC_RI=4490,
    WIKIDATA=10122,
)


def value_unit(native, reference, args):
    return {"value": compare(native(*args), reference(*args))}


def partial_unit(native, reference, args):
    values = []
    for i, argument in enumerate(args):

        def n(x, i=i):
            a = list(args)
            a[i] = x
            return native(*a)

        def r(x, i=i):
            a = list(args)
            a[i] = x
            return reference(*a)

        value, slope = derivative_values(n, argument)
        values.append(
            {
                "axis": i,
                "value": compare(value, r(argument)),
                "slope": differences(slope, r, argument),
            }
        )
    return {"partials": values}


def r8_unit(t, molar, expected):
    rho = molar * 1000 * 0.018015268
    actual = water_permittivity_iapws(t, rho)
    reference = permittivity.permittivity_IAPWS(t, rho)
    rounded_range = [
        permittivity.permittivity_IAPWS(t, (molar + offset) * 1000 * 0.018015268)
        for offset in (-0.000005, 0.000005)
    ]
    tolerance = 0.000005 + max(abs(x - reference) for x in rounded_range)
    return {
        "reference": compare(actual, reference),
        "official": compare(actual, expected, atol=tolerance, rtol=0),
        "molar_density_rounding": rounded_range,
        "official_tolerance": tolerance,
    }


def crc_unit(t):
    p = (487, -1.5, 0.00129, 0)
    actual, slope = derivative_values(lambda x: relative_permittivity_crc(x, *p), t)
    return {
        "value": compare(actual, permittivity.permittivity_CRC(t, *p)),
        "slope": differences(slope, lambda x: permittivity.permittivity_CRC(x, *p), t),
    }


def tde_unit(t, wavelength, bs, cs):
    def n(T, w):
        return refractive_index_tde(T, bs, cs, w)

    def r(T, w):
        return refractivity.TDE_RIXExpansion(T, bs, cs, w)

    return partial_unit(n, r, (t, wavelength))


def water_units():
    for t in (240, 300, 650, 870):
        for rho in (0, 1, 100, 1000):
            yield (
                f"water-epsilon/{t}/{rho}",
                dict(T=t, rho=rho),
                lambda t=t, rho=rho: value_unit(
                    water_permittivity_iapws, permittivity.permittivity_IAPWS, (t, rho)
                ),
            )
    for t in (273.15, 400, 700):
        for rho in (0, 100, 1000):
            for wave in (0.2265e-6, 0.589e-6, 1.01398e-6):
                yield (
                    f"water-index/{t}/{rho}/{wave}",
                    dict(T=t, rho=rho, wavelength=wave),
                    (
                        lambda t=t, r=rho, w=wave: value_unit(
                            water_refractive_index_iapws, refractivity.RI_IAPWS, (t, r, w)
                        )
                    ),
                )
    for t, molar, expected in R8_POINTS:
        yield (
            f"R8-table4/{t}/{molar}",
            dict(T=t, molar_density_mol_dm3=molar, epsilon=expected),
            (lambda t=t, m=molar, e=expected: r8_unit(t, m, e)),
        )
    for t in (300, 650):
        for rho in (40, 900):
            for wave in (0.3e-6, 0.6e-6, 0.9e-6):
                yield (
                    f"RI-partials/{t}/{rho}/{wave}",
                    dict(T=t, rho=rho, wavelength=wave),
                    (
                        lambda t=t, r=rho, w=wave: partial_unit(
                            water_refractive_index_iapws, refractivity.RI_IAPWS, (t, r, w)
                        )
                    ),
                )
    for t in (300, 500, 800):
        for rho in (40, 900):
            yield (
                f"dielectric-partials/{t}/{rho}",
                dict(T=t, rho=rho),
                (
                    lambda t=t, r=rho: partial_unit(
                        water_permittivity_iapws, permittivity.permittivity_IAPWS, (t, r)
                    )
                ),
            )
    for t in (400, 450, 500):
        yield f"CRC/{t}", dict(T=t), lambda t=t: crc_unit(t)
    for i, (bs, cs) in enumerate(
        (
            ([-0.000125041, 1.33245], [1.20771e-7, -3.56795e-5, 0]),
            ([1e-6, -0.001, 1.5], [2e-8, 0]),
            ([1.4], [0]),
        )
    ):
        for t in (280, 330):
            for w in (400e-9, 700e-9):
                yield (
                    f"TDE/{i}/{t}/{w}",
                    dict(T=t, wavelength=w, Bs=bs, Cs=cs),
                    (lambda t=t, w=w, bs=bs, cs=cs: tde_unit(t, w, bs, cs)),
                )
    for t, r, w in ((300, 1000, 1.5e-6), (800, 100, 600e-9)):
        yield (
            f"RI-extrapolated/{t}/{r}/{w}",
            dict(T=t, rho=r, wavelength=w),
            (
                lambda t=t, r=r, w=w: value_unit(
                    lambda *a: water_refractive_index_iapws(*a, extrapolate=True),
                    refractivity.RI_IAPWS,
                    (t, r, w),
                )
            ),
        )


def official_ri_unit(t, pressure, wavelength, expected):
    rho = iapws.iapws95_rho(t, pressure)
    actual = water_refractive_index_iapws(t, rho, wavelength)
    return {
        "rho_from_reference_iapws95": rho,
        "reference": compare(actual, refractivity.RI_IAPWS(t, rho, wavelength)),
        "official": compare(
            actual, float(expected), atol=10 ** -len(expected.split(".")[1]), rtol=0
        ),
    }


def published_dielectric_mp(t, rho):
    """Independent 50-digit transcription of the published IAPWS R8-97 equations."""
    t, rho = mp.mpf(str(t)), mp.mpf(str(rho))
    coefficients = (
        (".978224486826", 1, ".25"),
        ("-.957771379375", 1, "1"),
        (".237511794148", 1, "2.5"),
        (".714692244396", 2, "1.5"),
        ("-.298217036956", 3, "1.5"),
        ("-.108863472196", 3, "2.5"),
        (".0949327488264", 4, "2"),
        ("-.00980469816509", 5, "2"),
        (".000016516763497", 6, "5"),
        (".0000937359795772", 7, ".5"),
        ("-.00000000012317921872", 10, "10"),
    )
    delta, tau = rho / 322, mp.mpf("647.096") / t
    g = 1 + mp.fsum(mp.mpf(n) * delta**i * tau ** mp.mpf(j) for n, i, j in coefficients)
    g += mp.mpf(".00196096504426") * delta * (t / 228 - 1) ** mp.mpf("-1.2")
    epsilon0 = 1 / (mp.mpf("4e-7") * mp.pi * 299792458**2)
    molar = rho / mp.mpf(".018015268")
    a = (
        mp.mpf("6.0221367e23")
        * mp.mpf("6.138e-30") ** 2
        * molar
        * g
        / (epsilon0 * mp.mpf("1.380658e-23") * t)
    )
    b = mp.mpf("6.0221367e23") * mp.mpf("1.636e-40") * molar / (3 * epsilon0)
    return (1 + a + 5 * b + mp.sqrt(9 + 2 * a + 18 * b + a * a + 10 * a * b + 9 * b * b)) / (
        4 * (1 - b)
    )


def precise_partials(native, reference, args):
    result = []
    with mp.workdps(50):
        for i, argument in enumerate(args):

            def n(x, i=i):
                a = list(args)
                a[i] = x
                return native(*a)

            def r(x, i=i):
                a = [mp.mpf(str(v)) for v in args]
                a[i] = x
                return reference(*a)

            x = mp.mpf(str(argument))
            value, slope = derivative_values(n, argument)
            levels = []
            for divisor in (1, 2):
                h = x * mp.mpf(".001") / divisor
                levels.append(
                    float((r(x - 2 * h) - 8 * r(x - h) + 8 * r(x + h) - r(x + 2 * h)) / (12 * h))
                )
            compare(*levels, atol=1e-8, rtol=3e-5)
            result.append(
                dict(
                    axis=i,
                    value=compare(value, float(r(x))),
                    slopes=[compare(slope, v, atol=1e-8, rtol=3e-5) for v in levels],
                )
            )
    return result


def dielectric_constants_unit(t, rho, *, official=None, molar=None, partials=False):
    with mp.workdps(50):
        native = water_permittivity_iapws(t, rho)
        precise = published_dielectric_mp(t, rho)
        result = dict(
            original=compare(native, float(precise)),
            compatibility=compare(
                water_permittivity_iapws(t, rho, constants="chemicals_1_5_2"),
                permittivity.permittivity_IAPWS(t, rho),
            ),
        )
        if official is not None:
            rounded = [
                float(
                    published_dielectric_mp(
                        t, (mp.mpf(str(molar)) + offset) * 1000 * mp.mpf(".018015268")
                    )
                )
                for offset in (mp.mpf("-.000005"), mp.mpf(".000005"))
            ]
            tolerance = 0.000005 + max(abs(v - float(precise)) for v in rounded)
            result.update(
                official=compare(native, official, atol=tolerance, rtol=0),
                density_rounding=rounded,
                official_tolerance=tolerance,
            )
        if partials:
            result["original_partials"] = precise_partials(
                water_permittivity_iapws, published_dielectric_mp, (t, rho)
            )
            result["compatibility_partials"] = partial_unit(
                lambda t, r: water_permittivity_iapws(t, r, constants="chemicals_1_5_2"),
                permittivity.permittivity_IAPWS,
                (t, rho),
            )
        return result


def dielectric_constants_units():
    for t in (240, 300, 650, 870):
        for rho in (0, 1, 100, 1000):
            yield (
                f"constants/value/{t}/{rho}",
                dict(T=t, rho=rho),
                lambda t=t, r=rho: dielectric_constants_unit(t, r),
            )
    for t, molar, official in R8_POINTS:
        yield (
            f"constants/official/{t}/{molar}",
            dict(T=t, molar=molar, official=official),
            lambda t=t, m=molar, e=official: dielectric_constants_unit(
                t, m * 1000 * 0.018015268, official=e, molar=m
            ),
        )
    for t in (300, 500, 800):
        for rho in (40, 900):
            yield (
                f"constants/partials/{t}/{rho}",
                dict(T=t, rho=rho),
                lambda t=t, r=rho: dielectric_constants_unit(t, r, partials=True),
            )


def tde_precision_unit(t, wavelength, bs, cs):
    def reference(T, w):
        return mp.polyval([mp.mpf(str(b)) for b in bs], T - mp.mpf("298.15")) + mp.polyval(
            [mp.mpf(str(c)) for c in cs], w * 10**9 - mp.mpf("589.26")
        )

    return dict(
        partials=precise_partials(
            lambda T, w: refractive_index_tde(T, bs, cs, w), reference, (t, wavelength)
        )
    )


def tde_precision_units():
    for i, (bs, cs) in enumerate(
        (
            ([-0.000125041, 1.33245], [1.20771e-7, -3.56795e-5, 0]),
            ([1e-6, -0.001, 1.5], [2e-8, 0]),
            ([1.4], [0]),
        )
    ):
        for t in (280, 330):
            for w in (400e-9, 700e-9):
                yield (
                    f"TDE-precision/{i}/{t}/{w}",
                    dict(T=t, wavelength=w, Bs=bs, Cs=cs),
                    lambda t=t, w=w, bs=bs, cs=cs: tde_precision_unit(t, w, bs, cs),
                )


def official_ri_units():
    for wave, rows in R9_POINTS:
        for celsius, row in zip((0, 100, 200, 500), rows, strict=True):
            for pressure, expected in zip((0.1e6, 1e6, 10e6, 100e6), row, strict=True):
                yield (
                    f"R9-table3/{wave}/{celsius}/{pressure}",
                    dict(
                        T=celsius + 273.15, P=pressure, wavelength_m=wave * 1e-6, expected=expected
                    ),
                    (
                        lambda t=celsius + 273.15, p=pressure, w=wave * 1e-6, e=expected: (
                            official_ri_unit(t, p, w, e)
                        )
                    ),
                )


def lorentz_unit(index, volume):
    rm = molar_refractivity(index, volume)
    return {
        "Rm": compare(rm, refractivity.molar_refractivity_from_RI(index, volume)),
        "alpha_volume": compare(
            polarizability_volume(index, volume),
            refractivity.polarizability_from_RI(index, volume),
            atol=1e-40,
        ),
        "inverse": compare(
            refractive_index_from_molar_refractivity(rm, volume),
            refractivity.RI_from_molar_refractivity(rm, volume),
        ),
        "roundtrip": compare(refractive_index_from_molar_refractivity(rm, volume), index),
    }


def lorentz_units():
    for n in (0.8, 1, 1.3611, 2):
        for v in (0.00002, 0.0001):
            yield f"Lorentz/{n}/{v}", dict(n=n, Vm=v), lambda n=n, v=v: lorentz_unit(n, v)


def brix_unit(value):
    index = brix_to_refractive_index(value, extrapolate=not 0 <= value <= 95)
    return {
        "index": compare(index, refractivity.brix_to_RI(value)),
        "inverse": compare(
            refractive_index_to_brix(index, extrapolate=True),
            refractivity.RI_to_brix(index),
            atol=2e-10,
            rtol=0,
        ),
        "roundtrip": compare(
            refractive_index_to_brix(index, extrapolate=True), value, atol=2e-10, rtol=0
        ),
    }


def brix_units():
    for value in (*range(96), *(i + 0.5 for i in range(95)), -5, -0.1, 95.1, 100):
        yield f"Brix/{value}", dict(brix=value), lambda value=value: brix_unit(value)


@lru_cache(maxsize=1)
def identifier_database(directory):
    return IdentifierDatabase(directory / "Identifiers")


def native_card(database, record, *, component=None, **kwargs):
    # Use actual native identifier/formula records; no fabricated carrier formula.
    sources = ("example-user", "inorganic", "anion", "cation", "pubchem-small", "pubchem-large")
    if component is None:
        component = identifier_database(database.source_directory).component(
            record.cas_number, kind="cas", source_ids=sources, source_priority=sources
        )
    card = database.property_input(record.record_id, component, method=record.source_id, **kwargs)
    restored = PropertyInputCard.from_dict(json.loads(json.dumps(card.to_dict())))
    assert restored.to_dict() == card.to_dict()
    return restored


def source_evaluation(database, record, standalone, *, component=None, **kwargs):
    try:
        card = native_card(database, record, component=component, **kwargs)
    except UnknownIdentifierError as exc:
        if not standalone:
            raise
        if record.source_id == "CRC_PERMITTIVITY":
            curve = record.correlation(**kwargs)
        else:
            curve = RefractiveIndexObservation.from_record(record).correlation()
        curve = PropertyCorrelation.from_dict(json.loads(json.dumps(curve.to_dict())))

        def evaluate(property_id, *, temperature_K):
            assert curve.property_id == property_id
            return evaluate_correlation(curve, temperature_K=temperature_K)

        return curve, evaluate, dict(input_card=0, standalone_correlation=1, component_gap=str(exc))
    return (
        card.package.correlations[0],
        card.package.evaluate,
        dict(input_card=1, standalone_correlation=0),
    )


def optical_source_unit(database, record, raw, standalone=False, *, component=None):
    observation = RefractiveIndexObservation.from_record(record)
    restored = RefractiveIndexObservation.from_dict(json.loads(json.dumps(observation.to_dict())))
    assert restored == observation
    methods = database.methods(record.source_key, "RI", sources=REFRACTIVE_INDEX_SOURCES)
    # Wikidata stores an integer CAS; the reference public API needs its formatted string.
    cas = raw["CAS"]
    if cas.isdigit():
        cas = cas[:-3] + "-" + cas[-3:-1] + "-" + cas[-1]
    assert record.cas_number == cas
    assert {METHOD_MAP[m] for m in methods} == set(refractivity.RI_methods(cas))
    upstream = refractivity.RI(cas, method=METHOD_MAP[record.source_id])
    if raw["RI"] == "":
        assert observation.index.value is None and upstream == (None, None)
        try:
            restored.evaluate()
        except ValueError:
            return dict(comparisons=[], missing=1, unconsumed=0)
        raise AssertionError("fabricated missing RI")
    report = restored.evaluate()
    checks = [compare(report.value, float(raw["RI"])), compare(report.value, upstream[0])]
    integration = {}
    if raw["RIT"]:
        assert observation.temperature.value == float(raw["RIT"]) == upstream[1]
        _, evaluate, integration = source_evaluation(
            database, record, standalone, source_point=True, component=component
        )
        checks.append(
            compare(
                evaluate("refractive_index", temperature_K=float(raw["RIT"])).value,
                report.value,
            )
        )
    else:
        assert observation.temperature.value is None and upstream[1] is None and report.warnings
    checks.extend(lorentz_unit(report.value, 0.0001).values())
    if refractivity.brix_to_RI(0) <= report.value <= refractivity.brix_to_RI(95):
        checks.append(
            compare(
                brix_reading(report).value,
                refractivity.RI_to_brix(report.value),
                atol=2e-10,
                rtol=0,
            )
        )
    else:
        try:
            brix_reading(report)
        except ValueError:
            pass
        else:
            raise AssertionError("RI outside Brix range silently extrapolated")
    return dict(comparisons=checks, missing=0, unconsumed=0, **integration)


def dipole_source_unit(database, record, raw):
    value = record.value("dipole_moment")
    methods = database.methods(record.source_key, "dipole_moment", sources=DIPOLE_SOURCES)
    assert {METHOD_MAP[m] for m in methods} == set(dipole.dipole_moment_methods(record.source_key))
    upstream = dipole.dipole_moment(record.source_key, method=METHOD_MAP[record.source_id])
    if not raw["dipole_moment"]:
        assert value.value is None and upstream is None
        return dict(comparisons=[], missing=1, unconsumed=0)
    source = value.input_value("C*m").to("Debye")
    v = source.value
    checks = [
        compare(v, float(raw["dipole_moment"])),
        compare(v, upstream),
        compare(
            viscosity_value("Lucas_gas", 400, Tc=500, Pc=5e6, Zc=0.25, MW=80, dipole=v),
            viscosity.Lucas_gas(400, 500, 5e6, 0.25, 80, upstream),
        ),
    ]
    return dict(comparisons=checks, missing=0, unconsumed=0)


def dielectric_source_unit(database, record, raw, standalone=False, *, component=None):
    def numeric(k):
        return None if raw[k] == "" else float(raw[k])

    checks = []
    point = 0
    fit = 0
    rejected = 0
    integration = {}
    t, epsilon = numeric("T"), numeric("Permittivity")
    if t is not None and epsilon is not None and t > 0 and epsilon > 0:
        _, evaluate, integration = source_evaluation(
            database, record, standalone, source_point=True, component=component
        )
        report = evaluate("relative_permittivity", temperature_K=t)
        checks.append(compare(report.value, epsilon))
        point = 1
        cap = parallel_plate_capacitance(report, area_m2=0.01, separation_m=0.001)
        checks.append(compare(cap.value, (1 / (4e-7 * math.pi * 299792458**2)) * epsilon * 10))
    a = numeric("A")
    bounds = numeric("Tmin"), numeric("Tmax")
    coeff = [numeric(k) for k in "ABCD"]
    interior = any(coeff[i] is None and any(v is not None for v in coeff[i + 1 :]) for i in (1, 2))
    buildable = (
        a is not None
        and not interior
        and all(t is not None for t in bounds)
        and 0 < bounds[0] < bounds[1]
    )
    if buildable:
        curve, evaluate, integration = source_evaluation(
            database, record, standalone, component=component
        )
        p = [0 if v is None else v for v in coeff]
        for t in (bounds[0], sum(bounds) / 2, bounds[1]):
            expected = permittivity.permittivity_CRC(t, *p)
            if expected <= 0:
                try:
                    evaluate_correlation(curve, temperature_K=t)
                except ValueError:
                    rejected += 1
                    continue
                raise AssertionError("nonpositive dielectric prediction accepted")
            result = evaluate("relative_permittivity", temperature_K=t)
            checks.append(compare(result.value, expected))
            slope = correlation_temperature_derivative(curve, temperature_K=t).value
            checks.extend(differences(slope, lambda x: permittivity.permittivity_CRC(x, *p), t))
            checks.append(
                compare(
                    parallel_plate_capacitance(result, area_m2=0.01, separation_m=0.001).value,
                    (1 / (4e-7 * math.pi * 299792458**2)) * expected * 10,
                )
            )
        fit = 1
    else:
        try:
            record.correlation()
        except ValueError:
            pass
        else:
            raise AssertionError("unavailable source fit silently constructed")
    return dict(
        comparisons=checks,
        point=point,
        fit=fit,
        rejected=rejected,
        unconsumed=int(not (point or fit)),
        **integration,
    )


def source_unit(database, source, index, raw, standalone=False):
    record = database.records(source)[index]
    assert record.source_key == raw["CAS"]
    assert all(record.raw(k) == raw[k] for k in record.field_names)
    if source in REFRACTIVE_INDEX_SOURCES:
        return optical_source_unit(database, record, raw, standalone)
    if source in DIPOLE_SOURCES:
        return dipole_source_unit(database, record, raw)
    return dielectric_source_unit(database, record, raw, standalone)


def source_units(directory, standalone=False):
    database = ElectroOpticalDatabase(directory)
    for source in database.catalog:
        with (directory / source.relative_path).open(encoding="utf8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) == COUNTS[source.source_id] == len(database.records(source.source_id))
        for i, raw in enumerate(rows):
            yield (
                f"{source.source_id}/{i}",
                dict(source=source.source_id, row=i, raw=raw),
                (lambda s=source.source_id, i=i, r=raw: source_unit(database, s, i, r, standalone)),
            )


def main():
    parser = argparse.ArgumentParser()
    counts = {
        "water": 88,
        "official-ri": 48,
        "lorentz": 8,
        "brix": 195,
        "sources": 17268,
        "dielectric-constants": 32,
        "tde-precision": 12,
        "sources-consumers": 17268,
    }
    parser.add_argument("--block", choices=counts, required=True)
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert chemicals.__version__ == "1.5.2"
    units = list(
        {
            "water": water_units,
            "official-ri": official_ri_units,
            "lorentz": lorentz_units,
            "brix": brix_units,
            "dielectric-constants": dielectric_constants_units,
            "tde-precision": tde_precision_units,
            "sources": lambda: source_units(args.source_directory),
            "sources-consumers": lambda: source_units(args.source_directory, standalone=True),
        }[args.block]()
    )
    assert len(units) == counts[args.block]
    print(f"{args.block}: fixed denominator {len(units)}", flush=True)
    start, rows = time.perf_counter(), []
    for identity, inputs, run in units:
        row = {"unit": identity, "inputs": inputs}
        try:
            row.update(metrics=run(), status="passed")
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        if len(rows) % 200 == 0 or len(rows) == len(units):
            elapsed = time.perf_counter() - start
            print(
                f"{len(rows)}/{len(units)}; {len(rows) / elapsed:.1f}/s; "
                f"ETA {(len(units) - len(rows)) * elapsed / len(rows):.1f}s",
                flush=True,
            )
    summary = {
        "mode": "development",
        "block": args.block,
        "reference_version": chemicals.__version__,
        "total": len(rows),
        "passed": sum(r["status"] == "passed" for r in rows),
        "failed": sum(r["status"] == "failed" for r in rows),
        "seconds": time.perf_counter() - start,
        "results": rows,
    }
    if args.block in {"sources", "sources-consumers"}:
        summary["coverage"] = {
            source: {
                key: sum(
                    r.get("metrics", {}).get(key, 0)
                    for r in rows
                    if r["inputs"]["source"] == source
                )
                for key in (
                    "point",
                    "fit",
                    "missing",
                    "rejected",
                    "unconsumed",
                    "input_card",
                    "standalone_correlation",
                )
            }
            for source in COUNTS
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    for r in [r for r in rows if r["status"] == "failed"][:3]:
        print(r["unit"], r["failure"])
    return bool(summary["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
