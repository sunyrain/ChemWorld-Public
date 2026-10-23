"""Fixed thermal-conductivity blocks; only tests import the reference engine."""
# ruff: noqa: C408 -- compact fixed parameters

import argparse
import csv
import json
import math
import time
import traceback
from itertools import pairwise
from pathlib import Path

import chemicals
from chemicals import dippr
from chemicals import thermal_conductivity as reference
from chemicals.iapws import iapws95_P, iapws95_properties
from pressure_reference import finite_difference

from chemworld.data.conductivity import ConductivityDatabase
from chemworld.physchem.air_transport import air_conductivity_lemmon
from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.conductivity_kernels import conductivity_derivatives
from chemworld.physchem.conductivity_mixtures import (
    dippr9h_conductivity,
    dippr9i_conductivity,
    filippov_conductivity,
    lindsay_bromley_conductivity,
    wassiljewa_herning_zipperer_conductivity,
)
from chemworld.physchem.dense_conductivity import (
    compressed_liquid_conductivity_derivatives,
)
from chemworld.physchem.property_equations import (
    correlation_temperature_derivative,
    evaluate_correlation,
)
from chemworld.physchem.water_transport import water_conductivity_iapws, water_viscosity_iapws

METHODS = (
    ("Sheffy_Johnson", (280, 300, 450), dict(MW=47, Tm=280)),
    ("Sato_Riedel", (300, 400, 500), dict(MW=47, Tb=390, Tc=520)),
    ("Lakshmi_Prasad", (200, 250, 273.15), dict(MW=100)),
    ("Bahadori_liquid", (273.15, 320, 400), dict(MW=170)),
    ("Gharagheizi_liquid", (250, 300, 400), dict(MW=40, Tb=350, Pc=1e6, omega=0.27)),
    ("Nicola", (300, 400, 500), dict(MW=142.3, Tc=611.7, Pc=2110000, omega=0.49)),
    ("Nicola_original", (300, 400, 500), dict(MW=142.3, Tc=611.7, omega=0.49, Hfus=201853)),
    ("kl_Mersmann_Kind", (300, 400, 600), dict(MW=170.33484, Tc=658, Vc=0.000754, na=38)),
    ("PPDS3", (300, 400, 550), dict(Tc=470.008, a1=11.6366, a2=25.1191, a3=-7.21674)),
    ("PPDS8", (300, 400, 500), dict(Tc=562.05, a0=0.0641126, a1=0.61057, a2=-1.72442, a3=3.94394)),
    (
        "Chemsep_16",
        (250, 300, 400),
        dict(A=-0.12682, B=-1.5015, C=-1.0467, D=-0.00088709, E=-9.3679e-7),
    ),
    ("Eucken", (300, 373.15, 500), dict(MW=72.151, Cvm=135.9, mu=8.77e-6)),
    ("Eucken_modified", (300, 373.15, 500), dict(MW=72.151, Cvm=135.9, mu=8.77e-6)),
    ("DIPPR9B_monoatomic", (200, 300, 400), dict(MW=28.01, Cvm=20.826, mu=1.277e-5)),
    ("DIPPR9B_linear", (200, 300, 400), dict(MW=28.01, Cvm=20.826, mu=1.277e-5, Tc=132.92)),
    ("DIPPR9B_nonlinear", (200, 300, 400), dict(MW=28.01, Cvm=20.826, mu=1.277e-5)),
    ("Chung", (300, 373.15, 500), dict(MW=72.151, Tc=460.4, omega=0.227, Cvm=135.9, mu=8.77e-6)),
    ("Bahadori_gas", (250, 313.15, 450), dict(MW=20)),
)
SOURCES = (
    (
        "PERRY_GAS_CONDUCTIVITY",
        "Table 2-314 Vapor Thermal Conductivity of Inorganic and Organic Substances.tsv",
        345,
    ),
    (
        "PERRY_LIQUID_CONDUCTIVITY",
        "Table 2-315 Thermal Conductivity of Inorganic and Organic Liquids.tsv",
        340,
    ),
    ("VDI_GAS_CONDUCTIVITY", "VDI PPDS Thermal conductivity of gases.tsv", 275),
    ("VDI_LIQUID_CONDUCTIVITY", "VDI PPDS Thermal conductivity of saturated liquids.tsv", 271),
)


def compare(actual, expected, *, rtol=3e-10, atol=1e-12):
    assert abs(actual - expected) <= atol + rtol * abs(expected), (actual, expected)
    return {"actual": actual, "reference": expected, "error": abs(actual - expected)}


def slope(actual, function, t):
    values = finite_difference(function, t, 1)
    compare(*values, atol=1e-11, rtol=3e-6)
    return [compare(actual, v, atol=1e-11, rtol=3e-6) for v in values]


def method_units():
    for method, temperatures, p in METHODS:
        for t in temperatures:

            def run(method=method, t=t, p=p):
                def ref(x):
                    if method.startswith("DIPPR9B_"):
                        return reference.DIPPR9B(x, **p, chemtype=method.removeprefix("DIPPR9B_"))
                    if method.startswith("Eucken"):
                        return getattr(reference, method)(**p)
                    return getattr(reference, method)(x, **p)

                value, derivative = conductivity_derivatives(method, t, p)
                return {"value": compare(value, ref(t)), "derivatives": slope(derivative, ref, t)}

            yield f"{method}/{t}", {"T": t, **p}, run


def source_unit(database, source, index, raw):
    record = database.records(source)[index]
    assert record.source_key == raw["CAS"]
    assert record in database.records_for(raw["CAS"], method=source)
    for key in record.field_names:
        assert record.raw(key) == raw[key]
    bounds = (
        (float(raw["Tmin"]), float(raw["Tmax"]))
        if source.startswith("PERRY")
        else (300, 800)
        if source == "VDI_GAS_CONDUCTIVITY"
        else (250, 450)
    )
    if not 0 < bounds[0] < bounds[1]:
        try:
            record.correlation(temperature_bounds_K=bounds)
        except ValueError as exc:
            return {"unbuilt": 1, "rejection": str(exc), "bounds": bounds}
        raise AssertionError("native accepted invalid interval")
    curve = record.correlation(temperature_bounds_K=bounds)
    n = 4 if source == "PERRY_GAS_CONDUCTIVITY" else 5
    p = (
        [float(raw[f"C{i}"]) for i in range(1, n + 1)]
        if source.startswith("PERRY")
        else [float(raw[k]) for k in "ABCDE"]
    )
    equation = dippr.EQ102 if n == 4 else dippr.EQ100
    function = lambda t: equation(t, *p)  # noqa: E731
    metrics = {
        "bounds": bounds,
        "points": [],
        "value_comparisons": 0,
        "derivative_comparisons": 0,
        "rejected": 0,
    }
    for fraction in (0.25, 0.5, 0.75):
        t = bounds[0] + fraction * (bounds[1] - bounds[0])
        point = {"T": t}
        try:
            expected = function(t)
            valid = math.isfinite(expected) and expected > 0
            point["reference"] = expected if math.isfinite(expected) else str(expected)
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            valid = False
            point["reference_exception"] = repr(exc)
        if valid:
            actual = evaluate_correlation(curve, temperature_K=t, validity_policy="raise").value
            point["value"] = compare(actual, expected)
            point["derivatives"] = slope(
                correlation_temperature_derivative(curve, temperature_K=t).value, function, t
            )
            metrics["value_comparisons"] += 1
            metrics["derivative_comparisons"] += 2
        else:
            try:
                evaluate_correlation(curve, temperature_K=t, validity_policy="raise")
            except ValueError as exc:
                point["rejection"] = str(exc)
                metrics["rejected"] += 1
            else:
                raise AssertionError("native accepted nonpositive/nonfinite reference prediction")
        metrics["points"].append(point)
    return metrics


def source_units(root):
    database = ConductivityDatabase(root)
    for source, filename, count in SOURCES:
        with (root / "Thermal Conductivity" / filename).open(encoding="utf8", newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        assert len(rows) == len(database.records(source)) == count
        for i, raw in enumerate(rows):
            yield f"{source}/{i}", raw, lambda s=source, i=i, r=raw: source_unit(database, s, i, r)


FRACTIONS = ((1,), (1, 0), (0.258, 0.742), (0, 1), (0.2, 0, 0.8), (0.05, 0.15, 0.25, 0.2, 0.35))


def mixing_unit(method, t, vectors):
    native = {
        "DIPPR9H": dippr9h_conductivity,
        "DIPPR9I": dippr9i_conductivity,
        "Filippov": filippov_conductivity,
        "Lindsay_Bromley": lindsay_bromley_conductivity,
        "Wassiljewa_Herning_Zipperer": wassiljewa_herning_zipperer_conductivity,
    }[method]
    ref = getattr(reference, method)
    values = []
    for args in (vectors, tuple(tuple(reversed(v)) for v in vectors)):
        values.append(
            compare(native(t, *args), ref(t, *args))
            if method == "Lindsay_Bromley"
            else compare(native(*args), ref(*args))
        )
    metrics = {"values": values}
    if method == "Lindsay_Bromley":
        derivative = derivative_values(lambda temp: native(temp, *vectors), t)[1]
        metrics["derivatives"] = slope(derivative, lambda temp: ref(temp, *vectors), t)
    return metrics


def mixture_units():
    for method in ("DIPPR9H", "DIPPR9I", "Wassiljewa_Herning_Zipperer", "Lindsay_Bromley"):
        for index, fractions in enumerate(FRACTIONS):
            n = len(fractions)
            liquid = method in {"DIPPR9H", "DIPPR9I"}
            ks = tuple(0.1 + 0.03 * i if liquid else 0.01 + 0.007 * i for i in range(n))
            mw = tuple(20 + 30 * i for i in range(n))
            vectors = (
                (fractions, ks)
                if method == "DIPPR9H"
                else (fractions, tuple((60 + 20 * i) * 1e-6 for i in range(n)), ks)
                if method == "DIPPR9I"
                else (fractions, ks, mw)
                if method == "Wassiljewa_Herning_Zipperer"
                else (
                    fractions,
                    ks,
                    tuple((9 + i) * 1e-6 for i in range(n)),
                    tuple(180 + 35 * i for i in range(n)),
                    mw,
                )
            )
            for t in (250, 323.15, 600) if method == "Lindsay_Bromley" else (300,):
                yield (
                    f"{method}/{index}/{t}",
                    {"T": t, "vectors": vectors},
                    lambda m=method, t=t, v=vectors: mixing_unit(m, t, v),
                )
    for index, fractions in enumerate(((1, 0), (0, 1), (0.258, 0.742), (0.5, 0.5))):
        vectors = (fractions, (0.1692, 0.1528))
        yield (
            f"Filippov/{index}",
            {"vectors": vectors},
            lambda v=vectors: mixing_unit("Filippov", 300, v),
        )


def pressure_unit(method, t, p, tc, pc, kl, policy, derivatives):
    parameters = dict(
        method=method,
        critical_temperature_K=tc,
        critical_pressure_Pa=pc,
        reference_conductivity_W_m_K=kl,
        outside_table=policy,
    )
    ref = lambda t, p: getattr(reference, method)(t, p, tc, pc, kl)  # noqa: E731
    value = compressed_liquid_conductivity_derivatives(t, p, order=0, **parameters)[0]
    metrics = {"value": compare(value, ref(t, p))}
    if derivatives:
        metrics["temperature_derivatives"] = slope(
            compressed_liquid_conductivity_derivatives(t, p, **parameters)[1],
            lambda t: ref(t, p),
            t,
        )
        metrics["pressure_derivatives"] = slope(
            compressed_liquid_conductivity_derivatives(t, p, variable="pressure", **parameters)[1],
            lambda p: ref(t, p),
            p,
        )
    return metrics


def pressure_units():
    for t in (300, 400, 515.05):
        for p in (101325, 3.212e6, 3.92e7, 1e8):
            args = ("DIPPR9G", t, p, 579.15, 3.212e6, 0.07085, "reject", True)
            yield f"DIPPR9G/{t}/{p}", {"arguments": args}, lambda a=args: pressure_unit(*a)
    trs, prs = (0.5, 0.6, 0.7, 0.8), (1, 5, 10, 50, 100, 200)
    points = [(tr, pr, "reject", False) for tr in trs for pr in prs]
    points += [
        ((ta + tb) / 2, (pa + pb) / 2, "reject", True)
        for ta, tb in pairwise(trs)
        for pa, pb in pairwise(prs)
    ]
    points += [(304 / 591.8, 633e6 / 4.1e6, "reject", True)]
    points += [
        (tr, pr, "nearest", pr != 10)
        for tr, pr in (
            (0.4, 10),
            (0.9, 10),
            (0.65, 0.5),
            (0.65, 250),
            (0.4, 0.5),
            (0.9, 250),
            (0.4, 30),
        )
    ]
    for index, (tr, pr, policy, derivatives) in enumerate(points):
        args = ("Missenard", tr * 591.8, pr * 4.1e6, 591.8, 4.1e6, 0.129, policy, derivatives)
        yield f"Missenard/{index}", {"arguments": args}, lambda a=args: pressure_unit(*a)


def fluid_unit(fluid, t, rho, parameters, *, background=False, official=None, eos=False):
    p = dict(parameters)
    if eos:
        properties = iapws95_properties(t, iapws95_P(t, rho))
        reference_properties = iapws95_properties(970.644, iapws95_P(970.644, rho))
        p.update(
            Cp=properties[5],
            Cv=properties[4],
            drho_dP=properties[10],
            drho_dP_reference=reference_properties[10],
        )
        p["mu"] = water_viscosity_iapws(
            t, rho, drho_dP=p["drho_dP"], drho_dP_reference=p["drho_dP_reference"]
        )
    method = "k_IAPWS" if fluid == "water" else "k_air_lemmon"
    native = water_conductivity_iapws if fluid == "water" else air_conductivity_lemmon
    rp = {
        "drho_dP_Tr" if key == "drho_dP_reference" else key: v
        for key, v in p.items()
        if key != "reference_approximation"
    }
    expected = getattr(reference, method)(t, rho, **rp)
    value = native(t, rho, **p)
    metrics = {"state_inputs": p, "value": compare(value, expected)}
    if background:
        metrics["derivatives"] = slope(
            conductivity_derivatives(method, t, {"rho": rho})[1],
            lambda temp: getattr(reference, method)(temp, rho),
            t,
        )
    if official is not None:
        metrics["official"] = compare(value, official, atol=5e-9, rtol=0)
    return metrics


def water_air_units():
    for fluid, ts, rhos in (
        ("water", (298.15, 647.35, 873.15), (0, 1, 322, 750, 998)),
        ("air", (100, 132.64, 300, 1000), (0, 40, 10400, 20000)),
    ):
        for t in ts:
            for rho in rhos:
                yield (
                    f"{fluid}-background/{t}/{rho}",
                    {"T": t, "rho": rho},
                    lambda f=fluid, t=t, r=rho: fluid_unit(f, t, r, {}, background=True),
                )
    for t, rho, value in (
        (298.15, 0, 0.0184341883),
        (298.15, 998, 0.607712868),
        (298.15, 1200, 0.799038144),
        (873.15, 0, 0.0791034659),
    ):
        yield (
            f"water-official-background/{t}/{rho}",
            {"T": t, "rho": rho, "official": value},
            lambda t=t, r=rho, v=value: fluid_unit("water", t, r, {}, official=v),
        )
    for rho, value in zip(
        (1, 122, 222, 272, 322, 372, 422, 750),
        (
            0.0519298924,
            0.130922885,
            0.367787459,
            0.757959776,
            1.44375556,
            0.650319402,
            0.448883487,
            0.600961346,
        ),
        strict=True,
    ):
        yield (
            f"water-official-critical/{rho}",
            {"T": 647.35, "rho": rho, "official": value},
            lambda r=rho, v=value: fluid_unit("water", 647.35, r, {}, official=v, eos=True),
        )
    p = dict(
        Cp=7634.337046792,
        Cv=3037.934412104,
        mu=70.905106751524e-6,
        drho_dP=5.209378197916e-6,
        reference_approximation=True,
    )
    yield (
        "water-approximation/example",
        {"T": 620, "rho": 613.227777440324, **p},
        lambda p=p: fluid_unit("water", 620, 613.227777440324, p),
    )
    for rho in (50, 150, 300, 500, 700):
        p = dict(Cp=10000, Cv=3000, mu=4e-5, drho_dP=1e-4, reference_approximation=True)
        yield (
            f"water-approximation/{rho}",
            {"T": 650, "rho": rho, **p},
            lambda r=rho, p=p: fluid_unit("water", 650, r, p),
        )
    for xi in (0, 4e-9, 4.752e-8, 4.848e-8, 4e-7, 1, 100):
        drho = 0.06 * (xi / 0.13) ** (1.239 / 0.63) * 322 / 22.064e6
        p = dict(Cp=10000, Cv=3000, mu=4e-5, drho_dP=drho, drho_dP_reference=0)
        yield (
            f"water-branch/{xi}",
            {"T": 647.35, "rho": 322, "xi_nm": xi, **p},
            lambda p=p: fluid_unit("water", 647.35, 322, p),
        )
    p = dict(Cp=10000, Cv=3000, mu=4e-5, drho_dP=0, drho_dP_reference=1e-6)
    yield (
        "water-branch/negative",
        {"T": 647.35, "rho": 322, **p},
        lambda p=p: fluid_unit("water", 647.35, 322, p),
    )
    p = dict(
        Cp=2137.078854678728,
        Cv=35.24316159996235,
        drho_dP=0.07417878614315769,
        drho_dP_reference=0.00035919027241528256,
        mu=1.7762253265868595e-5,
    )
    yield (
        "air-critical/example",
        {"T": 132.64, "rho": 10400, **p},
        lambda p=p: fluid_unit("air", 132.64, 10400, p),
    )
    for excess in (0, 1e-18, 1e-10, 0.01, 1, 100):
        p = dict(
            Cp=2000,
            Cv=35,
            mu=1.8e-5,
            drho_dP=excess * 10447.7**2 / (3.78502e6 * 10400),
            drho_dP_reference=0,
        )
        yield (
            f"air-critical/{excess}",
            {"T": 132.64, "rho": 10400, **p},
            lambda p=p: fluid_unit("air", 132.64, 10400, p),
        )
    p = dict(Cp=2000, Cv=35, mu=1.8e-5, drho_dP=0, drho_dP_reference=1e-6)
    yield (
        "air-critical/negative",
        {"T": 132.64, "rho": 10400, **p},
        lambda p=p: fluid_unit("air", 132.64, 10400, p),
    )


def high_precision_air_unit(inputs):
    import mpmath as mp

    p = {k: v for k, v in inputs.items() if k not in {"T", "rho"}}
    with mp.workdps(80):
        m = lambda x: mp.mpf(str(x))  # noqa: E731
        t, rho = m(inputs["T"]), m(inputs["rho"])
        tau, delta = m(132.6312) / t, rho / m(10447.7)
        collision = mp.exp(
            sum(
                m(b) * mp.log(t / m(103.3)) ** i
                for i, b in enumerate((0.431, -0.4623, 0.08406, 0.005341, -0.00331))
            )
        )
        eta = m(0.0266958) * mp.sqrt(m(28.9586) * t) / (m(0.36) ** 2 * collision)
        background = m(0.001) * (
            m(1.308) * eta
            + m(1.405) * tau ** m(-1.1)
            - m(1.036) * tau ** m(-0.3)
            + sum(
                m(n) * tau ** m(power) * delta**d * mp.exp(-g * delta**ell)
                for n, power, d, g, ell in (
                    (8.743, 0.1, 1, 0, 0),
                    (14.76, 0, 2, 0, 0),
                    (-16.62, 0.5, 3, 1, 2),
                    (3.793, 2.7, 7, 1, 2),
                    (-6.142, 0.3, 7, 1, 2),
                    (-0.3778, 1.3, 11, 1, 2),
                )
            )
        )
        excess = (
            m(3785020)
            * rho
            / m(10447.7) ** 2
            * (m(p["drho_dP"]) - m(p["drho_dP_reference"]) * m(265.262) / t)
        )
        enhancement = mp.mpf(0)
        if excess > 0:
            xi = m(0.11e-9) * (excess / m(0.055)) ** (m(0.63) / m(1.2415))
            y = xi / m(0.31e-9)
            cp, cv, mu = m(p["Cp"]), m(p["Cv"]), m(p["mu"])
            omega = 2 / mp.pi * ((1 - cv / cp) * mp.atan(y) + cv / cp * y)
            omega0 = 2 / mp.pi * (1 - mp.exp(-1 / (1 / y + y * y / (3 * delta * delta))))
            enhancement = (
                rho * cp * m(1.380658e-23) * m(1.01) * t / (6 * mp.pi * xi * mu) * (omega - omega0)
            )
        expected = float(background + enhancement)
        metrics = {
            "high_precision": mp.nstr(background + enhancement, 80),
            "enhancement": mp.nstr(enhancement, 80),
        }
    actual = air_conductivity_lemmon(inputs["T"], inputs["rho"], **p)
    metrics["value"] = compare(actual, expected)
    rp = {"drho_dP_Tr" if k == "drho_dP_reference" else k: v for k, v in p.items()}
    try:
        metrics["reference_library"] = reference.k_air_lemmon(inputs["T"], inputs["rho"], **rp)
    except Exception as exc:
        metrics["reference_exception"] = repr(exc)
    return metrics


def high_precision_air_units():
    for identity, inputs, _ in water_air_units():
        if identity.startswith("air-critical/"):
            yield identity, inputs, lambda p=inputs: high_precision_air_unit(p)


def gas_shape_unit(method, t, parameters):
    order = 0 if method == "Eli_Hanley" and t == 920.8 else 1
    values = conductivity_derivatives(method, t, parameters, order=order)
    ref = lambda temp: getattr(reference, method)(temp, **parameters)  # noqa: E731
    metrics = {"value": compare(values[0], ref(t))}
    if order:
        metrics["derivatives"] = slope(values[1], ref, t)
    return metrics


def gas_shape_units():
    p = dict(MW=72.151, Tc=460.4, Vc=3.06e-4, Zc=0.267, omega=0.227, Cvm=135.9)
    for t in (230.2, 373.15, 690.6, 920.8, 1151):
        yield f"Eli_Hanley/{t}", {"T": t, **p}, lambda t=t, p=p: gas_shape_unit("Eli_Hanley", t, p)
    for t in (300, 473, 700):
        for factor in (0.9, 2, 50):
            for dipole, association in ((0, 0), (0.4, 0), (1.2, 0.1)):
                p = dict(
                    MW=42.081,
                    Tc=364.9,
                    Vc=184.6e-6,
                    omega=0.142,
                    Cvm=82.67,
                    Vm=factor * 184.6e-6,
                    mu=1.34e-5,
                    dipole=dipole,
                    association=association,
                )
                yield (
                    f"Chung_dense/{t}/{factor}/{dipole}/{association}",
                    {"T": t, **p},
                    lambda t=t, p=p: gas_shape_unit("Chung_dense", t, p),
                )


def gas_completion_unit(method, t, parameters):
    order = 0 if method == "Eli_Hanley_dense" and t / parameters["Tc"] == 2 else 1
    values = conductivity_derivatives(method, t, parameters, order=order)
    ref = lambda temp: getattr(reference, method)(temp, **parameters)  # noqa: E731
    expected = ref(t)
    metrics = {"value": compare(values[0], expected)}
    if order:
        metrics["derivatives"] = slope(values[1], ref, t)
    return metrics


def gas_completion_units():
    cases = []
    for mw, tb, pc, omega in (
        (16.04246, 111.66, 4599000, 0.0115478),
        (44.09562, 231.04, 4251200, 0.1521),
        (72.151, 309.21, 3370000, 0.251),
    ):
        for t in (400, 580, 800, 1200):
            cases.append(("Gharagheizi_gas", t, dict(MW=mw, Tb=tb, Pc=pc, omega=omega)))
    for mw, tc, pc, vc, zc, kg in (
        (44.013, 309.6, 7240000, 97.4e-6, 0.274, 0.0234),
        (16.04246, 190.564, 4599000, 98.6e-6, 0.286, 0.034),
    ):
        for t in (250, 378.15, 600):
            for rho in (1e-10, 0.1, 0.499999, 0.5, 0.500001, 1.2, 1.999999, 2, 2.000001, 2.79):
                cases.append(
                    (
                        "Stiel_Thodos_dense",
                        t,
                        dict(MW=mw, Tc=tc, Pc=pc, Vc=vc, Zc=zc, kg=kg, Vm=vc / rho),
                    )
                )
    for mw, tc, vc, zc, omega, cv in (
        (42.081, 364.9, 1.81e-4, 0.274, 0.144, 82.70),
        (72.151, 460.4, 3.06e-4, 0.267, 0.227, 135.9),
    ):
        for tr in (0.6, 1, 1.8, 2, 2.4):
            for vr in (0.5, 0.95, 1.999999, 2, 10):
                cases.append(
                    (
                        "Eli_Hanley_dense",
                        tr * tc,
                        dict(MW=mw, Tc=tc, Vc=vc, Zc=zc, omega=omega, Cvm=cv, Vm=vr * vc),
                    )
                )
    cases.append(
        (
            "Eli_Hanley_dense",
            473,
            dict(MW=42.081, Tc=364.9, Vc=1.81e-4, Zc=0.274, omega=0.144, Cvm=82.70, Vm=1.721e-4),
        )
    )
    for i, (method, t, p) in enumerate(cases):
        yield (
            f"{method}/{i}",
            {"T": t, **p},
            lambda m=method, t=t, p=p: gas_completion_unit(m, t, p),
        )


def ely_translational_equation(temperature, parameters):
    """Independent 80-digit equation evaluation; no native arithmetic reused."""
    import mpmath as mp

    p = {key: mp.mpf(str(value)) for key, value in parameters.items()}
    c = lambda x: mp.mpf(str(x))  # noqa: E731
    t = mp.mpf(temperature)
    tau = min(t / p["Tc"], c(2))
    volume = min(p["Vm"] / p["Vc"], c(2))
    w = p["omega"] - c(0.011)
    z = c(0.288) / p["Zc"]
    theta = 1 + w * (
        c(0.09057) - c(0.86276) * mp.log(tau) + (c(0.31664) - c(0.46568) / tau) * (volume - c(0.5))
    )
    psi = z * (
        1
        + w * (c(0.3949) * (volume - c(1.02355)) - c(0.93281) * (volume - c(0.75464)) * mp.log(tau))
    )
    f, h = p["Tc"] * theta / c(190.4), p["Vc"] * psi / c(0.0000992)
    t0, rho = t / f, c(16.04) * h / p["Vm"] / 10**6
    coeffs = (
        "2907741.307",
        "-3312874.033",
        "1608101.838",
        "-433190.4871",
        "70624.8133",
        "-7116.62075",
        "432.51744",
        "-14.4591121",
        "0.2037119479",
    )

    def eta(temperature):
        return (
            mp.fsum(mp.mpf(a) * temperature ** (mp.mpf(i - 3) / 3) for i, a in enumerate(coeffs))
            / 10**7
        )

    dtheta = w * (-c(0.86276) / t + (volume - c(0.5)) * c(0.46568) * p["Tc"] / t**2)
    if t / p["Tc"] > 2:
        dtheta = 0
    x = ((1 - t * dtheta / theta) * z) ** c(1.5)
    h_factor = mp.sqrt(c(16.04) * f / p["MW"]) / h ** (mp.mpf(2) / 3)
    k0 = 1944 * eta(t0)
    k1 = (c(-0.2527692) + c(0.33432859) * (c(1.12) - mp.log(t0 / 168)) ** 2) * rho / 1000
    q = (c(12.47183) - c(984.6252) / t0 ** c(1.5)) * rho ** c(0.1)
    q += (
        (rho / c(0.1617) - 1)
        * mp.sqrt(rho)
        * (c(0.3594685) + c(69.79841) / t0 - c(872.8833) / t0**2)
    )
    k2 = mp.exp(c(-7.19771) + c(85.67822) / t0) * mp.expm1(q) / 1000
    fd = p["Tc"] / c(190.4) * (1 + w * (c(0.56553) - c(0.86276) * mp.log(tau) - c(0.69852) / tau))
    hd = p["Vc"] / c(0.0000992) * z * (1 + w * (c(0.3856) - c(1.1617) * mp.log(tau)))
    internal = eta(t / fd) * mp.sqrt(fd * p["MW"] / c(16.04)) / hd ** (mp.mpf(2) / 3)
    internal *= 1000 / p["MW"] * c(1.32) * (p["Cvm"] - c(1.5) * c(8.31446261815324))
    return (k0 + k1 + k2) * h_factor * x + internal


def ely_high_precision_unit(inputs):
    import mpmath as mp

    t = inputs["T"]
    p = {key: value for key, value in inputs.items() if key != "T"}
    order = 0 if t / p["Tc"] == 2 else 1
    values = conductivity_derivatives("Ely_Hanley_dense_translational", t, p, order=order)
    with mp.workdps(80):
        temperature = mp.mpf(str(t))
        expected = float(ely_translational_equation(temperature, p))
        metrics = {
            "value": compare(values[0], expected),
            "reference_compatibility_value": reference.Eli_Hanley_dense(t, **p),
        }
        if order:
            derivative = float(
                mp.diff(lambda temp: ely_translational_equation(temp, p), temperature)
            )
            metrics["derivative"] = compare(values[1], derivative, rtol=3e-6, atol=1e-11)
    return metrics


def ely_high_precision_units():
    for identity, inputs, _ in gas_completion_units():
        if identity.startswith("Eli_Hanley_dense/"):
            yield identity, inputs, lambda p=inputs: ely_high_precision_unit(p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block",
        choices=(
            "methods-core",
            "sources",
            "mixtures",
            "liquid-pressure",
            "water-air",
            "air-critical-high-precision",
            "gas-shapes",
            "gas-completion",
            "ely-translational-high-precision",
        ),
        required=True,
    )
    parser.add_argument("--source-directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    assert chemicals.__version__ == "1.5.2"
    units = list(
        {
            "methods-core": method_units,
            "sources": lambda: source_units(args.source_directory),
            "mixtures": mixture_units,
            "liquid-pressure": pressure_units,
            "water-air": water_air_units,
            "air-critical-high-precision": high_precision_air_units,
            "gas-shapes": gas_shape_units,
            "gas-completion": gas_completion_units,
            "ely-translational-high-precision": ely_high_precision_units,
        }[args.block]()
    )
    assert (
        len(units)
        == {
            "methods-core": 54,
            "sources": 1231,
            "mixtures": 40,
            "liquid-pressure": 59,
            "water-air": 65,
            "air-critical-high-precision": 8,
            "gas-shapes": 32,
            "gas-completion": 123,
            "ely-translational-high-precision": 51,
        }[args.block]
    )
    print(f"{args.block}: fixed denominator {len(units)}", flush=True)
    start = time.perf_counter()
    rows = []
    for identity, inputs, function in units:
        row = {"unit": identity, "inputs": inputs}
        try:
            row.update(metrics=function(), status="passed")
        except Exception:
            row.update(status="failed", failure=traceback.format_exc())
        rows.append(row)
        if len(rows) % 250 == 0 or len(rows) == len(units):
            seconds = time.perf_counter() - start
            print(
                f"{args.block}: {len(rows)}/{len(units)}; {len(rows) / seconds:.1f}/s; "
                f"ETA {(len(units) - len(rows)) * seconds / len(rows):.1f}s",
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
    if args.block == "sources":
        for key in ("value_comparisons", "derivative_comparisons", "rejected", "unbuilt"):
            summary[key] = sum(r.get("metrics", {}).get(key, 0) for r in rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    for row in [r for r in rows if r["status"] == "failed"][:5]:
        print(row["unit"], row["failure"])
    return bool(summary["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
