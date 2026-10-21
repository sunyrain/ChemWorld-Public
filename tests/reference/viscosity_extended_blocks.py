"""Predeclared full scales/index and 39 additional viscosity model units."""

from decimal import ROUND_HALF_EVEN, Decimal
from itertools import pairwise
from math import sqrt

from chemicals import viscosity as reference

from chemworld.physchem.dense_viscosity import lucas_liquid_viscosity
from chemworld.physchem.petroleum_viscosity import viscosity_index
from chemworld.physchem.viscosity_kernels import viscosity_derivatives
from chemworld.physchem.viscosity_mixtures import (
    brokaw_viscosity,
    lorentz_bray_clarke_viscosity,
)
from chemworld.physchem.viscosity_scale_data import (
    VISCOSITY_INDEX_H,
    VISCOSITY_INDEX_L,
    VISCOSITY_INDEX_NU,
    VISCOSITY_LINEAR_SCALES,
    VISCOSITY_SCALE_TABLES,
)
from chemworld.physchem.viscosity_scales import viscosity_converter


def scale_units(compare, *, independent_inverse=False, monotone=False):
    def convert(value, old, new):
        return viscosity_converter(
            value, old, new, interpolation="monotone" if monotone else "reference_cubic"
        )

    def run(name, value):
        if name in reference.viscosity_scales:
            assert VISCOSITY_SCALE_TABLES[name] == tuple(
                tuple(v) for v in reference.viscosity_scales[name]
            )
        if name in reference.viscosity_scales_linear:
            assert VISCOSITY_LINEAR_SCALES[name] == tuple(reference.viscosity_scales_linear[name])
        si = reference.viscosity_converter(value, name, "kinematic viscosity")
        actual_si = convert(value, name, "kinematic viscosity")
        if monotone:
            metrics = {
                "roundtrip": compare(
                    convert(actual_si, "kinematic viscosity", name), value, rtol=3e-9, atol=1e-9
                )
            }
            if name in reference.viscosity_scales and name not in {
                "kinematic viscosity",
                "saybolt universal",
            }:
                nodes, nus = reference.viscosity_scales[name]
                pairs = sorted(zip(nodes, nus, strict=True))
                if value in nodes:
                    metrics["node"] = compare(
                        actual_si, nus[nodes.index(value)] * 1e-6, rtol=3e-9, atol=1e-14
                    )
                else:
                    a, b = next((a, b) for a, b in pairwise(pairs) if a[0] < value < b[0])
                    assert min(a[1], b[1]) <= actual_si * 1e6 <= max(a[1], b[1])
                    metrics["bounded_cSt"] = [a[1], b[1]]
            else:
                metrics["reference"] = compare(actual_si, si, rtol=3e-9, atol=1e-14)
            return metrics
        reverse_diagnostic = {}
        if independent_inverse:
            expected_back = value
            try:
                reverse_diagnostic["value"] = reference.viscosity_converter(
                    si, "kinematic viscosity", name
                )
            except Exception as exc:
                reverse_diagnostic["failure"] = repr(exc)
        else:
            expected_back = reference.viscosity_converter(si, "kinematic viscosity", name)
        return {
            "reference_reverse": reverse_diagnostic,
            "forward": compare(actual_si, si, rtol=3e-9, atol=1e-14),
            "reverse": compare(
                convert(si, "kinematic viscosity", name), expected_back, rtol=3e-9, atol=1e-9
            ),
            "roundtrip": compare(
                convert(actual_si, "kinematic viscosity", name), value, rtol=3e-9, atol=1e-9
            ),
        }

    for name, (nodes, _) in reference.viscosity_scales.items():
        if name == "kinematic viscosity":
            continue
        values = [*nodes, *(sqrt(a * b) for a, b in pairwise(nodes))]
        for i, value in enumerate(values):
            yield (
                f"scales/{name}/{i}",
                {"scale": name, "value": value},
                (lambda n=name, v=value: run(n, v)),
            )
    for name, (_, minimum) in reference.viscosity_scales_linear.items():
        if name in reference.viscosity_scales:
            continue
        for multiplier in (1, 2, 10):
            value = minimum * multiplier
            yield (
                f"linear/{name}/{multiplier}",
                {"scale": name, "value": value},
                (lambda n=name, v=value: run(n, v)),
            )
    for value in (1e-6, 10e-6, 1000e-6):
        yield (
            f"identity/{value}",
            {"value_m2_s": value},
            (lambda v=value: run("kinematic viscosity", v)),
        )


def index_units(compare, *, standard_rounding=False):
    for a, b in (
        (VISCOSITY_INDEX_NU, reference.VI_nus),
        (VISCOSITY_INDEX_L, reference.VI_Ls),
        (VISCOSITY_INDEX_H, reference.VI_Hs),
    ):
        assert tuple(a) == tuple(b)

    def run(u, v):
        expected = reference.viscosity_index(u * 1e-6, v * 1e-6)
        actual = viscosity_index(u * 1e-6, v * 1e-6)
        rounded = viscosity_index(u * 1e-6, v * 1e-6, rounding=True)
        expected_rounded = reference.viscosity_index(u * 1e-6, v * 1e-6, rounding=True)
        if standard_rounding and expected is not None:
            standard = int(
                Decimal.from_float(expected).quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
            )
            assert rounded == standard, (rounded, standard)
        else:
            assert rounded == expected_rounded, (rounded, expected_rounded)
        if expected is None:
            assert actual is None
            return {"undefined": True}
        return {
            "raw": compare(actual, expected, atol=1e-8, rtol=1e-10),
            "rounded": rounded,
            "reference_rounded": expected_rounded,
            "rounding_agrees": rounded == expected_rounded,
        }

    cases = []
    for v, low, high in zip(reference.VI_nus, reference.VI_Ls, reference.VI_Hs, strict=True):
        cases.extend((u, v) for u in (0.8 * high, high, low))
    for i in range(len(reference.VI_nus) - 1):
        v = (reference.VI_nus[i] + reference.VI_nus[i + 1]) / 2
        u = (
            sum(
                (
                    reference.VI_Ls[i],
                    reference.VI_Ls[i + 1],
                    reference.VI_Hs[i],
                    reference.VI_Hs[i + 1],
                )
            )
            / 4
        )
        cases.append((u, v))
    cases.extend((10 * v, v) for v in (1, 1.999, 70, 100, 1000))
    for i, (u, v) in enumerate(cases):
        yield f"index/{i}", {"nu40_cSt": u, "nu100_cSt": v}, lambda u=u, v=v: run(u, v)


def model_units(compare, compare_slope):
    for tb, rho in ((400, 700), (672.3166, 895.5189), (900, 1000)):
        for t in (300, 338.7055, 450):

            def run(t=t, tb=tb, rho=rho):
                value, slope = viscosity_derivatives("Twu_1985", t, {"Tb": tb, "rho_60F": rho})
                return {
                    "value": compare(value, reference.Twu_1985(t, tb, rho)),
                    "derivatives": compare_slope(
                        slope, lambda x: reference.Twu_1985(x, tb, rho), t
                    ),
                }

            yield f"Twu/{tb}/{t}", {"T": t, "Tb": tb, "rho60F": rho}, run
    for t in (300, 450, 550):
        for pressure, psat in ((1e5, 2e5), (2e5, 2e5), (5e7, 2e5)):

            def run(t=t, p=pressure, psat=psat):
                parameters = {
                    "critical_temperature_K": 572.2,
                    "critical_pressure_Pa": 3.47e6,
                    "acentric_factor": 0.236,
                    "saturation_pressure_Pa": psat,
                    "saturation_viscosity_Pa_s": 0.00068,
                }
                if p < psat:
                    try:
                        lucas_liquid_viscosity(t, p, **parameters)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("below saturation must reject by default")
                return compare(
                    lucas_liquid_viscosity(t, p, **parameters, below_saturation="saturated_limit"),
                    reference.Lucas(t, p, 572.2, 3.47e6, 0.236, psat, 0.00068),
                )

            yield f"Lucas/{t}/{pressure}", {"T": t, "P": pressure, "Psat": psat}, run
    for delta in ((0, 0), (0.1, 0.1), (0.42, 0.19), (0, 0.42)):
        for x in ((0, 1), (0.05, 0.95), (1, 0)):

            def run(delta=delta, x=x):
                mus, mw, epsilon = (1.34e-5, 9.5029e-6), (64.06, 46.07), (347, 432)
                return compare(
                    brokaw_viscosity(
                        308.2, x, mus, mw, polar_parameters=delta, epsilon_k_K=epsilon
                    ),
                    reference.Brokaw(308.2, x, mus, mw, delta, epsilon),
                )

            yield f"Brokaw/{delta}/{x}", {"delta": delta, "x": x, "T": 308.2}, run
    for t in (300, 450, 600):
        for volume in (0.0023025, 0.0003, 0.00015):

            def run(t=t, volume=volume):
                x, mw = (0.4, 0.3, 0.3), (16.04246, 30.06904, 44.09562)
                tc, pc, vc = (
                    (190.564, 305.32, 369.83),
                    (4599000, 4872000, 4248000),
                    (9.86e-5, 0.0001455, 0.0002),
                )
                return compare(
                    lorentz_bray_clarke_viscosity(
                        t,
                        volume,
                        x,
                        molecular_weights_g_mol=mw,
                        critical_temperatures_K=tc,
                        critical_pressures_Pa=pc,
                        critical_volumes_m3_mol=vc,
                    ),
                    reference.Lorentz_Bray_Clarke(t, 1e6, volume, x, mw, tc, pc, vc),
                )

            yield f"LBC/{t}/{volume}", {"T": t, "Vm": volume}, run


def branch_units(compare, compare_slope, *, published_gharagheizi=False):
    cases = []
    for vf in (90e-6, 95e-6, 100e-6):
        for t in (250, 300, 383, 450, 500, 550):
            p = {
                "Tm": 178,
                "Tc": 591.8,
                "Pc": 41e5,
                "Vc": 316e-6,
                "Vf": vf,
                "omega": 0.263,
                "MW": 92.14,
            }
            cases.append(
                (
                    "Przedziecki_Sridhar",
                    t,
                    p,
                    lambda t, vf=vf: reference.Przedziecki_Sridhar(
                        t, 178, 591.8, 41e5, 316e-6, vf, 0.263, 92.14
                    ),
                    False,
                )
            )
    for reduced in (0.01, 0.022, 0.05, 0.075, 0.1):
        dipole = sqrt(reduced * 500**2 / (52.46 * 50))
        p = {"Tc": 500, "Pc": 5e6, "Zc": 0.25, "MW": 40, "dipole": dipole}
        for t in (250, 700):
            cases.append(("Lucas_gas", t, p, lambda t, p=p: reference.Lucas_gas(t, **p), False))
    for tc, pc, zc, mw, q, cas in (
        (5.1953, 227600, 0.302, 4.002602, 1.38, "7440-59-7"),
        (33.145, 1296400, 0.303, 2.01588, 0.76, "1333-74-0"),
        (38.34, 1679600, 0.306, 4.0282, 0.52, "7782-39-0"),
    ):
        p = {"Tc": tc, "Pc": pc, "Zc": zc, "MW": mw, "quantum_Q": q}
        for tr in (6, 12, 18):
            cases.append(
                (
                    "Lucas_gas",
                    tr * tc,
                    p,
                    lambda t, tc=tc, pc=pc, zc=zc, mw=mw, cas=cas: reference.Lucas_gas(
                        t, tc, pc, zc, mw, 0, CASRN=cas
                    ),
                    tr == 12,
                )
            )
    for t in (50, 99.5, 100, 100.5, 500):
        p = {"Tc": 500, "Pc": 5e6, "MW": 40, "low_temperature_limit": 1}
        cases.append(
            (
                "viscosity_gas_Gharagheizi",
                t,
                p,
                lambda t: reference.viscosity_gas_Gharagheizi(t, 500, 5e6, 40),
                t == 100,
            )
        )
    for i, (method, t, p, function, branch) in enumerate(cases):
        if published_gharagheizi and method != "viscosity_gas_Gharagheizi":
            continue

        def run(method=method, t=t, p=p, function=function, branch=branch):
            raw = function(t)
            reference_function = (lambda x: abs(function(x))) if published_gharagheizi else function
            if branch:
                value = viscosity_derivatives(method, t, p, order=0)[0]
                try:
                    viscosity_derivatives(method, t, p)
                except ValueError as exc:
                    return {
                        "value": compare(value, reference_function(t)),
                        "derivative_rejection": str(exc),
                        "reference_raw": raw,
                    }
                raise AssertionError("branch derivative should require a side")
            value, slope = viscosity_derivatives(method, t, p)
            return {
                "value": compare(value, reference_function(t)),
                "reference_raw": raw,
                "derivatives": compare_slope(slope, reference_function, t),
            }

        yield f"branches/{method}/{i}", {"T": t, **p}, run
