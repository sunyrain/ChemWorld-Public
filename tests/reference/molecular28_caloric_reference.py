"""Independent public thermo PR flash and analytic regular-solution oracle.

No native physics imports. All material parameters and caloric gauges are
supplied explicitly; no reference database supplies hidden heat capacities.
"""

from functools import lru_cache
from math import log

from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from thermo import (
    CEOSGas,
    CEOSLiquid,
    ChemicalConstantsPackage,
    FlashVL,
    HeatCapacityGas,
    PropertyCorrelationsPackage,
)
from thermo.eos_mix import PRMIX

R, T_REF, P_REF = 8.31446261815324, 298.15, 101325.0
CP = (80.0, 100.0)


@lru_cache(maxsize=1)
def oracle():
    constants = ChemicalConstantsPackage(
        names=["propane", "butane"],
        MWs=[44.09562, 58.1222],
        Tcs=[369.83, 425.12],
        Pcs=[4.248e6, 3.796e6],
        omegas=[0.152, 0.2],
    )
    heat_capacity = [HeatCapacityGas(poly_fit=(200.0, 500.0, [cp])) for cp in CP]
    correlations = PropertyCorrelationsPackage(
        constants, HeatCapacityGases=heat_capacity, skip_missing=True
    )
    kwargs = {
        "Tcs": constants.Tcs,
        "Pcs": constants.Pcs,
        "omegas": constants.omegas,
        "kijs": [[0.0, 0.0], [0.0, 0.0]],
    }
    gas = CEOSGas(PRMIX, kwargs, HeatCapacityGases=heat_capacity, T=T_REF, P=P_REF, zs=[0.6, 0.4])
    liquid = CEOSLiquid(
        PRMIX, kwargs, HeatCapacityGases=heat_capacity, T=T_REF, P=P_REF, zs=[0.6, 0.4]
    )
    return FlashVL(constants, correlations, gas, liquid)


def pr_state(z, *, temperature=None, enthalpy=None, total=1.0, support=0.0):
    flash = oracle()
    if temperature is None:
        if support:
            temperature = brentq(
                lambda t: (
                    total * flash.flash(T=t, P=6e5, zs=[z, 1 - z]).H()
                    + support * (t - T_REF)
                    - enthalpy
                ),
                280.0,
                320.0,
                xtol=1e-10,
            )
            result = flash.flash(T=temperature, P=6e5, zs=[z, 1 - z])
        else:
            result = flash.flash(H=enthalpy / total, P=6e5, zs=[z, 1 - z])
    else:
        result = flash.flash(T=temperature, P=6e5, zs=[z, 1 - z])
    two_phase = result.gas is not None and bool(result.liquids)
    if two_phase:
        phases = (result.gas, result.liquids[0])
        ns = (result.VF * total, (1 - result.VF) * total)
    else:
        phases = (result.gas,) if result.gas is not None else (result.liquids[0],)
        ns = (total,)
    t = result.T
    h = total * result.H() + support * (t - T_REF)
    s = total * result.S() + support * log(t / T_REF)
    v = sum(n * p.V() for n, p in zip(ns, phases, strict=True))
    return {
        "phase_status": "two_phase" if two_phase else "single_phase",
        "vapor_fraction": result.VF,
        "T": t,
        "H": h,
        "S": s,
        "U": h - 6e5 * v,
        "V": v,
        "amounts": [[n * x for x in p.zs] for n, p in zip(ns, phases, strict=True)],
        "densities": [1 / p.V() for p in phases],
    }


def lle_state(z, temperature):
    t = temperature
    a = 9000.0 / (R * t)
    x = brentq(lambda x: log(x / (1 - x)) + a * (1 - 2 * x), 1e-9, 0.49, xtol=1e-14)
    xs = (x, 1 - x)
    n0 = (1 - x - z) / (1 - 2 * x)
    ns = (n0, 1 - n0)
    h, s = 0.0, 0.0
    for n, y in zip(ns, xs, strict=True):
        cp = y * CP[0] + (1 - y) * CP[1]
        h += n * (cp * (t - T_REF) + 9000 * y * (1 - y))
        s += n * (
            cp * log(t / T_REF) - R * log(1e5 / P_REF) - R * (y * log(y) + (1 - y) * log(1 - y))
        )
    v = R * t / 1e5
    return {
        "T": t,
        "H": h,
        "S": s,
        "U": h - 1e5 * v,
        "V": v,
        "amounts": [[n * y, n * (1 - y)] for n, y in zip(ns, xs, strict=True)],
        "densities": [1 / v, 1 / v],
    }


def thermal_history(mode):
    start = pr_state(0.6, temperature=300.0, total=0.01, support=0.1)

    def state(h):
        result = pr_state(0.6, enthalpy=h, total=0.01, support=0.1)
        if result["phase_status"] != "two_phase":
            raise ValueError("independent thermal history leaves fixed two-phase branch")
        return result

    points = [start]
    h = start["H"]
    for begin, end in zip(
        (0.0, 40.0, 80.0, 120.0, 160.0), (40.0, 80.0, 120.0, 160.0, 200.0), strict=True
    ):
        if mode == "proportional":
            result = solve_ivp(
                lambda time, y: [0.001 * (301 - state(y[0])["T"])],
                (begin, end),
                [h],
                method="DOP853",
                rtol=1e-11,
                atol=1e-12,
            )
            if not result.success:
                raise ValueError(result.message)
            h = float(result.y[0, -1])
        else:
            power = -0.02 if mode == "cooler" else 0.0 if mode == "pause" and begin == 80 else 0.02
            h += power * (end - begin)
        points.append(state(h))
    for i, point in enumerate(points):
        point.update(
            time=float(40 * i), heat=point["H"] - start["H"], work=6e5 * (point["V"] - start["V"])
        )
    return points
