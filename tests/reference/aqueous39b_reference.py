"""Independent 50-digit evaluation of public IF97/R8-97/revised-HKF equations.

No production state, derivative arithmetic or HKF implementation is called.
The shared IF97 coefficient table is published numerical data, not an oracle.
"""

from functools import lru_cache

from mpmath import mp

from chemworld.physchem.water_if97_coefficients import REGION1

mp.dps = 50
M = mp.mpf
TR, PR = M("298.15"), M("100000")


def water_g(T, P):
    return (
        M("461.526")
        * T
        * mp.fsum(
            M(str(n)) * (M("7.1") - P / M("16530000")) ** i * (1386 / T - M("1.222")) ** j
            for i, j, n in REGION1
        )
    )


def density(T, P):
    return 1 / mp.diff(lambda p: water_g(T, p), P)


def dielectric(T, rho):
    ns = tuple(
        map(
            M,
            (
                ".978224486826",
                "-.957771379375",
                ".237511794148",
                ".714692244396",
                "-.298217036956",
                "-.108863472196",
                ".0949327488264",
                "-.00980469816509",
                ".0000165167634970",
                ".0000937359795772",
                "-.000000000123179218720",
            ),
        )
    )
    ij = (
        (1, 0.25),
        (1, 1),
        (1, 2.5),
        (2, 1.5),
        (3, 1.5),
        (3, 2.5),
        (4, 2),
        (5, 2),
        (6, 5),
        (7, 0.5),
        (10, 10),
    )
    factor = 1 + mp.fsum(
        n * (rho / 322) ** i * (M("647.096") / T) ** M(str(j))
        for n, (i, j) in zip(ns, ij, strict=True)
    )
    factor += M(".00196096504426") * rho / 322 * (T / 228 - 1) ** M("-1.2")
    vacuum = 1 / (M("4e-7") * mp.pi * 299792458**2)
    number = rho / M(".018015268") * M("6.0221367e23")
    A = M("6.138e-30") ** 2 * number * factor / (vacuum * M("1.380658e-23") * T)
    B = M("1.636e-40") * number / (3 * vacuum)
    return (1 + A + 5 * B + mp.sqrt((1 + A) ** 2 + 8 + 18 * B + 10 * A * B + 9 * B * B)) / (
        4 * (1 - B)
    )


def solvent_g(T, P):
    rho, t, p = density(T, P) / 1000, T - M("273.15"), P / 100000
    if rho >= 1:
        return M(0)
    a = M("-2.037662") + M(".005747") * t - M(".000006557892") * t * t
    b = M("6.107361") - M(".01074377") * t + M(".00001268348") * t * t
    correction = 0
    if 155 < t <= 355 and p < 1000:
        correction = (((t - 155) / 300) ** M("4.8") + M("36.66666") * ((t - 155) / 300) ** 16) * (
            M("-1.504956e-10") * (1000 - p) ** 3 + M("5.017997e-14") * (1000 - p) ** 4
        )
    return a * (1 - rho) ** b - correction


def omega(record, T, P):
    z, eta, w = record["charge"], M("166027") * M("4.184"), M(str(record["omega"])) * M("4.184")
    if not z:
        return w
    radius = z * z / (w / eta + z / M("3.082"))
    g = solvent_g(T, P)
    return eta * (z * z / (radius + abs(z) * g) - z / (M("3.082") + g))


def born(record, T, P):
    return omega(record, T, P) * (1 / dielectric(T, density(T, P)) - 1)


@lru_cache(maxsize=32)
def born_ref(charge, w):
    record = {"charge": charge, "omega": w}
    return born(record, TR, PR), mp.diff(lambda t: born(record, t, PR), TR)


def hkf_g(record, T, P, elemental):
    cal = M("4.184")
    S = M(str(record["S"])) * cal
    if record["Hf"] is None:
        es = sum(M(str(n)) * M(str(elemental[e])) for e, n in record["composition"].items())
        es += record["charge"] * M(str(elemental["charge"]))
        H = M(str(record["Gf"])) * cal + TR * (S - es)
    else:
        H = M(str(record["Hf"])) * cal
    a1, a2, a3, a4 = (M(str(x)) * cal for x in record["a"])
    c1, c2 = (M(str(x)) * cal for x in record["c"])
    p, pr, theta = P / 100000, PR / 100000, M(228)
    # Johnson et al Eq. 59, direct closed expression rather than the implementation's
    # H/S integration form; the final reference normalization fixes the chosen datum.
    thermal = -c1 * (T * mp.log(T / TR) - T + TR)
    thermal -= c2 * (
        (1 / (T - theta) - 1 / (TR - theta)) * (theta - T) / theta
        - T / theta**2 * mp.log(TR * (T - theta) / (T * (TR - theta)))
    )
    mechanical = (a1 + a3 / (T - theta)) * (p - pr) + (a2 + a4 / (T - theta)) * mp.log(
        (2600 + p) / (2600 + pr)
    )
    b, bt = born_ref(record["charge"], record["omega"])
    return H - T * S + thermal + mechanical + born(record, T, P) - b - bt * (T - TR)


def hkf_state(record, T, P, elemental):
    T, P = M(str(T)), M(str(P))

    def function(t, p):
        return hkf_g(record, t, p, elemental)

    g = function(T, P)
    s = -mp.diff(lambda t: function(t, P), T)
    cp = -T * mp.diff(lambda t: function(t, P), T, 2)
    v = mp.diff(lambda p: function(T, p), P)
    return {
        "G": float(g),
        "H": float(g + T * s),
        "S": float(s),
        "Cp": float(cp),
        "V": float(v),
        "density": float(density(T, P)),
        "epsilon": float(dielectric(T, density(T, P))),
        "omega": float(omega(record, T, P)),
    }
