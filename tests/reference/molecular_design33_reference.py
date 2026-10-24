"""Independent elementary ideal compressor/cooler calculus. No native imports."""

from itertools import pairwise
from math import log, sqrt

from scipy.optimize import brentq

R = 8.31446261815324
CASES = (
    (1.0, 320.0, 9e5, 0.85, 0.85, 50.0, 0.0, None, 6.5e5),
    (1.0, 320.0, 9e5, 0.70, 0.90, 50.0, 0.0, None, 6.5e5),
    (1.0, 300.0, 16e5, 0.85, 0.85, 50.0, 0.0, None, 6.5e5),
    (0.25, 330.0, 9e5, 0.80, 0.75, 50.0, 0.0, None, 6.5e5),
    (1.0, 320.0, 9e5, 0.85, 0.85, 50.0, 0.0, None, 2.5e5),
    (1.0, 320.0, 9e5, 0.85, 0.85, 50.0, 0.0, 420.0, 6.5e5),
    (1.0, 320.0, 9e5, 0.85, 0.85, 10.0, 0.0, None, 6.5e5),
    (1.0, 320.0, 9e5, 0.85, 0.85, 50.0, 0.25, None, 6.5e5),
)


def point(case, pressure):
    x, tin, pout, eta1, eta2, capacity, price, _, _ = case
    cp = R * (4.5 - x)
    exponent = R / cp
    u, v = (pressure / 1e5) ** exponent, (pout / pressure) ** exponent
    t1 = tin * (1 + (u - 1) / eta1)
    b = (v - 1) / eta2
    cooling = min(capacity, 0.01 * cp * (t1 - tin)) if b > price else 0.0
    tc = t1 - cooling / (0.01 * cp)
    dt1, db = tin * exponent * u / eta1, -exponent * v / eta2
    dc = 0.01 * cp * dt1 if b > price and cooling < capacity else 0.0
    work = 0.01 * cp * (t1 - tin + tc * b)
    derivative = 0.01 * cp * (dt1 + (dt1 - dc / (0.01 * cp)) * b + tc * db) + price * dc
    return {
        "pressure": pressure,
        "cooling": cooling,
        "cost": work + price * cooling,
        "temperature": tc * (1 + b),
        "stage1_temperature": t1,
        "cooler_temperature": tc,
        "heat": -cooling,
        "work": work,
        "d_cost_d_log_pressure": derivative,
    }


def optimum(case):
    x, tin, pout, eta1, eta2, capacity, price, tmax, upper = case
    cp, lower = R * (4.5 - x), 1.5e5
    exponent = R / cp
    if tmax is not None:
        upper = min(upper, 1e5 * (1 + eta1 * (tmax / tin - 1)) ** (1 / exponent))
    cuts = sorted(
        {
            lower,
            upper,
            *(
                p
                for p in (
                    1e5 * (1 + eta1 * capacity / (0.01 * cp * tin)) ** (1 / exponent),
                    pout / (1 + eta2 * price) ** (1 / exponent),
                )
                if lower < p < upper
            ),
        }
    )
    candidates = list(cuts)
    for lo, hi in pairwise(cuts):
        # Each open interval has a fixed cooling regime and derivative A*u+B*v.
        left, right = lo * (1 + 1e-10), hi * (1 - 1e-10)
        if (
            point(case, left)["d_cost_d_log_pressure"] * point(case, right)["d_cost_d_log_pressure"]
            < 0
        ):
            candidates.append(
                brentq(lambda p: point(case, p)["d_cost_d_log_pressure"], left, right)
            )
    best = min((point(case, p) for p in candidates), key=lambda row: row["cost"])
    return {**best, "candidate_count": len(candidates)}


def receiver(case, process, duration):
    x = case[0]
    cp, cv = R * (4.5 - x), R * (3.5 - x)
    h0 = -R * 300 * (4.5 - x)
    total = 0.05 + 0.01 * duration
    energy = 0.05 * (h0 + cv * 320) + 0.01 * (h0 + cp * process["temperature"]) * duration
    temperature = (energy - total * h0) / (total * cv)
    return {
        "amount": total,
        "temperature": temperature,
        "pressure": total * R * temperature / 0.02,
        "heat": process["heat"] * duration,
        "work": process["work"] * duration,
    }


def self_check():
    assert abs(optimum(CASES[0])["pressure"] - sqrt(1e5 * 9e5)) < 1e-6
    for case in CASES:
        best = optimum(case)
        assert best["cooler_temperature"] >= case[1] - 1e-9
        assert best["cooling"] <= case[5] + 1e-9
        if case[7] is not None:
            assert best["stage1_temperature"] <= case[7] + 1e-9
        for pressure in (1.7e5, 2.2e5):
            delta = 1e-5
            difference = (
                point(case, pressure * (1 + delta))["cost"]
                - point(case, pressure / (1 + delta))["cost"]
            ) / (2 * log(1 + delta))
            assert abs(difference - point(case, pressure)["d_cost_d_log_pressure"]) < 1e-6
    return {"cases": len(CASES), "derivative_checks": 16, "square_root_limit": True}
