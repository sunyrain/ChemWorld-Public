"""Independent ideal/Davies one-salt balances and analytic H/V; no native imports."""

from math import exp, log, sqrt

from scipy.optimize import brentq

R, T, P, MW = 8.31446261815324, 298.15, 1e5, 0.018015


def close(amounts, family):
    if family not in {"ideal", "Davies"}:
        raise ValueError("independent reference covers ideal and Davies only")
    result = dict(amounts)
    association = "NaCl(aq)" in result
    complex_n = result.get("NaCl(aq)", 0.0)
    na = result["Na+"] + complex_n + result["NaCl(s)"]
    cl = result["Cl-"] + complex_n + result["NaCl(s)"]
    delta = abs(na - cl)
    total = min(na, cl)
    mass = result["H2O"] * MW
    spectator = result["K+"] + result["Br-"]

    def ion_product(free):
        ionic = (2 * free + delta + spectator) / (2 * mass)
        gamma = (
            1.0
            if family == "ideal"
            else exp(-log(10) * 0.509 * (sqrt(ionic) / (1 + sqrt(ionic)) - 0.3 * ionic))
        )
        return gamma**2 * free * (free + delta) / mass**2

    def dissolved(free):
        return free + (10 * ion_product(free) * mass if association else 0.0)

    free = (
        brentq(lambda x: dissolved(x) - total, 0, total, xtol=1e-15)
        if association and total > 0
        else total
    )
    if ion_product(free) > 1e-4:
        free = brentq(lambda x: ion_product(x) - 1e-4, 0, free, xtol=1e-15)
    complex_n = 10 * ion_product(free) * mass if association else 0.0
    result.update(
        {
            "Na+": free + (delta if na > cl else 0),
            "Cl-": free + (delta if cl > na else 0),
            "NaCl(s)": max(0.0, total - free - complex_n),
        }
    )
    if association:
        result["NaCl(aq)"] = complex_n
    return result


def properties(amounts, family):
    mass = amounts["H2O"] * MW
    ions = sum(amounts[k] for k in ("Na+", "Cl-", "K+", "Br-"))
    solid, complex_n = amounts["NaCl(s)"], amounts.get("NaCl(aq)", 0.0)
    h = -1000 * solid - 300 * complex_n
    v = 3e-6 * amounts["H2O"] + 1e-6 * ions + 2e-6 * complex_n + 1.8e-6 * solid
    if family == "Davies":
        ionic = ions / (2 * mass)
        # Integral of log10 gamma: no native activity/thermo implementation.
        f = -4 * log(10) * 0.509 * (ionic / 2 - sqrt(ionic) + log(1 + sqrt(ionic)))
        f += log(10) * 0.509 * 0.3 * ionic**2
        h -= R * T * T * mass * 0.001 * f
        v += R * T * mass * 1e-8 * f
    return {"H": h, "V": v, "U": h - P * v}


def transfer(stocks, source, target, fraction, family):
    moved = {k: n * fraction for k, n in stocks[source].items() if k != "NaCl(s)"}
    candidate = {k: dict(v) for k, v in stocks.items()}
    for name, sign in ((source, -1), (target, 1)):
        candidate[name] = close(
            {k: n + sign * moved.get(k, 0.0) for k, n in stocks[name].items()}, family
        )
    return candidate
