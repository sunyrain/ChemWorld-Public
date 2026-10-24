"""Independent analytic latent/binodal oracle; no native imports."""

from math import log

from scipy.integrate import solve_ivp
from scipy.optimize import brentq

R, CP, OMEGA, TM, LATENT = 8.3145, 30.0, 12000.0, 600.0, 12000.0


def pure_state(*, temperature=None, beta=None, enthalpy=None, total=1.0, support=0.0):
    capacity = total * CP + support
    if enthalpy is not None:
        if enthalpy < 0:
            temperature, beta = TM + enthalpy / capacity, 0.0
        elif enthalpy > total * LATENT:
            temperature, beta = TM + (enthalpy - total * LATENT) / capacity, 1.0
        else:
            temperature, beta = TM, enthalpy / (total * LATENT)
    elif beta is None:
        beta = 0.0 if temperature < TM else 1.0
    h = capacity * (temperature - TM) + total * LATENT * beta
    s = capacity * log(temperature / TM) + total * LATENT / TM * beta
    phases = [
        {"name": name, "n": n, "x": {"FE": 1.0}, "sites": [{"FE": 1.0}]}
        for name, n in (("ALPHA", total * (1 - beta)), ("BETA", total * beta))
        if n > 1e-12 * total
    ]
    v = total * ((1 - beta) * 1e-5 + beta * 1.2e-5)
    return {
        "T": temperature,
        "H": h,
        "S": s,
        "V": v,
        "U": h - 101325 * v,
        "phases": phases,
        "beta": beta,
    }


def binary_state(z, *, temperature=None, enthalpy=None, total=1.0, support=0.0):
    if temperature is None:
        temperature = brentq(
            lambda t: binary_state(z, temperature=t, total=total, support=support)["H"] - enthalpy,
            500.0,
            1000.0,
            xtol=1e-10,
        )
    t = temperature
    if t >= OMEGA / (2 * R):
        xs, ns = [z], [total]
    else:
        x = brentq(
            lambda x: log(x / (1 - x)) + OMEGA / (R * t) * (1 - 2 * x), 1e-10, 0.499999, xtol=1e-14
        )
        if x < z < 1 - x:
            xs = [x, 1 - x]
            ns = [total * (1 - x - z) / (1 - 2 * x), total * (z - x) / (1 - 2 * x)]
        else:
            xs, ns = [z], [total]
    h = (total * CP + support) * (t - TM) + sum(
        n * OMEGA * x * (1 - x) for n, x in zip(ns, xs, strict=True)
    )
    s = (total * CP + support) * log(t / TM) - R * sum(
        n * (x * log(x) + (1 - x) * log(1 - x)) for n, x in zip(ns, xs, strict=True)
    )
    phases = [
        {"name": "ALPHA", "n": n, "x": {"FE": x, "NI": 1 - x}, "sites": [{"FE": x, "NI": 1 - x}]}
        for n, x in zip(ns, xs, strict=True)
    ]
    v = total * 1e-5
    return {"T": t, "H": h, "S": s, "V": v, "U": h - 101325 * v, "phases": phases}


def history(case):
    def close(**kw):
        return pure_state(**kw) if case["kind"] == "pure" else binary_state(0.3, **kw)

    initial = close(temperature=case["initial_T"], total=0.001, support=0.02)
    points, h = [initial], initial["H"]
    for start, end in zip(case["times"][:-1], case["times"][1:], strict=True):
        if "conductance" in case:

            def rhs(t, vector):
                current = close(enthalpy=vector[0], total=0.001, support=0.02)
                q = max(
                    -abs(case["power"]),
                    min(
                        abs(case["power"]), case["conductance"] * (case["setpoint"] - current["T"])
                    ),
                )
                return [q]

            result = solve_ivp(rhs, (start, end), [h], method="DOP853", rtol=1e-11, atol=1e-12)
            if not result.success:
                raise ValueError(result.message)
            h = float(result.y[0, -1])
        else:
            h += (0.0 if case["id"] == "binary-pause" and start == 80 else case["power"]) * (
                end - start
            )
        points.append(close(enthalpy=h, total=0.001, support=0.02))
    for time, point in zip(case["times"], points, strict=True):
        point.update(
            time=time, heat=point["H"] - initial["H"], work=101325 * (point["V"] - initial["V"])
        )
    return points
