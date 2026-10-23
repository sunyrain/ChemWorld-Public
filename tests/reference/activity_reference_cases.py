"""Fixed CONV23 activity inputs and independent public-equation references."""

import mpmath as mp
from thermo import NRTL, UNIQUAC, Wilson

from chemworld.physchem.equilibrium import ActivityModelSpec

MODELS = ("ideal", "margules", "wilson", "nrtl", "uniquac")
TEMPERATURES = (280.0, 320.0, 370.0)
COMPOSITIONS = {
    2: ((0.2, 0.8), (0.55, 0.45), (1e-10, 1 - 1e-10)),
    3: ((0.2, 0.3, 0.5), (0.65, 0.2, 0.15), (1e-10, 0.4, 0.6 - 1e-10)),
}


def model_case(model, n):
    ids = tuple("ABC"[:n])
    p = {}
    if model == "uniquac":
        for i, key in enumerate(ids):
            p[f"r:{key}"] = 0.9 + 1.1 * i
            p[f"q:{key}"] = 1.1 + 0.6 * i
    for i, left in enumerate(ids):
        for j, right in enumerate(ids):
            if i == j or model == "ideal":
                continue
            pair = f":{left}|{right}"
            if model == "margules":
                p["A" + pair] = 0.6 + 0.2 * i + 0.3 * j
            elif model == "nrtl":
                coefficients = (
                    0.03 * (i + 1) - 0.02 * (j + 1),
                    90 * (i + 1) - 55 * (j + 1),
                    0.006 * (i + j + 1),
                    1e-4 * (i - j),
                    130 * (i + j + 1),
                    -1e-7 * (i + j + 1),
                )
                for letter, value in zip("abefgh", coefficients, strict=True):
                    p[f"tau_{letter}{pair}"] = value
                p["alpha_c" + pair] = 0.2 + 0.02 * (i + j)
                p["alpha_d" + pair] = 1e-4 * (i + 1)
            else:
                prefix = "lambda" if model == "wilson" else "tau"
                coefficients = (
                    0.1 * (i - j),
                    80 * (i + 1) - 120 * (j + 1),
                    0.005 * (i + j + 1),
                    2e-4 * (i - j),
                    60 * (i + j + 1),
                    1e-7 * (i + j + 1),
                )
                for letter, value in zip("abcdef", coefficients, strict=True):
                    p[f"{prefix}_{letter}{pair}"] = value
    return ActivityModelSpec(f"fixed23-{model}-{n}", ids, model, p)


def upstream(spec, T, xs):
    ids, p = spec.component_ids, spec.parameters
    if spec.model not in {"wilson", "nrtl", "uniquac"}:
        return None
    prefix, letters = (
        ("lambda", "abcdef")
        if spec.model == "wilson"
        else ("tau", "abefgh")
        if spec.model == "nrtl"
        else ("tau", "abcdef")
    )
    kwargs = {
        f"{prefix}_{letter}s": [
            [p.get(f"{prefix}_{letter}:{left}|{right}", 0) for right in ids] for left in ids
        ]
        for letter in letters
    }
    if spec.model == "nrtl":
        kwargs.update(
            {
                f"alpha_{letter}s": [
                    [p.get(f"alpha_{letter}:{left}|{right}", 0) for right in ids] for left in ids
                ]
                for letter in "cd"
            }
        )
    if spec.model == "uniquac":
        kwargs.update(rs=[p[f"r:{key}"] for key in ids], qs=[p[f"q:{key}"] for key in ids])
    return {"wilson": Wilson, "nrtl": NRTL, "uniquac": UNIQUAC}[spec.model](
        T=T, xs=list(xs), **kwargs
    )


def mp_ge(spec, T, xs):
    """Independent scalar 80-digit evaluation of the cited public equations."""
    p = {key: mp.mpf(str(v)) for key, v in spec.parameters.items()}
    R = mp.mpf("8.31446261815324")
    n, ids = len(xs), spec.component_ids
    if spec.model == "ideal":
        return mp.mpf(0)
    if spec.model == "margules":
        return (
            R
            * T
            * mp.fsum(
                xs[i]
                * xs[j]
                * (
                    (p[f"A:{ids[i]}|{ids[j]}"] + p[f"A:{ids[j]}|{ids[i]}"]) / 2
                    + (p[f"A:{ids[j]}|{ids[i]}"] - p[f"A:{ids[i]}|{ids[j]}"]) * (xs[i] - xs[j]) / 2
                )
                for i in range(n)
                for j in range(i + 1, n)
            )
        )

    def parameter(i, j):
        if i == j:
            return mp.mpf(0) if spec.model == "nrtl" else mp.mpf(1)
        prefix = "lambda" if spec.model == "wilson" else "tau"
        letters = "abefgh" if spec.model == "nrtl" else "abcdef"
        coefficients = [p[f"{prefix}_{v}:{ids[i]}|{ids[j]}"] for v in letters]
        terms = (1, 1 / T, mp.log(T), T, 1 / T**2, T**2)
        value = mp.fsum(a * b for a, b in zip(coefficients, terms, strict=True))
        return value if spec.model == "nrtl" else mp.exp(value)

    matrix = [[parameter(i, j) for j in range(n)] for i in range(n)]
    if spec.model == "wilson":
        return (
            -R
            * T
            * mp.fsum(
                xs[i] * mp.log(mp.fsum(xs[j] * matrix[i][j] for j in range(n))) for i in range(n)
            )
        )
    if spec.model == "nrtl":
        g = [
            [
                mp.exp(
                    -(
                        p.get(f"alpha_c:{ids[i]}|{ids[j]}", 0)
                        + p.get(f"alpha_d:{ids[i]}|{ids[j]}", 0) * T
                    )
                    * matrix[i][j]
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
        return (
            R
            * T
            * mp.fsum(
                xs[i]
                * mp.fsum(xs[j] * matrix[j][i] * g[j][i] for j in range(n))
                / mp.fsum(xs[j] * g[j][i] for j in range(n))
                for i in range(n)
            )
        )
    rs, qs = ([p[f"{name}:{key}"] for key in ids] for name in ("r", "q"))
    rbar, qbar = (mp.fsum(x * v for x, v in zip(xs, values, strict=True)) for values in (rs, qs))
    theta = [xs[i] * qs[i] / qbar for i in range(n)]
    return (
        R
        * T
        * mp.fsum(
            xs[i]
            * (
                mp.log(rs[i] / rbar)
                + 5 * qs[i] * mp.log(qs[i] * rbar / (rs[i] * qbar))
                - qs[i] * mp.log(mp.fsum(theta[j] * matrix[j][i] for j in range(n)))
            )
            for i in range(n)
        )
    )


def high_precision_report(spec, T, xs):
    with mp.workdps(80):
        T = mp.mpf(str(T))
        xs = [mp.mpf(str(v)) for v in xs]
        n = len(xs)
        point = (T, *xs)

        def ge(*point):
            return mp_ge(spec, point[0], point[1:])

        def derivative(*indices):
            orders = [indices.count(i) for i in range(n + 1)]
            return mp.diff(ge, point, orders)

        g, gt, gtt, gttt = (derivative(*([0] * order)) for order in range(4))
        gx = [derivative(i + 1) for i in range(n)]
        gtx = [derivative(0, i + 1) for i in range(n)]
        h = [[derivative(i + 1, j + 1) for j in range(n)] for i in range(n)]

        def partial(g, grad):
            return [g + v - mp.fsum(x * s for x, s in zip(xs, grad, strict=True)) for v in grad]

        pg, pgt = partial(g, gx), partial(gt, gtx)
        R = mp.mpf("8.31446261815324")
        logg = [v / (R * T) for v in pg]
        mole = [
            [
                (
                    h[i][j]
                    - mp.fsum(xs[k] * h[k][j] for k in range(n))
                    - mp.fsum(xs[k] * h[i][k] for k in range(n))
                    + mp.fsum(xs[k] * xs[ell] * h[k][ell] for k in range(n) for ell in range(n))
                )
                / (R * T)
                for j in range(n)
            ]
            for i in range(n)
        ]
        return {
            "GE": float(g),
            "HE": float(g - T * gt),
            "SE": float(-gt),
            "CpE": float(-T * gtt),
            "dGE_dT": float(gt),
            "d2GE_dT2": float(gtt),
            "d3GE_dT3": float(gttt),
            "dGE_dxs": list(map(float, gx)),
            "d2GE_dxixjs": [list(map(float, row)) for row in h],
            "d2GE_dTdxs": list(map(float, gtx)),
            "gamma": [float(mp.exp(v)) for v in logg],
            "ln_gamma": list(map(float, logg)),
            "dln_gamma_dT": [float((dt - v / T) / (R * T)) for v, dt in zip(pg, pgt, strict=True)],
            "dln_gamma_dns": [list(map(float, row)) for row in mole],
            "partial_GE": list(map(float, pg)),
            "partial_HE": [float(v - T * dt) for v, dt in zip(pg, pgt, strict=True)],
            "partial_SE": [float(-dt) for dt in pgt],
        }


def expected_report(spec, T, xs):
    ref = upstream(spec, T, xs)
    if ref is None:
        return high_precision_report(spec, T, xs)
    with mp.workdps(80):
        temp = mp.mpf(str(T))
        fractions = [mp.mpf(str(v)) for v in xs]
        third = float(mp.diff(lambda v: mp_ge(spec, v, fractions), temp, 3))
    gammas = ref.gammas()
    return {
        "GE": ref.GE(),
        "HE": ref.HE(),
        "SE": ref.SE(),
        "CpE": ref.CpE(),
        "dGE_dT": ref.dGE_dT(),
        "d2GE_dT2": ref.d2GE_dT2(),
        "d3GE_dT3": third,
        "dGE_dxs": ref.dGE_dxs(),
        "d2GE_dxixjs": ref.d2GE_dxixjs(),
        "d2GE_dTdxs": ref.d2GE_dTdxs(),
        "gamma": gammas,
        "ln_gamma": ref.lngammas(),
        "dln_gamma_dT": ref.dlngammas_dT(),
        "dln_gamma_dns": [[v / gammas[i] for v in row] for i, row in enumerate(ref.dgammas_dns())],
        "partial_GE": ref.dnGE_dns(),
        "partial_HE": ref.dnHE_dns(),
        "partial_SE": [-v for v in ref.d2nGE_dTdns()],
    }
