"""80-digit public potentials; extensive mole derivatives are computed directly."""

import mpmath as mp


def expected(spec, temperature, fractions):
    with mp.workdps(80):
        p = {key: mp.mpf(str(value)) for key, value in spec.parameters.items()}
        T = mp.mpf(str(temperature))
        xs = tuple(mp.mpf(str(value)) for value in fractions)
        ids, n = spec.component_ids, len(xs)
        R = mp.mpf("8.31446261815324")

        def energy(temp, *amounts):
            volumes = [p[f"V:{key}"] for key in ids]
            denominator = mp.fsum(x * v for x, v in zip(amounts, volumes, strict=True))
            cohesive = mp.mpf(0)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if spec.model == "hansen":
                        a = (
                            p["alpha"]
                            / 2
                            * mp.fsum(
                                weight * (p[f"{key}:{ids[i]}"] - p[f"{key}:{ids[j]}"]) ** 2
                                for key, weight in (
                                    ("delta_d", 1),
                                    ("delta_p", mp.mpf(".25")),
                                    ("delta_h", mp.mpf(".25")),
                                )
                            )
                        )
                    else:
                        di, dj = p[f"delta:{ids[i]}"], p[f"delta:{ids[j]}"]
                        a = (di - dj) ** 2 / 2 + di * dj * p[f"lambda:{ids[i]}|{ids[j]}"]
                    cohesive += amounts[i] * amounts[j] * volumes[i] * volumes[j] * a
            value = cohesive / denominator
            if spec.model != "regular_solution":
                value += (
                    R
                    * temp
                    * mp.fsum(
                        x * mp.log(v / denominator) for x, v in zip(amounts, volumes, strict=True)
                    )
                )
            return value

        def extensive(temp, *amounts):
            total = mp.fsum(amounts)
            return total * energy(temp, *(v / total for v in amounts))

        def derivative(function, *indices):
            return mp.diff(function, (T, *xs), [indices.count(i) for i in range(n + 1)])

        ge, gt, gtt, gttt = (derivative(energy, *([0] * k)) for k in range(4))
        gx = [derivative(energy, i + 1) for i in range(n)]
        gtx = [derivative(energy, 0, i + 1) for i in range(n)]
        gxx = [[derivative(energy, i + 1, j + 1) for j in range(n)] for i in range(n)]
        pg = [derivative(extensive, i + 1) for i in range(n)]
        pgt = [derivative(extensive, 0, i + 1) for i in range(n)]
        mole = [[derivative(extensive, i + 1, j + 1) / (R * T) for j in range(n)] for i in range(n)]
        logs = [v / (R * T) for v in pg]
        return {
            "GE": float(ge),
            "HE": float(ge - T * gt),
            "SE": float(-gt),
            "CpE": float(-T * gtt),
            "dGE_dT": float(gt),
            "d2GE_dT2": float(gtt),
            "d3GE_dT3": float(gttt),
            "dGE_dxs": list(map(float, gx)),
            "d2GE_dxixjs": [list(map(float, row)) for row in gxx],
            "d2GE_dTdxs": list(map(float, gtx)),
            "gamma": [float(mp.exp(v)) for v in logs],
            "ln_gamma": list(map(float, logs)),
            "dln_gamma_dT": [float((b - a / T) / (R * T)) for a, b in zip(pg, pgt, strict=True)],
            "dln_gamma_dns": [list(map(float, row)) for row in mole],
            "partial_GE": list(map(float, pg)),
            "partial_HE": [float(a - T * b) for a, b in zip(pg, pgt, strict=True)],
            "partial_SE": [float(-v) for v in pgt],
        }
