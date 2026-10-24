"""Independent matrix-exponential transport and analytic acid/precursor chemistry."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal

R, T, F = 8.31446261815324, 298.15, 96485.33212
L, AREA, VW = 1e-3, 1e-3, 18e-6
TIMES = np.array([0.0, 10.0, 25.0, 50.0])


def parameters(kind, condition):
    dp, dm = (1e-9 if kind == "salt" and condition == 0 else 2e-9), 1e-9
    return {
        "q": (2e-9, 5e-9)[condition],
        "disp": (0.0, 2e-9)[condition],
        "da": 2 * dp * dm / (dp + dm),
        "dp": dp,
        "dm": dm,
        "kc": (1.0, 10.0)[condition],
        "kf": (0.002, 0.008)[condition],
        "ks": (2.0, 0.5)[condition],
    }


def ion(c, kc):
    return 2 * kc * c / (kc + np.sqrt(kc * kc + 4 * kc * c))


def linear_solution(cells, p, start, incoming):
    """Exact exponential of independently assembled conservative linear equations."""
    q, dx, volume = p["q"], L / cells, L * AREA / cells
    d, a = (p["da"] + p["disp"]) / dx**2, q / volume
    diagonal = np.full(cells, -2 * d - a)
    diagonal[0] = diagonal[-1] = -d - a
    similarity = ((d + a) / d) ** (np.arange(cells) / 2)
    eigenvalues, vectors = eigh_tridiagonal(diagonal, np.full(cells - 1, np.sqrt(d * (d + a))))
    weights = vectors.T @ (np.full(cells, start - incoming) / similarity)

    def evaluate(t):
        column = incoming + similarity * (vectors @ (np.exp(eigenvalues * t) * weights))
        integral = np.expm1(eigenvalues * t) / eigenvalues
        collected = q * (incoming * t + similarity[-1] * np.dot(vectors[-1], integral * weights))
        return column, float(collected)

    matrix = (
        np.diag(diagonal)
        + np.diag(np.full(cells - 1, d), 1)
        + np.diag(np.full(cells - 1, d + a), -1)
    )
    return evaluate, matrix


def structure(
    kind, p, column_total, collected_total, t, feed_volume, column_s=None, collected_s=0.0
):
    cells, q = len(column_total), p["q"]
    volumes = np.r_[feed_volume - q * t, np.full(cells, L * AREA / cells), q * t]
    names = ["feed", *(f"cell-{i:02d}" for i in range(cells)), "collector"]
    ct = collected_total / volumes[-1] if volumes[-1] else 0.0
    total = np.r_[20.0 if kind == "salt" else 38.0 if kind == "slow" else 30.0, column_total, ct]
    concentration = {"H2O": np.full(cells + 2, 1 / VW)}
    if kind == "salt":
        concentration.update({"M+": total, "A-": total})
    else:
        if kind == "slow":
            precursor = np.r_[8.0, column_s, collected_s / volumes[-1] if volumes[-1] else 0.0]
            total = total - precursor
            concentration["S"] = precursor
        ions = ion(total, p["kc"])
        concentration.update({"H+": ions, "A-": ions, "HA": total - ions})
    amounts = {
        v: {k: float(c[i] * volumes[i]) for k, c in concentration.items()}
        for i, v in enumerate(names)
    }
    return amounts, concentration, volumes


def face_values(concentration, p, cells):
    ids, q, dx = list(concentration), p["q"], L / cells
    faces = []
    hs = {
        k: 1000.0 if k == "H2O" else -1000.0 if k == "HA" else 500.0 if k == "S" else 0.0
        for k in ids
    }
    for i in range(cells + 1):
        adv = {k: float(q * c[i]) for k, c in concentration.items()}
        dispersion, molecular = dict.fromkeys(ids, 0.0), dict.fromkeys(ids, 0.0)
        potential = 0.0
        if 0 < i < cells:
            for k, c in concentration.items():
                if k != "H2O":
                    dispersion[k] = float(-AREA * p["disp"] * (c[i + 1] - c[i]) / dx)
                    molecular[k] = float(-AREA * p["da"] * (c[i + 1] - c[i]) / dx)
            key = "M+" if "M+" in concentration else "H+"
            potential = (
                -R
                * T
                / F
                * (p["dp"] - p["dm"])
                / (p["dp"] + p["dm"])
                * np.log(concentration[key][i + 1] / concentration[key][i])
            )
        total = {k: adv[k] + dispersion[k] + molecular[k] for k in ids}
        faces.append(
            {
                "advection": adv,
                "dispersion": dispersion,
                "molecular": molecular,
                "total": total,
                "potential_V": float(potential),
                "enthalpy_W": sum(hs[k] * total[k] for k in ids),
            }
        )
    return faces


def history(kind, condition, cells, *, times=TIMES, feed_volume=5e-6, max_step=0.25):
    p = parameters(kind, condition)
    start, incoming = (
        (5.0, 20.0) if kind == "salt" else (13.0, 38.0) if kind == "slow" else (10.0, 30.0)
    )
    exact, matrix = linear_solution(cells, p, start, incoming)
    nslow = cells + 1 if kind == "slow" else 0

    def unpack(t, y):
        total, collected = exact(t)
        return structure(
            kind,
            p,
            total,
            collected,
            t,
            feed_volume,
            y[:cells] if nslow else None,
            y[cells] if nslow else 0.0,
        )

    def rhs(t, y):
        _, conc, _ = unpack(t, y)
        faces = face_values(conc, p, cells)
        enthalpy = np.zeros(cells + 2)
        for i, face in enumerate(faces):
            enthalpy[i] -= face["enthalpy_W"]
            enthalpy[i + 1] += face["enthalpy_W"]
        if not nslow:
            return enthalpy
        slow = matrix @ y[:cells] - p["kf"] * y[:cells] + p["kf"] / p["ks"] * conc["HA"][1:-1]
        slow[0] += p["q"] / (L * AREA / cells) * 8.0
        return np.r_[slow, p["q"] * y[cells - 1], enthalpy]

    initial = np.r_[np.full(cells, 3.0), 0.0, np.zeros(cells + 2)] if nslow else np.zeros(cells + 2)
    solution = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        initial,
        method="DOP853",
        t_eval=times,
        rtol=2e-12,
        atol=1e-12,
        max_step=max_step,
    )
    if not solution.success:
        raise ValueError(solution.message)
    points = []
    for t, values in zip(times, solution.y.T, strict=True):
        amounts, concentrations, volumes = unpack(float(t), values)
        h = {
            v: sum(
                n * (1000.0 if k == "H2O" else -1000.0 if k == "HA" else 500.0 if k == "S" else 0.0)
                for k, n in ns.items()
            )
            for v, ns in amounts.items()
        }
        points.append(
            {
                "time_s": float(t),
                "amounts": amounts,
                "concentrations": concentrations,
                "volumes": volumes,
                "enthalpy": h,
                "transported_h": values[nslow:],
                "faces": face_values(concentrations, p, cells),
            }
        )
    initial_h = points[0]["enthalpy"]
    for point in points:
        point["heat"] = {
            v: h - initial_h[v] - float(point["transported_h"][i])
            for i, (v, h) in enumerate(point["enthalpy"].items())
        }
    return points


def front_errors(kind, condition, native_fronts):
    p = parameters(kind, condition)
    start, incoming = (
        (5.0, 20.0) if kind == "salt" else (13.0, 38.0) if kind == "slow" else (10.0, 30.0)
    )
    dense, _ = linear_solution(256, p, start, incoming)
    fine = dense(50.0)[0]
    errors = []
    for count in (4, 8, 16):
        grouped = fine.reshape(count, 256 // count).mean(axis=1)
        errors.append(float(np.mean(abs(np.asarray(native_fronts[count]) - grouped))))
    return {"errors_mol_m3": errors, "ratios": [errors[0] / errors[1], errors[1] / errors[2]]}
