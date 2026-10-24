"""Independent ambipolar modes, acid quadratics and scalar precursor ODE."""

import numpy as np
from scipy.integrate import solve_ivp

T, R, F = 298.15, 8.31446261815324, 96485.33212
LENGTH, AREA, VW = 1e-3, 1e-3, 18e-6
TIMES = (0.0, 25.0, 50.0, 75.0, 100.0)


def parameters(kind, condition):
    dp = 1e-9 if kind == "salt" and condition == 0 else 2e-9
    dm = 1e-9
    return (
        dp,
        dm,
        2 * dp * dm / (dp + dm),
        (1.0, 10.0)[condition],
        (0.002, 0.008)[condition],
        (2.0, 0.5)[condition],
    )


def ionic(total, kc):
    return 2 * kc * total / (kc + np.sqrt(kc * kc + 4 * kc * total))


def history(kind, condition, cells, *, max_step=0.5, continuous=False):
    dp, dm, da, kc, kf, ks = parameters(kind, condition)
    dx, volume = LENGTH / cells, LENGTH * AREA / cells
    mode = np.cos(np.pi * (np.arange(cells) + 0.5) / cells) * np.sinc(1 / (2 * cells))
    eigenvalue = (
        (np.pi / LENGTH) ** 2 if continuous else 4 * np.sin(np.pi / (2 * cells)) ** 2 / dx**2
    )
    matrix = (
        np.diag([-2.0] * cells) + np.diag(np.ones(cells - 1), 1) + np.diag(np.ones(cells - 1), -1)
    )
    matrix[0, 0] = matrix[-1, -1] = -1
    matrix *= da / dx**2
    slow = None
    if kind == "slow":

        def rhs(t, s):
            total = 25 + 4.75 * mode * np.exp(-da * eigenvalue * t)
            c = total - s
            return matrix @ s - kf * s + (kf / ks) * (c - ionic(c, kc))

        solved = solve_ivp(
            rhs,
            (0, 100),
            5 + 0.75 * mode,
            method="DOP853",
            t_eval=TIMES,
            rtol=2e-12,
            atol=1e-12,
            max_step=max_step,
        )
        if not solved.success:
            raise ValueError("independent scalar precursor integration failed")
        slow = solved.y.T
    points = []
    for j, t in enumerate(TIMES):
        mode_t = mode * np.exp(-da * eigenvalue * t)
        c = 20 + 4 * mode_t if kind != "salt" else 10 + 2 * mode_t
        values = {"H2O": np.full(cells, 1 / VW)}
        if kind == "salt":
            values.update({"M+": c, "A-": c})
        else:
            if kind == "slow":
                c = 25 + 4.75 * mode_t - slow[j]
                values["S"] = slow[j]
            ion = ionic(c, kc)
            values.update({"H+": ion, "A-": ion, "HA": c - ion})
        charged = values["M+" if kind == "salt" else "H+"]
        psi = -(dp - dm) / (dp + dm) * np.log(charged[1:] / charged[:-1])
        # All ionic pairs have identical concentration and ambipolar flux;
        # neutral DHA=DS=Damb was explicitly fixed before qualification.
        rates = {k: -AREA * da * np.diff(v) / dx for k, v in values.items() if k != "H2O"}
        dissipation = (
            -R * T * sum(float(rates[k] @ np.log(values[k][1:] / values[k][:-1])) for k in rates)
        )
        h = volume * float(
            sum(values.get("S", np.zeros(cells)) * 500 - values.get("HA", np.zeros(cells)) * 1000)
        )
        points.append(
            {
                "time_s": t,
                "concentrations": values,
                "potential_V": R * T / F * psi,
                "rates_mol_s": rates,
                "dissipation_W": dissipation,
                "enthalpy_J": h,
            }
        )
    for row in points:
        row["bath_heat_J"] = row["enthalpy_J"] - points[0]["enthalpy_J"]
    return points


def self_check():
    largest = 0.0
    for condition in (0, 1):
        for cells in (4, 8, 16):
            first = history("slow", condition, cells)
            refined = history("slow", condition, cells, max_step=0.25)
            for p, q in zip(first, refined, strict=True):
                values = p["concentrations"]
                kc = parameters("slow", condition)[3]
                assert np.max(np.abs(values["H+"] * values["A-"] / values["HA"] - kc)) < 1e-12
                assert abs(float(np.mean(values["S"] + values["HA"] + values["A-"])) - 25) < 1e-11
                largest = max(
                    largest,
                    max(float(np.max(abs(values[k] - q["concentrations"][k]))) for k in values),
                )
    assert largest < 1e-9, largest
    return {
        "reference_step_refinement_max_mol_m3": largest,
        "acid_quadratic_and_total_balance": "PASS",
    }


if __name__ == "__main__":
    print(self_check())
