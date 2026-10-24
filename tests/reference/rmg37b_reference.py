"""Manual group counts, independent quadrature and exact recombination solution."""

from math import exp, log

from scipy.integrate import quad

GRID = (298.15, 400, 600, 1000, 1500)
R = 8.31446261815324
COUNTS = {
    "methane": (1, 0, 0, 0),
    "ethane": (0, 2, 0, 0),
    "propane": (0, 2, 1, 0),
    "methyl": (1, 0, 0, 1),
}
H = (-75000, -40000, -20000, 200000)
S = (210, 120, 50, -5)
CP = ((35, 40, 50, 65, 75), (25, 30, 40, 50, 60), (20, 25, 30, 40, 50), (-2, -2, -2, -2, -2))
SIGMA = {"methane": 12, "methyl": 6, "ethane": 18, "propane": 18}


def thermo(name, T):
    weights = COUNTS[name]

    def cp(t):
        for j in range(len(GRID) - 1):
            if GRID[j] <= t <= GRID[j + 1]:
                fraction = (t - GRID[j]) / (GRID[j + 1] - GRID[j])
                return sum(
                    n * ((1 - fraction) * row[j] + fraction * row[j + 1])
                    for n, row in zip(weights, CP, strict=True)
                )
        raise ValueError("reference temperature outside fixed nodes")

    points = [x for x in GRID if 298.15 < x < T]
    h = sum(n * x for n, x in zip(weights, H, strict=True))
    h += quad(cp, 298.15, T, points=points, epsabs=1e-9, epsrel=1e-12)[0]
    s = sum(n * x for n, x in zip(weights, S, strict=True)) - R * log(SIGMA[name])
    s += quad(lambda t: cp(t) / t, 298.15, T, points=points, epsabs=1e-10, epsrel=1e-12)[0]
    return {"Cp": cp(T), "H": h, "S": s, "G": h - T * s}


def kinetic(tier, T):
    if tier == "ancestor":
        return 0.5 * T**0.5 * exp(-1000 / (R * T))
    return 2 if tier == "training" else 1


def trajectory(time_s):
    n = 0.1 / (1 + 0.4 * time_s)
    return {"H": n, "CH3": n, "H2": (0.1 - n) / 4, "CH4": (0.1 - n) / 2, "C2H6": (0.1 - n) / 4}
