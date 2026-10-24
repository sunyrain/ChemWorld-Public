"""Synthetic source declarations, independent of the implementation equations."""

from __future__ import annotations

import copy

THERMOS = {
    "NASA7": {
        "model": "NASA7",
        "temperature-ranges": [200, 1000, 3000],
        "data": [[3.5, 1e-4, -2e-8, 1e-12, -1e-17, -1200, 2]] * 2,
    },
    "NASA9": {
        "model": "NASA9",
        "temperature-ranges": [200, 1000, 3000],
        "data": [[100, -5, 3.5, 1e-4, -2e-8, 1e-12, -1e-17, -1200, 2]] * 2,
    },
    "Shomate": {
        "model": "Shomate",
        "temperature-ranges": [200, 1000, 3000],
        "data": [[30, 2, -0.5, 0.1, 0.01, -10, 50]] * 2,
    },
    "constant-cp": {
        "model": "constant-cp",
        "T0": 298.15,
        "h0": "-10 kJ/mol",
        "s0": "30 J/mol/K",
        "cp0": "32 J/mol/K",
        "T-min": 200,
        "T-max": 3000,
    },
    "piecewise-Gibbs": {
        "model": "piecewise-Gibbs",
        "h0": "-10 kJ/mol",
        "data": {200: "-17 kJ/mol", 298.15: "-20 kJ/mol", 1000: "-55 kJ/mol", 3000: "-175 kJ/mol"},
        "T-min": 200,
        "T-max": 3000,
    },
}


def payload(reaction=None, thermo="NASA7"):
    species = [
        {"name": name, "composition": {"H": 2}, "thermo": copy.deepcopy(THERMOS[thermo])}
        for name in ("A", "B", "C")
    ]
    return {
        "phases": [
            {
                "name": "synthetic",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "species": ["A", "B", "C"],
                "elements": ["H"],
            }
        ],
        "units": {"length": "m", "quantity": "kmol", "activation-energy": "J/kmol"},
        "species": species,
        "reactions": [
            reaction or {"equation": "A => B", "rate-constant": {"A": 2, "b": 0, "Ea": 0}}
        ],
    }


def rate_cases():
    result = {
        "arrhenius": {"equation": "A <=> B", "rate-constant": {"A": 25, "b": 0.2, "Ea": 2e6}},
        "third-body": {
            "equation": "A + M <=> B + M",
            "type": "three-body",
            "rate-constant": {"A": 2e5, "b": -0.3, "Ea": 1e6},
            "efficiencies": {"C": 2},
        },
        "P-log": {
            "equation": "A <=> B",
            "type": "pressure-dependent-Arrhenius",
            "rate-constants": [
                {"P": f"{p} atm", "A": a, "b": 0.1, "Ea": 1e6}
                for p, a in ((0.1, 10), (1, 20), (10, 40))
            ],
        },
        "Chebyshev": {
            "equation": "A <=> B",
            "type": "Chebyshev",
            "temperature-range": [300, 2000],
            "pressure-range": ["0.1 atm", "10 atm"],
            "data": [[1, 0.1], [0.2, -0.01]],
        },
        "Blowers-Masel": {
            "equation": "A <=> B",
            "type": "Blowers-Masel",
            "rate-constant": {"A": 50, "b": 0.1, "Ea0": 3e6, "w": 1e9},
        },
    }
    for kind in ("falloff", "chemically-activated"):
        for family, parameters in (
            ("Lindemann", None),
            ("Troe", {"A": 0.5, "T3": 100, "T1": 1000, "T2": 3000}),
            ("SRI", {"A": 0.7, "B": 500, "C": 1000, "D": 1.1, "E": 0.05}),
            ("Tsang", {"A": 0.6, "B": 1e-4}),
        ):
            node = {
                "equation": "A (+M) <=> B (+M)",
                "type": kind,
                "low-P-rate-constant": {"A": 2e5, "b": -0.2, "Ea": 1e6},
                "high-P-rate-constant": {"A": 30, "b": 0.1, "Ea": 2e6},
                "efficiencies": {"C": 2},
            }
            if parameters:
                node[family] = parameters
            result[f"{kind}/{family}"] = node
    return result
