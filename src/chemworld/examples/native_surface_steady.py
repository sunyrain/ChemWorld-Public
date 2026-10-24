"""Authored linear surface branching with inert and optionally disconnected pools.

A/B/C are mathematical same-formula species, not fitted real chemical isomers.
"""

from __future__ import annotations

from typing import Any

from chemworld.examples.native_imported_surface import SOURCE, authored_declaration
from chemworld.physchem.catalysis_analysis import (
    CatalyticAnalysisConditions,
    map_catalytic_conditions,
)
from chemworld.physchem.surface_mechanism_import import convert_surface_mechanism


def branching_declaration(*, disconnected: bool = False) -> dict[str, Any]:
    source = authored_declaration()

    def species(name: str, composition: dict[str, int]) -> dict[str, Any]:
        return {
            "name": name,
            "composition": composition,
            "thermo": {
                "model": "constant-cp",
                "T0": 298.15,
                "h0": 0,
                "s0": 100 if name in ("A", "B", "C") else 10,
                "cp0": 30 if name in ("A", "B", "C") else 15,
                "T-min": 300,
                "T-max": 2000,
            },
        }

    surface = ["E", "X", "Y", "Z", "I", *(["F", "W"] if disconnected else [])]
    source["phases"][0]["species"] = ["A", "B", "C"]
    source["phases"][1]["species"] = surface
    source["species"] = [species(k, {"H": 2}) for k in ("A", "B", "C")]
    source["species"] += [
        species(k, {"Pt": 1, **({"He": 1} if k == "I" else {} if k in ("E", "F") else {"H": 2})})
        for k in surface
    ]
    reactions = [
        ("A + E => X", 0.5, 500),
        ("X => Y", 2, 1000),
        ("X => Z", 1, 2000),
        ("Y => B + E", 4, 1500),
        ("Z => C + E", 3, 2500),
    ]
    if disconnected:
        reactions.extend([("A + F => W", 0.3, 750), ("W => C + F", 2.5, 1250)])
    source["reactions"] = [
        {"equation": equation, "rate-constant": {"A": a, "b": 0, "Ea": ea}}
        for equation, a, ea in reactions
    ]
    return source


def branching_mechanism(*, disconnected: bool = False) -> Any:
    return convert_surface_mechanism(
        branching_declaration(disconnected=disconnected),
        surface_phase="surface",
        source=SOURCE,
        minimum_temperature_K=300,
        maximum_temperature_K=1900,
    ).surface


def branching_seeds(*, disconnected: bool = False) -> list[dict[str, float]]:
    pool = 0.6 if disconnected else 0.9
    return [
        {
            "E": pool * e,
            "X": pool * x,
            "Y": pool * y,
            "Z": pool * z,
            "I": 0.1,
            **({"F": 0.3 * f, "W": 0.3 * (1 - f)} if disconnected else {}),
        }
        for (e, x, y, z), f in zip(
            ((0.4, 0.2, 0.2, 0.2), (0.1, 0.7, 0.1, 0.1), (0.1, 0.1, 0.4, 0.4)),
            (0.3, 0.6, 0.8),
            strict=True,
        )
    ]


def run_example() -> dict[str, Any]:
    spec = branching_mechanism(disconnected=True)
    points = [
        CatalyticAnalysisConditions(t, {"A": p, "B": 0, "C": 0}, {"surface": 1e-6})
        for t in (500.0, 900.0)
        for p in (2e4, 8e4)
    ]
    return map_catalytic_conditions(
        spec,
        points,
        branching_seeds(disconnected=True),
        target_species="B",
        product_species=("B", "C"),
        element_basis="H",
    )
