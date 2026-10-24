"""Authored, self-contained gas/surface declaration on the existing thermal world."""

from __future__ import annotations

from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.surface_mechanism_import import convert_surface_mechanism
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.descriptor_catalysis import descriptor_catalysis_world

SOURCE = DataSourceProvenance(
    "authored-surface-34B", 0, "ChemWorld authored ideal-surface example", "1", "CC0-1.0"
)


def authored_declaration(
    density_mol_m2: float = 1e-5, *, site_occupancy: int = 1
) -> dict[str, Any]:
    if type(site_occupancy) is not int or site_occupancy not in (1, 2, 3):
        raise ValueError("authored surface example supports integer 1/2/3-site molecules")

    def species(
        name: str, composition: dict[str, int], h: float, s: float, cp: float
    ) -> dict[str, Any]:
        return {
            "name": name,
            "composition": composition,
            "thermo": {
                "model": "constant-cp",
                "T0": 298.15,
                "h0": h,
                "s0": s,
                "cp0": cp,
                "T-min": 300,
                "T-max": 2000,
            },
        }

    declaration: dict[str, Any] = {
        "units": {"length": "m", "quantity": "mol", "activation-energy": "J/mol"},
        "phases": [
            {
                "name": "gas",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "species": ["A", "B"],
                "reactions": "none",
            },
            {
                "name": "surface",
                "thermo": "ideal-surface",
                "kinetics": "surface",
                "adjacent-phases": ["gas"],
                "species": ["Pt", "HA", "HB"],
                "site-density": density_mol_m2,
                "reactions": "all",
            },
        ],
        "species": [
            species("A", {"H": 2}, 0, 120, 30),
            species("B", {"H": 2}, -2000, 120, 30),
            species("Pt", {"Pt": 1}, 0, 0, 0),
            species("HA", {"H": 2, "Pt": 1}, -30000, 60, 15),
            species("HB", {"H": 2, "Pt": 1}, -32000, 60, 15),
        ],
        "reactions": [
            {
                "equation": "A + Pt <=> HA",
                "sticking-coefficient": {"A": 1e-7, "b": 0, "Ea": 0},
                "Motz-Wise": True,
                "coverage-dependencies": {"Pt": {"a": 0.12, "m": 0.1, "E": 500}},
            },
            {
                "equation": "HA <=> HB",
                "rate-constant": {"A": 4, "b": 0.1, "Ea": 1000},
                "coverage-dependencies": {"HA": {"a": 0.2, "m": 0, "E": -500}},
            },
            {
                "equation": "HB => B + Pt",
                "rate-constant": {"A": 6, "b": 0, "Ea": 2000},
                "orders": {"HB": 1.2},
            },
        ],
    }
    if site_occupancy != 1:
        for entry in declaration["species"]:
            if entry["name"] in ("HA", "HB"):
                entry["composition"]["Pt"] = site_occupancy
                entry["sites"] = site_occupancy
        extra = species("HAA", {"H": 4, "Pt": site_occupancy}, -45000, 90, 30)
        extra["sites"] = site_occupancy
        declaration["species"].append(extra)
        declaration["phases"][1]["species"].append("HAA")
        declaration["reactions"][0]["equation"] = f"A + {site_occupancy} Pt <=> HA"
        declaration["reactions"][2]["equation"] = f"HB => B + {site_occupancy} Pt"
        declaration["reactions"].append(
            {
                "equation": "A + HA <=> HAA",
                "sticking-coefficient": {"A": 1e-8, "b": 0, "Ea": 0},
                "coverage-dependencies": {"HAA": {"a": 0.05, "m": 0, "E": 250}},
            }
        )
    return declaration


def example_world(
    *,
    temperature_K: float = 700.0,
    density_mol_m2: float = 1e-5,
    isothermal: bool = True,
    site_occupancy: int = 1,
) -> WorldCompositionSpec:
    bundle = convert_surface_mechanism(
        authored_declaration(density_mol_m2, site_occupancy=site_occupancy),
        surface_phase="surface",
        source=SOURCE,
        minimum_temperature_K=300,
        maximum_temperature_K=1900,
    )
    from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R

    total = 1e5 * 0.001 / (R * temperature_K)
    capacity = 0.1 * density_mol_m2
    coverages = (
        {"Pt": 0.8, "HA": 0.1, "HB": 0.1}
        if site_occupancy == 1
        else {"Pt": 0.55, "HA": 0.2, "HB": 0.15, "HAA": 0.1}
    )
    scales = bundle.surface.coverage_amount_scales_mol({"surface": capacity})
    return descriptor_catalysis_world(
        bundle.surface,
        {"A": 0.7 * total, "B": 0.3 * total},
        capacities_mol={"surface": capacity},
        initial_adsorbates_mol={k: theta * scales[k] for k, theta in coverages.items()},
        volume_L=1,
        temperature_K=temperature_K,
        support_heat_capacity_J_K=1,
        stop_temperature_K=1800,
        maximum_pressure_Pa=1e6,
        isothermal=isothermal,
        gas_network=bundle.gas_network,
    )


def run_example() -> Any:
    from chemworld.runtime import ChemWorldRuntime

    world = ChemWorldRuntime(composition=example_world())
    try:
        world.reset(seed=0)
        return world.execute({"operation": "wait", "duration_s": 1})
    finally:
        world.close()
