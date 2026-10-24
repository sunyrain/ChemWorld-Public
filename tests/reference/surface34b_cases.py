"""Fixed source declarations and independent Cantera reference-side adapters."""

import json
from itertools import product

import cantera as ct

from chemworld.examples.native_imported_surface import authored_declaration

TEMPERATURES = (400.0, 700.0, 1000.0)
DENSITIES = (1e-5, 3e-5)
COVERAGES = ((0.8, 0.1, 0.1), (0.4, 0.3, 0.3), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8))
TIMES = (0.0, 0.025, 0.1, 0.5, 1.0)


def plain(value):
    if hasattr(value, "items"):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def source_payload():
    original = ct.Interface("ptcombust.yaml", "Pt_surf")
    gas = original.adjacent["gas"]
    # Explicit reference-side surface-only normalization: no homogeneous gas
    # reactions and no transport model. All 24 source surface reactions and all
    # 43 thermochemical species are retained through Cantera's public data API.
    species = [plain(s.input_data) for s in (*gas.species(), *original.species())]
    for item in species:
        item.pop("transport", None)
    return {
        "units": {"length": "m", "quantity": "kmol", "activation-energy": "J/kmol"},
        "phases": [
            {
                "name": "gas",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "species": gas.species_names,
                "reactions": "none",
            },
            {
                "name": "Pt_surf",
                "thermo": "ideal-surface",
                "kinetics": "surface",
                "adjacent-phases": ["gas"],
                "species": original.species_names,
                "site-density": original.site_density,
                "reactions": "all",
            },
        ],
        "species": species,
        "reactions": [plain(r.input_data) for r in original.reactions()],
    }


def reference(payload, temperature, pressure, coverages):
    name = payload["phases"][1]["name"]
    surface = ct.Interface(yaml=json.dumps(payload), name=name)
    gas = surface.adjacent["gas"]
    fractions = [0.7, 0.3] if gas.n_species == 2 else [1 / gas.n_species] * gas.n_species
    gas.TPX = temperature, pressure, fractions
    surface.TP = temperature, pressure
    surface.coverages = coverages
    return gas, surface


def cases():
    rows = []
    for i, (t, coverage, density) in enumerate(product(TEMPERATURES, COVERAGES, DENSITIES)):
        rows.append(
            {
                "name": f"authored-state-{i:02}",
                "kind": "state",
                "source": "authored",
                "temperature": t,
                "pressure": 1e5,
                "coverage": coverage,
                "density": density,
                "expected_comparisons": 9,
            }
        )
    for i, (t, coverage, p) in enumerate(product((700.0, 1000.0, 1500.0), range(3), (1e4, 1e5))):
        values = [1 / 11] * 11
        if coverage:
            values = [0.02] * 11
            values[0 if coverage == 1 else 10] = 0.8
        rows.append(
            {
                "name": f"ptcombust-state-{i:02}",
                "kind": "state",
                "source": "ptcombust",
                "temperature": t,
                "pressure": p,
                "coverage": values,
                "expected_comparisons": 9,
            }
        )
    for i, (t, density) in enumerate(product(TEMPERATURES, DENSITIES)):
        rows.append(
            {
                "name": f"finite-history-{i}",
                "kind": "history",
                "temperature": t,
                "density": density,
                "expected_comparisons": 25,
            }
        )
    for kind in ("isothermal", "adiabatic", "snapshot", "rollback"):
        rows.append({"name": "world-" + kind, "kind": kind, "expected_comparisons": 6})
    return rows


def payload_for(case, source):
    return (
        authored_declaration(case.get("density", 1e-5))
        if case.get("source", "authored") == "authored"
        else source
    )
