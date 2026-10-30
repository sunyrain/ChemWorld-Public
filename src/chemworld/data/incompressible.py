"""Explicit INCOMP parameter import and bundled, pinned engineering catalog.

Catalog identifiers are correlation identities, not chemical component IDs.
Volume-fraction fits and undocumented commercial blends are never silently
converted to mass fractions or assigned a molecular formula/CAS.
"""

import json
from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files
from typing import Any

from chemworld.physchem.incompressible import IncompressibleFluid

SOURCE_COMMIT = "ae81610e7d23efc57f9d051c8e70a4d66e87537f"


def load_incompressible_fluid(data: Mapping[str, Any], *, source: str) -> IncompressibleFluid:
    """Import a source-style parameter card with explicit provenance, no backend."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("incompressible import requires a source")
    correlations: dict[str, dict[str, Any] | None] = {}
    for name in ("density", "specific_heat", "viscosity", "conductivity",
                 "saturation_pressure", "T_freeze", "mass2input", "mole2input", "volume2input"):
        fit = data.get(name)
        if fit is None or fit["type"] == "notdefined":
            correlations[name] = None
            continue
        values = fit["coeffs"]
        if fit["type"] in {"exponential", "logexponential", "polyoffset"}:
            values = [[v] for v in values]
        correlations[name] = {"family": fit["type"], "coefficients": values}
    return IncompressibleFluid.from_dict({
        "model": "incompressible-fluid-v1", "fluid_id": data["name"],
        "description": data["description"], "citation": data["reference"], "source": source,
        "fraction_basis": data["xid"], "temperature_bounds_K": [data["Tmin"], data["Tmax"]],
        "fraction_bounds": [data.get("xmin", 0.0), data.get("xmax", 1.0)],
        "temperature_center_K": data.get("Tbase", 0.0),
        "fraction_center": data.get("xbase", 0.0),
        "minimum_saturation_temperature_K": data.get("TminPsat", 0.0),
        "correlations": correlations})


@lru_cache(maxsize=1)
def _catalog() -> tuple[IncompressibleFluid, ...]:
    payload = json.loads(files("chemworld.data").joinpath(
        "parameters/incompressible.json").read_text(encoding="utf-8"))
    if payload["schema"] != "incompressible-parameter-catalog-v1":
        raise ValueError("unknown incompressible catalog schema")
    fluids = tuple(IncompressibleFluid.from_dict(card) for card in payload["cards"])
    if len({fluid.fluid_id for fluid in fluids}) != len(fluids):
        raise ValueError("duplicate incompressible catalog identity")
    return fluids


def incompressible_catalog() -> tuple[IncompressibleFluid, ...]:
    """All immutable cards, including example/non-liquid/unidentified entries."""
    return _catalog()


def incompressible_fluid(name: str) -> IncompressibleFluid:
    for fluid in _catalog():
        if fluid.fluid_id == name:
            return fluid
    raise ValueError(f"unknown incompressible correlation identity: {name}")
