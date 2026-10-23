"""Explicit import of public pure-fluid Helmholtz coefficient declarations.

The caller supplies the JSON and provenance. Importing this module does not
read external files, discover a reference installation, or call CoolProp.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm


def load_helmholtz_fluid(data: Mapping[str, Any], *, source: str,
                         eos_index: int | None = None) -> HelmholtzFluid:
    """Import a public CoolProp-style JSON mapping, with explicit multi-EOS choice."""
    if not isinstance(data, Mapping) or not isinstance(source, str) or not source.strip():
        raise ValueError("Helmholtz import requires a JSON mapping and source")
    equations = data.get("EOS")
    if not isinstance(equations, list) or not equations:
        raise ValueError("Helmholtz JSON requires a nonempty EOS list")
    if eos_index is None:
        if len(equations) != 1:
            raise ValueError("multiple Helmholtz equations require an explicit eos_index")
        eos_index = 0
    if type(eos_index) is not int or not 0 <= eos_index < len(equations):
        raise ValueError("invalid Helmholtz EOS index")
    info, eos = data.get("INFO", {}), equations[eos_index]
    try:
        states = eos["STATES"]
        reducing = states["reducing"]
        minimum = states["sat_min_liquid"]["T"]
        expected_units = {"gas_constant_units": "J/mol/K", "molar_mass_units": "kg/mol",
                          "T_max_units": "K", "p_max_units": "Pa"}
        for key, expected in expected_units.items():
            if eos.get(key, expected) != expected:
                raise ValueError(f"Helmholtz {key} must be {expected}")
        for key, expected in {"T_units": "K", "rhomolar_units": "mol/m^3"}.items():
            if reducing.get(key, expected) != expected:
                raise ValueError(f"Helmholtz reducing {key} must be {expected}")
        if states["sat_min_liquid"].get("T_units", "K") != "K":
            raise ValueError("Helmholtz minimum temperature must use K")
        return HelmholtzFluid(
            fluid_id=info["NAME"], casrn=info.get("CAS", ""),
            molar_mass_kg_mol=eos["molar_mass"], gas_constant_J_mol_K=eos["gas_constant"],
            reducing_temperature_K=reducing["T"],
            reducing_molar_density_mol_m3=reducing["rhomolar"],
            minimum_temperature_K=minimum, maximum_temperature_K=eos["T_max"],
            maximum_pressure_Pa=eos["p_max"],
            ideal_terms=tuple(HelmholtzTerm.from_dict(t) for t in eos["alpha0"]),
            residual_terms=tuple(HelmholtzTerm.from_dict(t) for t in eos["alphar"]),
            source=source, eos_index=eos_index, pseudo_pure=eos["pseudo_pure"],
            ideal_prefactor=eos.get("alpha0_prefactor", 1.0),
            citation="; ".join(str(eos[k]) for k in ("BibTeX_EOS", "BibTeX_CP0") if eos.get(k)),
        )
    except KeyError as exc:
        raise ValueError(f"incomplete Helmholtz declaration: {exc.args[0]}") from exc


def read_helmholtz_fluid(path: str | Path, *, source: str,
                         eos_index: int | None = None) -> HelmholtzFluid:
    """Read one explicit JSON object (unwrap API-returned arrays before import)."""
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Helmholtz input exceeds the 16 MiB declaration limit")
    with path.open(encoding="utf-8") as stream:
        return load_helmholtz_fluid(json.load(stream), source=source, eos_index=eos_index)
