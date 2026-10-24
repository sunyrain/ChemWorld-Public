"""Explicit local-source multifluid import; never downloads or estimates pairs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chemworld.data.helmholtz import load_helmholtz_fluid
from chemworld.physchem.multifluid import (
    MultifluidBinaryPair,
    MultifluidDepartureTerm,
    MultifluidModel,
)


def _departure(data: Mapping[str, Any]) -> tuple[MultifluidDepartureTerm, ...]:
    kind = data.get("type")
    metadata = {"Name", "BibTeX", "aliases", "type"}
    if kind == "none":
        if set(data) - metadata:
            raise ValueError("unknown null departure fields")
        return ()
    if kind not in {"Exponential", "GERG-2004", "GERG-2008"}:
        raise ValueError(f"unsupported departure source type: {kind}")
    required = {"n", "d", "t"}
    if kind == "Exponential":
        required.add("l")
    allowed = metadata | required | {"l"}
    if kind.startswith("GERG"):
        required |= {"Npower", "eta", "epsilon", "beta", "gamma"}
        allowed |= required
    if required - set(data) or set(data) - allowed:
        raise ValueError("missing or unknown departure source fields")
    array_keys = required - {"Npower"}
    if "l" in data:
        array_keys.add("l")
    if any(not isinstance(data[k], (list, tuple)) or not data[k] for k in array_keys):
        raise ValueError("departure coefficient arrays must be nonempty")
    count = len(data["n"])
    if any(len(data[k]) != count for k in array_keys):
        raise ValueError("departure arrays must have equal lengths")
    npower = data.get("Npower", count)
    if type(npower) is not int or not 0 <= npower <= count:
        raise ValueError("invalid departure Npower")
    terms = []
    for i in range(count):
        parameters = {k: data[k][i] for k in ("n", "d", "t")}
        if i < npower:
            parameters["l"] = data["l"][i] if "l" in data else 0.0
            terms.append(MultifluidDepartureTerm("power", **parameters))
        else:
            if "l" in data and data["l"][i] != 0:
                raise ValueError("GERG tail cannot include power damping")
            parameters.update({k: data[k][i] for k in ("eta", "epsilon", "beta", "gamma")})
            terms.append(MultifluidDepartureTerm("gerg", **parameters))
    return tuple(terms)


def load_multifluid_model(
    data: Mapping[str, Any],
    *,
    source: str,
    eos_indexes: Sequence[int | None] | None = None,
) -> MultifluidModel:
    """Import explicit pure JSON, exactly selected binaries and departures.

    Fields: fluids, binary_pairs, departure_functions, gas_constant_J_mol_K,
    gas_constant_convention. Binary identities must match the pure source names
    AND CAS identifiers. Catalog selection/ambiguous identities belong to the
    caller; no fallback to approximate mixing parameters occurs here.
    """
    expected = {
        "fluids",
        "binary_pairs",
        "departure_functions",
        "gas_constant_J_mol_K",
        "gas_constant_convention",
    }
    if not isinstance(data, Mapping) or set(data) != expected:
        raise ValueError(f"multifluid source requires exactly {sorted(expected)}")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("multifluid import requires source")
    nodes = data["fluids"]
    if not isinstance(nodes, (list, tuple)) or not nodes:
        raise ValueError("multifluid source fluids must be a nonempty sequence")
    indexes = tuple(eos_indexes) if eos_indexes is not None else (None,) * len(nodes)
    if len(indexes) != len(nodes):
        raise ValueError("EOS index count differs from pure-fluid count")
    fluids = tuple(
        load_helmholtz_fluid(node, source=source, eos_index=index)
        for node, index in zip(nodes, indexes, strict=True)
    )
    by_name = {f.fluid_id: f for f in fluids}
    departures = {}
    for node in data["departure_functions"]:
        name = node.get("Name")
        if not isinstance(name, str) or not name or name in departures:
            raise ValueError("departure sources require unique explicit names")
        departures[name] = node
    used_departures = set()
    pairs = []
    required = {"Name1", "Name2", "CAS1", "CAS2", "betaT", "gammaT", "betaV", "gammaV", "F"}
    allowed = required | {"BibTeX", "function"}
    for node in data["binary_pairs"]:
        if not isinstance(node, Mapping) or required - set(node) or set(node) - allowed:
            raise ValueError("missing or unsupported binary source fields")
        ids = (node["Name1"], node["Name2"])
        if any(k not in by_name for k in ids) or any(
            by_name[k].casrn != node[f"CAS{i + 1}"] for i, k in enumerate(ids)
        ):
            raise ValueError("binary source name/CAS identity does not match pure declaration")
        name = node.get("function", "")
        if name and name not in departures:
            raise ValueError("binary source references a missing departure function")
        if name:
            used_departures.add(name)
        terms = _departure(departures[name]) if name else ()
        provenance = source + "; " + str(node.get("BibTeX", ""))
        if name:
            provenance += "; departure=" + name + "; " + str(departures[name].get("BibTeX", ""))
        pairs.append(
            MultifluidBinaryPair(
                ids,
                node["betaT"],
                node["gammaT"],
                node["betaV"],
                node["gammaV"],
                node["F"],
                terms,
                provenance,
            )
        )
    if used_departures != set(departures):
        raise ValueError("unselected departure sources are not part of this explicit model")
    return MultifluidModel(
        fluids, tuple(pairs), data["gas_constant_J_mol_K"], data["gas_constant_convention"], source
    )


def read_multifluid_model(
    path: str | Path, *, source: str, eos_indexes: Sequence[int | None] | None = None
) -> MultifluidModel:
    """Read one bounded, explicit JSON input; no directory/database discovery."""
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("multifluid source exceeds 32 MiB limit")
    return load_multifluid_model(
        json.loads(path.read_text(encoding="utf-8")), source=source, eos_indexes=eos_indexes
    )
