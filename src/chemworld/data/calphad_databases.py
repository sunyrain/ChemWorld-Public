"""Fixed, declarative CALPHAD input codecs; no class names or dynamic imports in JSON."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from chemworld.data.tdb import TDB_SCHEMA_VERSION, TdbDatabase

if TYPE_CHECKING:
    from chemworld.data.chem_sage import ChemSageDatabase


def calphad_database_from_dict(payload: object) -> TdbDatabase | ChemSageDatabase:
    if not isinstance(payload, Mapping):
        raise ValueError("CALPHAD input must be a declarative object")
    if payload.get("schema_version") == TDB_SCHEMA_VERSION:
        return TdbDatabase.from_dict(payload)
    if payload.get("schema_version") == "chemworld-chemsage-1":
        from chemworld.data.chem_sage import ChemSageDatabase

        return ChemSageDatabase.from_dict(payload)
    raise ValueError("unknown native CALPHAD database format/schema")
