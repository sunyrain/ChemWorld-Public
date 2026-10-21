"""Fixed, declarative CALPHAD input codecs; no class names or dynamic imports in JSON."""

from collections.abc import Mapping

from chemworld.data.tdb import TDB_SCHEMA_VERSION, TdbDatabase


def calphad_database_from_dict(payload):
    if not isinstance(payload, Mapping):
        raise ValueError("CALPHAD input must be a declarative object")
    if payload.get("schema_version") == TDB_SCHEMA_VERSION:
        return TdbDatabase.from_dict(payload)
    if payload.get("schema_version") == "chemworld-chemsage-1":
        from chemworld.data.chem_sage import ChemSageDatabase

        return ChemSageDatabase.from_dict(payload)
    raise ValueError("unknown native CALPHAD database format/schema")
