"""Provenanced original ChemSage DAT input and common native Gibbs declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType
from typing import Any

from chemworld.data.chem_sage_compiler import compile_chem_sage
from chemworld.data.chem_sage_parser import ChemSageParseError, parse_chem_sage
from chemworld.data.chem_sage_records import ChemSageInput, ChemSagePhase
from chemworld.data.scientific_inputs import InputSource, _keys
from chemworld.data.tdb import TdbDatabase

CHEMSAGE_SCHEMA_VERSION = "chemworld-chemsage-1"


@dataclass(frozen=True)
class ChemSageDatabase(TdbDatabase):
    declarations: ChemSageInput
    excess_pressure_unit: str | None = None

    @classmethod
    def from_string(
        cls, text: str, *, source: InputSource, excess_pressure_unit: str | None = None,
    ) -> ChemSageDatabase:
        if not isinstance(source, InputSource):
            raise ValueError("ChemSage requires the existing explicit InputSource provenance")
        if excess_pressure_unit not in {None, "Pa", "bar"}:
            raise ValueError("ChemSage excess pressure unit must be explicitly Pa or bar")
        declarations = parse_chem_sage(text)
        compiled = compile_chem_sage(declarations, excess_pressure_unit=excess_pressure_unit)
        return cls(source, text, *compiled, declarations, excess_pressure_unit)

    @classmethod
    def from_file(
        cls, path: str | PathLike[str], *, source: InputSource,
        excess_pressure_unit: str | None = None,
    ) -> ChemSageDatabase:
        file_path = Path(path)
        if file_path.stat().st_size > 20*1024*1024:
            raise ChemSageParseError("ChemSage exceeds 20 MiB")
        return cls.from_string(file_path.read_text(encoding="utf-8-sig"), source=source,
                               excess_pressure_unit=excess_pressure_unit)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": CHEMSAGE_SCHEMA_VERSION, "source": self.source.to_dict(),
                "text": self.text, "excess_pressure_unit": self.excess_pressure_unit}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChemSageDatabase:
        _keys(payload, ("schema_version", "source", "text", "excess_pressure_unit"))
        if payload["schema_version"] != CHEMSAGE_SCHEMA_VERSION:
            raise ValueError("unsupported ChemSage schema version")
        return cls.from_string(payload["text"], source=InputSource.from_dict(payload["source"]),
                               excess_pressure_unit=payload["excess_pressure_unit"])

    @property
    def phase_records(self) -> Mapping[str, ChemSagePhase]:
        return MappingProxyType(dict(zip(self.phases, self.declarations.phases, strict=True)))
