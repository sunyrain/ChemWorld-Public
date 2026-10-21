"""Provenanced original ChemSage DAT input and common native Gibbs declarations."""

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from chemworld.data.chem_sage_compiler import compile_chem_sage
from chemworld.data.chem_sage_parser import ChemSageParseError, parse_chem_sage
from chemworld.data.chem_sage_records import ChemSageInput
from chemworld.data.scientific_inputs import InputSource, _keys
from chemworld.data.tdb import TdbDatabase

CHEMSAGE_SCHEMA_VERSION = "chemworld-chemsage-1"


@dataclass(frozen=True)
class ChemSageDatabase(TdbDatabase):
    declarations: ChemSageInput
    excess_pressure_unit: str | None = None

    @classmethod
    def from_string(cls, text, *, source, excess_pressure_unit=None):
        if not isinstance(source, InputSource):
            raise ValueError("ChemSage requires the existing explicit InputSource provenance")
        if excess_pressure_unit not in {None, "Pa", "bar"}:
            raise ValueError("ChemSage excess pressure unit must be explicitly Pa or bar")
        declarations = parse_chem_sage(text)
        compiled = compile_chem_sage(declarations, excess_pressure_unit=excess_pressure_unit)
        return cls(source, text, *compiled, declarations, excess_pressure_unit)

    @classmethod
    def from_file(cls, path, *, source, excess_pressure_unit=None):
        path = Path(path)
        if path.stat().st_size > 20*1024*1024:
            raise ChemSageParseError("ChemSage exceeds 20 MiB")
        return cls.from_string(path.read_text(encoding="utf-8-sig"), source=source,
                               excess_pressure_unit=excess_pressure_unit)

    def to_dict(self):
        return {"schema_version": CHEMSAGE_SCHEMA_VERSION, "source": self.source.to_dict(),
                "text": self.text, "excess_pressure_unit": self.excess_pressure_unit}

    @classmethod
    def from_dict(cls, payload):
        _keys(payload, ("schema_version", "source", "text", "excess_pressure_unit"))
        if payload["schema_version"] != CHEMSAGE_SCHEMA_VERSION:
            raise ValueError("unsupported ChemSage schema version")
        return cls.from_string(payload["text"], source=InputSource.from_dict(payload["source"]),
                               excess_pressure_unit=payload["excess_pressure_unit"])

    @property
    def phase_records(self):
        return MappingProxyType(dict(zip(self.phases, self.declarations.phases, strict=True)))
