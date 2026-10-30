"""Bounded, declarative ChemSage token reader with original record ordering.

The reader recognizes records; physical consumers decide whether a declared
option has an implemented equation. No source text is executed or imported.
"""

import re
from collections.abc import Callable
from dataclasses import replace
from itertools import pairwise
from math import isfinite, prod
from typing import TypeVar

from chemworld.data.chem_sage_records import (
    ChemSageEndmember,
    ChemSageHeader,
    ChemSageInput,
    ChemSageInteraction,
    ChemSageInterval,
    ChemSageMQMQA,
    ChemSagePhase,
)

LIMIT = 100000
_T = TypeVar("_T")


class ChemSageParseError(ValueError):
    """No partially usable database is returned for an invalid declaration."""


class _Reader:
    def __init__(self, text: str) -> None:
        if not isinstance(text, str) or not text.strip() or len(text) > 20*1024*1024:
            raise ChemSageParseError("ChemSage requires nonempty text up to 20 MiB")
        lines = text.splitlines()
        self.title = lines[0]
        self.lines = lines
        self.tokens: list[tuple[str, int, int]] = []
        self.status: dict[int, str] = {}
        for line_number, original in enumerate(lines[1:], 2):
            line = original
            if len(line) > 25 and line[25] in "!#":
                self.status[line_number] = "dormant" if line[25] == "!" else "eliminated"
                line = line[:25]+" "+line[26:]
            self.tokens.extend((m[0].upper(), line_number, m.start())
                               for m in re.finditer(r"\S+", line))
        if len(self.tokens) > 2000000:
            raise ChemSageParseError("ChemSage exceeds two million tokens")
        self.position = 0

    def word(self) -> str:
        if self.position == len(self.tokens):
            raise ChemSageParseError("truncated ChemSage declaration")
        value, _, _ = self.tokens[self.position]
        self.position += 1
        return value

    def integer(self) -> int:
        value = self.word()
        if len(value) > 12 or not re.fullmatch(r"[+-]?\d+", value):
            raise ChemSageParseError(f"expected bounded ChemSage integer, got {value[:80]}")
        result = int(value)
        if abs(result) > LIMIT:
            raise ChemSageParseError("ChemSage integer/count exceeds supported bounds")
        return result

    def count(self, *, minimum: int = 0) -> int:
        value = self.integer()
        if value < minimum:
            raise ChemSageParseError("ChemSage count has an invalid sign or is empty")
        return value

    def real(self) -> float:
        value = self.word()
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[ED][+-]?\d+)?", value):
            raise ChemSageParseError(f"expected ChemSage real number, got {value[:80]}")
        result = float(value.replace("D", "E"))
        if not isfinite(result):
            raise ChemSageParseError("ChemSage coefficients must be finite")
        return result

    def many(self, size: int, method: Callable[[], _T]) -> tuple[_T, ...]:
        if not 0 <= size <= LIMIT:
            raise ChemSageParseError("ChemSage array exceeds supported bounds")
        return tuple(method() for _ in range(size))

    def name(self) -> tuple[str, str]:
        if self.position == len(self.tokens):
            raise ChemSageParseError("missing ChemSage name")
        line = self.tokens[self.position][1]
        name = self.word()
        if len(name) > 128 or not re.fullmatch(r"[A-Z0-9_+*/.\-():\[\]#]+", name):
            raise ChemSageParseError(f"invalid ChemSage identifier {name[:80]}")
        return name, self.status.get(line, "entered")

    def peek(self) -> str | None:
        return None if self.position == len(self.tokens) else self.tokens[self.position][0]


def _interval(
    reader: _Reader, header: ChemSageHeader, option: int, *, transition: bool = False,
) -> ChemSageInterval:
    reduced = (option-1) % 12+1
    heat = reduced >= 7
    latent = reader.real() if transition else 0.0
    upper = reader.real()
    coefficients = reader.many(4 if heat else len(header.gibbs_indices), reader.real)
    extended = reduced in {4, 5, 6, 10, 11, 12}
    extra = tuple((reader.real(), reader.real()) for _ in range(reader.count())) if extended else ()
    pressure = reader.many(11, reader.real) if reduced in {3, 6, 9, 12} else ()
    if upper <= 0:
        raise ChemSageParseError("ChemSage temperature limits must be positive")
    return ChemSageInterval(upper, coefficients, extra, pressure, latent)


def _endmember(
    reader: _Reader, header: ChemSageHeader, *, family: str = "CEF", zeta: float | None = None,
) -> ChemSageEndmember:
    name, status = reader.name()
    if reader.peek() == "#":
        reader.word()  # Authored dummy marker, retained in the original text.
    # Some sources include two zero-valued legacy fields after a species name.
    if ((next_word := reader.peek()) is not None and not re.fullmatch(r"[+-]?\d+", next_word)
            and (reader.real() != 0 or reader.real() != 0)):
        raise ChemSageParseError("nonzero legacy species fields lack a declared meaning")
    option = reader.integer()
    if not 1 <= option <= 24:
        raise ChemSageParseError("ChemSage thermodynamic option must be 1 through 24")
    reduced = (option-1) % 12+1
    if reduced in {2, 5, 8, 11}:
        raise ChemSageParseError("constant-volume layout is not implemented; no partial input")
    interval_count = reader.count(minimum=1)
    stoichiometry = reader.many(len(header.components)+(family == "AQUEOUS"), reader.real)
    charge = stoichiometry[0] if family == "AQUEOUS" else None
    if charge is not None:
        stoichiometry = stoichiometry[1:]
    h298 = s298 = None
    if reduced >= 7:
        if header.gibbs_indices != (1, 2, 3, 4, 5, 6):
            raise ChemSageParseError("Cp records require the six standard Gibbs basis indices")
        h298, s298 = reader.real(), reader.real()
    intervals = tuple(_interval(reader, header, option, transition=reduced >= 7 and i > 0)
                      for i in range(interval_count))
    if any(b.upper_K < a.upper_K for a, b in pairwise(intervals)):
        raise ChemSageParseError("ChemSage intervals must follow increasing temperature")
    tc, moment = (reader.real(), reader.real()) if option > 12 else (None, None)
    result = ChemSageEndmember(name, status, option, stoichiometry, intervals, h298, s298,
                              tc, moment, aqueous_charge=charge)
    if family == "QKTO":
        result = replace(result, stoichiometric_factor=reader.real(),
                         chemical_group=reader.integer())
    if family in {"SUBQ", "SUBG"}:
        result = replace(result, pair_stoichiometry=reader.many(5, reader.real),
                         zeta=reader.real() if zeta is None else zeta)
    if family == "IDVD":
        result = replace(result, real_gas_parameters=reader.many(5, reader.real))
    return result


def _excess(
    reader: _Reader, header: ChemSageHeader, *, family: str = "RKM",
) -> tuple[tuple[ChemSageInteraction, ...], tuple[tuple[str, ...], ...]]:
    records: list[ChemSageInteraction] = []
    overrides: tuple[tuple[str, ...], ...] = ()
    while True:
        size = reader.integer()
        if size == 0:
            break
        if size < 0:
            if family != "QKT":
                raise ChemSageParseError("negative excess count only defines QKTO group overrides")
            overrides = tuple(reader.many(10, reader.word) for _ in range(-size))
            break
        indices = reader.many(size, reader.integer)
        if family == "QKT":
            exponents = reader.many(size, reader.integer)
            records.append(ChemSageInteraction("QKT", indices, reader.many(
                len(header.excess_indices), reader.real), exponents=exponents))
        elif family == "PITZ":
            # Pitzer's three index fields also carry an unused third field in
            # binary records. Preserve it; the aqueous consumer owns its meaning.
            extra = reader.many(3-size, reader.integer)
            records.append(ChemSageInteraction("PITZ", indices, reader.many(
                len(header.excess_indices), reader.real), metadata=extra))
        else:
            for order in range(reader.count()):
                coefficients = reader.many(2 if family == "MAGNETIC" else
                                           len(header.excess_indices), reader.real)
                records.append(ChemSageInteraction(family, indices, coefficients, order))
        if len(records) > LIMIT:
            raise ChemSageParseError("too many ChemSage excess parameters")
    return tuple(records), overrides


def _mqmqa(
    reader: _Reader, header: ChemSageHeader, name: str, status: str,
    family: str, size: int, gas_slot: bool,
) -> ChemSagePhase:
    zeta = reader.real() if family == "SUBG" else None
    pair_count, coordination_count = reader.count(minimum=1), reader.count()
    members = tuple(_endmember(reader, header, family=family, zeta=zeta)
                    for _ in range(pair_count))
    nc, na = reader.count(minimum=1), reader.count(minimum=1)
    cations, anions = reader.many(nc, reader.word), reader.many(na, reader.word)
    charges_c, groups_c = reader.many(nc, reader.real), reader.many(nc, reader.integer)
    charges_a, groups_a = reader.many(na, reader.real), reader.many(na, reader.integer)
    if nc*na != pair_count or nc*(nc+1)*na*(na+1)//4 != size:
        raise ChemSageParseError("MQMQA pair/coordinate counts disagree with the authored lattices")
    indices_c, indices_a = reader.many(pair_count, reader.integer), reader.many(
        pair_count, reader.integer)
    pairs = tuple(zip(indices_c, indices_a, strict=True))
    if (set(pairs) != {(i+1, j+1) for i in range(nc) for j in range(na)}
            or len(set(zip(cations, charges_c, strict=True))) != nc
            or len(set(zip(anions, charges_a, strict=True))) != na):
        raise ChemSageParseError("MQMQA needs unique charged species and complete pair indices")
    coordinations = tuple((reader.many(4, reader.integer), reader.many(4, reader.real))
                         for _ in range(coordination_count))
    records: list[ChemSageInteraction] = []
    overrides: tuple[tuple[str, ...], ...] = ()
    while True:
        mixing_type = reader.integer()
        if mixing_type == 0:
            break
        if mixing_type < 0:
            overrides = tuple(reader.many(10, reader.word) for _ in range(-mixing_type))
            break
        code = reader.word()
        indices, exponents = reader.many(4, reader.integer), reader.many(4, reader.integer)
        metadata = reader.many(12, reader.real)
        additional, additional_anion = reader.integer(), reader.integer()
        coefficients = reader.many(len(header.excess_indices), reader.real)
        records.append(ChemSageInteraction("MQMX", indices, coefficients, exponents=exponents,
                       mixing_type=mixing_type, mixing_code=code, metadata=metadata,
                       additional_constituent=additional,
                       additional_anion_constituent=additional_anion))
        if len(records) > LIMIT:
            raise ChemSageParseError("too many MQMQA parameters")
    mqmqa = ChemSageMQMQA(cations, anions, charges_c, charges_a, groups_c, groups_a, pairs,
                         coordinations)
    return ChemSagePhase(name, family, status, gas_slot, size, members,
                         interactions=tuple(records), overrides=overrides, mqmqa=mqmqa)


def _phase(reader: _Reader, header: ChemSageHeader, size: int, index: int) -> ChemSagePhase:
    name, status = reader.name()
    family = reader.word()
    if family in {"SUBQ", "SUBG"}:
        return _mqmqa(reader, header, name, status, family, size, index == 0)
    if family not in {"IDMX", "RKMP", "RKMPM", "SUBL", "SUBLM", "QKTO", "IDVD", "IDWZ",
                      "PITZ"}:
        raise ChemSageParseError(f"unknown or unsupported ChemSage phase family {family}")
    afm, structure = ((reader.real(), reader.real()) if family in {"RKMPM", "SUBLM"}
                      else (None, None))
    base = family[:-1] if family in {"RKMPM", "SUBLM"} else family
    member_family = "AQUEOUS" if base in {"IDWZ", "PITZ"} else base
    members = tuple(_endmember(reader, header, family=member_family) for _ in range(size))
    ratios: tuple[float, ...] = (1.0,)
    constituents: tuple[tuple[str, ...], ...] = ()
    indices: tuple[tuple[int, ...], ...] = ()
    if base == "SUBL":
        count = reader.count(minimum=1)
        ratios = reader.many(count, reader.real)
        if ":" in name:
            _, atom_count_text = name.rsplit(":", 1)
            try:
                atom_count = float(atom_count_text)
            except ValueError as exc:
                raise ChemSageParseError("invalid sublattice atom-count suffix") from exc
            if not isfinite(atom_count) or atom_count <= 0:
                raise ChemSageParseError("sublattice atom count must be positive")
            ratios = tuple(r*atom_count for r in ratios)
        counts = reader.many(count, reader.count)
        if prod(counts) != size:
            raise ChemSageParseError("sublattice product disagrees with its declared size")
        constituents = tuple(reader.many(n, reader.word) for n in counts)
        index_rows = tuple(reader.many(size, reader.integer) for _ in range(count))
        indices = tuple(zip(*index_rows, strict=True))
        if any(not 1 <= value <= counts[i] for row in indices for i, value in enumerate(row)):
            raise ChemSageParseError("sublattice endmember index is out of bounds")
    interactions: tuple[ChemSageInteraction, ...] = ()
    overrides: tuple[tuple[str, ...], ...] = ()
    if base in {"RKMP", "SUBL"}:
        magnetic, _ = _excess(reader, header, family="MAGNETIC") if afm is not None else ((), ())
        regular, _ = _excess(reader, header)
        interactions = (*magnetic, *regular)
    elif base in {"QKTO", "PITZ"}:
        interactions, overrides = _excess(reader, header, family="QKT" if base == "QKTO" else base)
    return ChemSagePhase(name, family, status, index == 0, size, members, ratios, constituents,
                         indices, interactions, overrides, afm, structure)


def _parse_reader(reader: _Reader) -> ChemSageInput:
    components, solutions = reader.count(minimum=1), reader.count()
    sizes = reader.many(solutions, reader.count)
    stoichiometric = reader.count()
    names = reader.many(components, reader.word)
    masses = reader.many(components, reader.real)
    gibbs = reader.many(reader.count(minimum=1), reader.integer)
    excess = reader.many(reader.count(minimum=1), reader.integer)
    if (len(set(names)) != components or any(mass < 0 for mass in masses)
            or len(set(gibbs)) != len(gibbs) or set(gibbs) - set(range(1, 7))
            or len(set(excess)) != len(excess) or set(excess) - set(range(1, 9))
            or sum(sizes)+stoichiometric > LIMIT):
        raise ChemSageParseError("invalid ChemSage header identities, basis indices or counts")
    header = ChemSageHeader(reader.title, sizes, stoichiometric, names, masses, gibbs, excess)
    phases = [_phase(reader, header, size, i) for i, size in enumerate(sizes) if size]
    for _ in range(stoichiometric):
        member = _endmember(reader, header)
        afm, structure = (reader.real(), reader.real()) if member.option > 12 else (None, None)
        phases.append(ChemSagePhase(member.name, "PURE", member.status, False, 1, (member,),
                                    magnetic_afm_factor=afm, magnetic_structure_factor=structure))
    trailer = ""
    if reader.position < len(reader.tokens):
        _, line, offset = reader.tokens[reader.position]
        trailer = "\n".join([reader.lines[line-1][offset:], *reader.lines[line:]])
    return ChemSageInput(header, tuple(phases), trailer)


def parse_chem_sage(text: str) -> ChemSageInput:
    """Return immutable syntax, preserving every declared phase occurrence."""
    reader = _Reader(text)
    try:
        return _parse_reader(reader)
    except ChemSageParseError as exc:
        line = (reader.tokens[min(reader.position, len(reader.tokens)-1)][1]
                if reader.tokens else 1)
        raise ChemSageParseError(f"ChemSage near line {line}: {exc}") from exc
