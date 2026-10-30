"""Normalize validated ChemSage declarations to the common Gibbs data records."""

import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from types import MappingProxyType
from typing import cast

from chemworld.data.chem_sage_functions import basis_expression, endmember_function
from chemworld.data.chem_sage_records import (
    ChemSageEndmember,
    ChemSageHeader,
    ChemSageInput,
    ChemSagePhase,
)
from chemworld.data.tdb import (
    TdbFunction,
    TdbParameter,
    TdbPhase,
    TdbSegment,
    TdbSpecies,
    TdbTypeDefinition,
)
from chemworld.physchem.elements import ELEMENTS
from chemworld.physchem.tdb_expressions import TdbExpression


@dataclass(frozen=True)
class ChemSageElement:
    name: str
    molar_mass_g_mol: float
    reference_phase: None = None
    enthalpy_298_J_mol: None = None
    entropy_298_J_mol_K: None = None


_ConstituentArray = tuple[tuple[str, ...], ...]
CompiledChemSage = tuple[
    Mapping[str, ChemSageElement], Mapping[str, TdbSpecies], Mapping[str, TdbPhase],
    Mapping[str, TdbFunction], tuple[TdbParameter, ...], tuple[TdbTypeDefinition, ...],
    tuple[tuple[str, str], ...],
]


def is_auxiliary(name: str) -> bool:
    return name.startswith("E(") or name in {"E", "EA", "E-", "/-"}


def phase_identifiers(phases: Sequence[ChemSagePhase]) -> tuple[str, ...]:
    counts = Counter(p.name for p in phases)
    seen: Counter[str] = Counter()
    result = []
    for phase in phases:
        seen[phase.name] += 1
        result.append(phase.name if counts[phase.name] == 1 else
                      f"{phase.name}#OCCURRENCE{seen[phase.name]}")
    if len(set(result)) != len(result):
        raise ValueError("ChemSage duplicate occurrences collide with an authored phase identifier")
    return tuple(result)


def _from_formula(name: str, elements: Collection[str]) -> TdbSpecies:
    if name == "VA":
        return TdbSpecies(name, (("VA", 1.),))
    if is_auxiliary(name):
        return TdbSpecies(name, (("/-", 1.),), -1)
    base, charge = name, 0
    match = re.fullmatch(r"(.+)\[(\d+)?([+-])\]", name)
    if match:
        base = match[1]
        charge = int(match[2] or "1")*(1 if match[3] == "+" else -1)
    if base in elements:
        return TdbSpecies(name, ((base, 1.),), charge)
    # Only declared elemental symbols may appear in a derived formula. Prefer
    # the longest valid symbol, as in the original uppercase ChemSage names.
    atoms: dict[str, float] = {}
    offset = 0
    while offset < len(base):
        candidates = [e for e in elements if base.startswith(e, offset)]
        if not candidates:
            raise ValueError(f"ChemSage species {name} needs an explicit elemental interpretation")
        element = max(candidates, key=len)
        offset += len(element)
        number = re.match(r"(?:\d+(?:\.\d*)?|\.\d+)", base[offset:])
        amount = float(number[0]) if number else 1.
        offset += len(number[0]) if number else 0
        if not isfinite(amount) or amount <= 0:
            raise ValueError("ChemSage species atom counts must be positive and finite")
        atoms[element] = atoms.get(element, 0.)+amount
    return TdbSpecies(name, tuple(sorted(atoms.items())), charge)


def _member_species(
    member: ChemSageEndmember, header: ChemSageHeader, elements: Collection[str],
) -> TdbSpecies:
    atoms = tuple(sorted((e, n) for e, n in zip(header.components, member.stoichiometry,
                                              strict=True) if e in elements and n != 0))
    if any(n < 0 for _, n in atoms):
        raise ValueError("ChemSage endmember cannot have a negative real-element count")
    if not atoms:
        atoms = (("VA", 1.),) if member.name == "VA" else (("/-", 1.),)
    charge = _from_formula(member.name, elements).charge if "[" in member.name else 0
    return TdbSpecies(member.name, atoms, charge)


def _ion_composition(name: str, elements: Collection[str]) -> TdbSpecies:
    # Several source files repeat a valence in the species label as well as
    # the separate charge column. It is not an atom-count suffix (Fe2+, Na+1).
    for element in sorted(elements, key=len, reverse=True):
        suffix = name[len(element):] if name.startswith(element) else ""
        if re.fullmatch(r"(?:[+-]\d+(?:\.\d+)?|\d+(?:\.\d+)?[+-])", suffix):
            return _from_formula(element, elements)
    return _from_formula(name, elements)


def _constant_curve(name: str, value: float) -> TdbFunction:
    return TdbFunction(name, (TdbSegment(.01, 10000., TdbExpression(repr(float(value)))),))


def _interaction_array(indices: Sequence[int], array: _ConstituentArray) -> _ConstituentArray:
    result: list[list[str]] = [[] for _ in array]
    for raw in indices:
        offset = raw-1
        for i, subl in enumerate(array):
            if 0 <= offset < len(subl):
                result[i].append(subl[offset])
                break
            offset -= len(subl)
        else:
            raise ValueError("ChemSage interaction constituent index is out of bounds")
    if any(not subl or len(set(subl)) != len(subl) for subl in result):
        raise ValueError("ChemSage interaction must occupy every sublattice without duplicates")
    return tuple(tuple(subl) for subl in result)


def compile_chem_sage(
    parsed: ChemSageInput, *, excess_pressure_unit: str | None = None,
) -> CompiledChemSage:
    header = parsed.header
    known = {name.upper() for name in ELEMENTS}
    elements = {e: ChemSageElement(e, mass) for e, mass in zip(
        header.components, header.masses_g_mol, strict=True) if e in known}
    unknown = tuple(e for e in header.components if e not in known and not is_auxiliary(e)
                    and e != "VA")
    species = {e: TdbSpecies(e, ((e, 1.),)) for e in elements}
    phases: dict[str, TdbPhase] = {}
    parameters: list[TdbParameter] = []

    def insert(item: TdbSpecies) -> None:
        if item.name in species and species[item.name] != item:
            raise ValueError(f"ChemSage species {item.name} has conflicting elemental identities")
        species[item.name] = item

    for phase_index, (key, source) in enumerate(zip(phase_identifiers(parsed.phases), parsed.phases,
                                                   strict=True)):
        hints: dict[str, object] = {
            "source_format": "ChemSage", "source_phase_index": phase_index,
            "phase_family": source.phase_type, "initial_status": source.status,
        }
        groups: dict[str, int] = {}
        array: _ConstituentArray
        ratios: tuple[float, ...]
        limits = [f"component basis requires explicit mapping: {e}" for e in unknown]
        family = source.phase_type
        if family in {"IDVD", "IDWZ", "PITZ"}:
            limits.append(f"{family} requires its declared real-gas/aqueous physical model")
        if source.mqmqa:
            ions = []
            for names, charges, sign in ((source.mqmqa.cations, source.mqmqa.cation_charges, 1),
                                         (source.mqmqa.anions,
                                          source.mqmqa.anion_charge_magnitudes, -1)):
                lattice = []
                for name, magnitude in zip(names, charges, strict=True):
                    if magnitude <= 0:
                        raise ValueError("MQMQA formal charge magnitudes must be positive")
                    label = f"{name}[{magnitude:g}{'+' if sign > 0 else '-'}]"
                    base = _ion_composition(name, elements)
                    if base.name == "VA":
                        insert(base)  # Explicit vacancy component for active-species selection.
                    insert(TdbSpecies(label, base.constituents, sign*magnitude))
                    lattice.append(label)
                ions.append(tuple(lattice))
            array, ratios = tuple(ions), (1., 1.)
        elif family == "PURE":
            member = source.endmembers[0]
            amounts = {e: n for e, n in zip(header.components, member.stoichiometry, strict=True)
                       if e in elements and n != 0}
            array = tuple((e,) for e in sorted(amounts))
            ratios = tuple(amounts[e] for e in sorted(amounts))
            if not array:
                insert(TdbSpecies(member.name, (("/-", 1.),), -1))
                array, ratios = ((member.name,),), (1.,)
                limits.append("zero-atom auxiliary reservoir requires an electron/charge consumer")
        elif source.constituents:
            array, ratios = source.constituents, source.site_ratios
            for subl in array:
                for name in subl:
                    insert(_from_formula(name, elements))
        else:
            array, ratios = (tuple(m.name for m in source.endmembers),), (1.,)
            for member in source.endmembers:
                insert(_member_species(member, header, elements))
        if any(not isfinite(r) or r <= 0 for r in ratios):
            raise ValueError("ChemSage lattice ratios must be positive")
        if any(len(set(subl)) != len(subl) for subl in array):
            raise ValueError("ChemSage phase has duplicate charged constituent identities")
        if source.magnetic_afm_factor is not None:
            if source.magnetic_afm_factor <= 0 or source.magnetic_structure_factor is None:
                raise ValueError("ChemSage magnetic phase requires a positive AFM source factor")
            hints.update(magnetic_afm_factor=-1/source.magnetic_afm_factor,
                         magnetic_structure_factor=source.magnetic_structure_factor)
        aux: dict[str, list[tuple[_ConstituentArray, float]]] = {
            e: [] for e in header.components if is_auxiliary(e)
        }
        suppressed = []
        end_array: _ConstituentArray
        for i, member in enumerate(source.endmembers):
            if source.mqmqa:
                c, a = source.mqmqa.pair_indices[i]
                end_array = ((array[0][c-1],), (array[1][a-1],))
            elif family == "PURE":
                end_array = array
            elif source.endmember_indices:
                end_array = tuple((array[j][v-1],)
                                  for j, v in enumerate(source.endmember_indices[i]))
            else:
                end_array = ((member.name,),)
            if member.status != "entered":
                suppressed.append((end_array, member.status))
            if not source.mqmqa and family not in {"IDVD", "IDWZ", "PITZ"}:
                derived: dict[str, float] = {}
                for ratio, subl in zip(ratios, end_array, strict=True):
                    for e, count in species[subl[0]].constituents:
                        if e not in {"VA", "/-"}:
                            derived[e] = derived.get(e, 0.)+ratio*count
                if any(not isclose(derived.get(e, 0), n, rel_tol=1e-8, abs_tol=1e-10)
                       for e, n in zip(header.components, member.stoichiometry, strict=True)
                       if e in elements):
                    raise ValueError(f"ChemSage {key}: endmember atoms disagree with lattice atoms")
                for e, n in zip(header.components, member.stoichiometry, strict=True):
                    if e in aux and n != 0:
                        aux[e].append((end_array, n))
            try:
                curve = endmember_function(member, header, f"CS_{phase_index}_END_{i}")
            except ValueError as exc:
                if "mechanical" not in str(exc) and "volume" not in str(exc):
                    raise
                limits.append(str(exc))
                continue
            parameters.append(TdbParameter("MQMG" if source.mqmqa else "G", key, end_array, 0,
                                           curve, source_index=i))
            if member.curie_temperature_K is not None:
                for kind, value in (("TC", member.curie_temperature_K),
                                    ("BMAGN", cast(float, member.magnetic_moment))):
                    parameters.append(TdbParameter(kind, key, end_array, 0, _constant_curve(
                        f"CS_{phase_index}_{kind}_{i}", value), source_index=i))
            if member.chemical_group is not None:
                groups[member.name] = member.chemical_group
                hints["chemical_groups"] = tuple(groups.items())
                if not isclose(cast(float, member.stoichiometric_factor), 1.,
                               rel_tol=0, abs_tol=1e-12):
                    limits.append("non-unity QKTO stoichiometric factor needs its declared model")
        for i, item in enumerate(source.interactions):
            # MQ metadata stays in immutable source records; curves still
            # participate in the common segment-domain/derivative checks.
            end_array = () if source.mqmqa else _interaction_array(item.indices, array)
            if item.kind == "MAGNETIC":
                for kind, value in zip(("TC", "BMAGN"), item.coefficients, strict=True):
                    parameters.append(TdbParameter(kind, key, end_array, item.order,
                        _constant_curve(f"CS_{phase_index}_{kind}_XS_{i}", value), source_index=i))
                continue
            text = basis_expression(header.excess_indices, item.coefficients,
                                    excess_pressure_unit=excess_pressure_unit)
            curve = TdbFunction(f"CS_{phase_index}_XS_{i}", (TdbSegment(
                .01, 10000., TdbExpression(text)),))
            parameters.append(TdbParameter("L" if item.kind == "RKM" else item.kind, key,
                                           end_array, item.order, curve,
                                           exponents=tuple(v-1 for v in item.exponents)
                                           if item.kind == "QKT" else item.exponents,
                                           source_index=i))
        hints["auxiliary_balances"] = tuple((e, tuple(terms)) for e, terms in aux.items() if terms)
        hints["suppressed_endmembers"] = tuple(suppressed)
        hints["chemical_group_overrides"] = source.overrides
        hints["unsupported_physics"] = tuple(dict.fromkeys(limits))
        phases[key] = TdbPhase(key, tuple(ratios), tuple(tuple(s) for s in array),
                               frozenset({"G"}) if source.gas_slot else frozenset(), "",
                               MappingProxyType(hints))
    return (MappingProxyType(elements), MappingProxyType(species), MappingProxyType(phases),
            MappingProxyType({}), tuple(parameters), (), ())
