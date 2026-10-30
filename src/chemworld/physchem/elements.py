"""Native element, isotope and formula accounting, independent of property models."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from math import fsum, isfinite
from numbers import Integral, Real

from chemworld.physchem.atomic_data import AtomicEstimate, element_data, isotope_data


@dataclass(frozen=True)
class ElementSpec:
    symbol: str
    atomic_number: int
    atomic_weight_g_mol: float | None
    name: str = ""
    weight_source: str = "caller"
    standard_weight: AtomicEstimate | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.symbol, str)
            or re.fullmatch(r"[A-Z][a-z]?", self.symbol) is None
            or isinstance(self.atomic_number, bool)
            or not isinstance(self.atomic_number, Integral)
            or not 1 <= self.atomic_number <= 118
        ):
            raise ValueError("element requires a symbol and atomic number in 1..118")
        if self.atomic_weight_g_mol is not None and (
            not isfinite(self.atomic_weight_g_mol) or self.atomic_weight_g_mol <= 0
        ):
            raise ValueError("atomic weight must be positive and finite, or explicitly unavailable")

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "atomic_number": self.atomic_number,
            "atomic_weight_g_mol": self.atomic_weight_g_mol,
            "name": self.name,
            "weight_source": self.weight_source,
        }


_COMPATIBILITY_ELEMENTS: dict[str, ElementSpec] = {
    "H": ElementSpec("H", 1, 1.008),
    "He": ElementSpec("He", 2, 4.002602),
    "Li": ElementSpec("Li", 3, 6.94),
    "Be": ElementSpec("Be", 4, 9.0121831),
    "B": ElementSpec("B", 5, 10.81),
    "C": ElementSpec("C", 6, 12.011),
    "N": ElementSpec("N", 7, 14.007),
    "O": ElementSpec("O", 8, 15.999),
    "F": ElementSpec("F", 9, 18.998403163),
    "Ne": ElementSpec("Ne", 10, 20.1797),
    "Na": ElementSpec("Na", 11, 22.98976928),
    "Mg": ElementSpec("Mg", 12, 24.305),
    "Al": ElementSpec("Al", 13, 26.9815385),
    "Si": ElementSpec("Si", 14, 28.085),
    "P": ElementSpec("P", 15, 30.973761998),
    "S": ElementSpec("S", 16, 32.06),
    "Cl": ElementSpec("Cl", 17, 35.45),
    "Ar": ElementSpec("Ar", 18, 39.948),
    "K": ElementSpec("K", 19, 39.0983),
    "Ca": ElementSpec("Ca", 20, 40.078),
    "Fe": ElementSpec("Fe", 26, 55.845),
    "Cu": ElementSpec("Cu", 29, 63.546),
    "Zn": ElementSpec("Zn", 30, 65.38),
    "Br": ElementSpec("Br", 35, 79.904),
    "I": ElementSpec("I", 53, 126.90447),
    "Pt": ElementSpec("Pt", 78, 195.084),
}

ELEMENTS = {
    symbol: ElementSpec(
        symbol,
        item.atomic_number,
        _COMPATIBILITY_ELEMENTS[symbol].atomic_weight_g_mol
        if symbol in _COMPATIBILITY_ELEMENTS
        else item.standard_weight.value,
        item.name,
        "chemworld-original-26" if symbol in _COMPATIBILITY_ELEMENTS else "CIAAW2021",
        item.standard_weight,
    )
    for symbol, item in element_data().items()
}
_COUNTS = re.compile(r"(?:\d+(?:\.\d+)?|\.\d+)")
_SYMBOL = re.compile(r"[A-Z][a-z]?")
_ISOTOPE = re.compile(r"\[([1-9][0-9]*)([A-Z][a-z]?)\]")
_ALIASES = {"D": "[2H]", "T": "[3H]"}


def get_element(identity: object) -> ElementSpec:
    """Resolve symbol, English name or atomic number without loading isotopes."""
    if isinstance(identity, bool):
        raise ValueError("an element identity cannot be boolean")
    if isinstance(identity, Integral) or (
        isinstance(identity, str) and identity.isascii() and identity.isdigit()
    ):
        number = int(identity)
        matches = [item for item in ELEMENTS.values() if item.atomic_number == number]
    elif isinstance(identity, str):
        identity = {"aluminium": "Al", "caesium": "Cs", "sulphur": "S"}.get(
            identity.lower(),
            identity,
        )
        matches = [
            item
            for item in ELEMENTS.values()
            if item.symbol == identity or item.name.lower() == identity.lower()
        ]
    else:
        matches = []
    if len(matches) != 1:
        raise ValueError(f"Unknown element identity: {identity!r}")
    return matches[0]


def get_isotope(identity, mass_number=None):
    if mass_number is not None:
        if isinstance(mass_number, bool) or not isinstance(mass_number, Integral):
            raise ValueError("isotope mass number must be an integer")
        identity = f"[{mass_number}{get_element(identity).symbol}]"
    if not isinstance(identity, str):
        raise ValueError("isotope identity must be text")
    identity = _ALIASES.get(identity, identity)
    if not isinstance(identity, str) or _ISOTOPE.fullmatch(identity) is None:
        raise ValueError("isotope identity must use [mass-number element], e.g. [13C]")
    try:
        return isotope_data()[identity]
    except KeyError as exc:
        raise ValueError(f"Unknown isotope: {identity}") from exc


def _atom(label):
    label = _ALIASES.get(label, label)
    if label in ELEMENTS:
        return label, ELEMENTS[label].atomic_number, 0
    if isinstance(label, str) and _SYMBOL.fullmatch(label):
        raise ValueError(f"Unknown element symbol: {label}")
    isotope = get_isotope(label)
    return isotope.label, isotope.atomic_number, isotope.mass_number


def _weight(label):
    label, _, mass_number = _atom(label)
    value = get_isotope(label).mass.value if mass_number else ELEMENTS[label].atomic_weight_g_mol
    if value is None:
        raise ValueError(f"No standard atomic weight for {label}; select a known isotope")
    return value


def _split_charge(formula):
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("Formula cannot be empty")
    formula = formula.strip()
    patterns = (r"\((\+\d+|-\d+|\d+[+-]|\++|-+)\)$", r"\^(\d+[+-])$", r"([+-]\d+)$", r"(\++|-+)$")
    for pattern in patterns:
        match = re.search(pattern, formula)
        if match is None:
            continue
        token = match[1]
        sign = -1 if "-" in token else 1
        digits = token.replace("+", "").replace("-", "")
        magnitude = int(digits) if digits else len(token)
        if magnitude == 0:
            raise ValueError("an explicit charge magnitude must be positive")
        return formula[: match.start()], sign * magnitude, True
    return formula, 0, False


def parse_formula(formula: str) -> dict[str, float]:
    """Parse isotope-preserving counts; a decimal point is never a hydrate dot.

    A final single +/- is unit charge: NH4+ retains four hydrogens. Use Fe+3
    or Fe^3+ for explicit trivalent iron. D/T canonicalize to [2H]/[3H].
    """
    cleaned, _, _ = _split_charge(formula)
    parsed = {}
    for term in cleaned.split("·"):
        position = 0

        def multiplier(term=term):
            nonlocal position
            match = _COUNTS.match(term, position)
            if match is None:
                return 1.0
            position = match.end()
            value = float(match[0])
            if not isfinite(value) or value <= 0:
                raise ValueError("Formula multipliers must be positive and finite")
            return value

        def group(closing=None, term=term):
            nonlocal position
            counts = {}
            while position < len(term):
                character = term[position]
                if character in ")]}":
                    if character != closing or not counts:
                        raise ValueError("empty or mismatched formula group")
                    position += 1
                    return counts
                isotope = _ISOTOPE.match(term, position)
                if isotope is not None:
                    atom = get_isotope(isotope[0]).label
                    position = isotope.end()
                    _merge_counts(counts, {atom: 1}, multiplier())
                elif character in "([{":
                    position += 1
                    nested = group({"(": ")", "[": "]", "{": "}"}[character])
                    _merge_counts(counts, nested, multiplier())
                else:
                    symbol = _SYMBOL.match(term, position)
                    if symbol is None:
                        raise ValueError(f"Unsupported formula syntax near {term[position:]!r}")
                    position = symbol.end()
                    atom, _, _ = _atom(symbol[0])
                    _merge_counts(counts, {atom: 1}, multiplier())
            if closing is not None or not counts:
                raise ValueError("empty or unclosed formula group")
            return counts

        factor = multiplier()
        _merge_counts(parsed, group(), factor)
    return parsed


def formula_charge(formula):
    parse_formula(formula)
    return _split_charge(formula)[1]


def explicit_formula_charge(formula):
    """Return None when the formula does not declare a charge."""
    _, charge, declared = _split_charge(formula)
    return charge if declared else None


def serialize_formula(formula):
    charge = formula_charge(formula)
    return hill_formula(parse_formula(formula)) + (
        ""
        if charge == 0
        else ("+" if charge > 0 else "-") + ("" if abs(charge) == 1 else str(abs(charge)))
    )


def _composition(composition):
    answer = {}
    for symbol, count in composition.items():
        if (
            isinstance(count, bool)
            or not isinstance(count, Real)
            or not isfinite(count)
            or count < 0
        ):
            raise ValueError("atom counts must be finite nonnegative numbers")
        label, _, _ = _atom(symbol)
        if count > 0:
            answer[label] = answer.get(label, 0.0) + count
    if not answer:
        raise ValueError("Total atom count must be positive")
    return answer


def molecular_weight(composition: dict[str, float]) -> float:
    """Return molecular weight in g/mol from an element-count mapping."""

    weight = fsum(_weight(symbol) * count for symbol, count in _composition(composition).items())
    if not isfinite(weight) or weight <= 0:
        raise ValueError("Molecular weight must be positive and finite")
    return weight


def atom_fractions(composition: dict[str, float]) -> dict[str, float]:
    composition = _composition(composition)
    total = fsum(composition.values())
    return {symbol: count / total for symbol, count in composition.items()}


def mass_fractions_from_formula(composition: dict[str, float]) -> dict[str, float]:
    composition = _composition(composition)
    weight = molecular_weight(composition)
    return {symbol: _weight(symbol) * count / weight for symbol, count in composition.items()}


def element_matrix(
    compositions: Iterable[dict[str, float]],
    element_order: tuple[str, ...] | None = None,
) -> tuple[tuple[tuple[float, ...], ...], tuple[str, ...]]:
    rows = [_composition(row) for row in compositions]
    if element_order is None:
        symbols = sorted(
            {symbol for composition in rows for symbol in composition},
            key=lambda symbol: _atom(symbol)[1:],
        )
    else:
        symbols = [_atom(symbol)[0] for symbol in element_order]
        if len(set(symbols)) != len(symbols):
            raise ValueError("element order cannot contain duplicate atoms")
    matrix = tuple(tuple(row.get(symbol, 0.0) for symbol in symbols) for row in rows)
    return matrix, tuple(symbols)


def hill_formula(composition: dict[str, float]) -> str:
    """Return a Hill-system formula string for display and stable IDs."""

    composition = _composition(composition)
    carbon = any(_atom(symbol)[1] == 6 for symbol in composition)

    def order(symbol):
        _, number, mass = _atom(symbol)
        base = get_element(number).symbol
        priority = (0 if base == "C" else 1 if base == "H" else 2) if carbon else 0
        return priority, base, mass

    ordered = sorted(composition, key=order)
    return "".join(f"{symbol}{_format_count(composition[symbol])}" for symbol in ordered)


def _merge_counts(target: dict[str, float], source: dict[str, float], multiplier: float) -> None:
    if not isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Formula multipliers must be positive")
    for symbol, count in source.items():
        value = target.get(symbol, 0.0) + count * multiplier
        if not isfinite(value):
            raise ValueError("formula counts exceed the finite numeric domain")
        target[symbol] = value


def _format_count(count: float) -> str:
    if count == 1:
        return ""
    if count == int(count):
        return str(int(count))
    return format(Decimal(str(float(count))), "f")


def mixture_atomic_composition(compositions, mole_fractions):
    rows, fractions = list(compositions), tuple(mole_fractions)
    if len(rows) != len(fractions) or any(
        isinstance(z, bool) or not isfinite(z) or z < 0 for z in fractions
    ):
        raise ValueError("mixture requires one finite nonnegative mole fraction per composition")
    if abs(fsum(fractions) - 1) > 1e-12:
        raise ValueError("mixture mole fractions must sum to one")
    answer = {}
    for composition, fraction in zip(rows, fractions, strict=True):
        values = _composition(composition)
        if fraction > 0:
            _merge_counts(answer, values, fraction)
    return answer


def similarity_variable(composition):
    composition = _composition(composition)
    return fsum(composition.values()) / molecular_weight(composition)


def index_hydrogen_deficiency(composition):
    atoms = {}
    for symbol, count in _composition(composition).items():
        base = get_element(_atom(symbol)[1]).symbol
        if base not in {"C", "H", "N", "O", "S", "F", "Cl", "Br", "I"}:
            raise ValueError("hydrogen deficiency assumes C/H/N/O/S and monovalent halogens")
        atoms[base] = atoms.get(base, 0) + count
    return (
        2 * atoms.get("C", 0)
        + 2
        + atoms.get("N", 0)
        - atoms.get("H", 0)
        - sum(atoms.get(symbol, 0) for symbol in ("F", "Cl", "Br", "I"))
    ) / 2
