"""Exact stoichiometric linear algebra on the native element/isotope contract.

This module declares no reaction kinetics, identities, inventories or clocks.
Rational elimination uses caller-supplied compositions and charge, retaining
underdetermined reactions instead of selecting an arbitrary nullspace vector.
"""

from fractions import Fraction
from functools import reduce
from math import fsum, gcd, isfinite, lcm
from numbers import Real

from chemworld.physchem.elements import _atom, element_matrix


def _fraction(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("stoichiometry requires finite real coefficients, not booleans")
    return value if isinstance(value, Fraction) else Fraction(str(value))


def stoichiometric_matrix(atomss, reactants, *, charges=None):
    """Rows are conserved atoms/isotopes (and explicit charge); columns are species.

    Reactants have positive columns and products negative columns. An explicit
    atom-free charged carrier is allowed, e.g. {} with charge -1 for an electron.
    It does not become a ComponentSpec or gain a fabricated molecular weight.
    """
    atoms, flags = tuple(atomss), tuple(reactants)
    if (
        len(atoms) != len(flags)
        or len(atoms) < 2
        or any(type(v) is not bool for v in flags)
        or not any(flags)
        or all(flags)
    ):
        raise ValueError("one boolean side per species, with reactants and products, is required")
    charge = (Fraction(0),) * len(atoms) if charges is None else tuple(map(_fraction, charges))
    if len(charge) != len(atoms):
        raise ValueError("charge must align with every species")
    if any(not atom and not q for atom, q in zip(atoms, charge, strict=True)):
        raise ValueError("atom-free carriers require an explicit nonzero charge")
    nonempty = [atom for atom in atoms if atom]
    _, symbols = element_matrix(nonempty) if nonempty else ((), ())
    canonical = []
    for atom in atoms:
        row = {}
        for label, count in atom.items():
            label = _atom(label)[0]
            row[label] = row.get(label, Fraction(0)) + _fraction(count)
        canonical.append(row)
    signs = tuple(1 if value else -1 for value in flags)
    matrix = tuple(
        tuple(
            row.get(label, Fraction(0)) * sign for row, sign in zip(canonical, signs, strict=True)
        )
        for label in symbols
    )
    if any(charge):
        matrix += (tuple(q * s for q, s in zip(charge, signs, strict=True)),)
    return matrix


def balance_stoichiometry(matrix, *, allow_fractional=False, fixed_coefficients=None):
    """Unique positive balance, with exact elimination and optional fixed entries.

    Without fixed entries, normalize the first species to one before solving.
    Integer output is reduced to its smallest positive ratio. With fixed entries
    the supplied scale is retained. Extra degrees of freedom require more data.
    """
    rows = [list(map(_fraction, row)) for row in matrix]
    if not rows or not rows[0] or len({len(row) for row in rows}) != 1:
        raise ValueError("stoichiometric matrix must be nonempty and rectangular")
    width = len(rows[0])
    fixed = {0: Fraction(1)} if fixed_coefficients is None else dict(fixed_coefficients)
    if not fixed:
        raise ValueError("fixed coefficients must constrain the reaction scale")
    augmented = [[*row, Fraction(0)] for row in rows]
    for index, value in fixed.items():
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < width:
            raise ValueError("fixed coefficient index is outside the matrix")
        value = _fraction(value)
        if value <= 0:
            raise ValueError("fixed coefficients must be positive")
        augmented.append([Fraction(int(i == index)) for i in range(width)] + [value])
    pivot_row, columns = 0, []
    for column in range(width):
        found = next((i for i in range(pivot_row, len(augmented)) if augmented[i][column]), None)
        if found is None:
            continue
        augmented[pivot_row], augmented[found] = augmented[found], augmented[pivot_row]
        divisor = augmented[pivot_row][column]
        augmented[pivot_row] = [v / divisor for v in augmented[pivot_row]]
        for index, row in enumerate(augmented):
            if index != pivot_row and row[column]:
                multiplier = row[column]
                augmented[index] = [
                    a - multiplier * b for a, b in zip(row, augmented[pivot_row], strict=True)
                ]
        columns.append(column)
        pivot_row += 1
    if any(not any(row[:-1]) and row[-1] for row in augmented):
        raise ValueError("reaction and fixed stoichiometric coefficients are inconsistent")
    if len(columns) < width:
        raise ValueError("reaction is underdetermined; supply additional fixed coefficients")
    result = [Fraction(0)] * width
    for index, column in enumerate(columns):
        result[column] = augmented[index][-1]
    if any(value <= 0 for value in result):
        raise ValueError("declared reaction sides have no strictly positive balance")
    if not allow_fractional:
        if fixed_coefficients is None:
            multiple = lcm(*(v.denominator for v in result))
            integers = [int(v * multiple) for v in result]
            divisor = reduce(gcd, integers)
            result = [Fraction(v // divisor) for v in integers]
        elif any(v.denominator != 1 for v in result):
            raise ValueError("fixed scale produces fractions; explicitly allow fractional output")
    return tuple(result)


def standard_formation_reaction(atoms, *, elemental_forms=None, allow_fractional=False):
    """Compound coefficient, elemental coefficients and their atom dictionaries.

    Coefficients describe elements -> compound (not a kinetic mechanism). Default
    elemental units are diatomic H/N/O/F/Cl/Br/I and atom-equivalent others;
    no elemental phase/allotrope or thermodynamic value is inferred. Explicit
    elemental forms allow other molecular conventions and isotope references.
    """
    _, labels = element_matrix((atoms,))
    if elemental_forms is None:
        elemental_forms = {
            label: {label: 2 if _atom(label)[1] in {1, 7, 8, 9, 17, 35, 53} else 1}
            for label in labels
        }
    if set(elemental_forms) != set(labels):
        raise ValueError("supply exactly one elemental reference form for every atom/isotope")
    forms = tuple(dict(elemental_forms[label]) for label in labels)
    for label, form in zip(labels, forms, strict=True):
        _, form_labels = element_matrix((form,))
        if form_labels != (label,):
            raise ValueError("each elemental reference must contain only its declared atom/isotope")
    matrix = stoichiometric_matrix((atoms, *forms), (False, *(True for _ in forms)))
    values = balance_stoichiometry(matrix, allow_fractional=allow_fractional)
    return values[0], values[1:], forms


def _mass_inputs(coefficients, molecular_weights):
    coefficients, weights = tuple(coefficients), tuple(molecular_weights)
    if not coefficients or len(coefficients) != len(weights):
        raise ValueError("one molecular weight is required for every coefficient")
    for value in coefficients:
        _fraction(value)
    if any(_fraction(weight) <= 0 for weight in weights):
        raise ValueError("molecular weights must be positive")
    return coefficients, weights


def stoichiometry_molar_to_mass(coefficients, molecular_weights):
    """Molar coefficients times explicit g/mol weights; signed quantities retain sign."""
    values, weights = _mass_inputs(coefficients, molecular_weights)
    return tuple(float(v * w) for v, w in zip(values, weights, strict=True))


def stoichiometry_mass_to_molar(coefficients, molecular_weights):
    values, weights = _mass_inputs(coefficients, molecular_weights)
    return tuple(float(v / w) for v, w in zip(values, weights, strict=True))


def stoichiometry_mass_error(coefficients, molecular_weights, reactants):
    masses = stoichiometry_molar_to_mass(coefficients, molecular_weights)
    flags = tuple(reactants)
    if len(flags) != len(masses) or any(type(v) is not bool for v in flags):
        raise ValueError("one boolean side is required for every mass coefficient")
    return fsum(mass if flag else -mass for mass, flag in zip(masses, flags, strict=True))
