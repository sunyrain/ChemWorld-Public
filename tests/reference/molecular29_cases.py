"""Explicit same-EOS declarations; source-only coexistence initializes the fixture."""

from math import sqrt

import numpy as np

from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.molecular_interface import (
    SingularSquareGradientInfluence,
    SquareGradientInfluence,
)
from reference.molecular29_source_reference import (
    declarations,
    miscibility_temperature,
    source_endpoints,
)


def native_model(family):
    materials, interaction = declarations(family)
    return CPAModel(
        tuple(
            CPAComponent(
                identifier,
                0.4572355289213825 * 8.314**2 * tc**2 / (pc * 1e5),
                0.07779607390388854 * 8.314 * tc / (pc * 1e5),
                tc,
                0.37464 + 1.54226 * omega - 0.26992 * omega**2,
                "Explicit public PR material constants; same-EOS source fixture",
            )
            for identifier, tc, pc, omega, _ in materials
        ),
        tuple(tuple(float(v) for v in row) for row in interaction),
        (200.0, 500.0),
        "Parent29 explicit PR declarations; source R and kappa convention",
        cubic_model="peng_robinson",
        gas_constant_J_mol_K=8.314,
    )


def native_influence(family, beta=0.5):
    materials, _ = declarations(family)
    identifiers = tuple(row[0] for row in materials)
    coefficients = tuple(row[-1] for row in materials)
    if beta == 0 and len(materials) > 1:
        return SingularSquareGradientInfluence(
            identifiers,
            tuple((sqrt(c),) for c in coefficients),
            (200.0, 500.0),
            "Explicit rank-one C=FF.T; PhasePy beta=0 model declaration",
        )
    return SquareGradientInfluence(
        identifiers,
        tuple(
            tuple(c if i == j else (1 - beta) * sqrt(c * d) for j, d in enumerate(coefficients))
            for i, c in enumerate(coefficients)
        ),
        (200.0, 500.0),
        "Explicit same-EOS constant influence parameters",
    )


def native_pair(family, temperature, pressure=None, *, beta=0.5, endpoints=None):
    pair = source_endpoints(family, temperature, pressure) if endpoints is None else endpoints
    model = native_model(family)
    states = tuple(
        molecular_eos_state(
            model, temperature, sum(pair[side]), tuple(c / sum(pair[side]) for c in pair[side])
        )
        for side in ("left", "right")
    )
    return model, states, native_influence(family, beta), pair


def fixed_cases():
    critical = miscibility_temperature()
    pure = [
        {
            "name": f"pure_{temperature:.6f}",
            "family": "pure",
            "T": temperature,
            "P": None,
            "beta": 0.5,
        }
        for temperature in (280.0, 300.0, 0.99 * 369.83)
    ]
    profiles = [
        ("binary_PD", "binary", 300.0, 6e5, 0.5, (3e-9, 6e-9)),
        ("ternary_PD", "ternary", 320.0, 6e5, 0.5, (3e-9, 6e-9)),
        ("LLE_099_PD", "symmetric_lle", 0.99 * critical, 5e6, 0.5, (20e-9, 40e-9)),
        ("LLE_0999_PD", "symmetric_lle", 0.999 * critical, 5e6, 0.5, (20e-9, 40e-9)),
        ("binary_rank1", "binary", 300.0, 6e5, 0.0, (3e-9, 6e-9)),
        ("ternary_rank1", "ternary", 320.0, 6e5, 0.0, (3e-9, 6e-9)),
        ("LLE_099_rank1", "symmetric_lle", 0.99 * critical, 5e6, 0.0, (3e-9, 6e-9)),
    ]
    return pure + [
        dict(zip(("name", "family", "T", "P", "beta", "lengths"), row, strict=True))
        for row in profiles
    ]


def initial_tanh(left, right, nodes):
    coordinate = np.linspace(0.0, 1.0, nodes)
    fraction = 0.5 * (1 + np.tanh(10 * (coordinate - 0.503)))
    fraction = (fraction - fraction[0]) / (fraction[-1] - fraction[0])
    return np.asarray(left) + (np.asarray(right) - left) * fraction[:, None]
