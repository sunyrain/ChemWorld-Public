"""Authored neutral synthetic chemistry and explicit PR parameter declarations."""

from copy import deepcopy

from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.eos import R_J_PER_MOL_K as R
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.physchem.molecular_mixture import MolecularIsochoricThermodynamics
from chemworld.physchem.molecular_reaction import mechanism_caloric_model

from .mechanism34_cases import THERMOS

# Public Peng--Robinson critical constraints (unrounded cubic critical root).
OMEGA_A = 0.45723552892138218
OMEGA_B = 0.0777960739038885


def source(family="constant-cp", *, multiple=True, reversible=True):
    species = []
    for index, name in enumerate(("A", "B", "C")):
        thermo = deepcopy(THERMOS[family])
        if family == "constant-cp":
            thermo["h0"] = f"{-10 - 2 * index} kJ/mol"
        elif family == "NASA7":
            for row in thermo["data"]:
                row[5] -= 200 * index
        elif family == "NASA9":
            for row in thermo["data"]:
                row[7] -= 200 * index
        elif family == "Shomate":
            for row in thermo["data"]:
                row[5] -= 2 * index
        species.append(
            {"name": name, "composition": {"H": 4 if name == "C" else 2}, "thermo": thermo}
        )
    arrow = "<=>" if reversible else "=>"
    reactions = [{"equation": f"A {arrow} B", "rate-constant": {"A": 0.2, "b": 0.1, "Ea": 1e6}}]
    if multiple:
        reactions.append(
            {"equation": f"2 B {arrow} C", "rate-constant": {"A": 0.03, "b": 0, "Ea": 0}}
        )
    return {
        "phases": [
            {
                "name": "synthetic",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "species": ["A", "B", "C"],
                "elements": ["H"],
            }
        ],
        "units": {"length": "m", "quantity": "kmol", "activation-energy": "J/kmol"},
        "species": species,
        "reactions": reactions,
    }


def residual(*, interaction=0.0):
    components = []
    for index, name in enumerate(("A", "B", "C")):
        b, tc, omega = (3 + index) * 1e-5, 280 + 40 * index, 0.05 + 0.03 * index
        kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
        components.append(
            CPAComponent(
                name, OMEGA_A / OMEGA_B * b * R * tc, b, tc, kappa, "authored synthetic PR"
            )
        )
    return CPAModel(
        tuple(components),
        tuple(tuple(0.0 if i == j else interaction for j in range(3)) for i in range(3)),
        (200.0, 3000.0),
        "authored synthetic PR",
        cubic_model="peng_robinson",
    )


def inputs(family="constant-cp", *, volume_L=1.0, multiple=True, reversible=True, interaction=0.0):
    network = convert_mechanism(source(family, multiple=multiple, reversible=reversible))
    return network, MolecularIsochoricThermodynamics(
        mechanism_caloric_model(network, residual(interaction=interaction)), volume_L
    )


def cantera_source(family="constant-cp", *, multiple=True, reversible=True, interaction=0.0):
    data = source(family, multiple=multiple, reversible=reversible)
    data["phases"][0]["thermo"] = "Peng-Robinson"
    model = residual(interaction=interaction)
    for i, (species, component) in enumerate(zip(data["species"], model.components, strict=True)):
        species["equation-of-state"] = {
            "model": "Peng-Robinson",
            "a": component.attraction_Pa_m6_mol2 * 1e6,
            "b": component.covolume_m3_mol * 1000,
            "acentric-factor": 0.05 + 0.03 * i,
        }
        if interaction:
            species["equation-of-state"]["binary-a"] = {
                other.component_id: (component.attraction_Pa_m6_mol2 * other.attraction_Pa_m6_mol2)
                ** 0.5
                * (1 - interaction)
                * 1e6
                for other in model.components
                if other != component
            }
    return data
