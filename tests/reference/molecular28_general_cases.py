"""Fixed parent28 authored native declarations; independent expectations live separately."""

from dataclasses import dataclass

import numpy as np

from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_mixture import MolecularCaloricModel
from chemworld.physchem.molecular_multiphase import MolecularMultiphaseSpec
from reference.molecular28_general_reference import FAMILIES, MATERIALS, P_REF, T_REF, R


@dataclass(frozen=True)
class Regular:
    component_ids: tuple[str, ...] = ("a", "b", "c")
    gas_constant_J_mol_K: float = R
    omega: float = 9000.0

    def alphar(self, temperature, density, fractions):
        return (
            self.omega
            / (R * temperature)
            * sum(fractions[i] * fractions[j] for i in range(3) for j in range(i + 1, 3))
        )


def pr_model(family="vle", *, pure=False):
    ids = ("propane",) if pure else FAMILIES[family]
    r = 8.314 if family in {"vlle", "azeotrope"} else R
    values = [MATERIALS[k] for k in ids]
    interactions = np.zeros((len(ids), len(ids)))
    if family == "vlle":
        interactions[0, 2] = interactions[2, 0] = 0.5
        interactions[1, 2] = interactions[2, 1] = 0.5
    return CPAModel(
        tuple(
            CPAComponent(
                k,
                0.4572355289213822 * r * r * tc * tc / pc,
                0.07779607390388846 * r * tc / pc,
                tc,
                0.37464 + 1.54226 * w - 0.26992 * w * w,
                "explicit PR",
            )
            for k, (tc, pc, w, _, _) in zip(ids, values, strict=True)
        ),
        tuple(tuple(row) for row in interactions),
        (100.0, 1000.0),
        "independent source matched authored PR",
        gas_constant_J_mol_K=r,
        cubic_model="peng_robinson",
    )


def ideal_refs(ids, gas_constant):
    return tuple(
        HelmholtzFluid(
            k,
            "",
            MATERIALS[k][3] if k in MATERIALS else 0.04 + 0.01 * i,
            gas_constant,
            T_REF,
            P_REF / (gas_constant * T_REF),
            100.0,
            1000.0,
            1e10,
            (
                HelmholtzTerm(
                    "IdealGasHelmholtzLead", {"a1": cp / gas_constant - 1, "a2": -cp / gas_constant}
                ),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": cp / gas_constant - 1}),
            ),
            (),
            "authored constant Cp H=S=0 gauge",
        )
        for i, k in enumerate(ids)
        for cp in [MATERIALS[k][4] if k in MATERIALS else 80.0 + 10 * i]
    )


def native_spec(family, *, omega=9000.0):
    residual = Regular(omega=omega) if family == "regular" else pr_model(family)
    if family == "regular":
        basins = (MolecularDensityBasin("fluid", (1.0, 500.0), 40.0),)
        tb, pb, kinds = (250.0, 700.0), (1e4, 1e6), {"fluid": "liquid"}
    elif family == "vlle":
        basins = (
            MolecularDensityBasin("gas", (1.0, 7000.0), 1000.0),
            MolecularDensityBasin("hydrocarbon_liquid", (5000.0, 20000.0), 10000.0),
            MolecularDensityBasin("water_liquid", (25000.0, 52000.0), 47000.0),
        )
        tb, pb = (260.0, 360.0), (5e5, 7e6)
        kinds = {"gas": "gas", "hydrocarbon_liquid": "liquid", "water_liquid": "liquid"}
    else:
        basins = (
            MolecularDensityBasin("gas", (1.0, 6000.0), 500.0),
            MolecularDensityBasin("liquid", (5000.0, 20000.0), 11000.0),
        )
        tb, pb, kinds = (250.0, 400.0), (1e5, 2e6), {"gas": "gas", "liquid": "liquid"}
    return MolecularMultiphaseSpec(
        MolecularCaloricModel(
            residual, ideal_refs(residual.component_ids, residual.gas_constant_J_mol_K)
        ),
        basins,
        tb,
        pb,
        phase_kind_by_basin=kinds,
    )


def regular_spec(omega=9000.0):
    return native_spec("regular", omega=omega)
