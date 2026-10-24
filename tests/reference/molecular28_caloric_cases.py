"""Authored fixed 28B cases; declarations precede qualification execution."""

from dataclasses import dataclass

from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_caloric_flash import (
    MolecularCaloricFlashSeed,
    MolecularCaloricFlashSpec,
)
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_mixture import MolecularCaloricModel
from chemworld.physchem.residual_system import RootPolicy
from chemworld.runtime.molecular_flash_thermal import MolecularFlashThermalSpec
from chemworld.runtime.molecular_phase_inventory import MolecularPhaseTarget

R = 8.31446261815324
T_REF, P_REF = 298.15, 101325.0
TC, PC, OMEGA = (369.83, 425.12), (4.248e6, 3.796e6), (0.152, 0.2)
CP, MASS = (80.0, 100.0), (0.04409562, 0.0581222)
# Exact PR critical constraints; rounded textbook .45724/.07780 are not
# numerically identical constants and must not be compared as the same EOS.
PR_A, PR_B = 0.4572355289213822, 0.07779607390388846
OMEGA_LLE = 9000.0


@dataclass(frozen=True)
class RegularSolution:
    component_ids: tuple[str, ...] = ("a", "b")
    gas_constant_J_mol_K: float = R

    def alphar(self, temperature_K, molar_density_mol_m3, mole_fractions):
        return OMEGA_LLE / (R * temperature_K) * mole_fractions[0] * mole_fractions[1]


def ideal_refs(ids):
    return tuple(
        HelmholtzFluid(
            k,
            "",
            MASS[i],
            R,
            T_REF,
            P_REF / (R * T_REF),
            200.0,
            500.0,
            1e9,
            (
                HelmholtzTerm("IdealGasHelmholtzLead", {"a1": CP[i] / R - 1, "a2": -CP[i] / R}),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": CP[i] / R - 1}),
            ),
            (),
            "authored constant ideal Cp; H=S=0 at 298.15 K, 101325 Pa",
        )
        for i, k in enumerate(ids)
    )


def caloric_spec(kind="PR", policy=0):
    if kind == "PR":
        ids = ("propane", "butane")
        residual = CPAModel(
            tuple(
                CPAComponent(
                    k,
                    PR_A * R * R * tc * tc / pc,
                    PR_B * R * tc / pc,
                    tc,
                    0.37464 + 1.54226 * w - 0.26992 * w * w,
                    "authored PR critical constants",
                )
                for k, tc, pc, w in zip(ids, TC, PC, OMEGA, strict=True)
            ),
            ((0.0, 0.0), (0.0, 0.0)),
            (200.0, 500.0),
            "explicit PR binary",
            cubic_model="peng_robinson",
        )
        basins = (
            MolecularDensityBasin("vapor", (1.0, 2000.0), 250.0),
            MolecularDensityBasin("liquid", (7500.0, 13500.0), 10000.0),
        )
        names, pressure = ("vapor", "liquid"), 6e5
    else:
        residual = RegularSolution()
        ids = residual.component_ids
        basins = (MolecularDensityBasin("fluid", (1.0, 100.0), 40.0),)
        names, pressure = ("fluid", "fluid"), 1e5
    return MolecularCaloricFlashSpec(
        MolecularCaloricModel(residual, ideal_refs(ids)),
        basins,
        names,
        pressure,
        (280.0, 320.0),
        root_policy=RootPolicy((2e-9, 2e-10)[policy], (100, 150)[policy]),
    )


def seed(spec, feed, temperature=300.0):
    ids = spec.model.component_ids
    fractions = (0.8, 0.3) if ids[0] == "propane" else (0.05, 0.95)
    return MolecularCaloricFlashSeed(
        temperature,
        (
            {k: feed[k] * fractions[i] for i, k in enumerate(ids)},
            {k: feed[k] * (1 - fractions[i]) for i, k in enumerate(ids)},
        ),
        (250.0, 11500.0) if ids[0] == "propane" else (40.0, 40.0),
    )


def closure_cases():
    cases = [
        {"kind": "PR", "temperature_K": t, "z": z} for t in (298.0, 302.0) for z in (0.5, 0.6, 0.7)
    ]
    cases += [{"kind": "LLE", "temperature_K": t, "z": 0.4} for t in (290.0, 310.0)]
    return [
        {
            **case,
            "coordinate": coordinate,
            "policy": policy,
            "id": f"{case['kind']}-{case['temperature_K']:g}-{case['z']:g}-{coordinate}-{policy}",
        }
        for case in cases
        for coordinate in ("H", "S")
        for policy in (0, 1)
    ]


def thermal_spec(method="BDF"):
    return MolecularFlashThermalSpec(
        caloric_spec(),
        (
            MolecularPhaseTarget("vapor", "vapor", "gas"),
            MolecularPhaseTarget("liquid", "liquid", "liquid", True),
        ),
        support_heat_capacity_J_K=0.1,
        ode_method=method,
    )
