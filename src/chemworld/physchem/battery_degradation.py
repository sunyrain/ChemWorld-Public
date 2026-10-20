"""Finite solvent-diffusion SEI, independently implemented for native inventories.

Equation reference: PyBaMM sei_growth.py, BSD-3-Clause,
a4ea370d9ff844e586907cea7109c38f36ce3660. The explicit reaction here consumes
two bound Li and one EC, making one carbonate and one retained ethylene.
Upstream's effective one-electron coefficient is not copied as our stoichiometry.
"""

from dataclasses import dataclass
from math import exp, isfinite

from chemworld.physchem.electrochemistry import R_J_PER_MOL_K

SEI_SPECIES = {"sei:EC": "EC", "sei:carbonate": "Li2CO3", "sei:ethylene": "C2H4"}
SEI_FORMULAS = {"EC": "C3H4O3", "Li2CO3": "Li2CO3", "C2H4": "C2H4"}


@dataclass(frozen=True)
class SEISpec:
    solvent_diffusivity_m2_s: float = 1e-20
    activation_energy_J_mol: float = 20000.0
    reference_temperature_K: float = 298.15
    solvent_volume_m3: float = 1e-6
    initial_solvent_concentration_mol_m3: float = 1000.0
    molar_volume_m3_mol: float = 1e-5
    initial_thickness_m: float = 1e-8
    film_resistivity_ohm_m: float = 1e5
    reference_driving_energy_J_mol: float = 50000.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"SEI {name} must be finite and nonnegative")
        if any(
            getattr(self, name) <= 0
            for name in (
                "reference_temperature_K",
                "solvent_volume_m3",
                "molar_volume_m3_mol",
                "initial_thickness_m",
            )
        ):
            raise ValueError(
                "SEI reference temperature, volumes and initial thickness must be positive"
            )

    def initial_amounts(self, area_m2):
        return {
            "sei:EC": self.solvent_volume_m3 * self.initial_solvent_concentration_mol_m3,
            "sei:carbonate": area_m2 * self.initial_thickness_m / self.molar_volume_m3_mol,
            "sei:ethylene": 0.0,
        }

    def diffusivity(self, temperature_K):
        return self.solvent_diffusivity_m2_s * exp(
            -self.activation_energy_J_mol
            / R_J_PER_MOL_K
            * (1 / temperature_K - 1 / self.reference_temperature_K)
        )

    def thickness_m(self, state, area_m2):
        return state.amounts_mol["sei:carbonate"] * self.molar_volume_m3_mol / area_m2

    def reaction_rate_mol_s(self, state, area_m2):
        thickness = self.thickness_m(state, area_m2)
        if thickness <= 0:
            raise ValueError("solvent-limited SEI requires an existing film")
        return (
            area_m2
            * self.diffusivity(state.temperature_K)
            * (state.amounts_mol["sei:EC"] / self.solvent_volume_m3)
            / thickness
        )

    def resistance_ohm(self, state, area_m2):
        return self.film_resistivity_ohm_m * self.thickness_m(state, area_m2) / area_m2
