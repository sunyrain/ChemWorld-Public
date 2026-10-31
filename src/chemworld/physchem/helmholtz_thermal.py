"""Same-EOS fixed-volume thermal consumption on the existing CoupledSystem.

One caller-owned pure homogeneous inventory supplies n, T and V. This adapter
owns parameters only, and neither moves a piston nor performs two-phase flash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from chemworld.physchem.coupled_system import CoupledStateView, ProcessContribution
from chemworld.physchem.helmholtz import HelmholtzFluid, HelmholtzState
from chemworld.physchem.helmholtz_solvers import homogeneous_state_derivatives


@dataclass(frozen=True)
class HelmholtzIsochoricThermodynamics:
    fluid: HelmholtzFluid
    inventory_key: str
    volume_L: float
    support_heat_capacity_J_K: float = 0.0
    reference_temperature_K: float = 298.15

    def __post_init__(self) -> None:
        if not self.inventory_key:
            raise ValueError("Helmholtz thermal inventory requires one explicit key")
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0
               for v in (self.volume_L, self.reference_temperature_K)):
            raise ValueError("Helmholtz thermal volume and support reference must be positive")
        if (not isfinite(self.support_heat_capacity_J_K)
                or self.support_heat_capacity_J_K < 0):
            raise ValueError("support heat capacity must be finite and nonnegative")

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return (self.inventory_key,)

    def molar_state(self, state: CoupledStateView) -> HelmholtzState[float]:
        if set(state.amounts_mol) != {self.inventory_key}:
            raise ValueError("Helmholtz thermal inventory must contain only its declared fluid")
        if state.volume_L != self.volume_L:
            raise ValueError("isochoric Helmholtz model rejects a changing declared volume")
        amount = state.amounts_mol[self.inventory_key]
        if amount <= 0:
            raise ValueError("Helmholtz thermal model requires positive inventory; empty phase")
        return self.fluid.state(temperature_K=state.temperature_K,
                                molar_density_mol_m3=amount/(self.volume_L*1e-3))

    def validate_state(self, state: CoupledStateView) -> None:
        self.molar_state(state)

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        return (state.amounts_mol[self.inventory_key]*self.molar_state(state).cv_J_mol_K
                + self.support_heat_capacity_J_K)

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return (state.amounts_mol[self.inventory_key]*self.molar_state(state).internal_energy_J_mol
                + self.support_heat_capacity_J_K
                * (state.temperature_K-self.reference_temperature_K))

    def pressure_pa(self, state: CoupledStateView) -> float:
        return self.molar_state(state).pressure_Pa

    def partial_inventory_energy_j_mol(self, state: CoupledStateView) -> float:
        current = self.molar_state(state)
        urho = homogeneous_state_derivatives(self.fluid, current)["U"][1]
        return current.internal_energy_J_mol+current.molar_density_mol_m3*urho

    def chemical_energy_rate_w(self, state: CoupledStateView,
                               derivatives: Mapping[str, float],
                               contributions: Sequence[ProcessContribution]) -> float:
        if set(derivatives)-{self.inventory_key}:
            raise ValueError("Helmholtz energy rate refers to an undeclared fluid")
        if any(c.legacy_reaction_energy_W is not None for c in contributions):
            raise ValueError("same-EOS Helmholtz U cannot consume duplicate reaction heat")
        rate = derivatives.get(self.inventory_key, 0.0)
        if not isfinite(rate):
            raise ValueError("Helmholtz inventory rate must be finite")
        if rate == 0:
            return 0.0
        return self.partial_inventory_energy_j_mol(state)*rate

    def to_dict(self) -> dict[str, Any]:
        return {"model": "helmholtz-isochoric-thermal-v1", "fluid": self.fluid.to_dict(),
                "inventory_key": self.inventory_key, "volume_L": self.volume_L,
                "support_heat_capacity_J_K": self.support_heat_capacity_J_K,
                "reference_temperature_K": self.reference_temperature_K}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HelmholtzIsochoricThermodynamics:
        values = dict(data)
        if values.pop("model", None) != "helmholtz-isochoric-thermal-v1":
            raise ValueError("unknown Helmholtz thermal declaration")
        values["fluid"] = HelmholtzFluid.from_dict(values["fluid"])
        return cls(**values)
