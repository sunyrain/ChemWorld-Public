"""Fixed-volume liquid/gas equilibrium energy on the existing coupled clock."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from chemworld.physchem.coupled_system import CoupledStateView, ProcessContribution
from chemworld.physchem.helmholtz import HelmholtzState
from chemworld.physchem.helmholtz_flash import HelmholtzTwoPhaseState, coexistence_derivatives
from chemworld.physchem.helmholtz_solvers import (
    HelmholtzPhaseModel,
    homogeneous_state_derivatives,
)


@dataclass(frozen=True)
class HelmholtzEquilibriumIsochoricThermodynamics:
    phase_model: HelmholtzPhaseModel
    inventory_key: str
    volume_L: float
    support_heat_capacity_J_K: float = 0.0
    reference_temperature_K: float = 298.15

    def __post_init__(self) -> None:
        if not self.inventory_key or self.phase_model.fluid.pseudo_pure:
            raise ValueError("equilibrium thermal model needs one named pure fluid")
        if any(not isfinite(v) or v <= 0 for v in (self.volume_L, self.reference_temperature_K)):
            raise ValueError("equilibrium thermal volume and support reference must be positive")
        if (not isfinite(self.support_heat_capacity_J_K)
                or self.support_heat_capacity_J_K < 0):
            raise ValueError("support heat capacity must be finite and nonnegative")

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return (self.inventory_key,)

    def molar_state(
        self, state: CoupledStateView,
    ) -> HelmholtzState[float] | HelmholtzTwoPhaseState:
        if set(state.amounts_mol) != {self.inventory_key}:
            raise ValueError("equilibrium thermal inventory contains an undeclared fluid")
        if state.volume_L != self.volume_L:
            raise ValueError("isochoric equilibrium model rejects changing volume")
        amount = state.amounts_mol[self.inventory_key]
        if amount <= 0:
            raise ValueError("equilibrium thermal model requires positive inventory; empty phase")
        rho = amount/(self.volume_L*1e-3)
        T = state.temperature_K
        model = self.phase_model
        if model.critical_temperature_K > T:
            saturation = model.saturation(T)
            if not saturation.success:
                raise ValueError("equilibrium thermal saturation:"+saturation.failure_reason)
            liquid, gas = saturation.liquid, saturation.gas
            if liquid is None or gas is None:
                raise AssertionError("invariant violated: liquid or gas is None")
            if gas.molar_density_mol_m3 <= rho <= liquid.molar_density_mol_m3:
                quality = ((1/rho-1/liquid.molar_density_mol_m3)
                           /(1/gas.molar_density_mol_m3-1/liquid.molar_density_mol_m3))
                return HelmholtzTwoPhaseState(liquid, gas, quality)
        return model.fluid.state(temperature_K=T, molar_density_mol_m3=rho)

    def validate_state(self, state: CoupledStateView) -> None:
        self.molar_state(state)

    def _energy_derivatives(self, current: HelmholtzState[float] | HelmholtzTwoPhaseState
                            ) -> tuple[float, float]:
        if isinstance(current, HelmholtzState):
            return homogeneous_state_derivatives(self.phase_model.fluid, current)["U"]
        derivatives = coexistence_derivatives(self.phase_model, current)
        uT, uq = derivatives["U"]
        rhoT, rhoq = derivatives["rho"]
        # Transform the coexistence (T,Q) coordinates to physical (T,n/V).
        # This equilibrium isochoric response includes latent heat; it is not
        # a weighted average of the homogeneous endpoint Cv values.
        return uT-uq*rhoT/rhoq, uq/rhoq

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        cv = self._energy_derivatives(self.molar_state(state))[0]
        if not isfinite(cv) or cv <= 0:
            raise ValueError("nonpositive equilibrium isochoric thermal response")
        return state.amounts_mol[self.inventory_key]*cv+self.support_heat_capacity_J_K

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return (state.amounts_mol[self.inventory_key]*self.molar_state(state).internal_energy_J_mol
                + self.support_heat_capacity_J_K*(state.temperature_K-self.reference_temperature_K))

    def pressure_pa(self, state: CoupledStateView) -> float:
        return self.molar_state(state).pressure_Pa

    def partial_inventory_energy_j_mol(self, state: CoupledStateView) -> float:
        current = self.molar_state(state)
        return (current.internal_energy_J_mol
                + current.molar_density_mol_m3*self._energy_derivatives(current)[1])

    def chemical_energy_rate_w(self, state: CoupledStateView,
                               derivatives: Mapping[str, float],
                               contributions: Sequence[ProcessContribution]) -> float:
        if set(derivatives)-{self.inventory_key}:
            raise ValueError("equilibrium energy rate refers to an undeclared fluid")
        if any(c.legacy_reaction_energy_W is not None for c in contributions):
            raise ValueError("same-EOS equilibrium U rejects duplicate reaction heat")
        rate = derivatives.get(self.inventory_key, 0.0)
        if not isfinite(rate):
            raise ValueError("equilibrium inventory rate must be finite")
        return self.partial_inventory_energy_j_mol(state)*rate if rate else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"model": "helmholtz-equilibrium-isochoric-v1",
                "phase_model": self.phase_model.to_dict(), "inventory_key": self.inventory_key,
                "volume_L": self.volume_L,
                "support_heat_capacity_J_K": self.support_heat_capacity_J_K,
                "reference_temperature_K": self.reference_temperature_K}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HelmholtzEquilibriumIsochoricThermodynamics:
        values = dict(data)
        if values.pop("model", None) != "helmholtz-equilibrium-isochoric-v1":
            raise ValueError("unknown equilibrium thermal declaration")
        values["phase_model"] = HelmholtzPhaseModel.from_dict(values["phase_model"])
        return cls(**values)
