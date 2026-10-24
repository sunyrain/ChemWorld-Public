"""Absolute homogeneous molecular mixtures on existing inventory/flow contracts.

Each component supplies its explicit ideal Helmholtz reference. The molecular
residual owns mixing; the common CoupledSystem owns amounts, energy and time.
No global mixture phase stability or multiphase closure is inferred here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log, sqrt
from typing import Any

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.coupled_system import CoupledStateView, ProcessContribution
from chemworld.physchem.flash_units import phase_advection_rate
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.molecular_consumers import (
    molecular_model_from_dict,
    molecular_model_to_dict,
)
from chemworld.physchem.molecular_eos import (
    MolecularEOSState,
    ResidualHelmholtzModel,
    molecular_density_from_pressure,
    molecular_eos_state,
)


@dataclass(frozen=True)
class MolecularCaloricState(MolecularEOSState):
    component_ids: tuple[str, ...]
    internal_energy_J_mol: float
    enthalpy_J_mol: float
    entropy_J_mol_K: float
    helmholtz_energy_J_mol: float
    gibbs_energy_J_mol: float
    cv_J_mol_K: float
    cp_J_mol_K: float
    mass_density_kg_m3: float
    speed_of_sound_m_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "energy_reference": "explicit_component_ideal_plus_residual",
            "stability_scope": "homogeneous_local_mechanical_and_thermal_only",
        }


@dataclass(frozen=True)
class MolecularCaloricModel:
    residual_model: ResidualHelmholtzModel
    ideal_references: tuple[HelmholtzFluid, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ideal_references", tuple(self.ideal_references))
        if tuple(ref.fluid_id for ref in self.ideal_references) != self.component_ids:
            raise ValueError("ideal reference identities and molecular component order must agree")
        if not self.ideal_references or any(
            type(ref) is not HelmholtzFluid
            or ref.residual_terms
            or ref.pseudo_pure
            or ref.ideal_prefactor != 1
            or ref.gas_constant_J_mol_K != self.gas_constant_J_mol_K
            for ref in self.ideal_references
        ):
            raise ValueError("each component requires a pure ideal reference with the same R")

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self.residual_model.component_ids

    @property
    def gas_constant_J_mol_K(self) -> float:  # noqa: N802
        return self.residual_model.gas_constant_J_mol_K

    def state(
        self, temperature_K: float, molar_density_mol_m3: float, mole_fractions: Sequence[float]
    ) -> MolecularCaloricState:
        residual = molecular_eos_state(
            self.residual_model, temperature_K, molar_density_mol_m3, mole_fractions
        )
        xs, t, rho, gas = (
            residual.mole_fractions,
            temperature_K,
            molar_density_mol_m3,
            (self.gas_constant_J_mol_K),
        )
        ideal = tuple(
            ref.state(temperature_K=t, molar_density_mol_m3=rho) for ref in self.ideal_references
        )

        def mean(name: str) -> float:
            return sum(x * getattr(value, name) for x, value in zip(xs, ideal, strict=True))

        u = mean("internal_energy_J_mol") + residual.residual_internal_energy_J_mol
        h = u + residual.pressure_Pa / rho
        # Pure reference entropies are evaluated at total ideal density; the
        # exact ideal mixing contribution supplies partial-density normalization.
        entropy = (
            mean("entropy_J_mol_K")
            + residual.residual_entropy_J_mol_K
            - gas * sum(x * log(x) for x in xs if x > 0)
        )
        cv = mean("cv_J_mol_K") + residual.residual_cv_J_mol_K
        cp = mean("cp_J_mol_K") + residual.residual_cp_J_mol_K
        if not isfinite(cv) or cv <= 0 or not isfinite(cp) or cp <= 0:
            raise ValueError("homogeneous molecular mixture is thermally unstable")
        mass = sum(
            x * ref.molar_mass_kg_mol for x, ref in zip(xs, self.ideal_references, strict=True)
        )
        speed = sqrt(
            (residual.dP_drho_T_Pa_m3_mol + t * residual.dP_dT_rho_Pa_K**2 / (rho * rho * cv))
            / mass
        )
        return MolecularCaloricState(
            **residual.__dict__,
            component_ids=self.component_ids,
            internal_energy_J_mol=u,
            enthalpy_J_mol=h,
            entropy_J_mol_K=entropy,
            helmholtz_energy_J_mol=u - t * entropy,
            gibbs_energy_J_mol=h - t * entropy,
            cv_J_mol_K=cv,
            cp_J_mol_K=cp,
            mass_density_kg_m3=rho * mass,
            speed_of_sound_m_s=speed,
        )

    def state_tp(
        self,
        temperature_K: float,
        pressure_Pa: float,
        mole_fractions: Sequence[float],
        *,
        initial_density_mol_m3: float,
        density_bounds_mol_m3: tuple[float, float],
    ) -> MolecularCaloricState:
        """Reuse the declared local density basin; this does not select a global phase."""
        root = molecular_density_from_pressure(
            self.residual_model,
            temperature_K,
            pressure_Pa,
            mole_fractions,
            initial_density_mol_m3=initial_density_mol_m3,
            density_bounds_mol_m3=density_bounds_mol_m3,
        )
        return self.state(temperature_K, root.molar_density_mol_m3, mole_fractions)

    def partial_inventory_energies(
        self, temperature_K: float, volume_m3: float, amounts_mol: Mapping[str, float]
    ) -> tuple[float, ...]:
        """Partial extensive U at fixed T,V, including zero-component limits."""
        if (
            set(amounts_mol) != set(self.component_ids)
            or any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in amounts_mol.values())
            or not isfinite(volume_m3)
            or volume_m3 <= 0
        ):
            raise ValueError("inventory energies need matching nonnegative amounts and positive V")
        total = sum(amounts_mol.values())
        if total <= 0:
            raise ValueError("molecular inventory cannot be empty")
        rho = total / volume_m3
        self.state(temperature_K, rho, tuple(amounts_mol[k] / total for k in self.component_ids))
        values = {
            "T": temperature_K,
            **{f"n{i}": amounts_mol[key] for i, key in enumerate(self.component_ids)},
        }

        def extensive_alpha(v: Mapping[str, Any]) -> Any:
            amounts = tuple(v[f"n{i}"] for i in range(len(self.component_ids)))
            n = sum(amounts)
            return n * self.residual_model.alphar(
                v["T"], n / volume_m3, tuple(a / n for a in amounts)
            )

        factor = -self.gas_constant_J_mol_K * temperature_K**2
        return tuple(
            ref.state(temperature_K=temperature_K, molar_density_mol_m3=rho).internal_energy_J_mol
            + factor * partial_derivative(extensive_alpha, values, ("T", f"n{i}"))
            for i, ref in enumerate(self.ideal_references)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld-molecular-caloric-v1",
            "residual_model": molecular_model_to_dict(self.residual_model),
            "ideal_references": [ref.to_dict() for ref in self.ideal_references],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularCaloricModel:
        if data.get("model") != "chemworld-molecular-caloric-v1":
            raise ValueError("unknown molecular caloric model encoding")
        return cls(
            molecular_model_from_dict(data["residual_model"]),
            tuple(HelmholtzFluid.from_dict(ref) for ref in data["ideal_references"]),
        )


@dataclass(frozen=True)
class MolecularIsochoricThermodynamics:
    model: MolecularCaloricModel
    volume_L: float
    support_heat_capacity_J_K: float = 0.0
    reference_temperature_K: float = 298.15

    def __post_init__(self) -> None:
        if any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (self.volume_L, self.reference_temperature_K)
        ) or (
            isinstance(self.support_heat_capacity_J_K, bool)
            or not isfinite(self.support_heat_capacity_J_K)
            or self.support_heat_capacity_J_K < 0
        ):
            raise ValueError(
                "molecular thermal volume/reference must be positive, support nonnegative"
            )

    @property
    def inventory_keys(self) -> tuple[str, ...]:
        return self.model.component_ids

    def molar_state(self, state: CoupledStateView) -> MolecularCaloricState:
        if set(state.amounts_mol) != set(self.inventory_keys) or state.volume_L != self.volume_L:
            raise ValueError("molecular thermal inventory identities and fixed volume must agree")
        amounts = tuple(state.amounts_mol[k] for k in self.inventory_keys)
        if any(not isfinite(v) or v < 0 for v in amounts) or sum(amounts) <= 0:
            raise ValueError("molecular thermal inventory must be nonnegative and nonempty")
        n = sum(amounts)
        return self.model.state(
            state.temperature_K, n / (self.volume_L * 1e-3), tuple(a / n for a in amounts)
        )

    def validate_state(self, state: CoupledStateView) -> None:
        self.molar_state(state)

    def heat_capacity_j_k(self, state: CoupledStateView) -> float:
        return sum(state.amounts_mol.values()) * self.molar_state(state).cv_J_mol_K + (
            self.support_heat_capacity_J_K
        )

    def internal_energy_j(self, state: CoupledStateView) -> float:
        return sum(state.amounts_mol.values()) * self.molar_state(state).internal_energy_J_mol + (
            self.support_heat_capacity_J_K * (state.temperature_K - self.reference_temperature_K)
        )

    def pressure_pa(self, state: CoupledStateView) -> float:
        return self.molar_state(state).pressure_Pa

    def chemical_energy_rate_w(
        self,
        state: CoupledStateView,
        derivatives: Mapping[str, float],
        contributions: Sequence[ProcessContribution],
    ) -> float:
        if set(derivatives) - set(self.inventory_keys) or any(
            not isfinite(v) for v in derivatives.values()
        ):
            raise ValueError(
                "molecular inventory rate must refer to declared components and be finite"
            )
        if any(c.legacy_reaction_energy_W is not None for c in contributions):
            raise ValueError("molecular U cannot also consume duplicate reaction heat")
        if not any(derivatives.values()):
            return 0.0
        energies = self.model.partial_inventory_energies(
            state.temperature_K, self.volume_L * 1e-3, state.amounts_mol
        )
        return sum(
            energy * derivatives.get(key, 0.0)
            for key, energy in zip(self.inventory_keys, energies, strict=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "molecular-isochoric-thermal-v1",
            "caloric_model": self.model.to_dict(),
            "volume_L": self.volume_L,
            "support_heat_capacity_J_K": self.support_heat_capacity_J_K,
            "reference_temperature_K": self.reference_temperature_K,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularIsochoricThermodynamics:
        values = dict(data)
        if values.pop("model", None) != "molecular-isochoric-thermal-v1":
            raise ValueError("unknown molecular thermal encoding")
        values["model"] = MolecularCaloricModel.from_dict(values.pop("caloric_model"))
        return cls(**values)


@dataclass(frozen=True)
class MolecularStream:
    """A complete homogeneous bulk mixture stream, never a selective species flux."""

    state: MolecularCaloricState
    molar_rate_mol_s: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.molar_rate_mol_s, bool)
            or not isfinite(self.molar_rate_mol_s)
            or self.molar_rate_mol_s < 0
        ):
            raise ValueError("mixture stream rate must be finite and nonnegative")

    @property
    def enthalpy_rate_W(self) -> float:  # noqa: N802
        return self.molar_rate_mol_s * self.state.enthalpy_J_mol

    def contribution(self, ports_by_component: Mapping[str, str]) -> ProcessContribution:
        if (
            set(ports_by_component) != set(self.state.component_ids)
            or any(not isinstance(p, str) or not p for p in ports_by_component.values())
            or len(set(ports_by_component.values())) != len(ports_by_component)
        ):
            raise ValueError("bulk mixture stream requires distinct ports for all components")
        rates, _ = phase_advection_rate(
            dict(zip(self.state.component_ids, self.state.mole_fractions, strict=True)),
            self.state.enthalpy_J_mol,
            self.molar_rate_mol_s,
        )
        return ProcessContribution(
            port_rates_mol_s={ports_by_component[k]: v for k, v in rates.items()},
            port_enthalpies_J_mol=dict.fromkeys(
                ports_by_component.values(), self.state.enthalpy_J_mol
            ),
        )
