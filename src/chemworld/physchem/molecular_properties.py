"""Molecular residual EOS on the existing single-phase property/flash interface.

The declared density basin chooses a local homogeneous root. It is never a
global stability certificate. Ideal caloric datums remain in property packages.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Any

from chemworld.physchem.eos import R_J_PER_MOL_K, EOSState
from chemworld.physchem.molecular_consumers import (
    molecular_model_from_dict,
    molecular_model_to_dict,
)
from chemworld.physchem.molecular_eos import (
    ResidualHelmholtzModel,
    molecular_density_from_pressure,
)


@dataclass(frozen=True)
class MolecularEOSSpec:
    eos_id: str
    residual_model: ResidualHelmholtzModel
    phase: str
    density_bounds_mol_m3: tuple[float, float]
    initial_density_mol_m3: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "density_bounds_mol_m3", tuple(self.density_bounds_mol_m3))
        if not self.eos_id or self.phase not in {"vapor", "liquid"}:
            raise ValueError("molecular property EOS requires an identity and explicit phase")
        if self.residual_model.gas_constant_J_mol_K != R_J_PER_MOL_K:
            raise ValueError("molecular property EOS and ideal package must use the same R")
        if len(self.density_bounds_mol_m3) != 2:
            raise ValueError("molecular property EOS requires two density bounds")
        lo, hi = self.density_bounds_mol_m3
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (lo, hi)) or lo >= hi:
            raise ValueError("molecular density bounds must be positive and ordered")
        initial = self.initial_density_mol_m3
        if initial is not None and (
            isinstance(initial, bool) or not isfinite(initial) or not lo < initial < hi
        ):
            raise ValueError("molecular density seed must lie inside the declared basin")
        if self.phase == "liquid" and initial is None:
            raise ValueError("liquid molecular basin requires an explicit density seed")

    @property
    def component_ids(self) -> tuple[str, ...]:
        return self.residual_model.component_ids

    def evaluate(
        self,
        composition: Mapping[str, float],
        *,
        temperature_K: float,
        pressure_Pa: float,
        phase: str,
    ) -> tuple[EOSState, dict[str, float]]:
        if phase != self.phase:
            raise ValueError("requested phase disagrees with the declared molecular density basin")
        if set(composition) - set(self.component_ids) or any(
            isinstance(v, bool) or not isfinite(v) or v < 0 for v in composition.values()
        ):
            raise ValueError("molecular composition must use declared nonnegative components")
        total = sum(composition.values())
        if not isfinite(total) or total <= 0:
            raise ValueError("molecular composition requires positive total amount")
        if any(
            isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (temperature_K, pressure_Pa)
        ):
            raise ValueError("molecular properties require finite positive T/P")
        fractions = tuple(composition.get(k, 0.0) / total for k in self.component_ids)
        initial = self.initial_density_mol_m3
        if initial is None:
            initial = pressure_Pa / (R_J_PER_MOL_K * temperature_K)
        if not self.density_bounds_mol_m3[0] < initial < self.density_bounds_mol_m3[1]:
            raise ValueError("molecular density seed is outside the declared basin")
        state = molecular_density_from_pressure(
            self.residual_model,
            temperature_K,
            pressure_Pa,
            fractions,
            initial_density_mol_m3=initial,
            density_bounds_mol_m3=self.density_bounds_mol_m3,
        )
        rho, z = state.molar_density_mol_m3, state.compressibility_factor
        # Molecular residual entropy uses the ideal gas at the same density;
        # property packages use the ideal gas at the same pressure instead.
        entropy = state.residual_entropy_J_mol_K + R_J_PER_MOL_K * log(z)
        enthalpy = state.residual_enthalpy_J_mol
        assert state.log_fugacity_coefficients is not None
        phi = tuple(exp(v) for v in state.log_fugacity_coefficients)
        if any(not isfinite(v) or v <= 0 for v in phi):
            raise ValueError("nonfinite molecular fugacity coefficient")
        derivatives = {
            "cp_departure_J_mol_K": state.residual_cp_J_mol_K,
            "cp_minus_cv_J_mol_K": temperature_K
            * state.dP_dT_rho_Pa_K**2
            / (rho**2 * state.dP_drho_T_Pa_m3_mol),
            "d_molar_volume_dT_P": state.dP_dT_rho_Pa_K / (rho**2 * state.dP_drho_T_Pa_m3_mol),
            "d_molar_volume_dP_T": -1 / (rho**2 * state.dP_drho_T_Pa_m3_mol),
        }
        return EOSState(
            eos_id=self.eos_id,
            model="molecular_residual_helmholtz",
            phase=phase,
            root_selection_policy="explicit_local_density_basin",
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            composition=dict(zip(self.component_ids, fractions, strict=True)),
            compressibility_factor=z,
            molar_volume_m3_mol=1 / rho,
            fugacity_coefficients=dict(zip(self.component_ids, phi, strict=True)),
            molar_residual_enthalpy_J_mol=enthalpy,
            molar_residual_entropy_J_mol_K=entropy,
            molar_residual_gibbs_J_mol=enthalpy - temperature_K * entropy,
            roots=(z,),
            mixture_parameters={"residual_model": type(self.residual_model).__name__},
            residual_properties={
                "reference": "ideal_gas_at_same_T_P_composition",
                "stability_scope": "local_mechanical_only",
                "density_bounds_mol_m3": list(self.density_bounds_mol_m3),
            },
        ), derivatives

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld-molecular-property-eos-v1",
            "eos_id": self.eos_id,
            "residual_model": molecular_model_to_dict(self.residual_model),
            "phase": self.phase,
            "density_bounds_mol_m3": list(self.density_bounds_mol_m3),
            "initial_density_mol_m3": self.initial_density_mol_m3,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularEOSSpec:
        values = dict(data)
        if values.pop("model", None) != "chemworld-molecular-property-eos-v1":
            raise ValueError("unknown molecular property EOS encoding")
        values["residual_model"] = molecular_model_from_dict(values["residual_model"])
        return cls(**values)
