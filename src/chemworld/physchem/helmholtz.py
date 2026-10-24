"""Native pure-fluid Helmholtz properties with one analytic energy function.

Homogeneous states report local stability, not a global phase-equilibrium
claim. Parameters and reference offsets are caller supplied; no reference
package, hidden property database, or live material inventory is owned here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Any

from chemworld.physchem.caloric_math import compensated_sum, partial_derivative
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.second_order_pair import SecondOrderPair


@dataclass(frozen=True)
class HelmholtzState:
    fluid_id: str
    temperature_K: float
    molar_density_mol_m3: float
    mass_density_kg_m3: float
    pressure_Pa: float
    enthalpy_J_mol: float
    internal_energy_J_mol: float
    entropy_J_mol_K: float
    cp_J_mol_K: float
    cv_J_mol_K: float
    speed_of_sound_m_s: float
    gibbs_energy_J_mol: float
    helmholtz_energy_J_mol: float
    compressibility_factor: float
    dP_dT_rho_Pa_K: float
    dP_drho_T_Pa_m3_mol: float
    isothermal_compressibility_Pa_inv: float
    thermal_expansion_K_inv: float
    stability_margin: float
    locally_stable: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        result["stability_scope"] = "homogeneous_local_only"
        return result


@dataclass(frozen=True)
class HelmholtzFluid:
    fluid_id: str
    casrn: str
    molar_mass_kg_mol: float
    gas_constant_J_mol_K: float
    reducing_temperature_K: float
    reducing_molar_density_mol_m3: float
    minimum_temperature_K: float
    maximum_temperature_K: float
    maximum_pressure_Pa: float
    ideal_terms: tuple[HelmholtzTerm, ...]
    residual_terms: tuple[HelmholtzTerm, ...]
    source: str
    eos_index: int = 0
    pseudo_pure: bool = False
    ideal_prefactor: float = 1.0
    citation: str = ""

    def __post_init__(self) -> None:
        positive = (self.molar_mass_kg_mol, self.gas_constant_J_mol_K,
                    self.reducing_temperature_K, self.reducing_molar_density_mol_m3,
                    self.minimum_temperature_K, self.maximum_temperature_K,
                    self.maximum_pressure_Pa, self.ideal_prefactor)
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in positive):
            raise ValueError("Helmholtz constants and limits must be finite and positive")
        if self.minimum_temperature_K >= self.maximum_temperature_K:
            raise ValueError("Helmholtz temperature limits must be ordered")
        if not self.fluid_id or not self.source:
            raise ValueError("Helmholtz fluid requires identity and explicit source")
        if (type(self.eos_index) is not int or self.eos_index < 0
                or type(self.pseudo_pure) is not bool):
            raise ValueError("invalid Helmholtz EOS selection or pseudo-pure flag")
        for terms, prefix in ((self.ideal_terms, "IdealGasHelmholtz"),
                              (self.residual_terms, "ResidualHelmholtz")):
            if any(not isinstance(t, HelmholtzTerm) or not t.type.startswith(prefix)
                   for t in terms):
                raise ValueError("Helmholtz terms are in the wrong contribution")
        if sum(t.type == "IdealGasHelmholtzLead" for t in self.ideal_terms) != 1:
            raise ValueError("Helmholtz ideal contribution requires exactly one density lead")
        object.__setattr__(self, "ideal_terms", tuple(self.ideal_terms))
        object.__setattr__(self, "residual_terms", tuple(self.residual_terms))

    def alpha(self, kind: str, delta: Any, tau: Any) -> Any:
        """Evaluate a dimensionless contribution; accepts shared derivative seeds."""
        if kind not in {"ideal", "residual", "total"}:
            raise ValueError("Helmholtz contribution must be ideal, residual or total")
        contributions = []
        if kind in {"ideal", "total"}:
            contributions.append(self.ideal_prefactor * compensated_sum(
                term.evaluate(delta, tau, self.reducing_temperature_K)
                for term in self.ideal_terms))
        if kind in {"residual", "total"}:
            contributions.append(compensated_sum(
                term.evaluate(delta, tau, self.reducing_temperature_K)
                for term in self.residual_terms))
        return compensated_sum(contributions)

    def alpha_derivative(self, kind: str, *, delta: float, tau: float,
                         delta_order: int = 0, tau_order: int = 0) -> float:
        """Analytic dimensionless mixed partial, total order zero through three."""
        if (any(type(v) is not int or v < 0 for v in (delta_order, tau_order))
                or delta_order + tau_order > 3):
            raise ValueError("Helmholtz derivative total order must be zero through three")
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (delta, tau)):
            raise ValueError("Helmholtz reduced coordinates must be finite and positive")
        coordinates = ("delta",) * delta_order + ("tau",) * tau_order
        if not coordinates:
            result = self.alpha(kind, delta, tau)
            if not isfinite(result):
                raise ValueError("nonfinite Helmholtz energy")
            return result
        return partial_derivative(lambda x: self.alpha(kind, x["delta"], x["tau"]),
                                  {"delta": delta, "tau": tau}, coordinates)

    def _state_derivatives(self, delta: float, tau: float) -> tuple[float, ...]:
        """Evaluate each contribution once for all state-response derivatives."""
        density, temperature = SecondOrderPair(delta, dx=1), SecondOrderPair(tau, dy=1)
        ideal = SecondOrderPair(0) + self.alpha("ideal", density, temperature)
        residual = SecondOrderPair(0) + self.alpha("residual", density, temperature)
        total = ideal + residual
        values = (total.value, total.dy, total.dyy, residual.dx, residual.dxx, residual.dxy)
        if not all(isfinite(value) for value in values):
            raise ValueError("nonfinite Helmholtz state derivative")
        return values

    def state(self, *, temperature_K: float, molar_density_mol_m3: float,
              require_stable: bool = True) -> HelmholtzState:
        """Evaluate a supplied homogeneous state inside the declared T/P domain.

        This does not replace a two-phase T/rho flash. Mechanical and thermal
        stability are local; metastable homogeneous branches remain possible.
        """
        T, rho = temperature_K, molar_density_mol_m3
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (T, rho)):
            raise ValueError("Helmholtz temperature and density must be finite and positive")
        if not self.minimum_temperature_K <= T <= self.maximum_temperature_K:
            raise ValueError("temperature outside Helmholtz source validity")
        if type(require_stable) is not bool:
            raise ValueError("require_stable must be boolean")
        if self.ideal_prefactor != 1:
            raise ValueError("nonunit ideal prefactor changes density normalization; "
                             "only raw alpha evaluation is supported")
        delta, tau = rho/self.reducing_molar_density_mol_m3, self.reducing_temperature_K/T
        a, at, att, ard, ardd, ardt = self._state_derivatives(delta, tau)
        R = self.gas_constant_J_mol_K
        z = 1 + delta*ard
        stiffness = 1 + 2*delta*ard + delta**2*ardd
        thermal = 1 + delta*ard - delta*tau*ardt
        pressure = rho*R*T*z
        cv = -R*tau**2*att
        stable = stiffness > 0 and cv > 0 and pressure > 0
        if require_stable and not stable:
            raise ValueError("homogeneous Helmholtz state is mechanically or thermally unstable")
        if pressure > self.maximum_pressure_Pa:
            raise ValueError("pressure outside Helmholtz source validity")
        if stiffness == 0 or cv == 0:
            raise ValueError("singular Helmholtz response at stability boundary")
        cp = cv + R*thermal**2/stiffness
        speed2 = R*T/self.molar_mass_kg_mol * (stiffness + R*thermal**2/cv)
        if speed2 < 0:
            raise ValueError("homogeneous Helmholtz sound speed is not real")
        internal = R*T*tau*at
        enthalpy = internal + pressure/rho
        entropy = R*(tau*at-a)
        warnings: tuple[str, ...] = (() if stable else ("locally_unstable_homogeneous_state",))
        if 0 < stiffness < 1e-7:
            warnings += ("near_mechanical_stability_singularity",)
        result = HelmholtzState(
            self.fluid_id, float(T), float(rho), rho*self.molar_mass_kg_mol, pressure,
            enthalpy, internal, entropy, cp, cv, sqrt(speed2), enthalpy-T*entropy, R*T*a,
            z, rho*R*thermal, R*T*stiffness, 1/(rho*R*T*stiffness),
            thermal/(T*stiffness), stiffness, stable, warnings)
        if any(not isfinite(v) for v in asdict(result).values() if isinstance(v, float)):
            raise ValueError("nonfinite Helmholtz state response")
        return result

    def to_dict(self) -> dict[str, Any]:
        result = {key: value for key, value in self.__dict__.items()
                  if key not in {"ideal_terms", "residual_terms"}}
        result["ideal_terms"] = [term.to_dict() for term in self.ideal_terms]
        result["residual_terms"] = [term.to_dict() for term in self.residual_terms]
        result["model"] = "chemworld_pure_helmholtz_v1"
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HelmholtzFluid:
        if data.get("model") == "chemworld_molecular_helmholtz_v1":
            from chemworld.physchem.molecular_consumers import MolecularHelmholtzFluid

            return MolecularHelmholtzFluid.from_dict(data)
        values = dict(data)
        if values.pop("model", None) != "chemworld_pure_helmholtz_v1":
            raise ValueError("unknown native Helmholtz model")
        for key in ("ideal_terms", "residual_terms"):
            values[key] = tuple(HelmholtzTerm.from_dict(t) for t in values[key])
        return cls(**values)
