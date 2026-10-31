"""Pure-fluid phase closure using the shared residual root solver.

Ancillary fits supply guesses only. Accepted saturation endpoints use the
original energy function with equal pressure and chemical potential.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import exp, isfinite, log
from types import MappingProxyType
from typing import Any

from chemworld.physchem.helmholtz import HelmholtzFluid, HelmholtzState
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    solve_root,
)


@dataclass(frozen=True)
class HelmholtzAncillary:
    kind: str
    reducing_temperature_K: float
    reducing_value: float
    minimum_temperature_K: float
    maximum_temperature_K: float
    n: tuple[float, ...]
    t: tuple[float, ...]
    using_tau_r: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"rhoLnoexp", "rhoL", "rhoV", "pL", "pV"}:
            raise ValueError("unsupported Helmholtz saturation ancillary")
        if (not self.n or len(self.n) != len(self.t)
                or type(self.using_tau_r) is not bool):
            raise ValueError("invalid Helmholtz ancillary coefficient arrays")
        if any(not isfinite(v) for v in (*self.n, *self.t)):
            raise ValueError("Helmholtz ancillary coefficients must be finite")
        if any(not isfinite(v) or v <= 0 for v in (
                self.reducing_temperature_K, self.reducing_value,
                self.maximum_temperature_K)):
            raise ValueError("Helmholtz ancillary constants must be positive")
        if not isfinite(self.minimum_temperature_K) or self.minimum_temperature_K < 0:
            raise ValueError("Helmholtz ancillary lower fit limit must be nonnegative")
        if self.minimum_temperature_K >= self.maximum_temperature_K:
            raise ValueError("Helmholtz ancillary temperatures must be ordered")
        object.__setattr__(self, "n", tuple(self.n))
        object.__setattr__(self, "t", tuple(self.t))

    def value(self, temperature_K: float, *, extrapolate_initial_guess: bool = False) -> float:
        if not isfinite(temperature_K) or temperature_K <= 0:
            raise ValueError("ancillary evaluation temperature must be positive")
        if (not extrapolate_initial_guess
                and not self.minimum_temperature_K <= temperature_K <= self.maximum_temperature_K):
            raise ValueError("temperature outside ancillary initial-guess domain")
        theta = 1-temperature_K/self.reducing_temperature_K
        if theta < 0:
            raise ValueError("ancillary cannot supply a supercritical saturation guess")
        value = sum(n*theta**t for n, t in zip(self.n, self.t, strict=True))
        if self.using_tau_r:
            value *= self.reducing_temperature_K/temperature_K
        result = self.reducing_value*((1+value) if self.kind == "rhoLnoexp" else exp(value))
        if not isfinite(result) or result <= 0:
            raise ValueError("ancillary supplied a nonpositive or nonfinite guess")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "n": list(self.n), "t": list(self.t)}


@dataclass(frozen=True)
class HelmholtzSolveReport:
    state: HelmholtzState[float] | None
    success: bool
    failure_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class HelmholtzSaturationReport:
    temperature_K: float
    pressure_Pa: float | None
    liquid: HelmholtzState[float] | None
    gas: HelmholtzState[float] | None
    success: bool
    failure_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


PHASE_ROOT_POLICY = RootPolicy(tolerance=1e-10, maximum_iterations=80)


@dataclass(frozen=True)
class HelmholtzPhaseModel:
    fluid: HelmholtzFluid
    critical_temperature_K: float
    critical_molar_density_mol_m3: float
    critical_pressure_Pa: float
    ancillaries: Mapping[str, HelmholtzAncillary]

    def __post_init__(self) -> None:
        if any(not isfinite(v) or v <= 0 for v in (
                self.critical_temperature_K, self.critical_molar_density_mol_m3,
                self.critical_pressure_Pa)):
            raise ValueError("Helmholtz critical parameters must be positive")
        if set(self.ancillaries) - {"rhoL", "rhoV", "pS", "pL", "pV"}:
            raise ValueError("unknown Helmholtz phase ancillary role")
        if not {"rhoL", "rhoV"} <= self.ancillaries.keys():
            raise ValueError("Helmholtz phase model needs both density guesses")
        object.__setattr__(self, "ancillaries", MappingProxyType(dict(self.ancillaries)))

    def _trial(self, temperature: float, density: float) -> HelmholtzState[float]:
        try:
            state = self.fluid.state(temperature_K=temperature, molar_density_mol_m3=density,
                                    require_stable=False)
            if state.cv_J_mol_K <= 0 or state.dP_drho_T_Pa_m3_mol <= 0:
                raise ValueError("trial crosses homogeneous spinodal or thermal instability")
            return state
        except (ValueError, OverflowError) as exc:
            raise ResidualDomainError(str(exc)) from exc

    def saturation(self, temperature_K: float, *,
                   policy: RootPolicy = PHASE_ROOT_POLICY) -> HelmholtzSaturationReport:
        """Pure-fluid equal-P/equal-mu coexistence, not an ancillary lookup."""
        T = temperature_K
        if self.fluid.pseudo_pure:
            raise ValueError("equal-chemical-potential saturation rejects pseudo-pure EOS")
        if (not isfinite(T)
                or not self.fluid.minimum_temperature_K <= T <= self.fluid.maximum_temperature_K
                or self.critical_temperature_K <= T):
            raise ValueError("pure-fluid saturation requires source Tmin <= T <= Tmax "
                             "and T < Tcritical")
        liquid_guess, gas_guess = (self.ancillaries[key].value(T, extrapolate_initial_guess=True)
                                   for key in ("rhoL", "rhoV"))
        pressure_key = "pS" if "pS" in self.ancillaries else "pL"
        if pressure_key not in self.ancillaries:
            raise ValueError("pure-fluid saturation requires a pressure initial guess")
        # Dense-phase P subtracts O(rho*R*T) contributions; like solve_tp,
        # retain a 1e-5 Pa default absolute residual floor at low pressure.
        pressure_scale = max(1e5, self.ancillaries[pressure_key].value(
            T, extrapolate_initial_guess=True))
        RT = self.fluid.gas_constant_J_mol_K*T
        critical = self.critical_molar_density_mol_m3
        seed_diagnostics: list[dict[str, Any]] = []
        # An ancillary density may lie inside the spinodal near a fitted
        # critical point. Build stable starting coordinates, not accepted
        # states, by a declared finite outward search. No root is retried.
        # The source's nominal rhoc may differ from the EOS critical point;
        # splitting at it can exclude an actual coexistence endpoint.
        seeds = []
        for label, guess, factors in (("liquid", liquid_guess, (1, 1.02, 1.05, 1.1, 1.2, 1.5)),
                                      ("gas", gas_guess, (1, .98, .95, .9, .8, .5))):
            for factor in factors:
                try:
                    self._trial(T, guess*factor)
                except ResidualDomainError as exc:
                    seed_diagnostics.append({"branch": label, "density_multiplier": factor,
                                             "rejected_reason": str(exc)})
                else:
                    seeds.append(guess*factor)
                    seed_diagnostics.append({"branch": label, "density_multiplier": factor,
                                             "accepted_initial_guess": True})
                    break
            else:
                return HelmholtzSaturationReport(T, None, None, None, False,
                    "no_locally_stable_saturation_seed", {"seed_trials": seed_diagnostics})
        liquid_guess, gas_guess = seeds
        if liquid_guess <= gas_guess:
            return HelmholtzSaturationReport(T, None, None, None, False,
                                               "saturation_initial_branches_not_separated")
        separator = (liquid_guess+gas_guess)/2
        def states(x: Any) -> tuple[HelmholtzState[float], HelmholtzState[float]]:
            return self._trial(T, exp(x[0])), self._trial(T, exp(x[1]))

        def residual(x: Any) -> list[float]:
            liquid, gas = states(x)
            return [(liquid.pressure_Pa-gas.pressure_Pa)/pressure_scale,
                    (liquid.gibbs_energy_J_mol-gas.gibbs_energy_J_mol)/RT]

        def jacobian(x: Any) -> list[list[float]]:
            liquid, gas = states(x)
            return [[liquid.molar_density_mol_m3*liquid.dP_drho_T_Pa_m3_mol/pressure_scale,
                     -gas.molar_density_mol_m3*gas.dP_drho_T_Pa_m3_mol/pressure_scale],
                    [liquid.dP_drho_T_Pa_m3_mol/RT, -gas.dP_drho_T_Pa_m3_mol/RT]]

        root = solve_root(residual, [log(liquid_guess), log(gas_guess)], jacobian=jacobian,
            variables=(ResidualVariable("log_rho_liquid", "dimensionless", 1,
                           lower=log(separator*(1+1e-9)),
                           upper=log(max(20*critical, 5*liquid_guess))),
                       ResidualVariable("log_rho_gas", "dimensionless", 1,
                           lower=min(log(1e-16), log(gas_guess)-log(10)),
                           upper=log(separator*(1-1e-9)))),
            residual_scales=[1, 1], policy=policy)
        if not root.success:
            return HelmholtzSaturationReport(T, None, None, None, False,
                root.failure_reason, {**root.diagnostics, "seed_trials": seed_diagnostics})
        liquid, gas = states(root.value)
        if not liquid.locally_stable or not gas.locally_stable:
            return HelmholtzSaturationReport(T, None, None, None, False,
                                               "nonpositive_coexistence_pressure", root.diagnostics)
        separation = liquid.molar_density_mol_m3/gas.molar_density_mol_m3-1
        diagnostics = {**root.diagnostics, "density_separation_ratio": separation,
            "pressure_difference_Pa": liquid.pressure_Pa-gas.pressure_Pa,
            "chemical_potential_difference_J_mol": liquid.gibbs_energy_J_mol-gas.gibbs_energy_J_mol,
            "stability_scope": "two_locally_stable_coexistence_branches",
            "ancillaries_used_as_initial_guesses_only": True, "seed_trials": seed_diagnostics,
            "density_branch_separator_mol_m3": separator,
            "extrapolated_ancillary_initial_guesses": [k for k in ("rhoL", "rhoV", pressure_key)
                if not self.ancillaries[k].minimum_temperature_K <= T
                <= self.ancillaries[k].maximum_temperature_K]}
        if separation <= 1e-7:
            return HelmholtzSaturationReport(T, None, None, None, False,
                                               "unresolved_critical_branch_separation", diagnostics)
        return HelmholtzSaturationReport(T, gas.pressure_Pa, liquid, gas, True,
                                           diagnostics=diagnostics)

    def solve_tp(self, *, temperature_K: float, pressure_Pa: float,
                 phase_hint: str | None = "auto",
                 policy: RootPolicy = PHASE_ROOT_POLICY) -> HelmholtzSolveReport:
        """Resolve one homogeneous density with explicit gas/liquid or automatic branch."""
        T, pressure = temperature_K, pressure_Pa
        phase = "auto" if phase_hint is None else {"vapor": "gas"}.get(phase_hint, phase_hint)
        if phase not in {"auto", "gas", "liquid"}:
            raise ValueError("Helmholtz phase_hint must be auto, gas, liquid or vapor")
        if (not isfinite(pressure) or not 0 < pressure <= self.fluid.maximum_pressure_Pa
                or not self.fluid.minimum_temperature_K <= T <= self.fluid.maximum_temperature_K):
            raise ValueError("TP request outside Helmholtz source validity")
        if self.fluid.pseudo_pure:
            raise ValueError("pseudo-pure TP phase selection is not a pure-fluid equilibrium")
        RT = self.fluid.gas_constant_J_mol_K*T
        critical = self.critical_molar_density_mol_m3
        # Dense-liquid pressure subtracts O(rho*R*T) terms. A fixed 1e5 Pa
        # scale retains a 1e-5 Pa default absolute floor at low pressure,
        # instead of asking Newton to resolve sub-ulp liquid densities.
        pressure_scale = max(pressure, 1e5)
        lower, upper = 1e-16, 20*critical
        initial = pressure/RT
        diagnostics: dict[str, Any] = {"requested_phase": phase, "metastable": False}
        if self.critical_temperature_K > T:
            saturated = self.saturation(T, policy=policy)
            if not saturated.success:
                return HelmholtzSolveReport(None, False, "saturation:"+saturated.failure_reason,
                                            saturated.diagnostics)
            if saturated.pressure_Pa is None:
                raise AssertionError("invariant violated: saturated.pressure_Pa is None")
            if saturated.liquid is None or saturated.gas is None:
                raise AssertionError(
                    "invariant violated: saturated.liquid or saturated.gas is None"
                )
            stable_phase = "gas" if pressure < saturated.pressure_Pa else "liquid"
            if phase == "auto" and abs(pressure-saturated.pressure_Pa) <= max(
                    1e-4, saturated.pressure_Pa*1e-8):
                return HelmholtzSolveReport(None, False, "two_phase_TP_density_is_not_unique")
            if phase == "auto":
                phase = stable_phase
            diagnostics.update(selected_phase=phase, metastable=phase != stable_phase,
                saturation_pressure_Pa=saturated.pressure_Pa,
                stability_scope="relative_to_resolved_liquid_gas_coexistence")
            if phase == "gas":
                upper = (saturated.liquid.molar_density_mol_m3
                         +saturated.gas.molar_density_mol_m3)/2
                initial = min(pressure/RT, saturated.gas.molar_density_mol_m3)
            else:
                lower = (saturated.liquid.molar_density_mol_m3
                         +saturated.gas.molar_density_mol_m3)/2
                initial = saturated.liquid.molar_density_mol_m3
        else:
            diagnostics.update(selected_phase="supercritical_homogeneous",
                               stability_scope="homogeneous_local_only")

        def state(x: Any) -> HelmholtzState[float]:
            return self._trial(T, exp(x[0]))

        def residual(x: Any) -> list[float]:
            return [(state(x).pressure_Pa-pressure)/pressure_scale]

        def jacobian(x: Any) -> list[list[float]]:
            current = state(x)
            return [[current.molar_density_mol_m3*current.dP_drho_T_Pa_m3_mol/pressure_scale]]

        root = solve_root(residual, [log(initial)], jacobian=jacobian,
            variables=(ResidualVariable("log_rho", "dimensionless", 1,
                                         lower=log(lower), upper=log(upper)),),
            residual_scales=[1], policy=policy)
        diagnostics.update(root.diagnostics)
        final = state(root.value) if root.success else None
        if final is not None:
            diagnostics["pressure_difference_Pa"] = final.pressure_Pa-pressure
            diagnostics["pressure_residual_scale_Pa"] = pressure_scale
            if not final.locally_stable:
                return HelmholtzSolveReport(None, False, "nonpositive_final_pressure", diagnostics)
        return HelmholtzSolveReport(final, root.success, root.failure_reason, diagnostics)

    def solve_homogeneous_state(self, inputs: Mapping[str, float], *,
                                initial_state: HelmholtzState[float],
                                policy: RootPolicy = PHASE_ROOT_POLICY) -> HelmholtzSolveReport:
        """Two-input local homogeneous closure; two-phase flash remains separate.

        Coordinates use K, Pa, mol/m3, J/mol, and J/(mol K). Initial state is
        explicit because arbitrary caloric pairs can have several roots.
        """
        supported = {"T", "P", "rho", "H", "U", "S"}
        if len(inputs) != 2 or set(inputs)-supported:
            raise ValueError("homogeneous closure requires two distinct supported coordinates")
        if any(isinstance(v, bool) or not isfinite(v) for v in inputs.values()):
            raise ValueError("homogeneous target coordinates must be finite")
        if any(inputs[k] <= 0 for k in ("T", "P", "rho") if k in inputs):
            raise ValueError("temperature, pressure and density targets must be positive")
        if initial_state.fluid_id != self.fluid.fluid_id:
            raise ValueError("initial homogeneous state has a different fluid identity")
        keys = tuple(inputs)
        scales = [max(abs(inputs[k]), 1.0) for k in keys]

        def state(x: Any) -> HelmholtzState[float]:
            return self._trial(exp(x[0]), exp(x[1]))

        def values(current: HelmholtzState[float]) -> dict[str, float]:
            return {"T": current.temperature_K, "rho": current.molar_density_mol_m3,
                    "P": current.pressure_Pa, "H": current.enthalpy_J_mol,
                    "U": current.internal_energy_J_mol, "S": current.entropy_J_mol_K}

        def residual(x: Any) -> list[float]:
            current = values(state(x))
            return [(current[k]-inputs[k])/scale for k, scale in zip(keys, scales, strict=True)]

        def jacobian(x: Any) -> list[list[float]]:
            current = state(x)
            derivatives = homogeneous_state_derivatives(self.fluid, current)
            return [[derivatives[k][0]*current.temperature_K/scale,
                     derivatives[k][1]*current.molar_density_mol_m3/scale]
                    for k, scale in zip(keys, scales, strict=True)]

        root = solve_root(residual,
            [log(initial_state.temperature_K), log(initial_state.molar_density_mol_m3)],
            jacobian=jacobian, variables=(
                ResidualVariable("log_T", "dimensionless", 1,
                    lower=log(self.fluid.minimum_temperature_K),
                    upper=log(self.fluid.maximum_temperature_K)),
                ResidualVariable("log_rho", "dimensionless", 1,
                    lower=log(1e-16), upper=log(20*self.critical_molar_density_mol_m3))),
            residual_scales=[1, 1], policy=policy)
        final = state(root.value) if root.success else None
        if final is not None and not final.locally_stable:
            return HelmholtzSolveReport(None, False, "nonpositive_final_pressure", root.diagnostics)
        return HelmholtzSolveReport(final,
            root.success, root.failure_reason,
            {**root.diagnostics, "stability_scope": "homogeneous_local_only",
             "global_phase_selection_performed": False})

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld_helmholtz_phase_v1", "fluid": self.fluid.to_dict(),
                "critical_temperature_K": self.critical_temperature_K,
                "critical_molar_density_mol_m3": self.critical_molar_density_mol_m3,
                "critical_pressure_Pa": self.critical_pressure_Pa,
                "ancillaries": {k: v.to_dict() for k, v in self.ancillaries.items()}}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HelmholtzPhaseModel:
        values = dict(data)
        if values.pop("model", None) != "chemworld_helmholtz_phase_v1":
            raise ValueError("unknown Helmholtz phase model")
        values["fluid"] = HelmholtzFluid.from_dict(values["fluid"])
        values["ancillaries"] = {k: HelmholtzAncillary(**v)
                                 for k, v in values["ancillaries"].items()}
        return cls(**values)


def homogeneous_state_derivatives(fluid: HelmholtzFluid, state: HelmholtzState[float]
                                   ) -> dict[str, tuple[float, float]]:
    """Same-potential (d/dT|rho, d/drho|T) for caloric closure and real inventories."""
    if state.fluid_id != fluid.fluid_id:
        raise ValueError("Helmholtz derivative state has a different fluid identity")
    T, rho = state.temperature_K, state.molar_density_mol_m3
    tau = fluid.reducing_temperature_K/T
    artd = fluid.alpha_derivative("residual", delta=rho/fluid.reducing_molar_density_mol_m3,
                                 tau=tau, delta_order=1, tau_order=1)
    urho = fluid.gas_constant_J_mol_K*T*tau*artd/fluid.reducing_molar_density_mol_m3
    pT, prho, cv = state.dP_dT_rho_Pa_K, state.dP_drho_T_Pa_m3_mol, state.cv_J_mol_K
    return {"T": (1, 0), "rho": (0, 1), "P": (pT, prho), "U": (cv, urho),
            "H": (cv+pT/rho, urho+prho/rho-state.pressure_Pa/rho**2),
            "S": (cv/T, (urho-state.pressure_Pa/rho**2)/T)}
