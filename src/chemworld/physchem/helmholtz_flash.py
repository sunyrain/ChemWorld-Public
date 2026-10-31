"""Pure-fluid equilibrium closure on the same Helmholtz coexistence manifold.

Quality is molar vapor fraction (also mass fraction for one pure species).
Two-phase Cp, sound speed and transport are deliberately not manufactured from
weighted homogeneous responses: they require a declared process/time scale.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import exp, isfinite, log
from types import MappingProxyType
from typing import Any

from chemworld.physchem.helmholtz import HelmholtzState
from chemworld.physchem.helmholtz_solvers import (
    PHASE_ROOT_POLICY,
    HelmholtzPhaseModel,
    HelmholtzSaturationReport,
    homogeneous_state_derivatives,
)
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    solve_root,
)


@dataclass(frozen=True)
class HelmholtzTwoPhaseState:
    liquid: HelmholtzState[float]
    gas: HelmholtzState[float]
    vapor_fraction: float

    def __post_init__(self) -> None:
        liquid, gas, q = self.liquid, self.gas, self.vapor_fraction
        if isinstance(q, bool) or not isfinite(q) or not 0 <= q <= 1:
            raise ValueError("pure-fluid vapor fraction must be between zero and one")
        if liquid.fluid_id != gas.fluid_id:
            raise ValueError("two-phase endpoints have different fluid identities")
        if not liquid.locally_stable or not gas.locally_stable:
            raise ValueError("two-phase endpoints must be locally stable")
        if (abs(liquid.temperature_K-gas.temperature_K) > 1e-8
                or abs(liquid.pressure_Pa-gas.pressure_Pa) > max(1e-4, 1e-8*gas.pressure_Pa)
                or abs(liquid.gibbs_energy_J_mol-gas.gibbs_energy_J_mol) > 1e-4
                or liquid.molar_density_mol_m3 <= gas.molar_density_mol_m3):
            raise ValueError("two-phase endpoints do not satisfy resolved coexistence")

    @property
    def fluid_id(self) -> str:
        return self.liquid.fluid_id

    @property
    def temperature_K(self) -> float:  # noqa: N802 - same SI coordinate as HelmholtzState
        return self.liquid.temperature_K

    @property
    def pressure_Pa(self) -> float:  # noqa: N802
        return self.gas.pressure_Pa

    @property
    def molar_density_mol_m3(self) -> float:
        return 1/((1-self.vapor_fraction)/self.liquid.molar_density_mol_m3
                  + self.vapor_fraction/self.gas.molar_density_mol_m3)

    @property
    def mass_density_kg_m3(self) -> float:
        return self.molar_density_mol_m3*(
            self.liquid.mass_density_kg_m3/self.liquid.molar_density_mol_m3)

    def _weighted(self, attribute: str) -> float:
        return ((1-self.vapor_fraction)*getattr(self.liquid, attribute)
                + self.vapor_fraction*getattr(self.gas, attribute))

    @property
    def enthalpy_J_mol(self) -> float:  # noqa: N802
        return self._weighted("enthalpy_J_mol")

    @property
    def internal_energy_J_mol(self) -> float:  # noqa: N802
        return self._weighted("internal_energy_J_mol")

    @property
    def entropy_J_mol_K(self) -> float:  # noqa: N802
        return self._weighted("entropy_J_mol_K")

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld_helmholtz_two_phase_v1", "fluid_id": self.fluid_id,
                "vapor_fraction": self.vapor_fraction, "temperature_K": self.temperature_K,
                "pressure_Pa": self.pressure_Pa,
                "molar_density_mol_m3": self.molar_density_mol_m3,
                "enthalpy_J_mol": self.enthalpy_J_mol,
                "internal_energy_J_mol": self.internal_energy_J_mol,
                "entropy_J_mol_K": self.entropy_J_mol_K,
                "liquid": self.liquid.to_dict(), "gas": self.gas.to_dict()}


@dataclass(frozen=True)
class HelmholtzFlashReport:
    state: HelmholtzState[float] | HelmholtzTwoPhaseState | None
    success: bool
    phase: str = "unresolved"
    failure_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


def _values(state: HelmholtzState[float] | HelmholtzTwoPhaseState) -> dict[str, float]:
    result = {"T": state.temperature_K, "P": state.pressure_Pa,
              "rho": state.molar_density_mol_m3, "H": state.enthalpy_J_mol,
              "U": state.internal_energy_J_mol, "S": state.entropy_J_mol_K}
    if isinstance(state, HelmholtzTwoPhaseState):
        result["Q"] = state.vapor_fraction
    return result


def _temperature_bounds(model: HelmholtzPhaseModel) -> tuple[float, float]:
    if model.fluid.pseudo_pure:
        raise ValueError("pure-fluid equilibrium closure rejects pseudo-pure EOS")
    # Ancillary fit windows constrain guesses, not the original energy model.
    minimum = model.fluid.minimum_temperature_K
    maximum = min(model.critical_temperature_K*(1-1e-9), model.fluid.maximum_temperature_K)
    return minimum, maximum


def _two_phase(saturation: HelmholtzSaturationReport, quality: float) -> HelmholtzTwoPhaseState:
    if not saturation.success:
        raise ResidualDomainError("saturation:"+saturation.failure_reason)
    if saturation.liquid is None or saturation.gas is None:
        raise AssertionError("invariant violated: saturation.liquid or saturation.gas is None")
    return HelmholtzTwoPhaseState(saturation.liquid, saturation.gas, quality)


def coexistence_derivatives(model: HelmholtzPhaseModel,
                           state: HelmholtzTwoPhaseState) -> dict[str, tuple[float, float]]:
    """Analytic (d/dT|Q, d/dQ|T) on the pure-fluid equilibrium manifold."""
    if state.fluid_id != model.fluid.fluid_id:
        raise ValueError("coexistence state has a different fluid identity")
    liquid, gas, q = state.liquid, state.gas, state.vapor_fraction
    vl, vg = 1/liquid.molar_density_mol_m3, 1/gas.molar_density_mol_m3
    pT = (gas.entropy_J_mol_K-liquid.entropy_J_mol_K)/(vg-vl)
    result = {"T": (1.0, 0.0), "Q": (0.0, 1.0), "P": (pT, 0.0)}
    derivatives = []
    vT = []
    for endpoint in (liquid, gas):
        rhoT = (pT-endpoint.dP_dT_rho_Pa_K)/endpoint.dP_drho_T_Pa_m3_mol
        derivatives.append({k: dt+dr*rhoT for k, (dt, dr) in
                            homogeneous_state_derivatives(model.fluid, endpoint).items()})
        vT.append(-rhoT/endpoint.molar_density_mol_m3**2)
    result["rho"] = (-state.molar_density_mol_m3**2*((1-q)*vT[0]+q*vT[1]),
                     -state.molar_density_mol_m3**2*(vg-vl))
    liquid_values, gas_values = _values(liquid), _values(gas)
    for key in ("H", "U", "S"):
        result[key] = ((1-q)*derivatives[0][key]+q*derivatives[1][key],
                       gas_values[key]-liquid_values[key])
    return result


def saturation_at_pressure(model: HelmholtzPhaseModel, pressure_Pa: float, *,
                           initial_temperature_K: float | None = None,
                           policy: RootPolicy = PHASE_ROOT_POLICY) -> HelmholtzSaturationReport:
    """Invert same-EOS coexistence, using the source ancillary only for an initial guess."""
    lower, upper = _temperature_bounds(model)
    if not isfinite(pressure_Pa) or not 0 < pressure_Pa < model.critical_pressure_Pa:
        raise ValueError("saturation pressure must lie between zero and critical pressure")
    pressure_key = "pS" if "pS" in model.ancillaries else "pL"
    if pressure_key not in model.ancillaries:
        raise ValueError("saturation pressure closure requires a pressure ancillary guess")
    scale = max(1e4, pressure_Pa)
    variable = (ResidualVariable("log_T", "dimensionless", 1,
                                  lower=log(lower), upper=log(upper)),)
    if initial_temperature_K is None:
        ancillary = model.ancillaries[pressure_key]

        def guess_jacobian(x: Any) -> list[list[float]]:
            T = exp(x[0])
            theta = 1-T/ancillary.reducing_temperature_K
            coefficients = tuple(zip(ancillary.n, ancillary.t, strict=True))
            slope = sum(n*t*theta**(t-1) for n, t in coefficients if t != 0)
            if ancillary.using_tau_r:
                value = sum(n*theta**t for n, t in coefficients)
                return [[-ancillary.reducing_temperature_K/T*value-slope]]
            return [[-T/ancillary.reducing_temperature_K*slope]]

        guess = solve_root(lambda x: [log(ancillary.value(exp(x[0]),
            extrapolate_initial_guess=True)/pressure_Pa)],
                           [log((lower+upper)/2)], variables=variable,
                           residual_scales=[1], policy=policy, jacobian=guess_jacobian)
        if not guess.success:
            return HelmholtzSaturationReport((lower+upper)/2, None, None, None, False,
                                               "pressure_guess:"+guess.failure_reason,
                                               guess.diagnostics)
        initial_temperature_K = exp(guess.value[0])
    if not isfinite(initial_temperature_K) or not lower <= initial_temperature_K <= upper:
        raise ValueError("initial saturation temperature outside declared domain")
    cached: dict[float, HelmholtzTwoPhaseState] = {}

    def state(x: Any) -> HelmholtzTwoPhaseState:
        T = exp(x[0])
        if T not in cached:
            cached[T] = _two_phase(model.saturation(T, policy=policy), .5)
        return cached[T]

    root = solve_root(lambda x: [(state(x).pressure_Pa-pressure_Pa)/scale],
        [log(initial_temperature_K)],
        jacobian=lambda x: [[coexistence_derivatives(model, state(x))["P"][0]
                            *state(x).temperature_K/scale]],
        variables=variable, residual_scales=[1], policy=policy)
    if not root.success:
        return HelmholtzSaturationReport(initial_temperature_K, None, None, None, False,
                                           root.failure_reason, root.diagnostics)
    final = state(root.value)
    return HelmholtzSaturationReport(final.temperature_K, final.pressure_Pa,
        final.liquid, final.gas, True, diagnostics={**root.diagnostics,
            "pressure_difference_Pa": final.pressure_Pa-pressure_Pa,
            "equilibrium_model": "same_EOS_equal_pressure_and_chemical_potential"})


def solve_two_phase_state(model: HelmholtzPhaseModel, inputs: Mapping[str, float], *,
                          initial_temperature_K: float | None = None,
                          initial_vapor_fraction: float = .5,
                          policy: RootPolicy = PHASE_ROOT_POLICY) -> HelmholtzFlashReport:
    """All independent pairs from T/P/rho/H/U/S/Q on the coexistence manifold.

    T/P does not determine quality and is rejected. Pairs without T or P need
    an explicit temperature guess because their equilibrium solution may not
    be globally unique. Quality endpoints are accepted without clipping.
    """
    lower, upper = _temperature_bounds(model)
    supported = {"T", "P", "rho", "H", "U", "S", "Q"}
    if len(inputs) != 2 or set(inputs)-supported:
        raise ValueError("two-phase closure needs two distinct supported coordinates")
    if any(isinstance(v, bool) or not isfinite(v) for v in inputs.values()):
        raise ValueError("two-phase coordinates must be finite")
    if any(inputs[k] <= 0 for k in ("T", "P", "rho") if k in inputs):
        raise ValueError("temperature, pressure and density must be positive")
    if "Q" in inputs and not 0 <= inputs["Q"] <= 1:
        raise ValueError("vapor fraction must be between zero and one")
    if set(inputs) == {"T", "P"}:
        return HelmholtzFlashReport(None, False,
                                    failure_reason="two_phase_TP_quality_is_not_unique")
    if not isfinite(initial_vapor_fraction) or not 0 <= initial_vapor_fraction <= 1:
        raise ValueError("initial vapor fraction must be between zero and one")
    if "T" in inputs or "P" in inputs:
        saturation = (model.saturation(inputs["T"], policy=policy) if "T" in inputs else
            saturation_at_pressure(model, inputs["P"],
                                   initial_temperature_K=initial_temperature_K, policy=policy))
        if not saturation.success:
            return HelmholtzFlashReport(None, False, failure_reason=saturation.failure_reason,
                                        diagnostics=saturation.diagnostics)
        midpoint = _two_phase(saturation, .5)
        key = next(k for k in inputs if k not in {"T", "P"})
        if key == "Q":
            quality = inputs[key]
        elif key == "rho":
            quality = ((1/inputs[key]-1/midpoint.liquid.molar_density_mol_m3)
                       /(1/midpoint.gas.molar_density_mol_m3
                         -1/midpoint.liquid.molar_density_mol_m3))
        else:
            lv, gv = _values(midpoint.liquid)[key], _values(midpoint.gas)[key]
            quality = (inputs[key]-lv)/(gv-lv)
        if not 0 <= quality <= 1:
            return HelmholtzFlashReport(None, False, failure_reason="target_outside_coexistence",
                                        diagnostics={"unconstrained_quality": quality})
        return HelmholtzFlashReport(_two_phase(saturation, quality), True, "two_phase",
            diagnostics={**saturation.diagnostics, "global_uniqueness_claimed": False})
    if (initial_temperature_K is None or not isfinite(initial_temperature_K)
            or not lower <= initial_temperature_K <= upper):
        raise ValueError("caloric coexistence pair needs an explicit in-domain initial temperature")
    keys = tuple(inputs)
    scales = [max(1.0, abs(inputs[k])) for k in keys]
    cached: dict[float, HelmholtzSaturationReport] = {}

    def state(x: Any) -> HelmholtzTwoPhaseState:
        T = exp(x[0])
        if T not in cached:
            cached[T] = model.saturation(T, policy=policy)
        return _two_phase(cached[T], x[1])

    def residual(x: Any) -> list[float]:
        values = _values(state(x))
        return [(values[k]-inputs[k])/scale for k, scale in zip(keys, scales, strict=True)]

    def jacobian(x: Any) -> list[list[float]]:
        current = state(x)
        derivatives = coexistence_derivatives(model, current)
        return [[derivatives[k][0]*current.temperature_K/scale, derivatives[k][1]/scale]
                for k, scale in zip(keys, scales, strict=True)]

    root = solve_root(residual, [log(initial_temperature_K), initial_vapor_fraction],
        jacobian=jacobian, variables=(ResidualVariable(
            "log_T", "dimensionless", 1, lower=log(lower), upper=log(upper)),
            ResidualVariable("Q", "dimensionless", 1, lower=0, upper=1)),
        residual_scales=[1, 1], policy=policy)
    return HelmholtzFlashReport(state(root.value) if root.success else None,
        root.success, "two_phase" if root.success else "unresolved", root.failure_reason,
        {**root.diagnostics, "global_uniqueness_claimed": False,
         "phase_scope": "pure_fluid_liquid_gas_coexistence"})


def flash_at_pressure(model: HelmholtzPhaseModel, *, pressure_Pa: float,
                       coordinate: str, value: float,
                       initial_state: HelmholtzState[float] | None = None,
                       policy: RootPolicy = PHASE_ROOT_POLICY) -> HelmholtzFlashReport:
    """Stable pure-fluid PH/PS/PU/Prho flash, including the two-phase interval.

    Above critical pressure an explicit initial homogeneous state is needed.
    Below it, the saturation endpoints supply the appropriate branch guess.
    A successful single-phase result is checked against its own coexistence.
    """
    _temperature_bounds(model)
    if coordinate not in {"H", "U", "S", "rho"}:
        raise ValueError("pressure flash coordinate must be H, U, S or rho")
    if not isfinite(value) or (coordinate == "rho" and value <= 0):
        raise ValueError("pressure flash target must be finite with positive density")
    if not isfinite(pressure_Pa) or not 0 < pressure_Pa <= model.fluid.maximum_pressure_Pa:
        raise ValueError("flash pressure outside source validity")
    if pressure_Pa < model.critical_pressure_Pa:
        saturation = saturation_at_pressure(model, pressure_Pa, policy=policy)
        if not saturation.success:
            return HelmholtzFlashReport(None, False, failure_reason=saturation.failure_reason,
                                        diagnostics=saturation.diagnostics)
        if saturation.liquid is None or saturation.gas is None:
            raise AssertionError("invariant violated: saturation.liquid or saturation.gas is None")
        liquid, gas = saturation.liquid, saturation.gas
        lv, gv = _values(liquid)[coordinate], _values(gas)[coordinate]
        if min(lv, gv) <= value <= max(lv, gv):
            quality = ((1/value-1/lv)/(1/gv-1/lv) if coordinate == "rho"
                       else (value-lv)/(gv-lv))
            return HelmholtzFlashReport(_two_phase(saturation, quality), True, "two_phase",
                diagnostics={"phase_scope": "pure_fluid_liquid_gas_coexistence"})
        # H, U and S increase from liquid to gas; density is reversed.
        liquid_side = value > lv if coordinate == "rho" else value < lv
        initial_state = liquid if liquid_side else gas
    if initial_state is None:
        raise ValueError("above-critical pressure flash needs an explicit homogeneous "
                         "initial state")
    report = model.solve_homogeneous_state({"P": pressure_Pa, coordinate: value},
                                            initial_state=initial_state, policy=policy)
    if not report.success:
        return HelmholtzFlashReport(None, False, failure_reason=report.failure_reason,
                                    diagnostics=report.diagnostics)
    if report.state is None:
        raise AssertionError("invariant violated: report.state is not None")
    final = report.state
    if final.temperature_K < model.critical_temperature_K:
        stable = model.solve_tp(temperature_K=final.temperature_K, pressure_Pa=pressure_Pa,
                                policy=policy)
        if (not stable.success or stable.state is None
                or abs(stable.state.molar_density_mol_m3/final.molar_density_mol_m3-1) > 1e-7):
            return HelmholtzFlashReport(None, False,
                                        failure_reason="homogeneous_root_not_stable_phase",
                                        diagnostics=report.diagnostics)
        phase = str(stable.diagnostics["selected_phase"])
    else:
        phase = "supercritical_homogeneous"
    return HelmholtzFlashReport(final, True, phase, diagnostics={**report.diagnostics,
        "stability_scope": "relative_to_resolved_liquid_gas_coexistence",
        "global_phase_selection_performed": True})
