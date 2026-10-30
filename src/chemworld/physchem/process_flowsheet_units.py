"""Algebraic flowsheet devices, retaining the original molecular EOS and units.

No device here owns inventory or time. Heat exchange is a coupled two-sided PH
problem; pressure recovery executes the original PS/PH expander and compressor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import fsum, isfinite, log
from typing import Any

import numpy as np

from chemworld.physchem.molecular_mixture import MolecularStream
from chemworld.physchem.molecular_processes import (
    MolecularPressureStage,
    MolecularProcessDomain,
    MolecularProcessResult,
    molecular_heat_duty,
    molecular_mix_streams,
    molecular_process_state,
    molecular_split_stream,
)
from chemworld.physchem.residual_system import (
    DEFAULT_ROOT_POLICY,
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)


@dataclass(frozen=True)
class ProcessMixer:
    domain: MolecularProcessDomain
    outlet_pressure_Pa: float
    initial_temperature_K: float

    def evaluate(
        self, inlets: Sequence[MolecularStream], *, policy: RootPolicy = DEFAULT_ROOT_POLICY
    ) -> MolecularProcessResult:
        return molecular_mix_streams(
            self.domain,
            inlets,
            outlet_pressure_Pa=self.outlet_pressure_Pa,
            initial_temperature_K=self.initial_temperature_K,
            policy=policy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "mixer",
            "domain": self.domain.to_dict(),
            "outlet_pressure_Pa": self.outlet_pressure_Pa,
            "initial_temperature_K": self.initial_temperature_K,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProcessMixer:
        return cls(
            MolecularProcessDomain.from_dict(value["domain"]),
            value["outlet_pressure_Pa"],
            value["initial_temperature_K"],
        )


@dataclass(frozen=True)
class ProcessSplitter:
    domain: MolecularProcessDomain
    fractions: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fractions", tuple(self.fractions))
        if (
            not self.fractions
            or any(not isfinite(x) or x < 0 for x in self.fractions)
            or abs(fsum(self.fractions) - 1) > 1e-12
        ):
            raise ValueError("split fractions must be nonnegative and sum to one")

    def evaluate(self, inlet: MolecularStream) -> MolecularProcessResult:
        return molecular_split_stream(self.domain, inlet, self.fractions)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "splitter", "domain": self.domain.to_dict(), "fractions": self.fractions}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProcessSplitter:
        return cls(MolecularProcessDomain.from_dict(value["domain"]), tuple(value["fractions"]))


def _lmtd(first: float, second: float) -> float:
    if first < -1e-8 or second < -1e-8:
        raise ResidualDomainError("heat_exchanger_temperature_cross")
    if min(first, second) <= 0:
        return 0.0
    if abs(first - second) < 1e-6 * max(first, second):
        # The limit is continuous; second-order expansion avoids cancellation.
        mean = (first + second) / 2
        return mean - (first - second) ** 2 / (12 * mean)
    return (first - second) / log(first / second)


@dataclass(frozen=True)
class MolecularHeatExchanger:
    hot_domain: MolecularProcessDomain
    cold_domain: MolecularProcessDomain
    conductance_W_K: float
    hot_outlet_pressure_Pa: float | None = None
    cold_outlet_pressure_Pa: float | None = None
    maximum_trial_evaluations: int = 160

    def __post_init__(self) -> None:
        if (
            not isfinite(self.conductance_W_K)
            or self.conductance_W_K < 0
            or type(self.maximum_trial_evaluations) is not int
            or self.maximum_trial_evaluations < 1
            or any(
                p is not None and (not isfinite(p) or p <= 0)
                for p in (self.hot_outlet_pressure_Pa, self.cold_outlet_pressure_Pa)
            )
        ):
            raise ValueError("exchanger needs finite nonnegative UA, pressures and trial budget")

    def evaluate(
        self,
        hot: MolecularStream,
        cold: MolecularStream,
        *,
        policy: RootPolicy = DEFAULT_ROOT_POLICY,
    ) -> MolecularProcessResult:
        self.hot_domain.validate_stream(hot)
        self.cold_domain.validate_stream(cold)
        diagnostics: dict[str, Any] = {"mode": "molecular_countercurrent_UA", "trials": []}
        if min(hot.molar_rate_mol_s, cold.molar_rate_mol_s) <= 0:
            return MolecularProcessResult(False, failure_reason="exchanger_requires_positive_flows")
        if hot.state.temperature_K < cold.state.temperature_K:
            return MolecularProcessResult(False, failure_reason="exchanger_hot_inlet_is_colder")
        hp = self.hot_outlet_pressure_Pa or hot.state.pressure_Pa
        cp = self.cold_outlet_pressure_Pa or cold.state.pressure_Pa
        if any(
            p > s.state.pressure_Pa + max(1e-5, 1e-9 * s.state.pressure_Pa)
            for p, s in ((hp, hot), (cp, cold))
        ):
            return MolecularProcessResult(False, failure_reason="exchanger_pressure_increase")

        def endpoint(
            domain: MolecularProcessDomain,
            stream: MolecularStream,
            pressure: float,
            temperature: float,
        ) -> float:
            result = molecular_process_state(
                domain,
                pressure_Pa=pressure,
                coordinate="T",
                value=temperature,
                mole_fractions=stream.state.mole_fractions,
                initial_temperature_K=temperature,
                initial_density_mol_m3=stream.state.molar_density_mol_m3,
                policy=policy,
            )
            diagnostics.setdefault("endpoint_closures", []).append(result.to_dict())
            if not result.success or result.state is None:
                raise ResidualDomainError("exchanger_endpoint:" + result.failure_reason)
            return result.state.enthalpy_J_mol * stream.molar_rate_mol_s

        cache: dict[float, tuple[MolecularStream, MolecularStream, float]] = {}

        def trial(q: float) -> tuple[MolecularStream, MolecularStream, float]:
            if q in cache:
                return cache[q]
            if len(diagnostics["trials"]) >= self.maximum_trial_evaluations:
                raise ResidualDomainError("exchanger_trial_budget_exhausted")
            row: dict[str, Any] = {"internal_heat_W": q}
            diagnostics["trials"].append(row)
            h = molecular_heat_duty(
                self.hot_domain, hot, heat_duty_W=-q, outlet_pressure_Pa=hp, policy=policy
            )
            c = molecular_heat_duty(
                self.cold_domain, cold, heat_duty_W=q, outlet_pressure_Pa=cp, policy=policy
            )
            row.update(hot=h.to_dict(), cold=c.to_dict())
            if not h.success or not c.success or h.outlet is None or c.outlet is None:
                raise ResidualDomainError(
                    "exchanger_PH:" + h.failure_reason + ":" + c.failure_reason
                )
            dt1 = hot.state.temperature_K - c.outlet.state.temperature_K
            dt2 = h.outlet.state.temperature_K - cold.state.temperature_K
            residual = q - self.conductance_W_K * _lmtd(dt1, dt2)
            row.update(residual_W=residual, terminal_differences_K=[dt1, dt2])
            cache[q] = (h.outlet, c.outlet, residual)
            return cache[q]

        try:
            hot_limit = max(cold.state.temperature_K, self.hot_domain.temperature_range_K[0])
            cold_limit = min(hot.state.temperature_K, self.cold_domain.temperature_range_K[1])
            cap = min(
                hot.enthalpy_rate_W - endpoint(self.hot_domain, hot, hp, hot_limit),
                endpoint(self.cold_domain, cold, cp, cold_limit) - cold.enthalpy_rate_W,
            )
            diagnostics["heat_upper_bound_W"] = cap
            if cap < -1e-8:
                raise ResidualDomainError("exchanger_no_feasible_heat_interval")
            if cap <= 1e-12 or self.conductance_W_K == 0:
                q = 0.0
                hout, cout, residual = trial(q)
                if abs(residual) > policy.tolerance * max(1.0, abs(cap)):
                    raise ResidualDomainError("exchanger_no_feasible_UA_root")
            else:
                scale = np.array([max(cap, 1.0)])

                def function(point: np.ndarray) -> np.ndarray:
                    return np.array([trial(float(point[0]))[2]])

                root = solve_root(
                    function,
                    [cap * 0.5],
                    jacobian=lambda p: finite_jacobian(
                        function, p, scale, lower=[0.0], upper=[cap]
                    ),
                    variables=(ResidualVariable("heat", "W", scale[0], "algebraic", 0.0, cap),),
                    residual_scales=scale,
                    policy=policy,
                )
                diagnostics["root"] = root.diagnostics
                if not root.success:
                    raise ResidualDomainError("exchanger_root:" + root.failure_reason)
                q = float(root.value[0])
                hout, cout, residual = trial(q)
            energy = (
                hout.enthalpy_rate_W
                + cout.enthalpy_rate_W
                - (hot.enthalpy_rate_W + cold.enthalpy_rate_W)
            )
            entropy = fsum(
                (
                    out.molar_rate_mol_s * out.state.entropy_J_mol_K
                    - inlet.molar_rate_mol_s * inlet.state.entropy_J_mol_K
                )
                for inlet, out in ((hot, hout), (cold, cout))
            )
            diagnostics.update(
                internal_heat_W=q,
                energy_residual_W=energy,
                entropy_generation_W_K=entropy,
                UA_residual_W=residual,
                trial_evaluations=len(diagnostics["trials"]),
            )
            if entropy < -1e-7:
                raise ResidualDomainError("exchanger_negative_entropy_generation")
            return MolecularProcessResult(True, (hout, cout), diagnostics=diagnostics)
        except (ValueError, ArithmeticError) as exc:
            diagnostics["trial_evaluations"] = len(diagnostics["trials"])
            return MolecularProcessResult(False, failure_reason=str(exc), diagnostics=diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "heat_exchanger",
            "hot_domain": self.hot_domain.to_dict(),
            "cold_domain": self.cold_domain.to_dict(),
            "conductance_W_K": self.conductance_W_K,
            "hot_outlet_pressure_Pa": self.hot_outlet_pressure_Pa,
            "cold_outlet_pressure_Pa": self.cold_outlet_pressure_Pa,
            "maximum_trial_evaluations": self.maximum_trial_evaluations,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MolecularHeatExchanger:
        return cls(
            MolecularProcessDomain.from_dict(value["hot_domain"]),
            MolecularProcessDomain.from_dict(value["cold_domain"]),
            value["conductance_W_K"],
            value.get("hot_outlet_pressure_Pa"),
            value.get("cold_outlet_pressure_Pa"),
            value["maximum_trial_evaluations"],
        )


@dataclass(frozen=True)
class MolecularPressureRecovery:
    expander: MolecularPressureStage
    compressor: MolecularPressureStage
    transmission_efficiency: float = 1.0
    allow_external_work: bool = True

    def __post_init__(self) -> None:
        if (
            self.expander.mode != "expander"
            or self.compressor.mode != "compressor"
            or not isfinite(self.transmission_efficiency)
            or not 0 < self.transmission_efficiency <= 1
            or type(self.allow_external_work) is not bool
        ):
            raise ValueError(
                "pressure recovery requires an expander/compressor pair and efficiency"
            )

    def evaluate(
        self,
        donor: MolecularStream,
        receiver: MolecularStream,
        *,
        policy: RootPolicy = DEFAULT_ROOT_POLICY,
    ) -> MolecularProcessResult:
        expanded = self.expander.evaluate(donor, policy=policy)
        compressed = self.compressor.evaluate(receiver, policy=policy)
        diagnostics: dict[str, Any] = {
            "mode": "molecular_pressure_recovery",
            "expander": expanded.to_dict(),
            "compressor": compressed.to_dict(),
        }
        if (
            not expanded.success
            or not compressed.success
            or expanded.outlet is None
            or compressed.outlet is None
        ):
            return MolecularProcessResult(
                False, failure_reason="pressure_recovery_unit_failure", diagnostics=diagnostics
            )
        generated = -expanded.shaft_work_W
        recovered = self.transmission_efficiency * generated
        loss = (1 - self.transmission_efficiency) * generated
        external = compressed.shaft_work_W - recovered
        diagnostics.update(
            generated_work_W=generated,
            recovered_work_W=recovered,
            compressor_work_W=compressed.shaft_work_W,
            transmission_loss_W=loss,
            external_draw_W=max(external, 0.0),
            external_export_W=max(-external, 0.0),
        )
        if not self.allow_external_work and external > 1e-8:
            return MolecularProcessResult(
                False, failure_reason="insufficient_recovered_work", diagnostics=diagnostics
            )
        diagnostics["energy_residual_W"] = (
            expanded.outlet.enthalpy_rate_W
            + compressed.outlet.enthalpy_rate_W
            - donor.enthalpy_rate_W
            - receiver.enthalpy_rate_W
            - external
            + loss
        )
        return MolecularProcessResult(
            True,
            (expanded.outlet, compressed.outlet),
            heat_duty_W=-loss,
            shaft_work_W=external,
            diagnostics=diagnostics,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "pressure_recovery",
            "expander": self.expander.to_dict(),
            "compressor": self.compressor.to_dict(),
            "transmission_efficiency": self.transmission_efficiency,
            "allow_external_work": self.allow_external_work,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MolecularPressureRecovery:
        return cls(
            MolecularPressureStage.from_dict(value["expander"]),
            MolecularPressureStage.from_dict(value["compressor"]),
            value["transmission_efficiency"],
            value["allow_external_work"],
        )
