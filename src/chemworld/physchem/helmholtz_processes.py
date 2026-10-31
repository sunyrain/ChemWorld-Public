"""Same-EOS adiabatic pressure stages and explicit material-port consumption.

These algebraic devices own parameters only. The existing CoupledSystem owns
finite inventories, enthalpy ports and elapsed time. Stage shaft work is already
in the downstream enthalpy; it must not be added to a receiving tank again.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any

from chemworld.physchem.coupled_system import ProcessContribution
from chemworld.physchem.helmholtz import HelmholtzState
from chemworld.physchem.helmholtz_flash import (
    HelmholtzTwoPhaseState,
    flash_at_pressure,
)
from chemworld.physchem.helmholtz_solvers import (
    PHASE_ROOT_POLICY,
    HelmholtzPhaseModel,
)
from chemworld.physchem.residual_system import RootPolicy


@dataclass(frozen=True)
class HelmholtzStream:
    state: HelmholtzState[float] | HelmholtzTwoPhaseState
    molar_rate_mol_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.molar_rate_mol_s) or self.molar_rate_mol_s < 0:
            raise ValueError("stream molar rate must be finite and nonnegative")

    @property
    def enthalpy_rate_W(self) -> float:  # noqa: N802
        return self.molar_rate_mol_s*self.state.enthalpy_J_mol

    def contribution(self, port_id: str) -> ProcessContribution:
        """Explicit existing material port carries molar flow and source enthalpy."""
        if not port_id:
            raise ValueError("stream consumption needs an explicit material port")
        return ProcessContribution(port_rates_mol_s={port_id: self.molar_rate_mol_s},
            port_enthalpies_J_mol={port_id: self.state.enthalpy_J_mol})

    def separate(self) -> tuple[HelmholtzStream, HelmholtzStream]:
        """Equilibrium liquid and gas outlets, preserving moles and enthalpy."""
        if not isinstance(self.state, HelmholtzTwoPhaseState):
            raise ValueError("liquid/gas separator requires a resolved two-phase stream")
        q = self.state.vapor_fraction
        return (HelmholtzStream(self.state.liquid, self.molar_rate_mol_s*(1-q)),
                HelmholtzStream(self.state.gas, self.molar_rate_mol_s*q))


@dataclass(frozen=True)
class HelmholtzPressureStageReport:
    inlet: HelmholtzStream
    outlet: HelmholtzStream | None
    shaft_work_W: float | None
    success: bool
    failure_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


@dataclass(frozen=True)
class HelmholtzPressureStage:
    phase_model: HelmholtzPhaseModel
    mode: str
    outlet_pressure_Pa: float
    isentropic_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {"throttle", "compressor", "expander"}:
            raise ValueError("pressure stage mode must be throttle, compressor or expander")
        if (not isfinite(self.outlet_pressure_Pa)
                or not 0 < self.outlet_pressure_Pa <= self.phase_model.fluid.maximum_pressure_Pa):
            raise ValueError("stage outlet pressure outside EOS validity")
        if not isfinite(self.isentropic_efficiency) or not 0 < self.isentropic_efficiency <= 1:
            raise ValueError("isentropic efficiency must be in (0, 1]")
        if self.mode == "throttle" and self.isentropic_efficiency != 1:
            raise ValueError("throttle does not use an isentropic efficiency")
        if self.phase_model.fluid.pseudo_pure:
            raise ValueError("pure-fluid pressure stage rejects pseudo-pure EOS")

    def evaluate(self, inlet: HelmholtzStream, *,
                 policy: RootPolicy = PHASE_ROOT_POLICY) -> HelmholtzPressureStageReport:
        supplied = inlet.state
        model = self.phase_model
        if supplied.fluid_id != model.fluid.fluid_id:
            raise ValueError("pressure-stage inlet has a different fluid identity")
        if isinstance(supplied, HelmholtzTwoPhaseState):
            raise ValueError("pressure-stage inlet must be homogeneous; wet inlet is undeclared")
        # Recompute from our declaration; an object with the same name but a
        # different energy reference must not inject inconsistent inlet enthalpy.
        current = model.fluid.state(temperature_K=supplied.temperature_K,
                                     molar_density_mol_m3=supplied.molar_density_mol_m3)
        if any(abs(a-b) > 1e-6+1e-9*abs(b) for a, b in (
                (supplied.enthalpy_J_mol, current.enthalpy_J_mol),
                (supplied.entropy_J_mol_K, current.entropy_J_mol_K),
                (supplied.pressure_Pa, current.pressure_Pa))):
            raise ValueError("pressure-stage inlet disagrees with the declared EOS/reference")
        pressure = self.outlet_pressure_Pa
        if ((self.mode == "compressor" and pressure <= current.pressure_Pa)
                or (self.mode != "compressor" and pressure >= current.pressure_Pa)):
            raise ValueError("stage outlet pressure has the wrong operating direction")
        diagnostics: dict[str, Any] = {"mode": self.mode, "adiabatic": True,
            "shaft_work_sign": "positive_into_stream", "mass_basis": "mol/s"}
        target_h = current.enthalpy_J_mol
        if self.mode != "throttle":
            isentropic = flash_at_pressure(model, pressure_Pa=pressure, coordinate="S",
                value=current.entropy_J_mol_K, initial_state=current, policy=policy)
            if not isentropic.success:
                return HelmholtzPressureStageReport(inlet, None, None, False,
                    "isentropic_state:"+isentropic.failure_reason, isentropic.diagnostics)
            if isentropic.state is None:
                raise AssertionError("invariant violated: isentropic.state is not None")
            delta_h = isentropic.state.enthalpy_J_mol-current.enthalpy_J_mol
            target_h += (delta_h/self.isentropic_efficiency if self.mode == "compressor"
                         else self.isentropic_efficiency*delta_h)
            diagnostics["isentropic_enthalpy_J_mol"] = isentropic.state.enthalpy_J_mol
            diagnostics["isentropic_efficiency"] = self.isentropic_efficiency
        final = flash_at_pressure(model, pressure_Pa=pressure, coordinate="H", value=target_h,
                                   initial_state=current, policy=policy)
        if not final.success:
            return HelmholtzPressureStageReport(inlet, None, None, False,
                "outlet_state:"+final.failure_reason, final.diagnostics)
        if final.state is None:
            raise AssertionError("invariant violated: final.state is not None")
        outlet = HelmholtzStream(final.state, inlet.molar_rate_mol_s)
        work = inlet.molar_rate_mol_s*(target_h-current.enthalpy_J_mol)
        diagnostics.update(outlet_phase=final.phase,
            energy_residual_W=outlet.enthalpy_rate_W-inlet.enthalpy_rate_W-work,
            entropy_generation_W_K=inlet.molar_rate_mol_s*(
                final.state.entropy_J_mol_K-current.entropy_J_mol_K))
        if diagnostics["entropy_generation_W_K"] < -1e-6:
            return HelmholtzPressureStageReport(inlet, None, None, False,
                                               "negative_entropy_generation", diagnostics)
        return HelmholtzPressureStageReport(inlet, outlet, work, True, diagnostics=diagnostics)

    def to_dict(self) -> dict[str, Any]:
        return {"model": "helmholtz-pressure-stage-v1", "phase_model": self.phase_model.to_dict(),
                "mode": self.mode, "outlet_pressure_Pa": self.outlet_pressure_Pa,
                "isentropic_efficiency": self.isentropic_efficiency}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HelmholtzPressureStage:
        values = dict(data)
        if values.pop("model", None) != "helmholtz-pressure-stage-v1":
            raise ValueError("unknown Helmholtz pressure-stage declaration")
        values["phase_model"] = HelmholtzPhaseModel.from_dict(values["phase_model"])
        return cls(**values)
