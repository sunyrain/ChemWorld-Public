"""Homogeneous process units using the existing molecular caloric reference.

These steady algebraic units own no stock or clock. Declared local density
basins are not multiphase flash or a global stability assertion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import fsum, isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_mixture import (
    MolecularCaloricModel,
    MolecularCaloricState,
    MolecularStream,
)
from chemworld.physchem.residual_system import (
    DEFAULT_ROOT_POLICY,
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    solve_root,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MolecularProcessDomain:
    model: MolecularCaloricModel
    temperature_range_K: tuple[float, float]
    density_basin: MolecularDensityBasin

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        if (
            len(self.temperature_range_K) != 2
            or any(isinstance(v, bool) or not isfinite(v) for v in self.temperature_range_K)
            or not 0 < self.temperature_range_K[0] < self.temperature_range_K[1]
        ):
            raise ValueError("process domain requires positive ordered temperature bounds")
        if any(
            self.temperature_range_K[0] < r.minimum_temperature_K
            or self.temperature_range_K[1] > r.maximum_temperature_K
            for r in self.model.ideal_references
        ):
            raise ValueError("process temperature domain exceeds an ideal reference")

    def state(
        self, temperature: float, density: float, fractions: Sequence[float]
    ) -> MolecularCaloricState:
        if (
            not self.temperature_range_K[0] <= temperature <= self.temperature_range_K[1]
            or not self.density_basin.density_bounds_mol_m3[0]
            <= density
            <= self.density_basin.density_bounds_mol_m3[1]
        ):
            raise ValueError("process state outside declared temperature/density basin")
        return self.model.state(temperature, density, fractions)

    def validate_stream(self, stream: MolecularStream) -> MolecularCaloricState:
        s = stream.state
        if s.component_ids != self.model.component_ids:
            raise ValueError("process inlet component identities/order disagree")
        current = self.state(s.temperature_K, s.molar_density_mol_m3, s.mole_fractions)
        if any(
            abs(getattr(current, k) - getattr(s, k)) > 1e-7 + 1e-9 * abs(getattr(current, k))
            for k in ("pressure_Pa", "enthalpy_J_mol", "entropy_J_mol_K")
        ):
            raise ValueError("process inlet disagrees with the declared EOS/caloric reference")
        return current

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "molecular-process-domain-v1",
            "caloric_model": self.model.to_dict(),
            "temperature_range_K": self.temperature_range_K,
            "density_basin": asdict(self.density_basin),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularProcessDomain:
        if data.get("model") != "molecular-process-domain-v1":
            raise ValueError("unknown molecular process domain")
        return cls(
            MolecularCaloricModel.from_dict(data["caloric_model"]),
            tuple(data["temperature_range_K"]),
            MolecularDensityBasin(**data["density_basin"]),
        )


@dataclass(frozen=True)
class MolecularProcessStateResult:
    success: bool
    state: MolecularCaloricState | None = None
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "state": None if self.state is None else self.state.to_dict(),
            "failure_reason": self.failure_reason,
            "diagnostics": self.diagnostics,
        }


def molecular_process_state(
    domain: MolecularProcessDomain,
    *,
    pressure_Pa: float,
    coordinate: str,
    value: float,
    mole_fractions: Sequence[float],
    initial_temperature_K: float,
    initial_density_mol_m3: float | None = None,
    policy: RootPolicy = DEFAULT_ROOT_POLICY,
) -> MolecularProcessStateResult:
    """Local TP/PH/PS closure using shared exact thermodynamic derivatives.

    The root's unknowns are T and rho. No phase switching, saturation search or
    EOS equation is reimplemented. Domain failure leaves the caller untouched.
    """
    if (
        coordinate not in {"T", "H", "S"}
        or not isfinite(value)
        or not isfinite(pressure_Pa)
        or pressure_Pa <= 0
        or not isfinite(initial_temperature_K)
        or initial_temperature_K <= 0
    ):
        raise ValueError("process closure needs positive pressure and a finite T/H/S target")
    rho0 = (
        domain.density_basin.initial_density_mol_m3
        if initial_density_mol_m3 is None
        else initial_density_mol_m3
    )
    fractions = tuple(mole_fractions)
    tscale = initial_temperature_K
    gas = domain.model.gas_constant_J_mol_K
    scales = [max(pressure_Pa, 1.0), {"T": tscale, "H": gas * tscale, "S": gas}[coordinate]]

    def state(q: FloatArray) -> MolecularCaloricState:
        try:
            return domain.state(float(q[0]), float(q[1]), fractions)
        except (ValueError, ArithmeticError) as exc:
            raise ResidualDomainError(str(exc)) from exc

    def residual(q: FloatArray) -> FloatArray:
        s = state(q)
        actual = {"T": s.temperature_K, "H": s.enthalpy_J_mol, "S": s.entropy_J_mol_K}
        return np.array((s.pressure_Pa - pressure_Pa, actual[coordinate] - value))

    def jacobian(q: FloatArray) -> FloatArray:
        s = state(q)
        t, rho, pt, pr = (
            s.temperature_K,
            s.molar_density_mol_m3,
            (s.dP_dT_rho_Pa_K),
            s.dP_drho_T_Pa_m3_mol,
        )
        second = {
            "T": (1.0, 0.0),
            "H": (s.cv_J_mol_K + pt / rho, pr / rho - t * pt / rho**2),
            "S": (s.cv_J_mol_K / t, -pt / rho**2),
        }[coordinate]
        return np.array(((pt, pr), second))

    root = solve_root(
        residual,
        (initial_temperature_K, rho0),
        jacobian=jacobian,
        variables=(
            ResidualVariable(
                "T",
                "K",
                tscale,
                lower=domain.temperature_range_K[0],
                upper=domain.temperature_range_K[1],
            ),
            ResidualVariable(
                "rho",
                "mol/m3",
                domain.density_basin.initial_density_mol_m3,
                lower=domain.density_basin.density_bounds_mol_m3[0],
                upper=domain.density_basin.density_bounds_mol_m3[1],
            ),
        ),
        residual_scales=scales,
        policy=policy,
    )
    diagnostics = {
        "coordinate": coordinate,
        "basin": domain.density_basin.name,
        "stability_scope": "homogeneous_local_mechanical_and_thermal_only",
        "root": root.diagnostics,
    }
    if not root.success:
        return MolecularProcessStateResult(
            False, failure_reason=root.failure_reason, diagnostics=diagnostics
        )
    return MolecularProcessStateResult(True, state(root.value), diagnostics=diagnostics)


@dataclass(frozen=True)
class MolecularProcessResult:
    success: bool
    outlets: tuple[MolecularStream, ...] = ()
    heat_duty_W: float = 0.0
    shaft_work_W: float = 0.0
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def outlet(self) -> MolecularStream | None:
        return self.outlets[0] if len(self.outlets) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "outlets": [
                {"state": s.state.to_dict(), "molar_rate_mol_s": s.molar_rate_mol_s}
                for s in self.outlets
            ],
            "heat_duty_W": self.heat_duty_W,
            "shaft_work_W": self.shaft_work_W,
            "failure_reason": self.failure_reason,
            "diagnostics": self.diagnostics,
        }


def _single_result(
    inlet: MolecularStream,
    closed: MolecularProcessStateResult,
    *,
    heat: float = 0.0,
    work: float = 0.0,
    mode: str,
) -> MolecularProcessResult:
    if not closed.success:
        return MolecularProcessResult(
            False,
            failure_reason=closed.failure_reason,
            diagnostics={"mode": mode, **closed.diagnostics},
        )
    assert closed.state is not None
    outlet = MolecularStream(closed.state, inlet.molar_rate_mol_s)
    return MolecularProcessResult(
        True,
        (outlet,),
        heat,
        work,
        diagnostics={
            "mode": mode,
            "energy_sign": "heat_and_work_positive_into_stream",
            "energy_residual_W": outlet.enthalpy_rate_W - inlet.enthalpy_rate_W - heat - work,
            **closed.diagnostics,
        },
    )


@dataclass(frozen=True)
class MolecularPressureStage:
    domain: MolecularProcessDomain
    mode: str
    outlet_pressure_Pa: float
    isentropic_efficiency: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.mode not in {"compressor", "expander", "throttle"}
            or not isfinite(self.outlet_pressure_Pa)
            or self.outlet_pressure_Pa <= 0
            or not isfinite(self.isentropic_efficiency)
            or not 0 < self.isentropic_efficiency <= 1
            or (self.mode == "throttle" and self.isentropic_efficiency != 1)
        ):
            raise ValueError("pressure stage needs a mode, positive pressure and valid efficiency")

    def evaluate(
        self, inlet: MolecularStream, *, policy: RootPolicy = DEFAULT_ROOT_POLICY
    ) -> MolecularProcessResult:
        s = self.domain.validate_stream(inlet)
        if (self.mode == "compressor" and self.outlet_pressure_Pa <= s.pressure_Pa) or (
            self.mode != "compressor" and self.outlet_pressure_Pa >= s.pressure_Pa
        ):
            raise ValueError("pressure stage has the wrong operating direction")
        options: dict[str, Any] = {
            "pressure_Pa": self.outlet_pressure_Pa,
            "mole_fractions": s.mole_fractions,
            "initial_temperature_K": s.temperature_K,
            "initial_density_mol_m3": s.molar_density_mol_m3,
            "policy": policy,
        }
        target = s.enthalpy_J_mol
        isentropic = None
        if self.mode != "throttle":
            iso = molecular_process_state(
                self.domain, coordinate="S", value=s.entropy_J_mol_K, **options
            )
            if not iso.success:
                return MolecularProcessResult(
                    False,
                    failure_reason="isentropic_state:" + iso.failure_reason,
                    diagnostics=iso.diagnostics,
                )
            assert iso.state is not None
            isentropic = iso.state.enthalpy_J_mol
            delta = isentropic - target
            target += (
                delta / self.isentropic_efficiency
                if self.mode == "compressor"
                else delta * self.isentropic_efficiency
            )
            options.update(
                initial_temperature_K=iso.state.temperature_K,
                initial_density_mol_m3=iso.state.molar_density_mol_m3,
            )
        closed = molecular_process_state(self.domain, coordinate="H", value=target, **options)
        result = _single_result(
            inlet, closed, mode=self.mode, work=inlet.molar_rate_mol_s * (target - s.enthalpy_J_mol)
        )
        if result.outlet is not None:
            generated = inlet.molar_rate_mol_s * (
                result.outlet.state.entropy_J_mol_K - s.entropy_J_mol_K
            )
            result.diagnostics.update(
                entropy_generation_W_K=generated,
                isentropic_enthalpy_J_mol=isentropic,
                isentropic_efficiency=self.isentropic_efficiency,
            )
            if generated < -1e-7:
                return MolecularProcessResult(
                    False,
                    failure_reason="negative_entropy_generation",
                    diagnostics=result.diagnostics,
                )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "pressure",
            "domain": self.domain.to_dict(),
            "mode": self.mode,
            "outlet_pressure_Pa": self.outlet_pressure_Pa,
            "isentropic_efficiency": self.isentropic_efficiency,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularPressureStage:
        if data.get("kind") != "pressure":
            raise ValueError("unknown molecular pressure stage")
        return cls(
            MolecularProcessDomain.from_dict(data["domain"]),
            data["mode"],
            data["outlet_pressure_Pa"],
            data["isentropic_efficiency"],
        )


def molecular_heat_duty(
    domain: MolecularProcessDomain,
    inlet: MolecularStream,
    *,
    heat_duty_W: float,
    outlet_pressure_Pa: float | None = None,
    boundary_temperature_K: float | None = None,
    policy: RootPolicy = DEFAULT_ROOT_POLICY,
) -> MolecularProcessResult:
    """Prescribed steady heat load and pressure drop, optionally with one reservoir."""
    s = domain.validate_stream(inlet)
    pressure = s.pressure_Pa if outlet_pressure_Pa is None else outlet_pressure_Pa
    if (
        not isfinite(heat_duty_W)
        or not isfinite(pressure)
        or pressure <= 0
        or pressure > s.pressure_Pa + max(1e-5, 1e-9 * s.pressure_Pa)
        or (
            boundary_temperature_K is not None
            and (not isfinite(boundary_temperature_K) or boundary_temperature_K <= 0)
        )
    ):
        raise ValueError("heat duty needs finite heat, valid reservoir and nonincreasing pressure")
    if inlet.molar_rate_mol_s == 0:
        if heat_duty_W != 0:
            raise ValueError("nonzero heat duty cannot be assigned to zero stream flow")
        target = s.enthalpy_J_mol
    else:
        target = s.enthalpy_J_mol + heat_duty_W / inlet.molar_rate_mol_s
    closed = molecular_process_state(
        domain,
        pressure_Pa=pressure,
        coordinate="H",
        value=target,
        mole_fractions=s.mole_fractions,
        initial_temperature_K=s.temperature_K,
        initial_density_mol_m3=s.molar_density_mol_m3,
        policy=policy,
    )
    result = _single_result(inlet, closed, heat=heat_duty_W, mode="prescribed_heat_load")
    if result.outlet is not None and boundary_temperature_K is not None:
        out = result.outlet.state
        generated = inlet.molar_rate_mol_s * (out.entropy_J_mol_K - s.entropy_J_mol_K) - (
            heat_duty_W / boundary_temperature_K
        )
        result.diagnostics.update(
            boundary_temperature_K=boundary_temperature_K, entropy_generation_W_K=generated
        )
        if (
            generated < -1e-7
            or (
                heat_duty_W > 0 and boundary_temperature_K < max(s.temperature_K, out.temperature_K)
            )
            or (
                heat_duty_W < 0 and boundary_temperature_K > min(s.temperature_K, out.temperature_K)
            )
        ):
            return MolecularProcessResult(
                False, failure_reason="incompatible_heat_reservoir", diagnostics=result.diagnostics
            )
    return result


@dataclass(frozen=True)
class MolecularHeatDuty:
    domain: MolecularProcessDomain
    heat_duty_W: float
    outlet_pressure_Pa: float | None = None
    boundary_temperature_K: float | None = None

    def evaluate(
        self, inlet: MolecularStream, *, policy: RootPolicy = DEFAULT_ROOT_POLICY
    ) -> MolecularProcessResult:
        return molecular_heat_duty(
            self.domain,
            inlet,
            heat_duty_W=self.heat_duty_W,
            outlet_pressure_Pa=self.outlet_pressure_Pa,
            boundary_temperature_K=self.boundary_temperature_K,
            policy=policy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "heat",
            "domain": self.domain.to_dict(),
            "heat_duty_W": self.heat_duty_W,
            "outlet_pressure_Pa": self.outlet_pressure_Pa,
            "boundary_temperature_K": self.boundary_temperature_K,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularHeatDuty:
        if data.get("kind") != "heat":
            raise ValueError("unknown molecular heat device")
        return cls(
            MolecularProcessDomain.from_dict(data["domain"]),
            data["heat_duty_W"],
            data["outlet_pressure_Pa"],
            data["boundary_temperature_K"],
        )


def molecular_mix_streams(
    domain: MolecularProcessDomain,
    inlets: Sequence[MolecularStream],
    *,
    outlet_pressure_Pa: float,
    initial_temperature_K: float,
    policy: RootPolicy = DEFAULT_ROOT_POLICY,
) -> MolecularProcessResult:
    """Steady adiabatic mixing; all source amounts and enthalpies remain explicit."""
    if not inlets or not isfinite(outlet_pressure_Pa) or outlet_pressure_Pa <= 0:
        raise ValueError("mixing requires explicit nonempty inlets and positive pressure")
    for stream in inlets:
        domain.validate_stream(stream)
        if stream.molar_rate_mol_s > 0 and outlet_pressure_Pa > stream.state.pressure_Pa + max(
            1e-5, 1e-9 * stream.state.pressure_Pa
        ):
            raise ValueError("mixing cannot raise pressure without declared shaft work")
    rate = fsum(s.molar_rate_mol_s for s in inlets)
    if rate <= 0:
        raise ValueError("mixed stream cannot have zero total flow")
    fractions = tuple(
        fsum(s.molar_rate_mol_s * s.state.mole_fractions[i] for s in inlets) / rate
        for i in range(len(domain.model.component_ids))
    )
    enthalpy = fsum(s.enthalpy_rate_W for s in inlets)
    closed = molecular_process_state(
        domain,
        pressure_Pa=outlet_pressure_Pa,
        coordinate="H",
        value=enthalpy / rate,
        mole_fractions=fractions,
        initial_temperature_K=initial_temperature_K,
        policy=policy,
    )
    if not closed.success:
        return MolecularProcessResult(
            False, failure_reason=closed.failure_reason, diagnostics=closed.diagnostics
        )
    assert closed.state is not None
    out = MolecularStream(closed.state, rate)
    generated = rate * out.state.entropy_J_mol_K - fsum(
        s.molar_rate_mol_s * s.state.entropy_J_mol_K for s in inlets
    )
    diagnostics = {
        **closed.diagnostics,
        "mode": "steady_adiabatic_mixer",
        "energy_residual_W": out.enthalpy_rate_W - enthalpy,
        "entropy_generation_W_K": generated,
        "component_balance_mol_s": {
            k: rate * x - fsum(s.molar_rate_mol_s * s.state.mole_fractions[i] for s in inlets)
            for i, (k, x) in enumerate(zip(domain.model.component_ids, fractions, strict=True))
        },
    }
    if generated < -1e-7:
        return MolecularProcessResult(
            False, failure_reason="negative_entropy_generation", diagnostics=diagnostics
        )
    return MolecularProcessResult(True, (out,), diagnostics=diagnostics)


def molecular_split_stream(
    domain: MolecularProcessDomain, inlet: MolecularStream, fractions: Sequence[float]
) -> MolecularProcessResult:
    """Nonselective steady split at unchanged T/P/composition; zero branches allowed."""
    domain.validate_stream(inlet)
    if (
        not fractions
        or any(not isfinite(v) or v < 0 for v in fractions)
        or abs(fsum(fractions) - 1) > 1e-12
    ):
        raise ValueError("split fractions must be nonnegative and sum to one")
    outlets = tuple(MolecularStream(inlet.state, inlet.molar_rate_mol_s * v) for v in fractions)
    return MolecularProcessResult(
        True,
        outlets,
        diagnostics={
            "mode": "steady_nonselective_split",
            "energy_residual_W": fsum(s.enthalpy_rate_W for s in outlets) - inlet.enthalpy_rate_W,
        },
    )
