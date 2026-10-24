"""One-feed finite-power TP flash using original molecular phase/caloric models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any

from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_flash import MolecularDensityBasin, molecular_tp_flash
from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularStream
from chemworld.physchem.molecular_processes import MolecularProcessDomain, MolecularProcessResult
from chemworld.physchem.residual_system import RootPolicy


@dataclass(frozen=True)
class MolecularFlashSeparatorSpec:
    model: MolecularCaloricModel
    vapor_basin: MolecularDensityBasin
    liquid_basin: MolecularDensityBasin
    pressure_Pa: float
    device: ThermalDeviceSpec
    feed_phase: str = "liquid"
    temperature_range_K: tuple[float, float] = (200.0, 500.0)
    max_iterations: int = 200
    tpd_tolerance: float = 1e-8
    root_policy: RootPolicy = field(default_factory=lambda: RootPolicy(1e-9, 100))

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        if (
            len(self.model.component_ids) != 2
            or self.feed_phase not in {"gas", "liquid"}
            or self.vapor_basin.name == self.liquid_basin.name
            or isinstance(self.pressure_Pa, bool)
            or not isfinite(self.pressure_Pa)
            or self.pressure_Pa <= 0
            or type(self.max_iterations) is not int
            or self.max_iterations < 1
            or not isfinite(self.tpd_tolerance)
            or self.tpd_tolerance <= 0
            or self.device.mode != "isothermal"
            or not self.device.enabled
        ):
            raise ValueError(
                "separator requires a binary model, explicit phases and finite isothermal device"
            )
        for basin in self.basins:
            MolecularProcessDomain(self.model, self.temperature_range_K, basin)
        if not self.temperature_range_K[0] <= self.device.setpoint_K <= self.temperature_range_K[1]:
            raise ValueError("separator target temperature exceeds its declared source domain")

    @property
    def basins(self) -> tuple[MolecularDensityBasin, MolecularDensityBasin]:
        return self.vapor_basin, self.liquid_basin

    def basin(self, phase: str) -> MolecularDensityBasin:
        if phase not in {"gas", "liquid"}:
            raise ValueError("separator phase label is not supported")
        return self.vapor_basin if phase == "gas" else self.liquid_basin

    @property
    def feed_domain(self) -> MolecularProcessDomain:
        return MolecularProcessDomain(
            self.model, self.temperature_range_K, self.basin(self.feed_phase)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "molecular_flash_separator_v1",
            "caloric_model": self.model.to_dict(),
            "vapor_basin": asdict(self.vapor_basin),
            "liquid_basin": asdict(self.liquid_basin),
            "pressure_Pa": self.pressure_Pa,
            "device": self.device.to_dict(),
            "feed_phase": self.feed_phase,
            "temperature_range_K": self.temperature_range_K,
            "max_iterations": self.max_iterations,
            "tpd_tolerance": self.tpd_tolerance,
            "root_policy": asdict(self.root_policy),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularFlashSeparatorSpec:
        if data.get("model_type") != "molecular_flash_separator_v1":
            raise ValueError("unknown molecular flash separator encoding")
        return cls(
            MolecularCaloricModel.from_dict(data["caloric_model"]),
            MolecularDensityBasin(**data["vapor_basin"]),
            MolecularDensityBasin(**data["liquid_basin"]),
            data["pressure_Pa"],
            ThermalDeviceSpec.from_dict(data["device"]),
            data["feed_phase"],
            tuple(data["temperature_range_K"]),
            data["max_iterations"],
            data["tpd_tolerance"],
            RootPolicy(**data["root_policy"]),
        )


def evaluate_molecular_flash_separator(
    spec: MolecularFlashSeparatorSpec, inlet: MolecularStream
) -> MolecularProcessResult:
    """Actual phases alone form outlets, with explicit one-to-one phase labels.

    Flash extent is a one-mol algebraic basis. The original phase fraction is
    multiplied by the actual metered inlet; no basis inventory is installed.
    """
    diagnostic: dict[str, Any] = {
        "scope": "binary_isothermal_zero_accumulation_separator",
        "stability_scope": "finite_declared_basins_and_starts",
        "flash_basis_mol": 1.0,
        "max_iterations": spec.max_iterations,
        "tpd_tolerance": spec.tpd_tolerance,
        "outlet_phase_labels": [],
        "shaft_work_W": 0.0,
    }

    def failed(reason: str) -> MolecularProcessResult:
        return MolecularProcessResult(False, failure_reason=reason, diagnostics=diagnostic)

    try:
        source = spec.feed_domain.validate_stream(inlet)
        if inlet.molar_rate_mol_s <= 0 or abs(source.pressure_Pa / spec.pressure_Pa - 1) > 2e-7:
            raise ValueError(
                "separator needs positive same-pressure flow from its declared feed phase"
            )
        basis = dict(zip(spec.model.component_ids, source.mole_fractions, strict=True))
        flash = molecular_tp_flash(
            spec.model.residual_model,
            spec.device.setpoint_K,
            spec.pressure_Pa,
            basis,
            basins=spec.basins,
            max_iterations=spec.max_iterations,
            tpd_tolerance=spec.tpd_tolerance,
        )
        diagnostic["flash"] = flash.to_dict()
        if not flash.success:
            return failed("molecular_tp_flash:" + flash.failure_reason)
        by_basin = {p.basin: p for p in flash.phases}
        if len(by_basin) != len(flash.phases) or set(by_basin) - {b.name for b in spec.basins}:
            return failed("flash returned an undeclared or duplicate product phase")
        outlets, labels = [], []
        for label, basin in zip(("gas", "liquid"), spec.basins, strict=True):
            if basin.name not in by_basin:
                continue
            phase = by_basin[basin.name]
            if phase.amount_mol <= 0:
                return failed("flash returned a nonpositive claimed phase")
            s = phase.state
            closed = spec.model.state(s.temperature_K, s.molar_density_mol_m3, s.mole_fractions)
            outlets.append(MolecularStream(closed, inlet.molar_rate_mol_s * phase.amount_mol))
            labels.append(label)
        heat = sum(s.enthalpy_rate_W for s in outlets) - inlet.enthalpy_rate_W
        residual = {
            k: sum(s.molar_rate_mol_s * s.state.mole_fractions[i] for s in outlets)
            - inlet.molar_rate_mol_s * source.mole_fractions[i]
            for i, k in enumerate(spec.model.component_ids)
        }
        diagnostic.update(
            outlet_phase_labels=labels,
            component_residuals_mol_s=residual,
            required_heat_duty_W=heat,
            heater_W=max(heat, 0.0),
            condenser_W=max(-heat, 0.0),
        )
        if max(map(abs, residual.values())) > 1e-10:
            return failed("separator component balance failed")
        if (
            heat > spec.device.maximum_power_W + 1e-10
            or -heat > spec.device.maximum_cooling_power_W + 1e-10
        ):
            diagnostic["candidate_phase_labels"] = labels
            diagnostic["outlet_phase_labels"] = []
            return failed("insufficient_separator_heating_or_cooling_power")
        return MolecularProcessResult(True, tuple(outlets), heat, diagnostics=diagnostic)
    except (ValueError, ArithmeticError) as exc:
        return failed(str(exc))
