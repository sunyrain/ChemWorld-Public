"""Read-only design candidates compiled into the original executable process path."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from math import isfinite
from typing import Any

from chemworld.physchem.bounded_design import (
    BoundedDesignPolicy,
    BoundedDesignResult,
    DesignConstraint,
    DesignEvaluationError,
    DesignPoint,
    DesignVariable,
    solve_bounded_design,
)
from chemworld.physchem.molecular_processes import MolecularHeatDuty, MolecularPressureStage
from chemworld.physchem.molecular_recycle import MolecularRecyclePath
from chemworld.runtime.molecular_process_path import MolecularProcessPath
from chemworld.runtime.molecular_process_runtime import MolecularProcessReceiverSpec

ProcessPath = MolecularProcessPath | MolecularRecyclePath


@dataclass(frozen=True)
class MolecularDesignControl:
    variable: DesignVariable
    field: str
    device_index: int | None = None

    def __post_init__(self) -> None:
        expected = {"outlet_pressure_Pa": "Pa", "heat_duty_W": "W", "recycle_fraction": "1"}
        if self.field not in expected or self.variable.unit != expected[self.field]:
            raise ValueError("process control requires its original field and matching SI unit")
        if self.field == "recycle_fraction":
            if (
                self.device_index is not None
                or not 0 <= self.variable.bounds[0] < self.variable.bounds[1] < 1
            ):
                raise ValueError("recycle design requires no device index and bounds in [0,1)")
        elif type(self.device_index) is not int or self.device_index < 0:
            raise ValueError("process device control requires an explicit nonnegative index")
        if self.field == "outlet_pressure_Pa" and self.variable.bounds[0] <= 0:
            raise ValueError("design pressure bounds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MolecularDesignControl:
        if set(value) != {"variable", "field", "device_index"}:
            raise ValueError("unknown or incomplete design control")
        return cls(DesignVariable(**value["variable"]), value["field"], value["device_index"])


@dataclass(frozen=True)
class MolecularUtilityPrices:
    shaft_import_per_J: float = 1.0
    shaft_export_credit_per_J: float = 0.0
    heating_per_J: float = 0.0
    cooling_per_J: float = 0.0

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if not any(values) or any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in values):
            raise ValueError("utility prices require finite nonnegative currency/J declarations")

    def cost(self, observations: Mapping[str, float]) -> float:
        return (
            self.shaft_import_per_J * observations["shaft_import_W"]
            - self.shaft_export_credit_per_J * observations["shaft_export_W"]
            + self.heating_per_J * observations["heating_W"]
            + self.cooling_per_J * observations["cooling_W"]
        )


def _metric_units(path: ProcessPath) -> dict[str, str]:
    units = {
        "outlet_temperature_K": "K",
        "outlet_pressure_Pa": "Pa",
        "product_molar_rate_mol_s": "mol/s",
        "heat_duty_W": "W",
        "shaft_work_W": "W",
        "heating_W": "W",
        "cooling_W": "W",
        "shaft_import_W": "W",
        "shaft_export_W": "W",
    }
    for index in range(len(path.devices)):
        units.update(
            {
                f"device:{index}:{name}": unit
                for name, unit in (
                    ("temperature_K", "K"),
                    ("pressure_Pa", "Pa"),
                    ("molar_rate_mol_s", "mol/s"),
                    ("heat_duty_W", "W"),
                    ("shaft_work_W", "W"),
                )
            }
        )
    return units


@dataclass(frozen=True)
class MolecularProcessDesignSpec:
    path: ProcessPath
    controls: tuple[MolecularDesignControl, ...]
    constraints: tuple[DesignConstraint, ...]
    prices: MolecularUtilityPrices = field(default_factory=MolecularUtilityPrices)
    starts: tuple[dict[str, float], ...] = ()
    policy: BoundedDesignPolicy = field(default_factory=BoundedDesignPolicy)

    def __post_init__(self) -> None:
        for name in ("controls", "constraints"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "starts", tuple(dict(s) for s in self.starts))
        if not isinstance(self.path, (MolecularProcessPath, MolecularRecyclePath)):
            raise ValueError("design supports only original serial or single-recycle paths")
        if not self.controls or len({c.variable.parameter_id for c in self.controls}) != len(
            self.controls
        ):
            raise ValueError("process design controls require unique variable identities")
        bindings = {(c.device_index, c.field) for c in self.controls}
        if len(bindings) != len(self.controls):
            raise ValueError("two design variables cannot mutate the same device field")
        for control in self.controls:
            if control.field == "recycle_fraction":
                if not isinstance(self.path, MolecularRecyclePath):
                    raise ValueError("recycle control needs an existing original recycle path")
            else:
                index = control.device_index
                if index is None or index >= len(self.path.devices):
                    raise ValueError("design control references a missing device")
                device = self.path.devices[index]
                expected = (
                    MolecularHeatDuty if control.field == "heat_duty_W" else MolecularPressureStage
                )
                if not isinstance(device, expected):
                    raise ValueError("design field does not belong to this original device type")
        units = _metric_units(self.path)
        if any(units.get(c.metric) != c.unit for c in self.constraints):
            raise ValueError(
                "constraint requires an original device/process observable and SI unit"
            )

    def compile(self, values: Mapping[str, float]) -> ProcessPath:
        if set(values) != {c.variable.parameter_id for c in self.controls}:
            raise ValueError("selected design must specify exactly the original controls")
        devices = list(self.path.devices)
        extras: dict[str, Any] = {}
        for control in self.controls:
            value = values[control.variable.parameter_id]
            if isinstance(value, bool):
                raise ValueError("selected design requires real-valued controls")
            control.variable.encode(value)
            if control.field == "recycle_fraction":
                extras["recycle_fraction"] = value
            else:
                if control.device_index is None:
                    raise ValueError("device control requires an explicit index")
                device = devices[control.device_index]
                if control.field == "heat_duty_W" and isinstance(device, MolecularHeatDuty):
                    devices[control.device_index] = replace(device, heat_duty_W=value)
                elif isinstance(device, MolecularPressureStage):
                    devices[control.device_index] = replace(device, outlet_pressure_Pa=value)
                else:
                    raise ValueError("design control disagrees with its original device")
        return replace(self.path, devices=tuple(devices), **extras)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "molecular_process_design_v1",
            "path": self.path.to_dict(),
            "controls": [c.to_dict() for c in self.controls],
            "constraints": [asdict(c) for c in self.constraints],
            "utility_prices_currency_per_J": asdict(self.prices),
            "starts": self.starts,
            "policy": asdict(self.policy),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MolecularProcessDesignSpec:
        if (
            set(value)
            != {
                "model_type",
                "path",
                "controls",
                "constraints",
                "utility_prices_currency_per_J",
                "starts",
                "policy",
            }
            or value["model_type"] != "molecular_process_design_v1"
        ):
            raise ValueError("unknown or incomplete process design declaration")
        path = value["path"]
        if path.get("model") == "molecular-process-path-v1":
            restored: ProcessPath = MolecularProcessPath.from_dict(path)
        elif path.get("model") == "molecular-single-recycle-v1":
            restored = MolecularRecyclePath.from_dict(path)
        else:
            raise ValueError("unsupported original process path")
        return cls(
            restored,
            tuple(MolecularDesignControl.from_dict(c) for c in value["controls"]),
            tuple(DesignConstraint(**c) for c in value["constraints"]),
            MolecularUtilityPrices(**value["utility_prices_currency_per_J"]),
            tuple(value["starts"]),
            BoundedDesignPolicy(**value["policy"]),
        )


def evaluate_molecular_design(
    specification: MolecularProcessDesignSpec, values: Mapping[str, float]
) -> DesignPoint:
    path = specification.compile(values)
    result = path.evaluate()
    if not result.success or result.outlet is None:
        raise DesignEvaluationError(
            result.failure_reason or "missing_process_product", result.to_dict()
        )
    devices = result.diagnostics["devices"]
    if len(devices) != len(path.devices):
        raise DesignEvaluationError("incomplete_original_device_evaluations", result.to_dict())
    outlet = result.outlet
    energy = (
        outlet.enthalpy_rate_W
        - path.inlet.enthalpy_rate_W
        - result.heat_duty_W
        - result.shaft_work_W
    )
    material = max(
        abs(outlet.molar_rate_mol_s * x - path.inlet.molar_rate_mol_s * y)
        for x, y in zip(outlet.state.mole_fractions, path.inlet.state.mole_fractions, strict=True)
    )
    if (
        abs(energy)
        > 1e-5 + 1e-8 * max(abs(outlet.enthalpy_rate_W), abs(path.inlet.enthalpy_rate_W))
        or material > 1e-8 * path.inlet.molar_rate_mol_s
    ):
        raise DesignEvaluationError("original_process_boundary_residual_failed", result.to_dict())
    observations = {
        "outlet_temperature_K": outlet.state.temperature_K,
        "outlet_pressure_Pa": outlet.state.pressure_Pa,
        "product_molar_rate_mol_s": outlet.molar_rate_mol_s,
        "heat_duty_W": result.heat_duty_W,
        "shaft_work_W": result.shaft_work_W,
        "energy_residual_W": energy,
        "material_residual_mol_s": material,
        "heating_W": sum(max(d["heat_duty_W"], 0.0) for d in devices),
        "cooling_W": sum(max(-d["heat_duty_W"], 0.0) for d in devices),
        "shaft_import_W": sum(max(d["shaft_work_W"], 0.0) for d in devices),
        "shaft_export_W": sum(max(-d["shaft_work_W"], 0.0) for d in devices),
    }
    for index, device in enumerate(devices):
        point = device["outlets"][0]
        for name, value in (
            ("temperature_K", point["state"]["temperature_K"]),
            ("pressure_Pa", point["state"]["pressure_Pa"]),
            ("molar_rate_mol_s", point["molar_rate_mol_s"]),
            ("heat_duty_W", device["heat_duty_W"]),
            ("shaft_work_W", device["shaft_work_W"]),
        ):
            observations[f"device:{index}:{name}"] = value
    return DesignPoint(
        specification.prices.cost(observations),
        observations,
        {"original_process": result.to_dict(), "objective_unit": "currency/s"},
    )


@dataclass(frozen=True)
class MolecularProcessDesignResult:
    specification: MolecularProcessDesignSpec
    optimization: BoundedDesignResult
    selected_path: ProcessPath | None

    @property
    def success(self) -> bool:
        return self.optimization.success and self.selected_path is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "molecular_process_design_result_v1",
            "success": self.success,
            "specification": self.specification.to_dict(),
            "optimization": self.optimization.to_dict(),
            "selected_path": None if self.selected_path is None else self.selected_path.to_dict(),
        }


def optimize_molecular_process(
    specification: MolecularProcessDesignSpec,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> MolecularProcessDesignResult:
    result = solve_bounded_design(
        tuple(c.variable for c in specification.controls),
        specification.constraints,
        lambda v: evaluate_molecular_design(specification, v),
        starts=specification.starts,
        policy=specification.policy,
        progress=progress,
    )
    path = (
        specification.compile(result.values)
        if result.success and result.values is not None
        else None
    )
    return MolecularProcessDesignResult(specification, result, path)


def optimized_process_receiver(
    result: MolecularProcessDesignResult, template: MolecularProcessReceiverSpec
) -> MolecularProcessReceiverSpec:
    """Return the original receiver declaration; existing world/wait owns execution."""
    if not result.success or result.selected_path is None:
        raise ValueError("an unverified or exhausted design cannot supply an executable path")
    if template.path != result.specification.path:
        raise ValueError("receiver template differs from the original designed path")
    return replace(template, path=result.selected_path)
