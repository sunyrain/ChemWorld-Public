"""Bounded binary phi-phi countercurrent stages on the original molecular EOS.

Stage temperatures are imposed by finite-power isothermal devices. The steady
equipment owns no stock: internal phase rates follow component balances, not a
constant-molar-overflow approximation. Tanks and their clock belong elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.molecular_flash import MolecularDensityBasin, molecular_tp_flash
from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularStream
from chemworld.physchem.molecular_processes import MolecularProcessDomain, MolecularProcessResult
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)


@dataclass(frozen=True)
class MolecularCascadeSpec:
    model: MolecularCaloricModel
    vapor_basin: MolecularDensityBasin
    liquid_basin: MolecularDensityBasin
    pressure_Pa: float
    devices: tuple[ThermalDeviceSpec, ...]
    temperature_range_K: tuple[float, float] = (200.0, 500.0)
    component_flow_bounds_mol_s: tuple[float, float] = (1e-12, 0.1)
    root_policy: RootPolicy = field(default_factory=lambda: RootPolicy(1e-9, 100))
    tpd_max_iterations: int = 200
    tpd_tolerance: float = 1e-8
    source: str = "authored finite molecular equilibrium cascade"

    def __post_init__(self) -> None:
        for key in ("devices", "temperature_range_K", "component_flow_bounds_mol_s"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        if (
            len(self.model.component_ids) != 2
            or not 1 <= len(self.devices) <= 6
            or self.vapor_basin.name == self.liquid_basin.name
            or not isfinite(self.pressure_Pa)
            or self.pressure_Pa <= 0
            or type(self.tpd_max_iterations) is not int
            or self.tpd_max_iterations < 1
            or not isfinite(self.tpd_tolerance)
            or self.tpd_tolerance <= 0
            or not self.source
        ):
            raise ValueError(
                "cascade requires two components, distinct basins, 1..6 stages and budgets"
            )
        low, high = self.component_flow_bounds_mol_s
        if not 0 < low < high or not isfinite(high):
            raise ValueError("stage component flows need positive finite ordered bounds")
        for basin in self.basins:
            MolecularProcessDomain(self.model, self.temperature_range_K, basin)
        if any(
            d.mode != "isothermal"
            or not d.enabled
            or not self.temperature_range_K[0] <= d.setpoint_K <= self.temperature_range_K[1]
            for d in self.devices
        ):
            raise ValueError("each stage requires an enabled finite-power isothermal device")

    @property
    def basins(self) -> tuple[MolecularDensityBasin, MolecularDensityBasin]:
        return self.vapor_basin, self.liquid_basin

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "molecular_equilibrium_cascade_v1",
            "caloric_model": self.model.to_dict(),
            "vapor_basin": asdict(self.vapor_basin),
            "liquid_basin": asdict(self.liquid_basin),
            "pressure_Pa": self.pressure_Pa,
            "devices": [d.to_dict() for d in self.devices],
            "temperature_range_K": self.temperature_range_K,
            "component_flow_bounds_mol_s": self.component_flow_bounds_mol_s,
            "root_policy": asdict(self.root_policy),
            "tpd_max_iterations": self.tpd_max_iterations,
            "tpd_tolerance": self.tpd_tolerance,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularCascadeSpec:
        if data.get("model_type") != "molecular_equilibrium_cascade_v1":
            raise ValueError("unknown molecular cascade declaration")
        return cls(
            MolecularCaloricModel.from_dict(data["caloric_model"]),
            MolecularDensityBasin(**data["vapor_basin"]),
            MolecularDensityBasin(**data["liquid_basin"]),
            data["pressure_Pa"],
            tuple(ThermalDeviceSpec.from_dict(d) for d in data["devices"]),
            tuple(data["temperature_range_K"]),
            tuple(data["component_flow_bounds_mol_s"]),
            RootPolicy(**data["root_policy"]),
            data["tpd_max_iterations"],
            data["tpd_tolerance"],
            data["source"],
        )


def solve_molecular_cascade(
    spec: MolecularCascadeSpec,
    vapor_feed: MolecularStream,
    liquid_feed: MolecularStream,
) -> MolecularProcessResult:
    """Solve every stage's species, pressure and chemical-potential equations.

    Order is top to bottom; vapor enters the bottom, liquid the top. Returns
    vapor then liquid original streams. Failure carries no executable outlets.
    """
    diag: dict[str, Any] = {
        "scope": "binary_1_to_6_isothermal_equilibrium_stages_no_holdup",
        "stability_scope": "finite_declared_basins_and_starts",
        "root_policy": asdict(spec.root_policy),
        "tpd_max_iterations": spec.tpd_max_iterations,
        "tpd_tolerance": spec.tpd_tolerance,
        "stages": [],
        "phase_checks": [],
        "eos_evaluations": 0,
        "shaft_work_W": 0.0,
    }

    def fail(reason: str) -> MolecularProcessResult:
        return MolecularProcessResult(False, failure_reason=reason, diagnostics=diag)

    try:
        for stream, basin in zip((vapor_feed, liquid_feed), spec.basins, strict=True):
            MolecularProcessDomain(spec.model, spec.temperature_range_K, basin).validate_stream(
                stream
            )
            if (
                stream.molar_rate_mol_s <= 0
                or abs(stream.state.pressure_Pa / spec.pressure_Pa - 1) > 2e-7
            ):
                raise ValueError("cascade requires positive feeds at the common declared pressure")
        feed_v = vapor_feed.molar_rate_mol_s * np.array(vapor_feed.state.mole_fractions)
        feed_l = liquid_feed.molar_rate_mol_s * np.array(liquid_feed.state.mole_fractions)
        count = len(spec.devices)
        flow_scale = vapor_feed.molar_rate_mol_s + liquid_feed.molar_rate_mol_s
        variables: list[ResidualVariable] = []
        initial: list[float] = []
        scales: list[float] = []
        for j, device in enumerate(spec.devices):
            # Fixed composition starts are declarations, not reference answers.
            guesses = []
            for basin, x in zip(spec.basins, ((0.75, 0.25), (0.4, 0.6)), strict=True):
                guesses.append(
                    spec.model.state_tp(
                        device.setpoint_K,
                        spec.pressure_Pa,
                        x,
                        initial_density_mol_m3=basin.initial_density_mol_m3,
                        density_bounds_mol_m3=basin.density_bounds_mol_m3,
                    )
                )
            initial.extend(
                [
                    *(vapor_feed.molar_rate_mol_s * np.array(guesses[0].mole_fractions)),
                    *(liquid_feed.molar_rate_mol_s * np.array(guesses[1].mole_fractions)),
                    *(s.molar_density_mol_m3 for s in guesses),
                ]
            )
            variables.extend(
                ResidualVariable(
                    f"stage{j}:flow{k}",
                    "mol/s",
                    flow_scale,
                    lower=spec.component_flow_bounds_mol_s[0],
                    upper=spec.component_flow_bounds_mol_s[1],
                )
                for k in range(4)
            )
            variables.extend(
                ResidualVariable(
                    f"stage{j}:{b.name}:rho",
                    "mol/m3",
                    b.initial_density_mol_m3,
                    lower=b.density_bounds_mol_m3[0],
                    upper=b.density_bounds_mol_m3[1],
                )
                for b in spec.basins
            )
            scales.extend((flow_scale, flow_scale, 1.0, 1.0, spec.pressure_Pa, spec.pressure_Pa))

        def phase_states(q: NDArray[np.float64]) -> tuple[NDArray[np.float64], list[Any]]:
            rows = np.asarray(q).reshape(count, 6)
            states = []
            for j, row in enumerate(rows):
                pair = []
                for k in range(2):
                    ns = row[2 * k : 2 * k + 2]
                    diag["eos_evaluations"] += 1
                    pair.append(
                        molecular_eos_state(
                            spec.model.residual_model,
                            spec.devices[j].setpoint_K,
                            float(row[4 + k]),
                            tuple(float(v) for v in ns / ns.sum()),
                        )
                    )
                states.append(pair)
            return rows, states

        def residual(q: NDArray[np.float64]) -> NDArray[np.float64]:
            try:
                rows, pairs = phase_states(q)
                values = []
                for j, (row, pair) in enumerate(zip(rows, pairs, strict=True)):
                    incoming_v = feed_v if j == count - 1 else rows[j + 1, :2]
                    incoming_l = feed_l if j == 0 else rows[j - 1, 2:4]
                    if any(s.log_fugacity_coefficients is None for s in pair):
                        raise ValueError("stage has no positive-pressure chemical potentials")
                    chemical = [
                        np.log(s.mole_fractions) + np.array(s.log_fugacity_coefficients)
                        for s in pair
                    ]
                    values.extend(
                        [
                            *(row[:2] + row[2:4] - incoming_v - incoming_l),
                            *(chemical[0] - chemical[1]),
                            *(s.pressure_Pa - spec.pressure_Pa for s in pair),
                        ]
                    )
                return np.array(values)
            except (ValueError, ArithmeticError) as exc:
                raise ResidualDomainError(str(exc)) from exc

        lower = np.array([v.lower for v in variables])
        upper = np.array([v.upper for v in variables])
        columns = np.array([v.scale for v in variables])
        root = solve_root(
            residual,
            initial,
            variables=variables,
            residual_scales=scales,
            jacobian=lambda q: finite_jacobian(residual, q, columns, lower=lower, upper=upper),
            policy=spec.root_policy,
        )
        diag["root"] = root.diagnostics
        if not root.success or root.value is None:
            return fail("stage_balance_root:" + root.failure_reason)
        rows, pairs = phase_states(root.value)
        streams = []
        for row, pair in zip(rows, pairs, strict=True):
            streams.append(
                tuple(
                    MolecularStream(
                        spec.model.state(s.temperature_K, s.molar_density_mol_m3, s.mole_fractions),
                        float(row[2 * k : 2 * k + 2].sum()),
                    )
                    for k, s in enumerate(pair)
                )
            )
        duties = []
        for j, pair in enumerate(streams):
            vin = vapor_feed if j == count - 1 else streams[j + 1][0]
            lin = liquid_feed if j == 0 else streams[j - 1][1]
            duty = sum(s.enthalpy_rate_W for s in pair) - vin.enthalpy_rate_W - lin.enthalpy_rate_W
            duties.append(duty)
            diag["stages"].append(
                {
                    "index": j,
                    "temperature_K": spec.devices[j].setpoint_K,
                    "vapor": {
                        "molar_rate_mol_s": pair[0].molar_rate_mol_s,
                        "state": pair[0].state.to_dict(),
                    },
                    "liquid": {
                        "molar_rate_mol_s": pair[1].molar_rate_mol_s,
                        "state": pair[1].state.to_dict(),
                    },
                    "heat_duty_W": duty,
                }
            )
        raw = residual(root.value).reshape(count, 6)
        diag.update(
            component_residual_mol_s=float(np.max(np.abs(raw[:, :2]))),
            chemical_potential_error_RT=float(np.max(np.abs(raw[:, 2:4]))),
            pressure_relative_error=float(np.max(np.abs(raw[:, 4:])) / spec.pressure_Pa),
            heater_W=sum(max(q, 0.0) for q in duties),
            condenser_W=sum(max(-q, 0.0) for q in duties),
        )
        if any(
            q > d.maximum_power_W + 1e-10 or -q > d.maximum_cooling_power_W + 1e-10
            for q, d in zip(duties, spec.devices, strict=True)
        ):
            return fail("insufficient_stage_heating_or_cooling_power")
        for j, row in enumerate(rows):
            flash = molecular_tp_flash(
                spec.model.residual_model,
                spec.devices[j].setpoint_K,
                spec.pressure_Pa,
                dict(zip(spec.model.component_ids, row[:2] + row[2:4], strict=True)),
                basins=spec.basins,
                max_iterations=spec.tpd_max_iterations,
                tpd_tolerance=spec.tpd_tolerance,
            )
            diag["phase_checks"].append(flash.to_dict())
            if not flash.success or len(flash.phases) != 2:
                return fail("stage_phase_check:" + (flash.failure_reason or "not_two_phase"))
            by_basin = {p.basin: p for p in flash.phases}
            if any(
                b.name not in by_basin
                or np.max(
                    np.abs(
                        np.array(by_basin[b.name].state.mole_fractions)
                        - np.array(pairs[j][k].mole_fractions)
                    )
                )
                > 2e-5
                for k, b in enumerate(spec.basins)
            ):
                return fail("stage_phase_check_disagrees_with_balance_root")
        return MolecularProcessResult(
            True, (streams[0][0], streams[-1][1]), sum(duties), diagnostics=diag
        )
    except (ValueError, ArithmeticError) as exc:
        return fail(str(exc))
