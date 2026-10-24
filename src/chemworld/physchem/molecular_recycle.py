"""Explicit one-loop steady process using the original molecular unit models.

The tear is a numerical candidate, never a material inventory. Only the final
product and net device utilities can be consumed by a finite receiver.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from math import fsum, isclose, isfinite
from typing import Any

import numpy as np

from chemworld.physchem.molecular_mixture import MolecularStream
from chemworld.physchem.molecular_processes import (
    MolecularHeatDuty,
    MolecularPressureStage,
    MolecularProcessDomain,
    MolecularProcessResult,
    molecular_mix_streams,
    molecular_process_state,
    molecular_split_stream,
)
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)


@dataclass(frozen=True)
class MolecularRecyclePolicy:
    root: RootPolicy = field(default_factory=lambda: RootPolicy(tolerance=1e-9))
    relative_difference_step: float = 6e-6
    maximum_trial_evaluations: int = 1000

    def __post_init__(self) -> None:
        if (
            not isinstance(self.root, RootPolicy)
            or not isfinite(self.relative_difference_step)
            or self.relative_difference_step <= 0
            or type(self.maximum_trial_evaluations) is not int
            or self.maximum_trial_evaluations < 1
        ):
            raise ValueError("recycle requires explicit finite solver budgets and difference step")


@dataclass(frozen=True)
class MolecularRecyclePath:
    domain: MolecularProcessDomain
    source_id: str
    inlet: MolecularStream
    devices: tuple[MolecularPressureStage | MolecularHeatDuty, ...]
    recycle_fraction: float
    mixer_pressure_Pa: float
    initial_recycle_rates_mol_s: tuple[float, ...]
    initial_temperature_K: float
    policy: MolecularRecyclePolicy = field(default_factory=MolecularRecyclePolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "devices", tuple(self.devices))
        object.__setattr__(
            self, "initial_recycle_rates_mol_s", tuple(self.initial_recycle_rates_mol_s)
        )
        self.domain.validate_stream(self.inlet)
        if (
            not isinstance(self.source_id, str)
            or not self.source_id
            or self.inlet.molar_rate_mol_s <= 0
            or not isfinite(self.recycle_fraction)
            or not 0 <= self.recycle_fraction < 1
            or not isfinite(self.mixer_pressure_Pa)
            or self.mixer_pressure_Pa <= 0
            or self.mixer_pressure_Pa > self.inlet.state.pressure_Pa + 1e-5
            or len(self.initial_recycle_rates_mol_s) != len(self.domain.model.component_ids)
            or any(not isfinite(n) or n < 0 for n in self.initial_recycle_rates_mol_s)
            or not isfinite(self.initial_temperature_K)
            or not isinstance(self.policy, MolecularRecyclePolicy)
        ):
            raise ValueError(
                "recycle requires explicit positive feed, pressure and finite tear guess"
            )
        pressure = self.mixer_pressure_Pa
        for device in self.devices:
            if not isinstance(device, (MolecularPressureStage, MolecularHeatDuty)):
                raise ValueError("recycle supports declared molecular heat and pressure devices")
            if device.domain != self.domain:
                raise ValueError("recycle devices must use the same EOS and caloric reference")
            if device.outlet_pressure_Pa is not None:
                pressure = device.outlet_pressure_Pa
        if self.recycle_fraction and not isclose(
            pressure, self.mixer_pressure_Pa, rel_tol=1e-10, abs_tol=1e-5
        ):
            raise ValueError("recycle pressure loop must close through declared devices")

    def to_dict(self) -> dict[str, Any]:
        s = self.inlet.state
        return {
            "model": "molecular-single-recycle-v1",
            "source_id": self.source_id,
            "source_scope": "external_prescribed_reservoir",
            "domain": self.domain.to_dict(),
            "inlet": {
                "temperature_K": s.temperature_K,
                "molar_density_mol_m3": s.molar_density_mol_m3,
                "mole_fractions": s.mole_fractions,
                "molar_rate_mol_s": self.inlet.molar_rate_mol_s,
            },
            "devices": [d.to_dict() for d in self.devices],
            "recycle_fraction": self.recycle_fraction,
            "mixer_pressure_Pa": self.mixer_pressure_Pa,
            "initial_recycle_rates_mol_s": self.initial_recycle_rates_mol_s,
            "initial_temperature_K": self.initial_temperature_K,
            "policy": asdict(self.policy),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularRecyclePath:
        if (
            data.get("model") != "molecular-single-recycle-v1"
            or data.get("source_scope") != "external_prescribed_reservoir"
        ):
            raise ValueError("unknown recycle topology or undeclared finite source")
        domain = MolecularProcessDomain.from_dict(data["domain"])
        value = data["inlet"]
        inlet = MolecularStream(
            domain.state(
                value["temperature_K"], value["molar_density_mol_m3"], value["mole_fractions"]
            ),
            value["molar_rate_mol_s"],
        )
        devices = []
        for value in data["devices"]:
            kind = value.get("kind")
            if kind not in {"heat", "pressure"}:
                raise ValueError("unknown recycle device")
            devices.append(
                (MolecularHeatDuty if kind == "heat" else MolecularPressureStage).from_dict(value)
            )
        policy = dict(data["policy"])
        policy["root"] = RootPolicy(**policy["root"])
        return cls(
            domain,
            data["source_id"],
            inlet,
            tuple(devices),
            data["recycle_fraction"],
            data["mixer_pressure_Pa"],
            tuple(data["initial_recycle_rates_mol_s"]),
            data["initial_temperature_K"],
            MolecularRecyclePolicy(**policy),
        )

    def evaluate(self) -> MolecularProcessResult:
        """Solve the real loop, returning one product and one set of net utilities."""
        names, fraction = self.domain.model.component_ids, self.recycle_fraction
        count = len(names)
        trials: list[dict[str, Any]] = []
        last: dict[str, Any] = {}
        scales = np.array([self.inlet.molar_rate_mol_s] * count + [self.inlet.state.temperature_K])
        low = np.array([0.0] * count + [self.domain.temperature_range_K[0]])
        high = np.array([np.inf] * count + [self.domain.temperature_range_K[1]])

        def accepted(result: MolecularProcessResult, stage: str) -> MolecularStream:
            if not result.success or result.outlet is None:
                raise ResidualDomainError(f"{stage}:{result.failure_reason}")
            return result.outlet

        def physical(point: np.ndarray) -> np.ndarray:
            if len(trials) >= self.policy.maximum_trial_evaluations:
                raise ResidualDomainError("recycle_trial_budget_exhausted")
            row: dict[str, Any] = {
                "tear_rates_mol_s": point[:count].tolist(),
                "tear_temperature_K": float(point[-1]),
            }
            trials.append(row)
            try:
                total = fsum(point[:count])
                x = tuple(point[:count] / total) if total > 0 else self.inlet.state.mole_fractions
                closed = molecular_process_state(
                    self.domain,
                    pressure_Pa=self.mixer_pressure_Pa,
                    coordinate="T",
                    value=float(point[-1]),
                    mole_fractions=x,
                    initial_temperature_K=float(point[-1]),
                )
                if not closed.success or closed.state is None:
                    raise ResidualDomainError(f"tear_TP:{closed.failure_reason}")
                tear = MolecularStream(closed.state, total)
                mixed = molecular_mix_streams(
                    self.domain,
                    (self.inlet, tear),
                    outlet_pressure_Pa=self.mixer_pressure_Pa,
                    initial_temperature_K=self.inlet.state.temperature_K,
                )
                stream = accepted(mixed, "mixer")
                mixer_state = stream.state.to_dict()
                heat, work, devices = 0.0, 0.0, []
                for i, device in enumerate(self.devices):
                    result = device.evaluate(stream)
                    devices.append(result.to_dict())
                    stream = accepted(result, f"device_{i}")
                    heat += result.heat_duty_W
                    work += result.shaft_work_W
                if fraction and not isclose(
                    stream.state.pressure_Pa, self.mixer_pressure_Pa, rel_tol=1e-8, abs_tol=1e-4
                ):
                    raise ResidualDomainError("recycle_pressure_does_not_close")
                split = molecular_split_stream(self.domain, stream, (1 - fraction, fraction))
                product, recycled = split.outlets
                returned = np.array(
                    [recycled.molar_rate_mol_s * x for x in recycled.state.mole_fractions]
                )
                residual = np.r_[
                    point[:count] - returned, float(point[-1]) - recycled.state.temperature_K
                ]
                row.update(success=True, residual=residual.tolist())
                last.update(
                    tear=tear,
                    product=product,
                    heat=heat,
                    work=work,
                    mixer=mixer_state,
                    devices=devices,
                    residual=residual,
                )
                return residual
            except (ArithmeticError, ValueError) as error:
                row.update(success=False, reason=str(error))
                raise ResidualDomainError(str(error)) from error

        # At zero recycle no fictitious tear temperature equation is required.
        initial = np.array([*self.initial_recycle_rates_mol_s, self.initial_temperature_K])
        diagnostic: dict[str, Any] = {
            "topology": "one_mixer_serial_devices_nonselective_split_one_tear",
            "source_id": self.source_id,
            "source_scope": "external_prescribed_reservoir",
            "derivative_mode": "explicit_finite_difference_of_original_units",
            "policy": asdict(self.policy),
            "initial": initial.tolist(),
            "trials": trials,
            "unknown_count": count + 1 if fraction else 0,
            "equation_count": count + 1 if fraction else 0,
            "trial_resource_scope": "numerical_diagnostics_only; no_experiment_debit",
        }
        if fraction:
            variables = (
                *(
                    ResidualVariable(k, "mol/s", scales[i], "algebraic", lower=0)
                    for i, k in enumerate(names)
                ),
                ResidualVariable(
                    "tear_T", "K", scales[-1], "algebraic", lower=low[-1], upper=high[-1]
                ),
            )
            root = solve_root(
                physical,
                initial,
                variables=variables,
                residual_scales=scales,
                jacobian=lambda q: finite_jacobian(
                    physical,
                    q,
                    scales,
                    lower=low,
                    upper=high,
                    relative_step=self.policy.relative_difference_step,
                ),
                policy=self.policy.root,
            )
            diagnostic["root"] = root.diagnostics
            if not root.success:
                diagnostic["trial_evaluations"] = len(trials)
                return MolecularProcessResult(
                    False, failure_reason=root.failure_reason, diagnostics=diagnostic
                )
            final = root.value
        else:
            final = np.r_[np.zeros(count), self.inlet.state.temperature_K]
        try:
            physical(final)
        except ResidualDomainError as error:
            diagnostic["trial_evaluations"] = len(trials)
            return MolecularProcessResult(False, failure_reason=str(error), diagnostics=diagnostic)
        product = last["product"]
        balance = {
            k: product.molar_rate_mol_s * product.state.mole_fractions[i]
            - self.inlet.molar_rate_mol_s * self.inlet.state.mole_fractions[i]
            for i, k in enumerate(names)
        }
        energy = product.enthalpy_rate_W - self.inlet.enthalpy_rate_W - last["heat"] - last["work"]
        diagnostic.update(
            trial_evaluations=len(trials),
            tear=last["tear"].state.to_dict(),
            tear_molar_rate_mol_s=last["tear"].molar_rate_mol_s,
            tear_residual=last["residual"].tolist(),
            mixer=last["mixer"],
            devices=last["devices"],
            component_balance_mol_s=balance,
            energy_residual_W=energy,
        )
        if max(map(abs, balance.values())) > 1e-8 * self.inlet.molar_rate_mol_s or abs(
            energy
        ) > 1e-5 + 1e-8 * max(
            abs(product.enthalpy_rate_W),
            abs(self.inlet.enthalpy_rate_W),
            abs(last["heat"]),
            abs(last["work"]),
        ):
            return MolecularProcessResult(
                False, failure_reason="recycle_boundary_balance_failed", diagnostics=diagnostic
            )
        return MolecularProcessResult(
            True, (product,), last["heat"], last["work"], diagnostics=diagnostic
        )
