"""Shared reactor contracts and lightweight validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

SteadyStateStability = Literal["stable", "unstable", "marginal"]
JacketInterpolationMode = Literal["step", "linear"]
PressureBoundaryMode = Literal["fixed_liquid"]


@dataclass(frozen=True)
class HeatTransferSpec:
    rho_cp_J_per_L_K: float = 4180.0
    ua_W_per_K: float = 0.0
    environment_temperature_K: float = 298.15
    jacket_ua_W_per_K: float = 0.0
    jacket_temperature_K: float | None = None
    fixed_heat_W: float = 0.0
    maximum_jacket_heating_W: float | None = None
    maximum_jacket_cooling_W: float | None = None

    def __post_init__(self) -> None:
        _positive(self.rho_cp_J_per_L_K, "rho_cp_J_per_L_K")
        _nonnegative_finite(self.ua_W_per_K, "ua_W_per_K")
        _nonnegative_finite(self.jacket_ua_W_per_K, "jacket_ua_W_per_K")
        _positive(self.environment_temperature_K, "environment_temperature_K")
        if self.jacket_temperature_K is not None:
            _positive(self.jacket_temperature_K, "jacket_temperature_K")
        _finite(self.fixed_heat_W, "fixed_heat_W")
        for name in ("maximum_jacket_heating_W", "maximum_jacket_cooling_W"):
            limit = getattr(self, name)
            if limit is not None:
                _nonnegative_finite(limit, name)

    def jacket_heat_w(self, temperature_K: float) -> float:
        jacket = 0.0
        if self.jacket_temperature_K is not None:
            jacket = self.jacket_ua_W_per_K * (self.jacket_temperature_K - temperature_K)
        return self.fixed_heat_W + self.limit_jacket_heat_w(jacket)

    def limit_jacket_heat_w(self, jacket: float) -> float:
        """Bound jacket power continuously; the independent fixed heater is unchanged."""
        if self.maximum_jacket_heating_W is not None:
            jacket = min(jacket, self.maximum_jacket_heating_W)
        if self.maximum_jacket_cooling_W is not None:
            jacket = max(jacket, -self.maximum_jacket_cooling_W)
        return jacket

    def heat_loss_w(self, temperature_K: float) -> float:
        return self.ua_W_per_K * (temperature_K - self.environment_temperature_K)


@dataclass(frozen=True)
class JacketTemperatureProgram:
    """Time-dependent jacket setpoint for dynamic batch energy balances."""

    setpoints: tuple[tuple[float, float], ...]
    mode: JacketInterpolationMode = "step"

    def __post_init__(self) -> None:
        if not self.setpoints:
            raise ValueError("jacket setpoints cannot be empty")
        previous_time = -1.0
        for time_s, temperature_K in self.setpoints:
            if time_s < 0 or not isfinite(time_s):
                raise ValueError("jacket setpoint times cannot be negative")
            if time_s < previous_time:
                raise ValueError("jacket setpoints must be sorted by time")
            if temperature_K <= 0 or not isfinite(temperature_K):
                raise ValueError("jacket setpoint temperatures must be positive")
            previous_time = time_s
        if self.mode not in {"step", "linear"}:
            raise ValueError("jacket program mode must be 'step' or 'linear'")

    def temperature_at(self, time_s: float) -> float:
        if time_s <= self.setpoints[0][0]:
            return float(self.setpoints[0][1])
        for (left_time, left_temperature), (
            right_time,
            right_temperature,
        ) in zip(self.setpoints, self.setpoints[1:], strict=False):
            if left_time <= time_s <= right_time:
                if self.mode == "step" or right_time == left_time:
                    return float(left_temperature)
                fraction = (time_s - left_time) / (right_time - left_time)
                return float(
                    left_temperature
                    + fraction * (right_temperature - left_temperature)
                )
        return float(self.setpoints[-1][1])

    def to_dict(self) -> dict[str, object]:
        return {
            "setpoints": [
                {"time_s": time_s, "temperature_K": temperature_K}
                for time_s, temperature_K in self.setpoints
            ],
            "mode": self.mode,
        }


@dataclass(frozen=True)
class FeedStreamSpec:
    species_flows_mol_s: dict[str, float]
    volumetric_flow_L_s: float
    temperature_K: float = 298.15

    def __post_init__(self) -> None:
        _nonnegative_finite(self.volumetric_flow_L_s, "volumetric_flow_L_s")
        _positive(self.temperature_K, "temperature_K")
        if any(
            value < 0 or not isfinite(value) for value in self.species_flows_mol_s.values()
        ):
            raise ValueError("species flows must be finite and nonnegative")

    def flow_for(self, species_id: str) -> float:
        return float(self.species_flows_mol_s.get(species_id, 0.0))


@dataclass(frozen=True)
class SemiBatchFeedSpec:
    stream: FeedStreamSpec
    start_s: float = 0.0
    end_s: float = 0.0

    def __post_init__(self) -> None:
        if (
            self.start_s < 0
            or self.end_s < 0
            or not isfinite(self.start_s)
            or not isfinite(self.end_s)
        ):
            raise ValueError("feed times cannot be negative")
        if self.end_s < self.start_s:
            raise ValueError("end_s cannot be earlier than start_s")

    def is_active(self, time_s: float) -> bool:
        return self.start_s <= time_s <= self.end_s

    def species_flow(self, species_id: str, time_s: float) -> float:
        if not self.is_active(time_s):
            return 0.0
        return self.stream.flow_for(species_id)

    def volumetric_flow(self, time_s: float) -> float:
        if not self.is_active(time_s):
            return 0.0
        return self.stream.volumetric_flow_L_s


@dataclass(frozen=True)
class WithdrawalSpec:
    """A time-windowed, well-mixed liquid withdrawal.

    Species leave at the instantaneous reactor concentration.  This keeps the
    volume and material-out ledgers on the same basis and avoids treating a
    volume loss as disappearing material.
    """

    volumetric_flow_L_s: float
    start_s: float = 0.0
    end_s: float = 0.0
    label: str = "withdrawal"

    def __post_init__(self) -> None:
        if self.volumetric_flow_L_s < 0 or not isfinite(self.volumetric_flow_L_s):
            raise ValueError("volumetric_flow_L_s must be finite and nonnegative")
        if (
            self.start_s < 0
            or self.end_s < self.start_s
            or not isfinite(self.start_s)
            or not isfinite(self.end_s)
        ):
            raise ValueError("withdrawal times must be nonnegative and ordered")
        if not self.label.strip():
            raise ValueError("withdrawal label cannot be empty")

    def volumetric_flow(self, time_s: float) -> float:
        if self.start_s <= time_s <= self.end_s:
            return self.volumetric_flow_L_s
        return 0.0


@dataclass(frozen=True)
class PressureBoundarySpec:
    """Declared liquid-pressure boundary for a zero-dimensional reactor.

    The reference slice is intentionally not a gas-headspace model.  Pressure
    is therefore a fixed boundary condition that is carried through results
    and checked against the declared validity domain.
    """

    pressure_Pa: float = 101_325.0
    mode: PressureBoundaryMode = "fixed_liquid"

    def __post_init__(self) -> None:
        _positive(self.pressure_Pa, "pressure_Pa")
        if self.mode != "fixed_liquid":
            raise ValueError("only the fixed_liquid pressure boundary is supported")


@dataclass(frozen=True)
class ReactorValidityDomain:
    """Numerical and physical validity limits used by reactor diagnostics."""

    minimum_temperature_K: float = 200.0
    maximum_temperature_K: float = 800.0
    minimum_pressure_Pa: float = 1.0
    maximum_pressure_Pa: float = 5.0e6
    maximum_temperature_rate_K_s: float = 5.0
    material_balance_tolerance_mol: float = 1.0e-8

    def __post_init__(self) -> None:
        _positive(self.minimum_temperature_K, "minimum_temperature_K")
        _positive(self.maximum_temperature_K, "maximum_temperature_K")
        if self.maximum_temperature_K <= self.minimum_temperature_K:
            raise ValueError("maximum_temperature_K must exceed minimum_temperature_K")
        _positive(self.minimum_pressure_Pa, "minimum_pressure_Pa")
        _positive(self.maximum_pressure_Pa, "maximum_pressure_Pa")
        if self.maximum_pressure_Pa <= self.minimum_pressure_Pa:
            raise ValueError("maximum_pressure_Pa must exceed minimum_pressure_Pa")
        _positive(self.maximum_temperature_rate_K_s, "maximum_temperature_rate_K_s")
        _positive(self.material_balance_tolerance_mol, "material_balance_tolerance_mol")

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_temperature_K": self.minimum_temperature_K,
            "maximum_temperature_K": self.maximum_temperature_K,
            "minimum_pressure_Pa": self.minimum_pressure_Pa,
            "maximum_pressure_Pa": self.maximum_pressure_Pa,
            "maximum_temperature_rate_K_s": self.maximum_temperature_rate_K_s,
            "material_balance_tolerance_mol": self.material_balance_tolerance_mol,
        }


@dataclass(frozen=True)
class SamplingEventSpec:
    """A well-mixed destructive sample removed from a dynamic batch reactor."""

    time_s: float
    volume_L: float
    label: str = "sample"

    def __post_init__(self) -> None:
        if self.time_s < 0 or not isfinite(self.time_s):
            raise ValueError("sampling time_s cannot be negative")
        if self.volume_L <= 0:
            raise ValueError("sampling volume_L must be positive")
        if not self.label.strip():
            raise ValueError("sampling label cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "time_s": self.time_s,
            "volume_L": self.volume_L,
            "label": self.label,
        }


@dataclass(frozen=True)
class ReactorState:
    amounts_mol: dict[str, float]
    volume_L: float
    temperature_K: float
    time_s: float
    pressure_Pa: float | None = None
    energy_jacket_J: float = 0.0
    heat_reaction_J: float = 0.0
    heat_loss_J: float = 0.0
    material_in_mol: dict[str, float] = field(default_factory=dict)
    material_out_mol: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _positive(self.volume_L, "volume_L")
        _positive(self.temperature_K, "temperature_K")
        if self.time_s < 0 or not isfinite(self.time_s):
            raise ValueError("time_s cannot be negative")
        if self.pressure_Pa is not None:
            _positive(self.pressure_Pa, "pressure_Pa")
        for value, name in (
            (self.energy_jacket_J, "energy_jacket_J"),
            (self.heat_reaction_J, "heat_reaction_J"),
            (self.heat_loss_J, "heat_loss_J"),
        ):
            _finite(value, name)
        for label, values in {
            "amounts_mol": self.amounts_mol,
            "material_in_mol": self.material_in_mol,
            "material_out_mol": self.material_out_mol,
        }.items():
            if any(
                value < -1e-12 or not isfinite(value) for value in values.values()
            ):
                raise ValueError(f"{label} must contain finite nonnegative amounts")

    @property
    def concentrations_mol_L(self) -> dict[str, float]:  # noqa: N802 - unit suffix
        return {
            species_id: max(float(amount), 0.0) / self.volume_L
            for species_id, amount in self.amounts_mol.items()
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "amounts_mol": dict(self.amounts_mol),
            "volume_L": self.volume_L,
            "temperature_K": self.temperature_K,
            "time_s": self.time_s,
            "pressure_Pa": self.pressure_Pa,
            "concentrations_mol_L": self.concentrations_mol_L,
            "energy_jacket_J": self.energy_jacket_J,
            "heat_reaction_J": self.heat_reaction_J,
            "heat_loss_J": self.heat_loss_J,
            "material_in_mol": dict(self.material_in_mol),
            "material_out_mol": dict(self.material_out_mol),
        }


@dataclass(frozen=True)
class ReactorResult:
    reactor_id: str
    model_id: str
    network_id: str
    initial_state: ReactorState
    final_state: ReactorState
    times_s: tuple[float, ...]
    amounts_mol: dict[str, tuple[float, ...]]
    temperatures_K: tuple[float, ...]
    material_balance_error_mol: float
    volumes_L: tuple[float, ...] = ()
    pressures_Pa: tuple[float, ...] = ()
    validity_domain: ReactorValidityDomain | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        point_count = len(self.times_s)
        if point_count == 0:
            raise ValueError("reactor result trajectory cannot be empty")
        if any(not isfinite(value) for value in self.times_s):
            raise ValueError("reactor result times must be finite")
        if any(right < left for left, right in zip(self.times_s, self.times_s[1:], strict=False)):
            raise ValueError("reactor result times must be monotonic")
        if len(self.temperatures_K) != point_count:
            raise ValueError("temperature trajectory length must match times_s")
        if any(not isfinite(value) or value <= 0 for value in self.temperatures_K):
            raise ValueError("temperature trajectory must be finite and positive")
        for species_id, values in self.amounts_mol.items():
            if any(not isfinite(value) for value in values):
                raise ValueError(f"amount trajectory must be finite for {species_id}")
        for label, values in (("volumes_L", self.volumes_L), ("pressures_Pa", self.pressures_Pa)):
            if values and len(values) != point_count:
                raise ValueError(f"{label} length must match times_s")
            if any(not isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"{label} must be finite and positive")
        if self.material_balance_error_mol < 0 or not isfinite(self.material_balance_error_mol):
            raise ValueError("material_balance_error_mol must be finite and nonnegative")

    @property
    def diagnostics(self) -> ReactorDiagnostics:
        """Return deterministic conservation, positivity, and energy-ledger checks."""

        trajectory_values = [value for values in self.amounts_mol.values() for value in values]
        minimum_amount = min(trajectory_values, default=0.0)
        warnings: list[str] = []
        domain = self.validity_domain or ReactorValidityDomain()
        if minimum_amount < -1.0e-10:
            warnings.append("negative_species_amount")
        if self.material_balance_error_mol > domain.material_balance_tolerance_mol:
            warnings.append("material_balance_residual_exceeds_tolerance")
        minimum_temperature = min(self.temperatures_K, default=self.final_state.temperature_K)
        maximum_temperature = max(self.temperatures_K, default=self.final_state.temperature_K)
        maximum_temperature_rate = _maximum_rate(self.times_s, self.temperatures_K)
        if (
            minimum_temperature < domain.minimum_temperature_K
            or maximum_temperature > domain.maximum_temperature_K
        ):
            warnings.append("temperature_outside_validity_domain")
        if maximum_temperature_rate > domain.maximum_temperature_rate_K_s:
            warnings.append("thermal_runaway_rate_exceeds_limit")
        if self.pressures_Pa and (
            min(self.pressures_Pa) < domain.minimum_pressure_Pa
            or max(self.pressures_Pa) > domain.maximum_pressure_Pa
        ):
            warnings.append("pressure_outside_validity_domain")
        return ReactorDiagnostics(
            material_balance_error_mol=self.material_balance_error_mol,
            minimum_species_amount_mol=float(minimum_amount),
            nonnegative_species=minimum_amount >= -1.0e-10,
            energy_ledger_net_J=(
                self.final_state.energy_jacket_J
                + self.final_state.heat_reaction_J
                - self.final_state.heat_loss_J
            ),
            sensible_energy_input_J=(
                self.final_state.energy_jacket_J
                - self.final_state.heat_reaction_J
                - self.final_state.heat_loss_J
            ),
            material_balance_closed=(
                self.material_balance_error_mol <= domain.material_balance_tolerance_mol
            ),
            minimum_temperature_K=float(minimum_temperature),
            maximum_temperature_K=float(maximum_temperature),
            maximum_temperature_rate_K_s=float(maximum_temperature_rate),
            within_validity_domain=not any(
                warning
                in {
                    "temperature_outside_validity_domain",
                    "pressure_outside_validity_domain",
                    "thermal_runaway_rate_exceeds_limit",
                }
                for warning in warnings
            ),
            warnings=tuple(warnings),
        )

    def conversion(self, reactant_id: str) -> float:
        initial = self.initial_state.amounts_mol.get(reactant_id, 0.0)
        final = self.final_state.amounts_mol.get(reactant_id, 0.0)
        if initial <= 0:
            return 0.0
        return max(0.0, min(1.0, (initial - final) / initial))

    def yield_on(self, product_id: str, reactant_id: str) -> float:
        initial_reactant = self.initial_state.amounts_mol.get(reactant_id, 0.0)
        final_product = self.final_state.amounts_mol.get(product_id, 0.0)
        initial_product = self.initial_state.amounts_mol.get(product_id, 0.0)
        if initial_reactant <= 0:
            return 0.0
        return max(0.0, (final_product - initial_product) / initial_reactant)

    def to_dict(self) -> dict[str, object]:
        return {
            "reactor_id": self.reactor_id,
            "model_id": self.model_id,
            "network_id": self.network_id,
            "initial_state": self.initial_state.to_dict(),
            "final_state": self.final_state.to_dict(),
            "times_s": list(self.times_s),
            "amounts_mol": {
                species_id: list(values) for species_id, values in self.amounts_mol.items()
            },
            "temperatures_K": list(self.temperatures_K),
            "material_balance_error_mol": self.material_balance_error_mol,
            "volumes_L": list(self.volumes_L),
            "pressures_Pa": list(self.pressures_Pa),
            "validity_domain": (
                self.validity_domain.to_dict() if self.validity_domain is not None else None
            ),
            "diagnostics": self.diagnostics.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReactorDiagnostics:
    """Machine-readable checks attached to every reactor result."""

    material_balance_error_mol: float
    minimum_species_amount_mol: float
    nonnegative_species: bool
    energy_ledger_net_J: float
    sensible_energy_input_J: float = 0.0
    material_balance_closed: bool = True
    minimum_temperature_K: float = 0.0
    maximum_temperature_K: float = 0.0
    maximum_temperature_rate_K_s: float = 0.0
    within_validity_domain: bool = True
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "material_balance_error_mol": self.material_balance_error_mol,
            "minimum_species_amount_mol": self.minimum_species_amount_mol,
            "nonnegative_species": self.nonnegative_species,
            "energy_ledger_net_J": self.energy_ledger_net_J,
            "sensible_energy_input_J": self.sensible_energy_input_J,
            "material_balance_closed": self.material_balance_closed,
            "minimum_temperature_K": self.minimum_temperature_K,
            "maximum_temperature_K": self.maximum_temperature_K,
            "maximum_temperature_rate_K_s": self.maximum_temperature_rate_K_s,
            "within_validity_domain": self.within_validity_domain,
            "warnings": list(self.warnings),
        }


def _positive(value: float, name: str) -> None:
    if value <= 0 or not isfinite(value):
        raise ValueError(f"{name} must be finite and positive")


def _finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _nonnegative_finite(value: float, name: str) -> None:
    if value < 0 or not isfinite(value):
        raise ValueError(f"{name} must be finite and nonnegative")


def _maximum_rate(times: tuple[float, ...], values: tuple[float, ...]) -> float:
    rates = [
        abs((right_value - left_value) / (right_time - left_time))
        for left_time, right_time, left_value, right_value in zip(
            times,
            times[1:],
            values,
            values[1:],
            strict=False,
        )
        if right_time > left_time
    ]
    return max(rates, default=0.0)


__all__ = [
    "FeedStreamSpec",
    "HeatTransferSpec",
    "JacketInterpolationMode",
    "JacketTemperatureProgram",
    "PressureBoundaryMode",
    "PressureBoundarySpec",
    "ReactorDiagnostics",
    "ReactorResult",
    "ReactorState",
    "ReactorValidityDomain",
    "SamplingEventSpec",
    "SemiBatchFeedSpec",
    "SteadyStateStability",
    "WithdrawalSpec",
]
