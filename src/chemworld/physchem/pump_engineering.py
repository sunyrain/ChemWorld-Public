"""Pump and electrical-drive engineering; caller-owned data, existing flow state.

Corripio equations estimate centrifugal pump/motor efficiency. Sourced motor,
VFD and supply catalogs are configuration, not another device state or runtime.
No standards or country observations are redistributed with these interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite, log, sqrt
from typing import Any, cast

from chemworld.data.scientific_inputs import InputSource, ScientificRow, ScientificTable
from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.transport import PumpSpec, pump_operating_point


def _nonnegative(value: float, name: str) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _efficiency(value: float) -> float:
    if not isfinite(value) or not 0 < value <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    return value


def _cell(row: ScientificRow, name: str, unit: str = "dimensionless") -> float:
    cell = row.values[name].to(unit)
    if cell.quality != "valid" or cell.value is None:
        raise ValueError(f"valid {name} required")
    return cast(float, cell.value)


def corripio_pump_efficiency(volumetric_flow_m3_s: float) -> float:
    """Centrifugal estimate, restricted to the published 50-5000 US GPM range."""
    _positive(volumetric_flow_m3_s, "volumetric flow")
    flow_gpm = volumetric_flow_m3_s * 15850.323
    if not 50 <= flow_gpm <= 5000:
        raise ValueError("Corripio pump correlation requires 50-5000 GPM")
    x = log(flow_gpm)
    return _efficiency(-0.316 + 0.24015 * x - 0.01199 * x * x)


def corripio_motor_efficiency(rated_power_W: float) -> float:
    """Published estimate with its rounded W/hp constant; not a motor guarantee."""
    _positive(rated_power_W, "rated power")
    x = log(rated_power_W / 745.69987)
    return _efficiency(0.8 + 0.0319 * x - 0.00182 * x * x)


def pump_specific_speed(flow_m3_s: float, head_m: float, speed_rpm: float) -> float:
    """Dimensional engineering index n_rpm sqrt(Q_m3/s) / H_m**0.75."""
    for value, name in ((flow_m3_s, "flow"), (head_m, "head"), (speed_rpm, "speed")):
        _positive(value, name)
    return speed_rpm * sqrt(flow_m3_s) / head_m**0.75


def pump_specific_diameter(flow_m3_s: float, head_m: float, diameter_m: float) -> float:
    """Dimensional index D_m H_m**0.25 / sqrt(Q_m3/s), not dimensionless."""
    for value, name in ((flow_m3_s, "flow"), (head_m, "head"), (diameter_m, "diameter")):
        _positive(value, name)
    return diameter_m * head_m**0.25 / sqrt(flow_m3_s)


def motor_synchronous_speed(frequency_Hz: float, poles: int) -> float:
    """Synchronous speed in RPM; actual induction speed also requires slip."""
    _positive(frequency_Hz, "frequency")
    if isinstance(poles, bool) or not isinstance(poles, int) or poles < 2 or poles % 2:
        raise ValueError("motor poles must be a positive even integer")
    return 120 * frequency_Hz / poles


def ideal_motor_current(
    electrical_power_W: float, voltage_V: float, *, phases: int = 3, power_factor: float = 1
) -> float:
    """RMS line current; input is electrical active power, not mechanical output."""
    _nonnegative(electrical_power_W, "electrical power")
    _positive(voltage_V, "voltage")
    _efficiency(power_factor)
    if isinstance(phases, bool) or phases not in (1, 3):
        raise ValueError("only single-phase or balanced three-phase supply is supported")
    return electrical_power_W / (voltage_V * power_factor * (sqrt(3) if phases == 3 else 1))


@dataclass(frozen=True)
class DriveEfficiencyResult:
    efficiency: float
    requested: Mapping[str, float]
    evaluated: Mapping[str, float]
    clipped: bool
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class DriveEfficiencyMap:
    """Original multilinear table with explicit reject/clip and rounding choices.

    Axes have caller-specified names and SI units. The source table, including
    uncertainties and citations, is retained by the original scalar function.
    """

    function: TabulatedScalar

    def __post_init__(self) -> None:
        if self.function.output_unit != "dimensionless":
            raise ValueError("drive efficiency must be dimensionless")
        for bound in self.function.value_range:
            _efficiency(bound)

    def to_dict(self) -> dict[str, Any]:
        return {"function": self.function.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DriveEfficiencyMap:
        return cls(TabulatedScalar.from_dict(data["function"]))

    def evaluate(
        self, inputs: Mapping[str, float], *, outside: str = "reject", digits: int | None = None
    ) -> DriveEfficiencyResult:
        if outside not in ("reject", "clip"):
            raise ValueError("outside policy must be reject or clip")
        if inputs.keys() != self.function.domain.keys() or any(
            not isfinite(x) for x in inputs.values()
        ):
            raise ValueError("finite inputs on exactly the declared axes required")
        actual = dict(inputs)
        if outside == "clip":
            actual = {
                key: min(hi, max(lo, inputs[key])) for key, (lo, hi) in self.function.domain.items()
            }
        value = self.function.value(actual)
        if digits is not None:
            if isinstance(digits, bool) or not isinstance(digits, int) or not 0 <= digits <= 15:
                raise ValueError("rounding digits must be an integer from 0 to 15")
            value = round(value, digits)
        source_ids = tuple(
            sorted(
                {
                    row.source_id
                    for row in self.function.table.rows
                    if row.row_id in self.function.row_ids
                }
            )
        )
        return DriveEfficiencyResult(
            _efficiency(value), dict(inputs), actual, actual != dict(inputs), source_ids
        )


class MotorEfficiencyCatalog:
    """CSA-style full-load curves; groups are 'minimum|high:open|closed:poles'."""

    def __init__(self, table: ScientificTable) -> None:
        self.table = table
        self.curves = {
            group: DriveEfficiencyMap(
                TabulatedScalar(
                    table,
                    (("power_W", "power", "W"),),
                    "efficiency",
                    "dimensionless",
                    tuple(row.row_id for row in table.rows if row.group_id == group),
                )
            )
            for group in sorted({row.group_id for row in table.rows})
        }

    def efficiency(
        self,
        power_W: float,
        *,
        closed: bool = False,
        poles: int = 2,
        high_efficiency: bool = False,
        outside: str = "reject",
        digits: int | None = None,
    ) -> DriveEfficiencyResult:
        _positive(power_W, "rated power")
        group = (
            f"{'high' if high_efficiency else 'minimum'}:{'closed' if closed else 'open'}:{poles}"
        )
        if group not in self.curves:
            raise ValueError("no explicitly supplied efficiency curve for this motor class")
        return self.curves[group].evaluate({"power_W": power_W}, outside=outside, digits=digits)

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotorEfficiencyCatalog:
        return cls(ScientificTable.from_dict(data["table"]))


class MotorSizeCatalog:
    """Explicit standard power sizes; nominal horsepower may be another source column."""

    def __init__(self, table: ScientificTable) -> None:
        if table.kind != "parameters" or not table.rows:
            raise ValueError("a nonempty motor parameter table is required")
        self.table = table
        self.rows = tuple(sorted(table.rows, key=lambda row: _cell(row, "power", "W")))
        powers = tuple(_cell(row, "power", "W") for row in self.rows)
        if len(set(powers)) != len(powers) or min(powers) <= 0:
            raise ValueError("motor sizes must be distinct positive powers")

    def select(self, required_power_W: float) -> ScientificRow:
        _nonnegative(required_power_W, "required power")
        for row in self.rows:
            if _cell(row, "power", "W") >= required_power_W:
                return row
        raise ValueError("required power exceeds supplied motor size catalog")

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotorSizeCatalog:
        return cls(ScientificTable.from_dict(data["table"]))


class MotorPartLoadCatalog:
    """Sourced quartic load curves, selected by increasing upper rated power.

    Columns: upper_power (W), unity_load, c0..c4 (dimensionless). The final
    curve covers powers above its selection threshold, as in the source.
    That selection convention does not extend empirical material validity.
    """

    def __init__(self, table: ScientificTable) -> None:
        if table.kind != "parameters" or not table.rows:
            raise ValueError("a nonempty partial-load parameter table is required")
        self.table = table
        self.rows = tuple(sorted(table.rows, key=lambda r: _cell(r, "upper_power", "W")))
        limits = tuple(_cell(row, "upper_power", "W") for row in self.rows)
        if min(limits) <= 0 or len(set(limits)) != len(limits):
            raise ValueError("load curve selection thresholds must be distinct positive powers")
        for row in self.rows:
            _efficiency(_cell(row, "unity_load"))
            for i in range(5):
                if not isfinite(_cell(row, f"c{i}")):
                    raise ValueError("finite load curve coefficient required")

    def factor(self, rated_power_W: float, load: float) -> float:
        _positive(rated_power_W, "rated power")
        if not isfinite(load) or not 0 <= load <= 1:
            raise ValueError("load fraction must be in [0,1]")
        row = next(
            (r for r in self.rows if rated_power_W <= _cell(r, "upper_power", "W")), self.rows[-1]
        )
        if load > _cell(row, "unity_load"):
            return 1.0
        return _efficiency(sum(_cell(row, f"c{i}") * load**i for i in range(5)))

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table.to_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MotorPartLoadCatalog:
        return cls(ScientificTable.from_dict(data["table"]))


@dataclass(frozen=True)
class ElectricalSupplyRecord:
    code: str
    country: str
    sector: str
    frequency_Hz: float
    voltages_V: tuple[float, ...]
    plugs: tuple[str, ...] | None
    source: InputSource

    def __post_init__(self) -> None:
        if not self.code or not self.country or self.sector not in ("residential", "industrial"):
            raise ValueError("supply identity and residential/industrial sector required")
        _positive(self.frequency_Hz, "supply frequency")
        voltages = tuple(self.voltages_V)
        if not voltages:
            raise ValueError("at least one supply voltage required")
        for voltage in voltages:
            _positive(voltage, "supply voltage")
        object.__setattr__(self, "voltages_V", voltages)
        if self.plugs is not None:
            if any(not isinstance(p, str) or not p for p in self.plugs):
                raise ValueError("nonempty plug labels required")
            object.__setattr__(self, "plugs", tuple(self.plugs))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source"] = self.source.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ElectricalSupplyRecord:
        return cls(**{**data, "source": InputSource.from_dict(data["source"])})


@dataclass(frozen=True)
class ElectricalSupplyCatalog:
    """Labeled source records and source-declared options; not regulatory advice.

    Declared options are retained separately from the union of records, which
    need not be equal in a historical source. Industrial missing plugs stay None.
    """

    records: tuple[ElectricalSupplyRecord, ...]
    plug_types: tuple[str, ...]
    residential_frequencies_Hz: tuple[float, ...]
    residential_voltages_V: tuple[float, ...]
    industrial_voltages_V: tuple[float, ...]
    source: InputSource

    def __post_init__(self) -> None:
        for name in (
            "records",
            "plug_types",
            "residential_frequencies_Hz",
            "residential_voltages_V",
            "industrial_voltages_V",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        identities = [(r.sector, r.code) for r in self.records]
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate sector/country supply identity")
        for value in (
            *self.residential_frequencies_Hz,
            *self.residential_voltages_V,
            *self.industrial_voltages_V,
        ):
            _positive(value, "supply option")

    def lookup(self, sector: str, code: str) -> ElectricalSupplyRecord:
        for record in self.records:
            if (record.sector, record.code) == (sector, code):
                return record
        raise ValueError("unknown source country/sector supply")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["records"] = [r.to_dict() for r in self.records]
        result["source"] = self.source.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ElectricalSupplyCatalog:
        return cls(
            **{
                **data,
                "source": InputSource.from_dict(data["source"]),
                "records": tuple(ElectricalSupplyRecord.from_dict(r) for r in data["records"]),
            }
        )


@dataclass(frozen=True)
class PumpDriveSpec:
    """Electrical supply around the existing bounded ideal pump controller.

    Rated motor output is mechanical shaft power; motor/VFD efficiencies are
    independently selected configuration values. No fictitious heat is added
    to the fluid: electrical losses are reported for the surrounding process.
    """

    pump: PumpSpec
    motor_rated_output_W: float
    motor_efficiency: float
    vfd_efficiency: float = 1

    def __post_init__(self) -> None:
        _positive(self.motor_rated_output_W, "motor rated output")
        _efficiency(self.motor_efficiency)
        _efficiency(self.vfd_efficiency)

    @property
    def effective_pump(self) -> PumpSpec:
        return PumpSpec(
            self.pump.maximum_pressure_rise_Pa,
            min(self.pump.maximum_shaft_power_W, self.motor_rated_output_W),
            self.pump.efficiency,
        )

    def operating_point(self, requested_flow_m3_s: float, pressure_load: Any) -> dict[str, Any]:
        report = pump_operating_point(self.effective_pump, requested_flow_m3_s, pressure_load)
        shaft = report["shaft_power_W"]
        electrical = shaft / (self.motor_efficiency * self.vfd_efficiency)
        return {
            **report,
            "electrical_power_W": electrical,
            "drive_loss_W": electrical - shaft,
            "total_external_loss_W": electrical - report["hydraulic_power_W"],
            "motor_load_fraction": shaft / self.motor_rated_output_W,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PumpDriveSpec:
        return cls(**{**data, "pump": PumpSpec(**data["pump"])})
