"""Native public readings and the versioned scalar compatibility projection.

Readings own their values. Legacy dictionaries and raw acquisition packets are
views of these records, never separately writable observation state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from math import isfinite, isnan
from typing import Any, cast

from chemworld.foundation.units import unit_spec

LEGACY_PROJECTION_VERSION = "chemworld-gym-scalars-v1"
QUALITIES = frozenset(
    {"valid", "not_observed", "missing", "below_detection", "saturated", "out_of_range",
     "invalid", "fit_failed"}
)


@dataclass(frozen=True)
class ObservationTime:
    clock_id: str = "world"
    time_s: float = 0.0


DEFAULT_OBSERVATION_TIME = ObservationTime()


def legacy_scalar_projection(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """The sole v1 normalization formula; callers choose an explicit native range."""
    if not isfinite(value):
        return value
    return min(1.0, max(0.0, (value - lower) / (upper - lower)))


@dataclass(frozen=True)
class ChannelAxis:
    name: str
    unit: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True)
class ObservationChannel:
    channel_id: str
    unit: str
    value: float | tuple[float, ...] | None
    quality: str = "valid"
    axis: ChannelAxis | None = None
    valid_range: tuple[float, float] | None = None
    detection_limit: float | None = None
    uncertainty: float | None = None
    sample_id: str | None = None
    sampled_at: ObservationTime = ObservationTime()
    available_at: ObservationTime = ObservationTime()
    role: str = "raw"
    derived_from: tuple[str, ...] = ()
    acquisition_relation: str = "direct"

    def __post_init__(self) -> None:
        if (
            isinstance(self.value, float)
            and isnan(self.value)
            and self.quality in {
                "missing", "not_observed", "below_detection", "invalid", "fit_failed",
            }
        ):
            object.__setattr__(self, "value", None)
        if isinstance(self.value, (list, tuple)):
            object.__setattr__(self, "value", tuple(self.value))
        if self.valid_range is not None:
            if len(self.valid_range) != 2:
                raise ValueError("Channel range must contain two bounds")
            object.__setattr__(self, "valid_range", tuple(self.valid_range))
        object.__setattr__(self, "derived_from", tuple(self.derived_from))

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.value),) if isinstance(self.value, tuple) else ()

    @property
    def observed(self) -> bool:
        return self.quality not in {"not_observed", "missing", "invalid", "fit_failed"}

    @property
    def projectable(self) -> bool:
        """Legacy scalar availability; native censoring flags remain intact."""
        return self.value is not None and self.quality in {"valid", "saturated"}

    def errors(self) -> list[str]:
        errors: list[str] = []
        if not self.channel_id or self.quality not in QUALITIES:
            errors.append("identity_or_quality")
        if self.sample_id is not None and (
            not isinstance(self.sample_id, str) or not self.sample_id.strip()
        ):
            errors.append("sample_identity")
        try:
            unit_spec(self.unit)
        except ValueError:
            errors.append("unit")
        values = self.value if isinstance(self.value, tuple) else (self.value,)
        missing = self.value is None or (isinstance(self.value, float) and isnan(self.value))
        if missing:
            nullable = {"missing", "not_observed", "below_detection", "invalid", "fit_failed"}
            if self.role == "processed":
                nullable |= {"saturated", "out_of_range"}
            if self.quality not in nullable:
                errors.append("missing_quality")
        elif not values or not all(isinstance(v, (int, float)) and isfinite(v) for v in values):
            errors.append("finite_values")
        if self.valid_range is not None:
            lo, hi = self.valid_range
            if not isfinite(lo) or not isfinite(hi) or lo >= hi:
                errors.append("range")
            elif (
                not missing
                and self.quality in {"valid", "below_detection", "saturated"}
                and any(not lo <= cast(float, v) <= hi for v in values)
            ):
                errors.append("range_quality")
        for name, value in (
            ("detection_limit", self.detection_limit),
            ("uncertainty", self.uncertainty),
        ):
            if value is not None and (not isfinite(value) or value < 0):
                errors.append(name)
        if self.quality == "below_detection" and self.detection_limit is None:
            errors.append("detection_limit_required")
        elif (
            self.quality == "below_detection"
            and not missing
            and any(abs(cast(float, v)) > cast(float, self.detection_limit) for v in values)
        ):
            errors.append("detection_limit_quality")
        if self.role not in {"raw", "processed"}:
            errors.append("role")
        if self.acquisition_relation not in {
            "direct",
            "acquired_trace_fit",
            "legacy_parallel_readout",
            "legacy_public_estimate",
            "temperature_scale_conversion",
        }:
            errors.append("acquisition_relation")
        if isinstance(self.value, tuple):
            if self.axis is None or len(self.axis.values) != len(self.value):
                errors.append("axis_shape")
        elif self.axis is not None:
            errors.append("scalar_axis")
        if self.axis is not None:
            try:
                unit_spec(self.axis.unit)
            except ValueError:
                errors.append("axis_unit")
            axis = self.axis.values
            if (
                not axis
                or not all(isfinite(v) for v in axis)
                or (
                    len(axis) > 1
                    and not (
                        all(b > a for a, b in pairwise(axis))
                        or all(b < a for a, b in pairwise(axis))
                    )
                )
            ):
                errors.append("axis_coordinates")
        for stamp in (self.sampled_at, self.available_at):
            if not stamp.clock_id or not isfinite(stamp.time_s) or stamp.time_s < 0:
                errors.append("timestamp")
        if (
            self.sampled_at.clock_id == self.available_at.clock_id
            and self.available_at.time_s < self.sampled_at.time_s
        ):
            errors.append("timestamp_order")
        return errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        payload["derived_from"] = list(self.derived_from)
        if isinstance(self.value, tuple):
            payload["value"] = list(self.value)
        if self.axis is not None:
            payload["axis"]["values"] = list(self.axis.values)
        if self.valid_range is not None:
            payload["valid_range"] = list(self.valid_range)
        if isinstance(self.value, float) and isnan(self.value) and self.quality != "valid":
            payload["value"] = None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ObservationChannel:
        data = {key: value for key, value in payload.items() if key != "shape"}
        if data.get("axis") is not None:
            data["axis"] = ChannelAxis(**data["axis"])
        for key in ("sampled_at", "available_at"):
            if key in data:
                data[key] = ObservationTime(**data[key])
        channel = cls(**data)
        if "shape" in payload and list(channel.shape) != payload["shape"]:
            raise ValueError("Declared channel shape differs from its value")
        return channel


@dataclass(frozen=True)
class ScalarProjection:
    key: str
    channel_id: str
    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.lower) or not isfinite(self.upper) or self.lower >= self.upper:
            raise ValueError("Scalar projection requires finite increasing bounds")

    def project(self, channel: ObservationChannel) -> float | None:
        if isinstance(channel.value, tuple):
            raise ValueError("A finite curve requires an explicit scalar reduction")
        if not channel.projectable:
            return None
        # projectable excludes None; the scalar guard above excludes tuples.
        value = float(cast(float, channel.value))
        # Invalid finite contracts remain visible to validation, never clipped clean.
        return legacy_scalar_projection(value, self.lower, self.upper)


@dataclass(frozen=True, init=False)
class Observation:
    channels: tuple[ObservationChannel, ...]
    projection: tuple[ScalarProjection, ...]
    projection_version: str
    instrument_id: str | None
    cost: float
    sample_consumed_L: float
    _raw_template: dict[str, Any]
    _processed_links: tuple[tuple[str, str], ...]
    _uncertainty_links: tuple[tuple[str, str], ...]
    _mask_keys: tuple[str, ...]

    def __init__(
        self,
        values: dict[str, float | None] | None = None,
        units: dict[str, str] | None = None,
        observed_mask: dict[str, bool] | None = None,
        raw_signal: dict[str, Any] | None = None,
        processed_estimate: dict[str, float | None] | None = None,
        uncertainty: dict[str, float] | None = None,
        instrument_id: str | None = None,
        cost: float = 0.0,
        sample_consumed_L: float = 0.0,
        *,
        channels: tuple[ObservationChannel, ...] | None = None,
        projection: tuple[ScalarProjection, ...] = (),
        projection_version: str = LEGACY_PROJECTION_VERSION,
        sampled_at: ObservationTime = DEFAULT_OBSERVATION_TIME,
        available_at: ObservationTime | None = None,
        sample_id: str | None = None,
    ) -> None:
        if channels is not None and any(
            arg is not None
            for arg in (values, units, observed_mask, processed_estimate, uncertainty)
        ):
            raise ValueError("Native and legacy observation sources cannot be mixed")
        available_at = available_at or sampled_at
        readings = list(channels or ())
        links: list[tuple[str, str]] = []
        std_links: list[tuple[str, str]] = []
        mask_keys = tuple(p.key for p in projection)
        template = deepcopy(raw_signal or {})
        # The existing spectral provider's acquired trace is owned by a native
        # curve; its original packet is materialized through references below.
        raw_ids: list[str] = []
        # The older signal provider names the arrays at packet level. Move them
        # into the same native shape used by configurable spectroscopy.
        if isinstance(template.get("axis"), dict) and isinstance(template.get("raw_signal"), dict):
            axis, signal = template["axis"], template["raw_signal"]
            xkey, ykey = axis.get("key"), signal.get("key")
            if xkey in template and ykey in template and isinstance(template[ykey], list):
                coords = tuple(template[xkey])
                acquired = ObservationChannel(
                    "acquired_trace",
                    signal["unit"],
                    tuple(template[ykey]),
                    axis=ChannelAxis(xkey, axis["unit"], coords),
                    sampled_at=sampled_at,
                    available_at=available_at,
                    sample_id=sample_id,
                )
                readings.append(acquired)
                raw_ids.append(acquired.channel_id)
                template[xkey] = {"$channel_axis": acquired.channel_id}
                template[ykey] = {"$channel_value": acquired.channel_id}
                if "replicate_signals" in template:
                    for index, trace in enumerate(template["replicate_signals"]):
                        cid = f"acquired_replicate_{index}"
                        readings.append(replace(acquired, channel_id=cid, value=tuple(trace)))
                        template["replicate_signals"][index] = {"$channel_value": cid}
            elif ykey in template and isinstance(template[ykey], (int, float)):
                cid = "acquired_signal"
                readings.append(
                    ObservationChannel(
                        cid,
                        signal["unit"],
                        template[ykey],
                        sampled_at=sampled_at,
                        available_at=available_at,
                        sample_id=sample_id,
                    )
                )
                raw_ids.append(cid)
                template[ykey] = {"$channel_value": cid}
        if "axis" in template and isinstance(template.get("raw_signal"), dict):
            axis, signal = template["axis"], template["raw_signal"]
            if "values" in axis and "values" in signal:
                cid = "acquired_trace"
                readings.append(
                    ObservationChannel(
                        cid,
                        signal["unit"],
                        tuple(signal["values"]),
                        quality="saturated" if signal.get("saturated_points", 0) else "valid",
                        axis=ChannelAxis(
                            "spectral_coordinate", axis["unit"], tuple(axis["values"])
                        ),
                        uncertainty=template.get("noise", {}).get("standard_deviation"),
                        sampled_at=sampled_at,
                        available_at=available_at,
                        sample_id=sample_id,
                    )
                )
                raw_ids.append(cid)
                axis["values"] = {"$channel_axis": cid}
                signal["values"] = {"$channel_value": cid}
        if channels is None:
            values, units = values or {}, units or {}
            mask = observed_mask or {}
            uncertainty = uncertainty or {}
            processed_estimate = processed_estimate or {}
            mask_keys = tuple(mask)
            projection = tuple(ScalarProjection(key, key) for key in values)
            for key, value in values.items():
                std_key = f"{key}_std"
                is_processed = key in processed_estimate
                readings.append(
                    ObservationChannel(
                        key,
                        units.get(key, "dimensionless"),
                        value,
                        quality=(
                            "valid"
                            if mask.get(key, value is not None)
                            else "missing"
                            if value is None
                            else "not_observed"
                        ),
                        uncertainty=uncertainty.get(std_key),
                        sampled_at=sampled_at,
                        valid_range=(0.0, 1.0)
                        if units.get(key) in {"dimensionless", "risk"}
                        else None,
                        available_at=available_at,
                        sample_id=sample_id,
                        role="processed" if is_processed else "raw",
                        derived_from=(
                            tuple(raw_ids)
                            if is_processed and "values" in template.get("axis", {})
                            else ()
                        ),
                        acquisition_relation=(
                            "acquired_trace_fit"
                            if is_processed and "values" in template.get("axis", {})
                            else "legacy_parallel_readout"
                            if is_processed and raw_ids
                            else "legacy_public_estimate"
                            if is_processed
                            else "direct"
                        ),
                    )
                )
                if std_key in uncertainty:
                    std_links.append((std_key, key))
                if is_processed:
                    if processed_estimate[key] == value:
                        links.append((key, key))
                    else:
                        cid = f"processed:{key}"
                        readings.append(
                            replace(
                                readings[-1],
                                channel_id=cid,
                                value=processed_estimate[key],
                                derived_from=(key,),
                            )
                        )
                        links.append((key, cid))
            for key, std in uncertainty.items():
                if key not in dict(std_links):
                    readings.append(
                        ObservationChannel(
                            f"uncertainty:{key}",
                            "dimensionless",
                            None,
                            "not_observed",
                            uncertainty=std,
                        )
                    )
                    std_links.append((key, f"uncertainty:{key}"))
        else:
            links = [(c.channel_id, c.channel_id) for c in readings if c.role == "processed"]
            std_links = [
                (f"{c.channel_id}_std", c.channel_id) for c in readings if c.uncertainty is not None
            ]
        for name, field_value in {
            "channels": tuple(readings),
            "projection": tuple(projection),
            "projection_version": projection_version,
            "instrument_id": instrument_id,
            "cost": cost,
            "sample_consumed_L": sample_consumed_L,
            "_raw_template": template,
            "_processed_links": tuple(links),
            "_uncertainty_links": tuple(std_links),
            "_mask_keys": mask_keys,
        }.items():
            object.__setattr__(self, name, field_value)
        ids = [c.channel_id for c in self.channels]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate native observation channels")
        if any(p.channel_id not in ids or self.channel(p.channel_id).shape for p in projection):
            raise ValueError("Scalar projection must reference an existing scalar channel")

    def channel(self, channel_id: str) -> ObservationChannel:
        return next(c for c in self.channels if c.channel_id == channel_id)

    @property
    def values(self) -> dict[str, float | None]:
        return {p.key: p.project(self.channel(p.channel_id)) for p in self.projection}

    @property
    def units(self) -> dict[str, str]:
        return {p.key: self.channel(p.channel_id).unit for p in self.projection}

    @property
    def observed_mask(self) -> dict[str, bool]:
        by_key = {
            p.key: self.channel(p.channel_id).projectable
            for p in self.projection
        }
        return {key: by_key.get(key, False) for key in self._mask_keys}

    @property
    def processed_estimate(self) -> dict[str, float | tuple[float, ...] | None]:
        return {key: self.channel(cid).value for key, cid in self._processed_links}

    @property
    def uncertainty(self) -> dict[str, float]:
        # Links are created only for channels with a declared uncertainty.
        return {key: cast(float, self.channel(cid).uncertainty)
                for key, cid in self._uncertainty_links}

    @property
    def raw_signal(self) -> dict[str, Any]:
        def expand(value: object) -> object:
            if isinstance(value, dict):
                if set(value) == {"$channel_value"}:
                    reading = self.channel(value["$channel_value"]).value
                    return list(reading) if isinstance(reading, tuple) else reading
                if set(value) == {"$channel_axis"}:
                    axis = cast(ChannelAxis, self.channel(value["$channel_axis"]).axis)
                    return list(axis.values)
                if set(value) == {"$channel_uncertainty"}:
                    return self.channel(value["$channel_uncertainty"]).uncertainty
                return {key: expand(item) for key, item in value.items()}
            if isinstance(value, list):
                return [expand(item) for item in value]
            return deepcopy(value)

        return cast(dict[str, Any], expand(self._raw_template))

    def errors(self) -> list[str]:
        errors = [f"{c.channel_id}:{error}" for c in self.channels for error in c.errors()]
        ids = [c.channel_id for c in self.channels]
        if len(set(ids)) != len(ids):
            errors.append("duplicate_channel")
        if self.projection_version != LEGACY_PROJECTION_VERSION:
            errors.append("projection_version")
        if len({p.key for p in self.projection}) != len(self.projection):
            errors.append("duplicate_projection")
        for c in self.channels:
            if any(source not in ids or source == c.channel_id for source in c.derived_from):
                errors.append(f"{c.channel_id}:source")
            elif c.acquisition_relation == "acquired_trace_fit":
                if not c.derived_from:
                    errors.append(f"{c.channel_id}:fit_source")
                for source in c.derived_from:
                    raw = self.channel(source)
                    if (raw.sample_id, raw.sampled_at) != (c.sample_id, c.sampled_at):
                        errors.append(f"{c.channel_id}:source_acquisition")
                    if (
                        raw.available_at.clock_id == c.available_at.clock_id
                        and raw.available_at.time_s > c.available_at.time_s
                    ):
                        errors.append(f"{c.channel_id}:source_availability")
        remaining = {c.channel_id: set(c.derived_from) for c in self.channels}
        while remaining:
            ready = {key for key, sources in remaining.items() if not sources}
            if not ready:
                errors.append("cyclic_or_absent_source")
                break
            remaining = {
                key: sources - ready for key, sources in remaining.items() if key not in ready
            }
        for p in self.projection:
            if (
                p.channel_id not in ids
                or not isfinite(p.lower)
                or not isfinite(p.upper)
                or p.lower >= p.upper
            ):
                errors.append(f"{p.key}:projection")
            elif self.channel(p.channel_id).shape:
                errors.append(f"{p.key}:curve_projection")
        return errors

    def with_acquisition(self, acquisition: dict[str, Any]) -> Observation:
        """Bind the executed receipt once, including instantaneous instruments.

        The physical transaction (or shared-clock queue) owns timing and sample
        identity. Instrument callbacks own signals, never these execution facts.
        """
        native = self.native_record()
        timing = {
            key: deepcopy(acquisition[key])
            for key in ("sample_id", "sampled_at", "available_at")
        }
        native["channels"] = [{**channel, **timing} for channel in native["channels"]]
        native["raw_template"]["acquisition"] = deepcopy(acquisition)
        return type(self).from_native_record(native)

    def terminal_quality_errors(self, required: tuple[str, ...]) -> list[str]:
        """Detection-limit censoring is evidence; failed or absent assays are not.

        A declared negative crystal recovery can legitimately leave size/purity
        undefined; its explicit outcome remains visible alongside the assay.
        """
        negative = self.raw_signal.get("sample_outcome", {})
        undefined = (
            set(negative.get("undefined_observables", ()))
            if negative.get("status") == "negative_result" else set()
        )
        channels = {channel.channel_id: channel for channel in self.channels}
        keys = required or tuple(
            c.channel_id for c in self.channels if c.role == "processed"
        ) or tuple(channels)
        if not keys:
            return ["terminal:empty"]
        return [
            f"{key}:{channels[key].quality if key in channels else 'absent'}"
            for key in keys if key not in undefined
            and (key not in channels or channels[key].quality not in {"valid", "below_detection"})
        ]

    def native_record(self) -> dict[str, Any]:
        return {
            "schema_version": "chemworld-native-observation-v1",
            "channels": [c.to_dict() for c in self.channels],
            "projection_version": self.projection_version,
            "projection": [asdict(p) for p in self.projection],
            "raw_template": deepcopy(self._raw_template),
            "processed_links": [list(link) for link in self._processed_links],
            "uncertainty_links": [list(link) for link in self._uncertainty_links],
            "mask_keys": list(self._mask_keys),
            "instrument_id": self.instrument_id,
            "cost": self.cost,
            "sample_consumed_L": self.sample_consumed_L,
        }

    @classmethod
    def from_native_record(cls, payload: dict[str, Any]) -> Observation:
        keys = {
            "schema_version",
            "channels",
            "projection_version",
            "projection",
            "raw_template",
            "processed_links",
            "uncertainty_links",
            "mask_keys",
            "instrument_id",
            "cost",
            "sample_consumed_L",
        }
        if set(payload) != keys:
            raise ValueError("Invalid native observation fields")
        if payload.get("schema_version") != "chemworld-native-observation-v1":
            raise ValueError("Unknown native observation schema")
        obs = cls(
            channels=tuple(ObservationChannel.from_dict(c) for c in payload["channels"]),
            projection=tuple(ScalarProjection(**p) for p in payload["projection"]),
            projection_version=payload["projection_version"],
            instrument_id=payload["instrument_id"],
            cost=payload["cost"],
            sample_consumed_L=payload["sample_consumed_L"],
        )
        for field, key in (
            ("_raw_template", "raw_template"),
            ("_processed_links", "processed_links"),
            ("_uncertainty_links", "uncertainty_links"),
            ("_mask_keys", "mask_keys"),
        ):
            value = deepcopy(payload[key])
            if field != "_raw_template":
                value = tuple(tuple(v) if isinstance(v, list) else v for v in value)
            object.__setattr__(obs, field, value)
        ids = {c.channel_id for c in obs.channels}
        if any(cid not in ids for _, cid in (*obs._processed_links, *obs._uncertainty_links)):
            raise ValueError("Native observation link refers to an absent channel")
        return obs

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "units": self.units,
            "observed_mask": self.observed_mask,
            "observed_keys": [key for key, observed in self.observed_mask.items() if observed],
            "raw_signal": self.raw_signal,
            "processed_estimate": self.processed_estimate,
            "uncertainty": self.uncertainty,
            "instrument_id": self.instrument_id,
            "cost": self.cost,
            "sample_consumed_L": self.sample_consumed_L,
            "native": self.native_record(),
        }
