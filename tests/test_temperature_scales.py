"""Native temperature calibration contracts and unchanged original observations."""

import json
import math

import pytest

from chemworld.data.scientific_inputs import InputValue
from chemworld.foundation.observations import Observation, ObservationChannel, ObservationTime
from chemworld.foundation.units import Quantity
from chemworld.physchem.temperature_scales import (
    its90_68_difference,
    temperature_convert,
    temperature_input,
    temperature_observation,
    temperature_quantity,
    temperature_scale_bounds,
    temperature_scale_conversion,
)


@pytest.mark.parametrize("scale", ["ITS-68", "ITS-48", "ITS-76", "ITS-27"])
def test_physical_endpoint_images_and_reversible_conversion(scale):
    for t in temperature_scale_bounds(scale):
        converted = temperature_convert(t, scale, "ITS-90")
        assert temperature_convert(converted, "ITS-90", scale) == t
    lower, upper = temperature_scale_bounds(scale, basis="ITS-90")
    for t in (lower - 1e-5, upper + 1e-5):
        with pytest.raises(ValueError, match="outside"):
            temperature_convert(t, "ITS-90", scale)
    assert temperature_convert(500, "IPTS-68", "IPTS-48") == pytest.approx(499.9470, abs=1e-4)


@pytest.mark.parametrize("t", [-1, math.inf, math.nan, True])
def test_invalid_temperature_is_rejected(t):
    with pytest.raises(ValueError):
        temperature_convert(t, "ITS-90", "ITS-90")
    with pytest.raises(ValueError):
        its90_68_difference(t)


def test_scale_and_units_are_explicit_and_independent():
    q = Quantity(226.85, "degC")
    assert q.to("K").value == 500
    v, d = temperature_scale_conversion(500, "ITS-68", "ITS-90")
    assert temperature_quantity(q, "ITS-68", "ITS-90").value == v
    assert q.value == 226.85
    x = InputValue(q.value, q.unit, 0.02, "Original thermometer standard uncertainty")
    converted = temperature_input(x, "ITS-68", "ITS-90")
    assert converted.value == v and converted.uncertainty == pytest.approx(0.02 * d)
    assert json.loads(converted.uncertainty_note.split("Original input: ")[1]) == x.to_dict()
    assert InputValue.from_dict(converted.to_dict()) == converted
    unknown = temperature_input(InputValue(500, "K", None, "Unknown"), "ITS-68", "ITS-90")
    assert unknown.uncertainty is None
    with pytest.raises(ValueError):
        temperature_quantity(Quantity(5, "s"), "ITS-68", "ITS-90")
    with pytest.raises(ValueError, match="unknown"):
        temperature_convert(500, "K", "ITS-90")


@pytest.mark.parametrize("quality", ["missing", "invalid"])
def test_missing_input_is_not_invented(quality):
    value = temperature_input(
        InputValue(None, "K", None, "Not acquired", quality), "ITS-68", "ITS-90"
    )
    assert value.value is None and value.quality == quality and value.uncertainty is None


def test_scalar_observation_preserves_raw_sample_times_and_provenance():
    raw = ObservationChannel(
        "thermometer",
        "degC",
        26.85,
        uncertainty=0.02,
        sample_id="vial-1",
        valid_range=(20, 50),
        sampled_at=ObservationTime("world", 4),
        available_at=ObservationTime("world", 7),
    )
    before = raw.to_dict()
    observation = temperature_observation(raw, "IPTS-68", "ITS-90")
    converted = observation.channel("converted_temperature")
    assert not observation.errors()
    assert raw.to_dict() == before and observation.channel("thermometer") == raw
    assert converted.sampled_at == raw.sampled_at and converted.available_at == raw.available_at
    assert converted.sample_id == raw.sample_id and converted.derived_from == ("thermometer",)
    assert converted.value == temperature_convert(300, "ITS-68", "ITS-90")
    restored = Observation.from_native_record(
        json.loads(json.dumps(observation.native_record(), allow_nan=False))
    )
    assert restored.to_dict() == observation.to_dict()
    assert restored.raw_signal["temperature_conversion"]["source_value"] == raw.value
    assert restored.raw_signal["temperature_conversion"]["source_scale"] == "ITS-68"


@pytest.mark.parametrize("quality", ["missing", "not_observed", "invalid", "fit_failed"])
def test_nullable_observation_keeps_quality(quality):
    raw = ObservationChannel("old", "K", None, quality=quality)
    result = temperature_observation(raw, "ITS-68", "ITS-90")
    assert result.channel("converted_temperature").quality == quality
    assert result.channel("converted_temperature").value is None


def test_invalid_numeric_observations_are_not_silently_certified():
    with pytest.raises(ValueError, match="only valid numeric"):
        temperature_observation(
            ObservationChannel("x", "K", 300, quality="saturated"), "ITS-68", "ITS-90"
        )
    with pytest.raises(ValueError, match="distinct"):
        temperature_observation(
            ObservationChannel("x", "K", 300), "ITS-68", "ITS-90", channel_id="x"
        )
    with pytest.raises(ValueError, match="invalid numeric"):
        temperature_input(InputValue(300, "K", 0.1, "Invalid", "invalid"), "ITS-68", "ITS-90")


def test_public_difference_examples_and_undocumented_overlap():
    assert its90_68_difference(1000) == pytest.approx(0.01231818956580355, abs=2e-11)
    assert its90_68_difference(70) == pytest.approx(0.006819, abs=5e-7)
    assert its90_68_difference(140) == pytest.approx(0.013228, abs=5e-7)
    assert its90_68_difference(510) == pytest.approx(-0.040455, abs=5e-7)
    assert its90_68_difference(1090) == pytest.approx(0.044543, abs=5e-7)
    # Harvey's fifth check uses a different, unrounded radiation coefficient.
    assert its90_68_difference(77) > 0
    assert its90_68_difference(77, convention="chemicals_1_5_2") == 0
    assert its90_68_difference(0) == 0
