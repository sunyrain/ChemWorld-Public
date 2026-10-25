"""Lightweight declared-site coverage observation using the installed original state."""

from dataclasses import replace
from typing import Any

from chemworld.foundation import Observation
from chemworld.foundation.observations import ObservationChannel
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.electrode_instruments import electrical_instrument_spec


def read_chemostatted_coverage(state: Any, action: Any, rng: Any) -> Observation:
    from chemworld.runtime.chemostatted_catalysis import read_chemostatted_electrical

    reading = read_chemostatted_electrical(state)
    return Observation(
        channels=tuple(
            ObservationChannel(
                "coverage_" + key, "dimensionless", value, uncertainty=0, valid_range=(0, 1)
            )
            for key, value in sorted(reading.coverages.items())
        ),
        instrument_id="chemostatted_coverage",
        cost=0.01,
        sample_consumed_L=0,
    )


def chemostatted_coverage_instrument_spec() -> InstrumentSpec:
    base = electrical_instrument_spec("chemostatted_coverage", read_chemostatted_coverage)
    return replace(
        base,
        definition=replace(
            base.definition,
            name="Declared occupied-site coverage",
            observable_keys=(),
            noise_std={},
        ),
        contract=replace(
            base.contract,
            observable_keys=(),
            axis_contract={"shape": "scalar-per-adsorbate"},
            raw_signal_schema={"coverage_*": {"unit": "dimensionless"}},
            uncertainty_model="noiseless synthetic occupied-site fractions",
            noise_model={"coverage_std": 0.0},
            saturation_contract={},
            calibration_profile="declared-site-coverage-v1",
            synthetic_boundary=(
                "Synthetic noiseless surface coverage; not a calibrated experimental detector"
            ),
        ),
    )
