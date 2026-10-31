"""A calibrated mol gauge, independent of the reaction's process random stream."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from chemworld.foundation import Instrument, Observation, ObservationChannel, WorldState
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.instruments import InstrumentContract

if TYPE_CHECKING:
    from chemworld.world.composition import WorldCompositionSpec

SDE_METER = "sde_amount_meter"
SDE_METER_SD_MOL = 1.0 / AVOGADRO
SDE_ALIQUOT_METER = "sde_aliquot_meter"
SDE_ALIQUOT_VOLUME_L = 1e-16


def sde_observer_factory(
    composition: WorldCompositionSpec | None,
) -> Callable[[WorldState, Mapping[str, object], np.random.Generator], Observation]:
    from chemworld.runtime.stochastic_adaptive import AdaptiveHybridStochasticRuntime
    from chemworld.runtime.stochastic_sde import SDEStochasticRuntime

    if composition is None or composition.physics is None:
        raise ValueError("SDE gauge requires its actual physical model declaration")
    runtime = SDEStochasticRuntime(composition)
    adaptive = AdaptiveHybridStochasticRuntime(composition)

    def read(
        state: WorldState, action: Mapping[str, object], rng: np.random.Generator,
    ) -> Observation:
        (adaptive if adaptive.supports(state) else runtime).validate(state)
        aliquot = action["instrument"] == SDE_ALIQUOT_METER
        fraction = SDE_ALIQUOT_VOLUME_L / state.volume_L if aliquot else 1.0
        prefix = "sample_amount_" if aliquot else "amount_"
        channels = tuple(
            ObservationChannel(
                prefix + key,
                "mol",
                value * fraction + float(rng.normal(0, SDE_METER_SD_MOL)),
                uncertainty=SDE_METER_SD_MOL,
            )
            for key, value in sorted(state.species_amounts.items())
        )
        return Observation(
            channels=channels,
            instrument_id=SDE_ALIQUOT_METER if aliquot else SDE_METER,
            cost=0.01,
            sample_consumed_L=SDE_ALIQUOT_VOLUME_L if aliquot else 0.0,
        )

    return read


def sde_instrument_spec() -> InstrumentSpec:
    definition = Instrument(
        SDE_METER, "Synthetic continuous reaction amount gauge", (), 0.01, 0.0, {}
    )
    contract = InstrumentContract(
        instrument_id=SDE_METER,
        observable_keys=(),
        input_state_schema={"required_phase": "liquid"},
        axis_contract={"shape": "scalar"},
        raw_signal_schema={"amount_*": {"unit": "mol", "required": False}},
        processed_estimate_schema={},
        uncertainty_model="independent signed Gaussian detector",
        noise_model={"sd_mol": SDE_METER_SD_MOL},
        calibration_contract={"mol_gain": 1.0},
        detection_contract={"signed_readout": True},
        saturation_contract={},
        baseline_drift_contract={},
        missingness_contract={},
        cost=0.01,
        latency_s=0.0,
        sample_consumption_L=0.0,
        destructive=False,
        requires_terminated=False,
        calibration_profile="continuous-sde-amount-gauge-v1",
        synthetic_boundary=(
            "Declared continuous mol gauge with one-particle-equivalent SD; no integer count truth"
        ),
        clock_semantics="instantaneous_snapshot",
        analysis_time_s=0.0,
    )
    return InstrumentSpec(definition, contract, observer_factory=sde_observer_factory)


def sde_aliquot_instrument_spec() -> InstrumentSpec:
    from dataclasses import replace

    original = sde_instrument_spec()
    definition = replace(
        original.definition,
        id=SDE_ALIQUOT_METER,
        name="Synthetic representative liquid aliquot amount gauge",
        sample_volume_L=SDE_ALIQUOT_VOLUME_L,
    )
    contract = replace(
        original.contract,
        instrument_id=SDE_ALIQUOT_METER,
        raw_signal_schema={"sample_amount_*": {"unit": "mol", "required": False}},
        sample_consumption_L=SDE_ALIQUOT_VOLUME_L,
        destructive=True,
        clock_semantics="offline_snapshot",
        analysis_time_s=1.0,
        calibration_profile="continuous-sde-aliquot-gauge-v1",
        synthetic_boundary=(
            "Deterministic representative 1e-16 L liquid withdrawal; "
            "independent signed detector, no stochastic capture or pipetting error"
        ),
    )
    return InstrumentSpec(definition, contract, observer_factory=sde_observer_factory)
