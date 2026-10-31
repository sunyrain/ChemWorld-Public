"""Declared formal quantity gauge; the existing detector RNG supplies noise."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from chemworld.foundation import Instrument, Observation, ObservationChannel, WorldState
from chemworld.foundation.units import COHERENT_SI_UNITS
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.instruments import InstrumentContract

if TYPE_CHECKING:
    from chemworld.world.composition import WorldCompositionSpec

FORMAL_METER = "formal_quantity_meter"
NOISE_FLOORS = {"mol": 1e-9, "L": 1e-9, "mol/L": 1e-6}


def formal_observer_factory(
    composition: WorldCompositionSpec | None,
) -> Callable[[WorldState, Mapping[str, object], np.random.Generator], Observation]:
    from chemworld.runtime.sbml_models import validate_model_state

    if composition is None or composition.physics is None:
        raise ValueError("formal gauge requires its immutable native model declaration")
    models = composition.physics.models_of("sbml")

    def read(
        state: WorldState, action: Mapping[str, object], rng: np.random.Generator,
    ) -> Observation:
        if state.vessel_id not in models:
            raise ValueError("formal gauge has no model at this workspace")
        model = models[state.vessel_id]
        _, coordinates, _ = validate_model_state(state, model)
        values = model.readout(coordinates, state.ledger.time_s)
        channels = []
        for observable in model.observables:
            value = values[observable.channel_id]
            sigma = model.readout_relative_sd*max(
                abs(value), NOISE_FLOORS.get(observable.unit, 1e-12))
            channels.append(ObservationChannel(observable.channel_id, observable.unit,
                value+float(rng.normal(0., sigma)), uncertainty=sigma))
        return Observation(channels=tuple(channels), instrument_id=FORMAL_METER,
                           cost=.01, sample_consumed_L=0.)

    return read


def formal_instrument_spec() -> InstrumentSpec:
    definition = Instrument(FORMAL_METER, "Declared formal entity quantity gauge", (), .01, 0., {})
    contract = InstrumentContract(instrument_id=FORMAL_METER, observable_keys=(),
        input_state_schema={"required_phase": "formal"}, axis_contract={"shape": "scalar"},
        raw_signal_schema={"amount_*": {"unit": "mol", "required": False},
                           "concentration_*": {"unit": "mol/L", "required": False},
                           "volume_*": {"unit": "L", "required": False},
                           "parameter_*": {"units": list(COHERENT_SI_UNITS), "required": False},
                           "reaction_rate_*": {"unit": "mol/s", "required": False}},
        processed_estimate_schema={},
        uncertainty_model="Independent signed Gaussian detector; relative SD and unit floors",
        noise_model={"unit_floors": dict(NOISE_FLOORS), "other_coherent_SI_floor": 1e-12},
        calibration_contract={"native_quantity_gain": 1.},
        detection_contract={"signed_readout": True}, saturation_contract={},
        baseline_drift_contract={}, missingness_contract={}, cost=.01, latency_s=0.,
        sample_consumption_L=0., destructive=False, requires_terminated=False,
        calibration_profile="formal-quantity-gauge-v1",
        synthetic_boundary="Formal gauge; chemical spectrum, composition and heat are undeclared",
        clock_semantics="instantaneous_snapshot", analysis_time_s=0.)
    return InstrumentSpec(definition, contract, observer_factory=formal_observer_factory)
