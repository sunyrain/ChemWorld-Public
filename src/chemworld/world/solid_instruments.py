"""Synthetic committed-solid phase and temperature measurements."""

from dataclasses import replace

from chemworld.foundation import Instrument, Observation, ObservationChannel
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.transport_instruments import transport_instrument_spec


def read_solid(state, action, rng):
    domains = sorted(
        (
            p.species_amounts_mol["A"] / sum(p.species_amounts_mol.values()),
            sum(p.species_amounts_mol.values()),
        )
        for p in state.phases.phases.values()
    )
    total = sum(n for _, n in domains)
    values = {
        "temperature": ("K", state.temperature_K),
        "solid_domain_count": ("dimensionless", len(domains)),
        "low_A_fraction": ("dimensionless", domains[0][0] if domains else None),
        "high_A_fraction": ("dimensionless", domains[-1][0] if domains else None),
        "low_domain_fraction": ("dimensionless", domains[0][1] / total if total else None),
    }
    return Observation(
        channels=tuple(
            ObservationChannel(
                key, unit, value, uncertainty=0, quality="missing" if value is None else "valid"
            )
            for key, (unit, value) in values.items()
        ),
        instrument_id="solid_phase_meter",
        cost=0.01,
        sample_consumed_L=0,
    )


def solid_instrument_spec():
    base = transport_instrument_spec()
    definition = Instrument(
        "solid_phase_meter", "Synthetic binary phase and temperature gauge", (), 0.01, 0, {}
    )
    contract = replace(
        base.contract,
        instrument_id=definition.id,
        raw_signal_schema={"temperature": {"unit": "K"}},
        calibration_profile="binary-solid-gauge-v1",
        synthetic_boundary="committed solid phases; no crystallographic spectrum",
    )
    return InstrumentSpec(definition, contract, native_observer=read_solid)
