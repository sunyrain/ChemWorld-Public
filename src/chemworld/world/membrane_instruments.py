"""Public pressure, volume and salinity measurements for dilute membranes."""

from dataclasses import replace

from chemworld.foundation import Instrument, Observation, ObservationChannel
from chemworld.physchem.membranes import (
    FOULANT_MOLAR_MASS_KG,
    film_mass,
    membrane_operating_point,
    rinse_operating_point,
    salt_concentration,
)
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.transport_instruments import transport_instrument_spec


def read_membrane(state, action, rng):
    from chemworld.runtime.membranes import unpack_membrane

    record, spec, inventory = unpack_membrane(state)
    flow = record.settings.get("flow_control", {}).get("requested_flow_m3_s", 0.0)
    path = record.settings.get("flow_control", {}).get("flow_path", "process")
    collector = state.inactive_vessels[spec.collector_vessel_id]
    if path == "process" and (
        state.volume_L <= spec.pickup_volume_L + 1e-12
        or collector.volume_L >= spec.collector_capacity_L - 1e-12
    ):
        flow = 0.0
    point = membrane_operating_point(
        spec,
        inventory.amounts_mol[0],
        flow if path == "process" else 0.0,
        film_mass(spec, inventory),
    )
    # At the supported concentration boundary the process is stopped.
    if point["interface_concentration_mol_m3"] >= spec.concentration_limit_mol_m3 * (1 - 1e-10):
        point = membrane_operating_point(
            spec, inventory.amounts_mol[0], 0.0, film_mass(spec, inventory)
        )
    total_volume = state.volume_L + collector.volume_L
    values = {
        "water_flux": ("m/s", point["water_flux_m_s"]),
        "membrane_pressure": ("Pa", point["membrane_pressure_Pa"]),
        "shaft_power": ("W", point["shaft_power_W"]),
        "feed_salinity": ("mol/m^3", salt_concentration(inventory.amounts_mol[0])),
        "permeate_outlet_salinity": ("mol/m^3", point["permeate_concentration_mol_m3"]),
        "collected_salinity": ("mol/m^3", salt_concentration(inventory.amounts_mol[1])),
        "collected_volume": ("L", collector.volume_L),
        "collected_fraction": (
            "dimensionless",
            collector.volume_L / total_volume if total_volume else 0.0,
        ),
    }
    if spec.fouling:
        wash = state.inactive_vessels[spec.fouling.wash_vessel_id]
        waste = state.inactive_vessels[spec.fouling.waste_vessel_id]
        rinse_flow = 0.0
        if (
            path == "rinse"
            and wash.volume_L > spec.pickup_volume_L + 1e-12
            and (waste.volume_L < spec.fouling.waste_capacity_L - 1e-12)
        ):
            rinse = rinse_operating_point(spec, flow)
            rinse_flow = rinse["actual_flow_m3_s"]
            values["shaft_power"] = ("W", rinse["shaft_power_W"])
        values.update(
            {
                "wash_volume": ("L", wash.volume_L),
                "waste_volume": ("L", waste.volume_L),
                "rinse_flow": ("mL/min", rinse_flow * 6e7),
                "waste_foulant_mass": (
                    "g",
                    waste.species_amounts["foulant"] * FOULANT_MOLAR_MASS_KG * 1000,
                ),
            }
        )
    return Observation(
        channels=tuple(ObservationChannel(k, u, v, uncertainty=0) for k, (u, v) in values.items()),
        instrument_id="membrane_meter",
        cost=0.01,
        sample_consumed_L=0,
    )


def membrane_instrument_spec():
    gauge = transport_instrument_spec()
    definition = Instrument("membrane_meter", "Synthetic dilute membrane gauges", (), 0.01, 0, {})
    contract = replace(
        gauge.contract,
        instrument_id=definition.id,
        raw_signal_schema={"water_flux": {"unit": "m/s"}, "feed_salinity": {"unit": "mol/m^3"}},
        calibration_profile="dilute-ro-gauge-v1",
        synthetic_boundary="permeate flux, pressure, actual bulk and outlet salinity/volumes",
    )
    return InstrumentSpec(definition, contract, native_observer=read_membrane)
