"""Synthetic gas sensors using native units and the shared measurement lifecycle."""

from functools import partial, update_wrapper

from chemworld.foundation import Instrument, Observation
from chemworld.foundation.observations import ObservationChannel
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.instruments import InstrumentContract


def _read_gas(state, action, rng, *, definition, quantities):
    channels = []
    total = sum(state.species_amounts.values())
    for name, unit, upper in quantities:
        if name == "pressure":
            truth = state.pressure_Pa
        elif name == "CO2_amount":
            truth = state.species_amounts.get("CO2", 0.0)
        else:
            truth = (
                state.species_amounts.get(name.removesuffix("_fraction"), 0.0) / total
                if total
                else None
            )
        std = definition.noise_std[name]
        value = None if truth is None else float(truth + rng.normal(0.0, std))
        quality = "missing" if value is None else "valid"
        if value is not None and value < 0:
            value, quality = 0.0, "below_detection"
        elif upper is not None and value is not None and value > upper:
            value, quality = upper, "saturated"
        channels.append(
            ObservationChannel(
                name,
                unit,
                value,
                quality=quality,
                valid_range=None if upper is None else (0.0, upper),
                detection_limit=0.0,
                uncertainty=std,
            )
        )
    return Observation(
        channels=tuple(channels),
        instrument_id=definition.id,
        cost=definition.cost,
        sample_consumed_L=definition.sample_volume_L,
    )


def gas_instrument_specs():
    """Finite development sensors, not experimentally qualified spectroscopy."""
    fractions = tuple(
        (f"{species}_fraction", "dimensionless", 1.0) for species in ("CO", "O2", "CO2")
    )
    result = []
    for name, quantities, cost, acquisition_s, analysis_s, terminal, std in (
        ("gas_pressure", (("pressure", "Pa", 500000.0),), 0.02, 1.0, 0.0, False, 50.0),
        ("gas_composition", fractions, 0.08, 5.0, 0.0, False, 0.001),
        (
            "gas_final_assay",
            (*fractions, ("CO2_amount", "mol", None)),
            0.10,
            0.0,
            5.0,
            True,
            0.0002,
        ),
    ):
        noise = {key: 1e-8 if key == "CO2_amount" else std for key, _, _ in quantities}
        definition = Instrument(
            name, name.replace("_", " "), tuple(noise), cost, 0.0, noise, terminal
        )
        contract = InstrumentContract(
            instrument_id=name,
            observable_keys=definition.observable_keys,
            input_state_schema={"required_phase": "gas", "sample_consumption_L": 0.0},
            axis_contract={"shape": "scalar"},
            raw_signal_schema={
                key: {"unit": unit, "range": [0, upper]} for key, unit, upper in quantities
            },
            processed_estimate_schema={},
            uncertainty_model="independent Gaussian readout noise",
            noise_model=noise,
            calibration_contract={"gain": 1.0, "offset": 0.0},
            detection_contract={"lower_bound": 0.0},
            saturation_contract={"policy": "flag saturated endpoint; no exact-value claim"},
            baseline_drift_contract={"model": "none"},
            missingness_contract={"empty_gas": "composition unavailable"},
            cost=cost,
            latency_s=acquisition_s,
            sample_consumption_L=0.0,
            destructive=False,
            requires_terminated=terminal,
            calibration_profile="synthetic-gas-readout-v1",
            synthetic_boundary="No spectral or molecular mechanism identification implied",
            clock_semantics="offline_snapshot" if terminal else "in_situ_end",
            analysis_time_s=analysis_s,
        )
        observer = update_wrapper(
            partial(_read_gas, definition=definition, quantities=quantities), _read_gas
        )
        result.append(
            InstrumentSpec(
                definition,
                contract,
                native_observer=observer,
            )
        )
    return tuple(result)
