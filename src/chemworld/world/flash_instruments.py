"""Finite synthetic phase-inventory sensor using the normal acquisition clock."""

from chemworld.foundation import Instrument, Observation
from chemworld.foundation.observations import ObservationChannel
from chemworld.world.domain_modules import InstrumentSpec
from chemworld.world.instruments import InstrumentContract


def read_phase_state(state, action, rng):
    # A column contains separate physical compartments; this local probe
    # observes the declared reboiler rather than counting trays as LLE phases.
    phases = list(state.phases.phases.values())
    if any("column_cell" in p.metadata for p in phases):
        phases = [p for p in phases if p.metadata.get("column_cell") == 0]
    if any("concentration_cell" in p.metadata for p in phases):
        phases = [p for p in phases if p.metadata.get("concentration_cell") == 0]
    total = sum(sum(p.species_amounts_mol.values()) for p in phases)
    vapor = sum(sum(p.species_amounts_mol.values()) for p in phases
                if p.phase_type == "gas")
    liquids = [sum(p.species_amounts_mol.values())
               for p in sorted(phases,key=lambda p:p.phase_id)
               if p.phase_type == "liquid" and sum(p.species_amounts_mol.values()) > 0]
    return Observation(channels=(
        ObservationChannel("temperature", "K", state.temperature_K, uncertainty=0.0),
        ObservationChannel("vapor_fraction", "dimensionless", vapor / total if total else None,
                           quality="valid" if total else "missing", uncertainty=0.0),
        ObservationChannel("liquid_phase_count", "dimensionless", float(len(liquids)),
                           uncertainty=0.0),
        ObservationChannel("second_liquid_fraction", "dimensionless",
                           liquids[1] / sum(liquids) if len(liquids) == 2 else 0.0,
                           uncertainty=0.0),
    ), instrument_id="phase_state", cost=.02, sample_consumed_L=0.0)


def flash_instrument_specs():
    definition = Instrument("phase_state", "Synthetic equilibrium phase probe",
                            ("temperature", "vapor_fraction", "liquid_phase_count",
                             "second_liquid_fraction"), .02, 0,
                            {"temperature": 0.0, "vapor_fraction": 0.0,
                             "liquid_phase_count": 0.0, "second_liquid_fraction": 0.0})
    contract = InstrumentContract(
        instrument_id=definition.id, observable_keys=definition.observable_keys,
        input_state_schema={"sample_consumption_L": 0.0}, axis_contract={"shape": "scalar"},
        raw_signal_schema={"temperature": {"unit": "K"},
                           "vapor_fraction": {"unit": "dimensionless", "range": [0, 1]},
                           "liquid_phase_count": {"unit": "dimensionless", "range": [0, 2]},
                           "second_liquid_fraction": {"unit": "dimensionless", "range": [0, 1]}},
        processed_estimate_schema={}, uncertainty_model="explicit noiseless synthetic fixture",
        noise_model=definition.noise_std, calibration_contract={"gain": 1, "offset": 0},
        detection_contract={}, saturation_contract={}, baseline_drift_contract={"model": "none"},
        missingness_contract={"empty_material": "phase fraction unavailable"}, cost=.02,
        latency_s=1, sample_consumption_L=0, destructive=False, requires_terminated=False,
        calibration_profile="synthetic-phase-probe-v1",
        synthetic_boundary="Exact model readout; not experimental phase identification",
        clock_semantics="in_situ_end", analysis_time_s=0)
    return (InstrumentSpec(definition, contract, native_observer=read_phase_state),)
