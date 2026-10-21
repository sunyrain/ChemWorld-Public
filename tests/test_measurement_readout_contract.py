"""Execution receipts, failed inversions and instrument declaration boundaries."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.foundation import ChannelAxis, Observation, ObservationChannel, ObservationTime
from chemworld.physchem.configurable_spectra import fit_packet
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.chromatography_instruments import fit_public_trace
from chemworld.world.surface_world import surface_world_spec


def liquid_session(noise="sequential"):
    runtime = ChemWorldRuntime(task_id="reaction-to-assay", observation_noise_mode=noise)
    runtime.reset(seed=5)
    for action in (
        {"operation": "add_solvent", "volume_L": .02, "solvent": 0},
        {"operation": "add_reagent", "amount_mol": .01},
    ):
        assert runtime.execute(action).info["transaction_status"] == "committed"
    return runtime


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_instantaneous_sampling_receipt_owns_identity_even_with_retained_previous_process(noise):
    runtime = liquid_session(noise)
    # This mode intentionally gives the observer the previous process/device
    # view. It must still use the executed receipt, never that old use counter.
    runtime.full_process_contract_id = "phase-resolved-process-v5"
    records = []
    for _ in range(2):
        result = runtime.execute({"operation": "measure", "instrument": "hplc"})
        assert result.info["measurement_status"] == "valid"
        receipt = result.info["outcome_diagnostics"]["acquisition"]
        assert result.observation.raw_signal["acquisition"] == receipt
        assert all(c.sample_id == receipt["sample_id"] for c in result.observation.channels)
        assert all(c.sampled_at.time_s == receipt["sample_time_s"]
                   for c in result.observation.channels)
        records.append(result.observation)
    assert records[0].channels[0].sample_id != records[1].channels[0].sample_id
    frozen = records[0].native_record()
    runtime.observe()
    assert records[0].native_record() == frozen


@pytest.mark.parametrize(
    "quality", ["missing", "saturated", "out_of_range", "fit_failed", "invalid"],
)
@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_terminal_quality_failure_keeps_destructive_sample_and_public_failure(
    monkeypatch, quality, noise,
):
    runtime = liquid_session(noise)
    runtime.execute({"operation": "terminate"})
    instrument = runtime.constitution.instruments["final_assay"]
    allowed = runtime.observation_kernel.observation_contract.observable_keys_for_instrument(
        "final_assay", instrument.observable_keys,
    )
    failed_key = allowed[0]

    def read(state, action, rng):
        rng.normal()
        return Observation(
            channels=tuple(
                ObservationChannel(
                    key, "dimensionless",
                    None if key == failed_key and quality in {"missing", "fit_failed", "invalid"}
                    else .5,
                    quality=quality if key == failed_key else "valid",
                ) for key in allowed
            ),
            instrument_id="final_assay", cost=instrument.cost,
            sample_consumed_L=instrument.sample_volume_L,
        )

    monkeypatch.setattr(runtime.observation_kernel, "observe", read)
    before = runtime._state
    result = runtime.execute({"operation": "measure", "instrument": "final_assay"})
    assert result.info["transaction_status"] == "committed"
    assert result.info["measurement_status"] == "invalid"
    assert result.info["outcome_diagnostics"]["observation_failure"] == (
        "terminal_readout_quality_failed"
    )
    assert not result.terminated and result.reward == 0
    assert result.observation.channel(failed_key).quality == quality
    assert result.observation.channel(failed_key).sample_id
    assert runtime._state.volume_L == pytest.approx(before.volume_L - instrument.sample_volume_L)
    assert runtime._state.ledger.cost == pytest.approx(before.ledger.cost + instrument.cost)
    assert runtime._state.process.last_observation_record == before.process.last_observation_record


@pytest.mark.parametrize("stage", ["nnls", "covariance"])
def test_spectral_failed_fit_keeps_trace_and_single_native_estimate_owner(monkeypatch, stage):
    runtime = liquid_session()
    before = runtime._state

    def no_fit(*args, **kwargs):
        raise np.linalg.LinAlgError("controlled inversion nonconvergence")

    with monkeypatch.context() as patch:
        patch.setattr(
            "chemworld.physchem.configurable_spectra."
            + ("nnls" if stage == "nnls" else "np.linalg.pinv"), no_fit,
        )
        result = runtime.execute({"operation": "measure", "instrument": "nmr"})
        packet = result.observation.raw_signal
        assert packet["processed_estimates"] == fit_packet(packet)
    assert result.info["transaction_status"] == "committed"
    assert runtime._state.volume_L < before.volume_L
    obs = result.observation
    assert obs.channel("acquired_trace").quality == "valid"
    for reporter in ("reactant_reporter", "target_reporter", "impurity_reporter"):
        channel = obs.channel(f"concentration:{reporter}")
        assert channel.quality == "fit_failed" and channel.value is None
        assert channel.derived_from == ("acquired_trace",)
    assert not obs.errors()
    assert Observation.from_native_record(json.loads(json.dumps(obs.native_record()))).channels == (
        obs.channels
    )
    # All public fit values remain projections of the native channels.
    record = obs.native_record()
    cid = "concentration:reactant_reporter"
    for channel in record["channels"]:
        if channel["channel_id"] == cid:
            channel.update(value=.25, quality="valid", uncertainty=.01)
    changed = Observation.from_native_record(record)
    changed_fit = changed.raw_signal["processed_estimates"]
    assert changed_fit["concentration_mol_L"]["reactant_reporter"] == .25
    assert changed_fit["standard_error_mol_L"]["reactant_reporter"] == .01
    assert obs.channel(cid).value is None


@pytest.mark.parametrize("defect", ["unit", "role", "shape", "missing"])
def test_declared_native_pressure_contract_rejects_wrong_channel(defect):
    contract = surface_world_spec().modules.instrument_specs()["gas_pressure"].contract
    channel = ObservationChannel("pressure", "Pa", 100000)
    if defect == "unit":
        channel = replace(channel, unit="V")
    elif defect == "role":
        channel = replace(channel, role="processed")
    elif defect == "shape":
        channel = replace(channel, value=(100000, 110000),
                          axis=ChannelAxis("t", "s", (0, 1)))
    observation = Observation(
        channels=() if defect == "missing" else (channel,),
        instrument_id="gas_pressure", cost=contract.cost,
    )
    with pytest.raises(ValueError):
        contract.validate_native_readout(observation)


@pytest.mark.parametrize("defect", ["identity", "time", "availability", "absent_source"])
def test_fit_cannot_derive_from_another_acquisition_or_unavailable_signal(defect):
    raw = ObservationChannel("trace", "V", 4.2, sample_id="first",
                             sampled_at=ObservationTime(time_s=1),
                             available_at=ObservationTime(time_s=2))
    fit = replace(raw, channel_id="fit", role="processed", derived_from=("trace",),
                  acquisition_relation="acquired_trace_fit")
    if defect == "identity":
        fit = replace(fit, sample_id="second")
    elif defect == "time":
        fit = replace(fit, sampled_at=ObservationTime(time_s=0))
    elif defect == "availability":
        fit = replace(fit, available_at=ObservationTime(time_s=1))
    else:
        fit = replace(fit, derived_from=())
    assert Observation(channels=(raw, fit)).errors()


@pytest.mark.parametrize("quality", ["fit_failed", "saturated", "out_of_range", "below_detection"])
def test_unavailable_processed_quality_roundtrip_is_not_a_zero_readout(quality):
    channel = ObservationChannel("fit", "mol/L", None, quality=quality, role="processed",
                                 detection_limit=.001 if quality == "below_detection" else None)
    assert not channel.errors()
    assert not channel.projectable
    restored = ObservationChannel.from_dict(json.loads(json.dumps(channel.to_dict())))
    assert restored == channel


@pytest.mark.parametrize("slope", [0, -1, float("nan")])
def test_public_chromatogram_rejects_invalid_calibration_without_access_to_state(slope):
    axis = np.arange(6, dtype=float)
    calibration = [{"slope": 1, "intercept": 0} for _ in range(3)]
    calibration[1]["slope"] = slope
    with pytest.raises(ValueError, match="calibration"):
        fit_public_trace(axis, np.ones(6), np.ones((6, 3)), [True] * 3, calibration)


def test_terminal_censored_limit_and_explicit_negative_recovery_are_distinct_from_failure():
    assert Observation().terminal_quality_errors(()) == ["terminal:empty"]
    valid = Observation(channels=(
        ObservationChannel("amount", "mol", None, "below_detection", detection_limit=.001),
    ))
    assert not valid.terminal_quality_errors(("amount",))
    assert valid.terminal_quality_errors(("absent",))
    negative = Observation(
        channels=(ObservationChannel("crystal_size", "dimensionless", None, "missing"),),
        raw_signal={"sample_outcome": {"status": "negative_result",
                                      "undefined_observables": ["crystal_size"]}},
    )
    assert not negative.terminal_quality_errors(("crystal_size",))


@pytest.mark.parametrize("temperature", [298.15, 340.0])
def test_aqueous_ph_uses_returned_potential_and_temperature_calibration(monkeypatch, temperature):
    from chemworld.physchem.equilibrium_chemistry import solve_monoprotic_acid_base
    from chemworld.world.aqueous_instruments import read_aqueous_ph

    result = solve_monoprotic_acid_base(
        acid_total_mol=.001, volume_L=.01, pka=4.76, temperature_K=temperature,
    )
    monkeypatch.setattr(
        "chemworld.runtime.aqueous_mineral.committed_speciation", lambda state: result,
    )
    reading = read_aqueous_ph(None, {}, np.random.default_rng(1))
    calibration = reading.raw_signal["processing_calibration"]
    electrode = reading.channel("electrode_potential")
    ph = reading.channel("pH")
    assert calibration["reference_pH"] - electrode.value / calibration["slope_mV_pH"] == ph.value
    assert electrode.uncertainty == ph.uncertainty * calibration["slope_mV_pH"]
    assert ph.derived_from == ("electrode_potential",)
    assert not reading.errors()


@pytest.mark.parametrize("queued", [False, True])
def test_native_declaration_mismatch_keeps_actual_acquisition_and_time(queued):
    from chemworld.world.domain_modules import DomainModuleRegistry
    from chemworld.world.lab_world import shared_lab_world

    spec = shared_lab_world() if queued else surface_world_spec(include_surface=False)
    name = "programmed_hplc" if queued else "gas_pressure"
    original = spec.modules.instrument_specs()[name].native_observer

    def wrong_unit(state, action, rng):
        observation = original(state, action, rng)
        native = observation.native_record()
        native["channels"][0]["unit"] = "V"  # valid unit, wrong declared physical quantity
        return Observation.from_native_record(native)

    modules = DomainModuleRegistry(tuple(
        replace(module, instruments=tuple(
            replace(item, native_observer=wrong_unit) if item.definition.id == name else item
            for item in module.instruments
        )) for module in spec.modules.modules
    ))
    runtime = ChemWorldRuntime(composition=replace(spec, modules=modules))
    runtime.reset(seed=17)
    before = runtime._state
    if queued:
        runtime.execute({"operation": "queue_lab_measure"})
        result = runtime.execute({"operation": "wait", "duration_s": 60})
        job = result.info["lab_clock"]["completed"][0]
        assert job["measurement_status"] == "invalid"
        assert "observation" not in job
        assert runtime._state.ledger.analysis_time_s == 60
        assert runtime._state.ledger.sample_consumed_L == pytest.approx(.0002)
    else:
        result = runtime.execute({"operation": "measure", "instrument": name})
        assert result.info["measurement_status"] == "invalid"
        assert runtime._state.ledger.time_s == 1
    assert result.info["transaction_status"] == "committed"
    assert runtime._state.ledger.cost > before.ledger.cost
    assert runtime._state.process.last_observation_record == before.process.last_observation_record


def test_fitted_concentration_uncertainty_is_the_same_in_all_public_views():
    runtime = liquid_session()
    result = runtime.execute({"operation": "measure", "instrument": "nmr"})
    observation = result.observation
    assert result.info["measurement_status"] == "valid"
    packet = observation.raw_signal
    assert packet["processed_estimates"] == fit_packet(packet)
    for reporter in packet["processed_estimates"]["concentration_mol_L"]:
        cid = f"concentration:{reporter}"
        assert observation.uncertainty[f"{cid}_std"] == observation.channel(cid).uncertainty
        assert observation.uncertainty[f"{cid}_std"] == (
            packet["processed_estimates"]["standard_error_mol_L"][reporter]
        )
