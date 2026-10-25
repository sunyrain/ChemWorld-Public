from __future__ import annotations

from typing import Any

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.eval.observation_identifiability import PublicSpectrumArchive
from chemworld.eval.runner import _attach_spectrum_archive, _public_spectrum_packet
from chemworld.foundation import equipment_settings, instrument_equipment_id
from chemworld.physchem.spectroscopy_adapter_manifest import (
    FORBIDDEN_PACKET_KEYS,
    ValidatedInstrumentRuntimeProvider,
    instrument_runtime_adapter_manifest,
)
from chemworld.runtime.instrument_cost_services import ChemWorldInstrumentCostServices
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.transactions import TransactionResult
from chemworld.world.instruments import (
    INSTRUMENT_RUNTIME_MODEL_ID,
    INSTRUMENT_RUNTIME_PROVIDER_PATH,
    instrument_runtime_contract_hash,
)


class RecordingProvider(ValidatedInstrumentRuntimeProvider):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def evaluate(self, inputs: Any) -> ModelProviderResult:
        self.calls.append(dict(inputs))
        return super().evaluate(inputs)


class FailingProvider(ValidatedInstrumentRuntimeProvider):
    def evaluate(self, inputs: Any) -> ModelProviderResult:
        del inputs
        return ModelProviderResult(
            outputs={},
            diagnostics={"provider_failure_injected": True},
            success=False,
            failure_reason="injected instrument provider failure",
            provenance=self.model_contract.provenance,
        )


@pytest.mark.parametrize("instrument_id", ["hplc", "gc", "uvvis"])
def test_formal_measure_runtime_dynamically_calls_validated_provider(
    instrument_id: str,
) -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=13)
    try:
        env.reset(seed=13)
        base: Any = env.unwrapped
        provider = RecordingProvider()
        base.observation_kernel.instrument_provider = provider
        env.step({"operation": "add_solvent", "volume_L": 0.028, "solvent": 2})
        env.step({"operation": "add_reagent", "amount_mol": 0.010})
        volume_before = float(base._state.volume_L)

        _observation, _reward, _terminated, _truncated, info = env.step(
            {"operation": "measure", "instrument": instrument_id}
        )

        assert info["transaction_status"] == "committed"
        assert len(provider.calls) == 1
        assert provider.calls[0]["instrument_id"] == instrument_id
        assert provider.calls[0]["sample_basis_volume_L"] == pytest.approx(volume_before)
        assert base._state.volume_L < volume_before
        assert set(provider.calls[0].get("public_species_amounts_mol") or {}) <= {
            "reactant_public",
            "target_public",
            "impurity_public",
            "degradation_public",
        }
        packet = info["raw_signal"]
        assert {
            "sample_state",
            "raw_signal",
            "peaks",
            "assignments",
            "processed_estimates",
            "uncertainty",
            "calibration",
            "missingness",
        } <= set(packet)
        assert not (_nested_keys(packet) & FORBIDDEN_PACKET_KEYS)
        execution = base.observation_kernel.last_provider_execution
        assert execution["success"] is True
        assert execution["model_id"] == INSTRUMENT_RUNTIME_MODEL_ID
        assert execution["provider_path"] == INSTRUMENT_RUNTIME_PROVIDER_PATH
        assert execution["diagnostics"]["layered_packet"] is True
        assert execution["diagnostics"]["identity_safe"] is True

        settings = equipment_settings(
            base._state.equipment,
            instrument_equipment_id(instrument_id),
        )
        assert settings["model_id"] == INSTRUMENT_RUNTIME_MODEL_ID
        assert settings["provider_path"] == INSTRUMENT_RUNTIME_PROVIDER_PATH
        assert settings["provider_contract_hash"] == instrument_runtime_contract_hash()
        assert settings["diagnostics"]["sample_volume_exact"] is True
        assert len(settings["execution_history"]) == 1
    finally:
        env.close()


def test_ph_and_final_assay_use_layered_runtime_packets() -> None:
    equilibrium = gym.make("ChemWorld", task_id="equilibrium-characterization", seed=7)
    try:
        equilibrium.reset(seed=7)
        base: Any = equilibrium.unwrapped
        provider = RecordingProvider()
        base.observation_kernel.instrument_provider = provider
        equilibrium.step({"operation": "add_solvent", "volume_L": 0.025, "solvent": 2})
        equilibrium.step({"operation": "add_reagent", "amount_mol": 0.008})
        _obs, _reward, _terminated, _truncated, ph_info = equilibrium.step(
            {"operation": "measure", "instrument": "ph_meter"}
        )
        assert ph_info["transaction_status"] == "committed"
        assert provider.calls[-1]["instrument_id"] == "ph_meter"
        assert ph_info["raw_signal"]["calibration"]["slope_mV_per_pH"] == pytest.approx(-59.16)
    finally:
        equilibrium.close()

    assay = gym.make("ChemWorld", task_id="reaction-to-assay", seed=7)
    try:
        assay.reset(seed=7)
        base = assay.unwrapped
        provider = RecordingProvider()
        base.observation_kernel.instrument_provider = provider
        assay.step({"operation": "add_solvent", "volume_L": 0.025, "solvent": 2})
        assay.step({"operation": "add_reagent", "amount_mol": 0.008})
        assay.step({"operation": "terminate"})
        _obs, _reward, terminated, _truncated, info = assay.step(
            {"operation": "measure", "instrument": "final_assay"}
        )
        assert terminated is True
        assert provider.calls[-1]["instrument_id"] == "final_assay"
        assert info["raw_signal"]["kind"] == "final_assay_packet"
        assert {"hplc", "gc", "uvvis", "ph_meter"} <= set(info["raw_signal"]["spectra"])
        assert not (_nested_keys(info["raw_signal"]) & FORBIDDEN_PACKET_KEYS)
    finally:
        assay.close()


def test_repeated_nonterminal_measurements_retain_each_execution_record() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=9)
    try:
        env.reset(seed=9)
        base: Any = env.unwrapped
        provider = RecordingProvider()
        base.observation_kernel.instrument_provider = provider
        env.step({"operation": "add_solvent", "volume_L": 0.025, "solvent": 2})
        env.step({"operation": "add_reagent", "amount_mol": 0.008})
        for _ in range(2):
            _obs, _reward, _terminated, _truncated, info = env.step(
                {"operation": "measure", "instrument": "hplc"}
            )
            assert info["transaction_status"] == "committed"

        settings = equipment_settings(
            base._state.equipment,
            instrument_equipment_id("hplc"),
        )
        assert len(provider.calls) == 2
        assert settings["use_count"] == 2
        assert [record["measurement_index"] for record in settings["execution_history"]] == [1, 2]
        assert all(
            record["provider_contract_hash"] == instrument_runtime_contract_hash()
            for record in settings["execution_history"]
        )
    finally:
        env.close()


def test_destructive_measurement_observes_pre_withdrawal_sample_composition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=9, debug_truth=True)
    try:
        env.reset(seed=9)
        base: Any = env.unwrapped
        env.step({"operation": "add_solvent", "volume_L": 0.025, "solvent": 2})
        env.step({"operation": "add_reagent", "amount_mol": 0.008})
        env.step({"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1})
        env.step(
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1_500.0,
                "stirring_speed_rpm": 720.0,
            }
        )
        truth_before = base.observation_kernel._truth_values(base._state)
        species_before = dict(base._state.species_amounts)
        phase_amounts_before = base._state.phases.total_amounts_mol()
        volume_before = base._state.volume_L
        observed_states: list[tuple[dict[str, float], float]] = []
        original_truth_values = base.observation_kernel._truth_values

        def record_truth_state(state: Any) -> dict[str, float]:
            observed_states.append((dict(state.species_amounts), float(state.volume_L)))
            return original_truth_values(state)

        monkeypatch.setattr(base.observation_kernel, "_truth_values", record_truth_state)

        _observation, _reward, _terminated, _truncated, info = env.step(
            {"operation": "measure", "instrument": "hplc"}
        )

        assert info["transaction_status"] == "committed"
        assert observed_states == [(species_before, volume_before)]
        assert base._state.volume_L < volume_before
        remaining_fraction = base._state.volume_L / volume_before
        assert base._state.species_amounts == pytest.approx(
            {
                species_id: amount_mol * remaining_fraction
                for species_id, amount_mol in species_before.items()
            }
        )
        assert base._state.phases.total_amounts_mol() == pytest.approx(
            {
                species_id: amount_mol * remaining_fraction
                for species_id, amount_mol in phase_amounts_before.items()
            }
        )
        truth_after = original_truth_values(base._state)
        for metric in ("yield", "conversion", "selectivity"):
            assert truth_after[metric] == pytest.approx(truth_before[metric])
    finally:
        env.close()


def test_sequential_hplc_and_final_assay_preserve_composition_metrics() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=11, debug_truth=True)
    try:
        env.reset(seed=11)
        base: Any = env.unwrapped
        for action in (
            {"operation": "add_solvent", "volume_L": 0.025, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.008},
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1_500.0,
                "stirring_speed_rpm": 720.0,
            },
            {"operation": "quench"},
        ):
            env.step(action)
        truth_before = base.observation_kernel._truth_values(base._state)
        initial_before = dict(base._state.species.initial_amounts_mol)
        volume_before = float(base._state.volume_L)
        samples_before = float(base._state.ledger.sample_consumed_L)

        env.step({"operation": "measure", "instrument": "hplc"})
        truth_after_hplc = base.observation_kernel._truth_values(base._state)
        initial_after_hplc = dict(base._state.species.initial_amounts_mol)
        expected_remaining_fraction = base._state.volume_L / volume_before
        env.step({"operation": "terminate"})
        env.step({"operation": "measure", "instrument": "final_assay"})
        truth_after_final = base.observation_kernel._truth_values(base._state)

        for metric in ("yield", "conversion", "selectivity"):
            assert truth_after_hplc[metric] == pytest.approx(truth_before[metric])
            assert truth_after_final[metric] == pytest.approx(truth_before[metric])
        assert initial_after_hplc == pytest.approx(
            {
                species_id: amount_mol * expected_remaining_fraction
                for species_id, amount_mol in initial_before.items()
            }
        )
        assert base._state.ledger.sample_consumed_L > samples_before
    finally:
        env.close()


def test_rolled_back_final_assay_cannot_observe_reward_or_end_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=17)
    try:
        env.reset(seed=17)
        base: Any = env.unwrapped
        provider = RecordingProvider()
        base.observation_kernel.instrument_provider = provider
        env.step({"operation": "add_solvent", "volume_L": 0.025, "solvent": 2})
        env.step({"operation": "add_reagent", "amount_mol": 0.008})
        env.step({"operation": "measure", "instrument": "hplc"})
        env.step({"operation": "terminate"})
        cached_observation = dict(base._state.process.last_observation)
        cached_mask = dict(base._state.process.last_observed_mask)
        provider_call_count = len(provider.calls)
        transaction_manager = base.runtime.transaction_manager

        def reject_candidate(
            *,
            state: Any,
            operation_type: str,
            events: tuple[Any, ...],
            patches: tuple[Any, ...],
            action: dict[str, Any] | None = None,
            actual_duration_s: float | None = None,
        ) -> TransactionResult:
            del patches, action, actual_duration_s
            assert operation_type == "measure"
            return transaction_manager.rollback(
                state=state,
                operation_type=operation_type,
                rollback_reason="constitution_failed",
                failed_checks=("injected_measurement_failure",),
                events=events,
            )

        monkeypatch.setattr(transaction_manager, "commit", reject_candidate)

        _observation, reward, terminated, truncated, info = env.step(
            {"operation": "measure", "instrument": "final_assay"}
        )

        assert info["transaction_status"] == "rolled_back"
        assert info["rollback_reason"] == "constitution_failed"
        assert info["constraint_flags"]["constitution_failed"] is True
        assert reward == 0.0
        assert info["measurement_cost"] == 0.0
        assert info["sample_consumed"] == 0.0
        assert info["reward_source"] == "constitution_rollback"
        assert info["environment_reward"] == {
            "schema_version": "chemworld-environment-reward-0.2",
            "semantics": "fresh_measurement_score_delta",
            "fresh_measurement": False,
            "cached_observation_rewarded": False,
            "score_delta": 0.0,
        }
        assert terminated is False
        assert truncated is False
        assert info["experiment_ended"] is False
        assert len(provider.calls) == provider_call_count
        assert base._state.process.last_observation == cached_observation
        assert base._state.process.last_observed_mask == cached_mask
    finally:
        env.close()


def test_provider_seed_replay_and_numerical_domain_fail_closed() -> None:
    provider = ValidatedInstrumentRuntimeProvider()
    valid = {
        "instrument_id": "hplc",
        "public_values": {"yield": 0.6, "purity": 0.8},
        "public_species_amounts_mol": {
            "target_public": 0.20,
            "impurity_public": 0.03,
        },
        "sample_basis_volume_L": 1.0,
        "seed": 31,
        "replicate_count": 3,
    }
    first = provider.evaluate(valid)
    replay = provider.evaluate(valid)
    assert first.success is True
    assert first.outputs["packet"] == replay.outputs["packet"]

    for changes in (
        {"instrument_id": "imaginary"},
        {"sample_basis_volume_L": 0.0},
        {"sample_basis_volume_L": float("nan")},
        {"public_values": {"yield": float("inf")}},
        {"public_species_amounts_mol": {"private_species": 1.0}},
        {"replicate_count": 0},
    ):
        result = provider.evaluate({**valid, **changes})
        assert result.success is False
        assert result.outputs == {}
        assert result.failure_reason


def test_insufficient_sample_and_repeat_assay_fail_before_material_mutation() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=3)
    try:
        env.reset(seed=3)
        base: Any = env.unwrapped
        services = ChemWorldInstrumentCostServices(base.constitution)
        too_small = base._state.replace(
            volume_L=0.00010,
            species_amounts={"public_test": 0.001},
        )
        with pytest.raises(ValueError, match="insufficient sample volume"):
            services.apply_measurement_cost(
                too_small,
                {"operation": "measure", "instrument": "hplc"},
            )
        assert too_small.volume_L == pytest.approx(0.00010)
        assert too_small.species_amounts == {"public_test": 0.001}

        env.step({"operation": "add_solvent", "volume_L": 0.025, "solvent": 2})
        env.step({"operation": "add_reagent", "amount_mol": 0.008})
        env.step({"operation": "terminate"})
        _obs, _reward, _terminated, _truncated, info = env.step(
            {"operation": "measure", "instrument": "final_assay"}
        )
        completed_state = base._state
        with pytest.raises(ValueError, match="cannot be repeated"):
            services.apply_measurement_cost(
                completed_state,
                {"operation": "measure", "instrument": "final_assay"},
            )
        assert info["measurement_cost"] > 0.0
    finally:
        env.close()


def test_provider_failure_preserves_actual_cost_and_sample_but_withholds_signal() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=5)
    try:
        env.reset(seed=5)
        base: Any = env.unwrapped
        env.step({"operation": "add_solvent", "volume_L": 0.025, "solvent": 2})
        env.step({"operation": "add_reagent", "amount_mol": 0.008})
        volume_before = float(base._state.volume_L)
        samples_before = float(base._state.ledger.sample_consumed_L)
        species_before = dict(base._state.species_amounts)
        initial_before = dict(base._state.species.initial_amounts_mol)
        cost_before = base._state.ledger.cost
        base.observation_kernel.instrument_provider = FailingProvider()

        _obs, _reward, _terminated, _truncated, info = env.step(
            {"operation": "measure", "instrument": "hplc"}
        )

        assert info["transaction_status"] == "committed"
        assert info["operation_outcome"] == "success"
        assert info["measurement_status"] == "invalid"
        assert info["raw_signal"] == {}
        instrument = base.constitution.instruments["hplc"]
        fraction = 1.0 - instrument.sample_volume_L / volume_before
        assert info["measurement_cost"] == pytest.approx(instrument.cost)
        assert info["sample_consumed"] == pytest.approx(instrument.sample_volume_L)
        assert base._state.ledger.cost - cost_before == pytest.approx(instrument.cost)
        assert base._state.volume_L == pytest.approx(volume_before - instrument.sample_volume_L)
        assert base._state.ledger.sample_consumed_L == pytest.approx(
            samples_before + instrument.sample_volume_L,
        )
        assert base._state.species_amounts == pytest.approx({
            key: value * fraction for key, value in species_before.items()
        })
        assert base._state.species.initial_amounts_mol == pytest.approx({
            key: value * fraction for key, value in initial_before.items()
        })
    finally:
        env.close()


def test_history_catalog_exposes_only_requested_packet() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=21)
    archive = PublicSpectrumArchive()
    try:
        env.reset(seed=21)
        for step, action in enumerate((
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {"operation": "measure", "instrument": "hplc"},
            {"operation": "measure", "instrument": "uvvis"},
        ), start=1):
            _, _, _, _, info = env.step(action)
            assert info["transaction_status"] == "committed"
            view = {key: env.unwrapped.observation_view(key)
                    for key in ("tool_json", "lab_report")}
            if action["operation"] == "measure":
                packet = _public_spectrum_packet(view)
                assert packet is not None
                archive.record(f"step-{step}", packet, experiment_index=0,
                               measurement_step=step, measurement_cost=info["measurement_cost"])
        catalog = archive.catalog()
        assert [row["instrument_id"] for row in catalog] == ["hplc", "uvvis"]
        assert len({row["spectrum_id"] for row in catalog}) == 2
        assert all("raw_signal" not in row for row in catalog)
        attached, event = _attach_spectrum_archive(
            view, archive=archive, latest_spectrum_id="step-4", requested_spectrum_id="step-3",
        )
        requested = attached["tool_json"]["requested_historical_spectrum"]
        assert requested["spectrum_id"] == "step-3"
        assert requested["instrument_id"] == "hplc"
        assert requested["raw_signal"] == archive.retrieve("step-3")["raw_signal"]
        assert event["success"] is True
        assert "requested_historical_spectrum" not in view["tool_json"]
    finally:
        env.close()


def test_runtime_manifest_is_truthful() -> None:
    manifest = instrument_runtime_adapter_manifest()
    assert manifest.status == "integrated"
    assert manifest.provider_contract.role.value == "runtime"
    assert manifest.provider_contract.maturity.value == "reference_validated"
    assert manifest.provider_contract.model_id == INSTRUMENT_RUNTIME_MODEL_ID


def _nested_keys(payload: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys.update(_nested_keys(value))
    elif isinstance(payload, list | tuple):
        for value in payload:
            keys.update(_nested_keys(value))
    return keys
