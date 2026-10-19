from __future__ import annotations

from pathlib import Path

import pytest

from chemworld.agents.event import ScriptedChemistryAgent
from chemworld.data.logging import load_jsonl
from chemworld.eval.resource_accounting import (
    METHOD_RESOURCE_USAGE_VERSION,
    MethodResourceLedger,
    MethodResourceLimitError,
    MethodResourceLimits,
    evaluation_resource_limits,
    load_resource_protocol,
)
from chemworld.eval.runner import run_agent


def _usage(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": METHOD_RESOURCE_USAGE_VERSION,
        "accounting_complete": True,
        "usage_source": "test",
        "model_call_count": 0,
        "input_token_count": 0,
        "output_token_count": 0,
        "monetary_cost_usd": 0.0,
        "training_environment_step_count": 0,
        "cpu_time_s": 0.0,
        "gpu_time_s": 0.0,
        "model_provenance": {},
    }
    payload.update(overrides)
    return payload


def test_resource_protocol_defines_nonclaiming_hard_limits() -> None:
    protocol = load_resource_protocol()
    assert protocol["benchmark_claim_allowed"] is False
    assert protocol["evaluation_budget"]["complete_experiments"] == 40
    assert protocol["checkpoints"] == [4, 8, 12, 20, 40]
    assert protocol["llm_evidence_policy"]["private_chain_of_thought_required"] is False


def test_protocol_translates_to_family_aware_hard_limits() -> None:
    protocol = load_resource_protocol()
    classic = evaluation_resource_limits(
        protocol,
        operation_limit=400,
        requires_online_model=False,
    )
    llm = evaluation_resource_limits(
        protocol,
        operation_limit=400,
        requires_online_model=True,
    )
    assert classic.complete_experiment_limit == 40
    assert classic.checkpoint_complete_experiments == (4, 8, 12, 20, 40)
    assert classic.model_call_limit is None
    assert llm.model_call_limit == 2400
    assert llm.input_token_limit == 1_000_000


def test_resource_ledger_tracks_checkpoints_and_fails_closed() -> None:
    ledger = MethodResourceLedger(
        MethodResourceLimits(
            operation_limit=1,
            complete_experiment_limit=1,
            checkpoint_complete_experiments=(1,),
        )
    )
    ledger.record_decision(elapsed_s=0.01, agent_usage=_usage())
    ledger.record_outcome(experiment_ended=True, update_elapsed_s=0.02)
    snapshot = ledger.snapshot()
    assert snapshot["operation_count"] == 1
    assert snapshot["complete_experiment_count"] == 1
    assert snapshot["reached_checkpoints"] == [1]
    assert snapshot["accounting_complete"] is True
    with pytest.raises(MethodResourceLimitError, match="operation_count"):
        ledger.record_decision(elapsed_s=0.01, agent_usage=_usage())


def test_online_model_usage_requires_complete_provenance() -> None:
    with pytest.raises(ValueError, match="explicit provider request limit"):
        MethodResourceLedger(
            MethodResourceLimits(operation_limit=2),
            requires_online_model=True,
        )

    ledger = MethodResourceLedger(
        MethodResourceLimits(operation_limit=2, model_call_limit=12),
        requires_online_model=True,
    )
    with pytest.raises(ValueError, match="provenance"):
        ledger.record_decision(
            elapsed_s=0.01,
            agent_usage=_usage(model_call_count=1),
        )

    valid = MethodResourceLedger(
        MethodResourceLimits(operation_limit=2, model_call_limit=12),
        requires_online_model=True,
    )
    valid.record_decision(
        elapsed_s=0.01,
        agent_usage=_usage(
            model_call_count=1,
            input_token_count=200,
            output_token_count=50,
            monetary_cost_usd=0.01,
            model_provenance={
                "provider": "test-provider",
                "model_id": "test-model",
                "model_snapshot_or_access_date": "2026-07-11",
                "prompt_hash": "abc123",
                "request_parameters": {"temperature": 0.0},
                "tokenizer_or_provider_usage_source": "provider",
            },
        ),
    )
    assert valid.snapshot()["agent_usage"]["model_call_count"] == 1


def test_online_development_ledger_allows_only_explicit_unknown_monetary_accounting() -> None:
    provenance = {
        "provider": "test-provider",
        "model_id": "test-model",
        "model_snapshot_or_access_date": "2026-07-25",
        "prompt_hash": "abc123",
        "request_parameters": {"max_attempts": 3},
        "tokenizer_or_provider_usage_source": "provider",
    }
    usage = _usage(
        accounting_complete=False,
        provider_usage_accounting_complete=True,
        provider_call_accounting_complete=True,
        provider_token_accounting_complete=True,
        monetary_accounting_complete=False,
        model_call_count=1,
        input_token_count=200,
        output_token_count=50,
        monetary_cost_usd=0.0,
        model_provenance=provenance,
    )
    ledger = MethodResourceLedger(
        MethodResourceLimits(
            operation_limit=2,
            model_call_limit=6,
            monetary_cost_limit_usd=None,
        ),
        requires_online_model=True,
    )

    ledger.record_decision(elapsed_s=0.01, agent_usage=usage)

    snapshot = ledger.snapshot()
    assert snapshot["accounting_complete"] is False
    assert snapshot["development_monetary_accounting_incomplete"] is True

    capped = MethodResourceLedger(
        MethodResourceLimits(
            operation_limit=2,
            model_call_limit=6,
            monetary_cost_limit_usd=1.0,
        ),
        requires_online_model=True,
    )
    with pytest.raises(ValueError, match="uncapped monetary accounting"):
        capped.record_decision(elapsed_s=0.01, agent_usage=usage)

    missing_token_attestation = dict(usage)
    missing_token_attestation.pop("provider_token_accounting_complete")
    with pytest.raises(ValueError, match="uncapped monetary accounting"):
        MethodResourceLedger(
            MethodResourceLimits(operation_limit=2, model_call_limit=6),
            requires_online_model=True,
        ).record_decision(
            elapsed_s=0.01,
            agent_usage=missing_token_attestation,
        )


def test_online_model_ledger_allows_counted_retries_within_explicit_limit() -> None:
    ledger = MethodResourceLedger(
        MethodResourceLimits(operation_limit=2, model_call_limit=6),
        requires_online_model=True,
    )
    provenance = {
        "provider": "test-provider",
        "model_id": "test-model",
        "model_snapshot_or_access_date": "2026-07-11",
        "prompt_hash": "abc123",
        "request_parameters": {"max_attempts": 3},
        "tokenizer_or_provider_usage_source": "provider",
    }
    ledger.record_decision(
        elapsed_s=0.01,
        agent_usage=_usage(model_call_count=3, model_provenance=provenance),
    )
    ledger.record_outcome(experiment_ended=False, update_elapsed_s=0.01)
    ledger.record_decision(
        elapsed_s=0.01,
        agent_usage=_usage(model_call_count=6, model_provenance=provenance),
    )
    assert ledger.snapshot()["agent_usage"]["model_call_count"] == 6
    with pytest.raises(MethodResourceLimitError, match="model_call_count"):
        ledger.record_decision(
            elapsed_s=0.01,
            agent_usage=_usage(model_call_count=7, model_provenance=provenance),
        )


def test_reconcile_agent_usage_updates_session_tokens_without_counting_operation() -> None:
    provenance = {
        "provider": "test-provider",
        "model_id": "test-model",
        "model_snapshot_or_access_date": "2026-07-25",
        "prompt_hash": "abc123",
        "request_parameters": {"session_scope": "complete_experiment"},
        "tokenizer_or_provider_usage_source": "provider",
    }
    ledger = MethodResourceLedger(
        MethodResourceLimits(
            operation_limit=4,
            model_call_limit=2,
            input_token_limit=1_000,
            output_token_limit=200,
        ),
        requires_online_model=True,
    )
    ledger.record_decision(
        elapsed_s=0.01,
        agent_usage=_usage(model_provenance=provenance),
    )

    ledger.reconcile_agent_usage(
        _usage(
            model_call_count=1,
            input_token_count=700,
            output_token_count=80,
            model_provenance=provenance,
        )
    )

    snapshot = ledger.snapshot()
    assert snapshot["operation_count"] == 1
    assert snapshot["agent_usage"]["model_call_count"] == 1
    assert snapshot["agent_usage"]["input_token_count"] == 700
    assert snapshot["agent_usage"]["output_token_count"] == 80


def test_reconcile_agent_usage_enforces_session_level_token_limit() -> None:
    provenance = {
        "provider": "test-provider",
        "model_id": "test-model",
        "model_snapshot_or_access_date": "2026-07-25",
        "prompt_hash": "abc123",
        "request_parameters": {"session_scope": "complete_experiment"},
        "tokenizer_or_provider_usage_source": "provider",
    }
    ledger = MethodResourceLedger(
        MethodResourceLimits(
            operation_limit=4,
            model_call_limit=2,
            input_token_limit=100,
        ),
        requires_online_model=True,
    )
    ledger.record_decision(
        elapsed_s=0.01,
        agent_usage=_usage(model_provenance=provenance),
    )

    with pytest.raises(MethodResourceLimitError, match="input_token_count"):
        ledger.reconcile_agent_usage(
            _usage(
                model_call_count=1,
                input_token_count=101,
                model_provenance=provenance,
            )
        )


def test_uncached_input_tokens_have_an_independent_hard_limit() -> None:
    ledger = MethodResourceLedger(
        MethodResourceLimits(
            operation_limit=2,
            model_call_limit=2,
            input_token_limit=1_000,
            uncached_input_token_limit=100,
            output_token_limit=200,
        ),
        requires_online_model=True,
    )
    provenance = {
        "provider": "test-provider",
        "model_id": "test-model",
        "model_snapshot_or_access_date": "2026-08-05",
        "prompt_hash": "abc123",
        "request_parameters": {"session_scope": "complete_experiment"},
        "tokenizer_or_provider_usage_source": "provider",
    }
    with pytest.raises(MethodResourceLimitError, match="uncached_input_token_count"):
        ledger.reconcile_agent_usage(
            _usage(
                model_call_count=1,
                input_token_count=900,
                cached_input_token_count=850,
                uncached_input_token_count=50,
                output_token_count=80,
                model_provenance=provenance,
            )
        )
        ledger.reconcile_agent_usage(
            _usage(
                model_call_count=1,
                input_token_count=1_000,
                cached_input_token_count=850,
                uncached_input_token_count=150,
                output_token_count=80,
                model_provenance=provenance,
            )
        )


def test_online_ledger_allows_one_explicit_in_flight_experiment_session() -> None:
    provenance = {
        "provider": "test-provider",
        "model_id": "test-model",
        "model_snapshot_or_access_date": "2026-07-25",
        "prompt_hash": "abc123",
        "request_parameters": {"session_scope": "complete_experiment"},
        "tokenizer_or_provider_usage_source": "provider_turn_completed",
    }
    pending = _usage(
        accounting_complete=False,
        provider_usage_pending=True,
        in_flight_model_call_count=1,
        provider_usage_accounting_complete=False,
        provider_call_accounting_complete=True,
        provider_token_accounting_complete=False,
        provider_cache_accounting_complete=False,
        monetary_accounting_complete=False,
        model_call_count=1,
        model_provenance=provenance,
    )
    ledger = MethodResourceLedger(
        MethodResourceLimits(
            operation_limit=4,
            wall_time_limit_s=60.0,
            model_call_limit=2,
            input_token_limit=1_000,
            output_token_limit=200,
        ),
        requires_online_model=True,
    )

    ledger.record_decision(elapsed_s=0.01, agent_usage=pending)

    snapshot = ledger.snapshot()
    assert snapshot["operation_count"] == 1
    assert snapshot["provider_usage_pending"] is True
    assert snapshot["development_deferred_provider_usage"] is True
    assert snapshot["agent_usage"]["in_flight_model_call_count"] == 1


def test_online_ledger_rejects_unbounded_or_multiple_deferred_sessions() -> None:
    provenance = {
        "provider": "test-provider",
        "model_id": "test-model",
        "model_snapshot_or_access_date": "2026-07-25",
        "prompt_hash": "abc123",
        "request_parameters": {"session_scope": "complete_experiment"},
        "tokenizer_or_provider_usage_source": "provider_turn_completed",
    }
    pending = _usage(
        accounting_complete=False,
        provider_usage_pending=True,
        in_flight_model_call_count=1,
        provider_usage_accounting_complete=False,
        provider_call_accounting_complete=True,
        provider_token_accounting_complete=False,
        provider_cache_accounting_complete=False,
        monetary_accounting_complete=False,
        model_call_count=1,
        model_provenance=provenance,
    )
    with pytest.raises(ValueError, match="bounded experiment session"):
        MethodResourceLedger(
            MethodResourceLimits(operation_limit=4, model_call_limit=2),
            requires_online_model=True,
        ).record_decision(elapsed_s=0.01, agent_usage=pending)

    multiple = dict(pending)
    multiple["in_flight_model_call_count"] = 2
    multiple["model_call_count"] = 2
    with pytest.raises(ValueError, match="bounded experiment session"):
        MethodResourceLedger(
            MethodResourceLimits(
                operation_limit=4,
                model_call_limit=2,
                input_token_limit=1_000,
                output_token_limit=200,
            ),
            requires_online_model=True,
        ).record_decision(elapsed_s=0.01, agent_usage=multiple)


def test_runner_retains_method_resource_ledger(tmp_path: Path) -> None:
    output = tmp_path / "resource-ledger.jsonl"
    history = run_agent(
        env_id="ChemWorld",
        agent=ScriptedChemistryAgent(),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=3,
        task_id="reaction-to-assay",
        output_path=output,
        method_resource_limits={
            "complete_experiment_limit": 1,
            "checkpoint_complete_experiments": [1],
            "wall_time_limit_s": 30.0,
        },
    )
    records = load_jsonl(output)
    assert history[-1].method_resources["operation_count"] == len(history)
    assert records[-1]["method_resources"]["operation_count"] == len(records)
    assert records[-1]["method_resources"]["complete_experiment_count"] == 1
    assert records[-1]["method_resources"]["reached_checkpoints"] == [1]
    assert records[-1]["method_resources"]["agent_usage"]["model_call_count"] == 0
