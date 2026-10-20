from __future__ import annotations

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.campaign_resources import (
    CampaignResourceCard,
    CampaignResourceIntegrityError,
    CampaignResourceLedger,
)
from chemworld.foundation import WorldState
from chemworld.foundation.balances import MaterialExchange, TransitionBalance
from chemworld.foundation.samples import SampleLedger, StoredSample
from chemworld.runtime.transactions import StatePatch, TransactionManager


@pytest.fixture
def env():
    instance = gym.make(
        "ChemWorld", task_id="reaction-to-purification", budget_override=80, debug_truth=True
    )
    instance.reset(seed=0)
    instance.step({"operation": "add_solvent", "solvent": 0, "volume_L": 0.03})
    yield instance
    instance.close()


def commit(manager, before, candidate, operation="wait", action=None):
    return manager.commit(
        state=before,
        operation_type=operation,
        events=(),
        patches=(StatePatch("replace_state", ("species",), candidate),),
        action=action,
    )


def create_matter(env, state, amount=0.001):
    species = env.unwrapped.runtime.domain_services.species_view.feed_species[0]
    amounts = dict(state.species_amounts)
    amounts[species] = amounts.get(species, 0) + amount
    return state.replace(species_amounts=amounts)


@pytest.mark.parametrize("operation", ["wait", "add_reagent", "sample", "transfer"])
def test_legal_candidate_cannot_create_matter_by_using_an_open_operation_name(env, operation):
    base = env.unwrapped
    before = base._state
    candidate = create_matter(env, before)
    assert base.constitution.check_state(candidate).passed
    result = commit(TransactionManager(base.constitution), before, candidate, operation)
    assert result.transaction_status == "rolled_back"
    assert result.state.species_amounts == before.species_amounts
    assert result.state.ledger.time_s == before.ledger.time_s
    assert any(
        c["name"] == "material_conservation" and not c["passed"] for c in result.constitution_checks
    )


def test_fabricated_equal_amount_receipt_in_metadata_is_not_an_input_authorization(env):
    base = env.unwrapped
    before = base._state
    candidate = create_matter(env, before).replace(
        metadata={
            **before.metadata,
            "material_exchanges": [{"source": "stock", "amount_mol": 0.001}],
        }
    )
    result = commit(
        base.runtime.transaction_manager, before, candidate, action={"operation": "wait"}
    )
    assert result.transaction_status == "rolled_back"


def test_real_input_has_source_destination_and_actual_stock_debit(env):
    base = env.unwrapped
    action = {"operation": "add_reagent", "amount_mol": 0.004}
    before = base._state
    result = base.runtime.apply_transaction(before, action)
    assert result.kernel_result.transaction_status == "committed"
    record = result.operation_record
    balance = record.material_balance
    expected = base.runtime.domain_services.primitive.feed_additions(before, action)
    assert dict(balance.exchanges[0].amounts) == expected
    assert balance.exchanges[0].source == "supply:reagent"
    assert balance.exchanges[0].destination == f"vessel:{before.vessel_id}"
    assert dict(balance.stock_consumption)["reagent_mol"] == pytest.approx(sum(expected.values()))
    assert "exchanges" not in record.to_dict()["material_balance"]
    assert record.to_dict(include_hidden=True)["material_balance"]["exchanges"]


def test_unmodeled_catalyst_composition_is_declared_instead_of_a_zero_receipt(env):
    base = env.unwrapped
    assert base.runtime.domain_services.species_view.active_catalyst_species(base._state) is None
    result = base.runtime.apply_transaction(
        base._state,
        {
            "operation": "add_catalyst",
            "catalyst": 0,
            "catalyst_amount_mol": 0.0002,
        },
    )
    assert result.kernel_result.transaction_status == "committed"
    balance = result.operation_record.material_balance
    assert "catalyst_element_inventory" in balance.to_dict()["not_covered"]
    assert dict(balance.stock_consumption)["catalyst_mol"] == 0.0002
    assert not balance.exchanges


def test_forged_charge_counter_or_extra_material_rejects_real_feed(env):
    base = env.unwrapped
    before = base._state
    action = {"operation": "add_reagent", "amount_mol": 0.004}
    candidate, _ = base.runtime.domain_services.apply_operation(before, action)
    for bad in (candidate.replace(species=before.species), create_matter(env, candidate)):
        result = commit(base.runtime.transaction_manager, before, bad, "add_reagent", action)
        assert result.transaction_status == "rolled_back"


def test_record_retains_the_exact_candidate_failure_checks(env, monkeypatch):
    base = env.unwrapped
    services = base.runtime.domain_services
    original = services.primitive._add_feed
    monkeypatch.setattr(
        services.primitive,
        "_add_feed",
        lambda state, additions: create_matter(env, original(state, additions)),
    )
    captured = []
    original_commit = base.runtime.transaction_manager.commit

    def capture(**kwargs):
        result = original_commit(**kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(base.runtime.transaction_manager, "commit", capture)
    result = base.runtime.apply_transaction(
        base._state, {"operation": "add_reagent", "amount_mol": 0.004}
    )
    assert result.kernel_result.transaction_status == "rolled_back"
    assert result.operation_record.constitution_checks == list(captured[0].constitution_checks)
    assert any(not item["passed"] for item in result.operation_record.constitution_checks)


def test_sample_outlet_is_based_on_the_selected_source_not_the_candidate_residual(env):
    base = env.unwrapped
    env.step({"operation": "add_reagent", "amount_mol": 0.01})
    before = base._state
    action = {"operation": "sample", "sample_volume_L": 0.001}
    result = base.runtime.apply_transaction(before, action)
    assert result.kernel_result.transaction_status == "committed"
    exchange = result.operation_record.material_balance.exchanges[0]
    assert exchange.destination == "sink:sample"
    for key, amount in before.species_amounts.items():
        assert dict(exchange.amounts)[key] == pytest.approx(amount * 0.001 / before.volume_L)
    bad = create_matter(env, result.state)
    assert (
        commit(base.runtime.transaction_manager, before, bad, "sample", action).transaction_status
        == "rolled_back"
    )


@pytest.mark.parametrize("unit", ["g", "L", "mmol"])
def test_wrong_units_cannot_be_silently_counted_as_moles(unit):
    with pytest.raises(ValueError, match="require mol"):
        MaterialExchange("r1", "supply", "reactor", "in", (("A", 0.01),), unit)


def test_duplicate_receipts_are_rejected():
    receipt = MaterialExchange("r1", "supply", "reactor", "in", (("A", 0.01),))
    with pytest.raises(ValueError, match="duplicate"):
        TransitionBalance((receipt, receipt))


def test_inactive_vessel_samples_are_inside_the_same_fixed_boundary(env):
    base = env.unwrapped
    before = base._state
    key = base.runtime.domain_services.species_view.feed_species[0]
    sample = StoredSample(
        "stored", "other", "transfer_material", 0.0, 0.001, {key: 0.001}, 298.15, 0, False
    )
    # This is a distinct container, not a copy of the active vessel's retained
    # equipment and finite unused-media supply.
    local = WorldState(
        dict(before.species_amounts),
        before.volume_L,
        before.temperature_K,
        before.pressure_Pa,
        before.phase,
        "other",
        samples=SampleLedger({"stored": sample}),
    )
    before = before.replace(inactive_vessels={"other": local})
    after = before.replace(inactive_vessels={"other": local.replace(samples=SampleLedger())})
    assert not base.constitution.check_material_conservation(before, after).passed
    # Moving the retained sample to the active vessel crosses no world boundary.
    after = after.replace(samples=SampleLedger({"stored": sample}))
    assert base.constitution.check_material_conservation(before, after).passed


def test_entity_stock_debit_needs_a_declared_typed_external_packet(env):
    from chemworld.runtime.disposable_sorbent import MEDIA_QUANTITY, debit_media

    base = env.unwrapped
    before = base._state
    after, _packet = debit_media(before, 0.002)
    assert not base.constitution.check_material_conservation(before, after).passed
    receipt = MaterialExchange(
        "media", "equipment:kit", "sink:spent", "out", ((MEDIA_QUANTITY, 0.002),), "kg", "entity"
    )
    assert base.constitution.check_transition(before, after, TransitionBalance((receipt,))).passed
    wrong_unit = MaterialExchange(
        "media", "equipment:kit", "sink:spent", "out", ((MEDIA_QUANTITY, 0.002),), "g", "entity"
    )
    assert not base.constitution.check_transition(
        before, after, TransitionBalance((wrong_unit,))
    ).passed
    assert not base.constitution.check_material_conservation(before,before,
        allowed_owned_delta={(MEDIA_QUANTITY,"kg"):float("nan")}).passed
    # Merely naming the action dry cannot manufacture an outlet producer.
    result = commit(base.runtime.transaction_manager, before, after, "dry", {"operation": "dry"})
    assert result.transaction_status == "rolled_back"


def test_provider_outlets_are_typed_and_cannot_be_replayed_or_relabelled(env):
    base = env.unwrapped
    for action in [
        {"operation": "add_reagent", "amount_mol": 0.01},
        {"operation": "quench"},
        {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.01},
        {"operation": "add_extractant", "extractant": 3, "volume_L": 0.01},
        {"operation": "mix", "duration_s": 120.0, "stirring_speed_rpm": 500.0},
        {"operation": "settle", "duration_s": 180.0},
    ]:
        assert env.step(action)[-1]["transaction_status"] == "committed"
    before = base._state
    action = {"operation": "separate_phase", "target_phase": "organic"}
    result = base.runtime.apply_transaction(before, action)
    assert result.kernel_result.transaction_status == "committed"
    assert result.operation_record.material_balance.exchanges
    history = result.state.metadata["removed_phase_inventory_history"]
    for replacement in ([*history, history[-1]], [{**history[-1], "operation": "wait"}]):
        bad = result.state.replace(
            metadata={**result.state.metadata, "removed_phase_inventory_history": replacement}
        )
        assert (
            commit(
                base.runtime.transaction_manager, before, bad, "separate_phase", action
            ).transaction_status
            == "rolled_back"
        )
    changed = {key: {**value, "unit": "g"} for key, value in history[-1]["inventories"].items()}
    bad = result.state.replace(
        metadata={
            **result.state.metadata,
            "removed_phase_inventory_history": [{**history[-1], "inventories": changed}],
        }
    )
    assert (
        commit(
            base.runtime.transaction_manager, before, bad, "separate_phase", action
        ).transaction_status
        == "rolled_back"
    )


def test_nonfinite_balance_is_a_failure_even_with_another_finite_element(env):
    base = env.unwrapped
    before = base._state
    key = base.runtime.domain_services.species_view.feed_species[0]
    after = before.replace(species_amounts={**before.species_amounts, key: float("nan")})
    assert not base.constitution.check_material_conservation(before, after).passed


def test_closed_carrier_creation_cannot_be_disguised_as_solvent_addition(env):
    base = env.unwrapped
    before = base._state
    action = {"operation": "add_solvent", "solvent": 0, "volume_L": 0.01}
    candidate, _ = base.runtime.domain_services.apply_operation(before, action)
    assert base.constitution.check_material_conservation(before, candidate).passed
    result = commit(TransactionManager(base.constitution), before, candidate, "add_solvent")
    assert result.transaction_status == "rolled_back"
    assert any(
        c["name"] == "carrier_conservation" and not c["passed"] for c in result.constitution_checks
    )
    assert (
        commit(
            base.runtime.transaction_manager, before, candidate, "add_solvent", action
        ).transaction_status
        == "committed"
    )


def test_no_residual_derived_carrier_receipt_can_hide_an_extra_withdrawal(env):
    from chemworld.runtime.full_process_contract import withdraw_sample
    from chemworld.runtime.solvent_transport import finish_solvent_transition

    base = env.unwrapped
    before = base._state
    # The declared sample debit says 1 mL but the producer removed 2 mL.
    candidate = withdraw_sample(before, 0.002).replace(
        ledger=before.ledger.with_updates(sample_consumed_L=0.001)
    )
    with pytest.raises(ValueError, match="actual inputs/outputs"):
        finish_solvent_transition(
            before, candidate, {"operation": "sample", "sample_volume_L": 0.001}
        )


def test_actual_stock_receipts_debit_finite_supply_once_and_cannot_exceed_it(env):
    base = env.unwrapped
    action = {"operation": "add_reagent", "amount_mol": 0.004}
    physical = base.runtime.apply_transaction(base._state, action)
    amount = dict(physical.operation_record.material_balance.stock_consumption)["reagent_mol"]
    ledger = CampaignResourceLedger(
        CampaignResourceCard(
            card_id="transition-test",
            operation_attempt_limit=10,
            vessel_start_limit=1,
            final_assay_limit=1,
            nonfinal_instrument_use_limit=1,
            stock_limits={"reagent_mol": amount},
        ),
        reagent_charge_molar_multiplier=amount / 0.004,
    )
    ledger.preflight("feed", action)
    outcome = {"operation_committed": True, "material_stock_consumption": {"reagent_mol": amount}}
    ledger.record_outcome("feed", action, outcome)
    ledger.record_outcome("feed", action, outcome)
    assert ledger.stocks_used["reagent_mol"] == pytest.approx(amount)
    assert not ledger.preflight("second-feed", action).allowed
    with pytest.raises(CampaignResourceIntegrityError):
        ledger.record_outcome(
            "feed", action, {**outcome, "material_stock_consumption": {"reagent_mol": 2 * amount}}
        )


def test_physical_outlet_records_reach_debug_logs_but_not_public_species_views(env, tmp_path):
    import json

    from chemworld.data.logging import TrajectoryLogger

    action = {"operation": "add_reagent", "amount_mol": 0.004}
    obs, reward, terminated, truncated, info = env.step(action)
    assert info["material_balance"]["exchanges"]
    path = tmp_path / "balance.jsonl"
    with TrajectoryLogger(path) as logger:
        logger.log(
            task_info={**env.unwrapped.task_info(), **env.unwrapped.evaluator_provenance()},
            step=2,
            action=action,
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            agent_metadata={},
        )
    assert json.loads(path.read_text())["material_balance"] == info["material_balance"]
    assert "exchanges" not in env.unwrapped._last_operation_record.material_balance.to_dict()
    public_env = gym.make("ChemWorld", task_id="reaction-to-purification")
    try:
        public_env.reset(seed=0)
        public_info = public_env.step({"operation": "add_solvent", "solvent": 0, "volume_L": 0.03})[
            -1
        ]
        assert "exchanges" not in public_info["material_balance"]
    finally:
        public_env.close()


@pytest.mark.parametrize("solvent", [0, 2, 3])
def test_complete_purification_closes_all_actual_outlets(solvent):
    from chemworld.runtime.full_process_contract import FULL_PROCESS_CONTRACT

    # A finite public operation fixture must not import a study/provider runner.
    actions = [
        {"operation": "add_solvent", "volume_L": 0.028, "solvent": solvent},
        {"operation": "add_reagent", "amount_mol": 0.010},
        {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
        {
            "operation": "heat",
            "target_temperature_K": 385,
            "duration_s": 1500,
            "stirring_speed_rpm": 720,
        },
        {"operation": "wait", "duration_s": 900, "stirring_speed_rpm": 720},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "quench"},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
        {"operation": "add_extractant", "extractant": 3, "volume_L": 0.018},
        {"operation": "mix", "duration_s": 240, "stirring_speed_rpm": 850},
        {"operation": "settle", "duration_s": 420},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "separate_phase", "target_phase": "organic"},
        {"operation": "wash", "wash_volume_L": 0.008},
        {"operation": "dry"},
        {"operation": "concentrate", "duration_s": 600},
        {"operation": "transfer", "transfer_fraction": 0.97},
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]
    instance = gym.make(
        "ChemWorld",
        task_id="reaction-to-purification",
        budget_override=80,
        full_process_contract_id=FULL_PROCESS_CONTRACT,
        debug_truth=True,
    )
    try:
        instance.reset(seed=0)
        outlets = set()
        for action in actions:
            info = instance.step(action)[-1]
            assert info["transaction_status"] == "committed", (action, info)
            assert all(c["passed"] for c in info["constitution_checks"])
            outlets.update(
                x["destination"]
                for x in info["material_balance"]["exchanges"]
                if x["direction"] == "out"
            )
        assert {
            "sink:wash_aqueous",
            "sink:concentrate_condensate",
            "sink:transfer_source_heel",
        } <= outlets
    finally:
        instance.close()


def test_replay_checks_boundary_receipts_and_detects_changed_stock_consumption(tmp_path):
    from copy import deepcopy

    from chemworld.data.logging import TrajectoryLogger, load_jsonl
    from chemworld.eval.verify import verify_records

    instance = gym.make("ChemWorld", task_id="reaction-to-purification")
    path = tmp_path / "trajectory.jsonl"
    try:
        instance.reset(seed=0)
        task = {**instance.unwrapped.task_info(), **instance.unwrapped.evaluator_provenance()}
        with TrajectoryLogger(path) as logger:
            for i, action in enumerate(
                [
                    {"operation": "add_solvent", "volume_L": 0.03, "solvent": 0},
                    {"operation": "add_reagent", "amount_mol": 0.004},
                ],
                1,
            ):
                obs, reward, terminated, truncated, info = instance.step(action)
                logger.log(
                    task_info=task,
                    step=i,
                    action=action,
                    observation=obs,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    agent_metadata={},
                )
        records = load_jsonl(path)
        assert verify_records(records).verified
        changed = deepcopy(records)
        changed[1]["material_balance"]["stock_consumption"]["reagent_mol"] *= 2
        verified = verify_records(changed)
        assert not verified.verified
        assert any(m["field"].startswith("material_balance") for m in verified.mismatches)
    finally:
        instance.close()
