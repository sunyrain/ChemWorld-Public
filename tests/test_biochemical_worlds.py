"""Actual public actions, exact atomic accounting and independent random streams."""

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from chemworld.foundation.balances import TransitionBalance
from chemworld.foundation.discrete_inventory import AVOGADRO, counts_to_mol, mol_to_counts
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.biochemical_instruments import capture_outlet, capture_plan
from chemworld.world.biochemical_worlds import biochemical_network, biochemical_world


def session(kind="enzyme", seed=7):
    runtime = ChemWorldRuntime(composition=biochemical_world(kind, seed=seed))
    runtime.reset(seed=12)
    return runtime


def execute(runtime, **action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info.get("error_message")
    return result


@pytest.mark.parametrize("kind", ["reversible", "enzyme"])
def test_actual_add_wait_measure_capture_and_accounting(kind):
    runtime = session(kind)
    execute(runtime, operation="add_component", component=0, amount_mol=10 / AVOGADRO)
    assert sum(mol_to_counts(runtime._state.species_amounts).values()) == (
        110 if kind == "reversible" else 120
    )
    execute(runtime, operation="wait", duration_s=5)
    before = runtime._state
    selected, rng = capture_plan(before)
    count = execute(runtime, operation="measure", instrument="molecule_counter")
    assert runtime._state.species_amounts == before.species_amounts
    assert runtime._state.process.stochastic_continuation == before.process.stochastic_continuation
    assert all(c.unit == "counts" and c.uncertainty == 1 for c in count.observation.channels)
    result = execute(runtime, operation="measure", instrument="molecule_capture")
    after = runtime._state
    old, new = mol_to_counts(before.species_amounts), mol_to_counts(after.species_amounts)
    assert new == {k: n - selected[k] for k, n in old.items()}
    balance = TransitionBalance(capture_outlet(before, {}, {}))
    assert runtime.constitution.check_transition(before, after, balance).passed
    assert after.volume_L == before.volume_L == 1e-15
    assert after.ledger.time_s == before.ledger.time_s == 5
    assert after.ledger.analysis_time_s == 1
    assert after.ledger.sample_consumed_L == 0
    assert after.ledger.cost == pytest.approx(before.ledger.cost + 0.02)
    cont = after.process.stochastic_continuation
    assert cont["sampling_rng_state"] == rng
    assert cont["rng_state"] == before.process.stochastic_continuation["rng_state"]
    assert cont["pending"] is None
    assert all(
        c.sampled_at.time_s == 5 and c.available_at.time_s == 1 for c in result.observation.channels
    )
    assert "captured_counts" not in json.dumps(result.info)
    assert "sampling_rng_state" not in json.dumps(result.info)
    execute(runtime, operation="wait", duration_s=2)


@pytest.mark.parametrize("dose", [0.5, 10.1, -1, float("nan"), float("inf")])
def test_bad_dose_rejected_without_process_or_inventory_change(dose):
    runtime = session()
    before = runtime._state
    result = runtime.execute(
        {"operation": "add_component", "component": 0, "amount_mol": dose / AVOGADRO}
    )
    assert result.info["transaction_status"] == (
        "rolled_back" if dose in (0.5, 10.1) else "validation_failed"
    )
    assert runtime._state.species_amounts == before.species_amounts
    assert runtime._state.process.stochastic_continuation == before.process.stochastic_continuation


def test_exact_atom_gate_rejects_one_molecule_despite_old_mol_tolerance():
    runtime = session()
    before = runtime._state
    counts = mol_to_counts(before.species_amounts)
    counts["S"] -= 1
    after = before.replace(species_amounts=counts_to_mol(counts))
    check = runtime.constitution.check_transition(before, after, TransitionBalance())
    assert not check.passed and check.tolerance == 0


def test_wrong_boundary_and_fractional_candidate_rejected():
    runtime = session()
    before = runtime._state
    amounts = dict(before.species_amounts)
    amounts["S"] -= 0.5 / AVOGADRO
    check = runtime.constitution.check_transition(
        before, before.replace(species_amounts=amounts), TransitionBalance()
    )
    assert not check.passed
    receipts = capture_outlet(before, {}, {})
    execute(runtime, operation="measure", instrument="molecule_capture")
    bad = replace(receipts[0], amounts=tuple(counts_to_mol({"S": 1}).items()))
    assert not runtime.constitution.check_transition(
        before, runtime._state, TransitionBalance((bad,))
    ).passed


def test_read_frequency_preserves_sampling_and_process_streams():
    left, right = session(), session()
    for runtime in (left, right):
        execute(runtime, operation="wait", duration_s=10)
    for _ in range(4):
        execute(right, operation="measure", instrument="molecule_counter")
    for runtime in (left, right):
        execute(runtime, operation="measure", instrument="molecule_capture")
        execute(runtime, operation="wait", duration_s=10)
    assert left._state.species_amounts == right._state.species_amounts
    assert (
        left._state.process.stochastic_continuation == right._state.process.stochastic_continuation
    )


@pytest.mark.parametrize("scope", ["material", "batch", "vessel"])
def test_lifecycle_restores_discrete_process_not_legacy_ode(scope):
    runtime = session()
    execute(runtime, operation="wait", duration_s=10)
    execute(runtime, operation="measure", instrument="molecule_capture")
    runtime.reset(options={"scope": scope})
    assert mol_to_counts(runtime._state.species_amounts) == {"E": 10, "S": 100, "ES": 0, "P": 0}
    cont = runtime._state.process.stochastic_continuation
    assert cont["time_s"] == runtime._state.ledger.time_s
    assert "sampling_rng_state" in cont
    result = execute(runtime, operation="wait", duration_s=1)
    assert result.info["outcome_diagnostics"]["advance_strategy"] == "gillespie-direct"


def test_snapshot_new_process_with_capture_and_noise(tmp_path):
    runtime = session()
    execute(runtime, operation="wait", duration_s=10)
    execute(runtime, operation="measure", instrument="molecule_capture")
    source, output = tmp_path / "snapshot.json", tmp_path / "state.json"
    source.write_text(json.dumps(runtime.snapshot().to_dict()))
    actions = [
        {"operation": "add_component", "component": 1, "amount_mol": 10 / AVOGADRO},
        {"operation": "wait", "duration_s": 10},
        {"operation": "measure", "instrument": "molecule_capture"},
    ]
    for action in actions:
        execute(runtime, **action)
    code = (
        "import json; from pathlib import Path; from chemworld.runtime import ChemWorldRuntime; "
        f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
        f"[r.execute(a) for a in {actions!r}]; "
        f"Path({str(output)!r}).write_text(json.dumps(r._state.to_dict()))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert json.loads(output.read_text()) == json.loads(json.dumps(runtime._state.to_dict()))


@pytest.mark.parametrize("kind", ["reversible", "enzyme"])
def test_actual_gym_log_replays_native_interventions(tmp_path, kind):
    from chemworld.agent_interface import agent_view_bundle
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.eval.verify import verify_records

    e = ChemWorldEnv(composition=biochemical_world(kind))
    e.reset(seed=1)
    task = {**e.task_info(), **e.evaluator_provenance()}
    path = tmp_path / "trajectory.jsonl"
    actions = [
        {"operation": "add_component", "component": 0, "amount_mol": 5 / AVOGADRO},
        {"operation": "wait", "duration_s": 5},
        {"operation": "measure", "instrument": "molecule_counter"},
        {"operation": "measure", "instrument": "molecule_capture"},
        {"operation": "wait", "duration_s": 5},
    ]
    with TrajectoryLogger(path) as logger:
        for i, action in enumerate(actions, 1):
            obs, reward, terminated, truncated, info = e.step(action)
            assert info["transaction_status"] == "committed"
            logger.log(
                task_info=task,
                step=i,
                action=action,
                observation=observation_to_json(obs),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                agent_metadata={"name": "native-biochemical-test"},
                agent_view=agent_view_bundle(e, obs, info),
            )
    assert verify_records(load_jsonl(path)).verified


def test_synthetic_enzyme_explicit_three_channels_and_formulas():
    n = biochemical_network()
    assert len(n.reactions) == 3
    assert n.species_ids == ("E", "S", "ES", "P")
    assert n.stochastic["volume_L"] == 1e-15
    with pytest.raises(ValueError):
        biochemical_network(scale=0.5)
