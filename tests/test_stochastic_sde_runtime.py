"""Original world stocks, transactions, observations and saved SDE process memory."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.foundation.balances import TransitionBalance
from chemworld.foundation.discrete_inventory import AVOGADRO, integer_element_residual
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.physchem.stochastic_sde import initial_sde_continuation
from chemworld.physchem.stochastic_sde_specs import SDE_ALGORITHM, StochasticSDESpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.stochastic_sde import SDEStochasticRuntime
from chemworld.world.biochemical_worlds import biochemical_network, biochemical_world
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_sde_world import stochastic_sde_world


def composition(**changes):
    network = replace(biochemical_network("reversible"), stochastic=None)
    spec = StochasticSDESpec(**{"volume_L": 1e-15, "step_s": 0.125, **changes})
    return stochastic_sde_world(
        network, spec, {"A": 100.5 / AVOGADRO, "B": 20.25 / AVOGADRO}, seed=123
    )


def runtime(*, noise="sequential", **changes):
    instance = ChemWorldRuntime(composition=composition(**changes), observation_noise_mode=noise)
    instance.reset(seed=7)
    return instance


def committed(instance, action):
    result = instance.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.mark.parametrize("stop_flag", ["terminated", "quenched"])
def test_stopped_sde_control_returns_rejection_without_mutation(stop_flag):
    world = composition()
    state = world.physics.initial_states["reactor"].replace(**{stop_flag: True})
    before = state.to_dict()
    result = SDEStochasticRuntime(world).apply_parameters(
        state, {"operation": "set_sde_parameters", "values": {"unused": 1}},
    )
    assert not result.success and result.outcome == "rejected"
    assert result.failure_reason == "sde_control_stopped"
    assert result.actual_duration_s == 0 and result.state is None
    assert result.to_dict()["diagnostics"] == {}
    assert state.to_dict() == before


@pytest.mark.parametrize(
    "state_changes,action,options",
    [
        ({}, {"operation": "wait", "duration_s": 0.125}, {"heat": True}),
        ({}, {"operation": "wait", "duration_s": 0.125}, {"jacket_program": {}}),
        ({}, {"operation": "heat", "duration_s": 0.125}, {}),
        ({"terminated": True}, {"operation": "wait", "duration_s": 0.125}, {}),
        ({"quenched": True}, {"operation": "wait", "duration_s": 0.125}, {}),
    ],
)
def test_unsupported_sde_advance_returns_rejection_without_mutation(state_changes, action, options):
    world = composition()
    state = world.physics.initial_states["reactor"].replace(**state_changes)
    before = state.to_dict()
    result = SDEStochasticRuntime(world).advance(state, action, **options)
    assert not result.success and result.outcome == "rejected"
    assert result.failure_reason == "sde_fixed_boundary_only"
    assert result.actual_duration_s == 0 and result.state is None
    assert result.to_dict()["diagnostics"] == {}
    assert state.to_dict() == before


def test_real_world_wait_and_sole_phase_mol_inventory():
    instance = runtime()
    before = sum(instance._state.species_amounts.values())
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    state = instance._state
    assert state.ledger.time_s == state.process.stochastic_continuation["time_s"] == 0.0625
    assert state.process.stochastic_continuation["pending"] is not None
    assert next(iter(state.phases.phases.values())).species_amounts_mol == state.species_amounts
    assert sum(state.species_amounts.values()) == pytest.approx(before, rel=1e-14, abs=0)
    assert state.species_amounts["A"] * AVOGADRO != round(state.species_amounts["A"] * AVOGADRO)


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_snapshot_json_suffix_and_new_process_are_exact(tmp_path, noise):
    instance = runtime(noise=noise)
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    payload = json.loads(json.dumps(instance.snapshot().to_dict()))
    restored = ChemWorldRuntime.from_snapshot(payload)
    action = {"operation": "wait", "duration_s": 0.375}
    committed(instance, action)
    committed(restored, action)
    assert instance._state.to_dict() == restored._state.to_dict()
    source, output = tmp_path / "snapshot.json", tmp_path / "result.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    script = (
        "import json; from pathlib import Path; from chemworld.runtime import ChemWorldRuntime; "
        f"e=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
        f'r=e.execute({action!r}); assert r.info["transaction_status"]=="committed"; '
        f"Path({str(output)!r}).write_text(json.dumps(e._state.to_dict()))"
    )
    subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", script], check=True, cwd=tmp_path)
    assert json.loads(output.read_text()) == json.loads(json.dumps(instance._state.to_dict()))


def test_fractional_continuous_feed_invalidates_lookahead_preserves_rng():
    instance = runtime()
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    before = instance._state
    old_memory = deepcopy(before.process.stochastic_continuation)
    committed(
        instance, {"operation": "add_component", "component": 0, "amount_mol": 0.25 / AVOGADRO}
    )
    after = instance._state
    assert after.species_amounts["A"] - before.species_amounts["A"] == pytest.approx(
        0.25 / AVOGADRO, rel=1e-12, abs=0
    )
    memory = after.process.stochastic_continuation
    assert memory["pending"] is None and memory["invalidations"] == 1
    assert memory["rng_state"] == old_memory["rng_state"]
    assert memory["time_s"] == before.ledger.time_s
    committed(instance, {"operation": "wait", "duration_s": 0.25})


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_detector_is_independent_and_private_process_data_are_not_observations(noise):
    first, control = runtime(noise=noise), runtime(noise=noise)
    action = {"operation": "wait", "duration_s": 0.0625}
    committed(first, action)
    committed(control, action)
    memory = deepcopy(first._state.process.stochastic_continuation)
    result = committed(first, {"operation": "measure", "instrument": "sde_amount_meter"})
    assert result.observation.channels
    assert all(channel.unit == "mol" for channel in result.observation.channels)
    assert first._state.process.stochastic_continuation == memory
    assert first._state.ledger.time_s == control._state.ledger.time_s
    assert "rng_state" not in json.dumps(result.observation.to_dict())
    assert "stochastic_continuation" not in first._state.to_dict(include_hidden=False)["process"]
    committed(first, {"operation": "wait", "duration_s": 0.375})
    committed(control, {"operation": "wait", "duration_s": 0.375})
    assert first._state.species_amounts == control._state.species_amounts
    assert (
        first._state.process.stochastic_continuation
        == control._state.process.stochastic_continuation
    )


def test_budget_failure_uses_original_transaction_rollback():
    instance = runtime(max_steps_per_advance=1)
    before = instance._state.to_dict()
    result = instance.execute({"operation": "wait", "duration_s": 0.5})
    assert result.info["transaction_status"] == "rolled_back"
    assert result.info["operation_outcome"] == "numerical_failure"
    assert result.info["outcome_diagnostics"]["failure_reason"] == "sde_advance_failed"
    assert instance._state.species_amounts == before["species_amounts"]
    assert (
        instance._state.process.stochastic_continuation
        == before["process"]["stochastic_continuation"]
    )
    assert instance._state.ledger.time_s == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("algorithm", "gillespie-direct-v1"),
        ("pending", None),
        ("model_signature", None),
        ("inventory_signature", None),
        ("time_s", 0),
        ("unknown", 0),
    ],
)
def test_original_restore_checks_actual_model_and_numerical_memory(field, value):
    instance = runtime()
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    payload = json.loads(json.dumps(instance.snapshot().to_dict()))
    payload["world_state"]["process"]["stochastic_continuation"][field] = value
    with pytest.raises(ValueError):
        ChemWorldRuntime.from_snapshot(payload)


def test_integer_ssa_cannot_claim_sde_marker_to_bypass_actual_model_binding():
    payload = biochemical_world("reversible").to_dict()
    memory = payload["physics"]["initial_states"]["reactor"]["process"]["stochastic_continuation"]
    memory["algorithm"] = SDE_ALGORITHM
    with pytest.raises(ValueError, match="actual physical model"):
        ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))


@pytest.mark.parametrize("instrument", ["molecule_counter", "molecule_capture", "hplc"])
def test_unsupported_integer_or_consuming_instrument_rejected_at_authoring(instrument):
    payload = composition().to_dict()
    observation = next(c for c in payload["components"] if c["kind"] == "observation")
    observation["parameters"]["instruments"] = [instrument]
    payload["task"]["instruments"] = [instrument]
    with pytest.raises(ValueError, match="SDE"):
        ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))


def test_tiny_continuous_atom_loss_is_rejected_without_rounding_to_integer_counts():
    instance = runtime()
    state = instance._state
    changed = state.replace(
        species_amounts={**state.species_amounts, "A": state.species_amounts["A"] - 0.25 / AVOGADRO}
    )
    substances = instance.constitution.substances
    with pytest.raises(ValueError, match="SDE atom inventory"):
        integer_element_residual(state, changed, TransitionBalance(), substances)


def test_original_shared_clock_advances_two_sde_inventories_and_restores_exactly():
    authored = composition()
    payload = authored.to_dict()
    payload["vessels"].append({"id": "second", "capacity_L": 1e-15})
    for component in payload["components"]:
        component["vessel"] = "reactor"
    for component in tuple(payload["components"]):
        duplicate = deepcopy(component)
        duplicate.update(id="second:" + component["id"], vessel="second")
        payload["components"].append(duplicate)

    def rename(value):
        if isinstance(value, dict):
            return {rename(key): rename(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, str) and (value == "reactor" or value.startswith("reactor:")):
            return "second" + value[len("reactor") :]
        return value

    second = rename(deepcopy(payload["physics"]["initial_states"]["reactor"]))
    spec = authored.physics.models_of("stochastic_sde")["reactor"]
    second["process"]["stochastic_continuation"] = initial_sde_continuation(
        authored.physics.network,
        spec,
        authored.physics.initial_states["reactor"].species_amounts,
        seed=124,
    )
    payload["physics"]["initial_states"]["second"] = second
    payload["physics"]["models"]["stochastic_sde"]["bindings"]["second"] = spec.to_dict()
    payload["physics"]["lab_clock"] = LabClockSpec().to_dict()
    instance = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))
    instance.reset(seed=7)
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    state = instance._state
    other = state.inactive_vessels["second"]
    assert state.ledger.time_s == other.process.stochastic_continuation["time_s"] == 0.0625
    assert state.species_amounts != other.species_amounts
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(instance.snapshot().to_dict())))
    other_before = other.to_dict()
    for action in (
        {"operation": "add_component", "component": 0, "amount_mol": 0.25 / AVOGADRO},
        {"operation": "measure", "instrument": "sde_amount_meter"},
        {"operation": "wait", "duration_s": 0.375},
    ):
        committed(instance, action)
        committed(restored, action)
        if action["operation"] != "wait":
            assert instance._state.inactive_vessels["second"].to_dict() == other_before
    assert instance._state.to_dict() == restored._state.to_dict()
