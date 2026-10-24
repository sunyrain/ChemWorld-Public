"""Real original transport sessions, four owners, rollback and persisted continuation."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("thermo", reason="optional independent reference package")

from chemworld.examples.molecular_cascade import molecular_cascade_example
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.molecular_cascade import (
    MolecularCascadeAdapter,
    cascade_record,
    unpack_cascade,
)
from chemworld.world.molecular_cascade_world import build_molecular_cascade_world
from reference.molecular_cascade32_reference import steady


def setup(*, event=False, power=100, shared=True):
    spec, amounts, temperatures, _ = molecular_cascade_example()
    spec = replace(
        spec,
        cascade=replace(
            spec.cascade,
            devices=tuple(
                replace(d, maximum_power_W=power, maximum_cooling_power_W=power)
                for d in spec.cascade.devices
            ),
        ),
    )
    if event:
        flow = steady((300.0,) * 4, 0.75)["stages"][0]["rates"][0]
        spec = replace(spec, receiver_capacities_mol=(0.02 + 1.5 * flow, 0.1))
    world = build_molecular_cascade_world(
        spec,
        amounts,
        temperatures,
        formulas={"propane": "C3H8", "butane": "C4H10"},
        shared_clock=shared,
    )
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=32)
    return runtime, spec


@pytest.mark.parametrize("shared", [False, True])
def test_real_original_run_flow_and_shared_wait(shared):
    runtime, spec = setup(shared=shared)
    before = unpack_cascade(runtime._state)[2]
    result = runtime.execute({"operation": "wait" if shared else "run_flow", "duration_s": 1.0})
    assert result.info["transaction_status"] == "committed", result.info.get("error_message")
    assert result.info["operation_outcome"] == "success"
    after = unpack_cascade(runtime._state)[2]
    assert after.time_s == pytest.approx(1.0)
    assert after.amounts_mol[0]["propane"] < before.amounts_mol[0]["propane"]
    assert after.amounts_mol[2]["propane"] > before.amounts_mol[2]["propane"]
    assert runtime._state.ledger.flow_pump_work_J == 0
    assert runtime._state.ledger.boundary_work_J != 0
    assert cascade_record(runtime._state).settings["physical_model"] == spec.to_dict()
    denied = runtime.execute({"operation": "set_flow_rate", "flow_rate_mL_min": 1.0})
    assert denied.info["preconditions"]["operation_allowed_by_task"] is False


def test_earliest_event_and_fresh_process_exact_continuation(tmp_path):
    runtime, _ = setup(event=True)
    first = runtime.execute({"operation": "wait", "duration_s": 0.5})
    assert first.info["transaction_status"] == "committed"
    snapshot, output = tmp_path / "snapshot.json", tmp_path / "continued.json"
    snapshot.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
    second = runtime.execute({"operation": "wait", "duration_s": 2.0})
    assert second.info["transaction_status"] == "committed"
    assert second.info["operation_outcome"] == "stopped"
    assert runtime._state.ledger.time_s == pytest.approx(1.5, abs=2e-7)
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(Path(__file__).parent / "reference" / "molecular_cascade32_resume.py"),
            str(snapshot),
            str(output),
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    restored = json.loads(output.read_text(encoding="utf-8"))
    assert restored["status"] == "committed" and restored["outcome"] == "stopped"
    assert restored["snapshot"] == json.loads(json.dumps(runtime.snapshot().to_dict()))


def test_insufficient_device_power_rolls_back_all_physical_owners():
    runtime, _ = setup(power=1e-6)
    before = unpack_cascade(runtime._state)[2]
    result = runtime.execute({"operation": "wait", "duration_s": 1.0})
    assert result.info["transaction_status"] != "committed"
    assert result.info["operation_outcome"] == "numerical_failure"
    assert "insufficient_stage_heating_or_cooling_power" in str(result.info["outcome_diagnostics"])
    after = unpack_cascade(runtime._state)[2]
    assert before == after
    assert runtime._state.ledger.flow_heat_J == runtime._state.ledger.boundary_work_J == 0


def test_current_group_validator_does_not_substitute_initial_snapshot():
    runtime, _ = setup()
    result = runtime.execute({"operation": "wait", "duration_s": 0.5})
    assert result.info["transaction_status"] == "committed"
    current = runtime._state
    sources = dict(current.inactive_vessels)
    old = sources["gas_source"]
    sources["gas_source"] = old.replace(temperature_K=old.temperature_K + 1.0)
    with pytest.raises(ValueError, match=r"volume|ownership"):
        MolecularCascadeAdapter().validate_state(current.replace(inactive_vessels=sources))
    with pytest.raises(ValueError, match="fixed molar"):
        MolecularCascadeAdapter().configure_flow(current, {"flow_rate_mL_min": 1})
