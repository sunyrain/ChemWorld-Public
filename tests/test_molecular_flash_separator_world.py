"""Real three-tank local/shared flow, absent outlets, events and exact continuation."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("thermo", reason="optional independent reference package")

from chemworld.examples.molecular_flash_separator import molecular_flash_separator_example
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.molecular_flash_separator import separator_record, unpack_separator
from chemworld.world.molecular_flash_separator_world import build_molecular_flash_separator_world
from reference.molecular_flash_separator31_reference import steady


def physical(tank):
    return {
        "n": tank.species_amounts,
        "V": tank.volume_L,
        "T": tank.temperature_K,
        "P": tank.pressure_Pa,
        "phases": tank.phases.to_dict(),
        "slots": tank.state_slots.to_dict(),
    }


def setup(*, z=0.6, shared=True, power=100.0, event=None):
    spec, amounts, temperatures, _ = molecular_flash_separator_example(z=z)
    spec = replace(
        spec,
        separator=replace(
            spec.separator,
            device=replace(
                spec.separator.device, maximum_power_W=power, maximum_cooling_power_W=power
            ),
        ),
    )
    if event == "source":
        amounts = ({k: v * ((1e-5 + 0.002) / 0.02) for k, v in amounts[0].items()}, *amounts[1:])
    elif event == "receiver":
        rate = next(p["rate"] for p in steady(300.0, z)["phases"] if p["label"] == "gas")
        spec = replace(spec, receiver_capacities_mol=(0.02 + rate * 1.5, 0.1))
    elif event == "inactive_full":
        spec = replace(spec, receiver_capacities_mol=(0.02, 0.1))
    world = build_molecular_flash_separator_world(
        spec,
        amounts,
        temperatures,
        formulas={"propane": "C3H8", "butane": "C4H10"},
        shared_clock=shared,
    )
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=31)
    return runtime


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("z", [0.4, 0.6])
def test_real_single_and_two_phase_local_shared_paths(shared, z):
    runtime = setup(z=z, shared=shared)
    before = physical(runtime._state)
    result = runtime.execute({"operation": "wait" if shared else "run_flow", "duration_s": 1.0})
    assert result.info["transaction_status"] == "committed", result.info.get("error_message")
    assert runtime._state.ledger.time_s == pytest.approx(1.0)
    diagnostic = separator_record(runtime._state).settings["last_molecular_separator"]
    assert diagnostic["steady"]["diagnostics"]["outlet_phase_labels"] == (
        ["liquid"] if z == 0.4 else ["gas", "liquid"]
    )
    if z == 0.4:
        assert physical(runtime._state) == before
    else:
        assert runtime._state.species_amounts["propane"] > before["n"]["propane"]
    assert runtime._state.ledger.flow_pump_work_J == 0
    assert diagnostic["internal_energy_residual_J"] == pytest.approx(0, abs=2e-5)


def test_source_earliest_stop_and_nonreceiving_full_tank():
    runtime = setup(z=0.4, event="source")
    before = physical(runtime._state)
    result = runtime.execute({"operation": "wait", "duration_s": 3.0})
    assert result.info["transaction_status"] == "committed"
    assert result.info["operation_outcome"] == "stopped"
    assert runtime._state.ledger.time_s == pytest.approx(2.0, abs=2e-7)
    assert physical(runtime._state) == before
    full = setup(z=0.4, event="inactive_full")
    first = full.execute({"operation": "wait", "duration_s": 0.5})
    assert first.info["transaction_status"] == "committed"
    assert full._state.ledger.time_s == pytest.approx(0.5)


def test_receiver_event_and_exact_new_process_snapshot(tmp_path):
    runtime = setup(event="receiver")
    first = runtime.execute({"operation": "wait", "duration_s": 0.5})
    assert first.info["transaction_status"] == "committed"
    source, output = tmp_path / "snapshot.json", tmp_path / "continued.json"
    source.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
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
            str(Path(__file__).parent / "reference" / "molecular_flash_separator31_resume.py"),
            str(source),
            str(output),
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    fresh = json.loads(output.read_text(encoding="utf-8"))
    assert fresh["status"] == "committed" and fresh["outcome"] == "stopped"
    assert fresh["snapshot"] == json.loads(json.dumps(runtime.snapshot().to_dict()))


def test_power_rejection_preserves_three_physical_tanks():
    runtime = setup(power=1e-6)
    before = unpack_separator(runtime._state)[2]
    physical_before = {
        n: physical(s)
        for n, s in {
            runtime._state.vessel_id: runtime._state,
            **runtime._state.inactive_vessels,
        }.items()
    }
    outcome = runtime.execute({"operation": "wait", "duration_s": 1.0})
    assert outcome.info["operation_outcome"] == "numerical_failure"
    assert "insufficient_separator_heating_or_cooling_power" in str(
        outcome.info["outcome_diagnostics"]
    )
    assert unpack_separator(runtime._state)[2] == before
    assert physical_before == {
        n: physical(s)
        for n, s in {
            runtime._state.vessel_id: runtime._state,
            **runtime._state.inactive_vessels,
        }.items()
    }
    assert runtime._state.ledger.flow_heat_J == runtime._state.ledger.boundary_work_J == 0


def test_original_snapshot_validator_checks_current_group():
    runtime = setup()
    outcome = runtime.execute({"operation": "wait", "duration_s": 0.5})
    assert outcome.info["transaction_status"] == "committed"
    snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
    snapshot["world_state"]["inactive_vessels"]["source"]["temperature_K"] += 1.0
    with pytest.raises(ValueError, match=r"volume|ownership|inconsistent"):
        ChemWorldRuntime.from_snapshot(snapshot)
