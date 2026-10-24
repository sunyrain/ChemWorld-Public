"""Existing process physics now reaches the actual session boundary and replay."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from chemworld.physchem.molecular_processes import MolecularHeatDuty
from chemworld.runtime import ChemWorldRuntime, MolecularProcessReceiverSpec
from chemworld.runtime.kernel_contracts import ModelProviderResult
from chemworld.runtime.molecular_process_path import MolecularProcessPath
from chemworld.runtime.molecular_process_runtime import MolecularProcessRuntime
from chemworld.runtime.transactions import StatePatch
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.molecular_process_world import build_molecular_process_world
from reference.molecular_recycle_cases import path, receiver


def declared(kind="recycle", *, maximum=0.3, pressure_cycle=False):
    original = path(real=True, pressure_cycle=pressure_cycle)
    original = replace(
        original,
        inlet=replace(original.inlet, molar_rate_mol_s=0.001),
        devices=tuple(
            replace(d, heat_duty_W=0.1) if isinstance(d, MolecularHeatDuty) else d
            for d in original.devices
        ),
    )
    process = (
        original
        if kind == "recycle"
        else MolecularProcessPath(
            original.domain,
            original.source_id,
            original.inlet,
            original.devices,
        )
    )
    initial = receiver(process).replace(
        units={
            "amount": "mol",
            "volume": "L",
            "temperature": "K",
            "pressure": "Pa",
            "time": "s",
            "cost": "currency",
            "risk": "risk",
        }
    )
    spec = MolecularProcessReceiverSpec(process, "bulk", maximum)
    return build_molecular_process_world(
        spec,
        initial,
        formulas={"A": "Ar", "B": "C3H8"},
    )


def physical(state):
    return {
        "amounts": state.species_amounts,
        "phases": state.phases.to_dict(),
        "T": state.temperature_K,
        "P": state.pressure_Pa,
        "time": state.ledger.time_s,
        "Q": state.ledger.flow_heat_J,
        "W": state.ledger.flow_shaft_work_J,
        "jacket": state.ledger.energy_jacket_J,
    }


@pytest.mark.parametrize("kind", ("serial", "recycle"))
@pytest.mark.parametrize("cycle", (False, True))
def test_native_wait_admits_only_net_feed_and_settles_upstream_utilities_once(kind, cycle):
    world = declared(kind, pressure_cycle=cycle)
    spec = world.physics.models_of("molecular_process")["receiver"]
    expected = spec.path.evaluate()
    runtime = ChemWorldRuntime(composition=world)
    try:
        runtime.reset(seed=7)
        before = runtime._state
        result = runtime.execute({"operation": "wait", "duration_s": 1})
        assert result.info["transaction_status"] == "committed", result.info
        state = runtime._state
        assert state.species_amounts == pytest.approx({"A": 0.01275, "B": 0.03825})
        assert state.phases.phases["bulk"].species_amounts_mol == state.species_amounts
        assert state.ledger.time_s == 1
        assert state.vessel_elapsed_s["receiver"] == 1
        assert state.ledger.flow_heat_J == pytest.approx(expected.heat_duty_W)
        assert state.ledger.flow_shaft_work_J == pytest.approx(expected.shaft_work_W)
        assert state.ledger.energy_jacket_J == 0
        balance = MolecularProcessRuntime(world).material_boundary(
            before,
            state,
            {"operation": "wait", "duration_s": 1},
        )
        assert dict(balance.exchanges[0].amounts) == pytest.approx({"A": 0.00025, "B": 0.00075})
        assert result.info["outcome_diagnostics"]["receiver_energy_residual_J"] == pytest.approx(
            0,
            abs=1e-6,
        )
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", ("serial", "recycle"))
def test_real_inventory_event_books_only_actual_time(kind):
    runtime = ChemWorldRuntime(composition=declared(kind, maximum=0.0515))
    try:
        runtime.reset(seed=7)
        result = runtime.execute({"operation": "wait", "duration_s": 3})
        assert result.info["transaction_status"] == "committed", result.info
        assert result.info["operation_outcome"] == "stopped"
        assert runtime._state.ledger.time_s == pytest.approx(1.5, abs=1e-7)
        assert sum(runtime._state.species_amounts.values()) == pytest.approx(0.0515, abs=1e-10)
        assert runtime._state.ledger.flow_heat_J == pytest.approx(0.15, abs=1e-8)
    finally:
        runtime.close()


@pytest.mark.parametrize("kind", ("serial", "recycle"))
def test_complete_snapshot_continues_in_new_locked_process(tmp_path, kind):
    runtime = ChemWorldRuntime(composition=declared(kind))
    try:
        runtime.reset(seed=7)
        first = runtime.execute({"operation": "wait", "duration_s": 1})
        assert first.info["transaction_status"] == "committed"
        snapshot, output = tmp_path / "snapshot.json", tmp_path / "continued.json"
        runtime.snapshot().save(snapshot)
        second = runtime.execute({"operation": "wait", "duration_s": 1})
        assert second.info["transaction_status"] == "committed"
        code = (
            "import json,sys; from pathlib import Path; "
            "from chemworld.runtime import ChemWorldRuntime; "
            "r=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text())); "
            "o=r.execute({'operation':'wait','duration_s':1}); "
            "Path(sys.argv[2]).write_text(json.dumps({'status':o.info['transaction_status'],"
            "'state':r._state.to_dict(include_hidden=True)})); r.close()"
        )
        subprocess.run(
            [sys.executable, "-c", code, str(snapshot), str(output)],
            env={**os.environ},
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(output.read_text())
        assert data["status"] == "committed"
        assert data["state"] == json.loads(json.dumps(runtime._state.to_dict(include_hidden=True)))
    finally:
        runtime.close()


def test_solver_failure_rolls_back_physics_and_utilities(monkeypatch):
    runtime = ChemWorldRuntime(composition=declared())
    try:
        runtime.reset(seed=7)
        before = physical(runtime._state)
        monkeypatch.setattr(
            "chemworld.runtime.molecular_process_runtime.advance_molecular_process_receiver",
            lambda *a, **k: ModelProviderResult(
                {},
                {},
                success=False,
                failure_reason="injected_failure",
                outcome="numerical_failure",
            ),
        )
        result = runtime.execute({"operation": "wait", "duration_s": 1})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert physical(runtime._state) == before
    finally:
        runtime.close()


def test_external_receipt_is_authenticated_against_feed_not_inventory_deficit():
    world = declared()
    runner = MolecularProcessRuntime(world)
    before = world.physics.initial_states["receiver"]
    result = runner.advance(before, {"duration_s": 1})
    assert result.success
    metadata = json.loads(json.dumps(result.state.metadata))
    metadata["last_molecular_process_execution"]["exchanges"][0]["amount"] *= 2
    corrupted = result.state.replace(metadata=metadata)
    with pytest.raises(ValueError, match="metered feed"):
        runner.material_boundary(before, corrupted, {"operation": "wait", "duration_s": 1})
    with pytest.raises(ValueError, match="interval"):
        runner.material_boundary(before, result.state, {"operation": "wait", "duration_s": 0.1})


def test_public_codec_and_boundary_scope_are_explicit():
    world = declared()
    restored = WorldCompositionSpec.from_dict(json.loads(json.dumps(world.to_dict())))
    assert restored.to_dict() == world.to_dict()
    runner = MolecularProcessRuntime(restored)
    initial = restored.physics.initial_states["receiver"]
    assert not runner.advance(
        initial, {"duration_s": 1, "target_temperature_K": 400}, heat=True
    ).success
    payload = next(iter(runner.models.values())).to_dict()
    payload["path"]["model"] = "unregistered-arbitrary-path"
    with pytest.raises(ValueError, match="unsupported"):
        MolecularProcessReceiverSpec.from_dict(payload)
    runtime = ChemWorldRuntime(composition=restored)
    try:
        runtime.reset(seed=7)
        before = physical(runtime._state)
        denied = runtime.execute(
            {"operation": "heat", "duration_s": 1, "target_temperature_K": 400}
        )
        assert denied.info["operation_outcome"] == "rejected"
        assert physical(runtime._state) == before
    finally:
        runtime.close()


def test_material_reset_preserves_actual_resource_history_then_continues():
    runtime = ChemWorldRuntime(composition=declared())
    try:
        runtime.reset(seed=7)
        initial = dict(runtime._state.species_amounts)
        assert (
            runtime.execute({"operation": "wait", "duration_s": 1}).info["transaction_status"]
            == "committed"
        )
        ledger, operation = runtime._state.ledger, runtime._operation_id
        runtime.reset(options={"scope": "material", "vessel_ids": ["receiver"]})
        assert runtime._state.species_amounts == initial
        assert runtime._state.ledger == ledger
        assert runtime._operation_id == operation
        assert (
            runtime.execute({"operation": "wait", "duration_s": 1}).info["transaction_status"]
            == "committed"
        )
        assert runtime._state.ledger.flow_heat_J == pytest.approx(0.2)
    finally:
        runtime.close()


def test_valid_receipt_cannot_authorize_additional_candidate_matter():
    runtime = ChemWorldRuntime(composition=declared())
    try:
        runtime.reset(seed=7)
        before = runtime._state
        runner = MolecularProcessRuntime(runtime.compiled_composition.spec)
        candidate = runner.advance(before, {"duration_s": 1}).state
        amounts = {k: n + 0.001 for k, n in candidate.species_amounts.items()}
        phase = replace(candidate.phases.phases["bulk"], species_amounts_mol=amounts)
        candidate = candidate.replace(
            species_amounts=amounts,
            phases=replace(candidate.phases, phases={"bulk": phase}),
        )
        result = runtime.runtime.transaction_manager.commit(
            state=before,
            operation_type="wait",
            events=(),
            patches=(StatePatch("replace_state", ("species", "phases"), candidate),),
            action={"operation": "wait", "duration_s": 1},
            actual_duration_s=1,
        )
        assert result.transaction_status == "rolled_back"
        assert any(
            c["name"] == "material_conservation" and not c["passed"]
            for c in result.constitution_checks
        )
    finally:
        runtime.close()
