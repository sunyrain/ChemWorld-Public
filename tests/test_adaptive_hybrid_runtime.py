"""One native inventory/transaction/clock for the adaptive hybrid variant."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.physchem import (
    AdaptiveHybridModel,
    AdaptiveHybridSpec,
    advance_adaptive_hybrid,
    initial_adaptive_continuation,
)
from chemworld.physchem.stochastic_adaptive_hybrid import (
    ADAPTIVE_HYBRID_ALGORITHM,
    MOLECULAR_QUANTUM,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.biochemical_worlds import biochemical_network, biochemical_world
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_adaptive_world import stochastic_adaptive_world


def composition(method="LSODA", **changes):
    original = biochemical_network("reversible")
    network = replace(
        original,
        stochastic=None,
        reactions=tuple(
            replace(r, rate_law=replace(r.rate_law, parameters={"k": k}))
            for r, k in zip(original.reactions, (1.0, 0.0), strict=True)
        ),
    )
    spec = AdaptiveHybridSpec(
        **{
            "volume_L": 1e-15,
            "method": method,
            "step_s": 0.25,
            "lower_limit": 45,
            "upper_limit": 45,
            **changes,
        }
    )
    return stochastic_adaptive_world(
        network, spec, {"A": 50.25 * MOLECULAR_QUANTUM, "B": 50.75 * MOLECULAR_QUANTUM}, seed=17
    )


def runtime(method="LSODA", noise="sequential", **changes):
    result = ChemWorldRuntime(
        composition=composition(method, **changes), observation_noise_mode=noise
    )
    result.reset(seed=7)
    return result


def committed(session, action):
    result = session.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.mark.parametrize("method", ["RK4", "LSODA"])
def test_real_wait_matches_pure_candidate_and_sole_phase(method):
    session = runtime(method)
    try:
        state = session._state
        physics = session.compiled_composition.spec.physics
        spec = physics.models_of("stochastic_hybrid")["reactor"]
        expected = advance_adaptive_hybrid(
            physics.network, spec, state.species_amounts, state.process.stochastic_continuation, 1
        )
        result = committed(session, {"operation": "wait", "duration_s": 1})
        changes = result.info["outcome_diagnostics"]["partition_changes"]
        assert json.loads(json.dumps(changes)) == changes
        after = session._state
        assert after.species_amounts == expected.amounts_mol
        assert after.process.stochastic_continuation == expected.continuation
        assert next(iter(after.phases.phases.values())).species_amounts_mol == after.species_amounts
        assert after.ledger.time_s == 1 and after.process.stochastic_continuation["time_s"] == 1
    finally:
        session.close()


@pytest.mark.parametrize("method", ["RK4", "LSODA"])
def test_continuous_dose_commits_real_material_without_integer_rounding(method):
    session = runtime(method)
    try:
        committed(session, {"operation": "wait", "duration_s": 0.375})
        before = session._state
        old = before.process.stochastic_continuation
        committed(
            session,
            {"operation": "add_component", "component": 0, "amount_mol": 0.5 * MOLECULAR_QUANTUM},
        )
        after = session._state
        assert after.species_amounts["A"] == before.species_amounts["A"] + 0.5 * MOLECULAR_QUANTUM
        assert after.ledger.time_s == before.ledger.time_s
        assert after.process.stochastic_continuation["rng_state"] == old["rng_state"]
        committed(session, {"operation": "wait", "duration_s": 0.625})
    finally:
        session.close()


@pytest.mark.parametrize("method", ["RK4", "LSODA"])
@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_same_gauge_has_independent_signed_detector_and_no_process_consumption(method, noise):
    session = runtime(method, noise)
    try:
        committed(session, {"operation": "wait", "duration_s": 0.375})
        before = session._state
        result = committed(session, {"operation": "measure", "instrument": "sde_amount_meter"})
        assert session._state.species_amounts == before.species_amounts
        assert (
            session._state.process.stochastic_continuation == before.process.stochastic_continuation
        )
        assert session._state.ledger.time_s == before.ledger.time_s
        assert session._state.ledger.cost == pytest.approx(before.ledger.cost + 0.01)
        assert all(channel.unit == "mol" for channel in result.observation.channels)
    finally:
        session.close()


@pytest.mark.parametrize("method", ["RK4", "LSODA"])
def test_full_json_and_new_process_after_real_dose(method, tmp_path):
    session = runtime(method, "keyed")
    try:
        for action in (
            {"operation": "wait", "duration_s": 0.375},
            {"operation": "add_component", "component": 0, "amount_mol": 0.5 * MOLECULAR_QUANTUM},
            {"operation": "measure", "instrument": "sde_amount_meter"},
        ):
            committed(session, action)
        payload = json.loads(json.dumps(session.snapshot().to_dict()))
        restored = ChemWorldRuntime.from_snapshot(payload)
        action = {"operation": "wait", "duration_s": 0.625}
        try:
            committed(session, action)
            committed(restored, action)
            expected = json.loads(json.dumps(session._state.to_dict()))
            assert json.loads(json.dumps(restored._state.to_dict())) == expected
        finally:
            restored.close()
        source, output = tmp_path / "checkpoint.json", tmp_path / "state.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        code = (
            "import json,sys;from pathlib import Path;"
            "from chemworld.runtime import ChemWorldRuntime;"
            "e=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')));"
            "r=e.execute({'operation':'wait','duration_s':.625});"
            "assert r.info['transaction_status']=='committed';"
            "Path(sys.argv[2]).write_text(json.dumps(e._state.to_dict()),encoding='utf-8');e.close()"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", code, str(source), str(output)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(output.read_text(encoding="utf-8")) == expected
    finally:
        session.close()


@pytest.mark.parametrize("scope", ["material", "vessel", "batch"])
def test_local_lifecycle_and_next_snapshot_remain_valid(scope):
    session = runtime()
    try:
        initial = session.compiled_composition.spec.physics.initial_states["reactor"]
        committed(session, {"operation": "wait", "duration_s": 0.375})
        committed(session, {"operation": "measure", "instrument": "sde_amount_meter"})
        before = session._state
        session.reset(options={"scope": scope})
        after = session._state
        assert after.species_amounts == initial.species_amounts
        if scope == "batch":
            assert after.ledger.time_s == 0
            assert after.process.stochastic_continuation == initial.process.stochastic_continuation
        else:
            assert after.ledger == before.ledger
            assert (
                after.process.stochastic_continuation["rng_state"]
                == before.process.stochastic_continuation["rng_state"]
            )
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        try:
            committed(session, {"operation": "wait", "duration_s": 0.125})
            committed(restored, {"operation": "wait", "duration_s": 0.125})
            assert session._state.to_dict() == restored._state.to_dict()
        finally:
            restored.close()
    finally:
        session.close()


def test_equipment_replacement_keeps_original_reaction_memory():
    session = runtime()
    try:
        committed(session, {"operation": "measure", "instrument": "sde_amount_meter"})
        committed(session, {"operation": "wait", "duration_s": 0.375})
        before = session._state
        meter = next(
            r
            for r in before.equipment.equipment.values()
            if r.settings.get("instrument_id") == "sde_amount_meter"
        )
        new = replace(
            meter, equipment_id="replacement-meter", settings={"instrument_id": "sde_amount_meter"}
        )
        session.reset(
            options={
                "scope": "equipment",
                "equipment_id": meter.equipment_id,
                "replacement": new.to_dict(),
            }
        )
        assert session._state.species_amounts == before.species_amounts
        assert (
            session._state.process.stochastic_continuation == before.process.stochastic_continuation
        )
    finally:
        session.close()


def test_budget_failure_never_changes_real_inventory_clock_or_process_stream():
    session = runtime(max_evaluations_per_advance=1)
    try:
        before = session._state
        result = session.execute({"operation": "wait", "duration_s": 1})
        assert result.info["transaction_status"] != "committed"
        assert session._state.species_amounts == before.species_amounts
        assert (
            session._state.process.stochastic_continuation == before.process.stochastic_continuation
        )
        assert session._state.ledger == before.ledger.with_updates(
            cost=before.ledger.cost + 0.03, risk=min(1.0, before.ledger.risk + 0.08)
        )
    finally:
        session.close()


def test_runtime_rate_work_counts_actual_callbacks_including_saved_validation(monkeypatch):
    session = runtime("RK4")
    actual = {"drift": 0, "propensity": 0}
    old_drift, old_rates = AdaptiveHybridModel.deterministic_drift, AdaptiveHybridModel.propensities

    def drift(self, *args):
        actual["drift"] += 1
        return old_drift(self, *args)

    def rates(self, *args):
        actual["propensity"] += len(self.channels)
        return old_rates(self, *args)

    monkeypatch.setattr(AdaptiveHybridModel, "deterministic_drift", drift)
    monkeypatch.setattr(AdaptiveHybridModel, "propensities", rates)
    try:
        for duration in (0.125, 0.25):
            actual.update(drift=0, propensity=0)
            result = committed(session, {"operation": "wait", "duration_s": duration})
            work = result.info["outcome_diagnostics"]["stochastic_work"]
            assert actual == {
                "drift": work["drift_evaluations"],
                "propensity": work["propensity_evaluations"],
            }
    finally:
        session.close()


def test_integer_ssa_cannot_forge_an_adaptive_marker():
    payload = biochemical_world("reversible").to_dict()
    payload["physics"]["initial_states"]["reactor"]["process"]["stochastic_continuation"][
        "algorithm"
    ] = ADAPTIVE_HYBRID_ALGORITHM
    with pytest.raises(ValueError, match="actual physical model"):
        ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))


@pytest.mark.parametrize("instrument", ["molecule_counter", "sde_aliquot_meter"])
def test_fixed_volume_adaptive_world_rejects_unbound_integer_or_consuming_meter(instrument):
    payload = composition().to_dict()
    next(c for c in payload["components"] if c["kind"] == "observation")["parameters"][
        "instruments"
    ] = [instrument]
    payload["task"]["instruments"] = [instrument]
    with pytest.raises(ValueError, match="adaptive"):
        ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))


def test_original_shared_clock_and_local_feed_restore_exactly():
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
            return {rename(k): rename(v) for k, v in value.items()}
        if isinstance(value, list):
            return [rename(v) for v in value]
        if isinstance(value, str) and (value == "reactor" or value.startswith("reactor:")):
            return "second" + value[len("reactor") :]
        return value

    second = rename(deepcopy(payload["physics"]["initial_states"]["reactor"]))
    spec = authored.physics.models_of("stochastic_hybrid")["reactor"]
    second["process"]["stochastic_continuation"] = initial_adaptive_continuation(
        authored.physics.network,
        spec,
        authored.physics.initial_states["reactor"].species_amounts,
        seed=18,
    )
    payload["physics"]["initial_states"]["second"] = second
    payload["physics"]["models"]["stochastic_hybrid"]["bindings"]["second"] = spec.to_dict()
    payload["physics"]["lab_clock"] = LabClockSpec().to_dict()
    session = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))
    session.reset(seed=7)
    try:
        committed(session, {"operation": "wait", "duration_s": 0.375})
        before = deepcopy(session._state.inactive_vessels["second"].to_dict())
        committed(
            session,
            {"operation": "add_component", "component": 0, "amount_mol": 0.5 * MOLECULAR_QUANTUM},
        )
        assert session._state.inactive_vessels["second"].to_dict() == before
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        try:
            for instance in (session, restored):
                committed(instance, {"operation": "wait", "duration_s": 0.625})
            assert session._state.to_dict() == restored._state.to_dict()
            assert (
                session._state.ledger.time_s
                == session._state.inactive_vessels["second"].ledger.time_s
                == 1
            )
        finally:
            restored.close()
    finally:
        session.close()
