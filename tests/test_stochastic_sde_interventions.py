"""Actual bounded controls, finite liquid withdrawal and saved process memory."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from chemworld.foundation.dimensions import DimensionVector
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.physchem import (
    ReactionFlux,
    ReactionNoise,
    ReactionSDE,
    SDEControl,
    SDEParameter,
    StochasticSDESpec,
    advance_sde,
    initial_sde_continuation,
    rebind_sde_continuation,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.biochemical_worlds import biochemical_network, biochemical_world
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_sde_instruments import SDE_ALIQUOT_VOLUME_L
from chemworld.world.stochastic_sde_world import stochastic_sde_world


def math(body):
    return '<math xmlns="http://www.w3.org/1998/Math/MathML">' + body + "</math>"


def control_spec(**kwargs):
    return StochasticSDESpec(
        **{
            "volume_L": 1e-15,
            "step_s": 0.125,
            "minimum_volume_L": 0.75e-15,
            "controls": (
                SDEControl("forward_scale", "forward", "rate_multiplier", 0, 4),
                SDEControl("reverse_scale", "reverse", "rate_multiplier", 0, 4),
            ),
            **kwargs,
        }
    )


def session(*, noise="sequential", **kwargs):
    network = replace(biochemical_network("reversible"), stochastic=None)
    spec = control_spec(**kwargs)
    world = stochastic_sde_world(
        network, spec, {"A": 100.5 / AVOGADRO, "B": 20.25 / AVOGADRO}, seed=123
    )
    result = ChemWorldRuntime(composition=world, observation_noise_mode=noise)
    result.reset(seed=7)
    return result


def committed(instance, action):
    result = instance.execute(action)
    assert result.info["transaction_status"] == "committed", {
        key: result.info.get(key)
        for key in ("transaction_status", "error_message", "outcome_diagnostics", "preconditions")
    }
    return result


def test_flux_and_noise_parameter_controls_reach_same_physical_callback():
    network = replace(biochemical_network("reversible"), stochastic=None)
    flux = ReactionFlux(
        "forward",
        math("<apply><divide/><ci>n0</ci><ci>tau</ci></apply>"),
        (SDEParameter("tau", 2, DimensionVector(time=1)),),
    )
    noise = ReactionNoise(
        "forward", math("<ci>sigma</ci>"), (SDEParameter("sigma", 1, DimensionVector(time=-0.5)),)
    )
    spec = control_spec(
        fluxes=(flux,),
        noises=(noise,),
        controls=(
            SDEControl("speed", "forward", "rate_multiplier", 0, 4),
            SDEControl("tau_input", "forward", "flux_parameter", 1, 8, "tau"),
            SDEControl("sigma_input", "forward", "noise_parameter", 0, 4, "sigma"),
        ),
    )
    model = ReactionSDE(
        network, spec, control_values={"speed": 1, "tau_input": 4, "sigma_input": 3}
    )
    np.testing.assert_allclose(model.drift(0, [100, 20]), [-21, 21])
    np.testing.assert_allclose(model.diffusion(0, [100, 20]), [[-3], [3]])
    assert model.control_metadata["tau_input"]["dimension"]["time"] == 1
    restored = StochasticSDESpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec


def test_saved_partial_step_is_invalidated_by_chemical_control_without_reseeding():
    instance = session()
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    before = instance._state
    saved = deepcopy(before.process.stochastic_continuation)
    assert saved["pending"] is not None
    committed(
        instance,
        {"operation": "set_sde_parameters", "values": {"forward_scale": 0, "reverse_scale": 0}},
    )
    controlled = instance._state
    memory = controlled.process.stochastic_continuation
    assert memory["pending"] is None and memory["invalidations"] == saved["invalidations"] + 1
    assert memory["rng_state"] == saved["rng_state"]
    assert controlled.species_amounts == before.species_amounts
    assert controlled.ledger.time_s == before.ledger.time_s
    committed(instance, {"operation": "wait", "duration_s": 0.5})
    for key, value in before.species_amounts.items():
        assert instance._state.species_amounts[key] == pytest.approx(value, rel=1e-15, abs=0)


@pytest.mark.parametrize(
    "values",
    [
        {"forward_scale": 2, "reverse_scale": 5},
        {"missing": 1},
        {"forward_scale": True},
        {"forward_scale": float("nan")},
        {},
    ],
)
def test_invalid_multi_control_never_changes_any_live_parameter_or_process_draw(values):
    instance = session()
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    before = instance._state
    result = instance.execute({"operation": "set_sde_parameters", "values": values})
    assert result.info["transaction_status"] != "committed"
    assert instance._state.species_amounts == before.species_amounts
    assert instance._state.process.stochastic_continuation == before.process.stochastic_continuation
    assert instance._state.ledger.time_s == before.ledger.time_s


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_true_liquid_aliquot_debits_mol_volume_receipts_and_lookahead(noise):
    instance = session(noise=noise)
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    before = instance._state
    old = deepcopy(before.process.stochastic_continuation)
    result = committed(instance, {"operation": "measure", "instrument": "sde_aliquot_meter"})
    after = instance._state
    fraction = SDE_ALIQUOT_VOLUME_L / before.volume_L
    assert after.volume_L == pytest.approx(before.volume_L * (1 - fraction), rel=1e-15, abs=0)
    for key in before.species_amounts:
        assert after.species_amounts[key] == pytest.approx(
            before.species_amounts[key] * (1 - fraction), rel=1e-15, abs=0
        )
    memory = after.process.stochastic_continuation
    assert memory["volume_L"] == after.volume_L
    assert memory["pending"] is None and memory["invalidations"] == old["invalidations"] + 1
    assert memory["rng_state"] == old["rng_state"]
    assert memory["control_values"] == old["control_values"]
    assert after.ledger.time_s == before.ledger.time_s
    assert after.ledger.analysis_time_s - before.ledger.analysis_time_s == 1
    assert after.ledger.sample_consumed_L - before.ledger.sample_consumed_L == SDE_ALIQUOT_VOLUME_L
    assert after.ledger.cost - before.ledger.cost == pytest.approx(0.01)
    assert all(
        c.channel_id.startswith("sample_amount_") and c.unit == "mol"
        for c in result.observation.channels
    )
    committed(
        instance, {"operation": "add_component", "component": 0, "amount_mol": 0.25 / AVOGADRO}
    )
    committed(instance, {"operation": "wait", "duration_s": 0.125})


def test_sampling_floor_refuses_overwithdrawal_and_preserves_remaining_stock():
    instance = session()
    action = {"operation": "measure", "instrument": "sde_aliquot_meter"}
    committed(instance, action)
    committed(instance, action)
    before = instance._state
    result = instance.execute(action)
    assert result.info["transaction_status"] != "committed"
    assert result.info["preconditions"]["sde_sampling_volume_floor"] is False
    assert "sde_sampling_volume_floor" in result.info["error_message"]
    assert "sde_aliquot_meter" not in instance.action_schema("measure")["fields"][0]["choices"]
    assert instance._state.volume_L == before.volume_L
    assert instance._state.species_amounts == before.species_amounts
    assert instance._state.process.stochastic_continuation == before.process.stochastic_continuation


def test_changed_volume_and_controls_continue_exactly_in_new_process(tmp_path):
    instance = session(noise="keyed")
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    committed(instance, {"operation": "set_sde_parameters", "values": {"forward_scale": 0.5}})
    committed(instance, {"operation": "measure", "instrument": "sde_aliquot_meter"})
    payload = json.loads(json.dumps(instance.snapshot().to_dict()))
    restored = ChemWorldRuntime.from_snapshot(payload)
    action = {"operation": "wait", "duration_s": 0.375}
    committed(instance, action)
    committed(restored, action)
    assert instance._state.to_dict() == restored._state.to_dict()
    source, output = tmp_path / "checkpoint.json", tmp_path / "final.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    code = (
        "import json;from pathlib import Path;from chemworld.runtime import ChemWorldRuntime;"
        f"e=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text()));"
        f'r=e.execute({action!r});assert r.info["transaction_status"]=="committed";'
        f"Path({str(output)!r}).write_text(json.dumps(e._state.to_dict()))"
    )
    subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", code], check=True, cwd=tmp_path)
    assert json.loads(output.read_text()) == json.loads(json.dumps(instance._state.to_dict()))


def test_integer_ssa_has_no_sde_control_operation():
    instance = ChemWorldRuntime(composition=biochemical_world("reversible"))
    instance.reset(seed=7)
    assert "set_sde_parameters" not in instance.operation_validator.valid_operations(
        instance._state
    )


def test_control_and_geometry_rebind_are_pure_and_invalid_changes_are_atomic():
    network = replace(biochemical_network("reversible"), stochastic=None)
    spec = control_spec()
    amounts = {"A": 100 / AVOGADRO, "B": 20 / AVOGADRO}
    memory = initial_sde_continuation(network, spec, amounts, seed=123)
    first = advance_sde(network, spec, amounts, memory, 0.0625)
    before = deepcopy(first.continuation)
    with pytest.raises(ValueError):
        rebind_sde_continuation(
            network,
            spec,
            first.amounts_mol,
            first.amounts_mol,
            first.continuation,
            control_values={"forward_scale": 9},
        )
    assert first.continuation == before
    reduced = {key: value * 0.9 for key, value in first.amounts_mol.items()}
    changed = rebind_sde_continuation(
        network, spec, first.amounts_mol, reduced, before, volume_L=9e-16
    )
    assert changed["rng_state"] == before["rng_state"]
    assert changed["volume_L"] == 9e-16
    advance_sde(network, spec, reduced, changed, 0.125)


def test_public_control_metadata_is_bounded_and_reads_do_not_change_process():
    instance = session()
    before = instance.snapshot().to_dict()
    field = instance.action_schema("set_sde_parameters")["fields"][0]
    assert field["additionalProperties"] is False and field["partial_updates"] is True
    assert field["properties"]["forward_scale"]["minimum"] == 0
    assert field["properties"]["forward_scale"]["maximum"] == 4
    assert field["properties"]["forward_scale"]["current_value"] == 1
    assert instance.snapshot().to_dict() == before


@pytest.mark.parametrize("scope", ["material", "vessel"])
@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_partial_initialization_replaces_real_material_retains_stream_controls_and_clock(
    scope, noise
):
    instance = session(noise=noise)
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    committed(instance, {"operation": "set_sde_parameters", "values": {"forward_scale": 0.5}})
    committed(instance, {"operation": "measure", "instrument": "sde_aliquot_meter"})
    committed(instance, {"operation": "wait", "duration_s": 0.03125})
    before = instance._state
    memory = deepcopy(before.process.stochastic_continuation)
    assert memory["pending"] is not None
    instance.reset(options={"scope": scope})
    after = instance._state
    initial = instance.compiled_composition.spec.physics.initial_states["reactor"]
    assert after.species_amounts == initial.species_amounts
    assert after.volume_L == initial.volume_L
    assert after.ledger == before.ledger
    assert after.process.stochastic_continuation["rng_state"] == memory["rng_state"]
    assert after.process.stochastic_continuation["control_values"] == memory["control_values"]
    assert after.process.stochastic_continuation["pending"] is None
    assert after.process.stochastic_continuation["invalidations"] == memory["invalidations"] + 1
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(instance.snapshot().to_dict())))
    action = {"operation": "wait", "duration_s": 0.25}
    committed(instance, action)
    committed(restored, action)
    assert instance._state.to_dict() == restored._state.to_dict()


def test_batch_reinitializes_declared_seed_and_controls_with_original_device_usage():
    instance = session()
    initial = deepcopy(instance._state.process.stochastic_continuation)
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    committed(instance, {"operation": "set_sde_parameters", "values": {"forward_scale": 0.5}})
    committed(instance, {"operation": "measure", "instrument": "sde_amount_meter"})
    instance.reset(options={"scope": "batch"})
    assert instance._state.ledger.time_s == 0
    assert instance._state.process.stochastic_continuation == initial
    meter = next(
        r
        for r in instance._state.equipment.equipment.values()
        if r.settings.get("instrument_id") == "sde_amount_meter"
    )
    assert meter.settings["use_count"] == 1


def test_instrument_replacement_preserves_pending_reaction_and_stock():
    instance = session()
    committed(instance, {"operation": "measure", "instrument": "sde_amount_meter"})
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    before = instance._state
    meter = next(
        r
        for r in before.equipment.equipment.values()
        if r.settings.get("instrument_id") == "sde_amount_meter"
    )
    replacement = replace(
        meter, equipment_id="replacement-meter", settings={"instrument_id": "sde_amount_meter"}
    )
    instance.reset(
        options={
            "scope": "equipment",
            "equipment_id": meter.equipment_id,
            "replacement": replacement.to_dict(),
        }
    )
    after = instance._state
    assert after.species_amounts == before.species_amounts
    assert after.process.stochastic_continuation == before.process.stochastic_continuation
    assert after.ledger == before.ledger
    assert meter.equipment_id not in after.equipment.equipment
    committed(instance, {"operation": "measure", "instrument": "sde_amount_meter"})


def shared_composition():
    initial = session().compiled_composition.spec
    payload = initial.to_dict()
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
    spec = initial.physics.models_of("stochastic_sde")["reactor"]
    second["process"]["stochastic_continuation"] = initial_sde_continuation(
        initial.physics.network,
        spec,
        initial.physics.initial_states["reactor"].species_amounts,
        seed=124,
    )
    payload["physics"]["initial_states"]["second"] = second
    payload["physics"]["models"]["stochastic_sde"]["bindings"]["second"] = spec.to_dict()
    payload["physics"]["lab_clock"] = LabClockSpec().to_dict()
    return WorldCompositionSpec.from_dict(payload)


def test_shared_sampling_control_and_one_vessel_reset_preserve_the_other_process():
    instance = ChemWorldRuntime(composition=shared_composition())
    instance.reset(seed=7)
    committed(instance, {"operation": "wait", "duration_s": 0.0625})
    other = instance._state.inactive_vessels["second"].to_dict()
    committed(instance, {"operation": "set_sde_parameters", "values": {"forward_scale": 0.5}})
    committed(instance, {"operation": "measure", "instrument": "sde_aliquot_meter"})
    assert instance._state.inactive_vessels["second"].to_dict() == other
    instance.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert instance._state.inactive_vessels["second"].to_dict() == other
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(instance.snapshot().to_dict())))
    action = {"operation": "wait", "duration_s": 0.25}
    committed(instance, action)
    committed(restored, action)
    assert instance._state.to_dict() == restored._state.to_dict()


@pytest.mark.parametrize("entry", ["native", "gym"])
@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_logged_intervention_and_material_boundary_replay_exactly(entry, noise, tmp_path):
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.envs.spaces import to_observation
    from chemworld.eval.verify import verify_records

    source = session(noise=noise)
    spec = source.compiled_composition.spec
    source.close()
    cls = ChemWorldRuntime if entry == "native" else ChemWorldEnv
    instance = cls(composition=spec, observation_noise_mode=noise)
    instance.reset(seed=7)
    actions = [
        {"operation": "wait", "duration_s": 0.0625},
        {"operation": "set_sde_parameters", "values": {"forward_scale": 0.5}},
        {"operation": "measure", "instrument": "sde_aliquot_meter"},
        {"operation": "wait", "duration_s": 0.03125},
        {"operation": "measure", "instrument": "sde_amount_meter"},
        {"operation": "wait", "duration_s": 0.125},
        {"operation": "set_sde_parameters", "values": {"forward_scale": 0}},
        {"operation": "measure", "instrument": "sde_aliquot_meter"},
        {"operation": "wait", "duration_s": 0.125},
        {"operation": "terminate"},
    ]
    path = tmp_path / "interventions.jsonl"
    try:
        with TrajectoryLogger(path) as logger:
            for index, action in enumerate(actions, 1):
                if index == 6:
                    instance.reset(options={"scope": "material"})
                if entry == "native":
                    result = committed(instance, action)
                    observation = to_observation(result.observation.values)
                    reward, terminated, truncated, info = (
                        result.reward,
                        result.terminated,
                        result.truncated,
                        result.info,
                    )
                else:
                    observation, reward, terminated, truncated, info = instance.step(action)
                    assert info["transaction_status"] == "committed", info.get(
                        "outcome_diagnostics"
                    )
                logger.log(
                    task_info={**instance.task_info(), **instance.evaluator_provenance()},
                    step=index,
                    action=action,
                    observation=observation_to_json(observation),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    agent_metadata={"name": "sde-intervention-functional"},
                )
        records = load_jsonl(path)
        verified = verify_records(records, tolerance=0)
        assert verified.verified, verified.mismatches
        assert verified.checked_steps == len(actions)
        assert len(records[5]["initialization_replay"]) == 1
        assert "sample_amount_A" in path.read_text()
    finally:
        instance.close()


@pytest.mark.parametrize(
    "declaration",
    [
        SDEControl("missing", "absent", "rate_multiplier", 0, 4),
        SDEControl("excludes_initial", "forward", "rate_multiplier", 2, 4),
        SDEControl("unknown_parameter", "forward", "flux_parameter", 0, 4, "unknown"),
    ],
)
def test_control_binding_rejects_unknown_channel_parameter_or_initial_value(declaration):
    network = replace(biochemical_network("reversible"), stochastic=None)
    with pytest.raises(ValueError):
        ReactionSDE(network, control_spec(controls=(declaration,)))


def test_control_declaration_rejects_unknown_fields_and_duplicate_physical_targets():
    declaration = SDEControl("one", "forward", "rate_multiplier", 0, 4)
    with pytest.raises(ValueError):
        SDEControl.from_dict({**declaration.to_dict(), "unknown": 0})
    with pytest.raises(ValueError):
        control_spec(controls=(declaration, replace(declaration, name="two")))
    with pytest.raises(ValueError):
        SDEControl("one", "forward", "rate_multiplier", -1, 4)


def test_native_step_budget_reports_all_actual_drift_diffusion_callbacks(monkeypatch):
    instance = session(max_evaluations_per_advance=12)
    counts = []
    drift, diffusion = ReactionSDE.drift, ReactionSDE.diffusion

    def counted_drift(*args):
        counts.append("drift")
        return drift(*args)

    def counted_diffusion(*args):
        counts.append("diffusion")
        return diffusion(*args)

    monkeypatch.setattr(ReactionSDE, "drift", counted_drift)
    monkeypatch.setattr(ReactionSDE, "diffusion", counted_diffusion)
    for duration in (0.0625, 0.03125):
        counts.clear()
        result = committed(instance, {"operation": "wait", "duration_s": duration})
        work = result.info["outcome_diagnostics"]["stochastic_work"]
        assert len(counts) == work["evaluations"] == 12


def test_in_bounds_control_cannot_commit_an_undefined_current_flux():
    flux = ReactionFlux(
        "forward",
        math("<apply><divide/><ci>n0</ci><ci>tau</ci></apply>"),
        (SDEParameter("tau", 2, DimensionVector(time=1)),),
    )
    instance = session(
        fluxes=(flux,),
        controls=(SDEControl("tau_input", "forward", "flux_parameter", 0, 8, "tau"),),
    )
    before = instance._state
    result = instance.execute({"operation": "set_sde_parameters", "values": {"tau_input": 0}})
    assert result.info["transaction_status"] != "committed"
    assert instance._state.species_amounts == before.species_amounts
    assert instance._state.process.stochastic_continuation == before.process.stochastic_continuation


def test_initialization_rejects_undefined_current_flux_before_installing_a_process():
    flux = ReactionFlux(
        "forward",
        math("<apply><divide/><ci>n0</ci><ci>tau</ci></apply>"),
        (SDEParameter("tau", 0, DimensionVector(time=1)),),
    )
    with pytest.raises(ValueError):
        session(fluxes=(flux,))
