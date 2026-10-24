"""Model configuration is owned by the installed instance declarations."""

import json
from copy import deepcopy
from dataclasses import dataclass, fields, replace

import pytest

from chemworld.examples.native_domains import workflow
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec, compile_world_composition
from chemworld.world.domain_modules import DomainModuleRegistry, DomainModuleSpec, PhysicsModelSpec
from chemworld.world.physical_world import CompositionPhysics
from chemworld.world.surface_world import surface_world_spec


def make_world(name):
    if name in {"heater", "distillation", "crystal"}:
        return workflow(name)[0]
    if name == "flash":
        from chemworld.examples.native_flash_research import synthetic_vle_properties
        from chemworld.world.vle_world import vle_world_spec

        return vle_world_spec(synthetic_vle_properties())
    if name == "liquid":
        from chemworld.examples.native_liquid_research import synthetic_liquid_world

        return synthetic_liquid_world()
    if name == "concentration":
        from chemworld.examples.native_concentration_research import (
            initial_concentration_state,
            synthetic_concentration_spec,
        )
        from chemworld.world.concentration_world import concentration_world_spec

        model = synthetic_concentration_spec()
        return concentration_world_spec(model, initial_concentration_state(model))
    raise AssertionError(name)


def legacy_payload(spec):
    payload = deepcopy(spec.to_dict())
    for model_id, record in payload["physics"].pop("models").items():
        alias = spec.modules.model_specs()[model_id].legacy_field
        payload["physics"][alias] = record["bindings"]
    return payload


@pytest.mark.parametrize("name", (
    "heater", "distillation", "crystal", "flash", "liquid", "concentration",
))
def test_all_builtin_model_families_round_trip_and_normalize_legacy_input(name):
    world = make_world(name)
    payload = world.to_dict()
    canonical = WorldCompositionSpec.from_dict(json.loads(json.dumps(payload)))
    legacy = WorldCompositionSpec.from_dict(legacy_payload(world))
    assert canonical.to_dict() == legacy.to_dict() == payload
    assert not any(field.name.endswith("_models") for field in fields(world.physics))
    assert canonical.to_dict()["physics"]["initial_states"] == payload["physics"]["initial_states"]
    compile_world_composition(canonical)
    compile_world_composition(legacy)


def test_legacy_input_and_canonical_input_execute_the_same_real_history():
    spec = surface_world_spec()
    sessions = [
        ChemWorldRuntime(composition=world, observation_noise_mode="keyed")
        for world in (spec, WorldCompositionSpec.from_dict(legacy_payload(spec)))
    ]
    try:
        for session in sessions:
            session.reset(seed=12)
        for action in (
            {"operation": "heat", "target_temperature_K": 370, "duration_s": 2},
            {"operation": "measure", "instrument": "gas_pressure"},
            {"operation": "wait", "duration_s": 1},
        ):
            results = [s.execute(action) for s in sessions]
            assert all(r.info["transaction_status"] == "committed" for r in results)
            assert sessions[0]._state.to_dict() == sessions[1]._state.to_dict()
            assert results[0].info["native_observation"] == results[1].info["native_observation"]
        continued = ChemWorldRuntime.from_snapshot(sessions[0].snapshot().to_dict())
        try:
            action = {"operation": "wait", "duration_s": 1}
            sessions[0].execute(action)
            continued.execute(action)
            assert sessions[0]._state.to_dict() == continued._state.to_dict()
        finally:
            continued.close()
    finally:
        for session in sessions:
            session.close()


@dataclass(frozen=True)
class LocalEnvelope:
    minimum: float
    unit: str = "K"


def decode_envelope(payload):
    if set(payload) != {"minimum", "unit"} or payload["unit"] != "K":
        raise ValueError("local envelope requires native K units")
    if not 0 < payload["minimum"] < 1000:
        raise ValueError("local envelope temperature is out of domain")
    return LocalEnvelope(**payload)


def encode_envelope(model):
    return {"minimum": model.minimum, "unit": model.unit}


def validate_envelope(physics, composition):
    for vessel, model in physics.models_of("local-envelope").items():
        if physics.initial_states[vessel].temperature_K < model.minimum:
            raise ValueError("initial state outside local envelope")


LOCAL_MODEL = PhysicsModelSpec(
    "local-envelope", "test-1", decode_envelope, encode_envelope,
    legacy_field="local_envelope_models", validate_world=validate_envelope,
)


def local_registry(model=LOCAL_MODEL):
    return DomainModuleRegistry((
        *DomainModuleRegistry.default().modules,
        DomainModuleSpec("local-envelope", models=(model,)),
    ))


def test_local_model_codec_binding_and_world_validation_need_no_central_registration():
    base = surface_world_spec()
    modules = local_registry()
    payload = base.to_dict()
    payload["physics"]["models"]["local-envelope"] = {
        "version": "test-1", "bindings": {"reactor": {"minimum": 300, "unit": "K"}},
    }
    world = WorldCompositionSpec.from_dict(payload, modules=modules)
    assert world.physics.models_of("local-envelope")["reactor"] == LocalEnvelope(300)
    compile_world_composition(world)
    assert WorldCompositionSpec.from_dict(world.to_dict(), modules=modules).to_dict() == payload
    assert "local-envelope" not in base.modules.model_specs()
    assert "local-envelope" not in DomainModuleRegistry.default().model_specs()
    assert "local-envelope" not in base.physics.models
    with pytest.raises(ValueError, match="unknown physical model type"):
        WorldCompositionSpec.from_dict(payload)
    payload["physics"]["models"]["local-envelope"]["bindings"]["reactor"]["minimum"] = 400
    with pytest.raises(ValueError, match="outside local envelope"):
        compile_world_composition(WorldCompositionSpec.from_dict(payload, modules=modules))


@pytest.mark.parametrize("case, message", (
    ("unknown", "unknown physical model type"),
    ("version", "unsupported physical model version"),
    ("duplicate_input", "duplicate canonical and legacy"),
    ("record_shape", "requires version and bindings"),
    ("missing_vessel", "declared vessel"),
    ("native_units", "native mol/L/K/Pa/s"),
    ("missing_component", "matching component"),
    ("named_identity", "matching identities"),
))
def test_invalid_models_fail_before_execution(case, message):
    spec = surface_world_spec()
    payload = spec.to_dict()
    models = payload["physics"]["models"]
    if case == "unknown":
        models["untrusted.module:factory"] = {"version": "1", "bindings": {}}
    elif case == "version":
        models["thermal"]["version"] = "2"
    elif case == "duplicate_input":
        payload["physics"]["thermal_models"] = models["thermal"]["bindings"]
    elif case == "record_shape":
        models["thermal"]["loader"] = "untrusted.module"
    elif case == "missing_vessel":
        models["thermal"]["bindings"]["missing"] = models["thermal"]["bindings"]["reactor"]
    elif case == "native_units":
        payload["physics"]["initial_states"]["reactor"]["units"]["temperature"] = "degC"
    elif case == "missing_component":
        payload = make_world("flash").to_dict()
        payload["physics"]["models"]["flash"]["bindings"].pop("reactor")
    elif case == "named_identity":
        bindings = models["surface"]["bindings"]
        bindings["wrong-id"] = bindings.pop(next(iter(bindings)))
    with pytest.raises(ValueError, match=message):
        compile_world_composition(WorldCompositionSpec.from_dict(payload))


def test_custom_invalid_units_and_conflicting_ownership_are_rejected():
    base = surface_world_spec()
    payload = base.to_dict()
    payload["physics"]["models"]["local-envelope"] = {
        "version": "test-1", "bindings": {"reactor": {"minimum": 300, "unit": "degC"}},
    }
    with pytest.raises(ValueError, match="native K units"):
        WorldCompositionSpec.from_dict(payload, modules=local_registry())
    payload["physics"]["models"]["local-envelope"]["bindings"]["reactor"]["unit"] = "K"
    modules = local_registry(replace(LOCAL_MODEL, exclusive_group="thermal_evolution"))
    with pytest.raises(ValueError, match="conflicting physical model owners"):
        compile_world_composition(WorldCompositionSpec.from_dict(payload, modules=modules))


@pytest.mark.parametrize("duplicate_alias", (False, True))
def test_module_registry_rejects_duplicate_model_owners_or_input_aliases(duplicate_alias):
    declaration = replace(LOCAL_MODEL, model_id="other") if duplicate_alias else LOCAL_MODEL
    with pytest.raises(ValueError, match=r"Duplicate (physical model|legacy model alias) owner"):
        DomainModuleRegistry((
            *local_registry().modules, DomainModuleSpec("other", models=(declaration,)),
        ))


def test_model_storage_is_instance_owned_and_legacy_readers_do_not_store_a_copy():
    base = surface_world_spec().physics
    sibling = CompositionPhysics.from_dict(base.to_dict())
    assert base.thermal_models is base.models["thermal"]
    assert base.thermal_models is not sibling.models["thermal"]
    before = sibling.to_dict()
    base.models["thermal"].pop("receiver")
    assert sibling.to_dict() == before
    changed = replace(sibling, thermal_models={})
    canonical = replace(sibling, models={**sibling.models, "thermal": {}})
    assert canonical.to_dict() == changed.to_dict()
    assert canonical.models_of("surface") == sibling.models_of("surface")
    assert changed.models["thermal"] == {}
    assert sibling.to_dict() == before


@pytest.mark.parametrize("model_id", [
    "thermal", "flash", "liquid", "column", "concentration", "crystal", "sorbent", "surface",
])
def test_canonical_constructor_and_replace_preserve_other_model_families(model_id):
    base = surface_world_spec().physics
    before = base.to_dict()
    declarations = base.modules.model_specs()
    models = {**base.models, model_id: {}}
    old = replace(base, **{declarations[model_id].legacy_field: {}})
    canonical = replace(base, models=models)
    assert canonical.to_dict() == old.to_dict()
    common = {
        "parameter_seed": base.parameter_seed, "feed_sources": base.feed_sources,
        "lab_clock": base.lab_clock, "advance_bindings": base.advance_bindings,
        "modules": base.modules,
    }
    legacy = CompositionPhysics(
        base.network, base.initial_states, **common,
        **{declarations[key].legacy_field: bindings for key, bindings in models.items()},
    )
    current = CompositionPhysics(base.network, base.initial_states, **common, models=models)
    assert current.to_dict() == legacy.to_dict() == canonical.to_dict()
    for other in base.models.keys() - {model_id}:
        assert canonical.models_of(other) == base.models_of(other)
    assert base.to_dict() == before
