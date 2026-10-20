"""Declaration-to-runtime acceptance; no mocked dispatch or physics provider."""

from dataclasses import replace

import gymnasium as gym
import pytest

from chemworld.action_codec import ActionCodec
from chemworld.agent_interface import action_schema
from chemworld.foundation import Operation, equipment_settings, upsert_equipment_record
from chemworld.world.domain_modules import (
    ActionFieldSpec,
    ComponentCapability,
    DomainModuleRegistry,
    DomainModuleSpec,
    OperationSpec,
    ServiceFactorySpec,
)
from chemworld.world.operations import OPERATION_TYPES, PUBLIC_OBSERVATION_KEYS


class Agitator:
    def set_speed(self, state, action):
        return state.replace(
            equipment=upsert_equipment_record(
                state.equipment,
                equipment_id="bench_agitator",
                equipment_type="agitator",
                attached_vessel_id=state.vessel_id,
                settings={"speed_rpm": action["speed_rpm"]},
            )
        )


def agitator_module():
    return DomainModuleSpec(
        "agitator",
        operations=(
            OperationSpec(
                Operation(
                    "set_agitator", "Set shared agitator speed", ("speed_rpm",), ("not_terminated",)
                ),
                "mixing",
                "agitator",
                "set_speed",
                fields=(ActionFieldSpec("speed_rpm", "rpm", (0.0, 1200.0)),),
                affected_ledgers=("equipment",),
            ),
        ),
        services=(ServiceFactorySpec("agitator", "agitator", lambda ctx: Agitator(), "Agitator"),),
        components=(("agitator", ComponentCapability(("set_agitator",))),),
    )


def registry(*extras):
    return DomainModuleRegistry((*DomainModuleRegistry.default().modules, *extras))


def request():
    return {
        "schema_version": "chemworld-world-composition-0.1",
        "composition_id": "native-agitator",
        "world_split": "public-dev",
        "components": [
            {"kind": kind, "role": kind, "parameters": {}}
            for kind in ("reaction", "thermal", "observation", "agitator")
        ],
        "task": {"budget": 8, "description": "Native apparatus registration acceptance."},
    }


def test_new_declaration_reaches_public_schema_parser_and_actual_transaction():
    modules = registry(agitator_module())
    env = gym.make("ChemWorld", composition=request(), domain_modules=modules, seed=4)
    try:
        env.reset(seed=4)
        base = env.unwrapped
        assert base.runtime.modules is modules
        schema = action_schema(env, "set_agitator")
        assert schema["fields"][0]["unit"] == "rpm"
        assert "set_agitator" in base.allowed_operations
        assert base.runtime.domain_services.service_id_for_operation("set_agitator") == "agitator"
        _, _, _, _, info = env.step({"operation": "set_agitator", "payload": {"speed_rpm": 450.0}})
        assert info["transaction_status"] == "committed"
        assert info["affected_ledgers"] == ["equipment"]
        assert equipment_settings(base._state.equipment, "bench_agitator")["speed_rpm"] == 450.0
        _, _, _, _, rejected = env.step({"operation": "set_agitator", "speed_rpm": 1500.0})
        assert rejected["transaction_status"] != "committed"
        assert equipment_settings(base._state.equipment, "bench_agitator")["speed_rpm"] == 450.0
        with pytest.raises(ValueError, match="vector"):
            base.action_codec.encode_vector({"operation": "set_agitator", "speed_rpm": 450.0})
    finally:
        env.close()


def test_module_order_preserves_legacy_coordinates_and_world_isolation():
    modules = registry(agitator_module())
    reversed_modules = DomainModuleRegistry(tuple(reversed(modules.modules)))
    assert (
        ActionCodec(modules=modules).operation_types
        == ActionCodec(modules=reversed_modules).operation_types
    )
    # Fixed public anchors, independent of the registry that generates them.
    assert len(OPERATION_TYPES) == 43
    assert OPERATION_TYPES[6:9] == ("heat", "wait", "sample")
    assert OPERATION_TYPES[41:] == ("terminate", "measure")
    assert PUBLIC_OBSERVATION_KEYS[:3] == ("yield", "selectivity", "conversion")
    assert len(PUBLIC_OBSERVATION_KEYS) == 37
    plain = gym.make("ChemWorld", seed=4)
    extended = gym.make("ChemWorld", composition=request(), domain_modules=modules, seed=4)
    try:
        plain.reset(seed=4)
        extended.reset(seed=4)
        assert "set_agitator" not in plain.unwrapped.runtime.domain_services.operation_methods()
        extended.step({"operation": "set_agitator", "speed_rpm": 320.0})
        assert not equipment_settings(plain.unwrapped._state.equipment, "bench_agitator")
        assert action_schema(plain, "set_agitator")["valid_operation_type"] is False
    finally:
        plain.close()
        extended.close()


def test_uninstalled_reserved_action_is_rejected_and_never_sampled():
    from chemworld.envs.spaces import make_action_space

    modules = DomainModuleRegistry((agitator_module(),))
    codec = ActionCodec(modules=modules)
    with pytest.raises(ValueError, match="not installed"):
        codec.canonicalize({"operation": 6})
    space = make_action_space(modules=modules)
    space.seed(21)
    for _ in range(12):
        action = space.sample()
        assert space.contains(action)
        assert codec.canonicalize(action)["operation"] == "set_agitator"
    assert not space.contains({"operation": 7, "duration_s": 1.0})


def test_duplicate_owners_and_missing_dependencies_are_rejected():
    module = agitator_module()
    with pytest.raises(ValueError, match="Duplicate operation"):
        registry(module, replace(module, module_id="duplicate"))
    with pytest.raises(ValueError, match="Missing module dependencies"):
        registry(replace(module, dependencies=("absent",)))
    with pytest.raises(ValueError, match="service dependencies"):
        registry(replace(module, services=(replace(module.services[0], dependencies=("absent",)),)))
    with pytest.raises(ValueError, match="Cyclic module"):
        DomainModuleRegistry(
            (DomainModuleSpec("a", dependencies=("b",)), DomainModuleSpec("b", dependencies=("a",)))
        )
    with pytest.raises(ValueError, match="Missing operation owner"):
        registry(
            DomainModuleSpec(
                "bad_component", components=(("bad", ComponentCapability(("absent",))),)
            )
        )


def test_ports_checked_against_declared_units_and_direction():
    modules = DomainModuleRegistry(
        (
            DomainModuleSpec(
                "heat_link",
                components=(
                    (
                        "heater",
                        ComponentCapability((), outputs=("heat",), input_ports=(), port_unit="W"),
                    ),
                    (
                        "vessel",
                        ComponentCapability((), outputs=(), input_ports=("jacket",), port_unit="W"),
                    ),
                ),
            ),
        )
    )
    modules.validate_connection("heater", "heat", "vessel", "jacket", "W")
    with pytest.raises(ValueError, match="ports"):
        modules.validate_connection("heater", "out", "vessel", "jacket", "W")
    with pytest.raises(ValueError, match="units"):
        modules.validate_connection("heater", "heat", "vessel", "jacket", "mol")


def test_process_declaration_reuses_shared_operations():
    def process(context):
        return context

    modules = registry(DomainModuleSpec("surface", processes=(("surface_rates", process),)))
    assert modules.process_factories()["surface_rates"] is process
    assert (
        modules.operation_specs()["heat"]
        == DomainModuleRegistry.default().operation_specs()["heat"]
    )
    assert modules.operation_specs() == DomainModuleRegistry.default().operation_specs()
    with pytest.raises(ValueError, match="Duplicate process"):
        DomainModuleRegistry(
            (
                DomainModuleSpec("one", processes=(("rates", process),)),
                DomainModuleSpec("two", processes=(("rates", process),)),
            )
        )


def test_native_precondition_is_enforced_by_the_runtime_itself():
    modules = registry(agitator_module())
    env = gym.make("ChemWorld", composition=request(), domain_modules=modules, seed=4)
    try:
        env.reset(seed=4)
        result = env.unwrapped.runtime.apply_transaction(
            env.unwrapped._state.replace(terminated=True),
            {"operation": "set_agitator", "speed_rpm": 500.0},
        )
        assert result.kernel_result.outcome == "rejected"
        assert not equipment_settings(result.state.equipment, "bench_agitator")
    finally:
        env.close()


def test_instrument_definition_is_used_by_actual_sampling_without_cross_world_mutation():
    original = DomainModuleRegistry.default()
    observation = next(m for m in original.modules if m.module_id == "observation")
    instrument = next(i for i in observation.instruments if i.definition.id == "hplc")
    changed = replace(
        instrument,
        definition=replace(instrument.definition, cost=0.13),
        contract=replace(instrument.contract, cost=0.13),
    )
    instruments = tuple(
        changed if i.definition.id == "hplc" else i for i in observation.instruments
    )
    modules = DomainModuleRegistry(
        tuple(
            replace(m, instruments=instruments) if m.module_id == "observation" else m
            for m in original.modules
        )
    )
    env = gym.make("ChemWorld", domain_modules=modules, seed=4)
    try:
        env.reset(seed=4)
        env.step({"operation": "add_solvent", "volume_L": 0.03, "solvent": 2})
        env.step({"operation": "add_reagent", "amount_mol": 0.01})
        before = env.unwrapped._state.ledger.cost
        _, _, _, _, info = env.step({"operation": "measure", "instrument": "hplc"})
        assert info["transaction_status"] == "committed"
        assert env.unwrapped._state.ledger.cost - before == pytest.approx(0.13)
        assert env.unwrapped.constitution.instruments["hplc"].cost == 0.13
        assert DomainModuleRegistry.default().instrument_specs()["hplc"].definition.cost == 0.08
    finally:
        env.close()


def test_native_fields_are_not_reinterpreted_by_legacy_field_names():
    from chemworld.operation_validator import OperationValidator
    from chemworld.runtime.constitution_factory import make_chemworld_constitution
    from chemworld.world.state_factory import initial_chemworld_state

    module = agitator_module()
    spec = replace(
        module.operations[0],
        operation=Operation("set_agitator", "test", ("phase",)),
        fields=(ActionFieldSpec("phase", "categorical", choices=("gas", "solid")),),
    )
    modules = registry(replace(module, operations=(spec,)))
    codec = ActionCodec(modules=modules)
    validator = OperationValidator(
        constitution=make_chemworld_constitution(),
        allowed_operations={"set_agitator"},
        operation_types=codec.operation_types,
        action_codec=codec,
    )
    action = {"operation": "set_agitator", "phase": "gas"}
    assert codec.canonicalize(action) == action
    assert validator.validate(action, initial_chemworld_state()).is_valid
    assert not validator.validate(
        {**action, "phase": "aqueous"}, initial_chemworld_state()
    ).is_valid


def test_instrument_calibration_mappings_are_instance_scoped():
    first, second = DomainModuleRegistry.default(), DomainModuleRegistry.default()
    first.instrument_specs()["hplc"].definition.noise_std["yield"] = 9.0
    assert second.instrument_specs()["hplc"].definition.noise_std["yield"] == 0.012
