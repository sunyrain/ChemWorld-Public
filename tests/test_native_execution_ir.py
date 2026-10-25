"""Native CPU metadata is explicit, typed and generated from module declarations."""

import pytest

from chemworld.foundation.state_ownership import StateFieldSpec, StateSlotSpec
from chemworld.world.domain_modules import (
    ComponentCapability,
    DomainModuleRegistry,
    DomainModuleSpec,
    ExecutionEffectSpec,
    NativeCapabilityError,
    NativeExecutionSpec,
)


def native_registry():
    state = StateSlotSpec(
        slot_id="material:lot",
        owner_kind="material",
        owner_id="lot",
        state_type="inventory",
        fields=(
            StateFieldSpec(
                "amount",
                "mol",
                conserved_quantity="substance",
            ),
        ),
    )
    module = DomainModuleSpec(
        "native_fixture",
        components=(
            (
                "reactor",
                ComponentCapability(
                    operations=(),
                    state_owners=("material:lot",),
                    input_ports=("feed",),
                    outputs=("product",),
                    port_unit="mol",
                ),
            ),
        ),
        state_slots=(state,),
        execution=NativeExecutionSpec(
            lowering="expression",
            derivatives="analytic",
            thread_safety="instance_local",
            snapshot="supported",
            public_result_type="fixture.Result",
            private_diagnostic_type="fixture.Diagnostics",
        ),
        effects=(
            ExecutionEffectSpec(
                resource="material:lot.amount",
                access="read",
                owner="lot",
                unit="mol",
                mixing="sum",
                reset="clear",
                conserved_quantity="substance",
            ),
        ),
    )
    return DomainModuleRegistry((module,))


def test_lowering_exposes_typed_state_ports_effects_and_conservation():
    ir = native_registry().lower_execution_ir(backend="cpu")
    module = ir["modules"][0]
    assert ir["backend"] == "cpu"
    assert module["execution"]["lowering"] == "expression"
    assert module["state"][0]["fields"][0]["unit"] == "mol"
    assert module["state"][0]["fields"][0]["mixing"] == "sum"
    assert module["state"][0]["batch_reset"] == "clear"
    assert module["ports"][0]["inputs"][0]["unit"] == "mol"
    assert module["ports"][0]["outputs"][0]["id"] == "reactor.output.product"
    assert module["effects"][0]["access"] == "read"
    assert module["conservation"] == ["substance"]
    assert module["execution"]["public_result_type"] == "fixture.Result"
    assert module["execution"]["private_diagnostic_type"] == "fixture.Diagnostics"
    assert ir["units"]["canonical"]["volume"] == "m^3"
    assert {
        (item["source"], item["target"])
        for item in ir["units"]["conversions"]
    } >= {
        ("L", "m^3"),
        ("m^3", "L"),
        ("mol", "particle_count"),
        ("particle_count", "mol"),
        ("J", "kJ"),
    }
    assert ir["units"]["energy_reference"]["absolute"] == "J"


def test_undeclared_modules_select_reference_and_cpu_selection_fails_closed():
    registry = DomainModuleRegistry((DomainModuleSpec("opaque"),))
    ir = registry.lower_execution_ir()
    assert ir["backend"] == "reference"
    assert {item["field"] for item in ir["diagnostics"]} == {
        "lowering",
        "derivatives",
        "thread_safety",
        "snapshot",
    }
    with pytest.raises(NativeCapabilityError) as error:
        registry.lower_execution_ir(backend="cpu")
    assert len(error.value.diagnostics) == 4


def test_effects_must_be_unique_and_bound_to_declared_operations():
    duplicate = ExecutionEffectSpec("resource", "read", "owner")
    with pytest.raises(ValueError, match="Duplicate native effect"):
        DomainModuleRegistry(
            (
                DomainModuleSpec(
                    "duplicate",
                    effects=(duplicate, duplicate),
                ),
            )
        )
    with pytest.raises(ValueError, match="Missing operation owner"):
        DomainModuleRegistry(
            (
                DomainModuleSpec(
                    "missing-operation",
                    effects=(
                        ExecutionEffectSpec(
                            "resource", "read", "owner", operation_id="gone"
                        ),
                    ),
                ),
            )
        )
