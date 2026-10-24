"""Inventory projects original declarations and does not invent backend support."""

import json

from chemworld.examples.native_capabilities import capability_rows, execution_inventory
from chemworld.physchem import _MODULE_EXPORTS
from chemworld.world.domain_modules import DomainModuleRegistry, DomainModuleSpec


def test_execution_view_keeps_original_capabilities_and_codec_bindings():
    registry = DomainModuleRegistry.default()
    original = capability_rows(registry)
    view = execution_inventory(registry)
    # Existing parameter declarations use tuples; JSON represents those as arrays.
    decoded = json.loads(json.dumps(view, allow_nan=False))
    assert decoded["schema"] == view["schema"]
    assert [
        {k: v for k, v in row.items() if k != "execution"} for row in view["modules"]
    ] == original
    declarations = {module.module_id: module for module in registry.modules}
    for row in view["modules"]:
        actual = row["execution"]
        assert actual["cpu_lowering"] == actual["thread_safety"] == "not_declared"
        assert [m["id"] for m in actual["model_codecs"]] == [
            m.model_id for m in declarations[row["module"]].models
        ]
        assert [s["id"] for s in actual["advance_bindings"]] == row["advance_strategies"]
    assert capability_rows(registry) == original


def test_custom_module_is_not_guessed_to_be_compilable_or_thread_safe():
    registry = DomainModuleRegistry((DomainModuleSpec("opaque_extension"),))
    view = execution_inventory(registry)
    assert len(view["modules"]) == 1
    row = view["modules"][0]
    assert row["module"] == "opaque_extension"
    assert row["execution"]["cpu_lowering"] == "not_declared"
    assert row["execution"]["thread_safety"] == "not_declared"
    assert row["execution"]["model_codecs"] == []


def test_public_function_inventory_uses_original_lazy_facade():
    view = execution_inventory()
    assert {
        row["module"].removeprefix("chemworld.physchem."): row["symbols"]
        for row in view["public_physchem"]
    } == _MODULE_EXPORTS
