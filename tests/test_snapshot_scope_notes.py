"""Neutral declarations preserve continuation; execution edits still reject."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import declaration
from chemworld.world.domain_modules import (
    DomainModuleRegistry,
    ExecutionEffectSpec,
    NativeExecutionSpec,
)
from chemworld.world.mineral_world import mineral_world_spec


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_legacy_scope_notes_restore_real_history_without_losing_rng_or_resources(noise):
    session = ChemWorldRuntime(
        composition=mineral_world_spec(), observation_noise_mode=noise,
        observation_seed_override=12,
    )
    restored = None
    try:
        session.reset(seed=7)
        session.execute({"operation": "wait", "duration_s": 1})
        canonical = session.snapshot().to_dict()
        legacy = deepcopy(canonical)
        legacy["execution"]["modules"] = declaration(session.domain_modules.modules)
        original = deepcopy(legacy)
        changed_notes = DomainModuleRegistry(tuple(
            replace(module, scope_notes=("Updated capability documentation only",))
            for module in session.domain_modules.modules
        ))
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(legacy)), domain_modules=changed_notes,
        )
        assert legacy == original
        assert restored.snapshot().to_dict() == canonical
        assert all("scope_notes" not in m for m in canonical["execution"]["modules"])
        for action in (
            {"operation": "wait", "duration_s": 1},
            {"operation": "measure", "instrument": "aqueous_ph"},
        ):
            assert restored.execute(action) == session.execute(action)
            assert restored.snapshot().to_dict() == session.snapshot().to_dict()
    finally:
        if restored is not None:
            restored.close()
        session.close()


@pytest.mark.parametrize("change", [
    "runtime", "numpy_version", "scipy_version", "python_version", "rng",
    "operation_order", "module_version", "dependencies", "strategy",
    "unknown_module_field", "nested_scope_notes", "unknown_binding_field",
    "cpu_lowering", "cpu_snapshot", "unknown_cpu_field", "native_effects",
])
def test_scope_compatibility_does_not_weaken_real_execution_contracts(change):
    session = ChemWorldRuntime(composition=mineral_world_spec())
    try:
        session.reset(seed=7)
        before = session.snapshot().to_dict()
        bad = deepcopy(before)
        execution = bad["execution"]
        # Legacy display notes are accepted, but cannot hide any other change.
        execution["modules"] = declaration(session.domain_modules.modules)
        module = next(m for m in execution["modules"] if m["advance_strategies"])
        if change in {"runtime", "numpy_version", "scipy_version", "python_version", "rng"}:
            execution[change] = "different-execution"
        elif change == "operation_order":
            execution[change] = list(reversed(execution[change]))
        elif change == "module_version":
            module["version"] = "different-version"
        elif change == "dependencies":
            module[change].append("new-runtime-dependency")
        elif change == "strategy":
            module["advance_strategies"][0]["factory"] = {"callable": "other.executor"}
        elif change == "nested_scope_notes":
            module["advance_strategies"][0]["scope_notes"] = ["not-module-docs"]
        elif change == "unknown_module_field":
            module["unknown_execution_input"] = 1
        elif change == "cpu_lowering":
            module["execution"]["lowering"] = "native_algorithm"
        elif change == "cpu_snapshot":
            module["execution"]["snapshot"] = "supported"
        elif change == "unknown_cpu_field":
            module["execution"]["unknown_execution_input"] = 1
        elif change == "native_effects":
            module["effects"].append({"resource": "material", "access": "write"})
        else:
            execution["unknown_execution_input"] = 1
        with pytest.raises(ValueError, match="execution contracts changed"):
            session.restore(bad)
        assert session.snapshot().to_dict() == before
    finally:
        session.close()


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_snapshots_before_optional_cpu_declarations_keep_exact_native_continuation(noise):
    session = ChemWorldRuntime(
        composition=mineral_world_spec(), observation_noise_mode=noise,
        observation_seed_override=12,
    )
    restored = None
    try:
        session.reset(seed=7)
        session.execute({"operation": "wait", "duration_s": 1})
        canonical = session.snapshot().to_dict()
        legacy = deepcopy(canonical)
        for module in legacy["execution"]["modules"]:
            module.pop("execution", None)
            module.pop("effects", None)
        original = deepcopy(legacy)
        restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(legacy)))
        assert legacy == original
        assert restored.snapshot().to_dict() == canonical
        for action in (
            {"operation": "wait", "duration_s": 1},
            {"operation": "measure", "instrument": "aqueous_ph"},
        ):
            assert restored.execute(action) == session.execute(action)
            assert restored.snapshot().to_dict() == session.snapshot().to_dict()
    finally:
        if restored is not None:
            restored.close()
        session.close()


def test_explicit_cpu_contract_and_effects_stay_bound_to_their_registry():
    original = DomainModuleRegistry.default()
    explicit = NativeExecutionSpec(
        lowering="native_algorithm", derivatives="analytic", thread_safety="instance_local",
        snapshot="supported",
    )
    effects = (ExecutionEffectSpec("runtime.history", "read", "runtime"),)
    registry = DomainModuleRegistry((
        replace(original.modules[0], execution=explicit, effects=effects),
        *original.modules[1:],
    ))
    session = ChemWorldRuntime(composition=mineral_world_spec().to_dict(), domain_modules=registry)
    restored = None
    try:
        session.reset(seed=7)
        session.execute({"operation": "wait", "duration_s": 1})
        saved = session.snapshot().to_dict()
        module = saved["execution"]["modules"][0]
        assert module["execution"] == declaration(explicit)
        assert module["effects"] == declaration(effects)
        with pytest.raises(ValueError, match="execution contracts changed"):
            ChemWorldRuntime.from_snapshot(saved)
        restored = ChemWorldRuntime.from_snapshot(saved, domain_modules=registry)
        assert restored.execute({"operation": "wait", "duration_s": 1}) == session.execute(
            {"operation": "wait", "duration_s": 1}
        )
        assert restored.snapshot().to_dict() == session.snapshot().to_dict()
    finally:
        if restored is not None:
            restored.close()
        session.close()
