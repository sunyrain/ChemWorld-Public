"""Real dispatch, compilation failures and explicit legacy compatibility."""

from dataclasses import replace

import pytest

from advance_strategy_cases import CASES, case, trace
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.coupled_thermal import CoupledThermalRuntime
from chemworld.world.composition import (
    WorldCompositionSpec,
    check_world_composition_compatibility,
    compile_world_composition,
)
from chemworld.world.domain_modules import (
    AdvanceStrategySpec,
    DomainModuleRegistry,
    DomainModuleSpec,
)
from chemworld.world.physical_world import CompositionPhysics
from chemworld.world.surface_world import surface_world_spec

EXPECTED = {
    **dict.fromkeys(("inert", "bulk", "surface"), "owned-inventory-temperature"),
    "flash": "constant-pressure-flash-enthalpy",
    "flash-stop": "constant-pressure-flash-enthalpy",
    "liquid": "isothermal-liquid-equilibrium",
    "column": "finite-equilibrium-column",
    "concentration": "finite-vacuum-concentration-enthalpy",
    "crystal": "finite-crystal-population-temperature",
    "battery": "single-particle-battery",
    "ssa": "gillespie-direct",
    "legacy": "legacy-liquid-enthalpy",
    "legacy-stable": "legacy-liquid-enthalpy",
}


@pytest.mark.parametrize("name", CASES)
def test_real_heat_wait_control_and_stop_use_declared_instances(name):
    session, actions = case(name)
    try:
        session.reset(seed=7)
        thermal = session.runtime.domain_services.reaction_thermal
        advancer = thermal.advancer
        for spec in session.domain_modules.advance_specs().values():
            if spec.service_attribute:
                assert (
                    getattr(thermal, spec.service_attribute)
                    is advancer.strategies[spec.strategy_id]
                )
        for action in actions:
            result = session.execute(action)
            assert advancer.strategy(session._state)[0] == EXPECTED[name]
            actual_strategy = result.info["outcome_diagnostics"].get("advance_strategy")
            if actual_strategy is not None:
                assert actual_strategy == EXPECTED[name]
        if name == "flash-stop":
            assert result.info["operation_outcome"] == "stopped"
            assert 0 < result.info["actual_duration_s"] < 100
        if name.startswith("legacy"):
            assert advancer.supports(session._state)
            assert not advancer.supports_measurement_clock(session._state)
    finally:
        session.close()


def remove_strategy(registry, key):
    return DomainModuleRegistry(
        tuple(
            replace(
                module,
                advance_strategies=tuple(
                    spec for spec in module.advance_strategies if spec.strategy_id != key
                ),
            )
            for module in registry.modules
        )
    )


def local_thermal_factory(composition):
    return CoupledThermalRuntime(composition, composition.modules)


def with_local_thermal(base):
    modules = remove_strategy(base.modules, "owned-inventory-temperature")
    modules = DomainModuleRegistry(
        (
            *modules.modules,
            DomainModuleSpec(
                "local-thermal",
                advance_strategies=(AdvanceStrategySpec("local-thermal", local_thermal_factory),),
            ),
        )
    )
    return WorldCompositionSpec.from_dict(base.to_dict(), modules=modules)


def test_local_strategy_executes_without_a_central_attribute_or_fallback(monkeypatch):
    from chemworld.runtime.reaction_thermal_services import ChemWorldReactionThermalServices

    base = surface_world_spec()
    world = with_local_thermal(base)

    def no_fallback(*args, **kwargs):
        raise AssertionError("native strategy cannot fall back to the legacy liquid solver")

    monkeypatch.setattr(ChemWorldReactionThermalServices, "_integrate_legacy", no_fallback)
    session = ChemWorldRuntime(composition=world)
    try:
        session.reset(seed=7)
        assert not hasattr(session.runtime.domain_services.reaction_thermal, "coupled")
        result = session.execute(
            {"operation": "heat", "target_temperature_K": 380, "duration_s": 1}
        )
        assert result.info["transaction_status"] == "committed"
        assert result.info["outcome_diagnostics"]["advance_strategy"] == "local-thermal"
        assert session._state.ledger.time_s == 1
        assert session._state.temperature_K > 350
        assert "local-thermal" not in base.modules.advance_specs()
    finally:
        session.close()


def test_sorbent_dry_uses_selected_strategy_without_a_thermal_service_alias():
    from chemworld.physchem.sorbent_kinetics import loading_field
    from test_native_sorbent_runtime import session_for

    base, original = session_for()
    original.close()
    session = ChemWorldRuntime(composition=with_local_thermal(base))
    try:
        session.reset(seed=22)
        assert not hasattr(session.runtime.domain_services.reaction_thermal, "coupled")
        result = session.execute({"operation": "dry"})
        assert result.info["transaction_status"] == "committed"
        assert result.info["outcome_diagnostics"]["advance_strategy"] == "local-thermal"
        assert session._state.ledger.time_s == 300
        assert session._state.state_slots.slots[0].values[loading_field("B")] > 0
    finally:
        session.close()


def test_missing_native_strategy_is_rejected_at_world_compilation():
    base = surface_world_spec()
    modules = remove_strategy(base.modules, "owned-inventory-temperature")
    with pytest.raises(ValueError, match="no declared physical advance strategy"):
        compile_world_composition(WorldCompositionSpec.from_dict(base.to_dict(), modules=modules))


def test_missing_legacy_strategy_is_rejected_at_runtime_binding():
    modules = remove_strategy(DomainModuleRegistry.default(), "legacy-liquid-enthalpy")
    with pytest.raises(ValueError, match="no declared physical advance strategy"):
        ChemWorldRuntime(domain_modules=modules)


@pytest.mark.parametrize("binding", (None, "owned-inventory-temperature"))
def test_conflicting_strategy_owners_are_not_hidden_by_a_binding(binding):
    base = surface_world_spec()
    modules = DomainModuleRegistry(
        (
            *base.modules.modules,
            DomainModuleSpec(
                "conflict", advance_strategies=(("other-thermal", local_thermal_factory),)
            ),
        )
    )
    payload = base.to_dict()
    if binding is not None:
        payload["physics"]["advance_bindings"] = {"reactor": binding}
    report = check_world_composition_compatibility(
        WorldCompositionSpec.from_dict(payload, modules=modules)
    )
    assert not report.compatible
    assert any("multiple physical advance strategies" in d.message for d in report.diagnostics)
    with pytest.raises(ValueError, match="multiple physical advance strategies"):
        compile_world_composition(WorldCompositionSpec.from_dict(payload, modules=modules))


@pytest.mark.parametrize(
    "bindings",
    (
        {"reactor": "unknown-strategy"},
        {"missing-vessel": "owned-inventory-temperature"},
        {"reactor": "legacy-liquid-enthalpy"},
    ),
)
def test_invalid_or_conflicting_explicit_bindings_are_rejected(bindings):
    payload = surface_world_spec().to_dict()
    payload["physics"]["advance_bindings"] = bindings
    with pytest.raises(ValueError, match=r"advance binding|multiple physical advance strategies"):
        compile_world_composition(payload)


def test_old_plain_liquid_input_normalizes_once_and_explicit_empty_binding_does_not_fallback():
    from test_world_task_binding import request

    payload = request()
    payload["physics"].pop("advance_bindings")
    old = WorldCompositionSpec.from_dict(payload)
    assert old.physics.advance_bindings == {"reactor": "legacy-liquid-enthalpy"}
    assert WorldCompositionSpec.from_dict(old.to_dict()).to_dict() == old.to_dict()
    canonical = old.to_dict()
    canonical["physics"]["advance_bindings"] = {}
    with pytest.raises(ValueError, match="no declared physical advance strategy"):
        compile_world_composition(canonical)


def test_strategy_service_alias_has_one_owner():
    with pytest.raises(ValueError, match="Duplicate strategy service attribute"):
        DomainModuleRegistry(
            (
                DomainModuleSpec(
                    "a",
                    advance_strategies=(
                        AdvanceStrategySpec(
                            "a", local_thermal_factory, service_attribute="coupled"
                        ),
                    ),
                ),
                DomainModuleSpec(
                    "b",
                    advance_strategies=(
                        AdvanceStrategySpec(
                            "b", local_thermal_factory, service_attribute="coupled"
                        ),
                    ),
                ),
            )
        )


def test_input_compatibility_does_not_invent_a_liquid_solver_for_unbound_model_parameters():
    base = surface_world_spec().physics
    modules = remove_strategy(base.modules, "owned-inventory-temperature")
    copied = CompositionPhysics(
        base.network,
        base.initial_states,
        models=base.models,
        modules=modules,
        feed_sources=base.feed_sources,
    )
    assert not copied.advance_bindings


def test_explicit_legacy_selection_cannot_bypass_the_liquid_model_domain():
    from test_world_task_binding import request

    payload = request(inert=True)
    payload["physics"]["advance_bindings"] = {"reactor": "legacy-liquid-enthalpy"}
    with pytest.raises(ValueError, match=r"advance binding .* does not own"):
        compile_world_composition(payload)

    base = surface_world_spec()
    modules = remove_strategy(base.modules, "owned-inventory-temperature")
    payload = base.to_dict()
    payload["physics"]["advance_bindings"] = {"reactor": "legacy-liquid-enthalpy"}
    with pytest.raises(ValueError, match=r"advance binding .* does not own"):
        compile_world_composition(WorldCompositionSpec.from_dict(payload, modules=modules))


def test_noisy_queries_and_strategy_inspection_do_not_change_the_pending_ssa_event():
    session, _ = case("ssa")
    try:
        session.reset(seed=7)
        before = session.snapshot().to_dict()
        for _ in range(3):
            session.runtime.domain_services.reaction_thermal.advancer.strategy(session._state)
            session.compiled_composition.spec.physics.advance_strategy(
                session._state, session.domain_modules
            )
            session.action_schema("wait")
        assert session.snapshot().to_dict() == before
    finally:
        session.close()


def test_known_legacy_failure_remains_a_numerical_rollback_then_wait_can_continue():
    records = trace("legacy")
    assert records[2]["transaction_status"] == "rolled_back"
    assert records[2]["outcome"] == "numerical_failure"
    assert records[2]["failure_reason"] == "reaction_solver_failed"
    assert records[2]["duration_s"] == 0
    assert records[3]["outcome"] == "success"
