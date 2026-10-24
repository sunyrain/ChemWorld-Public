"""Real two-output recovery and restoration with exact empty finite vessels."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.aqueous_filtration import filter_vessel, synthetic_filtration_world
from chemworld.examples.aqueous_precipitation import route, synthetic_specification
from chemworld.foundation.state_ledgers import EquipmentLedger, VesselLedger
from chemworld.physchem.aqueous_mineral_equilibrium import solve_aqueous_mineral_equilibrium
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_filtration import filter_aqueous_suspension
from chemworld.world.aqueous_filtration_world import AqueousFilterSpec
from chemworld.world.aqueous_precipitation_world import build_aqueous_precipitation_world
from reference.aqueous_filtration40_reference import reference_history, totals


def members(state):
    return {state.vessel_id: state, **state.inactive_vessels}


def setup(family="ideal", eta=0.9, retention=0.08):
    spec, composition = synthetic_filtration_world(
        family, solid_retention=eta, liquid_retention=retention
    )
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=41)
    route(session, "feed-a", "feed-a-reactor", 0.5)
    route(session, "feed-b", "feed-b-reactor", 0.5)
    index = session._state.metadata["component_network"]["vessels"].index("reactor")
    assert (
        session.execute({"operation": "select_vessel", "vessel": index}).info["transaction_status"]
        == "committed"
    )
    return spec, composition, session


@pytest.mark.parametrize("family", ["ideal", "Davies"])
def test_original_filter_and_wash_against_independent_all_inventory_calorics(family):
    _, _, session = setup(family)
    expected = reference_history(family, 0.9, 0.08)
    try:
        first = filter_vessel(session, "reactor")
        assert first.info["actual_duration_s"] == 0
        assert all(n == 0 for n in session._state.species_amounts.values())
        assert all(p.volume_L == 0 for p in session._state.phases.phases.values())
        for key, state in members(session._state).items():
            assert state.species_amounts == pytest.approx(expected[3][key], abs=2e-9)
        receipt = session._state.metadata["aqueous_filter_history"][-1]
        assert len(receipt["packets"]) == 2
        assert receipt["packets"][1]["amounts_mol"]["NaCl(s)"] > 0
        snapshot = json.loads(json.dumps(session.snapshot().to_dict()))
        replay = ChemWorldRuntime.from_snapshot(snapshot)
        try:
            for instance in (session, replay):
                route(instance, "diluent", "wash-cake", 0.025)
                filter_vessel(instance, "cake")
            assert json.loads(json.dumps(session._state.to_dict())) == json.loads(
                json.dumps(replay._state.to_dict())
            )
        finally:
            replay.close()
        for key, state in members(session._state).items():
            assert state.species_amounts == pytest.approx(expected[-1][key], abs=2e-9)
        before, after = totals(expected[0], family), totals(expected[-1], family)
        ledger = session._state.ledger
        heat = ledger.energy_jacket_J - ledger.heat_loss_J
        assert heat == pytest.approx(after["H"] - before["H"], abs=2e-7)
        assert ledger.boundary_work_J == pytest.approx(1e5 * (after["V"] - before["V"]), abs=2e-7)
        assert after["U"] - before["U"] == pytest.approx(heat - ledger.boundary_work_J, abs=2e-7)
        assert ledger.time_s == 0
    finally:
        session.close()


@pytest.mark.parametrize("eta", [0.0, 1.0])
def test_zero_and_full_capture_keep_all_mineral_in_declared_outputs(eta):
    _, _, session = setup(eta=eta)
    try:
        mineral = session._state.species_amounts["NaCl(s)"]
        filter_vessel(session, "reactor")
        states = members(session._state)
        assert states["cake"].species_amounts["NaCl(s)"] == pytest.approx(eta * mineral, abs=2e-10)
        assert states["filtrate"].species_amounts["NaCl(s)"] == pytest.approx(
            (1 - eta) * mineral, abs=2e-10
        )
        for key in ("cake", "filtrate"):
            assert states[key].species_amounts["H2O"] > 0
    finally:
        session.close()


@pytest.mark.parametrize(
    "eta,retention", [(-0.1, 0.1), (1.1, 0.1), (0.9, 0.0), (0.9, 1.0), (True, 0.1)]
)
def test_invalid_retention_parameters_are_not_defaulted(eta, retention):
    with pytest.raises(ValueError):
        AqueousFilterSpec("filter", "source", "cake", "filtrate", eta, retention)


def test_true_empty_requires_explicit_permission_and_dry_mineral_is_rejected():
    spec = synthetic_specification("ideal")
    zero = dict.fromkeys((s.species_id for s in spec.database.species), 0.0)
    with pytest.raises(ValueError):
        build_aqueous_precipitation_world(spec, {"v": zero}, ())
    empty = build_aqueous_precipitation_world(spec, {"v": zero}, (), allow_empty_vessels=("v",))
    assert empty.physics.initial_states["v"].volume_L == 0
    with pytest.raises(ValueError):
        build_aqueous_precipitation_world(
            spec, {"v": {**zero, "NaCl(s)": 0.1}}, (), allow_empty_vessels=("v",)
        )


@pytest.mark.parametrize(
    "defect",
    [
        "cake_capacity",
        "filtrate_capacity",
        "target_solver",
        "occupied",
        "wrong_connection",
        "changed_capture",
    ],
)
def test_filter_failure_keeps_three_original_phases_resources_and_clock(defect, monkeypatch):
    _, composition, session = setup()
    try:
        state = session._state
        if defect.endswith("capacity"):
            key = defect.split("_")[0]
            target = state.inactive_vessels[key]
            vessel = target.vessels.vessels[key]
            target = target.replace(vessels=VesselLedger({key: replace(vessel, max_volume_L=1e-9)}))
            session._state = state.replace(inactive_vessels={**state.inactive_vessels, key: target})
        elif defect == "occupied":
            route(session, "diluent", "wash-cake", 0.001)
            index = session._state.metadata["component_network"]["vessels"].index("reactor")
            session.execute({"operation": "select_vessel", "vessel": index})
        elif defect == "wrong_connection":
            metadata = json.loads(json.dumps(state.metadata))
            next(
                e
                for e in metadata["component_network"]["connections"]
                if e["id"] == "first-filter:cake"
            )["target_vessel"] = "wash-filtrate"
            session._state = state.replace(metadata=metadata)
        elif defect == "changed_capture":
            records = dict(state.equipment.equipment)
            record = records["first-filter"]
            settings = json.loads(json.dumps(record.settings))
            settings["filter_specification"]["solid_retention_fraction"] = 0.1
            records[record.equipment_id] = replace(record, settings=settings)
            session._state = state.replace(equipment=EquipmentLedger(records))
        elif defect == "target_solver":
            from chemworld.runtime import aqueous_filtration

            original = aqueous_filtration.close_precipitation

            def fail(local):
                if local.vessel_id == "filtrate" and any(local.species_amounts.values()):
                    raise RuntimeError("injected second-target equilibrium failure")
                return original(local)

            monkeypatch.setattr(aqueous_filtration, "close_precipitation", fail)
        before, ledger = members(session._state), session._state.ledger
        result = session.execute({"operation": "filter_crystals"})
        assert result.info["transaction_status"] != "committed"
        if defect == "target_solver":
            assert result.info["operation_outcome"] == "numerical_failure"
        for key, value in members(session._state).items():
            assert value.phases == before[key].phases
            assert value.species_amounts == before[key].species_amounts
            assert value.thermal == before[key].thermal
        for key in ("time_s", "energy_jacket_J", "heat_loss_J", "boundary_work_J"):
            assert getattr(session._state.ledger, key) == getattr(ledger, key)
        if defect == "changed_capture":
            with pytest.raises(ValueError, match="filter declaration differs"):
                filter_aqueous_suspension(session._state, composition.physics)
    finally:
        session.close()


def test_restored_empty_source_cannot_change_filter_or_connection_authority():
    _, _, session = setup()
    try:
        filter_vessel(session, "reactor")
        snapshot = json.loads(json.dumps(session.snapshot().to_dict()))
        for defect in ("capture", "connection"):
            changed = json.loads(json.dumps(snapshot))
            if defect == "capture":
                changed["world_state"]["equipment"]["first-filter"]["settings"][
                    "filter_specification"
                ]["solid_retention_fraction"] = 0.2
            else:
                next(
                    e
                    for e in changed["world_state"]["metadata"]["component_network"]["connections"]
                    if e["id"] == "first-filter:filtrate"
                )["target"]["port"] = "out"
            with pytest.raises(ValueError, match="filter"):
                ChemWorldRuntime.from_snapshot(changed)
        restored = ChemWorldRuntime.from_snapshot(snapshot)
        restored.close()
    finally:
        session.close()


def test_thin_wet_cake_uses_actual_domain_valid_phase_support_only():
    spec = synthetic_specification("ideal")
    amounts = {
        "H2O": 0.011375 / 0.018015,
        "Na+": 0.00011375,
        "Cl-": 0.00011375,
        "K+": 0.000125,
        "Br-": 0.000125,
        "NaCl(s)": 0.0425,
    }
    result = solve_aqueous_mineral_equilibrium(spec, amounts, temperature_K=298.15, pressure_Pa=1e5)
    assert result.success
    assert result.amounts_mol == pytest.approx(amounts, abs=2e-10)
    absent, present = result.diagnostics["phase_sets"]
    assert absent["failure"] == "ionic strength outside declared activity-model domain"
    assert "strategy" not in absent["initialization"]
    assert present["initialization"]["strategy"] == "actual_feed_after_lp_domain_failure"
    assert present["initialization"]["feed_scaled_invariant_error"] <= 64 * 2.220446049250313e-16


def test_infeasible_lp_upper_bound_is_not_hidden_by_projection_or_feed(monkeypatch):
    from chemworld.physchem import aqueous_mineral_equilibrium

    spec, _, session = setup()
    actual = aqueous_mineral_equilibrium.linprog

    def infeasible(*args, **kwargs):
        result = actual(*args, **kwargs)
        if result.success:
            result.x[0] = 1.000001
        return result

    monkeypatch.setattr(aqueous_mineral_equilibrium, "linprog", infeasible)
    try:
        result = solve_aqueous_mineral_equilibrium(
            spec, session._state.species_amounts, temperature_K=298.15, pressure_Pa=1e5
        )
        assert not result.success
        assert all(
            r["failure"] == "initial_outside_bounds" for r in result.diagnostics["phase_sets"]
        )
    finally:
        session.close()
