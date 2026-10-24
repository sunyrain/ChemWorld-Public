"""Real finite-liquid routing, phase appearance and the original transaction."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.aqueous_precipitation import route, synthetic_world
from chemworld.foundation.state_ledgers import VesselLedger
from chemworld.physchem.aqueous_mineral_equilibrium import solve_aqueous_mineral_equilibrium
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_mineral import AqueousMineralRuntime
from chemworld.runtime.aqueous_precipitation import precipitation_record
from reference.aqueous_precipitation40_reference import P, close, properties, transfer


def members(session):
    state = session._state
    return {state.vessel_id: state, **state.inactive_vessels}


def setup(family="ideal", association=False):
    spec, composition = synthetic_world(family, association=association)
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=40)
    return spec, session


@pytest.mark.parametrize("family", ["ideal", "Davies"])
@pytest.mark.parametrize("association", [False, True])
def test_complete_route_against_independent_solubility_and_calorics(family, association):
    _, session = setup(family, association)
    try:
        stocks = {k: s.species_amounts for k, s in members(session).items()}
        initial = {k: properties(v, family) for k, v in stocks.items()}
        solids = []
        for source, edge, fraction in (
            ("feed-a", "feed-a-reactor", 0.5),
            ("feed-b", "feed-b-reactor", 0.5),
            ("diluent", "dilute-reactor", 0.25),
            ("diluent", "dilute-reactor", 0.75),
        ):
            stocks = transfer(stocks, source, "reactor", fraction, family)
            route(session, source, edge, fraction)
            actual = members(session)
            for name, expected in stocks.items():
                assert actual[name].species_amounts == pytest.approx(expected, abs=2e-9)
            final = {k: properties(v, family) for k, v in stocks.items()}
            heat = session._state.ledger.energy_jacket_J - session._state.ledger.heat_loss_J
            expected_heat = sum(final[k]["H"] - initial[k]["H"] for k in stocks)
            work = P * sum(final[k]["V"] - initial[k]["V"] for k in stocks)
            assert heat == pytest.approx(expected_heat, abs=2e-7)
            assert session._state.ledger.boundary_work_J == pytest.approx(work, abs=2e-7)
            assert sum(final[k]["U"] - initial[k]["U"] for k in stocks) == pytest.approx(
                heat - work, abs=2e-7
            )
            solids.append(actual["reactor"].species_amounts["NaCl(s)"])
        assert solids[0] == 0 < solids[2] < solids[1]
        assert solids[3] == 0
        assert session._state.samples.transfer_tools_used == 4
        assert session._state.ledger.time_s == 0
        assert session._state.ledger.cost >= 0.02
        assert len(session._state.samples.transfers) == 4
    finally:
        session.close()


@pytest.mark.parametrize("family", ["ideal", "Davies", "SIT", "Pitzer"])
def test_structural_zero_precursors_and_pure_water_are_exact(family):
    spec, composition = synthetic_world(family, association=True)
    for state in composition.physics.initial_states.values():
        result = solve_aqueous_mineral_equilibrium(
            spec,
            state.species_amounts,
            temperature_K=state.temperature_K,
            pressure_Pa=state.pressure_Pa,
        )
        assert result.success, result.to_dict()
        assert result.amounts_mol["NaCl(s)"] == 0
        assert result.amounts_mol["NaCl(aq)"] == 0
        assert result.saturation_indices["NaCl(s)"] is None
        assert result.diagnostics["structurally_blocked_reactions"] == ["dissolve", "associate"]


def test_mineral_bearing_source_moves_only_liquid_then_both_endpoints_equilibrate():
    _, session = setup("Davies", True)
    try:
        route(session, "feed-a", "feed-a-reactor", 0.5)
        route(session, "feed-b", "feed-b-reactor", 0.5)
        before = members(session)
        stocks = {k: s.species_amounts for k, s in before.items()}
        expected = transfer(stocks, "reactor", "feed-a", 0.1, "Davies")
        route(session, "reactor", "withdraw-reactor", 0.1)
        after = members(session)
        packet = session._state.metadata["last_material_packet"]
        assert "NaCl(s)" not in packet["amounts_mol"]
        assert packet["amounts_mol"] == pytest.approx(
            {
                k: 0.1 * v
                for k, v in before["reactor"].phases.phases["aqueous"].species_amounts_mol.items()
            }
        )
        for key in ("reactor", "feed-a"):
            assert after[key].species_amounts == pytest.approx(expected[key], abs=2e-9)
            assert set(after[key].phases.phases) == set(before[key].phases.phases)
            original = before[key].thermal.vessels[key]
            current = after[key].thermal.vessels[key]
            assert (
                current.energy_jacket_J
                - current.heat_loss_J
                - original.energy_jacket_J
                + original.heat_loss_J
            ) == pytest.approx(packet["vessel_heat_duty_J"][key], abs=2e-7)
    finally:
        session.close()


@pytest.mark.parametrize("defect", ["capacity", "solver", "drain", "mixing", "temperature"])
def test_route_failure_rolls_back_both_material_owners_and_resources(defect, monkeypatch):
    _, session = setup()
    try:
        state = session._state
        if defect == "capacity":
            reactor = state.inactive_vessels["reactor"]
            record = reactor.vessels.vessels["reactor"]
            reactor = reactor.replace(
                vessels=VesselLedger(
                    {"reactor": replace(record, max_volume_L=reactor.volume_L * 1.01)}
                )
            )
            session._state = state.replace(
                inactive_vessels={**state.inactive_vessels, "reactor": reactor}
            )
        elif defect == "temperature":
            session._state = state.replace(temperature_K=300.0)
        elif defect == "solver":
            from chemworld.runtime import aqueous_precipitation

            original = aqueous_precipitation.equilibrate_aqueous_mineral_inventory

            def fail_target(*args, **kwargs):
                if kwargs["vessel"].vessel_id == "reactor":
                    raise RuntimeError("injected target-only closure failure")
                return original(*args, **kwargs)

            monkeypatch.setattr(
                aqueous_precipitation, "equilibrate_aqueous_mineral_inventory", fail_target
            )
        before = members(session)
        ledger = session._state.ledger
        result = session.execute(
            {
                "operation": "route_material",
                "connection": 0,
                "transfer_fraction": 1.0 if defect == "drain" else 0.5,
                "mixing": 0 if defect == "mixing" else 1,
            }
        )
        assert result.info["transaction_status"] != "committed"
        for key, value in members(session).items():
            assert value.phases == before[key].phases
            assert value.species_amounts == before[key].species_amounts
            assert value.thermal == before[key].thermal
        for key in ("time_s", "energy_jacket_J", "heat_loss_J", "boundary_work_J"):
            assert getattr(session._state.ledger, key) == getattr(ledger, key)
    finally:
        session.close()


def test_snapshot_continuation_and_original_fixed_tp_wait():
    _, session = setup("Pitzer", True)
    replay = None
    try:
        route(session, "feed-a", "feed-a-reactor", 0.5)
        route(session, "feed-b", "feed-b-reactor", 0.5)
        replay = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        for instance in (session, replay):
            route(instance, "diluent", "dilute-reactor", 0.25)
            assert (
                instance.execute({"operation": "wait", "duration_s": 7}).info["transaction_status"]
                == "committed"
            )
        assert json.loads(json.dumps(session._state.to_dict())) == json.loads(
            json.dumps(replay._state.to_dict())
        )
        assert session._state.ledger.time_s == 7
        assert (
            not AqueousMineralRuntime()
            .advance(session._state, {"duration_s": 1}, heat=True)
            .success
        )
    finally:
        session.close()
        if replay:
            replay.close()


def test_restore_validator_checks_current_phase_geometry():
    _, composition = synthetic_world()
    state = composition.physics.initial_states["feed-a"]
    from chemworld.foundation.state_ledgers import PhaseLedger

    phase = state.phases.phases["aqueous"]
    broken = state.replace(
        phases=PhaseLedger(
            {**state.phases.phases, "aqueous": replace(phase, volume_L=phase.volume_L * 1.01)}
        ),
        volume_L=state.volume_L * 1.01,
    )
    with pytest.raises(ValueError, match="volume"):
        AqueousMineralRuntime().validate_state(broken)
    assert precipitation_record(None) is None


def test_independent_reference_has_no_precipitate_without_counterion():
    _, composition = synthetic_world("Davies", association=True)
    for state in composition.physics.initial_states.values():
        assert close(state.species_amounts, "Davies") == pytest.approx(state.species_amounts)


def test_actual_snapshot_cannot_replace_equilibrium_declaration():
    _, session = setup("Davies")
    try:
        snapshot = json.loads(json.dumps(session.snapshot().to_dict()))
        original = json.loads(json.dumps(snapshot))

        record = snapshot["world_state"]["equipment"]["feed-a:equilibrium"]
        record["settings"]["physical_model"]["database"]["activity_model"]["parameters"]["A"][
            "value"
        ] = 0.52
        with pytest.raises(ValueError, match="precipitation declaration differs"):
            ChemWorldRuntime.from_snapshot(snapshot)
        assert original != snapshot
        unchanged = ChemWorldRuntime.from_snapshot(original)
        unchanged.close()
    finally:
        session.close()
