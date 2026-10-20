"""Finite carbonate operations, existing transaction boundary and exact replay."""

import json
from copy import deepcopy

import pytest

from chemworld.physchem.aqueous_mineral import element_amounts
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_mineral import committed_speciation
from chemworld.world.aqueous_world import aqueous_world_spec


def execute(session, action):
    result = session.execute(action)
    assert result.info["transaction_status"] == "committed", result.info.get("error_message")
    if action["operation"] == "measure":
        assert result.info["measurement_status"] == "valid"
    return result


def global_elements(state):
    amounts = dict(state.species_amounts)
    for other in state.inactive_vessels.values():
        for k, n in other.species_amounts.items():
            amounts[k] += n
    return element_amounts(amounts)


@pytest.mark.parametrize("mode", ["sequential", "keyed"])
def test_finite_feed_sample_ph_and_exact_continuation(mode):
    session = ChemWorldRuntime(composition=aqueous_world_spec(), observation_noise_mode=mode)
    session.reset(seed=32)
    replay = None
    try:
        initial = global_elements(session._state)
        execute(session, {"operation": "wait", "duration_s": 2})
        before = deepcopy(session._state.species_amounts)
        measured = execute(session, {"operation": "measure", "instrument": "aqueous_ph"})
        assert session._state.species_amounts == pytest.approx(before, rel=1e-12, abs=1e-18)
        channels = {c.channel_id: c.value for c in measured.observation.channels}
        assert abs(channels["pH"] - committed_speciation(session._state).pH) <= 0.1
        assert session._state.ledger.sample_consumed_L == 0
        execute(
            session, {"operation": "add_component", "component": "HCl-solution", "amount_mol": 0.01}
        )
        assert committed_speciation(session._state).pH < 7
        replay = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        removed = None
        for action in (
            {"operation": "wait", "duration_s": 3},
            {"operation": "add_component", "component": "NaOH-solution", "amount_mol": 0.008},
            {"operation": "sample", "sample_volume_L": 0.001},
            {"operation": "measure", "instrument": "aqueous_ph"},
        ):
            before = session._state
            if action["operation"] == "sample":
                liquid = before.phases.phases["aqueous"]
                removed = element_amounts(
                    {k: n * 0.001 / liquid.volume_L for k, n in liquid.species_amounts_mol.items()}
                )
            left, right = execute(session, action), execute(replay, action)
            assert left.info == right.info
            assert left.observation.native_record() == right.observation.native_record()
            assert session._state.to_dict() == replay._state.to_dict()
            if action["operation"] == "sample":
                assert session._state.species_amounts["Calcite"] == pytest.approx(
                    before.species_amounts["Calcite"], rel=1e-10, abs=1e-14
                )
            elements = global_elements(session._state)
            for k, value in initial.items():
                expected = value - (removed[k] if removed else 0)
                assert abs(elements[k] - expected) <= 1e-10 * max(value, 1e-12)
            assert session._state.pressure_Pa == 101325
            assert session._state.temperature_K == 298.15
            assert all(p.solvents.volume_L == 0 for p in session._state.phases.phases.values())
        assert session._state.ledger.time_s == 7
        assert session._state.ledger.sample_consumed_L == 0.001
        assert session._state.ledger.energy_jacket_J == 0
        assert session._state.ledger.heat_reaction_J == 0
    finally:
        session.close()
        if replay:
            replay.close()


@pytest.mark.parametrize("failure", ["overstock", "temperature", "numerical"])
def test_failed_native_operations_preserve_physical_state(failure, monkeypatch):
    from chemworld.runtime import aqueous_mineral

    session = ChemWorldRuntime(composition=aqueous_world_spec())
    session.reset(seed=32)
    try:
        action = {"operation": "wait", "duration_s": 2}
        if failure == "overstock":
            action = {"operation": "add_component", "component": "HCl-solution", "amount_mol": 1}
        elif failure == "temperature":
            session._state = session._state.replace(temperature_K=310)
        else:

            def fail(*args, **kwargs):
                raise RuntimeError("injected aqueous numerical failure")

            monkeypatch.setattr(aqueous_mineral, "solve_carbonate_equilibrium", fail)
        before = session._state
        result = session.execute(action)
        assert result.info["transaction_status"] != "committed"
        after = session._state
        assert after.phases == before.phases
        assert after.equipment == before.equipment
        assert after.inactive_vessels == before.inactive_vessels
        assert after.temperature_K == before.temperature_K
        assert after.pressure_Pa == before.pressure_Pa
        assert after.ledger.time_s == before.ledger.time_s
    finally:
        session.close()


@pytest.mark.parametrize("identity", ["water_formula", "proton_charge"])
def test_authored_species_identity_cannot_redefine_the_fixed_chemistry(identity):
    payload = aqueous_world_spec().to_dict()
    species = payload["physics"]["network"]["species"]
    if identity == "water_formula":
        species[0]["formula"] = "H2"
        species[0]["composition"] = {"H": 2.0}
    else:
        species[1]["charge"] = 2
    with pytest.raises(ValueError, match="species identities"):
        ChemWorldRuntime(composition=payload)
def test_direct_thermal_rejection_returns_the_common_result_contract():
    from chemworld.runtime.aqueous_mineral import AqueousMineralRuntime

    result = AqueousMineralRuntime().advance(None, {"duration_s": 10}, heat=True)
    assert result.outcome == "rejected"
    assert result.failure_reason == "aqueous_temperature_data_unavailable"
    assert result.state is None
    assert result.diagnostics == {}
