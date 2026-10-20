"""Prespecified actual TPD and constrained two-liquid equilibrium checks."""

import json
from math import log

import pytest
from scipy.optimize import brentq

from chemworld.physchem.equilibrium import ActivityModelSpec, equilibrate_liquids


def margules(amplitude):
    return ActivityModelSpec("synthetic-symmetric-margules", ("A", "B"), "margules",
                             {"A:A|B": amplitude, "A:B|A": amplitude})


@pytest.mark.parametrize("amplitude", [0, 1, 3])
@pytest.mark.parametrize("fraction", [.1, .3, .5, .7, .9])
def test_fixed_margules_common_tangent_block(amplitude, fraction):
    result = equilibrate_liquids(margules(amplitude), {"A": fraction, "B": 1-fraction},
                                 temperature_K=298.15)
    assert result.material_balance_error_mol <= 1e-8
    assert result.chemical_potential_error_RT <= 1e-7
    assert all(report.minimum_tpd >= -1e-7 for report in result.post_split_stability)
    if amplitude < 2:
        assert result.phase_status == "single_liquid"
        assert result.phases[0] == {"A": fraction, "B": 1-fraction}
    else:
        endpoint = brentq(lambda x: log(x/(1-x)) + amplitude*(1-2*x), 1e-6, .49,
                          xtol=1e-14)
        assert result.phase_status == "two_liquid"
        actual = [phase["A"] / sum(phase.values()) for phase in result.phases]
        assert actual == pytest.approx([endpoint, 1-endpoint], rel=1e-5, abs=1e-7)
        for component in ("A", "B"):
            assert sum(phase[component] for phase in result.phases) == pytest.approx(
                result.feed_amounts_mol[component], abs=1e-8)


@pytest.mark.parametrize("composition", [(.45, .45, .1), (.1, .1, .8), (.7, .2, .1)])
def test_fixed_ternary_nrtl_balances_and_post_split_stability(composition):
    ids = ("A", "B", "C")
    parameters = {f"alpha:{a}|{b}": .2 for a in ids for b in ids if a != b}
    parameters.update({f"tau:{a}|{b}": 3 if {a,b} == {"A", "B"} else 0
                       for a in ids for b in ids if a != b})
    model = ActivityModelSpec("fixed-ternary-nrtl", ids, "nrtl", parameters)
    feed = dict(zip(ids, composition, strict=True))
    result = equilibrate_liquids(model, feed, temperature_K=298.15)
    assert len(result.phases) <= 2
    for key, value in feed.items():
        assert sum(phase[key] for phase in result.phases) == pytest.approx(value, abs=1e-8)
    assert result.chemical_potential_error_RT <= 1e-7
    assert all(report.minimum_tpd >= -1e-7 for report in result.post_split_stability)


@pytest.mark.parametrize("fraction", [0, .01, .99, 1])
def test_phase_disappearance_and_zero_components(fraction):
    result = equilibrate_liquids(margules(3), {"A": fraction, "B": 1-fraction},
                                 temperature_K=298.15)
    assert result.phase_status == "single_liquid"
    assert result.phases == ({"A": fraction, "B": 1-fraction},)


def test_unresolved_tpd_starts_are_reported_as_failure():
    with pytest.raises(RuntimeError, match="unresolved starts"):
        equilibrate_liquids(margules(3), {"A": .3, "B": .7}, temperature_K=298.15,
                            max_iterations=1)


@pytest.mark.parametrize("amplitude", [0, 3])
def test_actual_mix_wash_finite_source_outlet_and_single_clock(amplitude):
    from chemworld.examples.native_liquid_research import synthetic_liquid_world
    from chemworld.runtime import ChemWorldRuntime

    session = ChemWorldRuntime(composition=synthetic_liquid_world(amplitude))
    try:
        session.reset(seed=12)
        before = dict(session._state.species_amounts)
        source_before = dict(session._state.inactive_vessels["wash_supply"].species_amounts)
        mixed = session.execute({"operation": "mix", "duration_s": 2, "stirring_speed_rpm": 600})
        assert mixed.info["transaction_status"] == "committed"
        assert session._state.ledger.time_s == 2
        assert len(session._state.phases.phases) == (1 if amplitude == 0 else 2)
        washed = session.execute({"operation": "wash", "wash_volume_L": 1e-5})
        assert washed.info["transaction_status"] == "committed"
        assert session._state.ledger.time_s == 2
        assert washed.info["material_balance"]["stock_consumption"]["wash_solvent_L"] == 1e-5
        source_after = session._state.inactive_vessels["wash_supply"]
        assert source_before["B"] - source_after.species_amounts["B"] == pytest.approx(.0002)
        removed = session._state.metadata["last_liquid_wash"]["removed_mol"]
        for key in before:
            actual = (session._state.species_amounts[key] + source_after.species_amounts[key]
                      + removed[key])
            assert actual == pytest.approx(before[key] + source_before[key], abs=1e-8)
        if amplitude == 0:
            assert not sum(removed.values())
            assert len(session._state.phases.phases) == 1
        else:
            # Independent common-tangent endpoint and lever rule, then finite
            # droplet entrainment. No reported solver residual is used.
            low = brentq(lambda x: log(x/(1-x)) + 3*(1-2*x), 1e-6, .49, xtol=1e-14)
            high = 1 - low
            phase_beta = (.001 - .0022 * low) / (high - low)
            phase_alpha = .0022 - phase_beta
            assert removed["A"] == pytest.approx(.99 * phase_alpha * low, rel=1e-5)
            assert removed["B"] == pytest.approx(.99 * phase_alpha * high, rel=1e-5)
        read = session.execute({"operation": "measure", "instrument": "phase_state"})
        assert read.info["transaction_status"] == "committed"
        assert session._state.ledger.time_s == 3
        readings = {c.channel_id: c for c in session.observe().channels}
        assert readings["liquid_phase_count"].value == (1 if amplitude == 0 else 2)
        assert readings["liquid_phase_count"].sampled_at.time_s == 3
        assert all(not p.solvents.volume_L for p in session._state.phases.phases.values())
    finally:
        session.close()


def test_liquid_snapshot_restores_model_and_deterministic_continuation():
    from chemworld.examples.native_liquid_research import synthetic_liquid_world
    from chemworld.runtime import ChemWorldRuntime

    session = ChemWorldRuntime(composition=synthetic_liquid_world())
    session.reset(seed=12)
    session.execute({"operation": "wash", "wash_volume_L": 1e-5})
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(session.snapshot().to_dict())))
    try:
        action = {"operation": "wait", "duration_s": 2, "stirring_speed_rpm": 600}
        first, other = session.execute(action), restored.execute(action)
        assert first.info["transaction_status"] == other.info["transaction_status"] == "committed"
        assert first.observation.to_dict() == other.observation.to_dict()
        assert session._state.to_dict() == restored._state.to_dict()
        transfer = session.execute({"operation": "route_material",
                                    "connection": "reactor-to-receiver",
                                    "transfer_fraction": .2, "mixing": 1})
        assert transfer.info["transaction_status"] == "committed"
        assert session._state.ledger.time_s == 2
    finally:
        session.close()
        restored.close()


def test_native_liquid_numerical_failure_retains_inventory_and_clock(monkeypatch):
    from chemworld.examples.native_liquid_research import synthetic_liquid_world
    from chemworld.physchem.phase_equilibrium_units import StrictLiquidSystemSpec
    from chemworld.runtime import ChemWorldRuntime

    session = ChemWorldRuntime(composition=synthetic_liquid_world())
    try:
        session.reset(seed=12)
        phases = session._state.phases.to_dict()
        def fail(*args, **kwargs):
            raise RuntimeError("injected liquid solver failure")
        monkeypatch.setattr(StrictLiquidSystemSpec, "solve", fail)
        result = session.execute({"operation": "mix", "duration_s": 2, "stirring_speed_rpm": 600})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert result.info["transaction_status"] == "rolled_back"
        assert session._state.phases.to_dict() == phases
        assert session._state.ledger.time_s == 0
    finally:
        session.close()


def test_native_wash_cannot_overdraw_its_finite_molecular_source():
    from chemworld.examples.native_liquid_research import synthetic_liquid_world
    from chemworld.runtime import ChemWorldRuntime

    session = ChemWorldRuntime(composition=synthetic_liquid_world())
    try:
        session.reset(seed=12)
        before, source = session._state.phases.to_dict(), session._state.inactive_vessels[
            "wash_supply"].to_dict()
        result = session.execute({"operation": "wash", "wash_volume_L": .001})
        assert result.info["operation_outcome"] == "rejected"
        assert session._state.phases.to_dict() == before
        assert session._state.inactive_vessels["wash_supply"].to_dict() == source
        assert session._state.ledger.time_s == 0
    finally:
        session.close()


def test_empirical_partition_drive_is_not_a_strict_stability_certificate():
    from chemworld.physchem.equilibrium import lle_phase_stability_diagnostic

    empirical = lle_phase_stability_diagnostic(
        {"A": .5, "B": .5}, partition_coefficients={"A": 100, "B": .01},
        aqueous_volume_L=1, organic_volume_L=1)
    strict = equilibrate_liquids(margules(0), {"A": .5, "B": .5}, temperature_K=298.15)
    assert empirical.phase_status == "two_liquid"
    assert empirical.to_dict()["diagnostic_kind"] == "empirical_partition_contact"
    assert not empirical.to_dict()["thermodynamic_stability_certified"]
    assert strict.phase_status == "single_liquid"
    assert strict.stability.minimum_tpd >= -1e-9


def test_remaining_ternary_instability_rejects_an_unsupported_third_liquid():
    ids = ("A", "B", "C")
    parameters = {f"alpha:{a}|{b}": .2 for a in ids for b in ids if a != b}
    parameters.update({f"tau:{a}|{b}": 3 for a in ids for b in ids if a != b})
    model = ActivityModelSpec("synthetic-three-liquid", ids, "nrtl", parameters)
    with pytest.raises(ValueError, match="unsupported additional phase"):
        equilibrate_liquids(model, dict.fromkeys(ids, 1/3), temperature_K=298.15)


def test_component_phase_label_order_and_start_order_preserve_physical_split(monkeypatch):
    from chemworld.physchem import equilibrium

    feed = {"A": .3, "B": .7}
    baseline = equilibrate_liquids(margules(3), feed, temperature_K=298.15)
    model = ActivityModelSpec("permuted", ("B", "A"), "margules", margules(3).parameters)
    original = equilibrium._liquid_simplex_seeds
    monkeypatch.setattr(equilibrium, "_liquid_simplex_seeds", lambda z: list(reversed(original(z))))
    permuted = equilibrate_liquids(model, {"B": .7, "A": .3}, temperature_K=298.15)
    def canonical(result):
        return sorted(([phase["A"], phase["B"]] for phase in result.phases), key=lambda p: p[0])
    for actual, expected in zip(canonical(permuted), canonical(baseline), strict=True):
        assert actual == pytest.approx(expected, rel=1e-5, abs=1e-8)
