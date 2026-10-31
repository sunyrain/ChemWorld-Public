"""Inventory summaries follow every actual particle size across cycle endpoints."""

import pytest

from chemworld.physchem.battery_models import F
from chemworld.physchem.battery_plating import advance_plating_cycle, plating_inventory_ledger
from chemworld.physchem.coupled_system import CoupledEvent
from test_battery_mpm_plating import model
from test_battery_plating import program


@pytest.mark.parametrize("mode", ("reversible", "irreversible"))
@pytest.mark.parametrize("single_negative_size", (False, True))
def test_mpm_ledger_and_cycle_history_include_all_actual_negative_bins(mode, single_negative_size):
    changes = {"negative_distribution": None} if single_negative_size else {}
    spec = model(mode=mode, **changes)
    initial = spec.initial_state()
    before = plating_inventory_ledger(spec, initial)
    assert before["negative_mobile_mol"] == pytest.approx(
        spec.negative.capacity_mol * spec.initial_negative_fraction, rel=1e-14)

    cycle = advance_plating_cycle(
        spec, initial, ((program(.2), .1), (program(-.1, start=.1), .1)))
    assert cycle.outcome == "success"
    history = cycle.inventory_history(spec)
    assert len(history) == 3
    for record, state in zip(
        history, (initial, *(segment.final_state for segment in cycle.segments)), strict=True,
    ):
        mobile = sum(amount for key, amount in state.amounts_mol.items()
                     if key.startswith("negative:size:"))
        assert mobile > 0
        metal = state.amounts_mol["plating:Li"] + state.amounts_mol["dead:Li"]
        assert record["time_s"] == state.time_s
        assert record["negative_mobile_mol"] == pytest.approx(mobile, rel=1e-14)
        assert record["negative_total_mol"] == pytest.approx(mobile + metal, rel=1e-14)
        assert record["faradaic_residual_C"] == pytest.approx(0.0, abs=1e-10)
        assert record["negative_total_mol"] + record["discharged_charge_C"] / F == (
            pytest.approx(before["negative_total_mol"], abs=1e-12))
    assert history[-1]["discharged_charge_C"] == pytest.approx(.01, abs=1e-12)


def test_mpm_inventory_history_retains_only_the_accepted_event_prefix():
    spec = model()
    cycle = advance_plating_cycle(
        spec, spec.initial_state(), ((program(.2), .1), (program(-.1, start=.1), .1)),
        control_events=(CoupledEvent("inventory-review-stop", lambda state: .04-state.time_s),),
    )
    assert cycle.outcome == "stopped"
    assert cycle.stopping_event == "inventory-review-stop"
    assert len(cycle.segments) == 1
    history = cycle.inventory_history(spec)
    assert len(history) == 2
    assert history[-1]["time_s"] == pytest.approx(.04, abs=1e-8)
    assert history[-1]["faradaic_residual_C"] == pytest.approx(0.0, abs=1e-10)
