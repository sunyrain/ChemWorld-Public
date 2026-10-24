"""A sourced descriptor-built mechanism reaches the original actual world consumer."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.catalysis_network import make_catalytic_slot
from chemworld.physchem.catalysis_thermochemistry import CatalyticThermodynamics
from chemworld.physchem.heat_transfer_units import ThermalSystemSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.service_factories import surface_processes
from chemworld.world.descriptor_catalysis import descriptor_catalysis_world
from reference.catalysis53_fixtures import mechanism


@pytest.mark.parametrize("isothermal", [True, False])
def test_world_wait_commits_original_inventories_and_exact_snapshot_continuation(isothermal):
    spec = mechanism()
    composition = descriptor_catalysis_world(
        spec,
        {"He": 0.002},
        capacities_mol={"A": 1e-4, "B": 2e-4},
        initial_adsorbates_mol={"bound_A": 2e-5, "bound_B": 1e-5},
        volume_L=1,
        temperature_K=350,
        support_heat_capacity_J_K=5,
        stop_temperature_K=440,
        maximum_pressure_Pa=1e6,
        isothermal=isothermal,
    )
    original, resumed = ChemWorldRuntime(composition=composition), None
    try:
        original.reset(seed=53)
        first = original.execute({"operation": "wait", "duration_s": 2})
        assert first.info["transaction_status"] == "committed", first.info
        resumed = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(original.snapshot().to_dict()))
        )
        for runtime in (original, resumed):
            second = runtime.execute({"operation": "wait", "duration_s": 3})
            assert second.info["transaction_status"] == "committed", second.info
            assert runtime._state.ledger.time_s == 5
            assert runtime._state.species_amounts["He"] != 0.002
            slot = runtime._state.state_slots.slots[0]
            total = (
                runtime._state.species_amounts["He"]
                + slot.values["bound_A"]
                + slot.values["bound_B"]
            )
            assert total == pytest.approx(0.00203, abs=1e-11)
            assert (
                runtime._state.phases.phases["gas"].species_amounts_mol
                == runtime._state.species_amounts
            )
        assert original._state.to_dict() == resumed._state.to_dict()
    finally:
        original.close()
        if resumed is not None:
            resumed.close()


def test_gas_caloric_profile_rejects_fixed_pressure_and_factory_selects_its_mechanism():
    spec = mechanism()
    thermal = CatalyticThermodynamics(spec.species_by_id, 5)
    with pytest.raises(ValueError, match=r"ideal.gas"):
        ThermalSystemSpec(thermal, pressure_law="fixed_pressure")
    slot = make_catalytic_slot(
        "first", "reactor", spec, capacities_mol={"A": 1e-4, "B": 2e-4}, initial_amounts_mol={}
    )
    other = make_catalytic_slot(
        "second",
        "reactor",
        replace(spec, mechanism_id="other"),
        capacities_mol={"A": 1e-4, "B": 2e-4},
        initial_amounts_mol={},
    )
    tables = {
        "surface": {spec.mechanism_id: spec},
        "thermal": {"reactor": SimpleNamespace(thermodynamics=thermal)},
    }
    context = SimpleNamespace(
        component=SimpleNamespace(parameters={"surface_id": spec.mechanism_id}),
        physics=SimpleNamespace(models_of=lambda kind: tables[kind]),
        state=SimpleNamespace(vessel_id="reactor", state_slots=StateSlots((slot, other))),
    )
    processes = surface_processes(context)
    assert len(processes) == 1 and processes[0].process_id.endswith(slot.spec.slot_id)
    context.state.state_slots = StateSlots((other,))
    with pytest.raises(ValueError, match="owned site"):
        surface_processes(context)
