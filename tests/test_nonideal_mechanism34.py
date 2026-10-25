"""Source binding, original finite thermal operations, events and rollback."""

import json
import subprocess
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest

from chemworld.foundation import WorldState
from chemworld.foundation.state_ledgers import VesselLedger, VesselRecord
from chemworld.physchem.coupled_system import CoupledState, ProcessContribution
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec, ThermalSystemSpec
from chemworld.physchem.mechanism_ideal_reference import MechanismIdealReference
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.physchem.molecular_mixture import MolecularIsochoricThermodynamics
from chemworld.physchem.molecular_reaction import (
    mechanism_caloric_model,
    molecular_reaction_process,
    molecular_reaction_rates,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.nonideal_mechanism_world import nonideal_mechanism_world
from chemworld.world.physical_world import CompositionPhysics
from reference.nonideal34_cases import THERMOS, inputs, residual, source

INITIAL = {"A": 1.0, "B": 0.3, "C": 0.2}


def world(*, shared=False, **kwargs):
    network, _ = inputs()
    return nonideal_mechanism_world(network, residual(), INITIAL, shared_clock=shared, **kwargs)


@pytest.mark.parametrize("family", THERMOS)
def test_original_thermo_is_the_only_caloric_reference_and_json_is_exact(family):
    network, thermo = inputs(family)
    initial = CoupledState(INITIAL, 600.0, 1.0)
    before = network.to_dict()
    molar = thermo.molar_state(initial)
    for reference in thermo.model.ideal_references:
        original = network.imported_gas_thermochemistry[reference.fluid_id].evaluate(600.0)
        state = reference.state(temperature_K=600.0, molar_density_mol_m3=1500.0)
        assert state.cp_J_mol_K == original.cp_J_mol_K
        assert (
            state.internal_energy_J_mol
            == original.enthalpy_J_mol - 600 * reference.gas_constant_J_mol_K
        )
        changed = reference.source_declaration
        changed["composition"]["H"] = 99
        assert reference.source_declaration["composition"]["H"] != 99
    restored = MolecularIsochoricThermodynamics.from_dict(json.loads(json.dumps(thermo.to_dict())))
    assert restored == thermo
    assert restored.molar_state(initial) == molar
    assert molecular_reaction_rates(network, restored, initial) == molecular_reaction_rates(
        network, thermo, initial
    )
    assert network.to_dict() == before
    thermal = ThermalSystemSpec(
        thermo,
        pressure_law="molecular",
        minimum_temperature_K=400,
        maximum_temperature_K=1400,
        stop_temperature_K=1300,
        maximum_pressure_Pa=1e8,
    )
    assert ThermalSystemSpec.from_dict(json.loads(json.dumps(thermal.to_dict()))) == thermal


@pytest.mark.parametrize(
    "defect", ["srk", "caloric-source", "charge", "liquid", "orders", "missing", "falloff"]
)
def test_explicit_reacting_scope_refuses_unsupported_bindings(defect):
    network, thermo = inputs()
    if defect == "srk":
        thermo = replace(
            thermo,
            model=replace(thermo.model, residual_model=replace(residual(), cubic_model="srk")),
        )
    elif defect == "caloric-source":
        raw = thermo.model.ideal_references[0].source_declaration
        raw["thermo"]["h0"] = "0 J/mol"
        thermo = replace(
            thermo,
            model=replace(
                thermo.model,
                ideal_references=(
                    MechanismIdealReference("A", raw),
                    *thermo.model.ideal_references[1:],
                ),
            ),
        )
    elif defect == "charge":
        network = replace(
            network,
            species=tuple(replace(s, charge=int(s.composition["H"] / 2)) for s in network.species),
        )
    elif defect == "liquid":
        species = replace(network.species[0], phase="liquid")
        network = replace(network, species=(species, *network.species[1:]))
    elif defect == "orders":
        reaction = replace(network.reactions[0], forward_orders={"A": 2})
        network = replace(network, reactions=(reaction, *network.reactions[1:]))
    elif defect == "missing":
        network = replace(network, species=network.species[:2], reactions=network.reactions[:1])
    else:
        reaction = replace(
            network.reactions[0],
            rate_law=replace(network.reactions[0].rate_law, equation_id="third_body_arrhenius"),
        )
        network = replace(network, reactions=(reaction, *network.reactions[1:]))
    with pytest.raises(ValueError):
        molecular_reaction_process(network, thermo)


def test_no_duplicate_heat_or_unbound_inventory_and_exact_absent_species():
    network, thermo = inputs()
    initial = CoupledState({"A": 1.0, "B": 0.0, "C": 0.0}, 600.0, 1.0)
    rates = molecular_reaction_rates(network, thermo, initial)
    assert rates["reaction-001"] > 0 and rates["reaction-002"] == 0
    process = molecular_reaction_process(network, thermo)
    contribution = process.evaluate(initial)
    assert contribution.legacy_reaction_energy_W is None
    assert sum(thermo.model.partial_inventory_energies(600, 0.001, initial.amounts_mol)) != 0
    with pytest.raises(ValueError, match="duplicate"):
        thermo.chemical_energy_rate_w(
            initial, {"A": -1.0, "B": 1.0}, (ProcessContribution(legacy_reaction_energy_W=0.0),)
        )
    with pytest.raises(ValueError, match="inventory"):
        molecular_reaction_rates(network, thermo, CoupledState({"A": 1.0}, 600.0, 1.0))


def test_thermal_pairing_source_bounds_and_mechanical_domain_are_enforced():
    _, thermo = inputs()
    with pytest.raises(ValueError, match="same EOS"):
        ThermalSystemSpec(thermo)
    with pytest.raises(ValueError, match="thermochemistry"):
        thermo.validate_temperature_interval(100.0, 900.0)
    restricted = replace(
        thermo,
        model=replace(
            thermo.model, residual_model=replace(residual(), temperature_range_K=(500.0, 900.0))
        ),
    )
    with pytest.raises(ValueError, match="residual"):
        restricted.validate_temperature_interval(400.0, 800.0)
    with pytest.raises(ValueError, match="unstable"):
        thermo.molar_state(CoupledState({"A": 3.0, "B": 3.0, "C": 2.0}, 200.0, 1.0))
    with pytest.raises(ValueError, match="complete"):
        network, _ = inputs()
        mechanism_caloric_model(
            network,
            replace(
                residual(),
                components=residual().components[:2],
                interaction=((0.0, 0.0), (0.0, 0.0)),
            ),
        )


@pytest.mark.parametrize("shared", [False, True])
def test_actual_heat_clock_phase_energy_and_full_snapshot_continuation(shared):
    runtime, restored = ChemWorldRuntime(composition=world(shared=shared)), None
    try:
        runtime.reset(seed=34)
        first = runtime.execute(
            {
                "operation": "queue_lab_heat" if shared else "heat",
                "duration_s": 1.0 if shared else 0.3,
                "target_temperature_K": 650.0,
            }
        )
        assert first.info["transaction_status"] == "committed", first.info
        if shared:
            assert (
                runtime.execute({"operation": "wait", "duration_s": 0.3}).info["transaction_status"]
                == "committed"
            )
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        action = {"operation": "wait", "duration_s": 0.3}
        for item in (runtime, restored):
            result = item.execute(action)
            assert result.info["transaction_status"] == "committed", result.info
            assert item._state.ledger.time_s == pytest.approx(0.6)
            assert item._state.temperature_K > 600.0
            assert item._state.ledger.energy_jacket_J > 0
            assert (
                next(iter(item._state.phases.phases.values())).species_amounts_mol
                == item._state.species_amounts
            )
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        runtime.close()
        if restored:
            restored.close()


def test_original_temperature_event_stops_before_full_requested_duration():
    runtime = ChemWorldRuntime(composition=world(maximum_temperature_K=605.0))
    try:
        runtime.reset(seed=34)
        result = runtime.execute({"operation": "wait", "duration_s": 1.0})
        assert result.info["transaction_status"] == "committed", result.info
        assert result.info["operation_outcome"] == "stopped"
        assert 0 < runtime._state.ledger.time_s < 1.0
        assert runtime._state.temperature_K == pytest.approx(604.0, abs=1e-7)
        assert all(value >= 0 for value in runtime._state.species_amounts.values())
    finally:
        runtime.close()


def test_failure_rolls_back_original_physics_and_queue_without_partial_commit():
    runtime = ChemWorldRuntime(composition=world(shared=True))
    try:
        runtime.reset(seed=34)
        runtime.execute(
            {"operation": "queue_lab_heat", "duration_s": 1.0, "target_temperature_K": 650.0}
        )
        before = deepcopy(runtime._state.to_dict())
        with patch(
            "chemworld.physchem.coupled_system.CoupledSystem.advance",
            side_effect=RuntimeError("34D solver failure"),
        ):
            result = runtime.execute({"operation": "wait", "duration_s": 0.3})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert result.info["actual_duration_s"] == 0
        after = runtime._state.to_dict()
        for value in (before, after):
            for key in ("ledger", "process"):
                value[key].pop("cost")
                value[key].pop("risk")
        assert before == after
    finally:
        runtime.close()


def test_original_ideal_mechanism_remains_a_distinct_unchanged_consumer():
    network, thermo = inputs()
    original = deepcopy(network.to_dict())
    ideal = network.reaction_rates(INITIAL, volume_L=1.0, temperature_K=600.0)
    nonideal = molecular_reaction_rates(network, thermo, CoupledState(INITIAL, 600.0, 1.0))
    assert nonideal != ideal
    assert convert_mechanism(source()).to_dict() == original
    assert network.reaction_rates(INITIAL, volume_L=1.0, temperature_K=600.0) == ideal


def test_irreversible_absent_reactant_never_creates_a_reverse_reaction():
    network, thermo = inputs(multiple=False, reversible=False)
    rates = molecular_reaction_rates(
        network, thermo, CoupledState({"A": 0.0, "B": 1.0, "C": 0.0}, 600.0, 1.0)
    )
    assert rates == {"reaction-001": 0.0}
    network.metadata["gas_thermochemistry"]["A"]["thermo"]["h0"] = "0 J/mol"
    with pytest.raises(ValueError, match="source exactly"):
        molecular_reaction_process(network, thermo)


def test_world_refuses_unsupported_surface_kinetics_before_execution():
    raw = world().to_dict()
    raw["components"].append({"id": "unsupported", "kind": "surface_reaction", "vessel": "reactor"})
    with pytest.raises(ValueError, match="bulk chemistry only"):
        ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(raw))


def test_two_independent_finite_vessels_age_on_one_shared_clock():
    spec = world(shared=True)
    physics = spec.physics
    thermo = physics.models_of("thermal")["reactor"].thermodynamics
    amounts = {"A": 0.7, "B": 0.4, "C": 0.1}
    pressure = thermo.pressure_pa(CoupledState(amounts, 650.0, 1.0))
    other = WorldState(
        amounts,
        1.0,
        650.0,
        pressure,
        "gas",
        "other",
        units=physics.initial_states["reactor"].units,
        vessels=VesselLedger(
            {"other": VesselRecord("other", "sealed", 1.0, 1400.0, 1e8, ("gas",), 650.0, pressure)}
        ),
    )
    raw = spec.to_dict()
    raw["vessels"].append({"id": "other", "capacity_L": 1.0})
    for component in spec.to_dict()["components"]:
        raw["components"].append({**component, "id": "other:" + component["id"], "vessel": "other"})
    raw["physics"] = CompositionPhysics(
        physics.network,
        {**physics.initial_states, "other": other},
        lab_clock=replace(physics.lab_clock, vessel_capacity=2),
        models={
            "thermal": {
                "reactor": physics.models_of("thermal")["reactor"],
                "other": physics.models_of("thermal")["reactor"],
            }
        },
    ).to_dict()
    runtime = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(raw))
    try:
        runtime.reset(seed=34)
        runtime.execute(
            {"operation": "queue_lab_heat", "duration_s": 1.0, "target_temperature_K": 650.0}
        )
        result = runtime.execute({"operation": "wait", "duration_s": 0.3})
        assert result.info["transaction_status"] == "committed", result.info
        assert runtime._state.vessel_elapsed_s == {"reactor": 0.3, "other": 0.3}
        assert runtime._state.inactive_vessels["other"].species_amounts != amounts
        assert runtime._state.species_amounts != INITIAL
        assert runtime._state.inactive_vessels["other"].ledger.energy_jacket_J == 0
    finally:
        runtime.close()


def test_actual_isothermal_device_books_compensation_from_the_same_inventory_energy():
    device = ThermalDeviceSpec(
        setpoint_K=600.0,
        minimum_setpoint_K=400.0,
        maximum_setpoint_K=1400.0,
        enabled=True,
        mode="isothermal",
    )
    spec = world(thermal_device=device)
    thermo = spec.physics.models_of("thermal")["reactor"].thermodynamics
    runtime = ChemWorldRuntime(composition=spec)
    try:
        runtime.reset(seed=34)
        initial_u = thermo.internal_energy_j(CoupledState(INITIAL, 600.0, 1.0))
        result = runtime.execute({"operation": "wait", "duration_s": 0.3})
        assert result.info["transaction_status"] == "committed", result.info
        state = runtime._state
        assert state.temperature_K == 600.0
        assert state.species_amounts != INITIAL
        final_u = thermo.internal_energy_j(CoupledState(state.species_amounts, 600.0, 1.0))
        assert final_u - initial_u == pytest.approx(state.ledger.energy_jacket_J, abs=2e-6)
        assert abs(state.ledger.energy_jacket_J) > 1.0
    finally:
        runtime.close()


def test_complete_json_continues_exactly_in_a_fresh_locked_process(tmp_path):
    runtime = ChemWorldRuntime(composition=world(shared=True))
    try:
        runtime.reset(seed=34)
        runtime.execute(
            {"operation": "queue_lab_heat", "duration_s": 1.0, "target_temperature_K": 650.0}
        )
        runtime.execute({"operation": "wait", "duration_s": 0.3})
        source_file, target_file = tmp_path / "snapshot.json", tmp_path / "continued.json"
        source_file.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf8")
        script = """import json,sys
from pathlib import Path
from chemworld.runtime import ChemWorldRuntime
r=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text(encoding="utf8")))
try:
 result=r.execute({"operation":"wait","duration_s":.3})
 assert result.info["transaction_status"]=="committed",result.info
 Path(sys.argv[2]).write_text(json.dumps(r._state.to_dict()),encoding="utf8")
finally:r.close()
"""
        subprocess.run(
            ["uv", "run", "--no-sync", "python", "-c", script, str(source_file), str(target_file)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        result = runtime.execute({"operation": "wait", "duration_s": 0.3})
        assert result.info["transaction_status"] == "committed"
        assert json.loads(target_file.read_text(encoding="utf8")) == json.loads(
            json.dumps(runtime._state.to_dict())
        )
    finally:
        runtime.close()
