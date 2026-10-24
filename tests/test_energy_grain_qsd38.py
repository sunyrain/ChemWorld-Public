"""Read-only QSD boundary, live inventory and serialization contracts."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.examples.rrkm_gas_network import example_inputs
from chemworld.physchem.energy_grain_binding import energy_grain_world
from chemworld.physchem.energy_grain_reduction import energy_grain_qsd, energy_grain_qsd_table
from chemworld.physchem.energy_grain_specs import (
    EnergyGrainBathGas,
    EnergyGrainMasterEquationSpec,
    EnergyGrainState,
    EnergyGrainWellSpec,
    EnergyTransferBath,
    InternalModeBaseline,
    MicrocanonicalGrainChannel,
    R,
)
from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec, GasReactorNodeSpec
from chemworld.physchem.rrkm_grain_binding import compile_rrkm_grains, rrkm_grain_world
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.energy_grain_analysis import analyze_energy_grain_world_state


def two_grain():
    return EnergyGrainMasterEquationSpec(
        "qsd-two-grain",
        "H2",
        InternalModeBaseline(30, 0, 40, ("vibration",)),
        (
            EnergyGrainWellSpec(
                "A",
                (EnergyGrainState("A0", 0), EnergyGrainState("A1", 4000, 2)),
                (EnergyTransferBath("He", 3000, 4000),),
            ),
        ),
        (EnergyGrainBathGas("He", "He", 2.5 * R, 0, 20),),
        (MicrocanonicalGrainChannel("exit", "A1", "P", 5),),
        (EnergyGrainState("P", -3000),),
    )


def query(model=None, **kwargs):
    options = {"temperature_K": 600, "bath_concentrations_mol_L": {"He": 0.01}}
    return energy_grain_qsd(model or two_grain(), **(options | kwargs))


def test_two_grain_rate_and_qsd_match_closed_quadratic():
    model = two_grain()
    downward = 30 * np.sqrt(2) / 3 * np.exp(-1)
    upward = 30 * np.sqrt(2) * 2 / 3 * np.exp(-1 - 4000 / (R * 600))
    trace = upward + downward + 5
    slow = 2 * upward * 5 / (trace + np.sqrt(trace**2 - 4 * upward * 5))
    result = query(model)
    assert result["effective_rate_s_inverse"] == pytest.approx(slow, rel=1e-13)
    assert result["qsd_fractions"]["A1"] == pytest.approx(slow / 5, rel=1e-13)
    assert result["second_decay_rate_s_inverse"] == pytest.approx(trace - slow)
    assert result["normalization"] == pytest.approx({"sum_q": 1, "left_dot_q": 1})


@pytest.mark.parametrize("kind", ("multiwell", "one-grain", "no-exit", "no-bath", "underflow"))
def test_explicit_unsupported_or_unresolved_boundaries(kind):
    model = two_grain()
    if kind == "multiwell":
        model = replace(
            model,
            wells=(
                *model.wells,
                replace(model.wells[0], well_id="B", grains=(EnergyGrainState("B0", 0),)),
            ),
        )
    elif kind == "one-grain":
        model = replace(model, wells=(replace(model.wells[0], grains=(model.wells[0].grains[1],)),))
    elif kind == "no-exit":
        model = replace(model, channels=())
    elif kind == "no-bath":
        with pytest.raises(ValueError, match="connected"):
            query(model, bath_concentrations_mol_L={})
        return
    else:
        model = replace(
            model,
            wells=(
                replace(
                    model.wells[0], grains=(EnergyGrainState("A0", 0), EnergyGrainState("A1", 1e9))
                ),
            ),
        )
    with pytest.raises(ValueError):
        query(model)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"temperature_K": True},
        {"temperature_K": 10},
        {"bath_concentrations_mol_L": {"He": -1}},
        {"bath_concentrations_mol_L": {"unknown": 1}},
        {"grain_amounts_mol": {"P": 1}},
        {"grain_amounts_mol": {"A0": -1}},
    ),
)
def test_bad_conditions_or_non_grain_initial_stock_are_rejected(kwargs):
    with pytest.raises(ValueError):
        query(**kwargs)


def test_arbitrary_initial_state_is_not_silently_replaced_by_qsd():
    amounts = {"A0": 0.01, "A1": 0}
    report = query(grain_amounts_mol=amounts)
    population = report["current_population"]
    assert population["l1_distance_to_qsd"] > 0.01
    assert population["slow_mode_amplitude_mol"] != pytest.approx(0.01)
    assert population["initial_product_fluxes_mol_s"] == {"P": 0}
    assert amounts == {"A0": 0.01, "A1": 0}
    assert report["validity"]["dynamic_applicability_assessed"] is False
    empty = query(grain_amounts_mol={})["current_population"]
    assert empty["fractions"] is None and empty["slow_mode_amplitude_mol"] == 0


def test_unrepresentable_initial_flux_is_not_reported_as_valid():
    with pytest.raises(ValueError, match="finite"):
        query(grain_amounts_mol={"A1": 1e308})


def test_table_retains_failed_cells_and_declares_bath_pressure():
    report = energy_grain_qsd_table(
        two_grain(),
        temperatures_K=(600, 5000),
        bath_pressures_Pa=(1e3, 1e4),
        bath_mole_fractions={"He": 1},
    )
    assert report["requested_cells"] == 4 and report["resolved_cells"] == 2
    assert [r["status"] for r in report["rows"]] == ["resolved"] * 2 + ["failed"] * 2
    assert report["rows"][0]["report"]["bath_pressure_Pa"] == pytest.approx(1000)
    assert report["interpolation_or_dynamic_rate_law_created"] is False
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_current_rrkm_world_and_json_query_are_exactly_read_only():
    composition = rrkm_grain_world(*example_inputs())
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=38)
        before = runtime._state.to_dict()
        report = analyze_energy_grain_world_state(runtime._state, composition)
        assert report["world_query"]["total_gas_pressure_Pa"] > report["bath_pressure_Pa"]
        assert runtime._state.to_dict() == before
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        assert analyze_energy_grain_world_state(restored._state, composition) == report
        assert json.loads(json.dumps(report, allow_nan=False)) == report
    finally:
        runtime.close()
        if restored:
            restored.close()


def test_current_flowing_bath_is_used_and_global_chemistry_mutation_is_rejected():
    composition = rrkm_grain_world(*example_inputs(flowing=True))
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=38)
        initial = analyze_energy_grain_world_state(runtime._state, composition)
        outcome = runtime.execute({"operation": "wait", "duration_s": 0.1})
        assert outcome.info["transaction_status"] == "committed"
        before = runtime._state.to_dict()
        current = analyze_energy_grain_world_state(runtime._state, composition)
        assert (
            current["bath_concentrations_mol_L"]["He"] > initial["bath_concentrations_mol_L"]["He"]
        )
        assert current["effective_rate_s_inverse"] != initial["effective_rate_s_inverse"]
        assert runtime._state.to_dict() == before
        network = composition.physics.network
        reaction = network.reactions[-1]
        altered = replace(reaction, rate_law=replace(reaction.rate_law, parameters={"k": 99}))
        changed = replace(
            composition,
            physics=replace(
                composition.physics,
                network=replace(network, reactions=(*network.reactions[:-1], altered)),
            ),
        )
        with pytest.raises(ValueError, match="differs"):
            analyze_energy_grain_world_state(runtime._state, changed)
    finally:
        runtime.close()


def test_world_without_original_energy_grain_source_is_rejected():
    model = two_grain()
    reactor = GasReactorNetworkSpec((GasReactorNodeSpec("cell", 1),), (), 600, "cell")
    composition = energy_grain_world(model, reactor, {"cell": {"A0": 0.01, "He": 0.01}})
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=38)
        network = composition.physics.network
        metadata = {k: v for k, v in network.metadata.items() if k != "energy_grain_declaration"}
        changed = replace(
            composition,
            physics=replace(composition.physics, network=replace(network, metadata=metadata)),
        )
        with pytest.raises(ValueError, match="declaration"):
            analyze_energy_grain_world_state(runtime._state, changed)
    finally:
        runtime.close()


def test_rrkm_table_retains_existing_finite_sparse_grid_boundary():
    model, _, _ = example_inputs()
    report = energy_grain_qsd_table(
        compile_rrkm_grains(model),
        temperatures_K=(400, 900),
        bath_pressures_Pa=(1000, 100000),
        bath_mole_fractions={"He": 0.8, "Ar": 0.2},
    )
    assert report["resolved_cells"] == 4
    assert all("no continuum" in r["report"]["validity"]["grid_scope"] for r in report["rows"])
