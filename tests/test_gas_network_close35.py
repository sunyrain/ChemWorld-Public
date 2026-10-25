"""Species-H contracts, original steady boundaries and real read-only recovery."""

import importlib
import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.examples.gas_network_steady import example_scales, example_world
from chemworld.physchem.gas_network_sensitivity import (
    GasSensitivityParameter,
    GasSensitivityPolicy,
    gas_network_trajectory_sensitivities,
)
from chemworld.physchem.gas_network_steady import gas_network_steady_state
from chemworld.physchem.gas_reactor_network import gas_inventory_key, pack_gas_reactor_network_state
from chemworld.physchem.gas_thermochemistry_parameters import (
    GasThermochemistryParameter,
    species_enthalpy_parameter,
    with_species_enthalpy,
)
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_network_steady import analyze_gas_network_steady_world


@pytest.fixture(scope="module")
def cases():
    path = str(Path(__file__).parent / "reference")
    sys.path.insert(0, path)
    yield importlib.import_module("gas_network_close35_cases")
    sys.path.remove(path)


@pytest.mark.parametrize(
    "species,step,reference",
    [
        ("", 1, 300),
        (12, 1, 300),
        (" A", 1, 300),
        ("A", 0, 300),
        ("A", True, 300),
        ("A", 1, float("inf")),
    ],
)
def test_parameter_validation(species, step, reference):
    with pytest.raises(ValueError):
        GasThermochemistryParameter(species, step, reference)


def test_signed_and_dimensionless_original_thermo(cases):
    source = importlib.import_module("mechanism34_cases")
    raw = source.payload(thermo="piecewise-Gibbs")
    for species in raw["species"]:
        model = species["thermo"]
        model["dimensionless"] = True
        model["data"] = {
            t: float(g.split()[0]) * 1000 / (cases.R * t) for t, g in model["data"].items()
        }
    network = convert_mechanism(raw)
    before = deepcopy(network.to_dict())
    p = GasThermochemistryParameter("A", 1)
    baseline = species_enthalpy_parameter(network, p)
    for target in (0.0, -5000.0, 5000.0):
        changed = with_species_enthalpy(network, p, target)
        assert species_enthalpy_parameter(changed, p) == pytest.approx(target, abs=1e-9)
        replay = type(network).from_dict(changed.to_dict())
        for t in (350, 700, 1200):
            a = network.imported_gas_thermochemistry["A"].evaluate(t)
            b = replay.imported_gas_thermochemistry["A"].evaluate(t)
            assert b.enthalpy_J_mol - a.enthalpy_J_mol == pytest.approx(target - baseline)
            assert b.gibbs_J_mol - a.gibbs_J_mol == pytest.approx(target - baseline)
            assert b.cp_J_mol_K == pytest.approx(a.cp_J_mol_K)
            assert b.entropy_J_mol_K == pytest.approx(a.entropy_J_mol_K)
    assert network.to_dict() == before


def test_original_default_units_and_missing_species(cases):
    raw = cases.document()
    raw["species"][0]["thermo"]["h0"] = -1234000  # source J/kmol
    n = convert_mechanism(raw)
    p = GasThermochemistryParameter("A", 10)
    assert species_enthalpy_parameter(n, p) == pytest.approx(-1234)
    assert species_enthalpy_parameter(with_species_enthalpy(n, p, 0), p) == pytest.approx(0)
    with pytest.raises(ValueError, match="imported species"):
        species_enthalpy_parameter(n, GasThermochemistryParameter("absent", 1))


def test_mixed_log_and_zero_enthalpy_and_budget(cases):
    _, spec, network, amounts = cases.closed(True)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    # Imported thermo reverse is coupled; old explicit positive-prefactor path
    # uses its existing concentration Arrhenius contract in the old regression.
    parameter = GasThermochemistryParameter("A", 10)
    result = gas_network_trajectory_sensitivities(spec, network, state, (parameter,), (0.1,))
    assert result["status"] == "complete_finite_difference"
    assert result["parameters"][0]["baseline_value"] == 0
    assert all(
        c["normalized"] is None
        for c in result["responses"][0]["records"][0]["coefficients"].values()
    )
    assert result["responses"][0]["records"][0]["coefficients"]["node:r:temperature"]["dy_dp"] > 0
    incomplete = gas_network_trajectory_sensitivities(
        spec, network, state, (parameter,), (0.1, 0.2), policy=GasSensitivityPolicy(max_advances=1)
    )
    assert incomplete["status"] == "incomplete_finite_difference"
    assert incomplete["work"]["attempted_advances"] == 1
    assert all(row["coefficients"] is None for row in incomplete["responses"][0]["records"])


def test_old_log_response_unchanged_in_mixed_report(cases):
    raw = cases.document(irreversible=True)
    _, spec, _, amounts = cases.closed()
    network = convert_mechanism(raw)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    old = GasSensitivityParameter("reaction", network.reactions[0].reaction_id, "A")
    r1 = gas_network_trajectory_sensitivities(spec, network, state, (old,), (0.1,))
    r2 = gas_network_trajectory_sensitivities(
        spec, network, state, (old, GasThermochemistryParameter("B", 10)), (0.1,)
    )
    assert r1["responses"][0] == r2["responses"][0]
    assert r1["trajectories"] == r2["trajectories"][:5]


def steady_case(cases):
    _, spec, network, amounts = cases.closed(True)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    return spec, network, state, dict.fromkeys(state.amounts_mol, 0.02)


def test_original_adiabatic_steady_and_budget(cases):
    spec, network, state, scales = steady_case(cases)
    result = gas_network_steady_state(spec, network, state, amount_scales_mol=scales)
    assert result["status"] == "steady_state_found"
    assert result["temperature_K"] > 500
    assert abs(result["energy_invariant_errors_J"][0]) < 1e-7
    assert result["energy_residual_W"] < 1e-6
    failed = gas_network_steady_state(
        spec, network, state, amount_scales_mol=scales, maximum_rate_evaluations=1
    )
    assert failed["status"] == "failed"
    assert "budget" in failed["failure"]
    assert failed["original_rate_evaluations"] == 1


@pytest.mark.parametrize(
    "options",
    [
        {"held_vessel_ids": ("missing",)},
        {"held_vessel_ids": ("r",)},
        {"maximum_unknowns": 0},
        {"maximum_rate_evaluations": True},
        {"derivative_step": 0},
        {"temperature_scale_K": float("nan")},
    ],
)
def test_bad_analysis_boundaries_rejected(cases, options):
    spec, network, state, scales = steady_case(cases)
    with pytest.raises(ValueError):
        gas_network_steady_state(spec, network, state, amount_scales_mol=scales, **options)


def test_held_vessels_retain_real_derivatives_and_full_closed_failure(cases):
    _, spec, network, amounts = cases.stirred(900)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    scales = {gas_inventory_key("r", s): 0.01 for s in ("A", "B")}
    result = gas_network_steady_state(
        spec, network, state, amount_scales_mol=scales, held_vessel_ids=("feed", "sink")
    )
    assert result["status"] == "steady_state_found"
    assert result["scope"] == "held-boundary-internal-steady"
    assert result["material_rates_mol_s"][gas_inventory_key("feed", "A")] < 0
    assert result["material_rates_mol_s"][gas_inventory_key("sink", "B")] > 0
    closed = gas_network_steady_state(
        spec,
        network,
        state,
        amount_scales_mol=dict.fromkeys(state.amounts_mol, 0.01),
        maximum_rate_evaluations=100,
    )
    assert closed["status"] == "failed"
    reacting = replace(
        spec,
        nodes=tuple(
            replace(n, chemistry_enabled=True) if n.vessel_id == "feed" else n for n in spec.nodes
        ),
    )
    with pytest.raises(ValueError, match="nonreacting"):
        gas_network_steady_state(
            reacting, network, state, amount_scales_mol=scales, held_vessel_ids=("feed", "sink")
        )


def test_surface_slots_are_original_and_validated(cases):
    _, spec, network, amounts, slots = cases.surface()
    state = pack_gas_reactor_network_state(spec, network, amounts, surface_slots=slots)
    result = gas_network_steady_state(
        spec,
        network,
        state,
        surface_slots=slots,
        amount_scales_mol=dict.fromkeys(state.amounts_mol, 0.01),
    )
    assert result["status"] == "steady_state_found"
    assert result["reachable_dimension"] == 3
    with pytest.raises(ValueError, match="slots"):
        gas_network_steady_state(
            spec, network, state, amount_scales_mol=dict.fromkeys(state.amounts_mol, 0.01)
        )


def test_world_readonly_selected_member_and_fresh_process(tmp_path):
    composition = example_world()
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=35)
        runtime.execute({"operation": "wait", "duration_s": 0.1})
        runtime.execute({"operation": "select_vessel", "vessel": "cold"})
        before = runtime._state.to_dict()
        result = analyze_gas_network_steady_world(
            runtime._state, composition, amount_scales_mol=example_scales()
        )
        assert result["status"] == "steady_state_found"
        assert result["world_consumer"]["selected_vessel"] == "cold"
        assert result["world_consumer"]["anchor_vessel"] == "hot"
        assert runtime._state.to_dict() == before
        assert result["start_time_s"] == 0.1
        path = tmp_path / "input.json"
        path.write_text(
            json.dumps({"state": before, "composition": composition.to_dict()}), encoding="utf-8"
        )
        script = """import json,sys
from chemworld.foundation import WorldState
from chemworld.world.composition import WorldCompositionSpec
from chemworld.runtime.gas_network_steady import analyze_gas_network_steady_world
from chemworld.examples.gas_network_steady import example_scales
p=json.load(open(sys.argv[1],encoding="utf-8"))
s=WorldState.from_dict(p["state"])
r=analyze_gas_network_steady_world(s,WorldCompositionSpec.from_dict(p["composition"]),amount_scales_mol=example_scales())
assert json.loads(json.dumps(s.to_dict()))==p["state"]
assert "cantera" not in sys.modules
print(json.dumps(r,sort_keys=True,allow_nan=False))
"""
        restored = subprocess.run(
            [sys.executable, "-c", script, str(path)], check=False, capture_output=True, text=True
        )
        assert restored.returncode == 0, restored.stderr
        assert json.loads(restored.stdout) == result
        stopped = runtime._state.replace(terminated=True)
        with pytest.raises(ValueError, match="stopped"):
            analyze_gas_network_steady_world(
                stopped, composition, amount_scales_mol=example_scales()
            )
    finally:
        runtime.close()


def test_trial_extension_never_accepts_outside_original_temperature(cases):
    from chemworld.physchem.gas_reactor_network import gas_reactor_network_system

    raw, spec, _, amounts = cases.closed(True)
    for s in raw["species"]:
        s["thermo"]["T-min"] = 300
    network = convert_mechanism(raw)
    spec = replace(
        spec,
        energy_nodes=(
            replace(
                spec.energy_nodes[0],
                minimum_temperature_K=301,
                environment_temperature_K=200,
                loss_conductance_W_K=5000,
            ),
        ),
    )
    state = pack_gas_reactor_network_state(spec, network, amounts)
    system = gas_reactor_network_system(spec, network)
    system = replace(system, policy=replace(system.policy, first_step_s=0.0001))
    result = system.advance(state, 0.001)
    assert result.stopping_event == "gas:r:minimum_temperature"
    assert result.final_state.time_s < 0.001
    assert result.final_state.temperature_K == pytest.approx(301, abs=1e-8)
    assert all(s.temperature_K >= 301 - 1e-8 for s in result.trajectory)
    extension = result.solver_diagnostic["rhs_state_extension"]
    assert extension["extended_evaluations"] > 0
    assert extension["maximum_temperature_offset_K"] > 0
    assert not extension["accepted_states_projected"]
    assert result.solver_diagnostic["total_reported_rhs_evaluations"] == (
        result.solver_diagnostic["nfev"]
        + result.solver_diagnostic["feedback_jacobian_rhs_evaluations"]
    )
    with pytest.raises(ValueError, match="temperature limits"):
        system.validate_state(replace(state, temperature_K=299))


def test_independent_columns_and_extension_contract(cases):
    from chemworld.physchem.gas_reactor_network import gas_reactor_network_system

    spec, network, state, _ = steady_case(cases)
    system = gas_reactor_network_system(spec, network)
    mask = system.jacobian_sparsity().toarray()
    offset = len(state.amounts_mol) + 5
    for index, c in enumerate(system.coordinates):
        if c.name in system.rate_independent_coordinates:
            assert not mask[:, offset + index].any()
    with pytest.raises(ValueError, match="declared coordinates"):
        replace(system, rate_independent_coordinates=("unknown",))

    def invalid(trial):
        return replace(trial, volume_L=trial.volume_L * 2)

    with pytest.raises(ValueError, match="may only extend"):
        replace(system, trial_state_extension=invalid).advance(state, 0.1)


def test_original_surface_world_consumer_keeps_owned_slots(cases):
    from chemworld.world.gas_reactor_network import gas_reactor_network_world

    _, spec, network, amounts, slots = cases.surface()
    initial = {
        a.owner_id: {s: slot.values[s] for s in a.mechanism.adsorbates}
        for a, slot in zip(spec.surfaces, slots, strict=True)
    }
    composition = gas_reactor_network_world(
        spec, network, amounts, surface_amounts_by_owner=initial
    )
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=35)
        before = runtime._state.to_dict()
        state = pack_gas_reactor_network_state(spec, network, amounts, surface_slots=slots)
        report = analyze_gas_network_steady_world(
            runtime._state, composition, amount_scales_mol=dict.fromkeys(state.amounts_mol, 0.01)
        )
        assert report["status"] == "steady_state_found"
        assert report["reachable_dimension"] == 3
        assert runtime._state.to_dict() == before
    finally:
        runtime.close()


def test_running_or_queued_control_is_not_a_steady_boundary(cases):
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
    from chemworld.world.gas_reactor_network import gas_reactor_network_world

    _, spec, network, amounts = cases.closed(True)
    composition = gas_reactor_network_world(
        spec,
        network,
        amounts,
        anchor_thermal_device=ThermalDeviceSpec(enabled=False, maximum_setpoint_K=1000),
    )
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=35)
        runtime.execute(
            {"operation": "queue_lab_heat", "duration_s": 1, "target_temperature_K": 600}
        )
        assert runtime._state.lab_clock.queued
        before = runtime._state.to_dict()
        with pytest.raises(ValueError, match="queued or running"):
            analyze_gas_network_steady_world(
                runtime._state,
                composition,
                amount_scales_mol={gas_inventory_key("r", s): 0.02 for s in ("A", "B")},
            )
        assert runtime._state.to_dict() == before
    finally:
        runtime.close()
