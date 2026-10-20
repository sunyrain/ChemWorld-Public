"""Two conserved site classes, shared interaction energy and actual sessions."""

from dataclasses import replace
from math import log

import numpy as np
import pytest

from chemworld.foundation.state_ownership import StateSlot, StateSlots
from chemworld.physchem.inventory_interactions import InventoryInteraction
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.surface_rates import (
    MULTISITE_REACTIONS,
    adsorbate_key,
    evaluate_surface_rates,
    gas_amounts_from_pressure,
    make_surface_system,
    standard_surface_gibbs,
    surface_state,
    surface_thermodynamics,
)
from chemworld.physchem.surface_specs import (
    MULTISITE_ADSORBATES,
    MultisiteSurfaceSpec,
    make_surface_slot,
    surface_spec_from_dict,
    surface_species_energies,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.coupled_thermal import local_thermodynamics
from chemworld.world.surface_world import surface_world_spec


def interaction_matrix(strength):
    matrix = np.eye(4) * strength
    matrix[0, 2] = matrix[2, 0] = matrix[1, 3] = matrix[3, 1] = strength / 2
    return tuple(tuple(row) for row in matrix)


def fixture(strength=5000, temperature=350, *, mobile=False):
    spec = MultisiteSurfaceSpec(interaction_matrix_J_mol=interaction_matrix(strength))
    slot = make_surface_slot(
        "carrier",
        "reactor",
        spec=spec,
        initial_coverages=(0.3, 0.2),
        second_coverages=(0.2, 0.3),
        fixed=not mobile,
        phase_id="gas" if mobile else None,
    )
    gas = gas_amounts_from_pressure(
        {"CO": 0.2, "O2": 0.1, "CO2": 0.05, "N2": 0.65},
        pressure_Pa=100000,
        temperature_K=temperature,
        volume_L=0.01,
    )
    return spec, slot, surface_state(gas, (slot,), temperature_K=temperature, volume_L=0.01)


@pytest.mark.parametrize("strength", [-5000, 0, 5000])
def test_common_free_energy_gradient_and_each_pair_detailed_balance(strength):
    spec, slot, state = fixture(strength)
    term = InventoryInteraction.from_slot(slot, slot.spec.energy_interactions[0])
    _, gradient = term.energy_and_gradient(state.amounts_mol)
    for key in term.keys:
        delta = 1e-11
        plus, minus = dict(state.amounts_mol), dict(state.amounts_mol)
        plus[key] += delta
        minus[key] -= delta
        fd = (term.energy_and_gradient(plus)[0] - term.energy_and_gradient(minus)[0]) / (2 * delta)
        assert fd == pytest.approx(gradient[key], abs=1e-3)
    rates = evaluate_surface_rates(state, slot, spec)
    potentials = standard_surface_gibbs(spec, state.temperature_K)
    for key in MULTISITE_ADSORBATES:
        potentials[key] += gradient[adsorbate_key(slot, key)]
    activity = {k: v / 100000 for k, v in rates.partial_pressures_Pa.items()}
    activity.update(rates.coverages)
    for i, (reactants, products, _) in enumerate(MULTISITE_REACTIONS):
        delta = sum(v * (potentials[k] + R * 350 * log(activity[k])) for k, v in products.items())
        delta -= sum(v * (potentials[k] + R * 350 * log(activity[k])) for k, v in reactants.items())
        forward, reverse = rates.progress_rates_mol_s[2 * i : 2 * i + 2]
        assert log(forward / reverse) == pytest.approx(-delta / (R * 350), abs=1e-8)


def test_standard_gibbs_derivative_matches_caloric_enthalpy():
    spec, _, _ = fixture()
    temperature, dt = 375, 0.001
    values = standard_surface_gibbs(spec, temperature)
    plus, minus = (standard_surface_gibbs(spec, t) for t in (temperature + dt, temperature - dt))
    for key, caloric in surface_species_energies(spec).items():
        enthalpy = values[key] - temperature * (plus[key] - minus[key]) / (2 * dt)
        expected = caloric.u_ref_J_mol + caloric.cv_J_mol_K * (temperature - 350)
        if key in ("CO", "O2", "CO2", "N2"):
            expected += R * temperature
        assert enthalpy == pytest.approx(expected, abs=1e-4)


def test_authored_runtime_cannot_use_a_conflicting_caloric_reference():
    world = surface_world_spec(surface_spec=MultisiteSurfaceSpec())
    models = dict(world.physics.thermal_models)
    model = models["reactor"]
    models["reactor"] = replace(
        model, thermodynamics=replace(model.thermodynamics, reference_temperature_K=300)
    )
    with pytest.raises(ValueError, match="same declared table"):
        ChemWorldRuntime(
            composition=replace(world, physics=replace(world.physics, thermal_models=models))
        )


def test_interaction_extensivity_and_owned_split_merge_json():
    spec, slot, state = fixture(mobile=True)
    whole = surface_thermodynamics((slot,), support_heat_capacity_J_K=0, spec=spec)
    # Use only the carrier correction to isolate extensivity from the bulk gas.
    energy = whole.interactions[0].energy_and_gradient(state.amounts_mol)[0]
    parts = [slot.portion(f, "reactor") for f in (0.3, 0.7)]
    pieces = []
    for part in parts:
        packed = {adsorbate_key(part, k): part.values[k] for k in MULTISITE_ADSORBATES}
        pieces.append(
            InventoryInteraction.from_slot(
                part, part.spec.energy_interactions[0]
            ).energy_and_gradient(packed)[0]
        )
    assert sum(pieces) == pytest.approx(energy, abs=1e-15)
    merged = parts[0].merge(parts[1])
    assert merged.values == pytest.approx(slot.values)
    assert StateSlot.from_dict(slot.to_dict()) == slot
    assert surface_spec_from_dict(spec.to_dict()) == spec


def test_second_site_removal_and_two_separate_capacity_checks():
    spec = MultisiteSurfaceSpec(second_site_ratio=0)
    slot = make_surface_slot("c", "reactor", spec=spec)
    gas = gas_amounts_from_pressure(
        {"CO": 0.2, "O2": 0.1, "N2": 0.7}, pressure_Pa=100000, temperature_K=350, volume_L=0.01
    )
    rates = evaluate_surface_rates(
        surface_state(gas, (slot,), temperature_K=350, volume_L=0.01), slot, spec
    )
    assert rates.progress_rates_mol_s[6:] == (0,) * 8
    _, full, _ = fixture()
    with pytest.raises(ValueError, match="capacity"):
        replace(full, values={**full.values, "CO_B": full.values["sites_B_mol"]})
    with pytest.raises(ValueError, match="symmetric"):
        MultisiteSurfaceSpec(
            interaction_matrix_J_mol=((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        )


@pytest.mark.parametrize("isothermal", [True, False])
def test_coupled_new_rates_conserve_elements_sites_and_interaction_heat(isothermal):
    spec, slot, start = fixture()
    system = make_surface_system((slot,), spec=spec, isothermal=isothermal)
    result = system.advance(start, 20)
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-7)
    final = result.final_state
    for names in (("CO_ads", "O_ads"), ("CO_B", "O_B")):
        assert sum(final.amounts_mol[adsorbate_key(slot, k)] for k in names) <= slot.values[
            "sites_mol"
        ] * (1 + 1e-10)
    assert max(abs(v) for v in result.conservation_residuals.values()) < 1e-12
    if isothermal:
        assert final.temperature_K == 350
        delta_u = system.thermodynamics.internal_energy_j(
            final
        ) - system.thermodynamics.internal_energy_j(start)
        assert final.boundary_heat_J == pytest.approx(delta_u, abs=1e-7)


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", result.info
    return result


def native_session(*, mobile=False):
    spec = MultisiteSurfaceSpec(interaction_matrix_J_mol=interaction_matrix(5000))
    world = surface_world_spec(surface_spec=spec, mobile=mobile)
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=38)
    execute(runtime, "add_component", component="CO-feed", amount_mol=2e-5)
    execute(runtime, "add_component", component="O2-feed", amount_mol=1e-5)
    return runtime, world


def test_actual_session_material_transfer_keeps_both_site_classes_and_total_energy():
    runtime, world = native_session(mobile=True)
    execute(runtime, "heat", target_temperature_K=400, duration_s=20)
    before = runtime._state

    def energy(state):
        locals_ = {state.vessel_id: state, **state.inactive_vessels}
        return sum(
            thermo.internal_energy_j(packed)
            for name in ("reactor", "receiver")
            for thermo, packed in (
                local_thermodynamics(locals_[name], world.physics.thermal_models[name]),
            )
        )

    before_u = energy(before)
    slot = before.state_slots.slots[0]
    execute(
        runtime,
        "route_material",
        connection="reactor-to-receiver",
        transfer_fraction=0.5,
        mixing="allow",
    )
    after = runtime._state
    target = after.inactive_vessels["receiver"]
    for key in ("sites_mol", "sites_B_mol", *MULTISITE_ADSORBATES):
        assert after.state_slots.slots[0].values[key] + target.state_slots.slots[0].values[
            key
        ] == pytest.approx(slot.values[key], abs=1e-15)
    assert energy(after) == pytest.approx(before_u, abs=1e-7)
    execute(runtime, "select_vessel", vessel="receiver")
    execute(runtime, "wait", duration_s=10)
    assert (
        runtime._state.state_slots.slots[0].values["sites_B_mol"]
        == slot.values["sites_B_mol"] * 0.5
    )


def test_native_snapshot_control_measurement_and_numerical_failure(monkeypatch):
    import json

    from chemworld.physchem.coupled_system import CoupledSystem
    from chemworld.runtime.session_snapshot import SessionSnapshot

    runtime, _ = native_session()
    execute(runtime, "heat", target_temperature_K=400, duration_s=30)
    execute(runtime, "measure", instrument="gas_composition")
    restored = ChemWorldRuntime.from_snapshot(
        SessionSnapshot.from_dict(json.loads(json.dumps(runtime.snapshot().to_dict())))
    )
    execute(runtime, "wait", duration_s=20)
    execute(restored, "wait", duration_s=20)
    assert runtime._state.to_dict() == restored._state.to_dict()
    assert runtime._state.state_slots != StateSlots()
    before = runtime._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected coupled failure")

    monkeypatch.setattr(CoupledSystem, "advance", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 10})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.ledger.time_s == before.ledger.time_s


def test_independent_process_preserves_interaction_energy_and_pending_measurement(tmp_path):
    import json
    import subprocess
    import sys

    runtime, _ = native_session()
    execute(runtime, "heat", target_temperature_K=400, duration_s=10)
    path = tmp_path / "multisite.json"
    path.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
    code = """import json,sys
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
r=ChemWorldRuntime.from_snapshot(SessionSnapshot.load(sys.argv[1]))
v=r.execute({'operation':'measure','instrument':'gas_composition'})
assert v.info['transaction_status']=='committed',v.info
print(json.dumps(r._state.to_dict()))
"""
    restored = subprocess.run(
        [sys.executable, "-c", code, str(path)], check=True, capture_output=True, text=True
    )
    execute(runtime, "measure", instrument="gas_composition")
    assert json.loads(restored.stdout) == json.loads(json.dumps(runtime._state.to_dict()))
