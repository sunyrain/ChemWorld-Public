"""True latent plateaus, evolving original phase stocks, controls and failures."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from chemworld.data.tdb import TdbDatabase
from chemworld.foundation.state_ledgers import PhaseLedger, VesselLedger
from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.calphad_caloric_equilibrium import CalphadCaloricProblem, solve_calphad_ph
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.calphad_thermal import (
    CalphadThermalRuntime,
    CalphadThermalSpec,
    calphad_inventory_enthalpy_j,
    validate_calphad_thermal_state,
)
from chemworld.world.calphad_thermal_world import build_calphad_thermal_world
from reference.calphad42_thermal_cases import (
    COMMON,
    PURE,
    caloric_spec,
    device,
    thermal_cases,
    thermal_spec,
)
from reference.calphad42_thermal_reference import binary_state, history, pure_state
from reference.run_calphad42_thermal import phase_errors


def world(index=0, spec=None):
    case = thermal_cases()[index]
    spec = spec or thermal_spec(case["kind"], case["method"])
    feed = {"FE": 0.001} if case["kind"] == "pure" else {"FE": 0.0003, "NI": 0.0007}
    composition = build_calphad_thermal_world(
        spec, feed, enthalpy_J=history(case)[0]["H"], device=device(case)
    )
    return spec, composition


def execute(session, operation="wait", **kwargs):
    result = session.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.mark.parametrize("beta", [0.25, 0.75])
def test_pure_plateau_is_determined_by_augmented_h_row(beta):
    ref = pure_state(temperature=600.0, beta=beta)
    result = solve_calphad_ph(caloric_spec("pure"), {"FE": 1.0}, ref["H"])
    assert result.success, result.to_dict()
    assert result.temperature_K == pytest.approx(600, abs=1e-8)
    assert result.phases[1].fraction == pytest.approx(beta, abs=1e-9)
    assert result.diagnostics["attempts"][-1]["rank"] == 4
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("temperature,count", [(600, 2), (800, 1)])
def test_binary_joint_ph_selects_actual_phase_number(temperature, count):
    ref = binary_state(0.3, temperature=temperature)
    result = solve_calphad_ph(caloric_spec(), {"FE": 0.3, "NI": 0.7}, ref["H"])
    assert result.success and len(result.phases) == count, result.to_dict()
    assert result.temperature_K == pytest.approx(temperature, abs=2e-4)
    assert result.entropy_J_K == pytest.approx(ref["S"], abs=2e-4)
    accepted = result.diagnostics["attempts"][-1]
    assert accepted["minimum_local_curvature_RT"] >= -1e-7
    assert accepted["element_balance_error"] <= 1e-10
    if temperature == 600:
        assert any(
            a.get("failure_reason") == "selected_phase_is_tangent_saddle"
            for a in result.diagnostics["attempts"]
        )


def test_latent_plateau_exit_reduces_assemblage_without_truncating_stock():
    problem = CalphadCaloricProblem(caloric_spec("pure"), {"FE": 1.0})
    assert problem.solve(3000.0).success
    endpoint = problem.solve(0.0)
    assert endpoint.success and len(endpoint.phases) == 1, endpoint.to_dict()
    assert endpoint.phases[0].phase_name == "ALPHA"
    assert endpoint.phases[0].amount_mol_atoms == pytest.approx(1.0, abs=1e-12)
    assert endpoint.enthalpy_J == pytest.approx(0.0, abs=1e-7)
    assert any(
        a.get("failure_reason") == "vanishing_phase_requires_reduced_assemblage"
        for a in endpoint.diagnostics["attempts"]
    )


def test_real_latent_heat_original_ledger_and_measurement():
    spec, composition = world()
    assert (
        CalphadThermalSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).to_dict()
        == spec.to_dict()
    )
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=42)
    try:
        before = session._state
        execute(session, duration_s=65.0)
        after = session._state
        assert after.temperature_K == pytest.approx(600.0, abs=1e-7)
        assert after.phases.phases["beta"].species_amounts_mol["FE"] == pytest.approx(
            0.0005, abs=1e-10
        )
        assert after.ledger.time_s == 65
        assert after.ledger.energy_jacket_J == pytest.approx(6.5, abs=1e-7)
        assert after.ledger.boundary_work_J == pytest.approx(
            spec.pressure_Pa * (after.volume_L - before.volume_L) * 0.001
        )
        assert not after.state_slots.slots
        assert not any(
            "enthalpy" in key or "amount" in key
            for p in after.phases.phases.values()
            for key in p.metadata
        )
        execute(session, "measure", instrument="phase_state")
    finally:
        session.close()


def test_phase_disappearance_reappearance_pause_and_exact_snapshot():
    _, composition = world(4)
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=42)
    replay = None
    try:
        execute(session, duration_s=200.0)
        assert len(session._state.phases.phases) == 1
        execute(session, "pause_control")
        t = session._state.temperature_K
        execute(session, duration_s=5.0)
        assert session._state.temperature_K == pytest.approx(t, abs=1e-7)
        execute(session, "resume_control")
        replay = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        for runtime in (session, replay):
            execute(runtime, "heat", target_temperature_K=510.0, duration_s=200.0)
        assert len(session._state.phases.phases) == 2
        assert session._state.to_dict() == replay._state.to_dict()
    finally:
        session.close()
        if replay is not None:
            replay.close()


def test_holder_reset_keeps_sensible_heat_and_original_device():
    spec, composition = world()
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=42)
    try:
        template = session._state
        execute(session, duration_s=65.0)
        record = session._state.equipment.equipment["calphad-heater"]
        expected = calphad_inventory_enthalpy_j(template, spec) + 0.02 * (
            session._state.temperature_K - template.temperature_K
        )
        session.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
        assert calphad_inventory_enthalpy_j(session._state, spec) == pytest.approx(
            expected, abs=1e-7
        )
        assert session._state.equipment.equipment["calphad-heater"] == record
    finally:
        session.close()


@pytest.mark.parametrize("limit", ["capacity", "temperature"])
def test_boundary_returns_actual_stopped_time(limit):
    _, composition = world()
    state = composition.physics.initial_states["reactor"]
    record = state.vessels.vessels["reactor"]
    record = replace(
        record,
        **(
            {"max_volume_L": state.volume_L * 1.05}
            if limit == "capacity"
            else {"max_temperature_K": 595.0}
        ),
    )
    state = state.replace(vessels=VesselLedger({"reactor": record}))
    result = CalphadThermalRuntime(composition).advance(state, {"duration_s": 100.0})
    assert result.success and result.outcome == "stopped", result.diagnostics
    assert 0 < result.actual_duration_s < 100
    assert result.state.ledger.time_s == result.actual_duration_s
    assert result.diagnostics["stopping_event"] == (
        "vessel_capacity" if limit == "capacity" else "temperature_limit"
    )


def test_corrupt_configuration_and_numerical_budget_never_install():
    spec, composition = world(4)
    state = composition.physics.initial_states["reactor"]
    phases = dict(state.phases.phases)
    p = phases["alpha-left"]
    phases[p.phase_id] = replace(
        p, metadata={**p.metadata, "site_fractions": [{"FE": 0.5, "NI": 0.5}]}
    )
    with pytest.raises(ValueError, match="configuration"):
        validate_calphad_thermal_state(state.replace(phases=PhaseLedger(phases)), spec)
    bad = replace(spec.caloric, root_policy=replace(spec.caloric.root_policy, maximum_iterations=1))
    result = solve_calphad_ph(bad, {"FE": 0.3, "NI": 0.7}, binary_state(0.3, temperature=600)["H"])
    assert not result.success and result.diagnostics["attempts"]
    adapter = CalphadThermalRuntime(
        SimpleNamespace(
            physics=SimpleNamespace(models_of=lambda _: {"reactor": replace(spec, caloric=bad)})
        )
    )
    failed = adapter.advance(state, {"duration_s": 40.0})
    assert not failed.success and failed.state is None
    assert failed.diagnostics["failed_caloric_closure"]["diagnostics"]["attempts"]
    bad = replace(
        spec.caloric,
        equilibrium=replace(
            spec.caloric.equilibrium,
            policy=replace(spec.caloric.equilibrium.policy, maximum_iterations=1),
        ),
    )
    result = solve_calphad_ph(bad, {"FE": 0.3, "NI": 0.7}, binary_state(0.3, temperature=600)["H"])
    assert not result.success


def test_snapshot_restore_rejects_tampered_site_projection():
    _, composition = world(4)
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=42)
    try:
        snapshot = json.loads(json.dumps(session.snapshot().to_dict()))
        snapshot["world_state"]["phases"]["alpha-left"]["metadata"]["site_fractions"] = [
            {"FE": 0.5, "NI": 0.5}
        ]
        with pytest.raises(ValueError, match="configuration"):
            ChemWorldRuntime.from_snapshot(snapshot)
    finally:
        session.close()


def test_transfer_and_unsupported_device_leave_material_unchanged():
    _, composition = world()
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=42)
    try:
        before = session._state
        result = session.execute({"operation": "add_component", "component": "FE", "amount": 0.01})
        assert result.info["transaction_status"] != "committed"
        assert session._state.phases == before.phases
        assert session._state.ledger.time_s == 0
        result = CalphadThermalRuntime(composition).advance(
            before, {"duration_s": 1.0}, jacket_program={}
        )
        assert not result.success and result.state is None
    finally:
        session.close()


def test_formula_to_atom_normalization_and_holder_gibbs_identity():
    spec = caloric_spec("pure")
    text = PURE.replace("% 1 1", "% 2 1 1").replace(":FE: !", ":FE:FE: !")
    text = text.replace(",FE;0)", ",FE:FE;0)").replace("400 GCP#;", "400 2*GCP#;")
    text = text.replace("400 GCP#+12000*(1-T/600);", "400 2*GCP#+24000*(1-T/600);")
    db = TdbDatabase.from_string(text, source=spec.equilibrium.models[0].database.source)
    spec = replace(
        spec,
        equilibrium=replace(
            spec.equilibrium, models=tuple(CefPhaseModel(db, name) for name in ("ALPHA", "BETA"))
        ),
    )
    ref = pure_state(temperature=600.0, beta=0.25, total=0.001, support=0.02)
    result = solve_calphad_ph(spec, {"FE": 0.001}, ref["H"], support_heat_capacity_J_K=0.02)
    assert result.success, result.to_dict()
    assert result.phases[1].amount_mol_atoms == pytest.approx(0.00025, abs=1e-11)
    result = solve_calphad_ph(spec, {"FE": 0.001}, 15.0, support_heat_capacity_J_K=0.02)
    assert result.success, result.to_dict()
    assert result.gibbs_J == pytest.approx(
        result.enthalpy_J - result.temperature_K * result.entropy_J_K, abs=1e-10
    )


def test_undeclared_one_sided_tdb_boundary_is_not_silently_differentiated():
    spec = caloric_spec("pure")
    text = PURE.replace(
        "30*(T-600-T*LN(T/600)); 1100 N",
        "30*(T-600-T*LN(T/600)); 600 Y 30*(T-600-T*LN(T/600)); 1100 N",
    )
    db = TdbDatabase.from_string(text, source=spec.equilibrium.models[0].database.source)
    spec = replace(
        spec,
        equilibrium=replace(
            spec.equilibrium, models=tuple(CefPhaseModel(db, name) for name in ("ALPHA", "BETA"))
        ),
    )
    result = solve_calphad_ph(spec, {"FE": 1.0}, 3000.0)
    assert not result.success
    assert any(a.get("branch_diagnostics") for a in result.diagnostics["attempts"])


def test_thermal_reference_compares_element_amounts_with_canonical_phase_matching():
    reference = binary_state(0.3, temperature=600.0, total=0.001)["phases"]
    reordered = [{**p, "x": dict(reversed(list(p["x"].items())))} for p in reversed(reference)]
    assert phase_errors(reordered, reference) == (0.0, 0.0, 0.0, 0.0)
    perturbed = [
        {**p, "x": {"FE": p["x"]["FE"] + 0.01, "NI": p["x"]["NI"] - 0.01}} for p in reference
    ]
    error = phase_errors(perturbed, reference)
    assert error[1] == 0.0  # Same total phase amounts conceal changed elemental stocks.
    assert error[3] == pytest.approx(0.01 * max(p["n"] for p in reference))
    assert phase_errors(reference[:1], reference)[0] == 1.0


def two_sublattice_world(penalty):
    spec = thermal_spec()
    text = COMMON + "\nPHASE ALPHA % 2 1 1 !\nCONSTITUENT ALPHA :FE,NI:FE,NI: !\n"
    text += "\n".join(
        f"PARAMETER G(ALPHA,{a}:{b};0) 400 2*GCP#+{penalty if a == b else 0}; 1100 N !"
        for a in ("FE", "NI")
        for b in ("FE", "NI")
    )
    db = TdbDatabase.from_string(text, source=spec.caloric.equilibrium.models[0].database.source)
    spec = replace(
        spec,
        caloric=replace(
            spec.caloric,
            equilibrium=replace(spec.caloric.equilibrium, models=(CefPhaseModel(db, "ALPHA"),)),
        ),
    )
    h = binary_state(0.5, temperature=600.0, total=0.001)["H"] if penalty else 0.0
    return spec, build_calphad_thermal_world(
        spec, {"FE": 0.0005, "NI": 0.0005}, enthalpy_J=h, device=device(thermal_cases()[4])
    )


def test_snapshot_rejects_same_h_same_composition_internal_nonequilibrium():
    spec, composition = two_sublattice_world(0)
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=42)
    try:
        before = session._state
        phases = dict(before.phases.phases)
        key = next(iter(phases))
        changed_sites = [{"FE": 0.2, "NI": 0.8}, {"FE": 0.8, "NI": 0.2}]
        phases[key] = replace(
            phases[key], metadata={**phases[key].metadata, "site_fractions": changed_sites}
        )
        changed = before.replace(phases=PhaseLedger(phases))
        assert changed.species_amounts == before.species_amounts
        assert calphad_inventory_enthalpy_j(changed, spec) == pytest.approx(
            calphad_inventory_enthalpy_j(before, spec), abs=1e-12
        )
        snapshot = json.loads(json.dumps(session.snapshot().to_dict()))
        snapshot["world_state"]["phases"][key]["metadata"]["site_fractions"] = changed_sites
        with pytest.raises(ValueError, match="configuration"):
            ChemWorldRuntime.from_snapshot(snapshot)
    finally:
        session.close()


def test_snapshot_accepts_equivalent_sublattice_exchange():
    spec, composition = two_sublattice_world(12000)
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=42)
    replay = None
    try:
        snapshot = json.loads(json.dumps(session.snapshot().to_dict()))
        phases = snapshot["world_state"]["phases"]
        assert len(phases) == 1
        metadata = next(iter(phases.values()))["metadata"]
        original = metadata["site_fractions"]
        assert abs(original[0]["FE"] - original[1]["FE"]) > 0.1
        metadata["site_fractions"] = list(reversed(original))
        replay = ChemWorldRuntime.from_snapshot(snapshot)
        assert (
            replay._state.phases.phases[next(iter(phases))].metadata["site_fractions"] != original
        )
        assert calphad_inventory_enthalpy_j(replay._state, spec) == pytest.approx(
            calphad_inventory_enthalpy_j(session._state, spec), abs=1e-12
        )
    finally:
        session.close()
        if replay is not None:
            replay.close()
