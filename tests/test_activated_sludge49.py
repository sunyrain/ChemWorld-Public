"""Complete empirical ASM1 original world, resource owners and finite boundaries."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.activated_sludge import (
    advance_activated_sludge,
    flow_profile,
    inventory_quality,
)
from chemworld.physchem.activated_sludge_kinetics import (
    ASM1_COMPONENTS,
    ASM1_REACTIONS,
    ASM1Parameters,
    asm1_quality,
    asm1_rates,
)
from chemworld.physchem.activated_sludge_specs import ActivatedSludgeSpec
from chemworld.physchem.transport import PumpSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.activated_sludge import unpack_activated_sludge
from chemworld.world.activated_sludge_world import activated_sludge_world
from reference.activated_sludge49_fixtures import MIXED, inventory_case, spec_case, transient_cases
from reference.run_activated_sludge49_canary import commit, run


def setup(n=5, shared=True, spec=None):
    spec = spec or spec_case(n)
    rt = ChemWorldRuntime(
        composition=activated_sludge_world(spec, inventory_case(spec), shared_clock=shared)
    )
    rt.reset(seed=49)
    return rt


def original_inventory(rt):
    active = rt._state
    members = {active.vessel_id: active.replace(inactive_vessels={}), **active.inactive_vessels}
    return unpack_activated_sludge(
        members["reactor"].replace(
            inactive_vessels={k: v for k, v in members.items() if k != "reactor"}
        )
    )[2]


@pytest.mark.parametrize("n", [1, 5])
def test_complete_model_codec_and_no_pseudomolecular_fields(n):
    spec = spec_case(n)
    assert ActivatedSludgeSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    assert ASM1Parameters.from_dict(spec.parameters.to_dict()) == spec.parameters
    rt = setup(n)
    members = {rt._state.vessel_id: rt._state, **rt._state.inactive_vessels}
    for state in members.values():
        assert set(state.species_amounts) == {"H2O"}
        for slot in state.state_slots.slots:
            for field in slot.spec.fields:
                assert field.species_id is None
                assert field.conserved_quantity is None
    assert set(original_inventory(rt).components[1]) == set(ASM1_COMPONENTS)
    assert set(original_inventory(rt).extents[0]) == set(ASM1_REACTIONS)


def test_all_eight_source_rates_and_no_invented_nh_limitation():
    p = ASM1Parameters()
    ordinary = asm1_rates(p, MIXED)
    assert all(v > 0 for v in ordinary.values())
    without = asm1_rates(p, {**MIXED, "S_NH": 0.0})
    assert without["R1"] == ordinary["R1"]
    assert without["R2"] == ordinary["R2"]
    assert without["R3"] == 0.0
    zeros = asm1_rates(p, {**MIXED, "X_BH": 0.0, "X_S": 0.0})
    assert zeros["R7"] == zeros["R8"] == 0.0
    before = asm1_quality(p, MIXED)
    after = asm1_quality(p, {**MIXED, "X_ND": 10.0})
    assert before["TSS_kg_m3"] == after["TSS_kg_m3"]
    assert before["COD_kg_O2_m3"] == after["COD_kg_O2_m3"]
    assert after["TKN_kg_N_m3"] > before["TKN_kg_N_m3"]


def test_actual_consumer_complete_json_in_fresh_process(tmp_path):
    assert run(tmp_path / "canary")


@pytest.mark.parametrize("shared", [False, True])
def test_original_clock_energy_cost_and_single_owner(shared):
    rt = setup(shared=shared)
    commit(rt, "set_flow_rate", flow_rate_mL_min=0.1281)
    before = rt._state.ledger.cost
    if shared:
        commit(rt, "select_vessel", vessel="sludge")
        before = rt._state.ledger.cost
    result = commit(rt, "wait" if shared else "run_flow", duration_s=20.0)
    inventory = original_inventory(rt)
    assert inventory.time_s == pytest.approx(20.0)
    assert all(v > 0 for v in inventory.extents[0].values())
    assert inventory.water_mol[0] < inventory_case(spec_case()).water_mol[0]
    states = {rt._state.vessel_id: rt._state, **rt._state.inactive_vessels}
    assert all(s.ledger.time_s == 20.0 for s in states.values())
    anchor = states["reactor"]
    receipt = anchor.equipment.equipment["asm1_treatment"].settings["last_activated_sludge"]
    expected = (receipt["aeration_electricity_J"] + receipt["pump_shaft_work_J"]) / 250000
    assert rt._state.ledger.cost - before == pytest.approx(expected)
    assert result.info["actual_duration_s"] == pytest.approx(20.0)
    assert inventory_quality(spec_case(), inventory)["effluent"]["TSS_kg_m3"] >= 0.0


@pytest.mark.parametrize("index", list(range(11, 22)))
def test_finite_boundaries_preserve_every_accepted_quantity(index):
    name, spec, options, q, duration = transient_cases()[index]
    result = advance_activated_sludge(spec, inventory_case(spec, **options), q, duration)
    assert all(v >= 0 for row in result.final.components for v in row.values())
    assert all(v >= 0 for v in result.final.utilities.values())
    if name == "zero-reactive-pools":
        assert result.stopping_event is None
        assert result.actual_duration_s == duration
    else:
        assert result.stopping_event is not None
        assert result.actual_duration_s < duration


@pytest.mark.parametrize(
    "mutation",
    [
        "quality",
        "owner",
        "phase",
        "model",
        "basis",
        "receipt",
        "unit",
        "H",
        "extent",
        "molecular-alias",
    ],
)
def test_restore_rejects_source_history_and_inventory_tampering(mutation):
    rt = setup(1)
    commit(rt, "wait", duration_s=10.0)
    commit(rt, "select_vessel", vessel="asm1_1")
    payload = json.loads(json.dumps(rt.snapshot().to_dict()))
    local = payload["world_state"]
    anchor = local["inactive_vessels"]["reactor"]
    slot = local["state_slots"]["asm1_1:asm1:quality"]
    if mutation == "quality":
        slot["values"]["X_BH"] += 1e-4
    elif mutation == "owner":
        slot["spec"]["owner_id"] = "foreign"
    elif mutation == "phase":
        local["phases"]["liquid"]["vessel_id"] = "sludge"
    elif mutation == "model":
        anchor["equipment"]["asm1_treatment"]["settings"]["physical_model"]["parameters"]["Y_H"] = (
            0.6
        )
    elif mutation == "basis":
        anchor["equipment"]["asm1_treatment"]["settings"]["empirical_reaction_balance"][
            "reactions"
        ]["R1"]["stoichiometry"]["S_S"] = -1.0
    elif mutation == "receipt":
        slot["initial_values"]["X_BH"] += 1e-4
    elif mutation in ("unit", "molecular-alias"):
        slot["spec"]["fields"][0]["unit" if mutation == "unit" else "species_id"] = (
            "mol" if mutation == "unit" else "H2O"
        )
    elif mutation == "H":
        local["state_slots"]["asm1_1:water:enthalpy"]["values"]["enthalpy_J"] += 1.0
    else:
        local["state_slots"]["asm1_1:asm1:extents"]["values"]["R1"] += 1e-5
    with pytest.raises(ValueError):
        ChemWorldRuntime.from_snapshot(payload)


def test_reverse_aeration_does_not_require_unused_supply_and_zero_power_is_valid():
    spec = spec_case(1, electricity_mode="none", aeration_power_limit_W=0.0)
    mixed = {**MIXED, "X_BH": 0.0, "X_BA": 0.0, "S_O": 0.012}
    initial = inventory_case(spec, mixed=mixed, oxygen_supply=0.0)
    result = advance_activated_sludge(spec, initial, 0.0, 1.0)
    assert result.stopping_event is None
    assert result.final.utilities["oxygen_supply_kg"] == 0.0
    assert result.final.utilities["oxygen_offgas_kg"] > 0.0
    assert result.final.work["aeration_electricity_J"] == 0.0


def test_real_pump_limit_and_bath_stop():
    spec = spec_case(1, pump=PumpSpec(1e-8, 1e-12, 0.7), maximum_bath_cooling_W=1e-30)
    profile = flow_profile(spec, 1e-6)
    assert profile["actual_flow_m3_s"] < 1e-6
    result = advance_activated_sludge(spec, inventory_case(spec), 1e-6, 10.0)
    assert result.stopping_event == "asm1_bath_cooling_limit"
    assert result.actual_duration_s == 0.0


def test_incompatible_reactor_counts_and_incomplete_models_rejected():
    with pytest.raises(ValueError):
        spec_case(reactor_volumes_m3=(1e-5,) * 2, kla_per_hour=(1.0,) * 2)
    payload = spec_case().to_dict()
    del payload["parameters"]["Y_A"]
    with pytest.raises(ValueError):
        ActivatedSludgeSpec.from_dict(payload)
    initial = inventory_case(spec_case())
    with pytest.raises(ValueError):
        activated_sludge_world(spec_case(), replace(initial, time_s=1.0))


@pytest.mark.parametrize("index", [11, 14, 17, 18])
def test_original_shared_world_commits_finite_stop(index):
    _, spec, options, q, duration = transient_cases()[index]
    rt = ChemWorldRuntime(composition=activated_sludge_world(spec, inventory_case(spec, **options)))
    rt.reset(seed=49)
    if q:
        commit(rt, "set_flow_rate", flow_rate_mL_min=q * 60e6)
    commit(rt, "select_vessel", vessel="effluent")
    result = commit(rt, "wait", duration_s=duration)
    assert result.info["operation_outcome"] == "stopped"
    assert result.info["actual_duration_s"] < duration
    assert original_inventory(rt).time_s == pytest.approx(result.info["actual_duration_s"])
    ChemWorldRuntime.from_snapshot(json.loads(json.dumps(rt.snapshot().to_dict())))


def test_electricity_budget_receipt_cannot_be_erased_on_restore():
    rt = setup(1)
    commit(rt, "wait", duration_s=10.0)
    payload = json.loads(json.dumps(rt.snapshot().to_dict()))
    payload["world_state"]["state_slots"]["reactor:asm1:work"]["values"][
        "aeration_electricity_J"
    ] = 0.0
    with pytest.raises(ValueError, match="electricity receipt"):
        ChemWorldRuntime.from_snapshot(payload)
