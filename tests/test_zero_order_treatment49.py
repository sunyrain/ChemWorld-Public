"""Finite empirical splits, original world consumption and separate economics."""

import copy
import json
from dataclasses import replace

import pytest

from chemworld.physchem.water_treatment_economics import (
    empty_treatment_receipt,
    evaluate_water_treatment_economics,
    validate_treatment_receipt,
)
from chemworld.physchem.zero_order_treatment import advance_zero_order_treatment, treatment_profile
from chemworld.physchem.zero_order_treatment_specs import ZeroOrderTreatmentSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.zero_order_treatment import (
    analyze_treatment_world_economics,
    treatment_record,
)
from chemworld.world.zero_order_treatment_world import zero_order_treatment_world_spec
from reference.zero_order_treatment49_cases import (
    analytic,
    economic_policy,
    fixed_cases,
    inventory_case,
    spec_case,
)


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_complete_chain_matches_independent_splits_and_energy(stages):
    spec = spec_case(stages, 318.15)
    initial = inventory_case(spec)
    expected = analytic(spec, initial)
    result, diagnostic = advance_zero_order_treatment(spec, initial, 10 / 6e7, 10)
    for actual, reference in zip(result.final_state.amounts_mol, expected["amounts"], strict=True):
        assert actual == pytest.approx(reference, rel=3e-7, abs=1e-10)
    assert result.final_state.enthalpies_J == pytest.approx(expected["enthalpies"], abs=1e-7)
    receipt = diagnostic["receipt"]
    for key in (
        "feed_volume_m3",
        "product_volume_m3",
        "pump_work_J",
        "additional_electrical_work_J",
        "bath_heat_into_liquid_J",
    ):
        assert receipt[key] == pytest.approx(expected["receipt"][key])
    assert sum(result.component_residuals_mol.values()) == pytest.approx(0, abs=1e-10)
    assert result.enthalpy_residual_J == pytest.approx(0, abs=1e-7)


@pytest.mark.parametrize("index", range(18, 24))
def test_all_fixed_capacity_and_drive_boundaries(index):
    _, spec, composition, flow, duration = fixed_cases()[index]
    initial = inventory_case(spec, composition)
    expected = analytic(spec, initial, flow, duration)
    result, diagnostic = advance_zero_order_treatment(spec, initial, flow, duration)
    assert result.stopping_event == expected["stop"]
    assert result.actual_duration_s == pytest.approx(expected["dt"], abs=1e-6)
    assert diagnostic["receipt"]["feed_volume_m3"] == pytest.approx(
        expected["receipt"]["feed_volume_m3"],
        abs=1e-12,
    )


def test_declared_model_roundtrip_and_external_mutation_isolation():
    spec = spec_case()
    payload = spec.to_dict()
    restored = ZeroOrderTreatmentSpec.from_dict(json.loads(json.dumps(payload)))
    assert restored.to_dict() == payload
    payload["stages"][0]["removal_to_concentrate"]["ethanol"] = 0
    payload["properties"]["packages"][0]["correlations"][0]["coefficients"]["a"] = 999
    assert restored.to_dict() != payload
    assert restored.stages[0].removal_to_concentrate["ethanol"] == 0.6


@pytest.mark.parametrize("mutation", ["recovery", "removal", "dry", "domain", "unknown"])
def test_empirical_template_rejects_undeclared_or_unphysical_splits(mutation):
    spec = spec_case()
    stage = spec.stages[0]
    with pytest.raises(ValueError):
        if mutation == "recovery":
            replace(stage, water_recovery=0)
        elif mutation == "removal":
            replace(stage, removal_to_concentrate={"ethanol": 1.1, "acetone": 0.4})
        elif mutation == "dry":
            replace(stage, water_recovery=1)
        elif mutation == "unknown":
            replace(stage, model="mechanistic_gac")
        else:
            spec = replace(
                spec,
                stages=(
                    replace(
                        stage, maximum_inlet_concentration_mol_m3={"ethanol": 1e-6, "acetone": 1e-6}
                    ),
                ),
            )
            treatment_profile(spec, inventory_case(spec).amounts_mol[0], 10 / 6e7)


@pytest.mark.parametrize("discount", [0, 0.08])
def test_economics_uses_actual_utility_and_zero_product_is_undefined(discount):
    spec = spec_case()
    _, diagnostic = advance_zero_order_treatment(spec, inventory_case(spec), 10 / 6e7, 10)
    receipt = diagnostic["receipt"]
    before = copy.deepcopy(receipt)
    policy = economic_policy(spec, discount)
    report = evaluate_water_treatment_economics(receipt, policy)
    crf = 1 / 20 if discount == 0 else discount / (1 - (1 + discount) ** -20)
    assert report["capital_recovery_factor_per_year"] == pytest.approx(crf)
    assert report["annual_capital_charge"] == pytest.approx(600 * crf)
    assert report["observed_electricity_kWh"] == pytest.approx(
        (receipt["pump_work_J"] + receipt["additional_electrical_work_J"]) / 3.6e6,
    )
    assert receipt == before and report["currency"] == "USD" and report["price_base_year"] == 2026
    assert (
        evaluate_water_treatment_economics(empty_treatment_receipt(spec), policy)[
            "levelized_product_cost_per_m3"
        ]
        is None
    )


@pytest.mark.parametrize("shared", [False, True])
def test_original_world_receipt_and_snapshot_economics_are_read_only(shared):
    spec = spec_case()
    initial = inventory_case(spec)
    composition = zero_order_treatment_world_spec(
        spec,
        initial.amounts_mol[0],
        collectors=initial.amounts_mol[1:],
        shared_clock=shared,
    )
    runtime = ChemWorldRuntime(composition=composition)
    runtime.reset(seed=7)
    assert treatment_record(runtime._state).settings["treatment_totals"]["product_volume_m3"] == 0
    for action in (
        {"operation": "set_flow_rate", "flow_rate_mL_min": 10},
        {"operation": "wait", "duration_s": 10},
    ):
        result = runtime.execute(action)
        assert result.info["transaction_status"] == "committed", result.info
    snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
    before = runtime._state.to_dict()
    left = analyze_treatment_world_economics(runtime._state, economic_policy(spec))
    right = analyze_treatment_world_economics(runtime._state, economic_policy(spec))
    assert left == right and runtime._state.to_dict() == before
    assert left["actual_observed_receipt"]["elapsed_s"] == pytest.approx(10)
    assert left["actual_observed_receipt"]["product_volume_m3"] < (
        runtime._state.inactive_vessels[spec.product_vessel_id].volume_L / 1000
    )
    restored = ChemWorldRuntime.from_snapshot(snapshot)
    action = {"operation": "wait", "duration_s": 5}
    runtime.execute(action)
    restored.execute(action)
    assert runtime._state.to_dict() == restored._state.to_dict()
    assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
        treatment_record(runtime._state).settings["treatment_totals"][
            "additional_electrical_work_J"
        ],
    )
    if shared:
        assert set(runtime._state.vessel_elapsed_s.values()) == {15}


def test_corrupt_receipt_cannot_create_free_product():
    receipt = empty_treatment_receipt(spec_case())
    receipt["elapsed_s"] = 1
    receipt["product_volume_m3"] = 1
    with pytest.raises(ValueError, match="balance"):
        validate_treatment_receipt(receipt)


def test_zero_time_receipt_rejects_fabricated_stage_throughput():
    receipt = empty_treatment_receipt(spec_case())
    receipt["stage_inlet_volumes_m3"]["stage1"] = 1.0
    with pytest.raises(ValueError, match="zero-time"):
        validate_treatment_receipt(receipt)


def test_slot_ownership_rejects_extra_states():
    from chemworld.foundation.state_ownership import StateFieldSpec, StateSlot, StateSlotSpec
    from chemworld.runtime.zero_order_treatment import unpack_treatment

    spec = spec_case()
    initial = inventory_case(spec)
    world = zero_order_treatment_world_spec(
        spec, initial.amounts_mol[0], collectors=initial.amounts_mol[1:]
    )
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset()
    extra = StateSlot(
        StateSlotSpec("other", "equipment", "treatment", "other", (StateFieldSpec("x", "mol"),)),
        "reactor",
        {"x": 1.0},
    )
    state = runtime._state.replace(
        state_slots=replace(
            runtime._state.state_slots, slots=(*runtime._state.state_slots.slots, extra)
        )
    )
    with pytest.raises(ValueError, match="state slots"):
        unpack_treatment(state)


@pytest.mark.parametrize("kind", ["ion", "solid"])
def test_declared_liquid_scope_does_not_invent_ionic_or_sludge_properties(kind):
    spec = spec_case()
    packages = list(spec.properties.packages)
    packages[1] = replace(
        packages[1],
        component=replace(
            packages[1].component,
            **({"charge": 1} if kind == "ion" else {"default_phase": "solid"}),
        ),
    )
    with pytest.raises(ValueError, match=r"ionic|solid/sludge"):
        replace(spec, properties=replace(spec.properties, packages=tuple(packages)))


def test_actual_heat_rejection_and_initial_capacity_stop():
    from chemworld.runtime.zero_order_treatment import ZeroOrderTreatmentAdapter

    spec = spec_case(1)
    initial = inventory_case(spec)
    world = zero_order_treatment_world_spec(
        spec, initial.amounts_mol[0], collectors=initial.amounts_mol[1:]
    )
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset()
    before = runtime._state.to_dict()
    result = ZeroOrderTreatmentAdapter(world).advance(runtime._state, {"duration_s": 1}, heat=True)
    assert result.outcome == "rejected" and result.state is None
    assert runtime._state.to_dict() == before
    spec = replace(spec, product_capacity_L=0.001)
    result, diagnostic = advance_zero_order_treatment(spec, initial, 10 / 6e7, 20)
    assert result.stopping_event == "product_full" and result.actual_duration_s == 0
    assert diagnostic["receipt"]["product_volume_m3"] == 0


@pytest.mark.parametrize("mutation", ["currency", "year", "hours", "capital"])
def test_explicit_engineering_economics_contract(mutation):
    spec = spec_case()
    policy = economic_policy(spec)
    with pytest.raises(ValueError):
        if mutation == "currency":
            replace(policy, currency="currency")
        elif mutation == "year":
            replace(policy, price_base_year=True)
        elif mutation == "hours":
            replace(policy, operating_hours_per_year=9000)
        else:
            policy = replace(policy, installed_capital={"other": 10})
            evaluate_water_treatment_economics(empty_treatment_receipt(spec), policy)
