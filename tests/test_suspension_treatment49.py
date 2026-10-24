"""Original multiphase ownership, finite events and rejection regressions."""

import json
from dataclasses import replace

import pytest
from examples.suspension_treatment49 import example_feed, example_spec

from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.physchem.suspension_treatment import (
    advance_suspension_treatment,
    suspension_treatment_inventory,
)
from chemworld.physchem.suspension_treatment_specs import (
    ClarifierSplitSpec,
    JarTestEndpointSpec,
    SuspensionTankInventory,
    SuspensionTreatmentSpec,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.suspension_treatment import (
    SuspensionTreatmentAdapter,
    unpack_suspension_treatment,
)
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.suspension_treatment_world import suspension_treatment_world_spec
from reference.suspension_treatment49_cases import fixed_cases, independent_flows


@pytest.mark.parametrize("case", fixed_cases()[:18], ids=lambda c: c[0])
def test_fixed_finite_material_and_event_cases(case):
    _, spec, feed, flow, duration, _, _, event = case
    initial = suspension_treatment_inventory(spec, feed)
    result, diagnostic = advance_suspension_treatment(spec, initial, flow, duration)
    assert result.stopping_event == event
    assert abs(result.energy_balance_residual_J) < 3e-6
    assert abs(diagnostic["enthalpy_residual_J"]) < 3e-6
    assert all(abs(v) < 3e-9 for v in result.conservation_residuals.values())
    _, _, inlet, product, sludge = independent_flows(spec, feed, flow)
    for i, rate in enumerate(({k: -v for k, v in inlet.items()}, product, sludge)):
        for k, value in rate.items():
            assert result.final_state.amounts_mol[i][k] == pytest.approx(
                initial.amounts_mol[i][k] + value * result.actual_duration_s, abs=3e-10
            )


def make_runtime(shared=False):
    spec = example_spec()
    payload = suspension_treatment_world_spec(
        spec, example_feed(spec), shared_clock=shared
    ).to_dict()
    payload["task"]["operations"].append("select_vessel")
    runtime = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))
    runtime.reset()
    return runtime


def act(runtime, operation, **values):
    result = runtime.execute({"operation": operation, **values})
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.mark.parametrize("shared", [False, True])
def test_original_multiphase_world_and_exact_full_json_continuation(shared):
    runtime = make_runtime(shared)
    act(runtime, "set_flow_rate", flow_rate_mL_min=10)
    if shared:
        act(runtime, "select_vessel", vessel="sludge")
    op = "wait" if shared else "run_flow"
    act(runtime, op, duration_s=5)
    clone = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    act(runtime, op, duration_s=15)
    act(clone, op, duration_s=15)
    assert json.loads(json.dumps(clone.snapshot().to_dict())) == json.loads(
        json.dumps(runtime.snapshot().to_dict())
    )
    current = runtime._state
    states = {current.vessel_id: current, **current.inactive_vessels}
    assert current.ledger.flow_pump_work_J > 0
    assert current.ledger.flow_shaft_work_J > 0
    assert current.ledger.heat_loss_J > current.ledger.flow_shaft_work_J
    for tank in states.values():
        assert set(tank.phases.phases) == {"liquid", "dispersed", "floc"}
        assert len(tank.state_slots.slots) == 1
        assert tank.state_slots.slots[0].spec.phase_id is None
        assert tank.state_slots.slots[0].spec.owner_kind == "material"
        assert tank.species_amounts["silica"] == pytest.approx(
            sum(
                tank.phases.phases[name].species_amounts_mol["silica"]
                for name in ("dispersed", "floc")
            )
        )
    assert states["clarified"].phases.phases["floc"].species_amounts_mol["silica"] > 0
    assert states["sludge"].phases.phases["liquid"].species_amounts_mol["water"] > 0
    assert states["sludge"].phases.phases["liquid"].species_amounts_mol["glucose"] > 0


@pytest.mark.parametrize(
    "mutation", ["enthalpy", "phase_owner", "foreign_solid", "phase_volume", "bulk", "source"]
)
def test_state_disagreement_rejected_atomically(mutation):
    runtime = make_runtime()
    original = runtime._state
    adapter = SuspensionTreatmentAdapter(runtime.compiled_composition.spec)
    if mutation == "enthalpy":
        slots = original.state_slots.slots
        bad = original.replace(
            state_slots=replace(
                original.state_slots,
                slots=(
                    replace(slots[0], values={"enthalpy_J": slots[0].values["enthalpy_J"] + 1}),
                ),
            )
        )
    elif mutation == "phase_owner":
        slots = original.state_slots.slots
        bad = original.replace(
            state_slots=replace(
                original.state_slots,
                slots=(replace(slots[0], spec=replace(slots[0].spec, phase_id="liquid")),),
            )
        )
    elif mutation in {"foreign_solid", "phase_volume"}:
        phases = dict(original.phases.phases)
        phase = phases["floc"]
        phases["floc"] = (
            replace(phase, species_amounts_mol={"silica": 0, "rogue": 0})
            if mutation == "foreign_solid"
            else replace(phase, volume_L=0.001)
        )
        bad = original.replace(phases=PhaseLedger(phases))
    elif mutation == "bulk":
        bad = original.replace(species_amounts={**original.species_amounts, "silica": 0})
    else:
        from chemworld.foundation import upsert_equipment_record

        record, spec, _ = unpack_suspension_treatment(original)
        changed = replace(spec, clarifier=replace(spec.clarifier, floc_capture=0.5))
        bad = original.replace(
            equipment=upsert_equipment_record(
                original.equipment,
                equipment_id=record.equipment_id,
                equipment_type=record.equipment_type,
                attached_vessel_id=original.vessel_id,
                batch_reset="preserve",
                settings={"physical_model": changed.to_dict()},
            )
        )
    before = bad.to_dict()
    result = adapter.advance(bad, {"operation": "run_flow", "duration_s": 10})
    assert not result.success
    assert result.state is None
    assert bad.to_dict() == before
    assert runtime._state == original


def test_declarations_roundtrip_and_reject_unsupported_domains():
    spec = example_spec()
    assert (
        SuspensionTreatmentSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).to_dict()
        == spec.to_dict()
    )
    with pytest.raises(ValueError, match="non-increasing"):
        JarTestEndpointSpec(1, 2, 1, 0, "measured")
    with pytest.raises(ValueError, match="actual liquid"):
        ClarifierSplitSpec(0.1, 0.9, 0, "measured")
    with pytest.raises(ValueError, match="constant-volume"):
        package = spec.material.liquid.packages[0]
        correlations = tuple(
            replace(c, coefficients={**c.coefficients, "alpha": 1e-4})
            if c.property_id == "liquid_density"
            else c
            for c in package.correlations
        )
        replace(
            spec.material,
            liquid=replace(
                spec.material.liquid,
                packages=(
                    replace(package, correlations=correlations),
                    *spec.material.liquid.packages[1:],
                ),
            ),
        )


def test_actual_feed_must_match_measured_jar_domain():
    spec = example_spec()
    feed = example_feed(spec)
    with pytest.raises(ValueError, match="measured TSS"):
        suspension_treatment_inventory(spec, replace(feed, dispersed_mol=feed.dispersed_mol * 0.5))
    with pytest.raises(ValueError, match="viscosity domain"):
        too_small = replace(
            spec, material=replace(spec.material, maximum_solid_volume_fraction=1e-8)
        )
        suspension_treatment_inventory(too_small, feed)
    with pytest.raises(ValueError, match="identities"):
        suspension_treatment_inventory(spec, SuspensionTankInventory({"water": 1}, 0))


def test_no_conversion_never_fabricates_floc_or_heat():
    spec = example_spec(final_ntu=100)
    result, diagnostic = advance_suspension_treatment(
        spec, suspension_treatment_inventory(spec, example_feed(spec)), 10 / 6e7, 1000
    )
    assert result.stopping_event == "feed_pickup_exhausted"
    assert all(row["floc"] == 0 for row in result.final_state.amounts_mol)
    assert diagnostic["enthalpy_change_J"] == pytest.approx(0, abs=1e-10)
    assert diagnostic["bath_cooling_J"] == pytest.approx(
        diagnostic["hydraulic_work_J"] + diagnostic["mixing_work_J"], abs=1e-10
    )


def test_original_network_phase_identity_is_bound():
    from chemworld.physchem.reaction_network import SpeciesSpec

    spec = example_spec()
    composition = suspension_treatment_world_spec(spec, example_feed(spec))
    physics = composition.physics
    wrong_network = replace(
        physics.network,
        species=tuple(
            SpeciesSpec(s.species_id, s.formula, phase="liquid") for s in physics.network.species
        ),
    )
    with pytest.raises(ValueError, match="phase identities"):
        ChemWorldRuntime(
            composition=replace(composition, physics=replace(physics, network=wrong_network))
        )


@pytest.mark.parametrize("operation", ["heat", "decant", "new_experiment"])
def test_group_cannot_be_modified_by_undeclared_operation(operation):
    runtime = make_runtime()
    before = runtime._state.to_dict()
    result = runtime.execute({"operation": operation, "duration_s": 5, "target_temperature_K": 320})
    assert result.info["transaction_status"] != "committed"
    current = runtime._state.to_dict()
    # Original runtime records the cost/risk of an invalid request; material,
    # energy and physical time must remain unchanged.
    assert {k: v for k, v in current.items() if k not in {"ledger", "process"}} == {
        k: v for k, v in before.items() if k not in {"ledger", "process"}
    }
    assert current["ledger"]["time_s"] == before["ledger"]["time_s"]
    assert current["ledger"]["flow_pump_work_J"] == before["ledger"]["flow_pump_work_J"]
    assert current["ledger"]["flow_shaft_work_J"] == before["ledger"]["flow_shaft_work_J"]
