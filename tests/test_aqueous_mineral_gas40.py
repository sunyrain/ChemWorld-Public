"""Joint gas/solid geometry, thermodynamic coupling and original transaction regressions."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.aqueous_headspace import example_inputs as gas_inputs
from chemworld.examples.aqueous_headspace_minerals import example_inputs
from chemworld.physchem.aqueous_headspace_minerals import (
    AqueousHeadspaceMineralSpec,
    advance_aqueous_headspace_minerals,
    headspace_mineral_pressure,
    headspace_mineral_properties,
    headspace_mineral_volumes,
)
from chemworld.physchem.mineral_kinetics import MineralGeometry
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_gas_transfer import AqueousGasTransferRuntime
from chemworld.world.aqueous_gas_transfer_world import build_aqueous_headspace_world


@pytest.mark.parametrize("condition", [0, 1])
def test_solids_displace_actual_headspace_and_use_same_reference_pressure_energy(condition):
    spec, n, _ = example_inputs("double", condition)
    props = headspace_mineral_properties(spec, n)
    volumes = headspace_mineral_volumes(spec, n)
    assert volumes["headspace"] == pytest.approx(9.4e-5)
    assert sum(volumes.values()) == pytest.approx(spec.container_volume_m3)
    assert props["enthalpy_J"] - props["internal_energy_J"] == pytest.approx(
        props["pressure_Pa"] * spec.container_volume_m3
    )
    changed = headspace_mineral_properties(spec, {**n, "I(g)": n["I(g)"] * 2})
    dp = changed["pressure_Pa"] - props["pressure_Pa"]
    for mineral in spec.mineral_ids:
        before, after = props["minerals"][mineral], changed["minerals"][mineral]
        for key in ("enthalpy_J", "gibbs_energy_J"):
            assert after[key] - before[key] == pytest.approx(dp * before["volume_m3"])
        assert after["internal_energy_J"] == pytest.approx(before["internal_energy_J"])


def test_gas_uptake_drives_common_ion_precipitation_beyond_no_transfer_control():
    spec, n, g = example_inputs()
    control = replace(
        spec, transfers=tuple(replace(t, coefficient_mol_s=0.0) for t in spec.transfers)
    )
    uptake = advance_aqueous_headspace_minerals(spec, n, g, 600.0)
    fixed = advance_aqueous_headspace_minerals(control, n, g, 600.0)
    assert uptake.final["amounts_mol"]["MA(s)"] > n["MA(s)"]
    assert fixed.final["amounts_mol"]["MA(s)"] < n["MA(s)"]
    assert (
        uptake.final["properties"]["gas"]["volume_m3"]
        < fixed.final["properties"]["gas"]["volume_m3"]
    )


def test_exhaustion_continues_gas_in_one_joint_original_solver(monkeypatch):
    from chemworld.physchem import solver_backend

    spec, n, g = example_inputs("exhaustion")
    original = solver_backend.solve_ode
    dimensions = []

    def track(rhs, y0, **kwargs):
        dimensions.append(len(y0))
        return original(rhs, y0, **kwargs)

    monkeypatch.setattr(solver_backend, "solve_ode", track)
    report = advance_aqueous_headspace_minerals(spec, n, g, 30.0)
    assert dimensions == [2, 1]
    assert report.diagnostics["events"][0]["event"] == "mineral_exhausted"
    assert 0 < report.diagnostics["events"][0]["time_s"] < 30.0
    assert report.final["amounts_mol"]["MA(s)"] == 0.0
    assert report.final["amounts_mol"]["A(g)"] < n["A(g)"]


@pytest.mark.parametrize("kind", ["unseeded", "zero_mineral"])
def test_zero_mineral_stock_or_rate_cannot_create_new_seed(kind):
    spec, n, g = example_inputs(kind)
    report = advance_aqueous_headspace_minerals(spec, n, g, 30.0)
    assert report.final["amounts_mol"]["MA(s)"] == n["MA(s)"]
    assert report.final["amounts_mol"]["A(g)"] != n["A(g)"]
    assert not report.diagnostics["events"]


def test_real_vessel_pressure_limit_rejects_all_physical_candidate_updates():
    spec, n, g = example_inputs("degassing")
    runtime = ChemWorldRuntime(
        composition=build_aqueous_headspace_world(
            spec,
            n,
            mineral_geometries=g,
            max_pressure_Pa=headspace_mineral_pressure(spec, n) + 10.0,
        )
    )
    try:
        runtime.reset(seed=40)
        before = runtime._state
        reply = runtime.execute({"operation": "wait", "duration_s": 5.0})
        assert reply.info["transaction_status"] != "committed"
        assert "vessel operating limit" in json.dumps(reply.info)
        assert runtime._state.species_amounts == before.species_amounts
        assert runtime._state.state_slots == before.state_slots
        assert runtime._state.phases == before.phases
        assert runtime._state.pressure_Pa == before.pressure_Pa
        assert runtime._state.ledger.time_s == runtime._state.ledger.energy_jacket_J == 0.0
        assert runtime._state.ledger.risk > before.ledger.risk
    finally:
        runtime.close()


def test_exhausted_zero_phase_and_original_geometry_survive_json_continuation():
    spec, n, g = example_inputs("exhaustion")
    runtime = ChemWorldRuntime(
        composition=build_aqueous_headspace_world(spec, n, mineral_geometries=g)
    )
    restored = None
    try:
        runtime.reset(seed=40)
        slots = runtime._state.state_slots
        assert (
            runtime.execute({"operation": "wait", "duration_s": 5.0}).info["transaction_status"]
            == "committed"
        )
        phase = runtime._state.phases.phases["mineral:MA(s)"]
        assert phase.volume_L == phase.species_amounts_mol["MA(s)"] == 0.0
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for world in (runtime, restored):
            assert (
                world.execute({"operation": "wait", "duration_s": 5.0}).info["transaction_status"]
                == "committed"
            )
            assert world._state.state_slots == slots
        assert runtime.snapshot().to_dict() == restored.snapshot().to_dict()
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


@pytest.mark.parametrize("bad", ["owner", "phase", "value", "positive_stock_without_reference"])
def test_original_material_geometry_validation_rejects_foreign_or_reset_reference(bad):
    spec, n, g = example_inputs()
    if bad == "positive_stock_without_reference":
        g = {"MA(s)": MineralGeometry(0.0, 0.0)}
    composition = build_aqueous_headspace_world(spec, n, mineral_geometries=g)
    state = composition.physics.initial_states["reactor"]
    slot = state.state_slots.slots[0]
    if bad == "owner":
        slot = replace(slot, spec=replace(slot.spec, owner_id="foreign-lot"))
    elif bad == "phase":
        slot = replace(slot, spec=replace(slot.spec, phase_id="headspace"))
    elif bad == "value":
        slot = replace(slot, values={**slot.values, "reference_area_m2": 0.02})
    state = state.replace(state_slots=replace(state.state_slots, slots=(slot,)))
    with pytest.raises(ValueError):
        AqueousGasTransferRuntime(composition).validate_state(state)


@pytest.mark.parametrize("bad", ["container", "rates", "reaction", "water_reaction", "schema"])
def test_invalid_v2_declaration_fails_without_repair(bad):
    spec, _, _ = example_inputs()
    with pytest.raises(ValueError, match="nonvolatile water" if bad == "water_reaction" else None):
        if bad == "container":
            replace(spec, container_volume_m3=spec.liquid_volume_m3)
        elif bad == "rates":
            replace(spec, mineral_rates=())
        elif bad == "reaction":
            replace(spec, mineral_reaction_ids=("acid",))
        elif bad == "water_reaction":
            from chemworld.physchem.aqueous_database import AqueousReaction

            db = spec.equilibrium.database
            hydrogen = next(s for s in db.species if s.species_id == "H+")
            hydroxide = replace(
                hydrogen,
                species_id="OH-",
                composition={"O": 1, "H": 1},
                charge=-1,
                standard_state=replace(
                    hydrogen.standard_state,
                    thermo=replace(
                        hydrogen.standard_state.thermo,
                        species_id="OH-",
                        composition={"O": 1, "H": 1},
                    ),
                ),
            )
            reaction = AqueousReaction(
                "water-dissociation", {"H2O": -1, "H+": 1, "OH-": 1}, db.sources[0].source_id
            )
            db = replace(
                db,
                species=(*db.species, hydroxide),
                reactions=(*db.reactions, reaction),
                activity_model=replace(
                    db.activity_model, charges=(*db.activity_model.charges, ("OH-", -1))
                ),
            )
            replace(
                spec,
                equilibrium=replace(
                    spec.equilibrium,
                    database=db,
                    reaction_ids=("water-dissociation",),
                ),
            )
        else:
            AqueousHeadspaceMineralSpec.from_dict({**spec.to_dict(), "model": "unknown"})


def test_original_v1_builder_cannot_gain_solid_state_by_geometry_argument():
    spec, n = gas_inputs()
    with pytest.raises(ValueError, match="v1"):
        build_aqueous_headspace_world(spec, n, mineral_geometries={"MA(s)": MineralGeometry()})
