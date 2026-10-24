"""Imported sources reach original finite material, thermal and snapshot consumers."""

import json

import pytest

from chemworld.examples.native_imported_surface import SOURCE, authored_declaration, example_world
from chemworld.physchem.catalysis_network import (
    CatalyticMechanismSpec,
    evaluate_catalytic_rates,
    make_catalytic_slot,
    pack_catalytic_state,
)
from chemworld.physchem.surface_mechanism_import import (
    ImportedSurfaceMechanism,
    convert_surface_mechanism,
    import_surface_mechanism,
)
from chemworld.runtime import ChemWorldRuntime


def imported(payload=None):
    return convert_surface_mechanism(
        authored_declaration() if payload is None else payload,
        surface_phase="surface",
        source=SOURCE,
        minimum_temperature_K=300,
        maximum_temperature_K=1900,
    )


def state(spec):
    slot = make_catalytic_slot(
        "test",
        "reactor",
        spec,
        capacities_mol={"surface": 1e-6},
        initial_amounts_mol={"Pt": 8e-7, "HA": 1e-7, "HB": 1e-7},
    )
    return slot, pack_catalytic_state(
        spec, (slot,), {"A": 0.02, "B": 0.01}, temperature_K=700, volume_L=1
    )


def test_import_json_retains_source_rates_and_original_complete_material():
    bundle = imported()
    restored = ImportedSurfaceMechanism.from_dict(json.loads(json.dumps(bundle.to_dict())))
    assert restored.to_dict() == bundle.to_dict()
    slot, physical = state(bundle.surface)
    rates = evaluate_catalytic_rates(physical, slot, bundle.surface)
    assert all(v is None for v in rates.barriers_J_mol.values())
    assert rates.reverse_rates_mol_s["surface-3"] == 0
    assert sum(
        v for k, v in rates.amount_rates_mol_s.items() if k.startswith("catalyst:")
    ) == pytest.approx(0, abs=1e-18)
    assert bundle.surface.species_by_id["Pt"].species.composition == {"Pt": 1}


def test_imported_surface_uses_enthalpy_without_gas_pv_subtraction():
    bundle = imported()
    for species in bundle.surface.species:
        value = species.evaluate(700)
        offset = value.enthalpy_J_mol - species.internal_energy(700)
        assert offset == pytest.approx(
            0 if species.species.phase == "surface" else 8.31446261815324 * 700
        )


def test_original_source_mutation_cannot_change_cached_surface_thermo():
    raw = authored_declaration()
    bundle = imported(raw)
    before = bundle.surface.to_dict()
    raw["species"][3]["thermo"]["h0"] = -9e10
    assert bundle.surface.to_dict() == before


@pytest.mark.parametrize(
    "kind",
    [
        "extra",
        "unbalanced_sites",
        "transport",
        "phase",
        "order",
        "nonreactant",
        "thermo",
        "include",
        "rate",
    ],
)
def test_unconsumed_physics_is_rejected(kind):
    raw = authored_declaration()
    if kind == "extra":
        raw["reactions"][0]["electrochemistry"] = True
    if kind == "unbalanced_sites":
        # Multidentate declarations are supported, but changing only one size
        # without balanced site stoichiometry must still be rejected.
        raw["species"][3]["sites"] = 2
    if kind == "transport":
        raw["phases"][0]["transport"] = "mixture-averaged"
    if kind == "phase":
        raw["phases"][1]["thermo"] = "coverage-dependent-surface"
    if kind == "order":
        raw["reactions"][2]["orders"] = {"HB": -1}
    if kind == "nonreactant":
        raw["reactions"][2]["orders"] = {"Pt": 1}
    if kind == "thermo":
        raw["species"][3]["thermo"]["excess-energy"] = 3
    if kind == "include":
        raw["phases"][0]["species"] = [{"other.yaml/species": "all"}]
    if kind == "rate":
        raw["reactions"][0]["type"] = "electrochemical"
    with pytest.raises(ValueError):
        imported(raw)


def test_explicit_surface_inventory_cannot_leave_empty_undeclared_site_stock():
    spec = imported().surface
    with pytest.raises(ValueError, match="fill"):
        make_catalytic_slot(
            "test",
            "reactor",
            spec,
            capacities_mol={"surface": 1e-6},
            initial_amounts_mol={"HA": 1e-7},
        )


def test_positive_orders_have_correct_length_and_amount_dimensions():
    raw = authored_declaration()
    raw["reactions"][0] = {
        "equation": "A + Pt => HA",
        "rate-constant": {"A": "2 m^3/mol/s", "b": 0, "Ea": 0},
    }
    value = imported(raw).surface.reactions[0].coefficient
    raw["reactions"][0]["rate-constant"]["A"] = "2000 m^3/kmol/s"
    assert imported(raw).surface.reactions[0].coefficient == value
    raw["reactions"][0]["rate-constant"]["A"] = "2 m^2/mol/s"
    with pytest.raises(ValueError, match="units"):
        imported(raw)


def test_yaml_file_and_source_initial_state_are_explicit(tmp_path):
    import yaml

    raw = authored_declaration()
    raw["phases"][0]["state"] = {"T": 999, "P": 200000, "X": {"A": 1}}
    path = tmp_path / "source.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    bundle = import_surface_mechanism(
        path,
        surface_phase="surface",
        source=SOURCE,
        minimum_temperature_K=300,
        maximum_temperature_K=1900,
    )
    assert bundle.source_initial_states["gas"]["T"] == 999
    assert len(bundle.surface.reactions) == 3


@pytest.mark.parametrize("isothermal", [True, False])
def test_actual_world_consumes_finite_gas_and_restores_exactly(isothermal):
    runtime = ChemWorldRuntime(composition=example_world(isothermal=isothermal))
    restored = None
    try:
        runtime.reset(seed=34)
        before = runtime._state
        result = runtime.execute({"operation": "wait", "duration_s": 1})
        assert result.info["transaction_status"] == "committed", result.info
        assert runtime._state.species_amounts != before.species_amounts
        slot = runtime._state.state_slots.slots[0]
        assert sum(slot.values[k] for k in ("Pt", "HA", "HB")) == pytest.approx(1e-6, abs=1e-16)
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for world in (runtime, restored):
            reply = world.execute({"operation": "wait", "duration_s": 1})
            assert reply.info["transaction_status"] == "committed", reply.info
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        runtime.close()
        if restored:
            restored.close()


def test_old_codec_remains_valid_without_new_import_fields():
    from reference.catalysis53_fixtures import mechanism

    spec = mechanism()
    raw = spec.to_dict()
    assert "site_species_complete" not in raw
    assert CatalyticMechanismSpec.from_dict(raw).to_dict() == raw


def test_surface_analysis_uses_complete_site_chart_and_retains_imported_control_boundary():
    from chemworld.physchem.catalysis_analysis import (
        CatalyticAnalysisConditions,
        catalytic_rate_control,
        solve_catalytic_steady_state,
    )

    spec = imported().surface
    conditions = CatalyticAnalysisConditions(700, {"A": 7e4, "B": 3e4}, {"surface": 1e-6})
    seed = {"Pt": 0.8, "HA": 0.1, "HB": 0.1}
    result = solve_catalytic_steady_state(spec, conditions, seed, target_species="B")
    assert result["status"] == "converged", result
    assert sum(result["coverages"].values()) == pytest.approx(1.0, abs=1e-12)
    assert result["steady_chart"]["reachable_dimension"] == 2
    assert len(result["eigenvalues_s_inv"]) == 2
    with pytest.raises(ValueError, match="BEP"):
        catalytic_rate_control(spec, conditions, seed, target_species="B")


def test_local_gas_reaction_section_reaches_the_original_world_process():
    from chemworld.world.descriptor_catalysis import descriptor_catalysis_world

    raw = authored_declaration()
    raw["surface-reactions"] = raw.pop("reactions")
    raw["phases"][1]["reactions"] = ["surface-reactions"]
    raw["phases"][0]["reactions"] = ["gas-reactions"]
    raw["gas-reactions"] = [{"equation": "A <=> B", "rate-constant": {"A": 0.2, "b": 0, "Ea": 0}}]
    bundle = imported(raw)
    composition = descriptor_catalysis_world(
        bundle.surface,
        {"A": 0.02, "B": 0.01},
        capacities_mol={"surface": 1e-6},
        initial_adsorbates_mol={"Pt": 8e-7, "HA": 1e-7, "HB": 1e-7},
        volume_L=1,
        temperature_K=700,
        support_heat_capacity_J_K=1,
        stop_temperature_K=1800,
        maximum_pressure_Pa=1e6,
        gas_network=bundle.gas_network,
    )
    assert any(c.kind == "reaction" for c in composition.components)
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=34)
        result = runtime.execute({"operation": "wait", "duration_s": 1})
        assert result.info["transaction_status"] == "committed", result.info
        assert runtime._state.species_amounts["A"] < 0.019
    finally:
        runtime.close()


@pytest.mark.parametrize("model", ["NASA7", "NASA9", "Shomate", "constant-cp"])
def test_surface_source_thermo_families_share_same_caloric_contract(model):
    from reference.mechanism34_cases import THERMOS

    raw = authored_declaration()
    raw["species"][3]["thermo"] = THERMOS[model]
    bundle = imported(raw)
    species = bundle.surface.species_by_id["HA"]
    value = species.evaluate(700)
    assert species.internal_energy(700) == value.enthalpy_J_mol
    assert species.cv(700) == value.cp_J_mol_K


def test_motz_wise_is_a_physical_sticking_probability_correction():
    from dataclasses import replace

    from chemworld.physchem.surface_import_rates import imported_surface_rates

    reaction = replace(
        imported().surface.reactions[0], coefficient=0.4, coverage_dependencies={}, reversible=False
    )
    arguments = {
        "temperature_K": 700,
        "activities": {"A": 0.7, "B": 0.3, "Pt": 0.8, "HA": 0.1, "HB": 0.1},
        "gas_species": ("A", "B"),
        "reference_pressure_Pa": 101325,
        "site_density_mol_m2": 1e-5,
        "capacity_mol": 1e-6,
        "delta_g_J_mol": 0,
    }
    corrected, _ = imported_surface_rates(reaction, **arguments)
    plain, _ = imported_surface_rates(replace(reaction, motz_wise=False), **arguments)
    assert corrected / plain == pytest.approx(1 / (1 - 0.4 / 2))
    with pytest.raises(ValueError, match="probability"):
        imported_surface_rates(replace(reaction, coefficient=1.1), **arguments)


def test_snapshot_rejects_missing_explicit_surface_configurations():
    runtime = ChemWorldRuntime(composition=example_world())
    try:
        runtime.reset(seed=34)
        runtime.execute({"operation": "wait", "duration_s": 1})
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        slot = next(iter(snapshot["world_state"]["state_slots"].values()))
        slot["values"]["Pt"] *= 0.9
        with pytest.raises(ValueError, match="fill"):
            ChemWorldRuntime.from_snapshot(snapshot)
    finally:
        runtime.close()


@pytest.mark.parametrize("index,kind", [(0, "interface-Arrhenius"), (1, "sticking-Arrhenius")])
def test_conflicting_rate_type_is_not_ignored(index, kind):
    raw = authored_declaration()
    raw["reactions"][index]["type"] = kind
    with pytest.raises(ValueError, match="contradicts"):
        imported(raw)
