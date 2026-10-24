"""Scientific identities, declaration contracts and actual read-only world use."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.gac_design import gac_design_example
from chemworld.physchem.gac_design import GACDesignSpec, evaluate_gac_design
from chemworld.world.gac_design import analyze_gac_design_world
from reference.gac_design49_fixtures import commit, numerical_cases, world_case


@pytest.mark.parametrize("case", numerical_cases(), ids=lambda c: c[0])
def test_design_balances_geometry_and_fresh_breakthrough(case):
    _, spec, feed = case
    result = evaluate_gac_design(spec, feed)
    m = result["metrics"]
    assert m["bed_volume_m3"] == pytest.approx(
        feed.volumetric_flow_m3_s * spec.bed.empty_bed_contact_time_s
    )
    assert m["carbon_mass_kg"] == pytest.approx(m["bed_volume_m3"] * m["bulk_density_kg_m3"])
    assert m["carbon_usage_kg_s"] * m["operational_time_s"] == pytest.approx(m["carbon_mass_kg"])
    assert m["adsorbed_mass_kg"] + m["average_outlet_target_kg_s"] * m[
        "operational_time_s"
    ] == pytest.approx(
        feed.volumetric_flow_m3_s * feed.concentration_kg_m3 * m["operational_time_s"]
    )
    curve = result["breakthrough"]
    assert curve[0] == {"time_s": 0.0, "concentration_ratio": 0.0}
    assert curve[1]["concentration_ratio"] == 0.01
    assert curve[-1]["concentration_ratio"] == pytest.approx(spec.replacement_concentration_ratio)
    assert curve[-1]["time_s"] == pytest.approx(m["operational_time_s"])
    assert 0 < m["average_concentration_ratio"] < spec.replacement_concentration_ratio
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert GACDesignSpec.from_dict(spec.to_dict()) == spec


@pytest.mark.parametrize(
    "shared,advanced", [(False, False), (False, True), (True, False), (True, True)]
)
def test_current_original_feed_selected_vessel_and_finite_source_are_read_only(shared, advanced):
    runtime, spec = world_case(shared, advanced)
    before = json.dumps(runtime.snapshot().to_dict(), sort_keys=True)
    result = analyze_gac_design_world(runtime._state, spec)
    q = result["world_query"]
    assert q["anchor_vessel_id"] == "reactor"
    assert q["selected_vessel_id"] == ("product" if advanced else "reactor")
    assert q["world_advanced"] is False
    assert q["existing_media_used_as_fresh_bed"] is False
    assert q["source_covers_replacement"] is advanced
    assert 0 < q["actual_flow_m3_s"] < q["requested_flow_m3_s"]
    assert q["available_feed_volume_m3"] == pytest.approx(
        q["feed_volume_m3"] - q["pickup_volume_m3"]
    )
    assert q["inert_solutes"] == ["B"]
    assert result["feed"]["dynamic_viscosity_Pa_s"] == 0.001
    amounts = q["feed_amounts_mol"]
    # Original source is 50 mol/m3 A and 25 mol/m3 B, with zero solute volumes.
    assert amounts["A"] / q["feed_volume_m3"] == pytest.approx(50)
    assert amounts["B"] / q["feed_volume_m3"] == pytest.approx(25)
    assert result["feed"]["concentration_kg_m3"] == pytest.approx(50 * 0.030026)
    assert result["feed"]["liquid_density_kg_m3"] == pytest.approx(
        0.018015 / 18e-6 + 50 * 0.030026 + 25 * 0.046069
    )
    assert json.dumps(runtime.snapshot().to_dict(), sort_keys=True) == before


def test_query_does_not_claim_remaining_life_of_existing_loaded_media():
    runtime, spec = world_case()
    first = analyze_gac_design_world(runtime._state, spec)
    commit(runtime, "wait", duration_s=10.0)
    second = analyze_gac_design_world(runtime._state, spec)
    assert second["metrics"] == pytest.approx(first["metrics"])
    assert (
        second["world_query"]["feed_duration_before_pickup_s"]
        < first["world_query"]["feed_duration_before_pickup_s"]
    )
    assert first["diagnostics"]["fresh_bed"] is second["diagnostics"]["fresh_bed"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("freundlich_k", 0.0),
        ("freundlich_k", float("nan")),
        ("freundlich_exponent", 1.0),
        ("freundlich_exponent", True),
        ("source", ""),
        ("parameter_kind", "automatic"),
        ("concentration_unit", "mg/L"),
        ("loading_unit", "mol/kg"),
        ("concentration_bounds_kg_m3", (1.0, 0.1)),
        ("ratio_bounds", (0.02, 0.9)),
        ("ratio_bounds", (0.01, 1.0)),
        ("stanton_coefficients", (0.0, 0.0)),
        ("throughput_coefficients", (0.0, 0.0, 0.6, 0.0, 0.3)),
    ],
)
def test_rejects_unsourced_nonfinite_or_ambiguous_empirical_basis(field, value):
    spec, _ = gac_design_example()
    with pytest.raises(ValueError):
        replace(spec.empirical, **{field: value})


@pytest.mark.parametrize(
    "mutation", ["film", "surface", "diffusivity", "shape", "porosity", "tort"]
)
def test_independent_transport_modes_do_not_ignore_parameters(mutation):
    spec, _ = gac_design_example()
    correlated, _ = gac_design_example(calculated_film=True, calculated_surface=True)
    with pytest.raises(ValueError):
        if mutation == "film":
            replace(spec.transport, film_transfer_m_s=None)
        elif mutation == "surface":
            replace(spec.transport, surface_diffusion_m2_s=None)
        elif mutation == "diffusivity":
            replace(spec.transport, liquid_diffusivity_m2_s=1e-9)
        elif mutation == "shape":
            replace(spec.transport, shape_factor=1.0)
        elif mutation == "porosity":
            replace(correlated.transport, particle_porosity=1.0)
        else:
            replace(correlated.transport, tortuosity=0.5)


@pytest.mark.parametrize("where", ["root", "empirical", "bed", "transport"])
def test_codec_rejects_unknown_and_missing_members(where):
    spec, _ = gac_design_example()
    data = spec.to_dict()
    target = data if where == "root" else data[where]
    target["unrecognized"] = True
    with pytest.raises(ValueError):
        GACDesignSpec.from_dict(data)
    del target["unrecognized"]
    target.pop(next(iter(target)))
    with pytest.raises(ValueError):
        GACDesignSpec.from_dict(data)


@pytest.mark.parametrize("mode", ["unconfigured", "stopped", "rinse", "target", "anchor"])
def test_query_rejects_absent_or_incompatible_original_feed(mode):
    runtime, spec = world_case()
    with pytest.raises(ValueError):
        if mode == "unconfigured":
            runtime.reset(seed=49)
        elif mode == "stopped":
            commit(runtime, "set_flow_rate", flow_rate_mL_min=0.0)
        elif mode == "rinse":
            commit(runtime, "set_flow_rate", flow_rate_mL_min=10.0, flow_path="rinse")
        elif mode == "target":
            spec = replace(spec, target_species="water")
        else:
            analyze_gac_design_world(runtime._state, spec, anchor_vessel_id="product")
        analyze_gac_design_world(runtime._state, spec)


@pytest.mark.parametrize("value", [True, 1, 10001])
def test_no_degenerate_breakthrough_grid(value):
    spec, _ = gac_design_example()
    with pytest.raises(ValueError):
        replace(spec, breakthrough_elements=value)


def test_requested_flow_changes_geometry_and_usage_but_not_held_condition_breakthrough():
    spec, feed = gac_design_example()
    first = evaluate_gac_design(spec, feed)["metrics"]
    second = evaluate_gac_design(
        spec, replace(feed, volumetric_flow_m3_s=2 * feed.volumetric_flow_m3_s)
    )["metrics"]
    assert second["carbon_mass_kg"] == pytest.approx(2 * first["carbon_mass_kg"])
    assert second["carbon_usage_kg_s"] == pytest.approx(2 * first["carbon_usage_kg_s"])
    assert second["operational_time_s"] == first["operational_time_s"]
