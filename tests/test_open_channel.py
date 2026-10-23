"""Declared SI channel coefficients, free-weir domains and source consumption."""

import json
from math import sqrt

import pytest

from chemworld.data.scientific_inputs import ScientificTable
from chemworld.examples.scientific_inputs import synthetic_source
from chemworld.foundation.dimensions import dimension_definition
from chemworld.physchem.open_channel import (
    WEIR_METHODS,
    chezy_to_manning,
    chezy_velocity,
    manning_roughness_table,
    manning_to_chezy,
    manning_velocity,
    select_manning_roughness,
    uniform_channel_flow,
    weir_flow,
)


def test_channel_conversion_flow_and_physical_units():
    n, radius, slope = .03, .2859, .005236
    c = manning_to_chezy(n, radius)
    assert chezy_to_manning(c, radius) == pytest.approx(n)
    assert manning_velocity(radius, slope, n) == pytest.approx(1.0467781958118971)
    assert chezy_velocity(radius, slope, c) == pytest.approx(manning_velocity(radius, slope, n))
    assert uniform_channel_flow(2, 2/radius, slope, roughness=n) == pytest.approx(
        2*1.0467781958118971)
    assert uniform_channel_flow(2, 3, 0, roughness=n) == 0
    assert dimension_definition("manning_roughness").canonical_unit == "s/m^(1/3)"
    with pytest.raises(ValueError, match="slope"):
        manning_velocity(1, -.01, n)
    with pytest.raises(ValueError, match="positive"):
        chezy_to_manning(0, 1)


@pytest.mark.parametrize("method", WEIR_METHODS)
def test_weir_families_and_reference_boundaries(method):
    data = {"head_m": .15, "crest_height_m": .8, "width_m": 2.}
    if method in {"kindsvater_carter", "sia_contracted_reference"}:
        data["channel_width_m"] = 6.
    disputed = "reference" in method
    if disputed:
        with pytest.raises(ValueError, match="reference_only"):
            weir_flow(method, **data)
    report = weir_flow(method, **data, reference_only=disputed)
    assert report.flow_m3_s > 0
    assert bool(report.warnings) == disputed
    assert report.reference_only == disputed
    assert json.loads(json.dumps(report.to_dict()))["method"] == method
    with pytest.raises(ValueError):
        weir_flow(method, **{**data, "head_m": .001}, reference_only=disputed)


def test_source_examples_and_geometry_are_explicit():
    assert weir_flow("ackers_full", head_m=.9, crest_height_m=.6,
                     width_m=5).flow_m3_s == pytest.approx(9.251938159899948)
    assert weir_flow("rehbock_full", head_m=.3, crest_height_m=.4,
                     width_m=2).flow_m3_s == pytest.approx(.6486856330601333)
    assert weir_flow("kindsvater_carter_full", head_m=.3, crest_height_m=.4,
                     width_m=2).flow_m3_s == pytest.approx(.641560300081563)
    with pytest.raises(ValueError, match="channel width"):
        weir_flow("kindsvater_carter", head_m=.1, crest_height_m=.6, width_m=.5)
    with pytest.raises(ValueError, match="angle"):
        weir_flow("shen_v", head_m=.1, crest_height_m=.8, width_m=2, angle_degrees=110)


def test_roughness_tables_keep_paths_statistics_and_sources():
    table = manning_roughness_table("test-roughness", {"lined": {"smooth": (.01, .02, .03)}},
                                    source=synthetic_source())
    restored = ScientificTable.from_dict(json.loads(json.dumps(table.to_dict())))
    selected = select_manning_roughness(restored, ("lined", "smooth"))
    assert selected.value == .02 and selected.unit == "s/m^(1/3)"
    assert selected.uncertainty is None
    assert restored.sources == table.sources
    assert manning_velocity(1, .01, selected.value) == pytest.approx(sqrt(.01)/.02)
    with pytest.raises(ValueError, match="exactly once"):
        select_manning_roughness(restored, ("missing",))
    with pytest.raises(ValueError, match="ordered"):
        manning_roughness_table("bad", {"x": (.03, .01, .04)}, source=synthetic_source())
    with pytest.raises(ValueError, match="leaf"):
        manning_roughness_table("bad", {"x": (.03, .04)}, source=synthetic_source())
