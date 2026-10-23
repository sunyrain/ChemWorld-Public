import json

import pytest

from chemworld.physchem.equipment_geometry import (
    EquipmentGeometry,
    air_cooled_geometry,
    fin_exchanger_geometry,
    helical_coil_geometry,
)
from chemworld.physchem.geometric_primitives import (
    CoolingTowerGeometry,
    circular_segment_area,
    circular_segment_height,
)
from chemworld.physchem.transport import FluidState, pipe_pressure_drop


def test_coil_real_pipe_geometry_roundtrip_and_touching_boundary():
    coil = helical_coil_geometry(
        tube_diameter_m=0.01,
        total_diameter_m=0.21,
        total_height_m=0.03,
        turns=2,
        inner_diameter_m=0.008,
    )
    assert coil.metrics["pitch"] == 0.01
    restored = EquipmentGeometry.from_dict(json.loads(json.dumps(coil.to_dict())))
    assert restored.metrics == coil.metrics
    pipe = restored.tube_pipe()
    assert pipe.diameter_m == 0.008
    assert pipe.length_m == coil.metrics["tube_length"]
    assert (
        pipe_pressure_drop(
            pipe, FluidState(1000, 0.001), volumetric_flow_m3_s=1e-6
        ).pressure_drop_total_Pa
        > 0
    )
    with pytest.raises(ValueError, match="collide"):
        helical_coil_geometry(tube_diameter_m=0.01, coil_diameter_m=0.2, pitch_m=0.009, turns=2)


@pytest.mark.parametrize("height", [1e-12, 1e-8, 0.1, 0.5, 0.9])
def test_small_and_large_circle_segments_have_a_relative_inverse(height):
    area = circular_segment_area(1, height)
    assert circular_segment_height(1, area) == pytest.approx(height, rel=1e-8, abs=1e-20)


def test_pitch_ratio_air_cooler_and_full_fin_layer_height():
    cooler = air_cooled_geometry(
        tube_rows=2,
        tube_passes=1,
        tubes_per_row=10,
        tube_length_m=2,
        tube_diameter_m=0.025,
        fin_thickness_m=0.0003,
        angle_degrees=60,
        pitch_ratio=3.2,
        fin_diameter_m=0.05,
        fin_density_per_m=200,
        tube_thickness_m=0.002,
    )
    assert cooler.metrics["pitch"] == pytest.approx(0.08)
    assert cooler.metrics["Di"] == pytest.approx(0.021)
    assert cooler.metrics["A_min"] > 0
    assert (
        EquipmentGeometry.from_dict(json.loads(json.dumps(cooler.to_dict()))).metrics
        == cooler.metrics
    )
    fin = fin_exchanger_geometry(
        fin_height_m=0.005,
        fin_thickness_m=0.0001,
        fin_spacing_m=0.003,
        length_m=0.4,
        width_m=0.3,
        layers=10,
        plate_thickness_m=0.0005,
    )
    assert fin.metrics["height"] == pytest.approx(0.0555)
    assert fin.metrics["volume"] == pytest.approx(0.4 * 0.3 * 0.0555)


def test_tower_reconstructs_profile_from_json():
    tower = CoolingTowerGeometry(10, 100, 65, inlet_diameter_m=80)
    restored = CoolingTowerGeometry.from_dict(json.loads(json.dumps(tower.to_dict())))
    for actual, expected in zip(restored.sample_profile(11), tower.sample_profile(11), strict=True):
        assert actual == pytest.approx(expected)
    assert tower.diameter(tower.throat_height_m) == tower.throat_diameter_m
    assert tower.diameter(100) == pytest.approx(65)
