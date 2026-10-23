import json

import pytest

from chemworld.physchem.packed_towers import (
    TowerPackingSpec,
    demister_removal_efficiency,
    mesh_voidage_from_mass,
    stichlmair_flooding_velocity,
    stichlmair_tower_pressure,
)
from chemworld.physchem.transport import FluidState


def test_irrigated_lower_branch_height_and_flood_rejection():
    original = TowerPackingSpec(0.68, 260, (32, 7, 1), "Explicit synthetic packing")
    packing = TowerPackingSpec.from_dict(json.loads(json.dumps(original.to_dict())))
    gas, liquid = FluidState(5, 5e-5), FluidState(1200, 0.001)
    dry = stichlmair_tower_pressure(packing, gas, gas_velocity_m_s=0.4)
    wet = stichlmair_tower_pressure(
        packing, gas, gas_velocity_m_s=0.4, liquid=liquid, liquid_velocity_m_s=0.005
    )
    assert dry.pressure_drop_Pa == pytest.approx(236.80904286559885)
    assert wet.pressure_drop_Pa == pytest.approx(539.876823725352)
    assert 0 < wet.liquid_holdup_fraction < packing.voidage
    assert abs(wet.dimensionless_residual) < 2e-10
    tall = stichlmair_tower_pressure(
        packing, gas, gas_velocity_m_s=0.4, liquid=liquid, liquid_velocity_m_s=0.005, height_m=3
    )
    assert tall.pressure_drop_Pa == pytest.approx(3 * wet.pressure_drop_Pa, rel=1e-12)
    assert tall.liquid_holdup_fraction == wet.liquid_holdup_fraction
    flood = stichlmair_flooding_velocity(packing, gas, liquid, liquid_velocity_m_s=0.005)
    assert flood.gas_superficial_velocity_m_s == pytest.approx(0.6394323542746928, rel=2e-8)
    assert flood.liquid_holdup_fraction > wet.liquid_holdup_fraction
    at_fold = stichlmair_tower_pressure(
        packing,
        gas,
        gas_velocity_m_s=flood.gas_superficial_velocity_m_s,
        liquid=liquid,
        liquid_velocity_m_s=0.005,
    )
    assert at_fold.branch == "flooding fold"
    with pytest.raises(ValueError, match="flooding"):
        stichlmair_tower_pressure(
            packing,
            gas,
            gas_velocity_m_s=flood.gas_superficial_velocity_m_s * 1.001,
            liquid=liquid,
            liquid_velocity_m_s=0.005,
        )


def test_unphysical_geometry_and_efficiency_are_not_silently_repaired():
    with pytest.raises(ValueError, match="voidage"):
        mesh_voidage_from_mass(1e6, 8000, 0.1, 0.1)
    args = {
        "superficial_velocity_m_s": 7.5,
        "voidage": 0.974,
        "wire_diameter_m": 0.0002,
        "droplet_diameter_m": 0.005,
    }
    with pytest.raises(ValueError, match="above one"):
        demister_removal_efficiency(**args)
    clipped = demister_removal_efficiency(**args, endpoint_policy="clip")
    assert clipped.removal_fraction == 1 and clipped.raw_prediction > 1 and clipped.clipped
    packing = TowerPackingSpec(0.68, 260, (32, 7, 1), "Explicit synthetic packing")
    with pytest.raises(ValueError, match="holdup"):
        stichlmair_flooding_velocity(
            packing, FluidState(5, 5e-5), FluidState(1200, 0.001), liquid_velocity_m_s=1
        )
