import json
from dataclasses import replace

import pytest

from chemworld.physchem.packed_beds import (
    PACKED_BED_METHODS,
    PackedBedSpec,
    available_packed_bed_methods,
    packed_bed_pressure,
    packed_bed_voidage,
)
from chemworld.physchem.transport import FluidState, packed_bed_pressure_drop_ergun
from chemworld.physchem.transport_devices import TransportPathSpec, advance_transport
from test_transport_device_coupling import session, unpack_transport


def test_ergun_reuses_original_and_all_methods_have_physical_zero():
    fluid = FluidState(998, .001)
    for method in PACKED_BED_METHODS:
        spec = PackedBedSpec(method, .2, .001, .4,
                             tube_diameter_m=.0025 if method == "Guo_Sun" else None)
        assert packed_bed_pressure(spec, fluid, superficial_velocity_m_s=0).pressure_drop_Pa == 0
    spec = PackedBedSpec("Ergun", .2, .001, .4, .8)
    original = packed_bed_pressure_drop_ergun(
        fluid=fluid, superficial_velocity_m_s=.02, bed_length_m=.2,
        particle_diameter_m=.001, void_fraction=.4, sphericity=.8)
    assert packed_bed_pressure(spec, fluid, superficial_velocity_m_s=.02).pressure_drop_Pa == (
        original.pressure_drop_Pa)
    assert PackedBedSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_explicit_geometry_domain_and_nonphysical_prediction():
    assert "Guo_Sun" not in available_packed_bed_methods()
    assert len(available_packed_bed_methods(tube_diameter_m=.01)) == 14
    with pytest.raises(ValueError, match="explicit"):
        PackedBedSpec("Guo_Sun", .2, .001, .4)
    with pytest.raises(ValueError, match="does not consume"):
        PackedBedSpec("Ergun", .2, .001, .4, tube_diameter_m=.01)
    spec = PackedBedSpec("Guo_Sun", .2, .001, .4, tube_diameter_m=.1)
    with pytest.raises(ValueError, match="nonphysical"):
        packed_bed_pressure(spec, FluidState(998, .001), superficial_velocity_m_s=1)
    spec = PackedBedSpec("KTA", .2, .001, .5)
    assert packed_bed_pressure(spec, FluidState(998, .001),
                               superficial_velocity_m_s=.02).range_notes
    with pytest.raises(ValueError, match="voidage"):
        packed_bed_pressure(spec, FluidState(998, .001),
                            superficial_velocity_m_s=.02, range_policy="raise")
    with pytest.raises(ValueError, match="does not consume"):
        packed_bed_voidage(particle_diameter_m=.001, tube_diameter_m=.01,
                           particle_shape="cylindrical", sphericity=.8)


def test_selected_bed_law_reaches_original_pump_and_conserving_inventory():
    runtime = session()
    _, spec, initial = unpack_transport(runtime._state)
    path = replace(spec.path, bed_length_m=.2, particle_diameter_m=.0002,
                   bed_method="KTA", bed_sphericity=.9)
    restored = TransportPathSpec.from_dict(json.loads(json.dumps(path.to_dict())))
    assert restored == path
    _, plain = advance_transport(spec, initial, 20/6e7, 30)
    loaded, diagnostics = advance_transport(replace(spec, path=restored), initial, 20/6e7, 30)
    assert diagnostics["actual_flow_m3_s"] < plain["actual_flow_m3_s"]
    assert loaded.enthalpy_residual_J == pytest.approx(0, abs=1e-7)
    assert loaded.actual_duration_s == 30
    fluid = FluidState(998, .001)
    selected = path.hydraulic_diagnostics(fluid, 1e-8)["packed_bed"]
    assert selected["method"] == "KTA"
    hotter = path.hydraulic_diagnostics(replace(fluid, viscosity_Pa_s=.0005), 1e-8)["packed_bed"]
    assert hotter["pressure_drop_Pa"] < selected["pressure_drop_Pa"]
