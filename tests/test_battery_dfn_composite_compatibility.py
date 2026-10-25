"""Original state/codec consumers of the phase-resolved spatial DFN.

These contracts concern ownership and declared model boundaries. The physical
tests separately exercise current sharing, transport, Jacobians and events.
"""

import json
from collections.abc import Mapping
from dataclasses import fields, replace

import numpy as np
import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.battery_composite import ActiveMaterialPhaseSpec
from chemworld.physchem.battery_dfn_composite import (
    BatteryCompositeDFNSpec,
    CompositeDFNReadout,
    composite_dfn_fields,
    composite_dfn_readout,
)
from chemworld.physchem.battery_electrolyte import ELECTROLYTE_SPECIES, BatteryElectrolyteSpec
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import (
    BATTERY_COORDINATES,
    ParticleSpec,
    battery_chemical_energy_j,
    battery_chemical_potentials,
    battery_dissipation_w,
    battery_sei_affinity_j_mol,
    battery_spec_from_dict,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance


def model():
    """Explicit unequal cells, phases and capacities; no production default oracle."""
    def phase(name, cells, volume, radius, cmax, diffusion, j0, coefficients, theta):
        source = InputSource(DataSourceProvenance(
            "chemworld:composite-dfn-compat:" + name, 0,
            "Authored unequal phase/cell ownership fixture", "1", "MIT",
        ), "synthetic")
        return ActiveMaterialPhaseSpec(
            name,
            ParticleSpec(cells=cells, active_volume_m3=volume, radius_m=radius,
                         maximum_concentration_mol_m3=cmax, diffusivity_m2_s=diffusion,
                         exchange_current_scale_A_m2=j0),
            ElectrodeOCV("compat-" + name, source, coefficients,
                         supported_models=("DFN",)),
            theta, source,
        )

    return BatteryCompositeDFNSpec(
        negative_phases=(
            phase("alpha", 3, 3e-7, 5e-6, 30000., 3e-14, 2., (.25, -.2), .6),
            phase("beta", 5, 1e-7, 4e-6, 50000., 1e-14, 1.8, (.28, -.25), .55),
        ),
        positive_phase=phase("positive", 4, 4e-7, 6e-6, 32000., 2e-14, 3., (4.2, -.8), .3),
        electrolyte=BatteryElectrolyteSpec(
            cells=(3, 2, 4), area_m2=.01, lengths_m=(1e-4, 3e-5, 9e-5),
            porosities=(.4, .5, .4), solid_conductivity_S_m=(3., 7.),
        ),
        series_resistance_ohm=.03,
        provenance="synthetic-phase-cell-ownership-v1",
    )


def snapshot(state):
    return {f.name: dict(v) if isinstance(v, Mapping) else v
            for f in fields(state) for v in (getattr(state, f.name),)}


def test_phase_cell_and_salt_ownership_has_no_duplicate_mean_stock_or_housing():
    spec = model()
    state = spec.initial_state()
    expected = tuple(
        f"{electrode}:phase:{phase}:x:{x}:Li:{r}"
        for electrode, phase, nx, nr in (
            ("negative", "alpha", 3, 3), ("negative", "beta", 3, 5),
            ("positive", "positive", 4, 4),
        )
        for x in range(nx) for r in range(nr)
    )
    assert spec.inventory_keys == (*expected, *spec.electrolyte.keys)
    assert len(set(spec.inventory_keys)) == len(spec.inventory_keys) == 49
    assert not hasattr(spec, "negative") and not hasattr(spec, "positive")
    assert set(state.amounts_mol) == set(spec.inventory_keys)
    assert spec.inventory_species == {
        **dict.fromkeys(expected, "Li"),
        **dict.fromkeys(spec.electrolyte.keys, ELECTROLYTE_SPECIES),
    }
    assert spec.coordinate_keys == BATTERY_COORDINATES
    assert tuple(state.coordinates) == BATTERY_COORDINATES
    assert spec.hysteresis_keys == ()
    assert state.volume_L == pytest.approx(.01 * (1e-4 + 3e-5 + 9e-5) * 1000, rel=0, abs=1e-18)
    assert state.volume_L != 1000 * sum(p.particle.active_volume_m3 for p in spec.phases)

    expected_cell = (0, 1, 2, 0, 1, 2, 3, 4, 5, 6)
    expected_salt = (0, 1, 2, 0, 1, 2, 5, 6, 7, 8)
    assert tuple(spec.bin_cell_indices) == expected_cell
    assert tuple(spec.bin_electrolyte_indices) == expected_salt
    assert tuple(p.phase_id for p in spec.bin_phases) == (
        *("alpha" for _ in range(3)), *("beta" for _ in range(3)),
        *("positive" for _ in range(4)),
    )
    for phase in spec.phases:
        records = [(b, p) for b, p in zip(spec.particle_bins, spec.bin_phases, strict=True)
                   if p.phase_id == phase.phase_id]
        count = 4 if phase.phase_id == "positive" else 3
        assert len(records) == count
        assert sum(b[2].active_volume_m3 for b, _ in records) == pytest.approx(
            phase.particle.active_volume_m3, rel=1e-15,
        )
        assert sum(b[2].area_m2 for b, _ in records) == pytest.approx(
            3 * phase.particle.active_volume_m3 / phase.particle.radius_m, rel=1e-15,
        )
        phase_amount = 0.
        for (electrode, x, particle, keys, identity), stored_phase in records:
            assert stored_phase == phase
            assert identity == f"{electrode}:phase:{phase.phase_id}:x:{x}"
            assert keys == particle.keys(identity)
            local = np.array([state.amounts_mol[k] for k in keys])
            np.testing.assert_allclose(
                local / (particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3),
                phase.initial_fraction, rtol=1e-15, atol=0,
            )
            phase_amount += sum(local)
        assert phase_amount == pytest.approx(
            phase.particle.capacity_mol * phase.initial_fraction, rel=1e-15,
        )
    np.testing.assert_allclose(
        spec.electrolyte.concentrations(state), 1000., rtol=1e-15, atol=0,
    )


def test_json_codec_keeps_all_phase_sources_and_derived_maps_are_not_state():
    spec = model()
    _ = spec.particle_bins, spec.bin_phases, spec.bin_cell_indices, spec.bin_electrolyte_indices
    payload = json.loads(json.dumps(spec.to_dict(), sort_keys=True, allow_nan=False))
    decoded = battery_spec_from_dict(payload)
    assert type(decoded) is BatteryCompositeDFNSpec
    assert decoded.to_dict() == payload
    assert decoded.initial_state() == spec.initial_state()
    assert decoded.inventory_keys == spec.inventory_keys
    assert decoded.particle_bins == spec.particle_bins
    assert payload["model_family"] == "composite-dfn-isothermal-v1"
    assert not {"negative", "positive", "particle_bins", "bin_phases",
                "bin_cell_indices", "numerical_form"}.intersection(payload)
    for original, restored in zip(spec.phases, decoded.phases, strict=True):
        assert original.to_dict() == restored.to_dict()
        assert restored.ocv.supported_models == ("DFN",)
        assert restored.source.data_kind == restored.ocv.source.data_kind == "synthetic"
    assert decoded.numerical_form == "residual"


def test_readout_codec_roundtrips_three_phase_inventory_and_rejects_ambiguity():
    spec = model()
    readout = composite_dfn_readout(spec, spec.initial_state(), current_A=0.1)
    payload = json.loads(json.dumps(readout.to_dict(), allow_nan=False))
    restored = CompositeDFNReadout.from_dict(payload)
    assert restored == readout

    unknown = dict(payload, unexpected=True)
    with pytest.raises(ValueError, match="unknown composite DFN readout fields"):
        CompositeDFNReadout.from_dict(unknown)

    duplicate = json.loads(json.dumps(payload))
    duplicate["phase_readouts"][1]["phase_id"] = duplicate["phase_readouts"][0]["phase_id"]
    with pytest.raises(ValueError, match="identities must be unique"):
        CompositeDFNReadout.from_dict(duplicate)

    wrong_count = json.loads(json.dumps(payload))
    wrong_count["phase_readouts"].pop()
    with pytest.raises(ValueError, match="exactly three"):
        CompositeDFNReadout.from_dict(wrong_count)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_each_ocv_must_declare_dfn_not_inherit_spm_label(index):
    spec = model()
    phases = list(spec.phases)
    phases[index] = replace(
        phases[index], ocv=replace(phases[index].ocv, supported_models=("SPM",)),
    )
    with pytest.raises(ValueError, match="DFN"):
        replace(spec, negative_phases=tuple(phases[:2]), positive_phase=phases[2])


def test_phase_order_is_a_declared_identity_not_an_unlabelled_population():
    spec = model()
    reversed_spec = replace(spec, negative_phases=tuple(reversed(spec.negative_phases)))
    assert reversed_spec.inventory_keys != spec.inventory_keys
    assert dict(reversed_spec.initial_state().amounts_mol) == dict(spec.initial_state().amounts_mol)
    assert reversed_spec.to_dict()["negative_phases"][0]["phase_id"] == "beta"
    assert reversed_spec.bin_cell_indices == spec.bin_cell_indices


@pytest.mark.parametrize("key", [
    "negative", "positive", "thermal", "sei", "plating", "negative_ocp",
    "positive_ocp", "negative_distribution", "numerical_form", "initial_amounts_mol",
])
def test_unsupported_or_parallel_configuration_is_not_silently_discarded(key):
    payload = json.loads(json.dumps(model().to_dict()))
    payload[key] = None
    before = json.loads(json.dumps(payload))
    with pytest.raises((ValueError, TypeError), match="composite DFN"):
        battery_spec_from_dict(payload)
    assert payload == before


@pytest.mark.parametrize("key", ["negative_phases", "positive_phase", "electrolyte"])
def test_required_spatial_and_phase_declarations_cannot_fall_back_to_default(key):
    payload = json.loads(json.dumps(model().to_dict()))
    del payload[key]
    with pytest.raises((ValueError, TypeError, KeyError)):
        battery_spec_from_dict(payload)


def test_original_equipment_slot_owns_all_stocks_once():
    from chemworld.runtime.battery import initial_battery_slot, unpack_battery
    from chemworld.world.battery_world import battery_world_spec

    spec = model()
    world = battery_world_spec(spec)
    original = world.physics.initial_states["reactor"]
    before = original.to_dict()
    record, slot, decoded, state, _ = unpack_battery(original)
    assert slot == initial_battery_slot(record, decoded)
    assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
    assert set(slot.values) == {*spec.inventory_keys, *BATTERY_COORDINATES}
    assert tuple(f.name for f in slot.spec.fields) == (*spec.inventory_keys, *BATTERY_COORDINATES)
    assert len(slot.values) == 51
    assert state == spec.initial_state()
    composite_dfn_fields(decoded, state, current_A=.01)
    assert original.to_dict() == before


@pytest.mark.parametrize("operation", [
    "impedance", "soh", "energy", "chemical_potentials", "dissipation", "sei_affinity",
    "soc", "thermal",
])
def test_undeclared_analysis_does_not_approximate_two_phases_as_one(operation):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
    from chemworld.world.battery_world import battery_world_spec

    spec = model()
    state = spec.initial_state()
    before = snapshot(state), spec.to_dict()
    calls = {
        "impedance": lambda: battery_impedance(spec, [.01], state=state),
        "soh": lambda: soh_from_battery(spec, state),
        "energy": lambda: battery_chemical_energy_j(spec, state),
        "chemical_potentials": lambda: battery_chemical_potentials(spec, state),
        "dissipation": lambda: battery_dissipation_w(spec, state, .1),
        "sei_affinity": lambda: battery_sei_affinity_j_mol(spec, state, .1),
        "soc": lambda: battery_spec_at_soc(spec, None, .5),
        "thermal": lambda: battery_world_spec(spec, thermal_device=ThermalDeviceSpec()),
    }
    with pytest.raises(ValueError, match=r"(?i)composite.*DFN|DFN.*composite"):
        calls[operation]()
    assert (snapshot(state), spec.to_dict()) == before
