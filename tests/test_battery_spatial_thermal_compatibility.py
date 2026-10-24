"""State and legacy-consumer contracts for the spatial thermal SPM.

The heat capacities and the constant-OCV energy integral below are independent
of the production thermal reconstruction.  No new trajectory is used as its own
golden reference.
"""

import json
from collections.abc import Mapping
from dataclasses import fields, replace

import numpy as np
import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.foundation.state_ledgers import EquipmentRecord
from chemworld.physchem.battery_heat_stack import (
    BatteryHeatStackSpec,
    HeatStackCollector,
    HeatStackLayer,
)
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    BatteryThermalSpec,
    F,
    ParticleSpec,
    battery_spec_from_dict,
)
from chemworld.physchem.battery_spatial_thermal import (
    BatterySpatialThermalSPMSpec,
    advance_spatial_thermal,
    spatial_thermal_fields,
)
from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime.battery import BATTERY_TYPE, initial_battery_slot
from test_battery_hysteresis import model as hysteresis_model


def source():
    return InputSource(
        DataSourceProvenance(
            "spatial-thermal-compatibility", 0,
            "Authored unequal-capacity spatial thermal state contracts", "1", "CC0-1.0",
        ),
        "synthetic",
    )


def stack(*, cooling=False):
    # Unequal capacities distinguish capacity mean from volume or arithmetic mean.
    return BatteryHeatStackSpec(
        negative=HeatStackLayer(0.001, 1e6, 0.7, 2, source()),
        separator=HeatStackLayer(0.0005, 2e6, 0.2, 3, source()),
        positive=HeatStackLayer(0.002, 3e6, 1.3, 2, source()),
        negative_collector=HeatStackCollector(
            0.0002, 4e6, 2.0 if cooling else 0.0, 0.0, 0.01, source(),
        ),
        positive_collector=HeatStackCollector(
            0.0003, 5e6, 5.0 if cooling else 0.0, 0.0, 0.02, source(),
        ),
        width_m=0.04, height_m=0.05,
        edge_heat_transfer_W_m2_K=0.0,
        ambient_temperature_K=298.15, source=source(),
    )


def independent_volumes_and_capacities():
    area = 0.04 * 0.05
    widths = np.array([0.0002, *([0.001 / 2] * 2), *([0.0005 / 3] * 3),
                       *([0.002 / 2] * 2), 0.0003])
    volumetric_capacity = np.array([4e6, 1e6, 1e6, 2e6, 2e6, 2e6, 3e6, 3e6, 5e6])
    return area * widths, area * widths * volumetric_capacity


def spatial_model(*, cooling=False, profile=None):
    return BatterySpatialThermalSPMSpec(
        negative=ParticleSpec(cells=4), positive=ParticleSpec(cells=5),
        heat_stack=stack(cooling=cooling),
        negative_ocv=ElectrodeOCV("negative-flat", source(), (0.2,)),
        positive_ocv=ElectrodeOCV("positive-flat", source(), (4.2,)),
        initial_temperatures_K=profile,
    )


def checkpoint(state):
    payload = {
        field.name: dict(value) if isinstance(value, Mapping) else value
        for field in fields(state)
        for value in (getattr(state, field.name),)
    }
    return CoupledState(**json.loads(json.dumps(payload, sort_keys=True, allow_nan=False)))


def test_capacity_mean_and_signed_deviations_reconstruct_every_node_without_redundancy():
    heat_stack = stack()
    volumes, capacities = independent_volumes_and_capacities()
    temperatures = np.array([293.0, 296.0, 301.0, 304.0, 299.0, 294.0, 307.0, 303.0, 310.0])
    expected_mean = float(capacities @ temperatures / capacities.sum())
    mean, coordinates = heat_stack.pack_temperatures(temperatures)
    np.testing.assert_allclose(heat_stack.volumes_m3, volumes, rtol=1e-14, atol=0)
    np.testing.assert_allclose(heat_stack.heat_capacities_J_K, capacities, rtol=1e-14, atol=0)
    assert mean == pytest.approx(expected_mean, abs=1e-12)
    assert abs(mean - float(volumes @ temperatures / volumes.sum())) > 1.0
    assert abs(mean - float(temperatures.mean())) > 1.0
    assert tuple(coordinates) == tuple(f"battery:thermal:delta:{i}_K" for i in range(8))
    assert any(value < 0 for value in coordinates.values())
    assert any(value > 0 for value in coordinates.values())
    np.testing.assert_allclose(tuple(coordinates.values()), temperatures[:-1] - mean,
                               rtol=0, atol=1e-12)
    recovered = heat_stack.temperatures(mean, dict(reversed(tuple(coordinates.items()))))
    np.testing.assert_allclose(recovered, temperatures, rtol=0, atol=1e-12)
    assert capacities @ (np.asarray(recovered) - mean) == pytest.approx(0, abs=2e-12)
    # A different bulk temperature translates the full field, including the
    # dependent collector; there is no second independently stored temperature.
    np.testing.assert_allclose(heat_stack.temperatures(mean + 2, coordinates),
                               temperatures + 2, rtol=0, atol=1e-12)


def test_source_preserving_codec_initializes_unique_bulk_and_thermal_coordinates():
    profile = (293.0, 296.0, 301.0, 304.0, 299.0, 294.0, 307.0, 303.0, 310.0)
    spec = spatial_model(profile=profile)
    payload = json.loads(json.dumps(spec.to_dict(), sort_keys=True, allow_nan=False))
    restored = battery_spec_from_dict(payload)
    assert restored == spec
    assert restored.heat_stack.source == source()
    for layer in (restored.heat_stack.negative, restored.heat_stack.separator,
                  restored.heat_stack.positive, restored.heat_stack.negative_collector,
                  restored.heat_stack.positive_collector):
        assert layer.source == source()
    initial = spec.initial_state()
    assert restored.initial_state() == initial
    assert tuple(initial.coordinates) == (
        "discharged_charge_C", "delivered_work_J", *spec.thermal_coordinate_keys,
    )
    assert len(initial.coordinates) == 10  # two receipts + exactly N-1 temperature deviations
    assert len(initial.amounts_mol) == 9  # 4+5 radial inventories, not heat-stack cells
    _, capacities = independent_volumes_and_capacities()
    assert initial.temperature_K == pytest.approx(capacities @ profile / capacities.sum())
    np.testing.assert_allclose(spec.heat_stack.temperatures(
        initial.temperature_K, initial.coordinates), profile, rtol=0, atol=1e-12)
    outputs = spatial_thermal_fields(spec, initial, current_A=0.05)
    volumes, _ = independent_volumes_and_capacities()
    assert outputs["volume_mean_temperature_K"] == pytest.approx(volumes @ profile / volumes.sum())
    assert outputs["heat_capacity_mean_temperature_K"] == initial.temperature_K
    assert outputs["negative_mean_temperature_K"] == 298.5
    assert outputs["positive_mean_temperature_K"] == 305.0
    assert outputs["maximum_temperature_K"] == pytest.approx(310.0, rel=0., abs=1e-12)
    assert checkpoint(initial) == initial


def test_equipment_slot_keeps_kelvin_signed_and_has_no_duplicate_bulk_temperature():
    spec = spatial_model(profile=(293., 296., 301., 304., 299., 294., 307., 303., 310.))
    record = EquipmentRecord("cell", BATTERY_TYPE, "reactor", batch_reset="preserve")
    slot = initial_battery_slot(record, spec)
    assert set(slot.values) == {*spec.inventory_keys, *spec.coordinate_keys}
    thermal_fields = [
        field for field in slot.spec.fields if field.name in spec.thermal_coordinate_keys
    ]
    assert len(thermal_fields) == 8
    for field in thermal_fields:
        assert field.unit == "K" and field.mixing == "equal" and field.minimum is None
    assert any(slot.values[field.name] < 0 for field in thermal_fields)
    assert not any("mean" in field.name or field.name == "temperature_K"
                   for field in slot.spec.fields)


@pytest.mark.parametrize("family", ["lumped", "mpm", "hysteresis"])
def test_old_equipment_fields_do_not_gain_spatial_thermal_coordinates(family):
    spec = (
        BatterySPMSpec(negative=ParticleSpec(cells=3), positive=ParticleSpec(cells=4),
                       thermal=BatteryThermalSpec())
        if family == "lumped" else (
            replace(hysteresis_model(mpm=True), negative_ocp=None, positive_ocp=None)
            if family == "mpm" else hysteresis_model()
        )
    )
    record = EquipmentRecord("old-cell", BATTERY_TYPE, "reactor", batch_reset="preserve")
    slot = initial_battery_slot(record, spec)
    expected = [*( (key, "mol", 0.0) for key in spec.inventory_keys),
                ("discharged_charge_C", "C", None), ("delivered_work_J", "J", None),
                *( (key, "dimensionless", -1.0) for key in spec.hysteresis_keys)]
    assert [(field.name, field.unit, field.minimum) for field in slot.spec.fields] == expected
    assert slot.values == {**spec.initial_state().amounts_mol, **spec.initial_state().coordinates}
    assert not any(key.startswith("battery:thermal:") for key in slot.values)


def test_json_resume_preserves_each_thermal_degree_of_freedom_and_one_clock():
    spec = spatial_model(profile=(293., 296., 301., 304., 299., 294., 307., 303., 310.))
    initial = spec.initial_state()
    rest = ScalarControlProgram(((0., 0.),), "A")
    first = advance_spatial_thermal(spec, initial, rest, 0.2)
    assert first.outcome == "success"
    # Heat conduction changes the field while leaving its capacity mean fixed.
    assert first.final_state.coordinates != initial.coordinates
    assert first.final_state.temperature_K == pytest.approx(initial.temperature_K, abs=2e-12)
    saved = checkpoint(first.final_state)
    decoded = battery_spec_from_dict(json.loads(json.dumps(spec.to_dict(), sort_keys=True)))
    expected = advance_spatial_thermal(spec, first.final_state, rest, 0.2)
    actual = advance_spatial_thermal(decoded, saved, rest, 0.2)
    assert actual.outcome == expected.outcome == "success"
    assert actual.final_state == expected.final_state
    final = actual.final_state
    assert final.time_s == pytest.approx(0.4)
    assert final.coordinates["discharged_charge_C"] == 0
    assert final.coordinates["delivered_work_J"] == 0
    assert final.amounts_mol == pytest.approx(initial.amounts_mol, abs=1e-14)
    initial_t = np.asarray(spec.initial_temperatures_K)
    final_t = spec.heat_stack.temperatures(final.temperature_K, final.coordinates)
    _, capacities = independent_volumes_and_capacities()
    assert capacities @ (final_t - initial_t) == pytest.approx(0, abs=5e-11)
    assert final_t.max() < initial_t.max() and final_t.min() > initial_t.min()


def test_two_intervals_keep_single_electrical_and_boundary_receipts_in_exact_energy_integral():
    spec = spatial_model(cooling=True, profile=(303., 304., 305., 304., 303., 302.,
                                              307., 306., 308.))
    initial = spec.initial_state()
    state = initial
    for current, duration in ((0.1, 1.0), (-0.1, 0.5)):
        programme = ScalarControlProgram(((0., current),), "A")
        result = advance_spatial_thermal(spec, state, programme, duration)
        assert result.outcome == "success"
        assert abs(result.energy_balance_residual_J) < 5e-7
        state = result.final_state
    assert state.time_s == pytest.approx(1.5)
    assert state.coordinates["discharged_charge_C"] == pytest.approx(0.05, abs=1e-12)
    assert state.heat_loss_J > 0
    assert state.boundary_heat_J == 0  # ambient loss has its own receipt; no second thermostat
    delivered = state.coordinates["delivered_work_J"]
    assert state.exchange_energy_J == pytest.approx({"battery:terminal": -delivered}, abs=1e-10)
    assert state.work_J == 0  # terminal EnergyPort must not also enter legacy mechanical work
    # Constant sourced OCV makes G=-F sum(U_e n_e) an exact independent integral.
    chemical_change = -F * sum(
        voltage * sum(state.amounts_mol[key] - initial.amounts_mol[key]
                      for key in particle.keys(name))
        for name, voltage, particle in (("negative", 0.2, spec.negative),
                                        ("positive", 4.2, spec.positive))
    )
    assert chemical_change == pytest.approx(-4.0 * 0.05, abs=1e-9)
    assert state.chemical_energy_change_J == pytest.approx(chemical_change, abs=1e-9)
    _, capacities = independent_volumes_and_capacities()
    final_t = spec.heat_stack.temperatures(state.temperature_K, state.coordinates)
    thermal_change = capacities @ (final_t - np.asarray(spec.initial_temperatures_K))
    assert chemical_change + thermal_change == pytest.approx(
        -delivered - state.heat_loss_J, abs=5e-7,
    )
    # A duplicated terminal work or boundary loss would exceed this test's tolerance.
    assert abs(delivered) > 1e-3 and state.heat_loss_J > 1e-3
