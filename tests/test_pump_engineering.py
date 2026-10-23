import json

import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.pump_engineering import (
    DriveEfficiencyMap,
    ElectricalSupplyCatalog,
    ElectricalSupplyRecord,
    MotorEfficiencyCatalog,
    MotorPartLoadCatalog,
    MotorSizeCatalog,
    PumpDriveSpec,
    corripio_pump_efficiency,
    ideal_motor_current,
    motor_synchronous_speed,
)
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.transport import FluidState, PipeSpec, PumpSpec, pipe_pressure_drop


def table(group, units, records):
    source = InputSource(
        DataSourceProvenance("test", 0, "Synthetic", "1", "Synthetic"), "synthetic"
    )
    return ScientificTable(
        "test",
        "parameters",
        tuple(
            QuantityColumn(k, u, "parameter", (-1e9, 1e9), "Synthetic") for k, u in units.items()
        ),
        (source,),
        tuple(
            ScientificRow(
                str(i),
                "test",
                group,
                str(i),
                "parameters",
                {k: InputValue(value, units[k], None, "Synthetic") for k, value in record.items()},
            )
            for i, record in enumerate(records)
        ),
    )


def test_drive_maps_keep_clip_explicit_and_original_parameter_provenance():
    data = table(
        "minimum:open:2",
        {"power": "W", "efficiency": "dimensionless"},
        [
            {"power": 100, "efficiency": 0.8},
            {"power": 200, "efficiency": 0.9},
        ],
    )
    catalog = MotorEfficiencyCatalog.from_dict(
        json.loads(json.dumps(MotorEfficiencyCatalog(data).to_dict()))
    )
    assert catalog.efficiency(150).efficiency == pytest.approx(0.85)
    with pytest.raises(ValueError, match="outside"):
        catalog.efficiency(50)
    report = catalog.efficiency(50, outside="clip")
    assert report.clipped and report.requested == {"power_W": 50}
    assert report.evaluated == {"power_W": 100} and report.source_ids == ("test",)
    with pytest.raises(ValueError, match="curve"):
        catalog.efficiency(150, poles=8)
    invalid = table(
        "invalid",
        {"power": "W", "efficiency": "dimensionless"},
        [
            {"power": 100, "efficiency": -0.2},
            {"power": 200, "efficiency": 0.9},
        ],
    )
    with pytest.raises(ValueError, match="efficiency"):
        DriveEfficiencyMap(
            TabulatedScalar(invalid, (("p", "power", "W"),), "efficiency", "dimensionless")
        )


def test_motor_size_and_part_load_are_sourced_and_roundtrip():
    sizes = MotorSizeCatalog(table("sizes", {"power": "W"}, [{"power": 10}, {"power": 20}]))
    sizes = MotorSizeCatalog.from_dict(json.loads(json.dumps(sizes.to_dict())))
    assert sizes.select(11).values["power"].value == 20
    with pytest.raises(ValueError, match="exceeds"):
        sizes.select(21)
    units = {
        "upper_power": "W",
        "unity_load": "dimensionless",
        **{f"c{i}": "dimensionless" for i in range(5)},
    }
    loads = MotorPartLoadCatalog(
        table(
            "loads",
            units,
            [
                {
                    "upper_power": 20,
                    "unity_load": 0.8,
                    "c0": 0.5,
                    "c1": 0.5,
                    "c2": 0,
                    "c3": 0,
                    "c4": 0,
                },
            ],
        )
    )
    loads = MotorPartLoadCatalog.from_dict(json.loads(json.dumps(loads.to_dict())))
    assert loads.factor(10, 0.5) == 0.75
    assert loads.factor(10, 0.9) == 1
    with pytest.raises(ValueError, match="load fraction"):
        loads.factor(10, 1.1)


def test_source_supply_labels_duplicates_and_missing_plugs_survive_roundtrip():
    source = InputSource(
        DataSourceProvenance("test", 0, "Synthetic", "1", "Synthetic"), "synthetic"
    )
    record = ElectricalSupplyRecord(
        "test", "Test", "industrial", 50, (120, 240, 120), None, source
    )
    catalog = ElectricalSupplyCatalog((record,), ("TEST",), (50,), (230,), (120, 240), source)
    restored = ElectricalSupplyCatalog.from_dict(json.loads(json.dumps(catalog.to_dict())))
    assert restored.lookup("industrial", "test") == record
    assert restored.residential_voltages_V == (230,)
    with pytest.raises(ValueError, match="unknown"):
        restored.lookup("residential", "test")


def test_original_pump_current_fluid_and_electrical_energy_share_operating_point():
    pump = PumpSpec(1e5, 0.1, 0.7)
    drive = PumpDriveSpec.from_dict(
        json.loads(json.dumps(PumpDriveSpec(pump, 0.002, 0.9).to_dict()))
    )
    pipe = PipeSpec(1, 0.005)
    results = []
    for mu in (0.001, 0.1):
        fluid = FluidState(1000, mu)
        report = drive.operating_point(
            1e-5,
            lambda q, fluid=fluid: (
                pipe_pressure_drop(pipe, fluid, volumetric_flow_m3_s=q).pressure_drop_total_Pa
            ),
        )
        assert report["shaft_power_W"] == pytest.approx(0.002, rel=1e-8)
        assert report["electrical_power_W"] == pytest.approx(report["shaft_power_W"] / 0.9)
        assert report["total_external_loss_W"] + report["hydraulic_power_W"] == pytest.approx(
            report["electrical_power_W"]
        )
        results.append(report["actual_flow_m3_s"])
    assert results[1] < results[0]
    assert drive.operating_point(0, lambda q: q)["electrical_power_W"] == 0
    assert ideal_motor_current(230, 230, phases=1) == 1
    assert motor_synchronous_speed(50, 4) == 1500
    with pytest.raises(ValueError, match="poles"):
        motor_synchronous_speed(50, 3)
    with pytest.raises(ValueError, match="50"):
        corripio_pump_efficiency(1e-9)
