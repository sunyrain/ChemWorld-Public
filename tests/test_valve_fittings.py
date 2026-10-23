import json
from dataclasses import replace
from math import pi

import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.transport import FluidState, PipeSpec, pipe_pressure_drop
from chemworld.physchem.transport_devices import FiniteTransportSpec, TransportPathSpec
from chemworld.physchem.valve_fittings import (
    FittingCoefficientCatalog,
    change_loss_velocity_basis,
    crane_valve_coefficient,
    hooper_resistance,
    junction_loss_coefficients,
    minimum_valve_lift_velocity,
    nominal_size_from_bore,
    resistance_from_valve_coefficient,
    valve_coefficient_from_resistance,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.transport_devices import unpack_transport
from chemworld.world.transport_world import synthetic_transport_properties, transport_world_spec


def synthetic_table(columns, rows):
    source = InputSource(
        DataSourceProvenance("test", 0, "Synthetic fitting parameters", "1", "Test"), "synthetic"
    )
    return ScientificTable(
        "test",
        "parameters",
        tuple(QuantityColumn(k, u, "parameter", (0, 1e8), "Test") for k, u in columns),
        (source,),
        tuple(
            ScientificRow(
                str(i),
                "test",
                "test",
                str(i),
                "parameters",
                {
                    k: InputValue(v, u, None, "Test")
                    for (k, u), v in zip(columns, values, strict=True)
                },
            )
            for i, values in enumerate(rows)
        ),
    )


def test_catalog_units_roundtrip_and_nominal_extrapolation():
    table = synthetic_table(
        [("k1", "dimensionless"), ("k_infinity", "dimensionless")], [(500, 0.2)]
    )
    catalog = FittingCoefficientCatalog.from_dict(
        json.loads(json.dumps(FittingCoefficientCatalog(table, model="Hooper").to_dict()))
    )
    assert catalog.resistance("0", reynolds=1000, size=0.0254) == pytest.approx(0.9)
    assert catalog.to_dict()["table"] == table.to_dict()
    assert hooper_resistance(1000, 0.0508, k1=500, k_infinity=0.2) == pytest.approx(0.8)
    mapping = synthetic_table(
        [("inner_diameter", "mm"), ("nominal_size", "dimensionless")],
        [(10, 0.5), (20, 1.5), (30, 2.5)],
    )
    assert nominal_size_from_bore(mapping, 0.015) == pytest.approx(1)
    with pytest.raises(ValueError, match="outside"):
        nominal_size_from_bore(mapping, 0.035)
    assert nominal_size_from_bore(mapping, 0.035, extrapolate=True) == pytest.approx(3)


def test_valve_thresholds_and_geometry():
    low = crane_valve_coefficient("tilting check 5", pipe_diameter_m=0.2286, friction_factor=0.01)
    high = crane_valve_coefficient("tilting check 15", pipe_diameter_m=0.2286, friction_factor=0.01)
    assert low == 0.4
    assert high == 0.9
    assert minimum_valve_lift_velocity(
        "lift check straight", density_kg_m3=1000, throat_to_pipe_ratio=0.5
    ) == pytest.approx(12.5 / 1000**0.5)
    with pytest.raises(ValueError, match="throat"):
        crane_valve_coefficient("gate", pipe_diameter_m=0.1, throat_diameter_m=0.2)
    with pytest.raises(ValueError, match="unknown"):
        crane_valve_coefficient("unknown", pipe_diameter_m=0.1)


def test_signed_junction_recovery_and_combined_velocity_basis():
    result = junction_loss_coefficients(
        "diverging",
        pipe_diameter_m=0.1,
        branch_diameter_m=0.1,
        run_flow_m3_s=0.009,
        branch_flow_m3_s=0.001,
    )
    assert result.run < 0
    pressure = result.pressure_changes(1000)
    dynamic = 0.5 * 1000 * (0.01 / (pi * 0.1**2 / 4)) ** 2
    assert pressure[1] == pytest.approx(result.run * dynamic)
    assert result.basis == "combined_stream_velocity"
    with pytest.raises(ValueError, match="combined"):
        junction_loss_coefficients(
            "converging",
            pipe_diameter_m=0.1,
            branch_diameter_m=0.05,
            run_flow_m3_s=0,
            branch_flow_m3_s=0,
        )


def test_conversion_preserves_actual_pressure_for_different_velocity_basis():
    coefficient = resistance_from_valve_coefficient(2, diameter_m=0.05, scale="Cv")
    assert valve_coefficient_from_resistance(
        coefficient, diameter_m=0.05, scale="Cv"
    ) == pytest.approx(2)
    adjusted = change_loss_velocity_basis(coefficient, from_diameter_m=0.05, to_diameter_m=0.1)
    fluid = FluidState(1000, 0.001)
    first = pipe_pressure_drop(
        PipeSpec(1, 0.05, fittings_loss_coefficient=coefficient), fluid, volumetric_flow_m3_s=0.001
    )
    second = pipe_pressure_drop(
        PipeSpec(1, 0.1, fittings_loss_coefficient=adjusted), fluid, volumetric_flow_m3_s=0.001
    )
    bare_first = pipe_pressure_drop(PipeSpec(1, 0.05), fluid, volumetric_flow_m3_s=0.001)
    bare_second = pipe_pressure_drop(PipeSpec(1, 0.1), fluid, volumetric_flow_m3_s=0.001)
    assert first.pressure_drop_total_Pa - bare_first.pressure_drop_total_Pa == pytest.approx(
        second.pressure_drop_total_Pa - bare_second.pressure_drop_total_Pa
    )


def test_crane_resistance_consumed_by_finite_path_work_and_inventory():
    coefficient = crane_valve_coefficient("globe", pipe_diameter_m=0.001, friction_factor=0.02)
    original = TransportPathSpec(exchanger_area_m2=0)
    path = replace(original, pipe=replace(original.pipe, fittings_loss_coefficient=coefficient))
    spec = FiniteTransportSpec(synthetic_transport_properties(), path=path)
    runtime = ChemWorldRuntime(composition=transport_world_spec(spec))
    runtime.reset(seed=30)
    _, _, start = unpack_transport(runtime._state)
    for action in (
        {"operation": "set_flow_rate", "flow_rate_mL_min": 20},
        {"operation": "run_flow", "duration_s": 20},
    ):
        result = runtime.execute(action)
        assert result.info["transaction_status"] == "committed"
    _, _, final = unpack_transport(runtime._state)
    diagnostic = result.info["outcome_diagnostics"]["transport"]
    assert diagnostic["hydraulic_work_J"] > 0
    assert sum(final.enthalpies_J) - sum(start.enthalpies_J) == pytest.approx(
        diagnostic["hydraulic_work_J"], abs=1e-6
    )
    for component in start.amounts_mol[0]:
        assert sum(n[component] for n in start.amounts_mol) == pytest.approx(
            sum(n[component] for n in final.amounts_mol), abs=1e-12
        )
