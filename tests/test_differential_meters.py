import json
from dataclasses import replace
from math import nextafter, pi, sqrt

import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.differential_meters import DifferentialMeterSpec, solve_differential_meter
from chemworld.physchem.flow_meters import (
    LiquidOrificeSpec,
    liquid_orifice_mass_flow,
    orifice_discharge_coefficient,
)
from chemworld.physchem.meter_correlations import (
    meter_beta,
    meter_expansibility,
    miller_orifice_coefficient,
)
from chemworld.physchem.meter_tables import MeterTableSurface
from chemworld.physchem.transport import FluidState, PipeSpec
from chemworld.physchem.transport_devices import FiniteTransportSpec, TransportPathSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
from chemworld.runtime.transport_devices import unpack_transport
from chemworld.world.transport_world import synthetic_transport_properties, transport_world_spec


def test_shared_rhg_kernel_preserves_original_liquid_orifice():
    old = LiquidOrificeSpec(0.1, 0.05, installation_note="Synthetic compliant geometry")
    original = liquid_orifice_mass_flow(
        old, differential_pressure_Pa=2e4, density_kg_m3=1000, viscosity_Pa_s=0.001
    )
    new = solve_differential_meter(
        DifferentialMeterSpec(0.1, 0.05),
        FluidState(1000, 0.001),
        upstream_pressure_Pa=2e5,
        downstream_pressure_Pa=1.8e5,
    )
    assert new.mass_flow_kg_s == pytest.approx(original.mass_flow_kg_s, rel=1e-9)
    assert new.permanent_loss_Pa == pytest.approx(original.permanent_pressure_loss_Pa, rel=1e-9)
    assert original.discharge_coefficient == orifice_discharge_coefficient(
        original.reynolds, pipe_diameter_m=0.1, beta=0.5
    )
    with pytest.raises(ValueError, match="Reynolds"):
        old.coefficient(1)
    assert orifice_discharge_coefficient(1e5, pipe_diameter_m=0.1, beta=0.5, taps="flange") > 0


def test_gas_equal_pressure_isothermal_limit_and_effective_wedge_area():
    beta = meter_beta(0.1, 0.05, "wedge meter")
    assert beta == pytest.approx(sqrt(0.5))
    assert meter_beta(0.1, 0.05, "cone meter") == pytest.approx(sqrt(0.75))
    for k in (1, 1 + 1e-12, 1.4):
        assert meter_expansibility(beta, 1, k, method="nozzle") == 1
    assert meter_expansibility(beta, 0.9, 1, method="nozzle") == pytest.approx(
        meter_expansibility(beta, 0.9, 1 + 1e-12, method="nozzle"), rel=1e-11
    )
    with pytest.raises(ValueError, match="domain"):
        meter_expansibility(0.5, 0.5, 1.4)
    with pytest.raises(ValueError, match="unknown"):
        meter_expansibility(0.5, 1, 1.4, method="unknown")


def test_miller_threshold_and_tap_branches():
    assert miller_orifice_coefficient(0.3, 0.1, 150, subtype="conical orifice") == 0.734
    assert (
        miller_orifice_coefficient(
            0.3, 0.1, nextafter(150, float("inf")), subtype="conical orifice"
        )
        == 0.73
    )
    corner = miller_orifice_coefficient(0.6, 0.1, 1e5)
    assert miller_orifice_coefficient(0.6, 0.1, 1e5, taps="flange") != corner
    with pytest.raises(ValueError, match="supports"):
        miller_orifice_coefficient(0.6, 0.1, 1e5, subtype="eccentric orifice", taps="corner")


def test_all_inverse_directions_json_and_loss_availability():
    spec = DifferentialMeterSpec(0.1, 0.045, meter_type="cone meter", reynolds_bounds=(1, 1e7))
    spec = DifferentialMeterSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    fluid = FluidState(10, 2e-5)
    flow = solve_differential_meter(
        spec, fluid, upstream_pressure_Pa=2e5, downstream_pressure_Pa=1.8e5, heat_capacity_ratio=1.4
    )
    assert abs(flow.normalized_mass_residual) < 1e-9
    assert 0 < flow.permanent_loss_Pa < 2e4
    common = {"mass_flow_kg_s": flow.mass_flow_kg_s, "heat_capacity_ratio": 1.4}
    p2 = solve_differential_meter(spec, fluid, upstream_pressure_Pa=2e5, **common)
    p1 = solve_differential_meter(spec, fluid, downstream_pressure_Pa=1.8e5, **common)
    dim = solve_differential_meter(
        replace(spec, secondary_dimension_m=0.06),
        fluid,
        upstream_pressure_Pa=2e5,
        downstream_pressure_Pa=1.8e5,
        solve_secondary_dimension=True,
        **common,
    )
    assert p2.downstream_pressure_Pa == pytest.approx(1.8e5, rel=1e-9)
    assert p1.upstream_pressure_Pa == pytest.approx(2e5, rel=1e-9)
    assert dim.spec.secondary_dimension_m == pytest.approx(0.045, rel=1e-9)
    unknown = solve_differential_meter(
        DifferentialMeterSpec(0.1, 0.05, meter_type="venturi nozzle"),
        fluid,
        upstream_pressure_Pa=2e5,
        downstream_pressure_Pa=1.8e5,
        heat_capacity_ratio=1.4,
    )
    assert unknown.permanent_loss_Pa is None
    zero = solve_differential_meter(
        spec, fluid, upstream_pressure_Pa=2e5, downstream_pressure_Pa=2e5
    )
    assert zero.mass_flow_kg_s == zero.permanent_loss_Pa == 0
    with pytest.raises(ValueError, match="forward"):
        solve_differential_meter(spec, fluid, upstream_pressure_Pa=1e5, downstream_pressure_Pa=2e5)


def test_source_table_json_domains_and_actual_solved_flow():
    source = InputSource(
        DataSourceProvenance("synthetic", 0, "Synthetic meter calibration", "1", "Synthetic"),
        "synthetic",
    )
    columns = tuple(
        QuantityColumn(name, "dimensionless", "parameter", (0, 1e9), "Synthetic")
        for name in ("beta", "reynolds", "coefficient")
    )
    rows = tuple(
        ScientificRow(
            str(i),
            "synthetic",
            "calibration",
            str(i),
            "parameters",
            {
                "beta": InputValue(0.5, "dimensionless", None, "Synthetic"),
                "reynolds": InputValue(re, "dimensionless", None, "Synthetic"),
                "coefficient": InputValue(0.7, "dimensionless", None, "Synthetic"),
            },
        )
        for i, re in enumerate((1, 1e4, 1e8))
    )
    table = MeterTableSurface(ScientificTable("meter", "parameters", columns, (source,), rows))
    spec = DifferentialMeterSpec(
        0.1,
        0.05,
        meter_type="Hollingshead orifice",
        coefficient_table=table,
        reynolds_bounds=(1, 1e8),
        installation_note="Synthetic calibration scope",
    )
    spec = DifferentialMeterSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    flow = solve_differential_meter(
        spec, FluidState(1000, 0.001), upstream_pressure_Pa=2e5, downstream_pressure_Pa=1.9e5
    )
    expected = pi * 0.05**2 / 4 * 0.7 * sqrt(2 * 1000 * 1e4 / (1 - 0.5**4))
    assert flow.mass_flow_kg_s == pytest.approx(expected, rel=1e-9)
    path = TransportPathSpec(pipe=PipeSpec(1, 0.1), differential_meter=spec)
    restored = TransportPathSpec.from_dict(json.loads(json.dumps(path.to_dict())))
    assert restored.differential_meter.coefficient(1e4) == pytest.approx(0.7)
    assert restored.to_dict() == path.to_dict()
    with pytest.raises(ValueError, match="domain"):
        table.evaluate(0.5, 1e9)
    with pytest.raises(ValueError, match="tables"):
        DifferentialMeterSpec(0.1, 0.05, meter_type="Hollingshead orifice")


def test_actual_meter_path_work_inventory_and_snapshot_replay():
    meter = DifferentialMeterSpec(
        0.001,
        0.0006,
        meter_type="cone meter",
        specified_coefficient=0.82,
        reynolds_bounds=(1e-6, 1e8),
        installation_note="Synthetic calibrated cone; no real installation claim",
    )
    path = TransportPathSpec(differential_meter=meter, exchanger_area_m2=0)
    spec = FiniteTransportSpec(synthetic_transport_properties(), path=path)
    runtime = ChemWorldRuntime(composition=transport_world_spec(spec))
    runtime.reset(seed=30)
    _, _, start = unpack_transport(runtime._state)

    def execute(target, operation, **kwargs):
        result = target.execute({"operation": operation, **kwargs})
        assert result.info["transaction_status"] == "committed", result.info
        return result

    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    result = execute(runtime, "run_flow", duration_s=20)
    diagnostic = result.info["outcome_diagnostics"]["transport"]
    devices = diagnostic["hydraulic_devices"]
    actual = devices["differential_meter"]
    assert 0 < actual["permanent_pressure_loss_Pa"] < actual["differential_pressure_Pa"]
    assert diagnostic["pressure_rise_Pa"] == pytest.approx(
        devices["pipe_and_bed_pressure_drop_Pa"] + actual["permanent_pressure_loss_Pa"]
    )
    _, _, final = unpack_transport(runtime._state)
    assert sum(final.enthalpies_J) - sum(start.enthalpies_J) == pytest.approx(
        diagnostic["hydraulic_work_J"], abs=1e-6
    )
    for component in start.amounts_mol[0]:
        assert sum(n[component] for n in final.amounts_mol) == pytest.approx(
            sum(n[component] for n in start.amounts_mol), abs=1e-12
        )
    snapshot = SessionSnapshot.from_dict(json.loads(json.dumps(runtime.snapshot().to_dict())))
    resumed = ChemWorldRuntime.from_snapshot(snapshot)
    execute(runtime, "wait", duration_s=1)
    execute(resumed, "wait", duration_s=1)
    assert runtime._state.to_dict() == resumed._state.to_dict()


def test_meter_path_consumes_current_fluid_and_requires_pressure_recovery():
    meter = DifferentialMeterSpec(
        0.01,
        0.005,
        meter_type="long radius nozzle",
        reynolds_bounds=(1, 1e8),
        installation_note="Declared synthetic nozzle installation scope",
    )
    path = TransportPathSpec(pipe=PipeSpec(1, 0.01), differential_meter=meter)
    first = path.hydraulic_diagnostics(FluidState(1000, 0.001), 0.001)["differential_meter"]
    second = path.hydraulic_diagnostics(FluidState(800, 0.002), 0.001)["differential_meter"]
    for key in ("reynolds", "discharge_coefficient", "permanent_pressure_loss_Pa"):
        assert first[key] != second[key]
    with pytest.raises(ValueError, match="permanent-loss"):
        replace(path, differential_meter=replace(meter, meter_type="venturi nozzle"))
