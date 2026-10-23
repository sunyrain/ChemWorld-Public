"""Engineering inverses and finite native consumer behavior, without reference imports."""

import json
from dataclasses import replace
from math import log, pi

import pytest

from chemworld.physchem.flow_meters import (
    LiquidOrificeSpec,
    liquid_orifice_bore,
    liquid_orifice_mass_flow,
    liquid_orifice_pressure,
    liquid_orifice_pressure_endpoint,
)
from chemworld.physchem.fluids_devices import (
    LiquidControlValveSpec,
    liquid_valve_flow,
    liquid_valve_pressure_drop,
    size_liquid_control_valve,
    valve_reynolds,
)
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.transport import FluidState, PumpSpec
from chemworld.physchem.transport_devices import FiniteTransportSpec, TransportPathSpec
from chemworld.physchem.transport_devices import transport_rates as liquid_path_rates
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
from chemworld.runtime.transport_devices import unpack_transport
from chemworld.world.transport_world import synthetic_transport_properties, transport_world_spec


def valve(**kwargs):
    defaults = {"kv_m3_h": 0.002, "diameter_m": 0.001, "inlet_diameter_m": 0.001,
        "outlet_diameter_m": 0.001, "recovery_factor": 0.9, "style_modifier": 0.5,
        "source_note": "Synthetic declared trim; no physical valve calibration"}
    return LiquidControlValveSpec(**{**defaults, **kwargs})


def pure_transport_properties():
    original = synthetic_transport_properties()
    package = original.packages[0]
    vapor = PropertyCorrelation(
        correlation_id="synthetic-water-psat", property_id="vapor_pressure",
        equation_id="dippr101_vapor_pressure", coefficients={
            "A": log(2000), "B": 0, "C": 0, "D": 0, "E": 1},
        input_units={"temperature": "K"}, output_unit="Pa",
        validity_ranges={"temperature": (280.0, 340.0)},
        source_note="Synthetic constant vapor pressure for apparatus testing",
    )
    package = replace(package, component=replace(package.component,
        metadata={"critical_pressure_Pa": 22e6}), correlations=(*package.correlations, vapor))
    return replace(original, packages=(package,), caloric_references=None)


def native_spec(*, closed=False, meter=True, pump=None):
    orifice = LiquidOrificeSpec(0.001, 0.0005, method="calibrated",
        discharge_coefficient=0.65, reynolds_bounds=(0, 100000),
        installation_note="Synthetic constant-C calibration across stated Reynolds interval")
    path = TransportPathSpec(control_valve=valve(is_open=not closed),
        orifice_meter=orifice if meter else None, exchanger_area_m2=0)
    return FiniteTransportSpec(pure_transport_properties(), path=path,
        pump=pump or PumpSpec())


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.mark.parametrize("taps", ["corner", "flange", "D-D/2"])
def test_iso_orifice_inverses_and_pressure_recovery(taps):
    spec = LiquidOrificeSpec(0.07366, 0.05, taps=taps,
        installation_note="ISO concentric sharp edge, compliant straight runs and roughness")
    point = liquid_orifice_pressure(spec, mass_flow_kg_s=7.0,
        density_kg_m3=999.1, viscosity_Pa_s=0.0011)
    assert 0 < point.permanent_pressure_loss_Pa < point.differential_pressure_Pa
    inverse = liquid_orifice_mass_flow(spec,
        differential_pressure_Pa=point.differential_pressure_Pa,
        density_kg_m3=999.1, viscosity_Pa_s=0.0011)
    assert inverse.mass_flow_kg_s == pytest.approx(7, rel=2e-8)
    assert liquid_orifice_bore(replace(spec, bore_diameter_m=0.04), mass_flow_kg_s=7,
        differential_pressure_Pa=point.differential_pressure_Pa,
        density_kg_m3=999.1, viscosity_Pa_s=0.0011) == pytest.approx(0.05, rel=2e-8)
    upstream = liquid_orifice_pressure_endpoint(spec, known_pressure_Pa=100000,
        known_side="downstream", mass_flow_kg_s=7, density_kg_m3=999.1, viscosity_Pa_s=.0011)
    assert liquid_orifice_pressure_endpoint(spec, known_pressure_Pa=upstream,
        known_side="upstream", mass_flow_kg_s=7, density_kg_m3=999.1,
        viscosity_Pa_s=.0011) == pytest.approx(100000)


def test_meter_rejects_uncalibrated_geometry_low_re_and_high_drop():
    with pytest.raises(ValueError, match="geometry"):
        LiquidOrificeSpec(.001, .0005, installation_note="not industrial geometry")
    spec = LiquidOrificeSpec(.1, .05, installation_note="standard installation")
    for mass in (1e-5, 1000):
        with pytest.raises(ValueError):
            liquid_orifice_pressure(spec, mass_flow_kg_s=mass,
                density_kg_m3=1000, viscosity_Pa_s=.001)
    with pytest.raises(ValueError, match="sub-domain"):
        liquid_orifice_mass_flow(spec, differential_pressure_Pa=.001,
            density_kg_m3=1000, viscosity_Pa_s=.001)
    with pytest.raises(ValueError, match="Reynolds validity"):
        LiquidOrificeSpec(.001, .0005, method="calibrated", discharge_coefficient=.6,
            installation_note="missing validity")


@pytest.mark.parametrize("re_factor", [1 - 1e-6, 1, 1 + 1e-6])
def test_iso_reynolds_boundary_is_explicit(re_factor):
    spec = LiquidOrificeSpec(.1, .07, taps="flange", installation_note="standard installation")
    mass = spec.minimum_reynolds * re_factor * pi * .1 * .001 / 4
    if re_factor < 1:
        with pytest.raises(ValueError, match="Reynolds"):
            liquid_orifice_pressure(spec, mass_flow_kg_s=mass,
                density_kg_m3=1000, viscosity_Pa_s=.001)
    else:
        assert liquid_orifice_pressure(spec, mass_flow_kg_s=mass,
            density_kg_m3=1000, viscosity_Pa_s=.001).mass_flow_kg_s == mass


@pytest.mark.parametrize("kv", [.002, .02, .2])
def test_installed_valve_inverse_uses_reynolds_not_nominal_sizing(kv):
    spec, fluid = valve(kv_m3_h=kv), FluidState(1000, .001)
    point = liquid_valve_pressure_drop(spec, fluid, flow_m3_s=1e-7,
        downstream_pressure_Pa=101325, saturation_pressure_Pa=2000, critical_pressure_Pa=22e6)
    inverse = liquid_valve_flow(spec, fluid, upstream_pressure_Pa=point.upstream_pressure_Pa,
        downstream_pressure_Pa=101325, saturation_pressure_Pa=2000, critical_pressure_Pa=22e6)
    assert inverse.flow_m3_s == pytest.approx(1e-7, rel=2e-8)


def test_iec_engineering_choked_selection_is_separate_from_single_phase_path():
    spec = valve(kv_m3_h=200, diameter_m=.1, inlet_diameter_m=.1,
        outlet_diameter_m=.1, recovery_factor=.6, style_modifier=.98)
    fluid = FluidState(965.4, 3.1472e-4)
    result = size_liquid_control_valve(spec, fluid, flow_m3_s=.1,
        upstream_pressure_Pa=680e3, downstream_pressure_Pa=220e3,
        saturation_pressure_Pa=70100, critical_pressure_Pa=22120e3)
    assert result.choked
    assert result.kv_m3_h == pytest.approx(238.05817216710483, rel=1e-12)
    with pytest.raises(ValueError, match="choking"):
        liquid_valve_flow(replace(spec, kv_m3_h=result.kv_m3_h), fluid,
            upstream_pressure_Pa=680e3, downstream_pressure_Pa=220e3,
            saturation_pressure_Pa=70100, critical_pressure_Pa=22120e3)


def test_native_devices_change_workpoint_and_conserve_inventory_energy_snapshot():
    spec = native_spec()
    runtime = ChemWorldRuntime(composition=transport_world_spec(spec))
    runtime.reset(seed=30)
    _, _, start = unpack_transport(runtime._state)
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    outcome = execute(runtime, "run_flow", duration_s=20)
    diagnostic = outcome.info["outcome_diagnostics"]["transport"]
    _, _, final = unpack_transport(runtime._state)
    devices = diagnostic["hydraulic_devices"]
    assert devices["control_valve"]["pressure_drop_Pa"] > 0
    assert 0 < devices["orifice"]["permanent_pressure_loss_Pa"] < (
        devices["orifice"]["differential_pressure_Pa"])
    expected = (devices["pipe_and_bed_pressure_drop_Pa"]
        + devices["control_valve"]["pressure_drop_Pa"]
        + devices["orifice"]["permanent_pressure_loss_Pa"])
    assert diagnostic["pressure_rise_Pa"] == pytest.approx(expected)
    assert diagnostic["shaft_power_W"] <= spec.pump.maximum_shaft_power_W * (1 + 1e-8)
    assert sum(final.enthalpies_J) - sum(start.enthalpies_J) == pytest.approx(
        diagnostic["hydraulic_work_J"], abs=1e-6)
    assert sum(n["water"] for n in final.amounts_mol) == pytest.approx(
        sum(n["water"] for n in start.amounts_mol), abs=1e-12)
    snapshot = SessionSnapshot.from_dict(json.loads(json.dumps(runtime.snapshot().to_dict())))
    resumed = ChemWorldRuntime.from_snapshot(snapshot)
    execute(runtime, "wait", duration_s=5)
    execute(resumed, "wait", duration_s=5)
    assert runtime._state.to_dict() == resumed._state.to_dict()


def test_closed_valve_has_zero_transfer_and_work_but_shared_clock_advances():
    runtime = ChemWorldRuntime(composition=transport_world_spec(native_spec(closed=True)))
    runtime.reset(seed=30)
    _, _, before = unpack_transport(runtime._state)
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    outcome = execute(runtime, "run_flow", duration_s=20)
    _, _, after = unpack_transport(runtime._state)
    diagnostic = outcome.info["outcome_diagnostics"]["transport"]
    assert diagnostic["control_valve_closed"]
    assert diagnostic["actual_flow_m3_s"] == diagnostic["shaft_work_J"] == 0
    assert after.amounts_mol == before.amounts_mol
    assert after.time_s - before.time_s == pytest.approx(20)


def test_valve_rejects_mixture_or_missing_material_properties():
    with pytest.raises(ValueError, match="pure liquid"):
        FiniteTransportSpec(synthetic_transport_properties(),
            path=TransportPathSpec(control_valve=valve()))
    original = synthetic_transport_properties()
    properties = replace(original, packages=original.packages[:1], caloric_references=None)
    with pytest.raises(ValueError, match="critical_pressure"):
        FiniteTransportSpec(properties, path=TransportPathSpec(control_valve=valve()))


def test_device_spec_roundtrip_keeps_original_payload_compatible():
    spec = native_spec()
    payload = json.loads(json.dumps(spec.to_dict()))
    assert FiniteTransportSpec.from_dict(payload).to_dict() == spec.to_dict()
    original = FiniteTransportSpec(synthetic_transport_properties()).to_dict()
    original["path"].pop("control_valve")
    original["path"].pop("orifice_meter")
    assert FiniteTransportSpec.from_dict(original).path.control_valve is None


def test_actual_meter_loss_reduces_flow_and_current_viscosity_changes_valve_loss():
    spec = native_spec(pump=PumpSpec(20000, .005, .7))
    points = []
    for temperature in (288.15, 308.15):
        runtime = ChemWorldRuntime(composition=transport_world_spec(
            spec, feed_temperature_K=temperature))
        runtime.reset(seed=30)
        _, _, state = unpack_transport(runtime._state)
        _, with_meter = liquid_path_rates(spec, state, (temperature, 298.15), 20 / 6e7)
        without = replace(spec, path=replace(spec.path, orifice_meter=None))
        _, without_meter = liquid_path_rates(without, state, (temperature, 298.15), 20 / 6e7)
        assert with_meter["actual_flow_m3_s"] < without_meter["actual_flow_m3_s"]
        points.append(with_meter)
    assert points[0]["source_viscosity_Pa_s"] > points[1]["source_viscosity_Pa_s"]
    assert points[0]["actual_flow_m3_s"] < points[1]["actual_flow_m3_s"]


def test_shared_pump_root_can_reject_choked_trial_but_commit_admissible_workpoint():
    spec = native_spec()
    spec = replace(spec, path=replace(spec.path, control_valve=valve(kv_m3_h=.00001)))
    runtime = ChemWorldRuntime(composition=transport_world_spec(spec))
    runtime.reset(seed=30)
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    outcome = execute(runtime, "run_flow", duration_s=1)
    devices = outcome.info["outcome_diagnostics"]["transport"]["hydraulic_devices"]
    assert not devices["control_valve"]["choked"]
    assert devices["total_pressure_drop_Pa"] <= 20000 * (1 + 1e-8)


@pytest.mark.parametrize("kv", [.002, .02])
@pytest.mark.parametrize("reynolds", [1, 9.999, 10, 100])
def test_valve_inverse_keeps_creeping_and_transition_branches(kv, reynolds):
    spec, fluid = valve(kv_m3_h=kv), FluidState(1000, .001)
    flow = reynolds / valve_reynolds(spec, fluid, 1.0)
    point = liquid_valve_pressure_drop(spec, fluid, flow_m3_s=flow,
        downstream_pressure_Pa=101325, saturation_pressure_Pa=2000, critical_pressure_Pa=22e6)
    inverse = liquid_valve_flow(spec, fluid, upstream_pressure_Pa=point.upstream_pressure_Pa,
        downstream_pressure_Pa=101325, saturation_pressure_Pa=2000, critical_pressure_Pa=22e6)
    assert inverse.flow_m3_s == pytest.approx(flow, rel=2e-8, abs=1e-16)
