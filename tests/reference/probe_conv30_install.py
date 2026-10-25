"""Copied outside checkout; exercise installed engineering with explicit input data."""

import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import chemworld.physchem as public
from chemworld.physchem import (
    AnalyticDiameterDistribution,
    DifferentialMeterSpec,
    FluidState,
    IsothermalGasState,
    MSISAtmosphere,
    MSISParameters,
    PackedBedSpec,
    PipeSpec,
    PumpDriveSpec,
    SolarEphemeris,
    SolarParameters,
    TankGeometry,
    TankHead,
    clear_sky_irradiance,
    helical_coil_geometry,
    isothermal_gas_pipe,
    packed_bed_pressure,
    pipe_pressure_drop,
    solar_events,
    solve_differential_meter,
    solve_isothermal_gas_pipe,
    staged_diffuser_loss,
    standard_atmosphere,
)
from chemworld.physchem.transport import PumpSpec


def main():
    checks = {}
    # Exercise every lazy export in an actual filesystem installation. This
    # catches misrouted symbols such as the cooling-tower geometry facade.
    exports = [name for names in public._MODULE_EXPORTS.values() for name in names]
    checks["public_export_resolution"] = all(getattr(public, name) is not None for name in exports)
    air = standard_atmosphere(2000)
    pipe = PipeSpec(1, 0.01, fittings_loss_coefficient=2)
    pressure = pipe_pressure_drop(pipe, air.fluid_state(), volumetric_flow_m3_s=1e-4)
    checks["standard_atmosphere_actual_pipe"] = (
        pressure.pressure_drop_total_Pa > 0 and air.pressure_Pa < 101325
    )
    params = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    atmosphere = MSISAtmosphere(MSISParameters.from_dict(params["msis"])).evaluate(
        100000, latitude_degrees=35, longitude_degrees=110, day_of_year=172, seconds_utc=43200
    )
    checks["explicit_msis_parameter_import"] = (
        atmosphere.temperature_K > 100 and atmosphere.density_kg_m3 > 0
    )
    checks["msis_species_closure"] = math.isclose(sum(atmosphere.mole_fractions), 1, abs_tol=1e-12)
    ephemeris = SolarEphemeris(SolarParameters.from_dict(params["solar"]))
    moment = datetime(2024, 6, 21, 12, tzinfo=UTC)
    position = ephemeris.position(moment, 35, 0)
    irradiation = clear_sky_irradiance(
        position,
        moment,
        pressure_Pa=101325,
        elevation_m=0,
        linke_turbidity=3,
        surface_tilt_degrees=30,
        surface_azimuth_degrees=180,
    )
    checks["explicit_solar_parameters_and_power"] = irradiation.absorbed_power_w(2, 0.7) > 0
    events = solar_events(ephemeris, moment, 35, 0)
    checks["actual_civil_solar_events"] = events.sunrise < events.transit < events.sunset
    gas = IsothermalGasState(300, 287.05, 0.98)
    gas_geometry = {"diameter_m": 0.05, "length_m": 100, "darcy_friction_factor": 0.02}
    gas_flow = isothermal_gas_pipe(
        gas, upstream_pressure_Pa=1e6, downstream_pressure_Pa=8e5, **gas_geometry
    )
    inverse = solve_isothermal_gas_pipe(
        gas, upstream_pressure_Pa=1e6, mass_flow_kg_s=gas_flow.mass_flow_kg_s, **gas_geometry
    )
    checks["gas_engineering_inverse"] = math.isclose(
        inverse.downstream_pressure_Pa, 8e5, rel_tol=1e-9
    )
    fluid = FluidState(1000, 0.001)
    meter = DifferentialMeterSpec.from_dict(
        json.loads(json.dumps(DifferentialMeterSpec(0.1, 0.05).to_dict()))
    )
    flow = solve_differential_meter(
        meter, fluid, upstream_pressure_Pa=2e5, downstream_pressure_Pa=1.8e5
    )
    checks["meter_serialization_and_pressure"] = (
        0 < flow.permanent_loss_Pa < 2e4 and flow.mass_flow_kg_s > 0
    )
    bed = PackedBedSpec.from_dict(
        json.loads(json.dumps(PackedBedSpec("Ergun", 0.2, 0.001, 0.4).to_dict()))
    )
    checks["packed_bed_current_fluid"] = (
        packed_bed_pressure(bed, fluid, superficial_velocity_m_s=0.02).pressure_drop_Pa > 0
    )
    drive = PumpDriveSpec(PumpSpec(1e5, 0.1, 0.7), 0.002, 0.9)
    drive_report = drive.operating_point(
        1e-5,
        lambda q: (
            pipe_pressure_drop(
                PipeSpec(1, 0.005), fluid, volumetric_flow_m3_s=q
            ).pressure_drop_total_Pa
        ),
    )
    checks["pump_drive_energy"] = math.isclose(
        drive_report["electrical_power_W"],
        drive_report["total_external_loss_W"] + drive_report["hydraulic_power_W"],
        rel_tol=1e-12,
    )
    distribution = AnalyticDiameterDistribution("lognormal", 1e-4, 0.8, lower_m=2e-5, upper_m=2e-4)
    checks["weighted_psd_inverse"] = math.isclose(
        distribution.cdf(distribution.quantile(0.6)), 0.6, rel_tol=1e-9
    )
    loss = staged_diffuser_loss((0.05, 0.075, 0.1), (0.2, 0.3), friction_factor=0.02)
    raw = staged_diffuser_loss(
        (0.05, 0.075, 0.1), (0.2, 0.3), friction_factor=0.02, common_velocity_basis=False
    )
    checks["fitting_velocity_basis"] = 0 < loss < raw
    coil = helical_coil_geometry(
        tube_diameter_m=0.01, coil_diameter_m=0.2, pitch_m=0.02, turns=2, inner_diameter_m=0.008
    )
    checks["equipment_pipe_geometry"] = coil.tube_pipe().length_m > 2 * math.pi * 0.2
    tank = TankGeometry(2, 3, horizontal=False, head_a=TankHead("conical", 0.25))
    tank = TankGeometry.from_dict(json.loads(json.dumps(tank.to_dict())))
    liquid_volume = 7 * math.pi / 6
    liquid = tank.liquid_state(liquid_volume, density_kg_m3=fluid.density_kg_m3)
    checks["tank_geometry_inventory_and_level"] = (
        math.isclose(tank.total_volume_m3, 19 * math.pi / 6, rel_tol=1e-12)
        and math.isclose(liquid["height_m"], 1.5, rel_tol=1e-10)
        and math.isclose(liquid["bottom_gauge_pressure_Pa"], 14709.975, rel_tol=1e-10)
        and math.isclose(
            liquid["headspace_volume_m3"] + liquid_volume, tank.total_volume_m3, rel_tol=1e-12
        )
    )
    checks["no_reference_backend"] = not any(
        n in sys.modules for n in ("fluids", "thermo", "CoolProp", "pvlib")
    )
    summary = {
        "installed_engineering_checks": len(checks),
        "passed": sum(checks.values()),
        "public_symbols_resolved": len(exports),
        "failures": [name for name, value in checks.items() if not value],
    }
    print(json.dumps(summary))
    assert all(checks.values()), summary


if __name__ == "__main__":
    main()
