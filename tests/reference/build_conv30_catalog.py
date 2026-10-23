"""Assemble the final source-to-native navigation from delivered block evidence.

Metadata only: no scientific matrix reruns. Constants/classes/data aliases are
counted as exports, never as independent physical models or dynamic devices.
"""

import argparse
import importlib
import json
from pathlib import Path

from chemworld.physchem.engineering_numbers import ENGINEERING_GROUP_PARAMETERS

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "workstreams/world_foundation/reports/convergence/30"
ACTIVE = ROOT / "workstreams/world_foundation"


def read(name):
    path = REPORTS / name
    if not path.exists():
        path = ACTIVE / name
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build():
    original = read("NATIVE_CONV30_RESULTS.json")["public_catalog"]
    mappings = {
        module["module"]: {
            name: value["native"] if isinstance(value, dict) else value
            for name, value in module["mapped_native_subset"].items()
        }
        for module in original["modules"]
    }
    evidence = {name: [] for name in mappings}
    blocks = {
        "filters": ("FILTERS",),
        "friction": ("FRICTION", "ROUGHNESS"),
        "two_phase": ("PRESSURE", "REGIMES"),
        "two_phase_voidage": ("HOLDUP",),
        "packed_bed": ("PACKED_BED",),
        "saltation": ("SALTATION",),
        "piping": ("PIPING",),
        "packed_tower": ("TOWERS",),
    }
    for module, suffixes in blocks.items():
        for suffix in suffixes:
            name = f"NATIVE_CONV30_{suffix}_RESULTS.json"
            result = read(name)
            mappings[module].update(result.get("source_mapping", result.get("source_exports", {})))
            evidence[module].append(name)
    mix = read("NATIVE_CONV30_SEPARATION_MIXING_RESULTS.json")["source_exports"]
    for module in ("mixing", "separator"):
        mappings[module].update(mix[module])
        evidence[module].append("NATIVE_CONV30_SEPARATION_MIXING_RESULTS.json")
    mappings["atmosphere"].update(
        {
            "ATMOSPHERE_1976": (
                "standard_atmosphere / StandardAtmosphereState / standard_pressure_column"
            ),
            "airmass": "air_column_mass",
            "ATMOSPHERE_NRLMSISE00": "MSISAtmosphere.evaluate",
            "earthsun_distance": "SolarEphemeris.earth_sun_distance_m",
            "solar_position": "SolarEphemeris.position",
            "sunrise_sunset": "solar_events",
            "solar_irradiation": "clear_sky_irradiance",
        }
    )
    evidence["atmosphere"] = [
        f"NATIVE_CONV30_{s}_RESULTS.json" for s in ("STANDARD_ATMOSPHERE", "MSIS", "SOLAR")
    ]
    for name in ENGINEERING_GROUP_PARAMETERS:
        mappings["core"][name] = f"engineering_group('{name}', ...)"
    mappings["core"].update(
        {
            "gravity": "local_gravity",
            "c_ideal_gas": "ideal_gas_sound_speed",
            "nu_mu_converter": "convert_viscosity",
            "thermal_diffusivity": "thermal_diffusivity_si",
        }
    )
    for name in next(m["public_exports"] for m in original["modules"] if m["module"] == "core"):
        if len(name) == 3 and name[1] == "2":
            mappings["core"][name] = "foundation.units.convert_value with K/degC/degF/degR"
        elif name not in mappings["core"]:
            mappings["core"][name] = f"engineering_resistance_conversion('{name}', ...)"
    evidence["core"] = ["NATIVE_CONV30_CORE_RESULTS.json"]
    mappings["drag"].update(
        {
            "drag_sphere": "particle_drag_coefficient",
            "drag_sphere_methods": "available_particle_drag_methods",
            "v_terminal": "particle_terminal_velocity",
            "time_v_terminal_Stokes": "time_to_stokes_terminal",
            "integrate_drag_sphere": "advance_particle_motion",
        }
    )
    for name in next(m["public_exports"] for m in original["modules"] if m["module"] == "drag"):
        mappings["drag"].setdefault(name, f"particle_drag_coefficient(method='{name}')")
    evidence["drag"] = ["NATIVE_CONV30_DRAG_RESULTS.json"]
    mappings["jet_pump"].update(
        {
            "liquid_jet_pump": "solve_liquid_jet_pump",
            "liquid_jet_pump_ancillary": "solve_jet_pump_inlet",
            **{
                f"vacuum_air_leakage_{name}": f"vacuum_air_leakage(method='{name}')"
                for name in ("Coker_Worthington", "HEI2633", "Ryans_Croll", "Seider")
            },
        }
    )
    evidence["jet_pump"] = ["NATIVE_CONV30_JET_RESULTS.json"]
    mappings["open_flow"].update(
        {
            "V_Manning": "manning_velocity",
            "V_Chezy": "chezy_velocity",
            "n_Manning_to_C_Chezy": "manning_to_chezy",
            "C_Chezy_to_n_Manning": "chezy_to_manning",
        }
    )
    for name in next(
        m["public_exports"] for m in original["modules"] if m["module"] == "open_flow"
    ):
        mappings["open_flow"].setdefault(
            name,
            "manning_roughness_table / select_manning_roughness (caller source)"
            if name.startswith("n_")
            else f"weir_flow (source relation {name})",
        )
    evidence["open_flow"] = ["NATIVE_CONV30_OPEN_FLOW_RESULTS.json"]
    mappings["pump"].update(
        {
            "CSA_motor_efficiency": "MotorEfficiencyCatalog",
            "Corripio_motor_efficiency": "corripio_motor_efficiency",
            "Corripio_pump_efficiency": "corripio_pump_efficiency",
            "CountryPower": "ElectricalSupplyRecord",
            "VFD_efficiency": "DriveEfficiencyMap",
            "current_ideal": "ideal_motor_current",
            "motor_efficiency_underloaded": "MotorPartLoadCatalog",
            "motor_round_size": "MotorSizeCatalog",
            "specific_diameter": "pump_specific_diameter",
            "specific_speed": "pump_specific_speed",
            "speed_synchronous": "motor_synchronous_speed",
        }
    )
    for name in next(m["public_exports"] for m in original["modules"] if m["module"] == "pump"):
        mappings["pump"].setdefault(
            name,
            "MotorSizeCatalog / MotorEfficiencyCatalog (caller source)"
            if name.startswith("nema_")
            else "ElectricalSupplyCatalog (caller source)",
        )
    evidence["pump"] = ["NATIVE_CONV30_PUMP_RESULTS.json"]
    mappings["particle_size_distribution"].update(
        {
            "PSDCustom": "CustomDiameterDistribution",
            "PSDInterpolated": "InterpolatedDiameterDistribution",
            "ParticleSizeDistribution": "BinnedDiameterDistribution",
            "ParticleSizeDistributionContinuous": "ContinuousDiameterDistribution",
            "psd_spacing": "diameter_spacing",
            "Sieve": "SieveSpecification",
        }
    )
    for name in next(
        m["public_exports"]
        for m in original["modules"]
        if m["module"] == "particle_size_distribution"
    ):
        mappings["particle_size_distribution"].setdefault(
            name,
            "SieveCatalog (caller data and sieve series)"
            if name.startswith(("ISO_", "ASTM_"))
            else f"AnalyticDiameterDistribution (source family/function {name}; pdf/cdf/moment)",
        )
    evidence["particle_size_distribution"] = ["NATIVE_CONV30_PSD_RESULTS.json"]
    fittings = mappings["fittings"]
    fittings.update(
        {
            "CRANE_VALVES": "CRANE_VALVE_TYPES",
            "DARBY_VALVES": "FittingCoefficientCatalog(table, model='Darby')",
            "HOOPER_VALVES": "FittingCoefficientCatalog(table, model='Hooper')",
            "Crane_loss_coefficient": "crane_valve_coefficient",
            "Cv_to_K": "resistance_from_valve_coefficient",
            "Kv_to_K": "resistance_from_valve_coefficient",
            "K_to_Cv": "valve_coefficient_from_resistance",
            "K_to_Kv": "valve_coefficient_from_resistance",
            "Cv_to_Kv": "convert_valve_coefficient",
            "Kv_to_Cv": "convert_valve_coefficient",
            "Darby": "FittingCoefficientCatalog",
            "Darby3K": "darby_resistance",
            "Hooper": "FittingCoefficientCatalog",
            "Hooper2K": "hooper_resistance",
            "change_K_basis": "change_loss_velocity_basis",
            "v_lift_valve_Crane": "minimum_valve_lift_velocity",
            "diffuser_conical_staged": "staged_diffuser_loss",
            "helix": "coil_loss_coefficient(helix)",
            "spiral": "coil_loss_coefficient(spiral)",
        }
    )
    for name in next(m["public_exports"] for m in original["modules"] if m["module"] == "fittings"):
        if name in fittings:
            continue
        if name.startswith(("K_branch", "K_run")):
            target = "junction_loss_coefficients"
        elif name.startswith("K_"):
            target = "crane_valve_coefficient"
        elif name.startswith("bend_"):
            target = "bend_loss_coefficient"
        elif name.startswith(("entrance_", "exit_")):
            target = "entrance_loss_coefficient"
        else:
            target = "transition_loss_coefficient"
        fittings[name] = f"{target} (source relation {name})"
    evidence["fittings"] = [
        "NATIVE_CONV30_FITTINGS_VALVES_RESULTS.json",
        "NATIVE_CONV30_LOCAL_FITTINGS_RESULTS.json",
    ]
    meters = mappings["flow_meter"]
    meters.update(
        {
            "C_Miller_1996": "miller_orifice_coefficient",
            "all_meters": "METER_TYPES / METER_ALIASES",
            "C_Reader_Harris_Gallagher_wet_venturi_tube": "wet_venturi_correction",
            "dP_Reader_Harris_Gallagher_wet_venturi_tube": "wet_venturi_correction",
            "differential_pressure_meter_C_epsilon": (
                "meter_discharge_coefficient / meter_expansibility"
            ),
            "differential_pressure_meter_beta": "meter_beta",
            "differential_pressure_meter_dP": "DifferentialMeterSpec.permanent_loss",
            "differential_pressure_meter_solver": "solve_differential_meter",
            "discharge_coefficient_to_K": "meter_loss_coefficient",
            "K_to_discharge_coefficient": "meter_coefficient_from_loss",
            "flow_coefficient": "meter_flow_coefficient",
            "flow_meter_discharge": "meter_discharge",
            "velocity_of_approach_factor": "meter_flow_coefficient (approach factor)",
        }
    )
    for name in next(
        m["public_exports"] for m in original["modules"] if m["module"] == "flow_meter"
    ):
        if name in meters:
            continue
        target = (
            "METER_TYPES / METER_ALIASES / explicit installation strings"
            if name.isupper()
            else "meter_expansibility"
            if "expansibility" in name
            else "meter_beta"
            if name.startswith("diameter_ratio")
            else "DifferentialMeterSpec.permanent_loss"
            if name.startswith("dP_")
            else "meter_discharge_coefficient"
        )
        meters[name] = f"{target} (source relation {name})"
    evidence["flow_meter"] = ["NATIVE_CONV30_METERS_RESULTS.json"]
    mappings["geometry"].update(
        {
            "HelicalCoil": "helical_coil_geometry",
            "PlateExchanger": "plate_exchanger_geometry",
            "RectangularFinExchanger": "fin_exchanger_geometry",
            "RectangularOffsetStripFinExchanger": "fin_exchanger_geometry (offset_strip)",
            "AirCooledExchanger": "air_cooled_geometry",
            "HyperbolicCoolingTower": "CoolingTowerGeometry",
            "pitch_angle_solver": "solve_tube_pitch",
            "plate_enlargement_factor": "corrugated_plate_enlargement",
            "A_partial_circle": "circular_segment_area",
            "circle_segment_h_from_A": "circular_segment_height",
            "SA_partial_sphere": "partial_sphere_geometry",
            "V_partial_sphere": "partial_sphere_geometry",
            **dict.fromkeys(
                (
                    "A_cylinder",
                    "A_hollow_cylinder",
                    "A_multiple_hole_cylinder",
                    "V_cylinder",
                    "V_hollow_cylinder",
                    "V_multiple_hole_cylinder",
                ),
                "cylinder_geometry",
            ),
            **{
                name: f"shape_factor('{name}', ...)"
                for name in ("aspect_ratio", "circularity", "sphericity")
            },
        }
    )
    mappings["jet_pump"]["vacuum_air_leakage_Ryans_Croll"] = (
        "vacuum_air_leakage(method='Ryans-Croll')"
    )
    mappings["jet_pump"]["vacuum_air_leakage_Coker_Worthington"] = (
        "vacuum_air_leakage(method='Coker-Worthington')"
    )
    mappings["geometry"].update(
        read("NATIVE_CONV30_TANK_GEOMETRY_RESULTS.json")["source_catalog"]["mapping"]
    )
    evidence["geometry"] = [
        "NATIVE_CONV30_EQUIPMENT_GEOMETRY_RESULTS.json",
        "NATIVE_CONV30_TANK_GEOMETRY_RESULTS.json",
    ]
    for module in ("compressible", "control_valve", "safety_valve"):
        evidence[module] = ["NATIVE_CONV30_RESULTS.json"]
    modules = []
    for previous in original["modules"]:
        module = previous["module"]
        exports = previous["public_exports"]
        modules.append(
            {
                "module": module,
                "export_count": len(exports),
                "evidence": [
                    str(
                        (
                            REPORTS / name if (REPORTS / name).exists() else ACTIVE / name
                        ).relative_to(ROOT)
                    ).replace("\\", "/")
                    for name in evidence[module]
                ],
                "mapping": {name: mappings[module].get(name) for name in exports},
                "initial_mapping_scope": {
                    name: {k: v for k, v in value.items() if k != "native"}
                    for name, value in previous["mapped_native_subset"].items()
                    if isinstance(value, dict) and len(value) > 1
                },
                "unmapped": [name for name in exports if name not in mappings[module]],
            }
        )
    reference = importlib.import_module("fluids")
    assert reference.__version__ == "1.3.1", reference.__version__
    for module in modules:
        source = importlib.import_module("fluids." + module["module"])
        assert set(source.__all__) == set(module["mapping"]), module["module"]
        assert all((ROOT / path).is_file() for path in module["evidence"])
    return {
        "reference": original["reference"],
        "module_count": len(modules),
        "export_count": sum(m["export_count"] for m in modules),
        "modules": modules,
        "unmapped_count": sum(len(m["unmapped"]) for m in modules),
        "installed_reference_inventory_verified": True,
        "note": (
            "Export coverage includes constants/classes/aliases. Data tables require explicit "
            "caller sources; model boundaries and all scientific differences "
            "remain in linked evidence."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print({k: v for k, v in result.items() if k not in ("modules", "note")})
    for module in result["modules"]:
        if module["unmapped"]:
            print(module["module"], module["unmapped"])
