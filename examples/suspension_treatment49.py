"""Installed three-tank jar/clarifier example with explicitly synthetic properties."""

from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.suspension_treatment_specs import (
    ClarifierSplitSpec,
    CoagFlocMixingSpec,
    JarTestEndpointSpec,
    SuspensionMaterialSpec,
    SuspensionParticleSpec,
    SuspensionTankInventory,
    SuspensionTreatmentSpec,
)
from chemworld.physchem.transport import PipeSpec, PumpSpec
from chemworld.physchem.transport_devices import LiquidTransportProperties
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.suspension_treatment_world import suspension_treatment_world_spec


def example_spec(*, temperature=298.15, final_ntu=10.0, floc_capture=0.95):
    packages = []
    for name, formula, density, cp in (
        ("water", "H2O", 1000.0, 75.0),
        ("glucose", "C6H12O6", 1400.0, 200.0),
    ):
        rows = (
            ("liquid_heat_capacity", "cp_polynomial", {"a": cp}, "J/(mol*K)"),
            (
                "liquid_density",
                "linear_liquid_density",
                {"rho_ref": density, "T_ref": 298.15, "alpha": 0.0},
                "kg/m^3",
            ),
            ("liquid_viscosity", "andrade_viscosity", {"A": 0.001, "B": 0.0}, "Pa*s"),
            (
                "thermal_conductivity",
                "linear_thermal_conductivity",
                {"k_ref": 0.5, "T_ref": 298.15, "slope": 0.0},
                "W/(m*K)",
            ),
        )
        packages.append(
            ComponentPropertyPackage(
                ComponentSpec(name, formula),
                tuple(
                    PropertyCorrelation(
                        f"synthetic49F-{name}-{quantity}",
                        equation,
                        coefficients,
                        {"temperature": "K"},
                        unit,
                        property_id=quantity,
                        validity_ranges={"temperature": (280.0, 340.0)},
                        metadata={"phase": "liquid"},
                        source_note="Synthetic constants; not measured slurry properties",
                    )
                    for quantity, equation, coefficients, unit in rows
                ),
            )
        )
    material = SuspensionMaterialSpec(
        LiquidTransportProperties(
            tuple(packages),
            caloric_references={
                name: CaloricReference(
                    phase="liquid", energy_J_mol=h, source_note="Synthetic caloric datum"
                )
                for name, h in (("water", -1000), ("glucose", 500))
            },
        ),
        SuspensionParticleSpec(
            "silica",
            "SiO2",
            2.7e-5,
            45.0,
            -5000,
            298.15,
            "Synthetic inert silica constants; no measured accuracy claim",
        ),
        "water",
        0.0012,
        0.05,
        "Synthetic uniform Newtonian slurry below declared solids fraction",
    )
    return SuspensionTreatmentSpec(
        material,
        JarTestEndpointSpec(100, final_ntu, 2.0, 5.0, "Synthetic prescribed jar measurement"),
        CoagFlocMixingSpec(20, 2, 300, 300, 0.1, 0.02, 0.2, 1.2, 0.7, 2, 4),
        ClarifierSplitSpec(
            0.1, floc_capture, 0.08, "Synthetic measured component capture fractions"
        ),
        ThermalDeviceSpec(
            setpoint_K=temperature,
            minimum_setpoint_K=280,
            maximum_setpoint_K=340,
            maximum_power_W=10,
            maximum_cooling_power_W=10,
            mode="isothermal",
        ),
        PumpSpec(1e5, 10, 0.7),
        PipeSpec(0.1, 0.002),
        (0.1, 0.1, 0.1),
        0.005,
    )


def example_feed(spec, *, initial_floc_mol=0.0):
    volume = 0.02 / 1000
    dispersed = (
        spec.jar_test.initial_tss_kg_m3 * volume / spec.material.particle.molecular_weight_kg_mol
    )
    glucose = 2e-5
    water = (
        volume
        - (dispersed + initial_floc_mol) * spec.material.particle.molar_volume_m3_mol
        - glucose * spec.molar_volumes["liquid:glucose"]
    ) / spec.molar_volumes["liquid:water"]
    return SuspensionTankInventory(
        {"water": water, "glucose": glucose}, dispersed, initial_floc_mol
    )


def main():
    import json

    spec = example_spec()
    runtime = ChemWorldRuntime(
        composition=suspension_treatment_world_spec(spec, example_feed(spec))
    )
    runtime.reset()
    for action in (
        {"operation": "set_flow_rate", "flow_rate_mL_min": 10},
        {"operation": "run_flow", "duration_s": 20},
    ):
        result = runtime.execute(action)
        if result.info["transaction_status"] != "committed":
            raise RuntimeError(result.info)
    print(json.dumps(result.info, indent=2))


if __name__ == "__main__":
    main()
