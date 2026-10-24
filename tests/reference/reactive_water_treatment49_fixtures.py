"""Fixed synthetic neutral liquid fixtures; not wastewater material parameters."""

from dataclasses import replace

from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.reactive_water_treatment import (
    ReactiveWaterTreatmentSpec,
    reactive_water_treatment_inventory,
)
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.transport import PipeSpec, PumpSpec
from chemworld.physchem.transport_devices import LiquidTransportProperties

DATA = {
    "water": ("H2O", 1000.0, 75.0, 0.0),
    "A": ("CH2O", 1000.0, 80.0, 1000.0),
    "B": ("CH2O", 1000.0, 90.0, -1000.0),
    "C": ("C2H4O2", 1000.0, 160.0, -500.0),
}


def spec_case(kind="forward", temperature=298.15):
    packages = []
    for name, (formula, density, cp, _) in DATA.items():
        correlations = (
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
                        f"synthetic49B-{name}-{quantity}",
                        equation,
                        coefficients,
                        {"temperature": "K"},
                        unit,
                        property_id=quantity,
                        validity_ranges={"temperature": (280.0, 340.0)},
                        metadata={"phase": "liquid"},
                        source_note="Synthetic constant incompressible qualification properties",
                    )
                    for quantity, equation, coefficients, unit in correlations
                ),
            )
        )
    properties = LiquidTransportProperties(
        tuple(packages),
        caloric_references={
            k: CaloricReference(
                phase="liquid",
                energy_J_mol=v[3],
                source_note="Synthetic same-source reaction caloric datum",
            )
            for k, v in DATA.items()
        },
    )
    definitions = (
        [("forward", "A -> B", 0.04)] if kind != "nonlinear" else [("dimer", "2 A -> C", 0.8)]
    )
    if kind == "reversible":
        definitions.append(("reverse", "B -> A", 0.015))
    network = ReactionNetworkSpec(
        "synthetic-reactive-treatment-49B",
        tuple(SpeciesSpec(k, v[0], phase="liquid") for k, v in DATA.items()),
        tuple(
            ReactionSpec.from_equation(
                reaction_id=name,
                equation=eq,
                rate_law=RateLawSpec(name, "mass_action", {"k": rate}),
            )
            for name, eq, rate in definitions
        ),
    )
    return ReactiveWaterTreatmentSpec(
        properties,
        network,
        ThermalDeviceSpec(
            setpoint_K=temperature,
            minimum_setpoint_K=280.0,
            maximum_setpoint_K=340.0,
            maximum_power_W=10.0,
            maximum_cooling_power_W=10.0,
            mode="isothermal",
        ),
        PumpSpec(1e5, 10.0, 0.7),
        PipeSpec(0.1, 0.002),
        (0.1, 0.005, 0.1),
        0.005,
    )


def molar_volumes(spec):
    return {
        p.component.identifier: p.component.molecular_weight_g_mol / 1000 / 1000
        for p in spec.properties.packages
    }


def inventory_case(spec):
    volumes = molar_volumes(spec)

    def row(volume_L, a, b):
        n = {"A": volume_L * a, "B": volume_L * b, "C": 0.0}
        n["water"] = (volume_L / 1000 - sum(n[k] * volumes[k] for k in n)) / volumes["water"]
        return {k: n[k] for k in DATA}

    return reactive_water_treatment_inventory(
        spec, row(0.02, 0.1, 0), row(0.005, 0.02, 0.01), row(0.001, 0, 0)
    )


def fixed_cases():
    cases = [
        (f"{kind}-{temperature}-{flow}", spec_case(kind, temperature), flow / 6e7, 20.0, None)
        for kind in ("forward", "reversible", "nonlinear")
        for temperature in (288.15, 318.15)
        for flow in (0.0, 10.0)
    ]
    original = spec_case()
    cases.extend(
        [
            (
                "pressure-limit",
                replace(original, pump=PumpSpec(1.0, 10.0, 0.7)),
                10 / 6e7,
                200.0,
                None,
            ),
            (
                "shaft-limit",
                replace(original, pump=PumpSpec(1e5, 1e-7, 0.7)),
                10 / 6e7,
                200.0,
                None,
            ),
            ("pickup", original, 10 / 6e7, 200.0, None),
            ("product", replace(original, capacities_L=(0.1, 0.005, 0.002)), 10 / 6e7, 200.0, None),
            ("world-local", original, 10 / 6e7, 20.0, False),
            ("world-shared", original, 10 / 6e7, 20.0, True),
        ]
    )
    return cases
