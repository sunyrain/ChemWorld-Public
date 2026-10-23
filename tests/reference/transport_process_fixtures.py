"""Explicit synthetic companions for E5 source-to-process tests, not material fits."""

from math import exp, log

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentUncertainty, PropertyCorrelation


def synthetic_card(
    component,
    *,
    cp=100,
    rho=800,
    mu=0.001,
    k=0.2,
    omit=(),
    pressure=101325,
    variable=False,
    vapor_pressure=False,
):
    phase = component.default_phase
    declarations = [
        (
            "ideal_gas_heat_capacity" if phase == "gas" else "liquid_heat_capacity",
            "cp_polynomial",
            {"a": cp},
            "J/(mol*K)",
        ),
        (
            phase + "_viscosity",
            "andrade_viscosity",
            {"A": mu / exp(1) if variable else mu, "B": 300 if variable else 0},
            "Pa*s",
        ),
        (
            "thermal_conductivity",
            "linear_thermal_conductivity",
            {"k_ref": k, "T_ref": 300, "alpha": 1e-4 / k if variable else 0},
            "W/(m*K)",
        ),
    ]
    if phase == "liquid":
        declarations.append(
            (
                "liquid_density",
                "linear_liquid_density",
                {"rho_ref": rho, "T_ref": 300, "alpha": 0},
                "kg/m^3",
            )
        )
    if vapor_pressure:
        declarations.append(("vapor_pressure", "dippr101", {"A": log(20000), "B": 0}, "Pa"))
    source = InputSource(
        DataSourceProvenance(
            "E5-synthetic:" + component.identifier,
            0,
            "Fixed synthetic companion properties, not fitted material measurements",
            version="1",
            license="synthetic test declaration",
        ),
        "synthetic",
    )
    curves = tuple(
        PropertyCorrelation(
            component.identifier + ":synthetic:" + prop,
            eq,
            parameters,
            {"temperature": "K"},
            unit,
            prop,
            {"temperature": (280, 460)},
            source.provenance.citation,
            {"phase": phase, "source": source.provenance.to_dict()},
        )
        for prop, eq, parameters, unit in declarations
        if prop not in omit
    )
    return PropertyInputCard(
        ComponentPropertyPackage(component, curves),
        (source,),
        {
            c.correlation_id: ComponentUncertainty(
                c.correlation_id,
                unit=c.output_unit,
                source_id=source.provenance.source_id,
                coverage="synthetic",
                note="Defined model parameters; no material accuracy inferred",
            )
            for c in curves
        },
        CaloricReference(phase=phase, pressure_Pa=pressure),
        "Explicit zero-H datum at 298.15 K for the synthetic caloric companion",
    )
