"""Synthetic GAC design example; coefficients are not a material calibration."""

from chemworld.physchem.gac_design import (
    GACBedSpec,
    GACDesignSpec,
    GACEmpiricalParameters,
    GACFeedConditions,
    GACTransportSpec,
)


def gac_design_example(
    *,
    target_species: str = "A",
    calculated_film: bool = False,
    calculated_surface: bool = False,
) -> tuple[GACDesignSpec, GACFeedConditions]:
    """Call evaluate_gac_design(*gac_design_example()) for a JSON-safe report.

    All parameters and declared domains are authored synthetic demonstration
    choices for the public CPHSDM equations, not experimental material data.
    """
    source = "CC0 authored synthetic CPHSDM demonstration, not calibrated material data"
    parameters = GACEmpiricalParameters(
        2.0,
        0.65,
        (0.01, 0.2),
        (0.5, 0.4, 0.6, 0.01, 0.3),
        source,
        "synthetic",
        (1e-4, 10.0),
        (280.0, 340.0),
        (1e4, 2e6),
        (1e-6, 1e6),
    )
    transport = GACTransportSpec(
        source,
        None if calculated_film else 5e-5,
        None if calculated_surface else 4e-13,
        1e-9 if calculated_film or calculated_surface else None,
        1.0 if calculated_film else None,
        0.4 if calculated_surface else None,
        2.0 if calculated_surface else None,
        1.0 if calculated_surface else None,
    )
    return (
        GACDesignSpec(
            target_species,
            parameters,
            GACBedSpec(750.0, 0.001, 0.4, 600.0, 0.01, source),
            transport,
        ),
        GACFeedConditions(0.001, 0.02, 298.15, 101325.0, 1000.0, 0.001, source),
    )
