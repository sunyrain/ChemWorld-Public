"""Unified machine-readable capability inventory for native mechanism imports."""

from __future__ import annotations

from typing import Any


def cantera_mechanism_capabilities() -> dict[str, Any]:
    """Aggregate gas, surface and electrochemical importer boundaries."""

    from chemworld.physchem.chemkin_import import chemkin_import_capabilities
    from chemworld.physchem.electrochemical_mechanism_import import (
        electrochemical_mechanism_import_capabilities,
    )
    from chemworld.physchem.mechanism_import import mechanism_import_capabilities
    from chemworld.physchem.surface_mechanism_import import surface_import_capabilities

    gas = mechanism_import_capabilities()
    surface = surface_import_capabilities()
    electrochemical = electrochemical_mechanism_import_capabilities()
    chemkin = chemkin_import_capabilities()
    unsupported = sorted(
        {
            *gas["unsupported"],
            *surface["unsupported"],
            *electrochemical["unsupported"],
            *chemkin["unsupported"],
        }
    )
    return {
        "schema_version": "chemworld-cantera-capability-inventory-v1",
        "source_specification": (
            "Cantera 3.2 YAML/JSON plus explicit ChemWorld electrochemical extension"
        ),
        "gas": gas,
        "surface": surface,
        "electrochemical": electrochemical,
        "chemkin": chemkin,
        "supported_phase_models": sorted(
            {
                *gas["phase_models"],
                *surface["phases"],
                *electrochemical["phase_models"],
                *chemkin["phase_models"],
            }
        ),
        "unsupported": unsupported,
        "qualification_boundaries": {
            "point_rates": "each imported rate family is validated at declared points",
            "thermo_curves": "source thermo curves are compared independently",
            "reactor_trajectories": "trajectory qualification remains a separate runtime block",
            "source_completeness": "all selected declarations are converted or import rejects",
        },
    }
