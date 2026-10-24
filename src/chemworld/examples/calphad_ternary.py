"""Sourced synthetic ternary CEF section and original finite thermal world."""

from chemworld.data.scientific_inputs import InputSource
from chemworld.data.tdb import TdbDatabase
from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.calphad_caloric_equilibrium import CalphadCaloricSpec
from chemworld.physchem.calphad_equilibrium import (
    CalphadEquilibriumPolicy,
    CalphadEquilibriumSpec,
    calphad_tp_equilibrium,
)
from chemworld.physchem.calphad_ternary_diagram import CalphadTernaryPath, CalphadTernarySpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.runtime.calphad_phase_inventory import CalphadPhaseTarget
from chemworld.runtime.calphad_thermal import CalphadThermalSpec
from chemworld.world.calphad_thermal_world import build_calphad_thermal_world
from chemworld.world.composition import WorldCompositionSpec

COMPONENTS = ("FE", "NI", "CR")
_HEADER = """
ELEMENT FE BCC_A2 55.845 0 0 !
ELEMENT NI FCC_A1 58.69 0 0 !
ELEMENT CR BCC_A2 51.996 0 0 !
TYPE_DEFINITION % SEQ * !
FUNCTION GCP 400 30*(T-600-T*LN(T/600)); 1100 N !
"""


def ternary_tdb(family: str = "preferred") -> tuple[str, tuple[str, ...]]:
    """Original CC0 teaching models, not a calibrated Fe-Ni-Cr database."""
    if family in {"preferred", "excluded"}:
        energies = {
            "FE_PREF": (0, 12000, 12000),
            "NI_PREF": (12000, 0, 12000),
            "CR_PREF": (12000, 12000, 0),
            "HIGH": (20000, 20000, 20000),
        }
    elif family in {"ideal", "regular"}:
        energies = {"ALPHA": (0, 0, 0)}
    else:
        raise ValueError("unknown synthetic ternary family")
    text = _HEADER
    for phase, values in energies.items():
        text += f"\nPHASE {phase} % 1 1 !\nCONSTITUENT {phase} :FE,NI,CR: !\n"
        for element, value in zip(COMPONENTS, values, strict=True):
            text += f"PARAMETER G({phase},{element};0) 400 GCP#+{value}; 1100 N !\n"
    if family == "regular":
        text += "PARAMETER L(ALPHA,FE,NI;0) 400 18000; 1100 N !\n"
    allowed = tuple(p for p in energies if family != "excluded" or p != "CR_PREF")
    return text, allowed


def ternary_equilibrium_spec(family: str = "preferred") -> CalphadEquilibriumSpec:
    text, allowed = ternary_tdb(family)
    source = InputSource(
        DataSourceProvenance(
            "calphad42-ternary", 0, "Original synthetic ternary teaching models", "1", "CC0"
        ),
        "synthetic",
    )
    database = TdbDatabase.from_string(text, source=source)
    return CalphadEquilibriumSpec(
        tuple(CefPhaseModel(database, p) for p in allowed),
        CalphadEquilibriumPolicy(grid_intervals=8),
    )


def ternary_example_spec(family: str = "preferred", grid_intervals: int = 3) -> CalphadTernarySpec:
    return CalphadTernarySpec(
        ternary_equilibrium_spec(family),
        COMPONENTS,
        600.0 if family == "regular" else 700.0,
        101325.0,
        grid_intervals,
        ((0.30, 0.30, 0.40),),
        (CalphadTernaryPath("FE-to-NI", ((1.0, 0.0, 0.0), (0.30, 0.30, 0.40), (0.0, 1.0, 0.0))),),
    )


def build_calphad_ternary_example() -> WorldCompositionSpec:
    # TP sections cover multiple phases. This runtime example qualifies the
    # original thermal path with one ideal phase, not three-phase PH performance.
    equilibrium = ternary_equilibrium_spec("ideal")
    targets = tuple(
        CalphadPhaseTarget(
            p.phase_name.lower(), p.phase_name, 0.01, selected=p.phase_name == "ALPHA"
        )
        for p in equilibrium.models
    )
    spec = CalphadThermalSpec(
        CalphadCaloricSpec(equilibrium, 101325.0, (500.0, 1000.0), (600.0, 800.0), 3), targets
    )
    feed = {"FE": 0.0003, "NI": 0.0003, "CR": 0.0004}
    initial = calphad_tp_equilibrium(equilibrium, 700.0, 101325.0, feed)
    if not initial.success or initial.total_enthalpy_J is None:
        raise ValueError("initial ternary TP failed: " + initial.failure_reason)
    enthalpy = initial.total_enthalpy_J + spec.support_heat_capacity_J_K * (
        700.0 - spec.reference_temperature_K
    )
    return build_calphad_thermal_world(
        spec,
        feed,
        enthalpy_J=enthalpy,
        device=ThermalDeviceSpec(
            conductance_W_K=1.0,
            maximum_power_W=0.05,
            maximum_cooling_power_W=0.05,
            setpoint_K=800.0,
            minimum_setpoint_K=500.0,
            maximum_setpoint_K=1000.0,
        ),
    )
