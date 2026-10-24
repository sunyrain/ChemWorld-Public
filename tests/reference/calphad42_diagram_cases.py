"""Original declarations and fixed requests for the 42C development block."""

from chemworld.data.scientific_inputs import InputSource
from chemworld.data.tdb import TdbDatabase
from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.calphad_equilibrium import CalphadEquilibriumPolicy, CalphadEquilibriumSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.residual_system import RootPolicy

SCANS = {
    "REGULAR": (500.0, 600.0, 680.0, 740.0, 800.0),
    "LOOP": (500.0, 600.0, 700.0, 800.0, 900.0),
    "CROSSING": (600.0, 650.0, 700.0, 750.0, 800.0),
}
ANCHORS = {
    "REGULAR": (500.0, 550.0, 600.0, 650.0, 700.0),
    "LOOP": (575.0, 625.0, 700.0, 775.0, 825.0),
    "CROSSING": (),
}
RANGES = {"REGULAR": (450.0, 850.0), "LOOP": (450.0, 950.0), "CROSSING": (550.0, 850.0)}


def equilibrium_spec(family, policy=0):
    text = """
ELEMENT FE BCC_A2 55.845 0 0 !
ELEMENT NI FCC_A1 58.69 0 0 !
TYPE_DEFINITION % SEQ * !
FUNCTION GCP 400 30*(T-600-T*LN(T/600)); 1100 N !
PHASE ALPHA % 1 1 !
CONSTITUENT ALPHA :FE,NI: !
PARAMETER G(ALPHA,FE;0) 400 GCP#; 1100 N !
PARAMETER G(ALPHA,NI;0) 400 GCP#; 1100 N !
"""
    if family == "CROSSING":
        text += """
PHASE BETA % 1 1 !
CONSTITUENT BETA :FE,NI: !
PARAMETER G(BETA,FE;0) 400 GCP#+10000*(1-T/700); 1100 N !
PARAMETER G(BETA,NI;0) 400 GCP#+10000*(1-T/700); 1100 N !
"""
    else:
        omega = "12000" if family == "REGULAR" else "2*8.3145*T+0.1*(T-550)*(850-T)"
        text += f"PARAMETER L(ALPHA,FE,NI;0) 400 {omega}; 1100 N !\n"
    source = InputSource(
        DataSourceProvenance("calphad42-diagram", 0, "42C original TDB", "1", "CC0"), "synthetic"
    )
    db = TdbDatabase.from_string(text, source=source)
    models = tuple(
        CefPhaseModel(db, name)
        for name in (("ALPHA", "BETA") if family == "CROSSING" else ("ALPHA",))
    )
    return CalphadEquilibriumSpec(
        models,
        CalphadEquilibriumPolicy(
            grid_intervals=(8, 12)[policy],
            maximum_search_starts=(12, 18)[policy],
            maximum_column_rounds=(8, 12)[policy],
            maximum_iterations=(250, 400)[policy],
            root=RootPolicy((1e-10, 1e-11)[policy], (80, 120)[policy]),
        ),
    )


def diagram_spec(family, policy=0):
    from chemworld.physchem.calphad_phase_diagram import (
        CalphadDiagramSpec,
        CalphadTracePolicy,
        CriticalPointRequest,
        PhaseCrossingRequest,
    )

    critical = {
        "REGULAR": (CriticalPointRequest("ALPHA", 715.0, 0.48, (690.0, 750.0), (0.35, 0.65)),),
        "LOOP": (
            CriticalPointRequest("ALPHA", 560.0, 0.48, (520.0, 580.0), (0.35, 0.65)),
            CriticalPointRequest("ALPHA", 840.0, 0.52, (820.0, 880.0), (0.35, 0.65)),
        ),
        "CROSSING": (),
    }[family]
    return CalphadDiagramSpec(
        equilibrium_spec(family, policy),
        ("FE", "NI"),
        101325.0,
        RANGES[family],
        SCANS[family],
        (0.0, 0.25, 0.5, 0.75, 1.0),
        ANCHORS[family],
        ((600.0 if family == "REGULAR" else 700.0, 0.5),) if family != "CROSSING" else (),
        critical,
        (PhaseCrossingRequest(("ALPHA", "BETA"), 0.5, 695.0, (680.0, 720.0)),)
        if family == "CROSSING"
        else (),
        CalphadTracePolicy((50.0, 25.0)[policy], (0.005, 0.0025)[policy], (512, 1024)[policy]),
    )
