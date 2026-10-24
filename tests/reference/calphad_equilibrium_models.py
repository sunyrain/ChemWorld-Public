"""Fixed synthetic CC0 cases for the prospective 42A block."""

ELEMENTS = """
ELEMENT FE BCC_A2 55.845 0 0 !
ELEMENT NI FCC_A1 58.69 0 0 !
ELEMENT CR BCC_A2 51.996 0 0 !
TYPE_DEFINITION % SEQ * !
"""


def solution(name, members, energies):
    return (f"PHASE {name} % 1 1 !\nCONSTITUENT {name} :{','.join(members)}: !\n" +
            "\n".join(f"PARAMETER G({name},{e};0) 100 {g}; 2500 N !"
                      for e, g in zip(members, energies, strict=True)))


BINARY = ELEMENTS + solution("ALPHA", ("FE", "NI"), (0, 0)) + """
PARAMETER L(ALPHA,FE,NI;0) 100 12000; 2500 N !
"""
TERNARY = ELEMENTS + "\n".join(solution(name, ("FE", "NI", "CR"), energy)
    for name, energy in (("FE_PREF", (0, 12000, 12000)),
        ("NI_PREF", (12000, 0, 12000)), ("CR_PREF", (12000, 12000, 0)),
        ("HIGH", (20000, 20000, 20000))))
SUBLATTICE = ELEMENTS + solution("FCC", ("FE", "NI"), (-4000, 0)) + """
PHASE ORDERED % 2 1 2 !
CONSTITUENT ORDERED :FE,NI:FE,NI: !
PARAMETER G(ORDERED,FE:FE;0) 100 -5000; 2500 N !
PARAMETER G(ORDERED,FE:NI;0) 100 -14000; 2500 N !
PARAMETER G(ORDERED,NI:FE;0) 100 -2000; 2500 N !
PARAMETER G(ORDERED,NI:NI;0) 100 -6000; 2500 N !
"""
COMPOUND = ELEMENTS + solution("FCC", ("FE", "NI"), (0, 0)) + """
PHASE COMPOUND % 2 1 1 !
CONSTITUENT COMPOUND :FE:NI: !
PARAMETER G(COMPOUND,FE:NI;0) 100 -14000; 2500 N !
"""


def cases():
    return [
        {"id": "binary-gap", "tdb": BINARY, "phases": ["ALPHA"], "T": 600.,
         "feed": {"FE": .3, "NI": .7}},
        {"id": "binary-hot", "tdb": BINARY, "phases": ["ALPHA"], "T": 1000.,
         "feed": {"FE": .5, "NI": .5}},
        {"id": "binary-dilute", "tdb": BINARY, "phases": ["ALPHA"], "T": 600.,
         "feed": {"FE": .03, "NI": .97}},
        *[{"id": name, "tdb": TERNARY, "phases": phases, "T": t,
           "feed": dict(zip(("FE", "NI", "CR"), x, strict=True))}
          for name, phases, t, x in (
              ("ternary-three", ["FE_PREF", "NI_PREF", "CR_PREF", "HIGH"], 700., (.3, .3, .4)),
              ("ternary-hot", ["FE_PREF", "NI_PREF", "CR_PREF", "HIGH"], 1000., (.5, .25, .25)),
              ("ternary-excluded", ["FE_PREF", "NI_PREF", "HIGH"], 700., (.3, .3, .4)))],
        *[{"id": f"sublattice-{int(t)}", "tdb": SUBLATTICE, "phases": ["ORDERED", "FCC"],
           "T": t, "feed": {"FE": x, "NI": 1-x}} for t, x in ((400., .3), (700., .5), (1000., .7))],
        *[{"id": f"compound-{int(t)}", "tdb": COMPOUND, "phases": ["COMPOUND", "FCC"],
           "T": t, "feed": {"FE": x, "NI": 1-x}} for t, x in ((500., .3), (900., .7), (1500., .3))],
    ]


def native_spec(case, intervals=8, **policy):
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.data.tdb import TdbDatabase
    from chemworld.physchem.calphad import CefPhaseModel
    from chemworld.physchem.calphad_equilibrium import (
        CalphadEquilibriumPolicy,
        CalphadEquilibriumSpec,
    )
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    source = InputSource(DataSourceProvenance(
        "synthetic-calphad-equilibrium", 0, "42A fixed synthetic CEF block", "1", "CC0"),
        "synthetic")
    db = TdbDatabase.from_string(case["tdb"], source=source)
    return CalphadEquilibriumSpec(tuple(CefPhaseModel(db, p) for p in case["phases"]),
                                 CalphadEquilibriumPolicy(grid_intervals=intervals, **policy))
