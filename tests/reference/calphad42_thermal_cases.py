"""Fixed 42B original TDB declarations and finite numerical policies."""

from chemworld.data.scientific_inputs import InputSource
from chemworld.data.tdb import TdbDatabase
from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.calphad_caloric_equilibrium import CalphadCaloricSpec
from chemworld.physchem.calphad_equilibrium import CalphadEquilibriumPolicy, CalphadEquilibriumSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.residual_system import RootPolicy
from chemworld.runtime.calphad_phase_inventory import CalphadPhaseTarget
from chemworld.runtime.calphad_thermal import CalphadThermalSpec

COMMON = """
ELEMENT FE BCC_A2 55.845 0 0 !
ELEMENT NI FCC_A1 58.69 0 0 !
TYPE_DEFINITION % SEQ * !
FUNCTION GCP 400 30*(T-600-T*LN(T/600)); 1100 N !
"""
PURE = (
    COMMON
    + """
PHASE ALPHA % 1 1 !
CONSTITUENT ALPHA :FE: !
PARAMETER G(ALPHA,FE;0) 400 GCP#; 1100 N !
PHASE BETA % 1 1 !
CONSTITUENT BETA :FE: !
PARAMETER G(BETA,FE;0) 400 GCP#+12000*(1-T/600); 1100 N !
"""
)
BINARY = (
    COMMON
    + """
PHASE ALPHA % 1 1 !
CONSTITUENT ALPHA :FE,NI: !
PARAMETER G(ALPHA,FE;0) 400 GCP#; 1100 N !
PARAMETER G(ALPHA,NI;0) 400 GCP#; 1100 N !
PARAMETER L(ALPHA,FE,NI;0) 400 12000; 1100 N !
"""
)


def caloric_spec(kind="binary", policy=0):
    source = InputSource(
        DataSourceProvenance("calphad42-thermal", 0, "42B original fixed TDB", "1", "CC0"),
        "synthetic",
    )
    db = TdbDatabase.from_string(PURE if kind == "pure" else BINARY, source=source)
    models = tuple(
        CefPhaseModel(db, p) for p in (("ALPHA", "BETA") if kind == "pure" else ("ALPHA",))
    )
    equilibrium = CalphadEquilibriumSpec(
        models,
        CalphadEquilibriumPolicy(
            grid_intervals=8,
            maximum_search_starts=12,
            maximum_column_rounds=8,
            maximum_iterations=250,
        ),
    )
    return CalphadCaloricSpec(
        equilibrium,
        101325.0,
        (500.0, 1000.0),
        (550.0, 850.0),
        3,
        RootPolicy((1e-10, 1e-11)[policy], (80, 120)[policy]),
    )


def thermal_spec(kind="binary", method="BDF"):
    targets = (
        (
            CalphadPhaseTarget("alpha", "ALPHA", 0.01, selected=True),
            CalphadPhaseTarget("beta", "BETA", 0.012),
        )
        if kind == "pure"
        else (
            CalphadPhaseTarget("alpha-left", "ALPHA", 0.01, selected=True),
            CalphadPhaseTarget("alpha-right", "ALPHA", 0.01),
        )
    )
    return CalphadThermalSpec(caloric_spec(kind), targets, ode_method=method)


def ph_cases():
    points = [{"kind": "pure", "T": 600.0, "beta": b, "feed": {"FE": 1.0}} for b in (0.25, 0.75)]
    points += [
        {"kind": "binary", "T": t, "z": z, "feed": {"FE": z, "NI": 1 - z}}
        for t, z in ((600.0, 0.3), (800.0, 0.3), (650.0, 0.5), (800.0, 0.5))
    ]
    return [
        {**case, "policy": policy, "id": f"PH-{i}-{policy}"}
        for i, case in enumerate(points)
        for policy in (0, 1)
    ]


def thermal_cases():
    rows = [
        {
            "id": "pure-heater",
            "kind": "pure",
            "initial_T": 590.0,
            "power": 0.1,
            "setpoint": 700.0,
            "times": [0.0, 5.0, 35.0, 65.0, 95.0, 125.0, 160.0],
        },
        {
            "id": "pure-cooler",
            "kind": "pure",
            "initial_T": 610.0,
            "power": -0.1,
            "setpoint": 510.0,
            "times": [0.0, 5.0, 35.0, 65.0, 95.0, 125.0, 160.0],
        },
        {
            "id": "binary-heater",
            "kind": "binary",
            "initial_T": 600.0,
            "power": 0.05,
            "setpoint": 900.0,
            "times": [0.0, 30.0, 60.0, 90.0, 120.0, 160.0, 200.0],
        },
        {
            "id": "binary-cooler",
            "kind": "binary",
            "initial_T": 800.0,
            "power": -0.05,
            "setpoint": 510.0,
            "times": [0.0, 30.0, 60.0, 90.0, 120.0, 160.0, 200.0],
        },
        {
            "id": "binary-proportional",
            "kind": "binary",
            "initial_T": 600.0,
            "power": 0.05,
            "setpoint": 800.0,
            "times": [0.0, 30.0, 60.0, 90.0, 120.0, 160.0, 200.0],
            "conductance": 0.0005,
        },
        {
            "id": "binary-pause",
            "kind": "binary",
            "initial_T": 600.0,
            "power": 0.05,
            "setpoint": 900.0,
            "times": [0.0, 40.0, 80.0, 120.0, 160.0, 200.0, 240.0],
        },
    ]
    return [{**row, "method": method} for row in rows for method in ("BDF", "Radau")]


def device(case):
    return ThermalDeviceSpec(
        conductance_W_K=case.get("conductance", 1.0),
        maximum_power_W=abs(case["power"]),
        maximum_cooling_power_W=abs(case["power"]),
        setpoint_K=case["setpoint"],
        minimum_setpoint_K=500.0,
        maximum_setpoint_K=1000.0,
    )
