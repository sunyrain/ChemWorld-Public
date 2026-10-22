"""Predeclared authored ECM inputs; no measured-material claims or reference calls."""

from itertools import product

import numpy as np

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_thevenin import CONSTANT_UNITS, TheveninSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.tabulated_functions import TabulatedScalar

TIMES = np.linspace(0., 10., 21)
CASES = (*(
    {"rc_count": n, "target": current, "unit": "A", "limit_A": 10., "feedback": False}
    for n in (0, 1, 2, 4) for current in (.5, -.5)
),
    {"rc_count": 1, "target": 3.5, "unit": "V", "limit_A": 10., "feedback": False},
    {"rc_count": 1, "target": 3.9, "unit": "V", "limit_A": 10., "feedback": False},
    {"rc_count": 1, "target": .5, "unit": "A", "limit_A": 10., "feedback": True},
    {"rc_count": 1, "target": -.5, "unit": "A", "limit_A": 10., "feedback": True},
    {"rc_count": 1, "target": 3.3, "unit": "V", "limit_A": .1, "feedback": False},
    {"rc_count": 1, "target": 4., "unit": "V", "limit_A": .1, "feedback": False},
)


def table(identity, units, rows):
    source = InputSource(DataSourceProvenance("chemworld:thevenin-qualification-synthetic", 0,
        "Authored RC/two-node qualification equations; not measured material data", "1",
        "MIT (ChemWorld-authored qualification inputs)"), "synthetic")
    return ScientificTable(identity, "parameters", tuple(
        QuantityColumn(key, unit, "parameter", (-1e9, 1e9), "Authored synthetic inputs")
        for key, unit in units.items()), (source,), tuple(
        ScientificRow(str(i), source.provenance.source_id, identity, str(i), "parameters",
            {k: InputValue(v, units[k], None, "Input uncertainty not supplied")
             for k, v in row.items()}) for i, row in enumerate(rows)))


def law(identity, grids, unit, formula):
    inputs = {"soc": "dimensionless", "temperature_K": "K", "current_A": "A"}
    data = [{**dict(zip(grids, point, strict=True)), "value": formula(*point)}
            for point in product(*grids.values())]
    return TabulatedScalar(table(identity, {**{k: inputs[k] for k in grids}, "value": unit},
                                 data), tuple((k, k, inputs[k]) for k in grids), "value", unit)


def native_case(index):
    case = CASES[index-1]
    n = case["rc_count"]
    p = {"capacity_C": 100., "maximum_current_A": case["limit_A"],
         "minimum_voltage_V": .1, "maximum_voltage_V": 10., "initial_soc": .7,
         "minimum_soc": 0., "maximum_soc": 1., "initial_cell_temperature_K": 310.,
         "initial_jig_temperature_K": 310., "minimum_temperature_K": 250.,
         "maximum_temperature_K": 400., "cell_heat_capacity_J_K": 100.,
         "jig_heat_capacity_J_K": 500., "cell_jig_conductance_W_K": .5,
         "jig_ambient_conductance_W_K": .2, "reference_frame_temperature_K": 298.15,
         "housing_volume_L": .01, **{f"initial_rc_{k}_V": 0. for k in range(n)}}

    def constant(identity, value, unit):
        return law(identity, {}, unit, lambda: value)

    r0 = constant("series-r", .1, "Ohm")
    rc = tuple((constant(f"r{k}", .2*(k+1), "Ohm"),
                constant(f"c{k}", 5./(k+1), "F")) for k in range(n))
    if case["feedback"]:
        axes = {"soc": (0., .5, 1.), "temperature_K": (250., 298.15, 400.),
                "current_A": (-10., 0., 10.)}
        r0 = law("feedback-r0", axes, "Ohm",
                 lambda z, T, current: .08+.02*z+.0001*(T-298.15)+.002*current)
        rc = ((law("feedback-r1", axes, "Ohm",
                   lambda z, T, current: .15+.05*z+.0002*(T-298.15)+.001*current),
               law("feedback-c1", axes, "F",
                   lambda z, T, current: 4.+z+.002*(T-298.15)+.01*current)),)
    spec = TheveninSpec(table("circuit-constants", {
        **CONSTANT_UNITS, **{f"initial_rc_{k}_V": "V" for k in range(n)}}, [p]), "0",
        law("ocv", {"soc": (0., .5, 1.)}, "V", lambda z: 3+z), r0, rc,
        constant("entropy", 1e-4, "V/K"), ScalarControlProgram(((0., 298.15),), "K"))
    return spec, ScalarControlProgram(((0., case["target"]),), case["unit"])
