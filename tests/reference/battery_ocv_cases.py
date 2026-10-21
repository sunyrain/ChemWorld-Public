"""Fixed OCV coefficient data, with source/units, for the eight-unit block."""

import json

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_dfn import BatteryDFNSpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance


def source_curve(electrode):
    # Numerical mathematical relationships from Chen et al. and released
    # PyBaMM 25.4.2 parameter data; no upstream algorithm is copied or called.
    if electrode == "negative":
        polynomial = [0.2482]
        exponentials = [(1.9793, -39.3631, 0.0)]
        transitions = [
            (-0.0909, 29.8538, 0.1234),
            (-0.04478, 14.9159, 0.2769),
            (-0.0205, 30.4444, 0.6103),
        ]
    elif electrode == "positive":
        polynomial = [4.4875, -0.8090]
        exponentials = []
        transitions = [
            (-0.0428, 18.5138, 0.5542),
            (-17.7326, 15.7890, 0.3117),
            (17.5842, 15.9308, 0.3120),
        ]
    else:
        raise ValueError("unknown electrode")
    cells, columns = {}, []

    def add(name, value, unit):
        columns.append(
            QuantityColumn(
                name, unit, "parameter", (-1000, 1000), "theta dimensionless; electrode V vs Li/Li+"
            )
        )
        cells[name] = InputValue(value, unit, None, "Coefficient uncertainty not supplied")
        return name

    poly_names = [add(f"p{i}", p, "V") for i, p in enumerate(polynomial)]

    def terms(kind, values):
        return [
            tuple(
                add(f"{kind}{i}_{label}", value, unit)
                for label, value, unit in zip(
                    ("amplitude", "scale", "shift"),
                    term,
                    ("V", "dimensionless", "dimensionless"),
                    strict=True,
                )
            )
            for i, term in enumerate(values)
        ]

    exp_names, tanh_names = terms("exp", exponentials), terms("tanh", transitions)
    source = InputSource(
        DataSourceProvenance(
            "Chen2020-ocv",
            0,
            "Chen et al., J. Electrochem. Soc. 167 080534; doi:10.1149/1945-7111/ab9050; "
            "PyBaMM 25.4.2 Chen2020 OCP parameter relationships",
            "Chen2020 / PyBaMM 25.4.2",
            "PyBaMM parameter implementation BSD-3-Clause; article CC-BY-NC-ND-4.0",
        ),
        "literature",
    )
    table = ScientificTable(
        "Chen2020-" + electrode,
        "parameters",
        tuple(columns),
        (source,),
        (ScientificRow(electrode, "Chen2020-ocv", electrode, electrode, "parameters", cells),),
    )
    table = ScientificTable.from_json(json.dumps(table.to_dict()))
    return ElectrodeOCV.from_parameter_row(
        table,
        electrode,
        material_id="Chen2020-LGM50-" + ("graphite" if electrode == "negative" else "NMC"),
        polynomial=poly_names,
        exponential_terms=exp_names,
        tanh_terms=tanh_names,
    ), table


def models():
    negative, _ = source_curve("negative")
    positive, _ = source_curve("positive")
    kwargs = {
        "negative": ParticleSpec(cells=64),
        "positive": ParticleSpec(cells=64),
        "initial_negative_fraction": 0.8,
        "initial_positive_fraction": 0.3,
        "negative_ocv": negative,
        "positive_ocv": positive,
        "provenance": "synthetic-geometry-with-Chen2020-OCV",
    }
    electrolyte = BatteryElectrolyteSpec(cells=(24,) * 3)
    return [
        ("SPM", BatterySPMSpec(**kwargs)),
        ("SPMe", BatterySPMSpec(**kwargs, electrolyte=electrolyte)),
        ("DFN", BatteryDFNSpec(**kwargs, electrolyte=electrolyte)),
    ]


def released_functions():
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("fixed OCV reference requires PyBaMM 25.4.2")
    p = pybamm.ParameterValues("Chen2020")
    return {name: p[name.capitalize() + " electrode OCP [V]"] for name in ("negative", "positive")}
