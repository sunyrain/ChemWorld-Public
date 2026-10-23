"""Import published SPA coefficients explicitly for qualification."""

from fluids.optional import spa

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.solar_ephemeris import SOLAR_PARAMETER_SHAPES, SolarParameters


def source_parameters():
    provenance = InputSource(
        DataSourceProvenance(
            "SPA",
            0,
            "Reda/Andreas2004; fluids1.3.1 SPA tables",
            "1.3.1",
            "Caller supplied VSOP87/nutation terms in original angular convention",
        ),
        "literature",
    )
    columns = tuple(
        QuantityColumn(
            name, "dimensionless", "parameter", (-1e30, 1e30), "Encoded source coefficient"
        )
        for name in ("index", "value")
    )
    rows = []
    for name, (count, width) in SOLAR_PARAMETER_SHAPES.items():
        if name == "nutation":
            array = [
                [getattr(spa, f"NUTATION_YTERM_LIST_{j}")[i] for j in range(5)]
                + list(spa.NUTATION_ABCD_LIST[i])
                for i in range(63)
            ]
        elif name == "L5":
            array = [[1, 3.14, 0]]
        else:
            key = {"L": "LONG", "B": "LAT", "R": "RADIUS"}[name[0]]
            array = getattr(spa, f"HELIO_{key}_TABLE_LIST_{name[1]}")
        values = [v for row in array for v in row]
        assert len(values) == count * width
        for i, value in enumerate(values):
            rows.append(
                ScientificRow(
                    f"{name}:{i}",
                    "SPA",
                    name,
                    f"{name}:{i}",
                    "parameters",
                    {
                        "index": InputValue(i, "dimensionless", None, "Source array index"),
                        "value": InputValue(value, "dimensionless", None, "Source coefficient"),
                    },
                )
            )
    return SolarParameters(
        ScientificTable("SPA", "parameters", columns, (provenance,), tuple(rows))
    )
