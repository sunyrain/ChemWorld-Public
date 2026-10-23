"""Explicit import of empirical coefficient arrays, for qualification only."""

from fluids.nrlmsise00 import nrlmsise_00_data as source

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.msis_atmosphere import MSIS_PARAMETER_SHAPES, MSISParameters


def source_parameters():
    provenance = InputSource(
        DataSourceProvenance(
            "NRLMSISE00",
            0,
            "Picone et al. 2002; fluids1.3.1 distribution",
            "1.3.1",
            "Caller imported coefficient arrays; source model numeric convention",
        ),
        "literature",
    )
    columns = tuple(
        QuantityColumn(
            name, "dimensionless", "parameter", (-1e30, 1e30), "Encoded original-model coefficients"
        )
        for name in ("index", "value")
    )
    rows = []
    for name, (count, width) in MSIS_PARAMETER_SHAPES.items():
        array = getattr(source, name)
        values = array if count == 1 else [value for row in array for value in row]
        assert len(values) == count * width
        for i, value in enumerate(values):
            rows.append(
                ScientificRow(
                    f"{name}:{i}",
                    "NRLMSISE00",
                    name,
                    f"{name}:{i}",
                    "parameters",
                    {
                        "index": InputValue(i, "dimensionless", None, "Source array index"),
                        "value": InputValue(
                            value, "dimensionless", None, "Source empirical coefficient"
                        ),
                    },
                )
            )
    return MSISParameters(
        ScientificTable("NRLMSISE00", "parameters", columns, (provenance,), tuple(rows))
    )
