"""Convert public-domain atomic data literals; never execute reference algorithms.

Requires the explicitly supplied periodictable 2.1.0 wheel and the optional
chemicals 1.5.2 public element identities in the locked reference environment.
The runtime consumes only the resulting documented CSV resources.
"""

import argparse
import ast
import csv
import json
import re
import zipfile
from decimal import Decimal
from pathlib import Path


def number_fields(text):
    """Keep nominal uncertainty and natural intervals as separate quantities."""
    interval = re.search(r"\[([0-9.]+),([0-9.]+)\]", text)
    nominal = re.match(r"\s*([0-9.]+)(?:\(([0-9.]+)\))?", text)
    value = uncertainty = ""
    if nominal:
        value = nominal[1]
        if nominal[2]:
            uncertainty = str(
                Decimal(nominal[2])
                if "." in nominal[2]
                else Decimal(nominal[2]) * Decimal(10) ** Decimal(value).as_tuple().exponent
            )
    return (
        value,
        uncertainty,
        "" if interval is None else interval[1],
        ("" if interval is None else interval[2]),
    )


def source_tables(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if "periodictable-2.1.0.dist-info/METADATA" not in names:
            raise ValueError("the fixed data conversion requires periodictable 2.1.0")
        source = archive.read("periodictable/mass.py").decode("iso-8859-15")
    if "This program is public domain" not in source.split('u"""', 1)[0]:
        raise ValueError("the expected data-file public-domain declaration is absent")
    tree = ast.parse(source)
    return {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def convert(wheel, destination):
    import chemicals
    from chemicals.elements import periodic_table

    if chemicals.__version__ != "1.5.2":
        raise ValueError("fixed identity facts require chemicals 1.5.2")
    tables = source_tables(wheel)
    weights = {}
    for line in tables["element_mass"].splitlines():
        _number, symbol, _name, weight, *_ = line.split("\t")
        if symbol in weights:
            raise ValueError("duplicate standard atomic weight")
        weights[symbol] = number_fields(weight)
    element_rows = [
        (element.number, element.symbol, element.name, *weights.get(element.symbol, ("",) * 4))
        for element in periodic_table
    ]
    abundance = {}
    symbol = None
    for line in tables["isotope_abundance"].splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            _, symbol, _ = line.split("\t", 2)
        else:
            mass_number, value, *_ = line.strip().split("\t")
            key = (symbol, int(mass_number))
            if key in abundance:
                raise ValueError("duplicate isotope abundance")
            abundance[key] = number_fields(value.strip())
    isotope_rows = []
    for line in tables["isotope_mass"].splitlines():
        identity, mass, _, _ = line.split(",")
        number, symbol, mass_number = identity.split("-")
        value, uncertainty, _, _ = number_fields(mass)
        if not value:
            raise ValueError("isotope mass cannot be missing")
        isotope_rows.append(
            (
                int(number),
                symbol,
                int(mass_number),
                value,
                uncertainty,
                int("#" in mass),
                *abundance.get((symbol, int(mass_number)), ("",) * 4),
            )
        )
    if len(element_rows) != 118 or len(isotope_rows) != 2939 or len(weights) != 84:
        raise ValueError("fixed atomic source coverage changed")
    if len({(row[0], row[2]) for row in isotope_rows}) != len(isotope_rows):
        raise ValueError("duplicate isotope identity")
    if set(abundance) - {(row[1], row[2]) for row in isotope_rows}:
        raise ValueError("abundance record has no mass record")
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(
        destination / "elements.csv",
        (
            "atomic_number",
            "symbol",
            "name",
            "standard_weight_g_mol",
            "weight_uncertainty_g_mol",
            "weight_low_g_mol",
            "weight_high_g_mol",
        ),
        element_rows,
    )
    write_csv(
        destination / "isotopes.csv",
        (
            "atomic_number",
            "symbol",
            "mass_number",
            "mass_u",
            "mass_uncertainty_u",
            "estimated",
            "abundance_fraction",
            "abundance_uncertainty",
            "abundance_low",
            "abundance_high",
        ),
        isotope_rows,
    )
    metadata = {
        "version": "chemworld-atomic-data-1",
        "reference_distribution": "periodictable 2.1.0",
        "data_file_license": "Public domain (mass.py: Paul Kienzle)",
        "identity_reference": "chemicals 1.5.2 public periodic_table identity facts",
        "mass_reference": "Wang et al. (2021), AME2020, DOI:10.1088/1674-1137/abddaf",
        "weight_reference": "Prohaska et al. (2022), CIAAW 2021, DOI:10.1515/pac-2019-0603",
        "abundance_reference": "CIAAW 2021; representative N/Ar/U/Pb entries as stated by source",
        "abundance_notes": "Intervals retained; no midpoint normalization. Missing stays missing.",
        "standard_weight_notes": "Missing standard weights remain unknown, not mass numbers.",
        "compatibility_notes": "Original 26 default weights retain their separate native source.",
        "elements": len(element_rows),
        "isotopes": len(isotope_rows),
        "standard_weight_records": len(weights),
        "abundance_records": len(abundance),
    }
    (destination / "provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.reference_wheel, args.output)
