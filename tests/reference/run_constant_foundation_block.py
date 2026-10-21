"""Fixed 22B3 source preservation, dimensional conversion, and input consumers.

The oracle reads source literals with stdlib parsers and explicit arithmetic;
it never calls reference algorithms or the native parser/converter as oracle.
"""

import argparse
import csv
import json
import math
import time
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from chemworld.data.constants import ConstantDatabase, SourceCorrelation
from chemworld.data.scientific_inputs import QuantityColumn, ScientificTable

COUNTS = dict(
    zip(
        [
            "PSRK",
            "CRC_CRITICAL",
            "DIPPR_PINA_MARTINES",
            "FEDORS",
            "IUPAC",
            "MATHEWS",
            "OMEGA_PSAT",
            "PASSUT_DANNER",
            "WILSON_JASPERSON",
            "YAWS",
            "STAVELEY",
            "ALSHEHRI_HANSEN",
            "CCCBDB",
            "CHEMSEP_XML",
            "CHEMSEP_RG",
            "COMMON_CHEMISTRY",
            "CRC_RI",
            "DAHMEN_IQT",
            "LEGACY_ELEMENTS",
            "OPENBABEL_ELEMENTS",
            "FLORIAN_CETANE",
            "FLORIAN_MON",
            "FLORIAN_RON",
            "FLORIAN_ANN",
            "HEOS",
            "HSPIPY_HANSEN",
            "JOBACK",
            "MULLER",
            "CRC_INORGANIC",
            "CRC_ORGANIC",
            "POLING",
            "PSI4_DIPOLE",
            "PSI4_LINEAR",
            "PSI4_RG",
            "RUBEN_HANSEN",
            "SCHRIER_HANSEN",
            "COMBUSTDB_CETANE",
            "COMBUSTDB_MON",
            "COMBUSTDB_PREDICTIONS",
            "COMBUSTDB_RON",
            "VDI",
            "WEBBOOK",
            "WIKIDATA",
        ],
        (
            995,
            861,
            1799,
            57445,
            810,
            120,
            29784,
            192,
            29726,
            7549,
            189,
            888,
            424,
            431,
            423,
            79291,
            4490,
            58,
            112,
            118,
            85,
            255,
            257,
            532,
            145,
            248,
            52224,
            254,
            2438,
            10867,
            264,
            411,
            276,
            276,
            1211,
            113,
            408,
            278,
            1066,
            278,
            58,
            13779,
            10122,
        ),
        strict=True,
    )
)
TOTAL = 312235

# Independently stated source-to-canonical contracts; no native unit function.
TEXT_UNITS = {}
for names, unit, canonical, scale in (
    ("Tc Tt Tm Tb RIT T_flash T_autoignition POUR_POINT CLOUD_POINT Tc_error", "K", "K", 1),
    ("Pc Pt Pc_error Pt_uncertainty", "Pa", "Pa", 1),
    ("Vc Vms Vml Vc_error", "m^3/mol", "m^3/mol", 1),
    ("MW", "g/mol", "g/mol", 1),
    ("omega Zc RI logP LFL UFL RON MON CETANE linear", "dimensionless", "dimensionless", 1),
    ("Hfs Hfl Hfg Hf Hfus Hvap Hsub", "J/mol", "J/mol", 1),
    ("S0s S0l S0g S0", "J/(mol*K)", "J/(mol*K)", 1),
    ("rho", "kg/m^3", "kg/m^3", 1),
    ("RG", "m", "nm", 1e9),
    ("dipole_moment", "Debye", "C*m", 1e-21 / 299792458),
    ("IGNITION_DELAY", "s", "s", 1),
    ("HANSEN_DELTA_D HANSEN_DELTA_P HANSEN_DELTA_H", "Pa^0.5", "Pa^0.5", 1),
):
    TEXT_UNITS.update(dict.fromkeys(names.split(), (unit, canonical, scale)))
XML_UNITS = {
    "K": ("K", "K", 1),
    "Pa": ("Pa", "Pa", 1),
    "m3/kmol": ("m^3/kmol", "m^3/mol", 0.001),
    "_": ("dimensionless", "dimensionless", 1),
    "kg/kmol": ("kg/kmol", "g/mol", 1),
    "J0.5/m1.5": ("Pa^0.5", "Pa^0.5", 1),
    "Coulomb.m": ("C*m", "C*m", 1),
    "J/kmol": ("J/kmol", "J/mol", 0.001),
    "J/kmol/K": ("J/(kmol*K)", "J/(mol*K)", 0.001),
    "m": ("m", "nm", 1e9),
    "kg0.25.m3/s0.5/kmol": (
        "kg^0.25*m^3/(s^0.5*kmol)",
        "kg^0.25*m^3/(s^0.5*mol)",
        0.001,
    ),
    "m2/kmol": ("m^2/kmol", "m^2/mol", 0.001),
}
ELEMENT_NAMES = [
    "atomic_number",
    "symbol",
    "allred_rochow",
    "covalent_radius",
    "bond_order_radius",
    "vdw_radius",
    "max_bond_valence",
    "historical_mass",
    "pauling",
    "ionization_energy",
    "electron_affinity",
    "red",
    "green",
    "blue",
    "name",
]
ELEMENT_UNITS = dict.fromkeys(
    ["atomic_number", "allred_rochow", "max_bond_valence", "pauling", "red", "green", "blue"],
    ("dimensionless", "dimensionless", 1),
)
ELEMENT_UNITS.update(
    dict.fromkeys(("covalent_radius", "bond_order_radius", "vdw_radius"), ("angstrom", "nm", 0.1))
)
ELEMENT_UNITS.update(
    dict.fromkeys(("ionization_energy", "electron_affinity"), ("eV", "J", 1.602176634e-19))
)
ELEMENT_DEFAULTS = dict(
    zip(
        [
            "allred_rochow",
            "covalent_radius",
            "vdw_radius",
            "max_bond_valence",
            "pauling",
            "ionization_energy",
            "electron_affinity",
        ],
        (0, 1.6, 2, 6, 0, 0, 0),
        strict=True,
    )
)
VDI_UNITS = {
    "T": "K",
    "P": "Pa",
    "Density (l)": "kg/m^3",
    "Density (g)": "kg/m^3",
    "Hvap": "J/mol",
    "Cp (l)": "J/(mol*K)",
    "Cp (g)": "J/(mol*K)",
    "Mu (l)": "Pa*s",
    "Mu (g)": "Pa*s",
    "K (l)": "W/(m*K)",
    "K (g)": "W/(m*K)",
    "Pr (l)": "dimensionless",
    "Pr (g)": "dimensionless",
    "sigma": "N/m",
    "Beta": "1/K",
    "Volume (l)": "m^3/mol",
    "Volume (g)": "m^3/mol",
    "MW": "g/mol",
    "Tc": "K",
}


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=2e-14, abs_tol=1e-300), (actual, expected)
    return abs(actual - expected)


def names_for(headers):
    result = []
    counts = Counter()
    for i, header in enumerate(headers):
        name = header.strip() or f"unnamed_{i}"
        counts[name] += 1
        result.append(name if counts[name] == 1 else f"{name}__{counts[name]}")
    return result


def reference_rows(source, directory):
    path = directory / source.relative_path
    if source.kind == "xml":
        root = ET.parse(path).getroot()
        return [(list(node), dict(root.attrib)) for node in root]
    if source.kind == "vdi":
        return list(json.loads(path.read_text(encoding="utf-8")).items())
    with path.open(encoding="utf-8", newline="") as stream:
        if source.kind == "elements":
            rows = list(
                csv.reader(
                    (ln for ln in stream if ln.strip() and not ln.startswith("#")), delimiter="\t"
                )
            )
            headers = ELEMENT_NAMES
        else:
            headers, *rows = csv.reader(stream, delimiter="\t")
    return [(headers, row) for row in rows]


def check_scalar(record, field, raw, contract, counts, point=None):
    actual = record.value(field, point=point)
    quality, value = "valid", None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        quality = "missing"
    else:
        try:
            if field == "linear" and raw in ("True", "False", True, False):
                value = 1.0 if raw in ("True", True) else 0.0
            elif isinstance(raw, bool):
                raise ValueError("boolean")
            else:
                value = float(raw)
            if not math.isfinite(value):
                value, quality = None, "invalid"
        except (ValueError, TypeError):
            quality = "invalid"
    if (
        record.source_id == "OPENBABEL_ELEMENTS"
        and quality == "valid"
        and field in ELEMENT_DEFAULTS
        and value == ELEMENT_DEFAULTS[field]
    ):
        quality = "source_default_or_ambiguous"
    assert actual.raw_value == raw and actual.quality == quality, (field, raw, actual)
    assert actual.value == value, (field, value, actual.value)
    counts[quality] += 1
    assert actual.unit == (contract[0] if contract else None), (record.record_id, field)
    if contract and quality == "valid":
        quantity = actual.quantity()
        assert quantity.unit == contract[1]
        error = close(quantity.value, value * contract[2])
        counts["max_conversion_error"] = max(counts["max_conversion_error"], error)
        counts["quantity_checks"] += 1
    elif contract is None:
        counts["unknown_unit"] += 1
    counts["fields"] += 1


def check_record(db, source, index, reference, counts):
    record = db.records(source.source_id)[index]
    assert record.row_number == index + 1
    assert db.record(record.record_id, method=source.source_id) is record
    assert record in db.records_for(record.source_key, method=source.source_id)
    counts["lookups"] += 2
    if record.cas_number:
        assert record in db.records_for(record.cas_number, method=source.source_id)
        counts["lookups"] += 1
    if source.kind == "xml":
        nodes, metadata = reference
        assert record.document_metadata == metadata
        assert record.headers == tuple(n.tag for n in nodes)
        for node in nodes:
            assert record.source_attributes[node.tag] == node.attrib
            if "value" not in node.attrib or len(node):
                correlation = record.raw(node.tag)
                assert isinstance(correlation, SourceCorrelation)
                assert ET.tostring(ET.fromstring(correlation.xml)).strip() == (
                    ET.tostring(node).strip()
                )
                counts["correlations"] += 1
            else:
                check_scalar(
                    record,
                    node.tag,
                    node.attrib["value"],
                    XML_UNITS.get(node.attrib.get("units")),
                    counts,
                )
    elif source.kind == "vdi":
        key, fields = reference
        assert record.source_key == key and record.field_names == tuple(fields)
        for name, value in fields.items():
            assert record.raw(name) == (tuple(value) if isinstance(value, list) else value)
            if not isinstance(value, list):
                unit = VDI_UNITS.get(name)
                check_scalar(record, name, value, (unit, unit, 1) if unit else None, counts)
    else:
        headers, row = reference
        assert record.headers == tuple(headers) and record.raw_fields == tuple(row)
        names = names_for(headers)
        assert record.field_names == tuple(names)
        assert record.source_key == row[names.index("CAS") if "CAS" in names else 0]
        contracts = (
            {}
            if source.source_id == "LEGACY_ELEMENTS"
            else ELEMENT_UNITS
            if source.kind == "elements"
            else TEXT_UNITS
        )
        for name, raw in zip(names, row, strict=True):
            check_scalar(record, name, raw, contracts.get(name), counts)


def roundtrip(table):
    payload = table.to_dict()
    assert ScientificTable.from_json(json.dumps(payload)).to_dict() == payload
    metadata = {k: v for k, v in payload.items() if k != "rows"}
    assert ScientificTable.from_csv(table.to_csv(), metadata).to_dict() == payload


def critical_consumer(db, cas, reference):
    from chemworld.physchem.eos import EOSComponentSpec

    columns = tuple(
        QuantityColumn(name, unit, "parameter", bounds, "critical state")
        for name, unit, bounds in (
            ("Tc", "K", (1, 2000)),
            ("Pc", "Pa", (1, 1e9)),
            ("omega", "dimensionless", (-2, 2)),
        )
    )
    table = db.parameter_table(
        (cas,),
        method="HEOS",
        columns=columns,
        field_bindings={c.quantity_id: c.quantity_id for c in columns},
        dataset_id="HEOS-critical:" + cas,
        group_id=cas,
    )
    roundtrip(table)
    values = table.rows[0].values
    model = EOSComponentSpec(cas, values["Tc"].value, values["Pc"].value, values["omega"].value)
    for actual, field in (
        (model.critical_temperature_K, "Tc"),
        (model.critical_pressure_Pa, "Pc"),
        (model.acentric_factor, "omega"),
    ):
        close(actual, float(reference[field]))
    return {"component": model.to_dict(), "source": table.sources[0].to_dict()}


def crystal_consumer(db, expected_density):
    from chemworld.examples.native_crystal_research import (
        synthetic_crystal_model,
        synthetic_crystal_thermodynamics,
    )
    from chemworld.physchem.elements import molecular_weight
    from chemworld.world.crystal_world import crystal_world_spec
    from chemworld.world.parameter_inputs import ModelParameterTarget, bind_parameter_row

    mass = molecular_weight({"Na": 1, "Cl": 1}) / 1000
    model = synthetic_crystal_model(mass)
    physics = crystal_world_spec(
        model, synthetic_crystal_thermodynamics(), {"P": "NaCl", "I": "KCl"}
    ).physics
    before = physics.to_dict()
    col = QuantityColumn(
        "rho",
        "kg/m^3",
        "parameter",
        (100, 10000),
        "CRC solid NaCl density; remaining kinetics synthetic",
    )
    table = db.parameter_table(
        ("7647-14-5",),
        method="CRC_INORGANIC",
        columns=(col,),
        field_bindings={"rho": "rho"},
        dataset_id="NaCl-density",
        group_id="NaCl",
    )
    roundtrip(table)
    target = ModelParameterTarget(
        col,
        "crystal",
        physics.modules.model_specs()["crystal"].version,
        "reactor",
        ("kinetics", "crystal_density_kg_m3"),
    )
    imported = bind_parameter_row(physics, table, table.rows[0].row_id, (target,))
    actual = imported.physics.models_of("crystal")["reactor"]
    assert actual.kinetics.crystal_density_kg_m3 == expected_density
    assert physics.to_dict() == before
    assert imported.physics.initial_states == physics.initial_states
    diameter = 1e-4
    expected = expected_density * math.pi * diameter**3 / (6 * mass)
    error = close(actual.particle_moles(diameter), expected)
    assert actual.particle_moles(diameter) != model.particle_moles(diameter)
    json.dumps(imported.physics.to_dict())
    json.dumps(imported.receipt)
    return {
        "particle_moles_error": error,
        "receipt": imported.receipt,
        "scope": "density consumption only; other crystallization inputs synthetic",
    }


def run(directory, output):
    if output.exists():
        raise FileExistsError("preserve previous outputs; choose a new filename")
    assert version("chemicals") == "1.5.2", "this block requires the declared reference release"
    db = ConstantDatabase(directory)
    assert set(COUNTS) == {source.source_id for source in db.catalog}
    started = time.monotonic()
    failures, stages, consumer_results = [], {}, []
    completed = 0

    def unit(identity, action):
        nonlocal completed
        try:
            result = action()
        except Exception:
            failures.append({"identity": identity, "failure": traceback.format_exc()})
            result = None
        completed += 1
        if completed % 1000 == 0:
            progress(identity)
        return result

    def progress(stage):
        elapsed = max(time.monotonic() - started, 1e-9)
        rate = completed / elapsed
        print(
            f"{stage}: {completed}/{TOTAL}, failed={len(failures)}, "
            f"{rate:.1f} units/s, ETA={(TOTAL - completed) / max(rate, 1e-9):.1f}s",
            flush=True,
        )

    references = {}
    for source in db.catalog:
        count = COUNTS[source.source_id]
        counts = Counter()
        before_failed = len(failures)

        def layout(s=source, n=count):
            reference = reference_rows(s, directory)
            references[s.source_id] = reference
            assert len(reference) == n and len(db.records(s.source_id)) == n
            assert s.provenance.version and s.provenance.license and s.provenance.citation

        unit(source.source_id + ":layout", layout)
        for index in range(count):
            unit(
                f"{source.source_id}:{index + 1}",
                lambda i=index, s=source, c=counts: check_record(
                    db, s, i, references[s.source_id][i], c
                ),
            )
        stages[source.source_id] = {
            "records": count,
            "units": count + 1,
            "failed": len(failures) - before_failed,
            "metrics": dict(counts),
        }
        progress(source.source_id)

    vdi = references.get("VDI", [])
    point_count = 0
    for index in range(58):
        # The fixed distribution has ten points for each of its 58 compounds.
        for point in range(10):

            def check_point(i=index, p=point):
                key, fields = vdi[i]
                record = db.record(key, method="VDI")
                assert len(fields["T"]) == 10
                for name in VDI_UNITS.keys() - {"MW", "Tc"}:
                    raw = fields[name][p]
                    check_scalar(
                        record,
                        name,
                        raw,
                        (VDI_UNITS[name], VDI_UNITS[name], 1),
                        stages["VDI"].setdefault("points", Counter()),
                        point=p,
                    )
                    if raw is not None and name != "Beta":
                        assert raw >= 0
                    if name == "T":
                        assert raw > 0

            unit(f"VDI-point:{index}:{point}", check_point)
            point_count += 1

        def check_table(i=index):
            key, fields = vdi[i]
            names = [name for name in fields if isinstance(fields[name], list)]
            columns = tuple(
                QuantityColumn(
                    name,
                    VDI_UNITS[name],
                    "condition" if name == "T" else "observation",
                    (-1e15, 1e15),
                    "VDI saturation; no extrapolation",
                )
                for name in names
            )
            table = db.saturation_table(
                key,
                columns=columns,
                field_bindings={name: name for name in names},
                dataset_id="VDI:" + key,
                group_id=key,
                split="validation",
            )
            roundtrip(table)
            assert len(table.rows) == len(fields["T"])
            for row, p in zip(table.rows, range(len(fields["T"])), strict=True):
                for name in names:
                    assert row.values[name].value == fields[name][p]
            for name in names:
                temperatures, values = db.saturation_series(key, name).valid_points()
                expected = [
                    (t, value)
                    for t, value in zip(fields["T"], fields[name], strict=True)
                    if value is not None
                ]
                assert list(zip(temperatures, values, strict=True)) == expected

        unit(f"VDI-table:{index}", check_table)
    progress("VDI inputs")

    def text_record(source, identity):
        matches = [
            dict(zip(headers, row, strict=True))
            for headers, row in references[source]
            if row[headers.index("CAS")] == identity
        ]
        assert len(matches) == 1
        return matches[0]

    for cas in ("7732-18-5", "7727-37-9", "64-17-5"):
        consumer_results.append(
            unit(
                "critical-consumer:" + cas,
                lambda key=cas: critical_consumer(db, key, text_record("HEOS", key)),
            )
        )
    consumer_results.append(
        unit(
            "NaCl-crystal-consumer",
            lambda: crystal_consumer(db, float(text_record("CRC_INORGANIC", "7647-14-5")["rho"])),
        )
    )
    assert completed == TOTAL and point_count == 580
    summary = {
        "block": "22B3-constant-foundation",
        "mode": "development",
        "reference_version": "chemicals 1.5.2",
        "source_directory": str(directory.resolve()),
        "total": TOTAL,
        "passed": completed - len(failures),
        "failed": len(failures),
        "denominators": {
            "source_layouts": 43,
            "source_records": 311550,
            "vdi_points": 580,
            "vdi_tables": 58,
            "consumers": 4,
        },
        "elapsed_seconds": time.monotonic() - started,
        "sources": stages,
        "consumer_results": consumer_results,
        "failures": failures,
    }
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    progress("complete")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "total",
                    "passed",
                    "failed",
                    "elapsed_seconds",
                )
            }
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source_directory, args.output))
