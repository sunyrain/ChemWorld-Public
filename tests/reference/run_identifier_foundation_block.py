"""Fixed 22B2 block: 79871 source units, 24 queries and 8 native consumers.

The original small-only consumer binding has three known invalid prerequisites.
Keep those failures reproducible; run_identifier_consumer_block.py qualifies the
separate, explicitly source-bound eight-consumer block. Do not reinterpret this
block as a fully passing acceptance run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import threading
import time
import traceback
from pathlib import Path

from chemworld.data.identifiers import IdentifierDatabase

SOURCES = (
    ("pubchem-large", "chemical identifiers pubchem large.tsv", 71347),
    ("pubchem-small", "chemical identifiers pubchem small.tsv", 1815),
    ("example-user", "chemical identifiers example user db.tsv", 2326),
    ("anion", "Anion db.tsv", 203),
    ("cation", "Cation db.tsv", 202),
    ("inorganic", "Inorganic db.tsv", 607),
    ("fake-registry", "Fake CAS Registry.tsv", 999),
    ("mixtures", "Mixtures Compositions.tsv", 90),
    ("dippr-2014", "dippr_2014.csv", 2278),
    ("anion-preferences", "anion_preferences.json", 1),
    ("cation-preferences", "cation_preferences.json", 1),
    ("inorganic-preferences", "inorganic_preferences.json", 1),
    ("organic-preferences", "organic_preferences.json", 1),
)
QUERIES = (
    ("name", "water"),
    ("name", "ethanol"),
    ("name", "sodium chloride"),
    ("name", "acetic acid"),
    ("cas", "7732-18-5"),
    ("cas", "64-17-5"),
    ("cas", 7732185),
    ("cas", 64175),
    ("pubchem", 962),
    ("pubchem", 702),
    ("smiles", "O"),
    ("smiles", "CCO"),
    ("smiles", "C"),
    ("inchi", "InChI=1S/H2O/h1H2"),
    ("inchi", "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"),
    ("inchikey", "XLYOFNOQVPJJNP-UHFFFAOYSA-N"),
    ("inchikey", "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"),
    ("formula", "H2O"),
    ("formula", "C2H6O"),
    ("element", "C"),
    ("element", "O"),
    ("element", 26),
    ("element", "iron"),
    ("name", "chemworld-no-such-reference-molecule-22b2"),
)


def canonical_cas(text):
    if re.fullmatch(r"[1-9][0-9]{1,6}-[0-9]{2}-[0-9]", text) is None:
        return None
    sequence = text.replace("-", "")
    checksum = (
        sum(int(sequence[i]) * (len(sequence) - i - 1) for i in range(len(sequence) - 1)) % 10
    )
    return text if checksum == int(sequence[-1]) else None


def numeric_mass(text):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def same_number(actual, expected, *, atol=1e-15, rtol=2e-14):
    if expected is None:
        assert actual is None
        return 0
    assert actual is not None
    assert math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol), (actual, expected)
    return abs(actual - expected)


def read_sources(directory):
    references = {}
    for source, filename, count in SOURCES:
        with (directory / filename).open(encoding="utf-8", newline="") as stream:
            if filename.endswith(".json"):
                rows = [json.load(stream)]
            else:
                rows = list(csv.reader(stream, delimiter="\t"))
                if source == "mixtures":
                    assert rows.pop(0)[:3] == ["Primary Name", "Source", "N components"]
        assert len(rows) == count, (source, len(rows), count)
        references[source] = rows
    return references


def compound_unit(database, source, number, row):
    record = database.records(source)[number - 1]
    assert record.record_id == f"{source}:{number}"
    assert record.raw_fields == tuple(row)
    assert (
        record.formula,
        record.smiles,
        record.inchi,
        record.inchi_key,
        record.iupac_name,
        record.common_name,
        record.synonyms,
    ) == (row[2], row[4], row[5], row[6], row[7], row[8], tuple(row[9:]))
    assert record.registry_key == row[1]
    assert record.cas_number == canonical_cas(row[1])
    cid = int(row[0]) if row[0].isascii() and row[0].isdigit() else None
    assert record.pubchem_id == cid
    mass_error = same_number(record.molecular_weight_g_mol, numeric_mass(row[3]))
    queries = [
        ("source_key", row[1]),
        ("formula", row[2]),
        ("smiles", row[4]),
        ("inchi", row[5]),
        ("inchikey", row[6]),
    ]
    queries += [("name", name) for name in row[7:] if name.strip()]
    if record.cas_number:
        queries.append(("cas", row[1]))
    if cid is not None:
        queries.append(("pubchem", cid))
    checked = 0
    for kind, key in queries:
        if key == "":
            continue
        hits = database.search(key, kind=kind, source_ids=(source,))
        assert any(hit.record_id == record.record_id for hit in hits), (kind, key, record.record_id)
        checked += 1
    return {"query_count": checked, "mass_absolute_error": mass_error}


def auxiliary_unit(database, source, number, row):
    actual = database.records(source)[number - 1]
    if source.endswith("-preferences"):
        assert actual.preferred_keys == tuple(row["preferred_cas"])
        assert actual.unpreferred_keys == tuple(row["unpreferred_cas"])
        return {
            "preferred": len(actual.preferred_keys),
            "unpreferred": len(actual.unpreferred_keys),
        }
    if source == "dippr-2014":
        assert actual == row[0]
        assert row[0] in database.dippr_compounds()
        return {}
    assert actual.raw_fields == tuple(row)
    if source == "fake-registry":
        assert canonical_cas(actual.registry_key) is None
        assert actual in database.fake_records(row[0])
        return {}
    count = int(row[2])
    assert actual.component_count == count
    assert actual.registry_keys == tuple(row[3 : 3 + count])
    assert actual.component_names == tuple(row[3 + count : 3 + 2 * count])
    for values, offset in ((actual.mass_fractions, 2), (actual.mole_fractions, 3)):
        for value, reference in zip(
            values, row[3 + offset * count : 3 + (offset + 1) * count], strict=True
        ):
            same_number(value, float(reference))
    assert database.mixture(row[0]).raw_fields == tuple(row)
    return {"components": count}


def expected_query(references, kind, identity):
    if kind == "cas" and isinstance(identity, int):
        digits = str(identity)
        identity = f"{digits[:-3]}-{digits[-3:-1]}-{digits[-1]}"
    if kind == "inchi":
        identity = identity.split("/", 1)[1]
    answer = set()
    for source, _, _ in SOURCES[:6]:
        for number, row in enumerate(references[source], 1):
            if kind == "name":
                hit = " ".join(identity.lower().split()) in {
                    " ".join(value.lower().split()) for value in row[7:]
                }
            elif kind == "cas":
                hit = row[1] == identity
            elif kind == "pubchem":
                hit = row[0] == str(identity)
            else:
                field = {"formula": 2, "smiles": 4, "inchi": 5, "inchikey": 6}[kind]
                # Published formulas use Hill order. The two fixed queries also
                # accept the equivalent elementary expanded forms explicitly.
                variants = (
                    (
                        {"H2O", "OH2"}
                        if identity == "H2O"
                        else {"C2H6O", "C2H5OH", "CH3OCH3"}
                        if identity == "C2H6O"
                        else {identity}
                    )
                    if kind == "formula"
                    else {identity}
                )
                hit = row[field] in variants
            if hit:
                answer.add(f"{source}:{number}")
    return answer


def query_unit(database, references, kind, identity):
    if kind == "element":
        expected = {"C": ("C", 6), "O": ("O", 8), 26: ("Fe", 26), "iron": ("Fe", 26)}
        actual = database.element(identity)
        assert (actual.symbol, actual.atomic_number) == expected[identity]
        return {"element": actual.symbol}
    expected = expected_query(references, kind, identity)
    actual = database.search(identity, kind=kind)
    assert {record.record_id for record in actual} == expected, (
        "missing",
        sorted(expected - {r.record_id for r in actual}),
        "extra",
        sorted({r.record_id for r in actual} - expected),
    )
    return {"candidate_count": len(actual)}


def component_unit(database, references, name):
    rows = [r for r in references["pubchem-small"] if name in (r[7], r[8], *r[9:])]
    assert rows and len({r[1] for r in rows}) == 1
    component = database.component(name, source_ids=("pubchem-small",))
    row = rows[0]
    assert (component.cas_number, component.formula) == (row[1], row[2])
    assert component.charge == 0
    error = same_number(component.molecular_weight_g_mol, float(row[3]), atol=1e-12)
    return {"cas": component.cas_number, "mass_absolute_error": error}


def mixture_unit(database, references, name, basis):
    row = next(r for r in references["mixtures"] if r[0].strip() == name)
    count = int(row[2])
    keys = row[3 : 3 + count]
    masses = []
    for cas in keys:
        records = [r for r in references["pubchem-small"] if r[1] == cas]
        assert len(records) == 1
        masses.append(float(records[0][3]))
    offset = 2 if basis == "mass" else 3
    raw = [float(v) for v in row[3 + offset * count : 3 + (offset + 1) * count]]
    selected = [v / sum(raw) for v in raw]
    opposite = (
        [w / mass for w, mass in zip(selected, masses, strict=True)]
        if basis == "mass"
        else [z * mass for z, mass in zip(selected, masses, strict=True)]
    )
    opposite = [v / sum(opposite) for v in opposite]
    expected_z, expected_w = (opposite, selected) if basis == "mass" else (selected, opposite)
    actual = database.mixture_spec(
        name,
        basis=basis,
        phase="unknown",
        temperature_K=298.15,
        pressure_Pa=101325,
        source_ids=("pubchem-small",),
    )
    assert actual.component_ids == tuple(keys)
    errors = [
        same_number(v, ref, atol=1e-12, rtol=0)
        for values, refs in (
            (actual.mole_fractions, expected_z),
            (actual.mass_fractions, expected_w),
        )
        for v, ref in zip(values, refs, strict=True)
    ]
    mean_mass = sum(z * mass for z, mass in zip(expected_z, masses, strict=True))
    mass_error = same_number(actual.average_molecular_weight_g_mol, mean_mass, atol=1e-12)
    other_offset = 3 if basis == "mass" else 2
    source_other = [
        float(v) for v in row[3 + other_offset * count : 3 + (other_offset + 1) * count]
    ]
    return {
        "max_fraction_error": max(errors),
        "mass_absolute_error": mass_error,
        "source_opposite_basis_difference": max(
            abs(v - ref) for v, ref in zip(opposite, source_other, strict=True)
        ),
    }


def run(directory, output):
    if output.exists():
        raise FileExistsError("preserve previous results; choose a new output")
    started = time.monotonic()
    state = {"stage": "read_reference", "completed": 0, "failed": 0}
    stop = threading.Event()

    def progress():
        elapsed = time.monotonic() - started
        rate = state["completed"] / max(elapsed, 1e-9)
        eta = (79903 - state["completed"]) / rate if rate else None
        print(
            f"identifiers {state['stage']} {state['completed']}/79903 "
            f"failed={state['failed']} rate={rate:.1f}/s ETA={eta}",
            flush=True,
        )

    def heartbeat():
        while not stop.wait(30):
            progress()

    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    results = []

    def attempt(category, identity, evaluate):
        unit = {"category": category, "identity": identity}
        try:
            unit.update(status="passed", metrics=evaluate())
        except Exception:
            unit.update(status="failed", failure=traceback.format_exc())
            state["failed"] += 1
        results.append(unit)
        state["completed"] += 1
        if state["completed"] % 5000 == 0:
            progress()

    try:
        references = read_sources(directory)
        database = IdentifierDatabase(directory)
        for source, _, _ in SOURCES:
            state["stage"] = source
            progress()
            for number, row in enumerate(references[source], 1):
                evaluate = (
                    compound_unit if source in {s[0] for s in SOURCES[:6]} else auxiliary_unit
                )
                attempt(
                    "source",
                    f"{source}:{number}",
                    lambda s=source, n=number, r=row, fn=evaluate: fn(database, s, n, r),
                )
        state["stage"] = "queries"
        for kind, identity in QUERIES:
            attempt(
                "query",
                f"{kind}:{identity}",
                lambda k=kind, i=identity: query_unit(database, references, k, i),
            )
        state["stage"] = "consumers"
        for name in ("water", "ethanol", "sodium chloride", "acetic acid"):
            attempt("component", name, lambda n=name: component_unit(database, references, n))
        for name in ("Air", "R512A"):
            for basis in ("mass", "mole"):
                attempt(
                    "mixture",
                    f"{name}:{basis}",
                    lambda n=name, b=basis: mixture_unit(database, references, n, b),
                )
        assert len(results) == 79903
        summary = {
            "mode": "development",
            "block": "NATIVE-CONV-22B2",
            "reference": "chemicals 1.5.2 explicit Identifiers data directory",
            "source_directory": str(directory.resolve()),
            "denominators": {
                "source_records": 79871,
                "queries": 24,
                "consumers": 8,
                "total": 79903,
            },
            "source_counts": {source: count for source, _, count in SOURCES},
            "passed": 79903 - state["failed"],
            "failed": state["failed"],
            "skipped": 0,
            "elapsed_seconds": time.monotonic() - started,
            "units": results,
            "failures": [unit for unit in results if unit["status"] == "failed"],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        progress()
        print(
            json.dumps({k: v for k, v in summary.items() if k not in {"units", "failures"}}),
            flush=True,
        )
        return bool(state["failed"])
    finally:
        stop.set()
        worker.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source_directory, args.output))
