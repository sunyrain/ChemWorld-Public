"""Eight native consumers with explicitly bound existing data sources.

This is a separate block; the original small-only diagnostic remains intact.
"""

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

from chemworld.data.identifiers import IdentifierDatabase

BINDINGS = {
    "7732-18-5": "inorganic",
    "7727-37-9": "inorganic",
    "7782-44-7": "inorganic",
    "64-17-5": "pubchem-small",
    "7647-14-5": "pubchem-small",
    "64-19-7": "pubchem-small",
    "7440-37-1": "pubchem-small",
    "811-97-2": "pubchem-small",
    "75-37-6": "pubchem-small",
}


def check_close(actual, expected, *, atol=1e-12, rtol=2e-14):
    assert math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), (actual, expected)
    return abs(actual - expected)


def run(directory, output):
    if output.exists():
        raise FileExistsError("use a new output; preserve previous block results")
    references = {}
    for source, filename in (
        ("pubchem-small", "chemical identifiers pubchem small.tsv"),
        ("inorganic", "Inorganic db.tsv"),
    ):
        with (directory / filename).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream, delimiter="\t"))
        for key, selected_source in BINDINGS.items():
            if selected_source == source:
                matches = [row for row in rows if row[1] == key]
                assert len(matches) == 1, (key, selected_source)
                references[key] = matches[0]
    with (directory / "Mixtures Compositions.tsv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))[1:]
    mixtures = {row[0].strip(): row for row in rows}
    db = IdentifierDatabase(directory)

    def component(cas):
        actual = db.component(cas, kind="cas", source_ids=(BINDINGS[cas],))
        expected = references[cas]
        assert (actual.cas_number, actual.formula, actual.charge) == (cas, expected[2], 0)
        error = check_close(actual.molecular_weight_g_mol, float(expected[3]))
        return {"cas": cas, "source": BINDINGS[cas], "mass_absolute_error": error}

    def mixture(name, basis):
        expected = mixtures[name]
        count = int(expected[2])
        keys = expected[3 : 3 + count]
        masses = [float(references[key][3]) for key in keys]
        source_w = [float(value) for value in expected[3 + count * 2 : 3 + count * 3]]
        source_z = [float(value) for value in expected[3 + count * 3 : 3 + count * 4]]
        if basis == "mass":
            ws = [v / sum(source_w) for v in source_w]
            zs = [w / m for w, m in zip(ws, masses, strict=True)]
            zs = [v / sum(zs) for v in zs]
            source_difference = max(abs(z - ref) for z, ref in zip(zs, source_z, strict=True))
        else:
            zs = [v / sum(source_z) for v in source_z]
            ws = [z * m for z, m in zip(zs, masses, strict=True)]
            ws = [v / sum(ws) for v in ws]
            source_difference = max(abs(w - ref) for w, ref in zip(ws, source_w, strict=True))
        actual = db.mixture_spec(
            name,
            basis=basis,
            phase="unknown",
            temperature_K=298.15,
            pressure_Pa=101325,
            source_ids=("pubchem-small", "inorganic"),
        )
        assert actual.component_ids == tuple(keys)
        errors = [
            check_close(a, b, rtol=0)
            for values, refs in ((actual.mass_fractions, ws), (actual.mole_fractions, zs))
            for a, b in zip(values, refs, strict=True)
        ]
        mass_error = check_close(
            actual.average_molecular_weight_g_mol,
            sum(z * m for z, m in zip(zs, masses, strict=True)),
        )
        return {
            "sources": {key: BINDINGS[key] for key in keys},
            "max_fraction_error": max(errors),
            "mass_absolute_error": mass_error,
            "source_opposite_basis_difference": source_difference,
        }

    units = [
        (cas, lambda key=cas: component(key))
        for cas in ("7732-18-5", "64-17-5", "7647-14-5", "64-19-7")
    ]
    units += [
        (f"{name}:{basis}", lambda n=name, b=basis: mixture(n, b))
        for name in ("Air", "R512A")
        for basis in ("mass", "mole")
    ]
    started = time.monotonic()
    results = []
    for identity, evaluate in units:
        item = {"identity": identity}
        try:
            item.update(status="passed", metrics=evaluate())
        except Exception:
            item.update(status="failed", failure=traceback.format_exc())
        results.append(item)
        print(
            f"bound identifier consumers {len(results)}/8 "
            f"failed={sum(r['status'] == 'failed' for r in results)}",
            flush=True,
        )
    failures = [item for item in results if item["status"] == "failed"]
    summary = {
        "block": "22B2-explicit-source-consumers",
        "mode": "development",
        "reference_version": "chemicals 1.5.2",
        "source_bindings": BINDINGS,
        "total": 8,
        "passed": 8 - len(failures),
        "failed": len(failures),
        "skipped": 0,
        "elapsed_seconds": time.monotonic() - started,
        "units": results,
        "failures": failures,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key not in {"units", "failures"}}
        ),
        flush=True,
    )
    return bool(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.source_directory, args.output))
