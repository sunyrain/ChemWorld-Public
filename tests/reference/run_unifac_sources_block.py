"""Complete fixed UNIFAC source ingestion and numerical parameter consumption."""

import argparse
import json
import time
from pathlib import Path

import thermo
import thermo.unifac as reference
from run_activity_thermodynamics_block import compare, new_record

from chemworld.data.unifac_sources import (
    iter_ddbst_sqlite,
    iter_ddbst_text,
    read_unifac_interactions,
    read_unifac_subgroups,
    select_unifac_interactions,
    unifac_model_from_assignments,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.equilibrium import activity_coefficients
from chemworld.physchem.unifac import UNIFACSubgroup, unifac_activity_model, unifac_psi

TABLES = (
    ("UNIFAC original interaction parameters.tsv", 0, 1270),
    ("UNIFAC LLE interaction parameters.tsv", 0, 512),
    ("UNIFAC modified Dortmund interaction parameters 2006.tsv", 1, 1318),
    ("UNIFAC modified Dortmund interaction parameters.tsv", 1, 1516),
    ("UNIFAC modified NIST 2015 interaction parameters.tsv", 1, 1969),
    ("PSRK interaction parameters.tsv", 2, 1912),
    ("UNIFAC Lyngby interaction parameters.tsv", 4, 307),
    ("NIST KT 2011 interaction parameters.tsv", 5, 1436),
    ("UNIFAC 2.0 interaction parameters.tsv", 0, 2916),
    ("UNIFAC 2.0 Dortmund interaction parameters.tsv", 1, 3969),
    ("VTPR 2012 interaction parameters.tsv", 3, 130),
    ("VTPR 2014 interaction parameters.tsv", 3, 384),
    ("VTPR 2016 interaction parameters.tsv", 3, 50),
)
GROUPS = {
    "UFSG": 113,
    "LLEUFSG": 57,
    "DOUFSG": 124,
    "NISTUFSG": 201,
    "PSRKSG": 150,
    "VTPRSG": 108,
    "LUFSG": 45,
    "NISTKTUFSG": 138,
}


def source(name):
    return DataSourceProvenance(
        name,
        0,
        "thermo 0.6.0 distributed UNIFAC source: " + name,
        version="0.6.0",
        license="upstream source attribution retained; local development ingestion",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(thermo.__file__).parent / "Phase Change"
    result = {
        "block": "23C-UNIFAC-sources",
        "mode": "development",
        "units": [],
        "counts": {},
        "source_files": [],
    }
    start = time.monotonic()
    group_sets = {}
    for name, count in GROUPS.items():
        upstream = getattr(reference, name)
        path = args.output / (name + ".json")
        payload = {
            "source": source(name).to_dict(),
            "subgroups": [
                {
                    "group_id": key,
                    "main_group_id": v.main_group_id,
                    "name": v.group,
                    "main_group_name": v.main_group,
                    "R": v.R,
                    "Q": v.Q,
                }
                for key, v in upstream.items()
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        native, provenance = read_unifac_subgroups(path)
        group_sets[name] = (native, provenance)
        result["counts"][name] = {"expected": count, "actual": len(native)}
        for key, g in native.items():
            rec = new_record("subgroup", name, 1, row=key)
            result["units"].append(rec)
            ref = upstream[key]
            compare(
                rec,
                {
                    "main": g.main_group_id,
                    "R": g.R,
                    "Q": g.Q,
                    "roundtrip": float(UNIFACSubgroup(**g.to_dict()) == g),
                    "name": float(g.name == ref.group),
                    "main_name": float(g.main_group_name == ref.main_group),
                },
                {
                    "main": ref.main_group_id,
                    "R": ref.R,
                    "Q": ref.Q,
                    "roundtrip": 1.0,
                    "name": 1.0,
                    "main_name": 1.0,
                },
            )
    revisions = {}
    done = 0
    for filename, version, count in TABLES:
        path = root / filename
        records = read_unifac_interactions(path, source=source(filename))
        revisions[filename] = records
        result["counts"][filename] = {"expected": count, "actual": len(records)}
        result["source_files"].append(str(path))
        raw = path.read_text(encoding="utf-8").splitlines()
        for row in records:
            rec = new_record(
                "interaction",
                filename,
                2,
                row=row.row,
                main_groups=[row.left, row.right],
                same_main=row.left == row.right,
            )
            result["units"].append(rec)
            try:
                left, right = row.left, row.right
                groups = {
                    1: UNIFACSubgroup(1, left, 0.0 if version == 3 else 1.0, 1.0),
                    2: UNIFACSubgroup(2, right, 0.0 if version == 3 else 1.0, 1.0),
                }
                interactions = {(right, left): (0.0, 0.0, 0.0), (left, right): row.coefficients}
                model = unifac_activity_model(
                    "source-probe",
                    {"A": {1: 1}, "B": {2: 1}},
                    subgroups=groups,
                    interactions=interactions,
                    version=version,
                    provenance={
                        "source": row.to_dict(),
                        "reverse": "authored zero for this source-row probe",
                    },
                )
                data = {
                    left: {right: row.coefficients[0] if version == 0 else row.coefficients},
                    right: {left: 0.0 if version == 0 else (0.0, 0.0, 0.0)},
                }
                if left == right:
                    data = {}
                ref = reference.UNIFAC.from_subgroups(
                    T=320.0,
                    xs=[0.35, 0.65],
                    chemgroups=[{1: 1}, {2: 1}],
                    version=version,
                    subgroups={
                        key: reference.UNIFAC_subgroup(
                            key, str(key), g.main_group_id, "main", g.R, g.Q
                        )
                        for key, g in groups.items()
                    },
                    interaction_data=data,
                )
                gamma = activity_coefficients(model, {"A": 0.35, "B": 0.65}, temperature_K=320.0)
                fields = raw[row.row - 1].split("\t")
                target_coefficients = list(map(float, fields[2:5]))
                if len(target_coefficients) == 1:
                    target_coefficients += [0.0, 0.0]
                compare(
                    rec,
                    {
                        "coefficients": row.coefficients,
                        "psi": 1.0
                        if left == right
                        else unifac_psi(version, 320.0, *row.coefficients),
                        "gamma": list(gamma.values()),
                    },
                    {
                        "coefficients": target_coefficients,
                        "psi": ref.psis()[0][1],
                        "gamma": ref.gammas(),
                    },
                )
            except Exception as exc:
                rec["failures"].append({"exception": f"{type(exc).__name__}: {exc}"})
            done += 1
            if done % 1000 == 0:
                elapsed = max(time.monotonic() - start, 1e-9)
                print(
                    f"UNIFAC source equations {done}/17689; {done / elapsed:.2f}/s; "
                    f"ETA {(17689 - done) * elapsed / done:.1f}s",
                    flush=True,
                )
        (args.output / "partial.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
    selected, selection = select_unifac_interactions(
        [revisions[name] for name, _, _ in TABLES[-3:]], later_revisions_override=True
    )
    result["vtpr_revision_selection"] = {
        "raw_rows": 564,
        "selected": len(selected),
        "superseded": selection["superseded"],
    }
    reference.load_group_assignments_DDBST()
    reference_assignments = {
        method: getattr(reference, "DDBST_" + method + "_assignments")
        for method in ("UNIFAC", "MODIFIED_UNIFAC", "PSRK")
    }
    for kind, expected_count, rows in (
        (
            "text",
            31778,
            iter_ddbst_text(root / "DDBST UNIFAC assignments.tsv", source=source("DDBST text")),
        ),
        (
            "sqlite",
            31754,
            iter_ddbst_sqlite(
                root / "DDBST_UNIFAC_assignments.sqlite", source=source("DDBST SQLite")
            ),
        ),
    ):
        count = 0
        valid = dict.fromkeys(reference_assignments, 0)
        for count, records in enumerate(rows, 1):
            rec = new_record("assignment", kind, 1, row=count, identity=records[0].identity)
            result["units"].append(rec)
            for record in records:
                try:
                    target = (
                        reference_assignments[record.method].get(record.identity, {})
                        if kind == "text"
                        else reference.UNIFAC_group_assignment_DDBST(record.identity, record.method)
                    )
                    # Text preserves invalid authored fragments; the reference
                    # lookup exposes only valid ones.
                    compare(
                        rec,
                        {
                            record.method + "_valid": float(record.valid),
                            record.method + "_groups": float(dict(record.groups) == target)
                            if record.valid
                            else 1.0,
                        },
                        {
                            record.method + "_valid": float(bool(target)),
                            record.method + "_groups": 1.0,
                        },
                    )
                    valid[record.method] += int(record.valid)
                except Exception as exc:
                    rec["failures"].append({"method": record.method, "exception": str(exc)})
            if count % 5000 == 0:
                print(f"DDBST {kind} {count}/{expected_count}; valid {valid}", flush=True)
        result["counts"]["DDBST_" + kind] = {
            "expected": expected_count,
            "actual": count,
            "valid": valid,
        }
    for version, method, groups_name, file_name in (
        (0, "UNIFAC", "UFSG", TABLES[0][0]),
        (1, "MODIFIED_UNIFAC", "DOUFSG", TABLES[3][0]),
        (2, "PSRK", "PSRKSG", TABLES[5][0]),
    ):
        records = {
            cas: next(
                iter_ddbst_sqlite(
                    root / "DDBST_UNIFAC_assignments.sqlite",
                    source=source("DDBST SQLite"),
                    cas_number=cas,
                )
            )[version]
            for cas in ("64-17-5", "110-54-3")
        }
        groups, provenance = group_sets[groups_name]
        rec = new_record("molecular_consumer", method, 2)
        result["units"].append(rec)
        try:
            model = unifac_model_from_assignments(
                "ddbst-mixture",
                records,
                subgroups=groups,
                subgroup_source=provenance,
                revisions=(revisions[file_name],),
                version=version,
            )
            gamma = activity_coefficients(
                model, dict(zip(records, (0.35, 0.65), strict=True)), temperature_K=320.0
            )
            ref = reference.UNIFAC.from_subgroups(
                T=320.0,
                xs=[0.35, 0.65],
                chemgroups=[dict(r.groups) for r in records.values()],
                subgroups=getattr(reference, groups_name),
                version=version,
            )
            compare(rec, list(gamma.values()), ref.gammas())
        except Exception as exc:
            rec["failures"].append({"exception": str(exc)})
    result["comparisons"] = sum(r["comparisons"] for r in result["units"])
    result["failures"] = [
        {"unit": i, "kind": r["kind"], "model": r["model"], "row": r.get("row"), **f}
        for i, r in enumerate(result["units"])
        for f in r["failures"]
    ]
    result["elapsed_seconds"] = time.monotonic() - start
    result["passed"] = (
        not result["failures"]
        and all(v["expected"] == v["actual"] for v in result["counts"].values())
        and len(selected) == 552
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps({k: result[k] for k in ("comparisons", "passed", "elapsed_seconds")}), flush=True
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
