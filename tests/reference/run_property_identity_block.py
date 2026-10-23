"""Inventory every original F2 identity gap; alias candidates are not resolutions."""

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import chemicals

from chemworld.data.identifiers import IdentifierDatabase

SOURCES = ("example-user", "inorganic", "anion", "cation", "pubchem-small", "pubchem-large")


def inventory(record, query):
    return {
        "record_id": record.record_id,
        "source_id": record.source_id,
        "source": record.source.provenance.to_dict(),
        "primary_CAS": record.cas_number,
        "formula": record.formula,
        "molecular_weight_g_mol": record.molecular_weight_g_mol,
        "smiles": record.smiles,
        "inchi": record.inchi,
        "inchi_key": record.inchi_key,
        "pubchem_id": record.pubchem_id,
        "iupac_name": record.iupac_name,
        "common_name": record.common_name,
        "query": query,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = json.loads(args.prior_summary.read_text(encoding="utf8"))
    assert prior["block"] == "sources-consumers" and prior["total"] == 17268
    gaps = [row for row in prior["results"] if "component_gap" in row.get("metrics", {})]
    counts = Counter(row["inputs"]["source"] for row in gaps)
    assert counts == {"CRC_PERMITTIVITY": 64, "CRC_RI": 221} and len(gaps) == 285
    assert len({row["unit"] for row in gaps}) == 285
    args.output.mkdir(parents=True, exist_ok=False)
    db = IdentifierDatabase(Path(chemicals.__file__).parent / "Identifiers")
    results = []
    started = time.perf_counter()
    for index, original in enumerate(gaps, 1):
        raw = original["inputs"]["raw"]
        cas = raw["CAS"]
        row = {
            "unit": original["unit"],
            "source": original["inputs"]["source"],
            "source_CAS": cas,
            "source_name": raw.get("Chemical"),
            "original_gap": original["metrics"]["component_gap"],
        }
        try:
            candidates = []
            queries = [("primary_CAS", cas, "cas"), ("CAS_alias", cas, "name")]
            if row["source_name"] is not None:
                queries.append(("source_name", row["source_name"], "name"))
            row["unavailable_queries"] = [] if row["source_name"] is not None else ["source_name"]
            for label, identity, kind in queries:
                candidates.extend(
                    inventory(record, label)
                    for record in db.search(identity, kind=kind, source_ids=SOURCES)
                )
            status = (
                "primary_CAS_present"
                if any(c["query"] == "primary_CAS" for c in candidates)
                else "unverified_candidates"
                if candidates
                else "no_candidates"
            )
            row.update(status=status, candidates=candidates)
        except Exception:
            row.update(status="query_failed", failure=traceback.format_exc())
        results.append(row)
        if index % 25 == 0 or index == len(gaps):
            (args.output / "partial.json").write_text(
                json.dumps(
                    {"total": len(gaps), "completed": index, "results": results},
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf8",
            )
            elapsed = time.perf_counter() - started
            print(
                f"{index}/{len(gaps)} identity rows; {index / elapsed:.2f}/s "
                f"ETA {elapsed * (len(gaps) - index) / index:.1f}s",
                flush=True,
            )
    result = {
        "task": "NATIVE-CONV-22G1",
        "mode": "development",
        "block": "original-285-identity-inventory",
        "reference_version": chemicals.__version__,
        "source_counts": dict(counts),
        "source_CAS_count": len({r["source_CAS"] for r in results}),
        "total": len(gaps),
        "completed": len(results),
        "states": dict(Counter(r["status"] for r in results)),
        "new_verified_mappings": 0,
        "scope": (
            "Alias/name candidates require independent identity evidence; "
            "no source or component changed"
        ),
        "seconds": time.perf_counter() - started,
        "results": results,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
    return bool(result["states"].get("query_failed", 0))


if __name__ == "__main__":
    raise SystemExit(main())
