"""Two explicit identities recovered from a CRC diagram and primary page excerpts.

This is reference qualification code, not a runtime identifier resolver. Raw
documents and the unchanged web-connector response stay outside the repository.
"""

import argparse
import json
import re
import subprocess
import time
import traceback
from pathlib import Path

CRC_URL = "https://s3.amazonaws.com/thm-monocle-interactive/0PC9TGZOtD/file.pdf"
SIGMA_URL = "https://www.sigmaaldrich.com/US/en/product/aldrich/278777"
JST_URL = "https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=200907066237312881"
ROUTE = "reviewed_primary_document_structure"


def crc_trans_mol():
    """Transcribe CRC drawing 684: OH solid wedge, adjacent methyl dashed wedge.

    Atoms 1..6 traverse the drawn ring clockwise from its top carbon. Atoms 7
    and 8 are OH and methyl; a zero chiral flag does not select an enantiomer.
    The table's explicit (+/-) label supplies the racemic material scope.
    """
    atoms = [
        (0.0, 1.0, "C"), (0.866, 0.5, "C"), (0.866, -0.5, "C"),
        (0.0, -1.0, "C"), (-0.866, -0.5, "C"), (-0.866, 0.5, "C"),
        (0.0, 1.8, "O"), (1.559, 0.9, "C"),
    ]
    bonds = [(1, 2, 0), (2, 3, 0), (3, 4, 0), (4, 5, 0), (5, 6, 0),
             (6, 1, 0), (1, 7, 1), (2, 8, 6)]
    lines = ["CRC 684 trans-2-methylcyclohexanol", "  ChemWorld 2D", "",
             "  8  8  0  0  0  0            999 V2000"]
    lines += [f"{x:10.4f}{y:10.4f}{0.0:10.4f} {element:<3}"
              " 0  0  0  0  0  0  0  0  0  0  0  0" for x, y, element in atoms]
    lines += [f"{a:3d}{b:3d}{1:3d}{stereo:3d}  0  0  0" for a, b, stereo in bonds]
    return "\n".join([*lines, "M  END", ""])


def read_crc(directory):
    assert (directory / "crc-physical-constants.pdf").read_bytes().startswith(b"%PDF-")
    pages = (directory / "crc-physical-constants.txt").read_text(encoding="utf8").split("\f")
    assert "10,867" in pages[0] and "structure diagrams" in pages[0]
    # The layout extraction includes superscript temperature immediately after nD.
    match = re.search(
        r"684\s+trans-2-Methylcyclohexa-\s+C7H14O\s+(615-39-4)\s+"
        r"(114\.185)\s+liq[^\n]+(1\.4616)20[^\n]*\n\s*nol,\s*\(±\)-", pages[35]
    )
    assert match is not None
    assert "684" in pages[36] and "trans-2-Methylcyclohexanol, (±)-" in pages[36]
    return {"CAS": match[1], "formula": "C7H14O", "molecular_weight_g_mol": float(match[2]),
            "refractive_index": float(match[3]), "temperature_K": 293.15,
            "name": "trans-2-Methylcyclohexanol, (±)-", "table_entry": 684,
            "table_page": 36, "diagram_page": 37, "url": CRC_URL,
            "diagram_review": "adjacent OH solid wedge and methyl dashed wedge"}


def read_silane(directory):
    """Read each primary page's own saved identity excerpt, retaining access limits."""
    record = json.loads((directory / "indexed-primary-pages.json").read_text(encoding="utf8"))
    assert not record.get("isError")
    text = "\n".join(c["text"] for c in record["content"] if c["type"] == "text")
    fields = []
    for url in (SIGMA_URL, JST_URL):
        sections = [s for s in text.split("\n--------------------------------") if f"({url})" in s]
        assert len(sections) == 1
        page = sections[0]
        assert "998-41-4" in page and "Tributylsilane" in page
        assert "CCCC[SiH](CCCC)CCCC" in page
        values = set(re.findall(r"(?:InChI=)?(1S/C12H28Si/[^\s]+)", page))
        keys = set(re.findall(r"\bHIAZFYQNGXRLTF-UHFFFAOYSA-N\b", page))
        assert len(values) == len(keys) == 1
        inchi = "InChI=" + values.pop()
        assert "/h13H,4-12H2,1-3H3" in inchi
        fields.append({"url": url, "CAS": "998-41-4", "formula": "C12H28Si",
                       "inchi": inchi, "inchi_key": keys.pop(),
                       "smiles": "CCCC[SiH](CCCC)CCCC"})
    assert {f["inchi"] for f in fields} == {fields[0]["inchi"]}
    assert "[CH_{3}(CH_{2})_{3}]_{3}SiH" in text
    return {"CAS": "998-41-4", "formula": "C12H28Si", "name": "Tributylsilane",
            "molecular_weight_g_mol": 200.44, "pages": fields,
            "source_access": "saved_search_connector_indexed_primary_page_excerpts",
            "direct_download": "Sigma timed out; J-GLOBAL previously returned 503"}


def index_nist_sources(directory):
    """Extract identity fields from every saved query; no hit is not database absence."""
    plan = json.loads((directory / "query-plan.json").read_text(encoding="utf8"))
    assert plan["total"] == len(plan["CAS"]) == len(set(plan["CAS"])) == 52
    results = {cas: [] for cas in plan["CAS"]}
    batches, ignored = [], set()
    for number in range(1, 14):
        path = directory / f"batch-{number}.json"
        batch = json.loads(path.read_text(encoding="utf8"))
        assert batch["CAS"] == plan["CAS"][(number - 1) * 4:number * 4]
        response = batch["response"]
        if response["status"] != "fulfilled" or response["value"].get("isError"):
            batches.append({"batch": number, "CAS": batch["CAS"], "status": "failed"})
            continue
        batches.append({"batch": number, "CAS": batch["CAS"], "status": "received"})
        text = "\n".join(c["text"] for c in response["value"]["content"] if c["type"] == "text")
        for section in re.split(r"\n-{25,}\n", text):
            cas_match = re.search(r"^\s*\* CAS Registry Number:\s*([\d-]+)", section, re.M)
            if cas_match is None:
                continue
            cas = cas_match[1]
            if cas not in batch["CAS"]:
                ignored.add(cas)
                continue
            url = re.search(r"\((https://webbook\.nist\.gov/[^\s]+)\)\n", section)
            name = re.search(r"^# (?!NIST Chemistry WebBook)(.+)$", section, re.M)
            if url is None or name is None:
                continue
            inchi = re.search(r"IUPAC Standard InChI:\s*(InChI=\S+)", section)
            key = re.search(r"IUPAC Standard InChIKey:\s*([A-Z-]+)", section)
            formula = re.search(r"^\s*\* Formula:\s*(\S+)", section, re.M)
            weight = re.search(r"Molecular weight:\s*([0-9.]+)", section)
            record = {"CAS": cas, "name": name[1], "url": url[1],
                      "formula": re.sub(r"[_{}]", "", formula[1]) if formula else None,
                      "molecular_weight_g_mol": float(weight[1]) if weight else None,
                      "inchi": inchi[1] if inchi else None, "inchi_key": key[1] if key else None,
                      "access": "saved_search_connector_indexed_NIST_page_excerpt"}
            if record not in results[cas]:
                results[cas].append(record)
    return {"task": "NATIVE-CONV-22G1", "mode": "development",
            "block": "all-52-remaining-CAS-indexed-NIST-identities", "total": 52,
            "queried": sum(len(b["CAS"]) for b in batches), "batches": batches,
            "exact_CAS_hits": sum(bool(v) for v in results.values()),
            "no_exact_identity_excerpt": sum(not v for v in results.values()),
            "ignored_wrong_CAS": sorted(ignored), "results": results,
            "automatic_material_approvals": 0,
            "scope": "Indexed source excerpts, not direct HTTP reads or database completeness."}


def read_documents(directory, encoding_directory, indexed_directory=None):
    crc, silane = read_crc(directory), read_silane(directory / "primary-documents-1")
    encoding = json.loads((encoding_directory / "summary.json").read_text(encoding="utf8"))
    assert encoding["total"] == encoding["encoded"] == 2 and encoding["failed"] == 0
    assert encoding["reference_version"] == "InChI 1.07.5"
    assert (encoding_directory / "crc-684.mol").read_text(encoding="utf8") == crc_trans_mol()
    for row in encoding["results"]:
        raw = (encoding_directory / f"{row['mode']}.txt").read_text(encoding="utf8")
        assert re.findall(r"^InChI=.+$", raw, re.M) == [row["inchi"]]
        assert re.findall(r"^InChIKey=(.+)$", raw, re.M) == [row["inchi_key"]]
    documents = {"615-39-4": {"source": crc, "encoding": encoding},
                 "998-41-4": {"source": silane}}
    if indexed_directory is not None:
        indexed = index_nist_sources(indexed_directory)
        candidates = indexed["results"]["16177-46-1"]
        assert candidates
        unique = {tuple(r[k] for k in ("CAS", "name", "formula", "molecular_weight_g_mol",
                                       "inchi", "inchi_key")) for r in candidates}
        assert len(unique) == 1
        documents["16177-46-1"] = {"source": candidates[0]}
    return documents


def check_document_identity(component, documents, pubchem):
    scope, external = component.metadata["identity_scope"], component.metadata["external_identity"]
    assert documents is not None and component.cas_number in documents
    source = documents[component.cas_number]["source"]
    assert component.cas_number == scope["source_CAS"] == source["CAS"]
    assert component.formula == source["formula"]
    assert component.molecular_weight_g_mol == source["molecular_weight_g_mol"]
    assert component.charge == 0 and source["name"] in component.aliases
    assert external["primary_document_fields"] == source
    original = next(r for r in pubchem["original_row_comparison"]
                    if r["source_CAS"] == component.cas_number)
    expected_candidates = [{k: c[k] for k in ("CID", "InChI", "InChIKey", "MolecularFormula")}
                           for c in original["candidates"]]
    assert external["conflicting_PubChem_candidates"] == expected_candidates
    if component.cas_number == "998-41-4":
        assert scope["material_kind"] == "single_achiral_compound"
        assert scope["stereochemistry"] == "ACHIRAL"
        assert component.composition == {"C": 12.0, "H": 28.0, "Si": 1.0}
        assert scope["source_units"] == ["CRC_RI/2776"]
        assert len(component.provenance) == 2
        for provenance, page in zip(component.provenance, source["pages"], strict=True):
            assert provenance.source_path == page["url"] and provenance.source_key == source["CAS"]
            assert component.inchi == page["inchi"] and component.inchi_key == page["inchi_key"]
        assert all(c["MolecularFormula"] == "C12H27Si" for c in expected_candidates)
    elif component.cas_number == "615-39-4":
        assert scope["source_units"] == ["CRC_RI/2072"]
        assert component.composition == {"C": 7.0, "H": 14.0, "O": 1.0}
        assert scope["stereochemistry"] == "RACEMIC" and scope["stereo_centers"] == 2
        assert scope["enantiomer_mole_fractions"] == [0.5, 0.5]
        assert scope["absolute_configuration"] == "not_selected"
        assert len(component.provenance) == 2
        assert component.provenance[0].source_path == CRC_URL
        assert component.provenance[0].source_key == "684:615-39-4"
        rows = {r["mode"]: r for r in documents[component.cas_number]["encoding"]["results"]}
        representative, racemic = rows["absolute"], rows["racemic"]
        assert component.inchi == racemic["inchi"] and component.inchi_key == racemic["inchi_key"]
        assert "/t6-,7-/s3" in component.inchi and "/m" not in component.inchi
        transformation = component.metadata["identity_transformation"]
        assert transformation["representative_InChI"] == representative["inchi"]
        assert transformation["representative_InChIKey"] == representative["inchi_key"]
        assert transformation["options"] == racemic["options"]
        assert all("/t" not in c["InChI"] for c in expected_candidates)
    else:
        assert component.cas_number == "16177-46-1"
        assert scope["source_units"] == ["CRC_RI/4189"]
        assert scope["stereochemistry"] == "ACHIRAL_RELATIVE_RING_GEOMETRY"
        assert scope["relative_geometry"] == "cis"
        assert scope["absolute_configuration"] == "not_applicable"
        assert component.composition == {"C": 8.0, "H": 12.0}
        assert component.inchi == source["inchi"] and component.inchi_key == source["inchi_key"]
        assert "/t7-,8+" in component.inchi and "/m" not in component.inchi
        assert "/s" not in component.inchi and source["name"] == "cis-1,2-Divinylcyclobutane"
        assert len(component.provenance) == 1
        assert component.provenance[0].source_path == source["url"]
        assert component.provenance[0].source_key == component.cas_number
        assert all("/t" not in c["InChI"] for c in expected_candidates)


def document_negative_variants(component):
    """Scientific identity substitutions, independent of the positive fixture values."""
    if component.cas_number == "16177-46-1":
        for mode in ("wrong_CAS", "generic_graph", "opposite_ring_geometry",
                     "imposed_absolute_configuration", "generic_key", "wrong_source_URL"):
            data = json.loads(json.dumps(component.to_dict()))
            if mode == "wrong_CAS":
                data["cas_number"] = "6553-48-6"
            elif mode == "generic_graph":
                data["inchi"] = data["inchi"].split("/t")[0]
            elif mode == "opposite_ring_geometry":
                data["inchi"] = data["inchi"].replace("/t7-,8+", "/t7-,8-")
            elif mode == "imposed_absolute_configuration":
                data["inchi"] += "/m0/s1"
            elif mode == "generic_key":
                data["inchi_key"] = "UHHCYAAVGADGGP-UHFFFAOYSA-N"
            else:
                data["provenance"][0]["source_path"] = "https://webbook.nist.gov/"
            yield mode, data
        return
    modes = (
        ("wrong_CAS", "lost_racemate_scope", "unequal_enantiomers", "absolute_representative",
         "missing_relative_layer", "wrong_relative_geometry", "wrong_key", "wrong_diagram_page",
         "relative_not_racemic", "imposed_absolute_configuration")
        if component.cas_number == "615-39-4" else
        ("wrong_CAS", "missing_formula_hydrogen", "missing_SiH_layer", "old_hydrogen_free_key",
         "wrong_molecular_weight", "unknown_mixture", "charged_material", "wrong_supplier_url")
    )
    for mode in modes:
        data = json.loads(json.dumps(component.to_dict()))
        scope = data["metadata"]["identity_scope"]
        if mode == "wrong_CAS":
            data["cas_number"] = "615-38-3"
        elif mode == "lost_racemate_scope":
            scope["stereochemistry"] = "ACHIRAL"
        elif mode == "unequal_enantiomers":
            scope["enantiomer_mole_fractions"] = [0.75, 0.25]
        elif mode == "absolute_representative":
            data["inchi"] = data["metadata"]["identity_transformation"]["representative_InChI"]
        elif mode == "missing_relative_layer":
            data["inchi"] = re.sub(r"/t[^/]+", "", data["inchi"])
        elif mode == "wrong_relative_geometry":
            data["inchi"] = data["inchi"].replace("/t6-,7-", "/t6-,7+")
        elif mode in ("wrong_key", "old_hydrogen_free_key"):
            data["inchi_key"] = "ISEIIPDWJVGTQS-UHFFFAOYSA-N"
        elif mode == "wrong_diagram_page":
            data["metadata"]["external_identity"]["primary_document_fields"]["diagram_page"] = 36
        elif mode == "relative_not_racemic":
            data["inchi"] = data["inchi"].replace("/s3", "/s2")
        elif mode == "imposed_absolute_configuration":
            scope["absolute_configuration"] = "R,R"
        elif mode == "missing_formula_hydrogen":
            data["formula"], data["molecular_weight_g_mol"] = "C12H27Si", None
        elif mode == "missing_SiH_layer":
            data["inchi"] = data["inchi"].replace("/h13H,", "/h")
        elif mode == "wrong_molecular_weight":
            data["molecular_weight_g_mol"] = 199.43
        elif mode == "unknown_mixture":
            scope["material_kind"] = "mixture_unknown_composition"
        elif mode == "charged_material":
            data["charge"] = 1
        else:
            data["provenance"][0]["source_path"] = JST_URL
        yield mode, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--inchi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started, rows = time.perf_counter(), []
    try:
        source = read_crc(args.documents)
        source_file = args.output / "crc-684.mol"
        source_file.write_text(crc_trans_mol(), encoding="utf8")
        for mode in ("absolute", "racemic"):
            row = {"mode": mode, "source_CAS": source["CAS"]}
            try:
                options = ["/Key", "/AuxNone", "/NoLabels"]
                if mode == "racemic":
                    options.append("/SRac")
                command = [str(args.inchi), str(source_file)] + [
                    str(args.output / f"{mode}.{suffix}") for suffix in ("txt", "log", "prb")
                ] + options
                run = subprocess.run(command, capture_output=True, text=True, timeout=15)
                row.update(options=options, command=command, returncode=run.returncode)
                (args.output / f"{mode}.stderr").write_text(run.stderr, encoding="utf8")
                assert run.returncode == 0 and "1.07.5" in run.stderr
                raw = (args.output / f"{mode}.txt").read_text(encoding="utf8")
                (inchi,) = re.findall(r"^InChI=.+$", raw, re.M)
                (key,) = re.findall(r"^InChIKey=(.+)$", raw, re.M)
                row.update(inchi=inchi, inchi_key=key)
                assert "/C7H14O/" in inchi and "/t6-,7-" in inchi
                if mode == "racemic":
                    assert "/s3" in inchi and "/m" not in inchi
                else:
                    assert "/s1" in inchi
                row["status"] = "encoded"
            except Exception:
                row.update(status="failed", failure=traceback.format_exc())
            rows.append(row)
            print(f"{len(rows)}/2 CRC 684 {mode}: {row['status']}", flush=True)
        failure = None
    except Exception:
        failure = traceback.format_exc()
    result = {"task": "NATIVE-CONV-22G1", "mode": "development", "block": "CRC-684-encoding",
              "reference_version": "InChI 1.07.5", "total": 2, "results": rows,
              "encoded": sum(r["status"] == "encoded" for r in rows),
              "failed": sum(r["status"] == "failed" for r in rows),
              "setup_failure": failure, "seconds": time.perf_counter() - started}
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result))
    return result["encoded"] != 2


if __name__ == "__main__":
    raise SystemExit(main())
